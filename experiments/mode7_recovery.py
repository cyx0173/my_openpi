#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

OPENPI_ROOT = Path("/home/chengyuxuan/openpi")
for p in (OPENPI_ROOT / "experiments", OPENPI_ROOT / "third_party/libero"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mode4_collect as mode4  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


TASK_SUITE_NAME = "libero_10"
SEED = 7
RESIZE_SIZE = 224
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = 520

DEBUG_NOISE_HORIZON = 10
DEBUG_NOISE_DIM = 32

PRECISIONS = ("w4a4", "w4a8", "w4a16")
PRECISION_COST = {"w4a4": 1, "w4a8": 2, "w4a16": 4}

DEFAULT_ACTION_ROOT = Path("/home/chengyuxuan/openpi/experiments/smooth/baseline/trace/action_chunk")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")


def normalize_precision(p: str) -> str:
    p = str(p).lower().strip()
    if p in ("a4", "w4a4"):
        return "w4a4"
    if p in ("a8", "w4a8"):
        return "w4a8"
    if p in ("a16", "w4a16"):
        return "w4a16"
    raise ValueError(f"bad precision: {p}")


def parse_case_id(case_id: str) -> tuple[int, int]:
    m = re.search(r"task(\d+)_ep(\d+)", str(case_id))
    if not m:
        raise ValueError(f"bad case_id: {case_id}")
    return int(m.group(1)), int(m.group(2))


def max_chunk_index(max_steps: int = MAX_STEPS, replan_steps: int = REPLAN_STEPS) -> int:
    return int(math.ceil(max_steps / max(1, replan_steps))) - 1


def make_debug_noise(seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).standard_normal(
        (DEBUG_NOISE_HORIZON, DEBUG_NOISE_DIM)
    ).astype(np.float32)


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def make_env(task_id: int):
    task_suite = mode4.benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
    task = task_suite.get_task(int(task_id))
    env, task_description = mode4._get_libero_env(
        task,
        mode4.LIBERO_ENV_RESOLUTION,
        SEED,
    )
    return env, str(task_description)


def reset_to_chunk0(env, initial_state):
    env.reset()
    obs = env.set_init_state(np.asarray(initial_state))
    done = False
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, _ = env_step(env, mode4.LIBERO_DUMMY_ACTION)
        if done:
            raise RuntimeError("done during dummy wait")
    return obs


def get_policy_element(obs, task_description: str):
    if hasattr(mode4, "_get_policy_images_and_proprio"):
        element, _, _, _ = mode4._get_policy_images_and_proprio(obs, task_description, RESIZE_SIZE)
        return element
    return mode4._get_obs_element(obs, task_description, RESIZE_SIZE)


def make_clients(policy_host: str, port_a4: int, port_a8: int, port_a16: int):
    return {
        "w4a4": websocket_client_policy.WebsocketClientPolicy(policy_host, int(port_a4)),
        "w4a8": websocket_client_policy.WebsocketClientPolicy(policy_host, int(port_a8)),
        "w4a16": websocket_client_policy.WebsocketClientPolicy(policy_host, int(port_a16)),
    }


def infer_actions(client, obs, task_description: str, noise_seed: int):
    element = dict(get_policy_element(obs, task_description))
    element["debug_noise"] = make_debug_noise(noise_seed)
    result = client.infer(element)
    return np.asarray(result["actions"][:REPLAN_STEPS], dtype=np.float64)


def execute_actions(env, obs, actions):
    done = False
    n = 0
    for action in np.asarray(actions, dtype=np.float64).tolist():
        obs, _, done, _ = env_step(env, action)
        n += 1
        if done:
            break
    return obs, done, n

def frame_from_obs_for_video(obs: dict[str, Any]) -> np.ndarray:
    frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def draw_overlay_on_frame(
    frame: np.ndarray,
    *,
    case_id: str,
    chunk_idx: int,
    precision: str,
    noise_seed: int,
    step: int,
    done: bool,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame

    img = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    text = (
        f"{case_id} | chunk={chunk_idx:04d} | {precision} | "
        f"seed={noise_seed} | step={step} | done={done}"
    )

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.rectangle((0, 0, img.width, 24), fill=(0, 0, 0, 150))
    draw.text((6, 6), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def open_mp4_writer(video_path: str | Path, fps: int):
    import imageio.v2 as imageio

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return imageio.get_writer(str(video_path), fps=int(fps), codec="libx264", quality=8)
    except TypeError:
        return imageio.get_writer(str(video_path), fps=int(fps))


def save_final_video_and_actions(
    *,
    case_id: str,
    task_id: int,
    episode_idx: int,
    instruction: str,
    initial_state: Any,
    final_schedule: list[dict[str, Any]],
    policy_host: str,
    port_a4: int,
    port_a8: int,
    port_a16: int,
    final_actions_path: str | Path | None,
    final_video_path: str | Path | None,
    video_fps: int,
    video_overlay: bool,
    max_steps: int,
) -> dict[str, Any]:
    clients = make_clients(policy_host, port_a4, port_a8, port_a16)
    env, default_instruction = make_env(task_id)
    task_description = str(instruction or default_instruction)

    writer = None
    records: list[dict[str, Any]] = []
    t = NUM_STEPS_WAIT
    done = False

    try:
        obs = reset_to_chunk0(env, initial_state)

        if final_video_path is not None:
            writer = open_mp4_writer(final_video_path, fps=video_fps)
            frame = frame_from_obs_for_video(obs)
            if video_overlay:
                frame = draw_overlay_on_frame(
                    frame,
                    case_id=case_id,
                    chunk_idx=-1,
                    precision="start",
                    noise_seed=-1,
                    step=t,
                    done=False,
                )
            writer.append_data(frame)

        for row in final_schedule:
            chunk_idx = int(row["chunk_idx"])
            precision = normalize_precision(row["precision"])
            noise_seed = int(row["noise_seed"])

            planned_actions = infer_actions(
                clients[precision],
                obs,
                task_description,
                noise_seed,
            )

            step_start = int(t)
            executed_actions = []

            for action in np.asarray(planned_actions, dtype=np.float64).tolist():
                if t >= int(max_steps) + NUM_STEPS_WAIT:
                    break

                obs, _, done, _ = env_step(env, action)
                t += 1
                executed_actions.append(action)

                if writer is not None:
                    frame = frame_from_obs_for_video(obs)
                    if video_overlay:
                        frame = draw_overlay_on_frame(
                            frame,
                            case_id=case_id,
                            chunk_idx=chunk_idx,
                            precision=precision,
                            noise_seed=noise_seed,
                            step=t,
                            done=bool(done),
                        )
                    writer.append_data(frame)

                if done:
                    break

            records.append(
                {
                    "chunk_idx": chunk_idx,
                    "precision": precision,
                    "noise_seed": noise_seed,
                    "step_start": step_start,
                    "step_end": int(t),
                    "num_executed_actions": int(len(executed_actions)),
                    "actions": executed_actions,
                    "planned_actions": np.asarray(planned_actions, dtype=np.float64).tolist(),
                }
            )

            print(
                f"[FINAL-PATH] chunk={chunk_idx:04d} precision={precision} "
                f"seed={noise_seed} step={step_start}->{t} "
                f"n={len(executed_actions)} done={done}",
                flush=True,
            )

            if done:
                break

        action_obj = {
            "case_id": case_id,
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "instruction": task_description,
            "final_success": bool(done),
            "final_chunk": int(records[-1]["chunk_idx"]) if records else None,
            "final_step": int(t),
            "num_chunks": int(len(records)),
            "action_semantics": {
                "actions": "actually executed actions; final chunk may have fewer than REPLAN_STEPS actions",
                "planned_actions": "full policy output result['actions'][:REPLAN_STEPS]",
                "replan_steps": int(REPLAN_STEPS),
            },
            "chunks": records,
        }

        if final_actions_path is not None:
            save_json(final_actions_path, action_obj)

        return {
            "success": bool(done),
            "final_step": int(t),
            "num_chunks": int(len(records)),
            "final_actions_path": str(final_actions_path) if final_actions_path is not None else None,
            "final_video_path": str(final_video_path) if final_video_path is not None else None,
        }

    finally:
        if writer is not None:
            writer.close()
        env.close()

def find_path_values(obj: Any) -> list[str]:
    vals: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            vals.extend(find_path_values(v))
    elif isinstance(obj, list):
        for v in obj:
            vals.extend(find_path_values(v))
    elif isinstance(obj, str):
        if obj.endswith(".json") or "_bank.json" in obj:
            vals.append(obj)
    return vals


def find_bank(action_root: Path, case_id: str, precision: str) -> Path | None:
    precision = normalize_precision(precision)
    target = f"{case_id}_{precision}_bank.json"
    patterns = [
        f"*/{precision}/{precision}_action_bank/{target}",
        f"**/{precision}_action_bank/{target}",
        f"**/{target}",
    ]
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(action_root.glob(pat))
    hits = [p for p in hits if p.exists()]
    if not hits:
        return None
    return sorted(hits, key=lambda p: (0 if f"{precision}_action_bank" in str(p) else 1, len(str(p)), str(p)))[0]


def resolve_bank_paths_from_job(job: dict[str, Any], action_root: Path, case_id: str) -> dict[str, str]:
    paths: dict[str, str] = {}

    # Common explicit keys first.
    explicit_candidates = {
        "w4a4": [
            "a4_action_json",
            "w4a4_action_json",
            "a4_bank",
            "w4a4_bank",
            "a4_action_bank",
            "w4a4_action_bank",
        ],
        "w4a8": [
            "a8_action_json",
            "w4a8_action_json",
            "a8_bank",
            "w4a8_bank",
            "a8_action_bank",
            "w4a8_action_bank",
        ],
        "w4a16": [
            "a16_action_json",
            "w4a16_action_json",
            "a16_bank",
            "w4a16_bank",
            "a16_action_bank",
            "w4a16_action_bank",
        ],
    }

    for precision, keys in explicit_candidates.items():
        for key in keys:
            v = job.get(key)
            if isinstance(v, str) and Path(v).exists():
                paths[precision] = v
                break

    # Nested dictionaries such as source / bank_paths / action_jsons.
    for container_key in ("source", "sources", "bank_paths", "action_paths", "action_jsons", "banks"):
        c = job.get(container_key)
        if isinstance(c, dict):
            for precision in PRECISIONS:
                for key in (precision, precision.replace("w4", ""), f"{precision}_bank", f"{precision}_action_json"):
                    v = c.get(key)
                    if isinstance(v, str) and Path(v).exists():
                        paths[precision] = v

    # Last resort: scan every string path in job.
    for s in find_path_values(job):
        p = Path(s)
        if not p.exists():
            continue
        name = p.name.lower()
        for precision in PRECISIONS:
            if precision in name and precision not in paths:
                paths[precision] = str(p)

    # Search under action_root for missing precisions.
    for precision in PRECISIONS:
        if precision not in paths:
            p = find_bank(action_root, case_id, precision)
            if p is not None:
                paths[precision] = str(p)

    return paths


def load_banks(job: dict[str, Any], action_root: Path, case_id: str):
    paths = resolve_bank_paths_from_job(job, action_root, case_id)
    banks: dict[str, dict[str, Any]] = {}
    for precision, p in paths.items():
        if Path(p).exists():
            banks[normalize_precision(precision)] = load_json(p)
    return banks, paths


def choose_baseline(banks: dict[str, dict[str, Any]]):
    for p in PRECISIONS:
        b = banks.get(p)
        if b is not None and bool(b.get("success", False)):
            return p, b
    return None


class SeedProvider:
    def __init__(self, task_id: int, episode_idx: int, banks: dict[str, dict[str, Any]]):
        self.task_id = int(task_id)
        self.episode_idx = int(episode_idx)
        self.by_chunk: dict[int, int] = {}
        bases = []
        for bank in banks.values():
            for ch in bank.get("chunks", []) or []:
                if "chunk_idx" not in ch:
                    continue
                seed = ch.get("noise_seed", ch.get("debug_noise_seed", ch.get("seed")))
                if seed is None:
                    continue
                c = int(ch["chunk_idx"])
                seed = int(seed)
                self.by_chunk[c] = seed
                bases.append(seed - self.task_id * 100000 - self.episode_idx * 1000 - c)
        self.base_seed = 0
        if bases:
            count: dict[int, int] = {}
            for b in bases:
                count[int(b)] = count.get(int(b), 0) + 1
            self.base_seed = max(count.items(), key=lambda kv: kv[1])[0]

    def seed(self, chunk_idx: int) -> int:
        c = int(chunk_idx)
        if c in self.by_chunk:
            return int(self.by_chunk[c])
        return int(self.base_seed) + self.task_id * 100000 + self.episode_idx * 1000 + c


@dataclasses.dataclass
class Cost:
    total: int
    num_a4: int
    num_a8: int
    num_a16: int

    def to_dict(self):
        return {
            "a4": PRECISION_COST["w4a4"],
            "a8": PRECISION_COST["w4a8"],
            "a16": PRECISION_COST["w4a16"],
            "total": int(self.total),
            "num_a4": int(self.num_a4),
            "num_a8": int(self.num_a8),
            "num_a16": int(self.num_a16),
        }


@dataclasses.dataclass
class RolloutResult:
    success: bool
    final_chunk: int | None
    final_step: int
    cost: Cost
    executed_schedule: list[dict[str, Any]]
    new_action_records: list[dict[str, Any]]
    reason: str


def compute_cost(rows: list[dict[str, Any]]) -> Cost:
    counts = {"w4a4": 0, "w4a8": 0, "w4a16": 0}
    total = 0
    for r in rows:
        p = normalize_precision(r["precision"])
        counts[p] += 1
        total += PRECISION_COST[p]
    return Cost(total, counts["w4a4"], counts["w4a8"], counts["w4a16"])


def better_than(a: RolloutResult, b: RolloutResult | None) -> bool:
    if b is None:
        return bool(a.success)
    if a.success != b.success:
        return bool(a.success and not b.success)
    if not a.success:
        return False
    if a.cost.total != b.cost.total:
        return a.cost.total < b.cost.total
    if a.cost.num_a16 != b.cost.num_a16:
        return a.cost.num_a16 < b.cost.num_a16
    if a.cost.num_a8 != b.cost.num_a8:
        return a.cost.num_a8 < b.cost.num_a8
    if a.final_chunk is None:
        return False
    if b.final_chunk is None:
        return True
    return int(a.final_chunk) <= int(b.final_chunk)


def lower_candidates(p: str) -> list[str]:
    p = normalize_precision(p)
    if p == "w4a16":
        return ["w4a4", "w4a8"]
    if p == "w4a8":
        return ["w4a4"]
    return []


def schedule_all(precision: str, max_chunk: int) -> dict[int, str]:
    p = normalize_precision(precision)
    return {int(k): p for k in range(max_chunk + 1)}


def copy_schedule(s: dict[int, str]) -> dict[int, str]:
    return {int(k): normalize_precision(v) for k, v in s.items()}


def set_suffix(s: dict[int, str], *, start: int, precision: str, max_chunk: int) -> dict[int, str]:
    out = copy_schedule(s)
    p = normalize_precision(precision)
    for k in range(int(start), int(max_chunk) + 1):
        out[int(k)] = p
    return out


def suffix_has_a16(s: dict[int, str], *, start: int, max_chunk: int) -> bool:
    for k in range(int(start), int(max_chunk) + 1):
        if normalize_precision(s[k]) == "w4a16":
            return True
    return False


def suffix_all_precision(
    s: dict[int, str],
    *,
    start: int,
    max_chunk: int,
    precision: str,
) -> bool:
    """Return True if every chunk in [start, max_chunk] already has precision.

    Used to avoid redundant suffix probes, e.g. replacing an already-all-A4 tail
    with A4 again.
    """
    if int(start) > int(max_chunk):
        return True

    p = normalize_precision(precision)
    for k in range(int(start), int(max_chunk) + 1):
        if normalize_precision(s[int(k)]) != p:
            return False
    return True


def baseline_result_from_bank(precision: str, bank: dict[str, Any], seed_provider: SeedProvider) -> RolloutResult:
    precision = normalize_precision(precision)
    final_chunk = int(bank.get("num_chunks", len(bank.get("chunks", []))) or 0) - 1
    if final_chunk < 0:
        raise RuntimeError("baseline bank has no chunks")
    rows = [
        {
            "chunk_idx": k,
            "precision": precision,
            "noise_seed": int(seed_provider.seed(k)),
        }
        for k in range(final_chunk + 1)
    ]
    return RolloutResult(
        success=bool(bank.get("success", False)),
        final_chunk=final_chunk,
        final_step=int(bank.get("total_steps", -1)),
        cost=compute_cost(rows),
        executed_schedule=rows,
        new_action_records=[],
        reason="baseline_record",
    )


def rollout_from_action_prefix(
    *,
    case_id: str,
    task_id: int,
    instruction: str,
    initial_state: Any,
    schedule: dict[int, str],
    seed_provider: SeedProvider,
    prefix_action_records: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    start_chunk: int,
    policy_host: str,
    port_a4: int,
    port_a8: int,
    port_a16: int,
    cost_limit: int | None,
    max_chunk: int,
    max_steps: int,
    verbose: bool = False,
) -> RolloutResult:
    clients = make_clients(policy_host, port_a4, port_a8, port_a16)
    env, default_instruction = make_env(task_id)
    task_description = str(instruction or default_instruction)

    executed_rows = [dict(r) for r in prefix_rows]
    new_action_records: list[dict[str, Any]] = []
    running_cost = compute_cost(executed_rows).total
    t = NUM_STEPS_WAIT

    try:
        obs = reset_to_chunk0(env, initial_state)

        for rec in prefix_action_records:
            actions = np.asarray(rec["actions"], dtype=np.float64)
            obs, done, n = execute_actions(env, obs, actions)
            t += n
            if done:
                return RolloutResult(
                    success=True,
                    final_chunk=int(rec["chunk_idx"]),
                    final_step=t,
                    cost=compute_cost(executed_rows),
                    executed_schedule=executed_rows,
                    new_action_records=new_action_records,
                    reason="success_during_prefix_action_replay",
                )

        for k in range(int(start_chunk), int(max_chunk) + 1):
            p = normalize_precision(schedule[int(k)])
            next_cost = running_cost + PRECISION_COST[p]
            if cost_limit is not None and next_cost >= int(cost_limit):
                return RolloutResult(
                    success=False,
                    final_chunk=None,
                    final_step=t,
                    cost=compute_cost(executed_rows),
                    executed_schedule=executed_rows,
                    new_action_records=new_action_records,
                    reason=f"cost_pruned_at_chunk_{k}_next_cost_{next_cost}_limit_{cost_limit}",
                )

            noise_seed = int(seed_provider.seed(k))
            actions = infer_actions(clients[p], obs, task_description, noise_seed)

            row = {"chunk_idx": int(k), "precision": p, "noise_seed": int(noise_seed)}
            executed_rows.append(row)
            new_action_records.append(
                {
                    "chunk_idx": int(k),
                    "precision": p,
                    "noise_seed": int(noise_seed),
                    "actions": actions.copy(),
                }
            )
            running_cost = next_cost

            obs, done, n = execute_actions(env, obs, actions)
            t += n

            if verbose:
                print(f"[ROLLOUT] {case_id} chunk={k:04d} {p} step={t} done={done}", flush=True)

            if done:
                return RolloutResult(
                    success=True,
                    final_chunk=int(k),
                    final_step=int(t),
                    cost=compute_cost(executed_rows),
                    executed_schedule=executed_rows,
                    new_action_records=new_action_records,
                    reason="success",
                )

            if t >= int(max_steps) + NUM_STEPS_WAIT:
                break

        return RolloutResult(
            success=False,
            final_chunk=None,
            final_step=int(t),
            cost=compute_cost(executed_rows),
            executed_schedule=executed_rows,
            new_action_records=new_action_records,
            reason="timeout_or_schedule_end",
        )
    finally:
        env.close()


def run_trials_parallel(trials: list[dict[str, Any]], workers: int):
    if workers <= 1 or len(trials) <= 1:
        return [trial["fn"]() for trial in trials]

    out = [None] * len(trials)
    with futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(t["fn"]): i for i, t in enumerate(trials)}
        for fut in futures.as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def infer_job_identity(job: dict[str, Any], job_path: Path) -> tuple[str, int, int, str]:
    case_id = job.get("case_id")
    if not case_id:
        # Try path fragment .../task08/ep027/...
        m = re.search(r"task(\d+).*?ep(\d+)", str(job_path))
        if m:
            case_id = f"task{int(m.group(1)):02d}_ep{int(m.group(2)):03d}"
    if not case_id:
        task_id = job.get("task_id")
        ep = job.get("episode_idx", job.get("initial_state_idx"))
        if task_id is not None and ep is not None:
            case_id = f"task{int(task_id):02d}_ep{int(ep):03d}"
    if not case_id:
        raise ValueError(f"cannot infer case_id from {job_path}")

    task_id, episode_idx = parse_case_id(case_id)
    task_id = int(job.get("task_id", task_id))
    episode_idx = int(job.get("episode_idx", job.get("initial_state_idx", episode_idx)))
    instruction = str(job.get("instruction") or job.get("task_description") or job.get("language") or "")

    return str(case_id), task_id, episode_idx, instruction


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    job_path = Path(args.job)
    job = load_json(job_path)

    case_id, task_id, episode_idx, job_instruction = infer_job_identity(job, job_path)
    out_dir = Path(args.out) / case_id
    out_path = out_dir / "recovery_schedule.json"
    trial_path = out_dir / "trial_summary.jsonl"

    action_root = Path(args.action_root)
    banks, bank_paths = load_banks(job, action_root, case_id)
    if not banks:
        raise FileNotFoundError(f"no action banks found for {case_id}; action_root={action_root}")

    baseline_pair = choose_baseline(banks)
    if baseline_pair is None:
        record = {
            "case_id": case_id,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "instruction": job_instruction,
            "job_path": str(job_path),
            "final_success": False,
            "reason": "no_success_baseline",
            "bank_paths": bank_paths,
        }
        save_json(out_path, record)
        return record

    baseline_precision, baseline_bank = baseline_pair
    instruction = str(job_instruction or baseline_bank.get("task_description") or baseline_bank.get("instruction") or "")
    initial_state = job.get("initial_state", baseline_bank.get("initial_state"))
    if initial_state is None:
        raise KeyError("initial_state not found in job or baseline bank")

    seed_provider = SeedProvider(task_id, episode_idx, banks)

    max_steps = int(args.max_steps)
    max_chunk = max_chunk_index(max_steps, REPLAN_STEPS)
    current_schedule = schedule_all(baseline_precision, max_chunk)
    current_result = baseline_result_from_bank(baseline_precision, baseline_bank, seed_provider)
    baseline_result = current_result

    # Greedy maintains two schedules:
    #   current_schedule/current_result:
    #       the working greedy path, only updated by accepted single-chunk downgrades.
    #   global_best_schedule/global_best_result:
    #       the best successful schedule seen so far, updated by either accepted
    #       single downgrades or suffix probes.
    #
    # A suffix-A4 probe can become a cheaper successful incumbent, but it must not
    # collapse the working greedy path into an all-A4 tail and stop exploration,
    # because all-A4 can be successful but still finish later and cost more than a
    # mixed A8/A4 schedule.
    global_best_schedule = copy_schedule(current_schedule)
    global_best_result = current_result

    prefix_action_records: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    print(
        f"[任务开始] case={case_id} job={job_path}",
        flush=True,
    )
    print(
        f"[基线] baseline={baseline_precision} success=True "
        f"baseline_cost={baseline_result.cost.total} baseline_final_chunk={baseline_result.final_chunk}",
        flush=True,
    )

    if baseline_precision != "w4a4":
        for k in range(max_chunk + 1):
            if current_result.final_chunk is not None and k > int(current_result.final_chunk):
                break

            current_p = normalize_precision(current_schedule[k])
            single_candidates = lower_candidates(current_p)
            accepted_single_this_chunk = False

            single_trials = []
            for p in single_candidates:
                trial_schedule = copy_schedule(current_schedule)
                trial_schedule[k] = p

                def make_fn(ts=trial_schedule, pp=p):
                    return lambda: (
                        "single",
                        pp,
                        rollout_from_action_prefix(
                            case_id=case_id,
                            task_id=task_id,
                            instruction=instruction,
                            initial_state=initial_state,
                            schedule=ts,
                            seed_provider=seed_provider,
                            prefix_action_records=prefix_action_records,
                            prefix_rows=prefix_rows,
                            start_chunk=k,
                            policy_host=args.policy_host,
                            port_a4=args.port_a4,
                            port_a8=args.port_a8,
                            port_a16=args.port_a16,
                            cost_limit=current_result.cost.total,
                            max_chunk=max_chunk,
                            max_steps=max_steps,
                            verbose=False,
                        ),
                        ts,
                    )

                single_trials.append({"fn": make_fn()})

            single_results = run_trials_parallel(single_trials, int(args.parallel_trials))
            tried_single = []
            best_single = None

            for kind, p, res, trial_schedule in single_results:
                tried_single.append(
                    {
                        "kind": kind,
                        "chunk_idx": int(k),
                        "precision": p,
                        "success": bool(res.success),
                        "reason": res.reason,
                        "final_chunk": res.final_chunk,
                        "cost": res.cost.to_dict(),
                    }
                )
                if better_than(res, current_result):
                    if best_single is None or better_than(res, best_single[1]):
                        best_single = (trial_schedule, res, p)

            if best_single is not None:
                current_schedule, current_result, accepted_p = best_single
                accepted_single_this_chunk = True

                if better_than(current_result, global_best_result):
                    global_best_schedule = copy_schedule(current_schedule)
                    global_best_result = current_result

                event = {
                    "chunk_idx": int(k),
                    "event": "accept_single",
                    "accepted_precision": accepted_p,
                    "result": {
                        "final_chunk": current_result.final_chunk,
                        "cost": current_result.cost.to_dict(),
                    },
                    "tried": tried_single,
                }
                events.append(event)
                if args.save_trial_summary:
                    append_jsonl(trial_path, event)
                print(
                    f"[接受] chunk={k:04d} -> {accepted_p} "
                    f"cost={current_result.cost.total} final_chunk={current_result.final_chunk}",
                    flush=True,
                )
            elif tried_single:
                event = {
                    "chunk_idx": int(k),
                    "event": "reject_single",
                    "current_precision": current_p,
                    "current_cost": current_result.cost.to_dict(),
                    "tried": tried_single,
                }
                events.append(event)
                if args.save_trial_summary:
                    append_jsonl(trial_path, event)

            # Suffix probes.
            #
            # New rule:
            #   - suffix_a4 / suffix_a8 are incumbent probes only.
            #   - They can update global_best_schedule/global_best_result.
            #   - They must NOT update current_schedule/current_result.
            #   - They must NOT stop the greedy scan.
            #
            # Computation control:
            #   - do not test suffix_a4 if the suffix is already all A4.
            #   - do not test suffix_a8 unless the suffix contains A16.
            #   - optionally test suffix probes every N chunks, but always test
            #     after a single downgrade is accepted.
            suffix_start = int(k) + 1
            probe_every = max(1, int(args.suffix_probe_every))
            run_suffix_probe = (
                suffix_start <= int(max_chunk)
                and (
                    accepted_single_this_chunk
                    or int(k) == 0
                    or (int(k) % probe_every == 0)
                )
            )

            cert_specs = []
            if run_suffix_probe:
                if not suffix_all_precision(
                    current_schedule,
                    start=suffix_start,
                    max_chunk=max_chunk,
                    precision="w4a4",
                ):
                    cert_specs.append(
                        (
                            "suffix_a4",
                            "w4a4",
                            set_suffix(
                                current_schedule,
                                start=suffix_start,
                                precision="w4a4",
                                max_chunk=max_chunk,
                            ),
                        )
                    )

                if suffix_has_a16(current_schedule, start=suffix_start, max_chunk=max_chunk):
                    cert_specs.append(
                        (
                            "suffix_a8",
                            "w4a8",
                            set_suffix(
                                current_schedule,
                                start=suffix_start,
                                precision="w4a8",
                                max_chunk=max_chunk,
                            ),
                        )
                    )

            cert_trials = []
            for cert_kind, cert_p, cert_schedule in cert_specs:
                def make_fn(kind=cert_kind, pp=cert_p, ts=cert_schedule):
                    return lambda: (
                        kind,
                        pp,
                        rollout_from_action_prefix(
                            case_id=case_id,
                            task_id=task_id,
                            instruction=instruction,
                            initial_state=initial_state,
                            schedule=ts,
                            seed_provider=seed_provider,
                            prefix_action_records=prefix_action_records,
                            prefix_rows=prefix_rows,
                            start_chunk=k,
                            policy_host=args.policy_host,
                            port_a4=args.port_a4,
                            port_a8=args.port_a8,
                            port_a16=args.port_a16,
                            # Suffix probes are complete candidate schedules, so
                            # they only need to run while they can beat the current
                            # global incumbent.
                            cost_limit=global_best_result.cost.total,
                            max_chunk=max_chunk,
                            max_steps=max_steps,
                            verbose=False,
                        ),
                        ts,
                    )
                cert_trials.append({"fn": make_fn()})

            cert_results = run_trials_parallel(cert_trials, int(args.parallel_trials))
            tried_cert = []
            best_cert = None

            for kind, p, res, sched in cert_results:
                tried_cert.append(
                    {
                        "kind": kind,
                        "suffix_precision": p,
                        "success": bool(res.success),
                        "reason": res.reason,
                        "final_chunk": res.final_chunk,
                        "cost": res.cost.to_dict(),
                    }
                )
                if better_than(res, global_best_result):
                    if best_cert is None or better_than(res, best_cert[3]):
                        best_cert = (kind, p, sched, res)

            if best_cert is not None:
                kind, p, sched, res = best_cert
                global_best_schedule = copy_schedule(sched)
                global_best_result = res

                event = {
                    "chunk_idx": int(k),
                    "event": f"record_best_{kind}",
                    "suffix_precision": p,
                    "result": {
                        "final_chunk": res.final_chunk,
                        "cost": res.cost.to_dict(),
                    },
                    "current_greedy_cost": current_result.cost.to_dict(),
                    "global_best_cost": global_best_result.cost.to_dict(),
                    "tried": tried_cert,
                }
                events.append(event)
                if args.save_trial_summary:
                    append_jsonl(trial_path, event)
                print(
                    f"[后缀记录] chunk={k:04d} kind={kind} "
                    f"cost={res.cost.total} final_chunk={res.final_chunk}",
                    flush=True,
                )
            else:
                event = {
                    "chunk_idx": int(k),
                    "event": (
                        "skip_certificates"
                        if not cert_specs else "reject_certificates"
                    ),
                    "run_suffix_probe": bool(run_suffix_probe),
                    "accepted_single_this_chunk": bool(accepted_single_this_chunk),
                    "current_cost": current_result.cost.to_dict(),
                    "global_best_cost": global_best_result.cost.to_dict(),
                    "tried": tried_cert,
                }
                events.append(event)
                if args.save_trial_summary:
                    append_jsonl(trial_path, event)

            # Advance accepted prefix by one chunk using exact action replay route.
            advance = rollout_from_action_prefix(
                case_id=case_id,
                task_id=task_id,
                instruction=instruction,
                initial_state=initial_state,
                schedule=current_schedule,
                seed_provider=seed_provider,
                prefix_action_records=prefix_action_records,
                prefix_rows=prefix_rows,
                start_chunk=k,
                policy_host=args.policy_host,
                port_a4=args.port_a4,
                port_a8=args.port_a8,
                port_a16=args.port_a16,
                cost_limit=None,
                max_chunk=k,
                max_steps=max_steps,
                verbose=False,
            )

            if not advance.new_action_records:
                # Success occurred while replaying prefix, so we already have a complete prefix.
                break

            rec = advance.new_action_records[0]
            row = {
                "chunk_idx": int(rec["chunk_idx"]),
                "precision": normalize_precision(rec["precision"]),
                "noise_seed": int(rec["noise_seed"]),
            }
            prefix_action_records.append(rec)
            prefix_rows.append(row)

            if advance.success:
                break

    final_validation_schedule = global_best_schedule
    final_validation_source = (
        "global_best"
        if global_best_result is not current_result
        else "current_greedy"
    )

    final_result = rollout_from_action_prefix(
        case_id=case_id,
        task_id=task_id,
        instruction=instruction,
        initial_state=initial_state,
        schedule=final_validation_schedule,
        seed_provider=seed_provider,
        prefix_action_records=[],
        prefix_rows=[],
        start_chunk=0,
        policy_host=args.policy_host,
        port_a4=args.port_a4,
        port_a8=args.port_a8,
        port_a16=args.port_a16,
        cost_limit=None,
        max_chunk=max_chunk,
        max_steps=max_steps,
        verbose=bool(args.verbose_rollout),
    )

    if not final_result.success:
        record = {
            "case_id": case_id,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "instruction": instruction,
            "job_path": str(job_path),
            "baseline_precision": baseline_precision,
            "final_success": False,
            "reason": f"final_validation_failed:{final_result.reason}",
            "bank_paths": bank_paths,
            "baseline_cost": baseline_result.cost.to_dict(),
            "last_current_cost": current_result.cost.to_dict(),
            "global_best_cost": global_best_result.cost.to_dict(),
            "final_validation_source": final_validation_source,
            "greedy_events": events,
        }
        save_json(out_path, record)
        print(f"[保存失败结果] {out_path}", flush=True)
        return record
    final_actions_path = out_dir / "final_actions.json" if args.save_final_actions else None
    final_video_path = out_dir / "final_path.mp4" if args.save_video else None

    final_artifacts = save_final_video_and_actions(
        case_id=case_id,
        task_id=task_id,
        episode_idx=episode_idx,
        instruction=instruction,
        initial_state=initial_state,
        final_schedule=final_result.executed_schedule,
        policy_host=args.policy_host,
        port_a4=args.port_a4,
        port_a8=args.port_a8,
        port_a16=args.port_a16,
        final_actions_path=final_actions_path,
        final_video_path=final_video_path,
        video_fps=int(args.video_fps),
        video_overlay=bool(args.video_overlay),
        max_steps=max_steps,
    )

    if not final_artifacts.get("success"):
        print(
            f"[WARN] final artifact replay did not reach success: {final_artifacts}",
            flush=True,
        )
    record = {
        "case_id": case_id,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "instruction": instruction,
        "job_path": str(job_path),
        "baseline_precision": baseline_precision,
        "final_success": True,
        "final_chunk": int(final_result.final_chunk),
        "final_step": int(final_result.final_step),
        "cost": final_result.cost.to_dict(),
        "global_best_cost_before_validation": global_best_result.cost.to_dict(),
        "final_validation_source": final_validation_source,
        "chunk_schedule": final_result.executed_schedule,
        "final_actions_path": str(final_actions_path) if final_actions_path is not None else None,
        "final_video_path": str(final_video_path) if final_video_path is not None else None,
        "final_artifacts": final_artifacts,
        "bank_paths": bank_paths,
        "seed": {
            "base_seed": int(seed_provider.base_seed),
            "formula": "base_seed + task_id * 100000 + episode_idx * 1000 + chunk_idx",
        },
        "greedy_meta": {
            "method": "full_episode_greedy_action_replay_prefix_cache",
            "max_steps": int(max_steps),
            "max_chunk": int(max_chunk),
            "cost_pruning": True,
            "parallel_trials": int(args.parallel_trials),
            "suffix_probe_every": int(args.suffix_probe_every),
            "suffix_certificate_a4_stop": False,
            "suffix_certificate_accepts_current_schedule": False,
            "suffix_certificate_behavior": "incumbent_only_no_stop",
            "suffix_certificate_a8_continue": True,
            "final_validation_source": final_validation_source,
            "record_rule": "chunk_schedule is final validation executed_schedule from chunk 0 to final success chunk",
        },
        "greedy_events": events,
    }
    save_json(out_path, record)
    print(
        f"[保存结果] recovery_schedule 已保存: {out_path}",
        flush=True,
    )
    print(
        json.dumps(
            {
                "case_id": case_id,
                "final_success": True,
                "baseline_precision": baseline_precision,
                "final_chunk": final_result.final_chunk,
                "cost": final_result.cost.to_dict(),
                "schedule_path": str(out_path),
                "final_actions_path": str(final_actions_path) if final_actions_path is not None else None,
                "final_video_path": str(final_video_path) if final_video_path is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--port-a4", type=int, required=True)
    parser.add_argument("--port-a8", type=int, required=True)
    parser.add_argument("--port-a16", type=int, required=True)

    parser.add_argument("--action-root", default=str(DEFAULT_ACTION_ROOT))
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--parallel-trials", type=int, default=1)
    parser.add_argument(
        "--suffix-probe-every",
        type=int,
        default=4,
        help=(
            "Run suffix A4/A8 probes every N chunks, and always after an accepted "
            "single downgrade. Use 1 to probe every chunk."
        ),
    )
    parser.add_argument("--save-trial-summary", action="store_true")
    parser.add_argument("--verbose-rollout", action="store_true")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-final-actions", action=argparse.BooleanOptionalAction, default=True)


    args = parser.parse_args()

    t0 = time.time()
    try:
        record = run_job(args)
        dt = time.time() - t0
        print(f"[DONE] case={record.get('case_id')} final_success={record.get('final_success')} dt={dt:.1f}s", flush=True)
    except Exception as e:
        dt = time.time() - t0
        print(f"[ERROR] {type(e).__name__}: {e} dt={dt:.1f}s", flush=True)
        raise


if __name__ == "__main__":
    main()

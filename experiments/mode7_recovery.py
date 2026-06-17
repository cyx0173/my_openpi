#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

RECOVERY_PAIR_SAFE_VERSION = "fixed_a16_suffix_full9_greedy_v1"

BIT_COST = {4: 1, 8: 2, 16: 4}
VLM_WEIGHT = 3
ACTION_WEIGHT = 1
MIN_PAIR = (4, 4)
DIAGONAL_PAIRS = ((4, 4), (8, 8), (16, 16))
PAIR_ORDER = (
    (4, 4),
    (4, 8),
    (4, 16),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 4),
    (16, 8),
    (16, 16),
)

DEFAULT_ACTION_ROOT = Path("/home/chengyuxuan/openpi/experiments/smooth/baseline/trace/action_chunk")
SERVER_MID_ROOT = Path("/share/chengyuxuan-local/openpi/selector/recovery_data_selector")

# Selector target label space.  IMPORTANT:
#   PAIR_ORDER is used by the recovery/search schedule and means
#       (current_vlm_bits, current_action_bits).
#   SELECTOR_PAIR_ORDER is the supervised selector target and means
#       (current_action_bits, next_chunk_vlm_bits).
SELECTOR_PAIR_ORDER = (
    (4, 4),
    (4, 8),
    (4, 16),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 4),
    (16, 8),
    (16, 16),
)
SELECTOR_PAIR_TO_LABEL = {p: i for i, p in enumerate(SELECTOR_PAIR_ORDER)}
SELECTOR_LABEL_TO_PAIR = {i: p for p, i in SELECTOR_PAIR_TO_LABEL.items()}
BITS_TO_LABEL = {4: 0, 8: 1, 16: 2}


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


def parse_case_id(case_id: str) -> tuple[int, int]:
    m = re.search(r"task(\d+)_ep(\d+)", str(case_id))
    if not m:
        raise ValueError(f"bad case_id: {case_id}")
    return int(m.group(1)), int(m.group(2))


def max_chunk_index(max_steps: int = MAX_STEPS, replan_steps: int = REPLAN_STEPS) -> int:
    return int(math.ceil(max_steps / max(1, replan_steps))) - 1


def pair_cost(pair: tuple[int, int]) -> int:
    v, a = int(pair[0]), int(pair[1])
    if v not in BIT_COST or a not in BIT_COST:
        raise ValueError(f"bad pair: {pair}")
    return int(VLM_WEIGHT * BIT_COST[v] + ACTION_WEIGHT * BIT_COST[a])


def pair_name(pair: tuple[int, int]) -> str:
    return f"v{int(pair[0])}_a{int(pair[1])}"


def bits_to_precision(bits: int) -> str:
    bits = int(bits)
    if bits not in BIT_COST:
        raise ValueError(f"bad bits: {bits}")
    return f"w4a{bits}"


def selector_pair_label_id(action_bits: int, next_vlm_bits: int) -> int:
    key = (int(action_bits), int(next_vlm_bits))
    if key not in SELECTOR_PAIR_TO_LABEL:
        raise ValueError(
            f"bad selector pair label: action={action_bits}, next_vlm={next_vlm_bits}"
        )
    return int(SELECTOR_PAIR_TO_LABEL[key])


def selector_pair_name(label_id: int) -> str:
    action_bits, next_vlm_bits = SELECTOR_LABEL_TO_PAIR[int(label_id)]
    return f"a{action_bits}_nextv{next_vlm_bits}"


def selector_feature_tag(case_id: str, chunk_idx: int, vlm_bits: int) -> str:
    return (
        f"{case_id}__midprefix_{bits_to_precision(vlm_bits)}"
        f"__chunk{int(chunk_idx):04d}"
    )


def mid_feature_path(case_id: str, chunk_idx: int, vlm_bits: int) -> Path:
    return (
        SERVER_MID_ROOT
        / "cases"
        / str(case_id)
        / "mid"
        / bits_to_precision(vlm_bits)
        / f"chunk{int(chunk_idx):04d}.npz"
    )


def normalize_pair(x: Any) -> tuple[int, int]:
    if isinstance(x, str):
        s = x.lower().strip()
        aliases = {
            "w4a4": (4, 4), "a4": (4, 4), "v4_a4": (4, 4),
            "w4a8": (8, 8), "a8": (8, 8), "v8_a8": (8, 8),
            "w4a16": (16, 16), "a16": (16, 16), "v16_a16": (16, 16),
        }
        if s in aliases:
            return aliases[s]
        m = re.search(r"v\s*(\d+)\D+a\s*(\d+)", s)
        if m:
            return int(m.group(1)), int(m.group(2))
        parts = [p for p in re.split(r"[^0-9]+", s) if p]
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise ValueError(f"bad pair: {x!r}")


def pair_row(chunk_idx: int, pair: tuple[int, int], noise_seed: int) -> dict[str, Any]:
    p = normalize_pair(pair)
    return {
        "chunk_idx": int(chunk_idx),
        "pair": pair_name(p),
        "vlm_a_bits": int(p[0]),
        "action_a_bits": int(p[1]),
        "pair_cost": int(pair_cost(p)),
        "noise_seed": int(noise_seed),
    }


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
    env, task_description = mode4._get_libero_env(task, mode4.LIBERO_ENV_RESOLUTION, SEED)
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


def infer_actions(
    client: websocket_client_policy.WebsocketClientPolicy,
    obs,
    task_description: str,
    noise_seed: int,
    pair: tuple[int, int],
    *,
    debug_collect_tag: str | None = None,
) -> np.ndarray:
    p = normalize_pair(pair)
    element = dict(get_policy_element(obs, task_description))
    element["debug_noise"] = make_debug_noise(noise_seed)
    element["vlm_a_bits"] = int(p[0])
    element["action_a_bits"] = int(p[1])

    # Only final-validation data collection should pass this tag.
    # Greedy candidate search does not save large tensors.
    if debug_collect_tag is not None:
        element["debug_collect_tag"] = str(debug_collect_tag)

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


class SeedProvider:
    """Formula-only debug-noise seed provider.

    Do not read seeds from old action-bank files.  Every real chunk has exactly
    one deterministic seed shared by all candidate pairs:
        base_seed + task_id * 100000 + episode_idx * 1000 + chunk_idx
    """

    def __init__(self, task_id: int, episode_idx: int, base_seed: int = 0):
        self.task_id = int(task_id)
        self.episode_idx = int(episode_idx)
        self.base_seed = int(base_seed)

    def seed(self, chunk_idx: int) -> int:
        return int(self.base_seed + self.task_id * 100000 + self.episode_idx * 1000 + int(chunk_idx))


@dataclasses.dataclass
class Cost:
    total: int
    num_vlm_a4: int
    num_vlm_a8: int
    num_vlm_a16: int
    num_action_a4: int
    num_action_a8: int
    num_action_a16: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "bit_cost": dict(BIT_COST),
            "vlm_weight": int(VLM_WEIGHT),
            "action_weight": int(ACTION_WEIGHT),
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


@dataclasses.dataclass
class PreparedEnv:
    env: Any
    obs: Any
    t: int
    done: bool


def row_pair(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["vlm_a_bits"]), int(row["action_a_bits"])


def compute_cost(rows: list[dict[str, Any]]) -> Cost:
    total = 0
    counts = {
        "vlm_a4": 0, "vlm_a8": 0, "vlm_a16": 0,
        "action_a4": 0, "action_a8": 0, "action_a16": 0,
    }
    for r in rows:
        p = row_pair(r)
        total += pair_cost(p)
        counts[f"vlm_a{p[0]}"] += 1
        counts[f"action_a{p[1]}"] += 1
    return Cost(
        total=int(total),
        num_vlm_a4=int(counts["vlm_a4"]),
        num_vlm_a8=int(counts["vlm_a8"]),
        num_vlm_a16=int(counts["vlm_a16"]),
        num_action_a4=int(counts["action_a4"]),
        num_action_a8=int(counts["action_a8"]),
        num_action_a16=int(counts["action_a16"]),
    )


def prefix_cost(prefix_rows: list[dict[str, Any]]) -> int:
    return int(compute_cost(prefix_rows).total)


def _safe_final_step(x: RolloutResult) -> int:
    return int(x.final_step) if x.success and x.final_step is not None else 10**12


def _safe_final_chunk(x: RolloutResult) -> int:
    return int(x.final_chunk) if x.success and x.final_chunk is not None else 10**12


def better_than(a: RolloutResult, b: RolloutResult | None) -> bool:
    """Success, lower cost, then same-cost faster success.

    This keeps the policy you requested: same cost but fewer final steps may
    replace the old best.
    """
    if b is None:
        return bool(a.success)
    if a.success != b.success:
        return bool(a.success and not b.success)
    if not a.success:
        return False
    if int(a.cost.total) != int(b.cost.total):
        return int(a.cost.total) < int(b.cost.total)
    if _safe_final_step(a) != _safe_final_step(b):
        return _safe_final_step(a) < _safe_final_step(b)
    if _safe_final_chunk(a) != _safe_final_chunk(b):
        return _safe_final_chunk(a) < _safe_final_chunk(b)
    if a.cost.num_vlm_a16 + a.cost.num_action_a16 != b.cost.num_vlm_a16 + b.cost.num_action_a16:
        return (a.cost.num_vlm_a16 + a.cost.num_action_a16) < (b.cost.num_vlm_a16 + b.cost.num_action_a16)
    if a.cost.num_vlm_a8 + a.cost.num_action_a8 != b.cost.num_vlm_a8 + b.cost.num_action_a8:
        return (a.cost.num_vlm_a8 + a.cost.num_action_a8) < (b.cost.num_vlm_a8 + b.cost.num_action_a8)
    return False


def schedule_all(pair: tuple[int, int], max_chunk: int) -> dict[int, tuple[int, int]]:
    p = normalize_pair(pair)
    return {int(k): p for k in range(int(max_chunk) + 1)}


def copy_schedule(s: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    return {int(k): normalize_pair(v) for k, v in s.items()}


def schedule_from_prefix_current_tail(
    *,
    prefix_rows: list[dict[str, Any]],
    chunk_idx: int,
    current_pair: tuple[int, int],
    tail_pair: tuple[int, int],
    max_chunk: int,
) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for r in prefix_rows:
        out[int(r["chunk_idx"])] = row_pair(r)
    out[int(chunk_idx)] = normalize_pair(current_pair)
    tail = normalize_pair(tail_pair)
    for j in range(int(chunk_idx) + 1, int(max_chunk) + 1):
        out[int(j)] = tail
    return out


def schedule_from_prefix_tail(
    *,
    prefix_rows: list[dict[str, Any]],
    start_chunk: int,
    tail_pair: tuple[int, int],
    max_chunk: int,
) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for r in prefix_rows:
        out[int(r["chunk_idx"])] = row_pair(r)
    tail = normalize_pair(tail_pair)
    for j in range(int(start_chunk), int(max_chunk) + 1):
        out[int(j)] = tail
    return out


def schedule_from_result(result: RolloutResult, tail_pair: tuple[int, int], max_chunk: int) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for r in result.executed_schedule:
        out[int(r["chunk_idx"])] = row_pair(r)
    if out:
        last = out[max(out.keys())]
    else:
        last = normalize_pair(tail_pair)
    for k in range(max(out.keys()) + 1 if out else 0, int(max_chunk) + 1):
        out[int(k)] = normalize_pair(tail_pair if tail_pair is not None else last)
    return out


def prepare_prefix_env(env, initial_state: Any, prefix_action_records: list[dict[str, Any]]) -> PreparedEnv:
    obs = reset_to_chunk0(env, initial_state)
    t = NUM_STEPS_WAIT
    done = False

    for rec in prefix_action_records:
        actions = np.asarray(rec["actions"], dtype=np.float64)
        obs, done, n = execute_actions(env, obs, actions)
        t += n
        if done:
            break

    return PreparedEnv(env=env, obs=obs, t=int(t), done=bool(done))


def rollout_from_prepared_env(
    *,
    prepared: PreparedEnv,
    case_id: str,
    task_description: str,
    schedule: dict[int, tuple[int, int]],
    seed_provider: SeedProvider,
    prefix_rows: list[dict[str, Any]],
    start_chunk: int,
    client: websocket_client_policy.WebsocketClientPolicy,
    cost_limit: int | None,
    max_chunk: int,
    max_steps: int,
    verbose: bool = False,
    save_selector_features: bool = False,
) -> RolloutResult:
    env = prepared.env
    obs = prepared.obs
    t = int(prepared.t)
    executed_rows = [dict(r) for r in prefix_rows]
    new_action_records: list[dict[str, Any]] = []
    running_cost = compute_cost(executed_rows).total

    if prepared.done:
        return RolloutResult(
            success=True,
            final_chunk=int(executed_rows[-1]["chunk_idx"]) if executed_rows else None,
            final_step=t,
            cost=compute_cost(executed_rows),
            executed_schedule=executed_rows,
            new_action_records=[],
            reason="success_during_prefix_action_replay",
        )

    for k in range(int(start_chunk), int(max_chunk) + 1):
        p = normalize_pair(schedule[int(k)])
        next_cost = int(running_cost + pair_cost(p))

        # Use > rather than >= so equal-cost but faster candidates are allowed
        # to run to success and replace the current best.
        if cost_limit is not None and next_cost > int(cost_limit):
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
        debug_collect_tag = None
        if save_selector_features:
            # This tag matches the server-side save path:
            #   SERVER_MID_ROOT/cases/<case_id>/mid/w4aX/chunkXXXX.npz
            # It intentionally uses current VLM bits p[0], not action bits.
            debug_collect_tag = selector_feature_tag(case_id, k, p[0])

        actions = infer_actions(
            client,
            obs,
            task_description,
            noise_seed,
            p,
            debug_collect_tag=debug_collect_tag,
        )
        row = pair_row(k, p, noise_seed)

        executed_rows.append(row)
        new_action_records.append({
            "chunk_idx": int(k),
            "pair": pair_name(p),
            "vlm_a_bits": int(p[0]),
            "action_a_bits": int(p[1]),
            "pair_cost": int(pair_cost(p)),
            "noise_seed": int(noise_seed),
            "actions": actions.copy(),
        })
        running_cost = next_cost

        obs, done, n = execute_actions(env, obs, actions)
        t += n

        if verbose:
            print(f"[ROLLOUT] {case_id} chunk={k:04d} pair={pair_name(p)} seed={noise_seed} step={t} done={done}", flush=True)

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


def rollout_from_action_prefix(
    *,
    case_id: str,
    task_id: int,
    instruction: str,
    initial_state: Any,
    schedule: dict[int, tuple[int, int]],
    seed_provider: SeedProvider,
    prefix_action_records: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    start_chunk: int,
    client: websocket_client_policy.WebsocketClientPolicy,
    cost_limit: int | None,
    max_chunk: int,
    max_steps: int,
    verbose: bool = False,
    save_selector_features: bool = False,
) -> RolloutResult:
    """Single-env rollout.

    The only replay is committed action replay.  Prefix chunks are never
    re-inferred.  The online suffix uses seed_provider.seed(real_chunk_idx).
    """
    env, default_instruction = make_env(task_id)
    task_description = str(instruction or default_instruction)
    try:
        prepared = prepare_prefix_env(env, initial_state, prefix_action_records)
        return rollout_from_prepared_env(
            prepared=prepared,
            case_id=case_id,
            task_description=task_description,
            schedule=schedule,
            seed_provider=seed_provider,
            prefix_rows=prefix_rows,
            start_chunk=start_chunk,
            client=client,
            cost_limit=cost_limit,
            max_chunk=max_chunk,
            max_steps=max_steps,
            verbose=verbose,
            save_selector_features=bool(save_selector_features),
        )
    finally:
        env.close()


def evaluate_prefix_only(
    *,
    task_id: int,
    instruction: str,
    initial_state: Any,
    prefix_action_records: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
) -> RolloutResult:
    env, default_instruction = make_env(task_id)
    _ = str(instruction or default_instruction)
    try:
        prepared = prepare_prefix_env(env, initial_state, prefix_action_records)
        if prepared.done:
            return RolloutResult(
                success=True,
                final_chunk=int(prefix_rows[-1]["chunk_idx"]) if prefix_rows else None,
                final_step=int(prepared.t),
                cost=compute_cost(prefix_rows),
                executed_schedule=[dict(r) for r in prefix_rows],
                new_action_records=[],
                reason="prefix_already_success",
            )
        return RolloutResult(
            success=False,
            final_chunk=None,
            final_step=int(prepared.t),
            cost=compute_cost(prefix_rows),
            executed_schedule=[dict(r) for r in prefix_rows],
            new_action_records=[],
            reason="prefix_not_success_yet",
        )
    finally:
        env.close()


def infer_job_identity(job: dict[str, Any], job_path: Path) -> tuple[str, int, int, str]:
    case_id = job.get("case_id")
    if not case_id:
        m = re.search(r"task(\d+).*?ep(\d+)", str(job_path))
        if m:
            case_id = f"task{int(m.group(1)):02d}_ep{int(m.group(2)):03d}"
    if not case_id:
        task_id = job.get("task_id")
        ep = job.get("episode_idx", job.get("ep_id", job.get("episode_id", job.get("initial_state_idx"))))
        if task_id is not None and ep is not None:
            case_id = f"task{int(task_id):02d}_ep{int(ep):03d}"
    if not case_id:
        raise ValueError(f"cannot infer case_id from {job_path}")

    task_id, episode_idx = parse_case_id(case_id)
    task_id = int(job.get("task_id", task_id))
    episode_idx = int(job.get("episode_idx", job.get("ep_id", job.get("episode_id", job.get("initial_state_idx", episode_idx)))))
    instruction = str(job.get("instruction") or job.get("task_description") or job.get("language") or "")
    return str(case_id), task_id, episode_idx, instruction


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


def load_bank_fallbacks(job: dict[str, Any], action_root: Path, case_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    banks: dict[str, Any] = {}
    bank_paths: dict[str, str] = {}

    for s in find_path_values(job):
        p = Path(s)
        if not p.exists():
            continue
        for precision in ("w4a4", "w4a8", "w4a16"):
            if precision in p.name.lower():
                banks[precision] = load_json(p)
                bank_paths[precision] = str(p)

    for precision in ("w4a4", "w4a8", "w4a16"):
        if precision not in banks:
            p = find_bank(action_root, case_id, precision)
            if p is not None:
                banks[precision] = load_json(p)
                bank_paths[precision] = str(p)

    return banks, bank_paths


def find_action_record_for_chunk(result: RolloutResult, chunk_idx: int) -> dict[str, Any] | None:
    for rec in result.new_action_records:
        if int(rec.get("chunk_idx", -1)) == int(chunk_idx):
            return rec
    return None


def suffix_seed_preview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) <= 10:
        selected = rows
    else:
        selected = rows[:5] + [{"chunk_idx": -1, "pair": "...", "noise_seed": -1}] + rows[-5:]
    out = []
    for r in selected:
        if int(r.get("chunk_idx", -1)) < 0:
            out.append(dict(r))
        else:
            out.append({
                "chunk_idx": int(r["chunk_idx"]),
                "pair": r.get("pair", pair_name(row_pair(r))),
                "noise_seed": int(r["noise_seed"]),
            })
    return out


def search_current_chunk(
    *,
    case_id: str,
    task_id: int,
    instruction: str,
    initial_state: Any,
    k: int,
    tail_baseline: tuple[int, int],
    global_best_result: RolloutResult,
    current_certificate: RolloutResult,
    seed_provider: SeedProvider,
    prefix_action_records: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    client: websocket_client_policy.WebsocketClientPolicy,
    max_chunk: int,
    max_steps: int,
    candidate_outcome_path: Path,
    verbose: bool,
) -> tuple[RolloutResult, dict[int, tuple[int, int]], list[dict[str, Any]]]:
    """Run all 9 counterfactual current-pair candidates for a visited prefix.

    For the current committed prefix and current chunk k, every pair in PAIR_ORDER
    is rolled out fully with the same suffix tail_baseline.  No lower-bound skip
    and no cost-limit cut are applied inside this 9-way set.  The 9 outcomes are
    written to candidate_outcomes.jsonl and may be used later for counterfactual
    long-horizon analysis.

    Outer greedy may still stop between chunks when the committed prefix already
    succeeds or when prefix cost reaches/exceeds the best certified success cost.
    """
    events: list[dict[str, Any]] = []
    baseline_pair = normalize_pair(tail_baseline)
    best_current: RolloutResult | None = None
    best_schedule: dict[int, tuple[int, int]] | None = None
    pcost = prefix_cost(prefix_rows)
    current_noise_seed = int(seed_provider.seed(k))

    events.append({
        "chunk_idx": int(k),
        "event": "full9_counterfactual_start",
        "tail_baseline": pair_name(baseline_pair),
        "prefix_cost": int(pcost),
        "current_noise_seed": int(current_noise_seed),
        "num_pairs": int(len(PAIR_ORDER)),
        "note": "all 9 current-pair candidates are rolled out fully; no candidate cost pruning",
    })

    for pair in PAIR_ORDER:
        p = normalize_pair(pair)
        schedule = schedule_from_prefix_current_tail(
            prefix_rows=prefix_rows,
            chunk_idx=k,
            current_pair=p,
            tail_pair=baseline_pair,
            max_chunk=max_chunk,
        )
        r = rollout_from_action_prefix(
            case_id=case_id,
            task_id=task_id,
            instruction=instruction,
            initial_state=initial_state,
            schedule=schedule,
            seed_provider=seed_provider,
            prefix_action_records=prefix_action_records,
            prefix_rows=prefix_rows,
            start_chunk=k,
            client=client,
            cost_limit=None,
            max_chunk=max_chunk,
            max_steps=max_steps,
            verbose=verbose,
        )

        new_rows = r.executed_schedule[len(prefix_rows):]
        current_pair_cost = int(pair_cost(p))
        total_cost = int(r.cost.total)
        suffix_cost = int(total_cost - int(pcost) - current_pair_cost)
        cost_breakdown = {
            "prefix_cost": int(pcost),
            "current_pair_cost": int(current_pair_cost),
            "suffix_cost": int(suffix_cost),
            "total_cost": int(total_cost),
        }
        outcome_row = {
            "case_id": str(case_id),
            "task_id": int(task_id),
            "instruction": str(instruction),
            "chunk_idx": int(k),
            "candidate_pair": pair_name(p),
            "candidate_vlm_a_bits": int(p[0]),
            "candidate_action_a_bits": int(p[1]),
            "tail_baseline": pair_name(baseline_pair),
            "suffix_baseline": pair_name(baseline_pair),
            "baseline_fixed": True,
            "prefix_cost": int(pcost),
            "candidate_pair_cost": int(current_pair_cost),
            "cost_breakdown": cost_breakdown,
            "current_noise_seed": int(current_noise_seed),
            "success": bool(r.success),
            "reason": str(r.reason),
            "final_chunk": r.final_chunk,
            "final_step": int(r.final_step),
            "cost": r.cost.to_dict(),
            "num_new_rows": int(len(new_rows)),
            "suffix_seed_preview": suffix_seed_preview(new_rows),
            "outcome_known": True,
            "pruned": False,
            "source": "fixed_a16_suffix_full9_counterfactual_rollout",
        }
        append_jsonl(candidate_outcome_path, outcome_row)

        event = {
            "chunk_idx": int(k),
            "event": "candidate_rollout",
            "pair": pair_name(p),
            "tail_baseline": pair_name(baseline_pair),
            "baseline_fixed": True,
            "success": bool(r.success),
            "reason": r.reason,
            "cost": r.cost.to_dict(),
            "cost_breakdown": cost_breakdown,
            "final_chunk": r.final_chunk,
            "final_step": r.final_step,
            "current_noise_seed": int(current_noise_seed),
            "num_new_rows": int(len(new_rows)),
            "suffix_seed_preview": suffix_seed_preview(new_rows),
            "candidate_outcome_logged": True,
        }
        events.append(event)

        if not r.success:
            continue

        rec = find_action_record_for_chunk(r, k)
        if rec is None:
            events.append({
                "chunk_idx": int(k),
                "event": "candidate_ignore_missing_current_action_record",
                "pair": pair_name(p),
                "available_new_action_chunks": [int(x.get("chunk_idx", -1)) for x in r.new_action_records],
            })
            continue

        first_pair = normalize_pair((rec.get("vlm_a_bits"), rec.get("action_a_bits")))
        if first_pair != p:
            events.append({
                "chunk_idx": int(k),
                "event": "candidate_ignore_bad_current_action_pair",
                "pair": pair_name(p),
                "record_pair": pair_name(first_pair),
            })
            continue

        if better_than(r, best_current):
            best_current = r
            best_schedule = schedule
            events.append({
                "chunk_idx": int(k),
                "event": "candidate_updates_best_current",
                "pair": pair_name(p),
                "best_current_cost": int(r.cost.total),
                "best_current_final_step": int(r.final_step),
            })

    if best_current is None or best_schedule is None:
        # This should be rare because current_certificate is supposed to be a
        # certified fallback.  Keep the old certificate as an emergency fallback
        # so the outer loop can stop cleanly instead of crashing.
        best_current = current_certificate
        best_schedule = schedule_from_prefix_current_tail(
            prefix_rows=prefix_rows,
            chunk_idx=k,
            current_pair=baseline_pair,
            tail_pair=baseline_pair,
            max_chunk=max_chunk,
        )
        events.append({
            "chunk_idx": int(k),
            "event": "fallback_to_previous_certificate_after_full9_no_success",
            "tail_baseline": pair_name(baseline_pair),
            "certificate_success": bool(current_certificate.success),
            "certificate_reason": current_certificate.reason,
            "certificate_cost": current_certificate.cost.to_dict(),
        })

    return best_current, best_schedule, events

def frame_from_obs_for_video(obs: dict[str, Any]) -> np.ndarray:
    frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def draw_overlay_on_frame(frame: np.ndarray, *, case_id: str, chunk_idx: int, pair: str, step: int, done: bool) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame
    img = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    text = f"{case_id} | chunk={chunk_idx:04d} | {pair} | step={step} | done={done}"
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
    final_action_records: list[dict[str, Any]],
    final_actions_path: str | Path | None,
    final_video_path: str | Path | None,
    video_fps: int,
    video_overlay: bool,
    max_steps: int,
) -> dict[str, Any]:
    """Save final video/actions by replaying validated final actions.

    This does not re-infer actions from policy servers.  It uses
    final_action_records produced by final validation.
    """
    env, default_instruction = make_env(task_id)
    task_description = str(instruction or default_instruction)
    writer = None
    records: list[dict[str, Any]] = []
    t = NUM_STEPS_WAIT
    done = False
    action_by_chunk = {int(r["chunk_idx"]): r for r in final_action_records}

    try:
        obs = reset_to_chunk0(env, initial_state)
        if final_video_path is not None:
            writer = open_mp4_writer(final_video_path, fps=video_fps)
            frame = frame_from_obs_for_video(obs)
            if video_overlay:
                frame = draw_overlay_on_frame(frame, case_id=case_id, chunk_idx=-1, pair="start", step=t, done=False)
            writer.append_data(frame)

        for row in final_schedule:
            k = int(row["chunk_idx"])
            p = row_pair(row)
            noise_seed = int(row["noise_seed"])

            rec = action_by_chunk.get(k)
            if rec is None:
                raise RuntimeError(f"final_action_records missing chunk {k}; refusing to re-infer for video/actions")

            rec_pair = normalize_pair((rec["vlm_a_bits"], rec["action_a_bits"]))
            rec_seed = int(rec["noise_seed"])
            if rec_pair != p or rec_seed != noise_seed:
                raise RuntimeError(
                    f"final_action_record mismatch at chunk {k}: "
                    f"schedule={pair_name(p)} seed={noise_seed}, "
                    f"record={pair_name(rec_pair)} seed={rec_seed}"
                )

            planned_actions = np.asarray(rec["actions"], dtype=np.float64)
            step_start = int(t)
            executed_actions = []

            for action in planned_actions.tolist():
                if t >= int(max_steps) + NUM_STEPS_WAIT:
                    break
                obs, _, done, _ = env_step(env, action)
                t += 1
                executed_actions.append(action)
                if writer is not None:
                    frame = frame_from_obs_for_video(obs)
                    if video_overlay:
                        frame = draw_overlay_on_frame(frame, case_id=case_id, chunk_idx=k, pair=pair_name(p), step=t, done=done)
                    writer.append_data(frame)
                if done:
                    break

            records.append({
                "chunk_idx": int(k),
                "pair": pair_name(p),
                "vlm_a_bits": int(p[0]),
                "action_a_bits": int(p[1]),
                "pair_cost": int(pair_cost(p)),
                "noise_seed": int(noise_seed),
                "step_start": step_start,
                "step_end": int(t),
                "num_executed_actions": int(len(executed_actions)),
                "actions": executed_actions,
                "planned_actions": planned_actions.tolist(),
            })
            print(f"[FINAL-PATH] chunk={k:04d} pair={pair_name(p)} step={step_start}->{t} done={done}", flush=True)
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
            "chunks": records,
            "source": "replay_validated_final_action_records_no_reinfer",
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



def export_selector_pair_labels(
    *,
    out_dir: Path,
    case_id: str,
    task_id: int,
    episode_idx: int,
    instruction: str,
    final_result: RolloutResult,
    schedule_path: Path,
) -> dict[str, Any]:
    """Export 9-way selector labels from the final validated schedule.

    Feature/label alignment:
      feature = chunk k prefix hidden under current_vlm_bits_k
      label   = (action_bits_k, next_chunk_vlm_bits_{k+1})

    The final chunk is skipped because it has no next VLM decision.
    """
    rows = [dict(r) for r in final_result.executed_schedule]
    label_path = Path(out_dir) / "selector_pair_labels.jsonl"

    if label_path.exists():
        label_path.unlink()

    n = 0
    skipped = 0

    for i, row in enumerate(rows):
        if i + 1 >= len(rows):
            skipped += 1
            continue

        chunk_idx = int(row["chunk_idx"])
        current_vlm_bits = int(row["vlm_a_bits"])
        action_bits = int(row["action_a_bits"])
        next_vlm_bits = int(rows[i + 1]["vlm_a_bits"])

        label_id = selector_pair_label_id(action_bits, next_vlm_bits)
        feature_tag = selector_feature_tag(case_id, chunk_idx, current_vlm_bits)
        mid_path = mid_feature_path(case_id, chunk_idx, current_vlm_bits)

        label_row = {
            "case_id": str(case_id),
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "instruction": str(instruction),
            "chunk_idx": int(chunk_idx),

            "feature_tag": feature_tag,
            "mid_path": str(mid_path),

            # selector input condition
            "current_vlm_a_bits": int(current_vlm_bits),
            "current_vlm_label_id": int(BITS_TO_LABEL[int(current_vlm_bits)]),

            # auxiliary old 3-way labels
            "action_a_bits": int(action_bits),
            "action_label_id": int(BITS_TO_LABEL[int(action_bits)]),
            "next_vlm_a_bits": int(next_vlm_bits),
            "next_vlm_label_id": int(BITS_TO_LABEL[int(next_vlm_bits)]),

            # main 9-way selector label
            "pair_label_id": int(label_id),
            "pair_label_name": selector_pair_name(label_id),

            "noise_seed": int(row["noise_seed"]),
            "pair": str(row.get("pair", pair_name(row_pair(row)))),
            "pair_cost": int(row.get("pair_cost", pair_cost(row_pair(row)))),

            "source": "final_validated_schedule",
            "final_success": True,
            "schedule_path": str(schedule_path),
        }

        append_jsonl(label_path, label_row)
        n += 1

    return {
        "selector_pair_label_path": str(label_path),
        "num_selector_pair_labels": int(n),
        "num_final_chunks": int(len(rows)),
        "skipped_final_chunk_without_next_vlm": int(skipped),
        "label_semantics": (
            "9-way selector target over (current action precision, next chunk VLM precision); "
            "current_vlm_a_bits is selector input condition; final chunk is skipped"
        ),
    }

def complete_schedule_dict(schedule: dict[int, tuple[int, int]], max_chunk: int) -> dict[int, tuple[int, int]]:
    if not schedule:
        return {}
    out = {int(k): normalize_pair(v) for k, v in schedule.items()}
    last_pair = out[min(out.keys())]
    for k in range(0, int(max_chunk) + 1):
        if k in out:
            last_pair = out[k]
        else:
            out[k] = last_pair
    return out


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    job_path = Path(args.job)
    job = load_json(job_path)
    case_id, task_id, episode_idx, job_instruction = infer_job_identity(job, job_path)
    out_dir = Path(args.out) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recovery_schedule.json"
    trial_path = out_dir / "trial_summary.jsonl"
    candidate_outcome_path = out_dir / "candidate_outcomes.jsonl"

    # Avoid appending stale rows when this case is re-run.
    if trial_path.exists():
        trial_path.unlink()
    if candidate_outcome_path.exists():
        candidate_outcome_path.unlink()

    banks, bank_paths = load_bank_fallbacks(job, Path(args.action_root), case_id)

    instruction = str(job_instruction or "")
    if not instruction:
        for b in banks.values():
            instruction = str(b.get("task_description") or b.get("instruction") or "")
            if instruction:
                break

    initial_state = job.get("initial_state")
    if initial_state is None:
        for b in banks.values():
            if b.get("initial_state") is not None:
                initial_state = b.get("initial_state")
                break
    if initial_state is None:
        raise KeyError("initial_state not found in job or available fallback banks")

    seed_provider = SeedProvider(task_id, episode_idx, base_seed=int(args.base_seed))
    max_steps = int(args.max_steps)
    max_chunk = max_chunk_index(max_steps, REPLAN_STEPS)
    client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, int(args.policy_port))

    print(f"[任务开始] case={case_id} job={job_path}", flush=True)
    print(
        f"[SEED] base_seed={seed_provider.base_seed} formula=base+task*100000+ep*1000+chunk "
        f"chunk0_seed={seed_provider.seed(0)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Fixed-A16 safety certificate.
    # If full v16_a16 cannot solve this episode, this case is skipped.
    # ------------------------------------------------------------------
    tail_baseline = (16, 16)
    print(f"[A16-SAFETY] pair={pair_name(tail_baseline)}", flush=True)
    a16_schedule = schedule_all(tail_baseline, max_chunk)
    a16_safety_result = rollout_from_action_prefix(
        case_id=case_id,
        task_id=task_id,
        instruction=instruction,
        initial_state=initial_state,
        schedule=a16_schedule,
        seed_provider=seed_provider,
        prefix_action_records=[],
        prefix_rows=[],
        start_chunk=0,
        client=client,
        cost_limit=None,
        max_chunk=max_chunk,
        max_steps=max_steps,
        verbose=bool(args.verbose_rollout),
    )

    a16_safety_record = {
        "pair": pair_name(tail_baseline),
        "event": "a16_a16_safety_rollout",
        "source": "fixed_a16_safety_rollout",
        "success": bool(a16_safety_result.success),
        "reason": a16_safety_result.reason,
        "final_chunk": a16_safety_result.final_chunk,
        "final_step": int(a16_safety_result.final_step),
        "cost": a16_safety_result.cost.to_dict(),
        "baseline_fixed": True,
        "note": "episode is skipped if this full v16_a16 rollout fails",
    }
    if args.save_trial_summary:
        append_jsonl(trial_path, a16_safety_record)
    print(
        f"[A16-SAFETY-DONE] success={a16_safety_result.success} "
        f"reason={a16_safety_result.reason} step={a16_safety_result.final_step} "
        f"cost={a16_safety_result.cost.total}",
        flush=True,
    )

    if not a16_safety_result.success:
        record = {
            "case_id": case_id,
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "instruction": instruction,
            "job_path": str(job_path),
            "final_success": False,
            "reason": "skip_a16_a16_baseline_failed",
            "tail_baseline_final": pair_name(tail_baseline),
            "a16_a16_safety": a16_safety_record,
            "candidate_outcome_path": str(candidate_outcome_path),
            "bank_paths": bank_paths,
            "seed": {
                "base_seed": int(seed_provider.base_seed),
                "formula": "base_seed + task_id * 100000 + episode_idx * 1000 + chunk_idx",
            },
            "greedy_meta": {
                "method": "fixed_a16_suffix_full9_greedy",
                "implementation_version": RECOVERY_PAIR_SAFE_VERSION,
                "skip_rule": "skip episode if full v16_a16 safety rollout fails",
                "fixed_tail_baseline": pair_name(tail_baseline),
            },
        }
        save_json(out_path, record)
        print(f"[跳过] case={case_id} reason=skip_a16_a16_baseline_failed 保存={out_path}", flush=True)
        return record

    global_best_result: RolloutResult = a16_safety_result
    global_best_schedule: dict[int, tuple[int, int]] = a16_schedule
    current_certificate: RolloutResult = a16_safety_result

    prefix_action_records: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [a16_safety_record]

    print(
        f"[固定安全尾巴] tail_baseline={pair_name(tail_baseline)} "
        f"safety_cost={a16_safety_result.cost.total} final_chunk={a16_safety_result.final_chunk}",
        flush=True,
    )

    for k in range(max_chunk + 1):
        pcost = prefix_cost(prefix_rows)

        # Prefix-only success: if the committed actions already solve the task,
        # update global best and stop.
        if prefix_action_records:
            prefix_result = evaluate_prefix_only(
                task_id=task_id,
                instruction=instruction,
                initial_state=initial_state,
                prefix_action_records=prefix_action_records,
                prefix_rows=prefix_rows,
            )
            if prefix_result.success:
                if better_than(prefix_result, global_best_result):
                    global_best_result = prefix_result
                    global_best_schedule = {int(r["chunk_idx"]): row_pair(r) for r in prefix_result.executed_schedule}
                event = {
                    "chunk_idx": int(k),
                    "event": "stop_prefix_already_success",
                    "prefix_cost": int(prefix_result.cost.total),
                    "final_step": int(prefix_result.final_step),
                    "final_chunk": prefix_result.final_chunk,
                    "global_best_cost": global_best_result.cost.to_dict(),
                }
                events.append(event)
                if args.save_trial_summary:
                    append_jsonl(trial_path, event)
                print(f"[提前停止] chunk={k:04d} committed prefix already succeeds", flush=True)
                break

        # Stop between chunks only.  Within a visited chunk all 9 candidates are
        # still rolled out fully.
        if global_best_result.success and pcost >= int(global_best_result.cost.total):
            event = {
                "chunk_idx": int(k),
                "event": "stop_prefix_cost_reaches_global_best",
                "prefix_cost": int(pcost),
                "global_best_cost": global_best_result.cost.to_dict(),
                "note": "committed prefix cost already reaches/exceeds best successful schedule cost",
            }
            events.append(event)
            if args.save_trial_summary:
                append_jsonl(trial_path, event)
            print(f"[提前停止] chunk={k:04d} prefix_cost >= global_best_cost", flush=True)
            break

        if global_best_result.success and pcost + pair_cost(MIN_PAIR) > int(global_best_result.cost.total):
            event = {
                "chunk_idx": int(k),
                "event": "stop_prefix_lower_bound_exceeds_global_best",
                "prefix_cost": int(pcost),
                "min_future_pair_cost": int(pair_cost(MIN_PAIR)),
                "global_best_cost": global_best_result.cost.to_dict(),
            }
            events.append(event)
            if args.save_trial_summary:
                append_jsonl(trial_path, event)
            print(f"[提前停止] chunk={k:04d} prefix_cost+min_pair > global_best", flush=True)
            break

        winning_result, winning_schedule, search_events = search_current_chunk(
            case_id=case_id,
            task_id=task_id,
            instruction=instruction,
            initial_state=initial_state,
            k=k,
            tail_baseline=tail_baseline,
            global_best_result=global_best_result,
            current_certificate=current_certificate,
            seed_provider=seed_provider,
            prefix_action_records=prefix_action_records,
            prefix_rows=prefix_rows,
            client=client,
            max_chunk=max_chunk,
            max_steps=max_steps,
            candidate_outcome_path=candidate_outcome_path,
            verbose=bool(args.verbose_rollout),
        )
        events.extend(search_events)
        if args.save_trial_summary:
            for ev in search_events:
                append_jsonl(trial_path, ev)

        if winning_result is None or not winning_result.success:
            event = {
                "chunk_idx": int(k),
                "event": "stop_no_certified_candidate_for_current_chunk",
                "tail_baseline": pair_name(tail_baseline),
                "prefix_cost": int(pcost),
                "global_best_cost": global_best_result.cost.to_dict(),
                "note": "unexpected: fixed v16_a16 safety tail should provide a fallback candidate",
            }
            events.append(event)
            if args.save_trial_summary:
                append_jsonl(trial_path, event)
            print(f"[停止] chunk={k:04d} no certified candidate", flush=True)
            break

        rec = find_action_record_for_chunk(winning_result, k)
        if rec is None:
            event = {
                "chunk_idx": int(k),
                "event": "stop_missing_action_record_for_current_chunk",
                "tail_baseline": pair_name(tail_baseline),
                "prefix_cost": int(pcost),
                "winning_reason": winning_result.reason,
                "winning_final_chunk": winning_result.final_chunk,
                "winning_cost": winning_result.cost.to_dict(),
                "available_new_action_chunks": [int(x.get("chunk_idx", -1)) for x in winning_result.new_action_records],
            }
            events.append(event)
            if args.save_trial_summary:
                append_jsonl(trial_path, event)
            print(f"[停止] chunk={k:04d} certificate missing action record", flush=True)
            break

        p = normalize_pair((rec["vlm_a_bits"], rec["action_a_bits"]))
        row = pair_row(rec["chunk_idx"], p, rec["noise_seed"])
        prefix_action_records.append(rec)
        prefix_rows.append(row)

        if better_than(winning_result, global_best_result):
            global_best_result = winning_result
            global_best_schedule = copy_schedule(winning_schedule)

        # Under the fixed-A16 setting, the winning current rollout itself is the
        # new certificate for the committed prefix.  We never change tail_baseline.
        current_certificate = winning_result

        event = {
            "chunk_idx": int(k),
            "event": "commit_current_pair",
            "accepted_pair": pair_name(row_pair(row)),
            "prefix_cost_after_commit": int(prefix_cost(prefix_rows)),
            "winning_result": {
                "success": bool(winning_result.success),
                "reason": winning_result.reason,
                "final_chunk": winning_result.final_chunk,
                "final_step": winning_result.final_step,
                "cost": winning_result.cost.to_dict(),
            },
            "global_best_cost": global_best_result.cost.to_dict(),
            "tail_baseline": pair_name(tail_baseline),
            "tail_baseline_fixed": True,
        }
        events.append(event)
        if args.save_trial_summary:
            append_jsonl(trial_path, event)

        print(
            f"[提交] chunk={k:04d} pair={pair_name(row_pair(row))} "
            f"fixed_tail={pair_name(tail_baseline)} prefix_cost={prefix_cost(prefix_rows)} "
            f"winner_cost={winning_result.cost.total} global_best={global_best_result.cost.total}",
            flush=True,
        )

        if winning_result.success and winning_result.final_chunk is not None and int(winning_result.final_chunk) <= int(k):
            break

    final_validation_schedule = complete_schedule_dict(global_best_schedule, max_chunk)
    final_validation_source = "fixed_a16_suffix_global_best_certified_single_env"

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
        client=client,
        cost_limit=None,
        max_chunk=max_chunk,
        max_steps=max_steps,
        verbose=bool(args.verbose_rollout),
        save_selector_features=bool(args.save_selector_features),
    )

    if not final_result.success:
        record = {
            "case_id": case_id,
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "instruction": instruction,
            "job_path": str(job_path),
            "tail_baseline_final": pair_name(tail_baseline),
            "final_success": False,
            "reason": f"final_validation_failed:{final_result.reason}",
            "bank_paths": bank_paths,
            "a16_a16_safety": a16_safety_record,
            "global_best_cost_before_validation": global_best_result.cost.to_dict(),
            "final_validation_source": final_validation_source,
            "candidate_outcome_path": str(candidate_outcome_path),
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
        final_action_records=final_result.new_action_records,
        final_actions_path=final_actions_path,
        final_video_path=final_video_path,
        video_fps=int(args.video_fps),
        video_overlay=bool(args.video_overlay),
        max_steps=max_steps,
    )

    selector_label_artifacts = export_selector_pair_labels(
        out_dir=out_dir,
        case_id=case_id,
        task_id=task_id,
        episode_idx=episode_idx,
        instruction=instruction,
        final_result=final_result,
        schedule_path=out_path,
    )

    if not final_artifacts.get("success"):
        print(f"[WARN] final action replay did not reach success: {final_artifacts}", flush=True)

    record = {
        "case_id": case_id,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "instruction": instruction,
        "job_path": str(job_path),
        "tail_baseline_final": pair_name(tail_baseline),
        "tail_baseline_fixed": True,
        "final_success": True,
        "final_chunk": int(final_result.final_chunk),
        "final_step": int(final_result.final_step),
        "cost": final_result.cost.to_dict(),
        "a16_a16_safety": a16_safety_record,
        "global_best_cost_before_validation": global_best_result.cost.to_dict(),
        "final_validation_source": final_validation_source,
        "chunk_schedule": final_result.executed_schedule,
        "final_actions_path": str(final_actions_path) if final_actions_path is not None else None,
        "final_video_path": str(final_video_path) if final_video_path is not None else None,
        "final_artifacts": final_artifacts,
        "selector_label_artifacts": selector_label_artifacts,
        "selector_pair_label_path": selector_label_artifacts["selector_pair_label_path"],
        "candidate_outcome_path": str(candidate_outcome_path),
        "bank_paths": bank_paths,
        "seed": {
            "base_seed": int(seed_provider.base_seed),
            "formula": "base_seed + task_id * 100000 + episode_idx * 1000 + chunk_idx",
        },
        "greedy_meta": {
            "method": "fixed_a16_suffix_full9_greedy",
            "implementation_version": RECOVERY_PAIR_SAFE_VERSION,
            "pair_cost": {pair_name(p): pair_cost(p) for p in PAIR_ORDER},
            "trial_execution": "single_env_per_rollout_committed_action_replay_only",
            "certified_fallback": True,
            "fixed_tail_baseline": pair_name(tail_baseline),
            "a16_safety_gate": "full v16_a16 rollout must succeed before collecting labels; otherwise episode is skipped",
            "candidate_counterfactuals": "for each visited prefix/chunk, all 9 current-pair candidates are rolled out fully with suffix fixed to v16_a16 and logged to candidate_outcomes.jsonl",
            "cost_pruning": "no within-chunk candidate pruning; only between-chunk prefix success/cost-bound stopping is used",
            "equal_cost_faster_replaces_best": True,
            "tail_lowering": "disabled; tail_baseline never changes during label collection",
            "stagnation_probe": "disabled",
            "max_steps": int(max_steps),
            "max_chunk": int(max_chunk),
            "final_validation_source": final_validation_source,
            "record_rule": "chunk_schedule is final validation executed_schedule from chunk 0 to final success chunk",
            "selector_label_rule": (
                "pair_label_id is derived from final validation executed_schedule: "
                "(action_bits at chunk k, vlm_bits at chunk k+1); final chunk skipped; "
                "candidate_outcomes.jsonl should be treated as the main counterfactual outcome-surface supervision"
            ),
            "server_feature_collection": (
                "enabled only during final validation when --save-selector-features is true; "
                "server saves current_vlm prefix hidden/state via debug_collect_tag"
            ),
        },
        "greedy_events": events,
    }
    save_json(out_path, record)

    print(f"[保存结果] {out_path}", flush=True)
    print(json.dumps({
        "case_id": case_id,
        "final_success": True,
        "final_chunk": final_result.final_chunk,
        "final_step": final_result.final_step,
        "cost": final_result.cost.to_dict(),
        "schedule_path": str(out_path),
        "final_actions_path": str(final_actions_path) if final_actions_path is not None else None,
        "final_video_path": str(final_video_path) if final_video_path is not None else None,
        "selector_pair_label_path": selector_label_artifacts["selector_pair_label_path"],
        "candidate_outcome_path": str(candidate_outcome_path),
    }, indent=2, ensure_ascii=False), flush=True)
    return record

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--action-root", default=str(DEFAULT_ACTION_ROOT))
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--save-trial-summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-rollout", action="store_true")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-final-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-selector-features", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    t0 = time.time()
    try:
        record = run_job(args)
        print(f"[DONE] case={record.get('case_id')} final_success={record.get('final_success')} dt={time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e} dt={time.time() - t0:.1f}s", flush=True)
        raise


if __name__ == "__main__":
    main()

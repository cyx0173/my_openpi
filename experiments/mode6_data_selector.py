#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as websocket_policy


# Client-side index/debug artifacts.
# Training should use server_state saved in server mid npz, not the 8-dim client debug state.
CLIENT_DATA_ROOT = Path("/home/chengyuxuan/openpi/experiments/selector_dataset")
SERVER_MID_ROOT = Path("/share/chengyuxuan-local/openpi/selector/recovery_data_selector")
A4_BANK_ROOT = Path("/home/chengyuxuan/openpi/experiments/smooth/baseline/trace/action_chunk")
TASK_SUITE_NAME = "libero_10"
SEED = 7
RESIZE_SIZE = 224
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = 520

DEBUG_NOISE_HORIZON = 10
DEBUG_NOISE_DIM = 32

BITS = (4, 8, 16)
BITS_TO_LABEL = {4: 0, 8: 1, 16: 2}
LABEL_TO_BITS = {v: k for k, v in BITS_TO_LABEL.items()}
EXPECTED_CANDIDATE_PAIRS = [f"v{v}_a{a}" for v in BITS for a in BITS]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def bits_to_precision(bits: int) -> str:
    bits = int(bits)
    if bits not in BITS_TO_LABEL:
        raise ValueError(f"bad activation bits: {bits}")
    return f"w4a{bits}"


def pair_name(vlm_bits: int, action_bits: int) -> str:
    return f"v{int(vlm_bits)}_a{int(action_bits)}"


def parse_case_id(case_id: str) -> tuple[int | None, int | None]:
    m = re.fullmatch(r"task(\d+)_ep(\d+)", str(case_id))
    if m is None:
        return None, None
    return int(m.group(1)), int(m.group(2))


def normalize_precision_to_bits(p: str | int) -> int:
    if isinstance(p, (int, np.integer)):
        bits = int(p)
        if bits in BITS_TO_LABEL:
            return bits
        raise ValueError(f"bad precision bits: {p}")

    s = str(p).lower().strip()
    if s in {"4", "a4", "w4a4", "v4", "vlm4"}:
        return 4
    if s in {"8", "a8", "w4a8", "v8", "vlm8"}:
        return 8
    if s in {"16", "a16", "w4a16", "v16", "vlm16"}:
        return 16
    raise ValueError(f"bad precision: {p}")


def parse_pair_bits(row: dict[str, Any]) -> tuple[int, int]:
    """Return (vlm_a_bits, action_a_bits) from a schedule row.

    Supports new pair rows such as v4_a8 as well as old single-precision rows
    such as w4a8, where VLM and action bits are assumed equal.
    """
    vlm_keys = ("vlm_a_bits", "current_vlm_a_bits", "schedule_vlm_a_bits", "vlm_bits")
    act_keys = ("action_a_bits", "current_action_a_bits", "action_bits")

    vlm = None
    action = None

    for k in vlm_keys:
        if k in row and row[k] is not None:
            vlm = normalize_precision_to_bits(row[k])
            break

    for k in act_keys:
        if k in row and row[k] is not None:
            action = normalize_precision_to_bits(row[k])
            break

    if vlm is not None and action is not None:
        return int(vlm), int(action)

    for k in ("pair", "precision", "accepted_pair", "name"):
        if k not in row or row[k] is None:
            continue

        s = str(row[k]).lower().strip()

        m = re.fullmatch(r"v(\d+)_a(\d+)", s)
        if m is not None:
            return int(m.group(1)), int(m.group(2))

        m = re.fullmatch(r"vlm(\d+)_action(\d+)", s)
        if m is not None:
            return int(m.group(1)), int(m.group(2))

        try:
            bits = normalize_precision_to_bits(s)
            return int(bits), int(bits)
        except ValueError:
            pass

    raise KeyError(f"cannot parse vlm/action bits from schedule row: {row}")


def debug_noise_seed(base_seed: int, task_id: int, episode_idx: int, chunk_idx: int) -> int:
    return int(base_seed) + int(task_id) * 100000 + int(episode_idx) * 1000 + int(chunk_idx)


def make_debug_noise(seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).standard_normal(
        (DEBUG_NOISE_HORIZON, DEBUG_NOISE_DIM)
    ).astype(np.float32)


def quat_to_axisangle(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def video_frame(obs: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1])


def make_policy_input(obs: dict[str, Any], task_description: str):
    agent = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(video_frame(obs), RESIZE_SIZE, RESIZE_SIZE)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            RESIZE_SIZE,
            RESIZE_SIZE,
        )
    )

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    eef_axis_angle = quat_to_axisangle(obs["robot0_eef_quat"])
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    observation_state = np.concatenate([eef_pos, eef_axis_angle, gripper_qpos]).astype(np.float32)

    element = {
        "observation/image": agent,
        "observation/wrist_image": wrist,
        "observation/state": observation_state,
        "prompt": str(task_description),
    }

    state = {
        "selector_state": observation_state,
        "observation_state": observation_state,
        "eef_pos": eef_pos.astype(np.float32),
        "eef_axis_angle": eef_axis_angle.astype(np.float32),
        "gripper_qpos": gripper_qpos.astype(np.float32),
    }

    return element, state


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def fresh_obs(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "_get_observations"):
            try:
                try:
                    obs = cur._get_observations(force_update=True)
                except TypeError:
                    obs = cur._get_observations()
                if isinstance(obs, dict):
                    return obs
            except Exception:
                pass
        cur = getattr(cur, "env", None)
    raise RuntimeError("cannot get fresh obs")


def make_env(task_id: int):
    task = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]().get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(SEED)
    return env, task.language


def find_a4_bank(case_id: str, schedule: dict[str, Any]) -> Path:
    bank_path = (schedule.get("bank_paths") or {}).get("w4a4")
    if bank_path and Path(bank_path).exists():
        return Path(bank_path)

    src_path = (schedule.get("source") or {}).get("a4_action_json")
    if src_path and Path(src_path).exists():
        return Path(src_path)

    target = f"{case_id}_w4a4_bank.json"
    matches = list(A4_BANK_ROOT.glob(f"*/w4a4/w4a4_action_bank/{target}")) or list(A4_BANK_ROOT.rglob(target))
    if not matches:
        raise FileNotFoundError(f"cannot find A4 bank for {case_id}")

    return sorted(matches, key=lambda p: (0 if "w4a4_action_bank" in str(p) else 1, len(str(p))))[0]


def extract_debug_noise_base_seed(schedule: dict[str, Any]) -> int:
    """Return the base seed used by recovery debug noise."""
    value = schedule.get("debug_noise_base_seed", None)
    if value is None:
        value = schedule.get("seed", 0)

    if isinstance(value, dict):
        for key in ("debug_noise_base_seed", "base_seed", "seed", "value"):
            if key in value and value[key] is not None:
                return int(value[key])
        return 0

    if value is None:
        return 0

    return int(value)


def resolve_outcome_source(schedule_path: Path) -> Path:
    """Return the JSONL file containing 9-pair counterfactual outcomes.

    Prefer the newer candidate_outcomes.jsonl name; fall back to existing
    trial_summary.jsonl, which already contains candidate_vlm/action rows in
    the current recovery pipeline.
    """
    candidates = [
        schedule_path.parent / "candidate_outcomes.jsonl",
        schedule_path.parent / "trial_summary.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Keep a deterministic pointer even if the outcome file is produced later.
    return candidates[0]


def schedule_rows(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    raw = schedule.get("selector_chunk_schedule") or schedule.get("chunk_schedule")
    if not raw:
        raise KeyError("missing selector_chunk_schedule / chunk_schedule")

    case_id = str(schedule.get("case_id", ""))
    task_id = schedule.get("task_id")
    episode_idx = schedule.get("episode_idx")

    if task_id is None or episode_idx is None:
        parsed_task, parsed_ep = parse_case_id(case_id)
        if task_id is None:
            task_id = parsed_task
        if episode_idx is None:
            episode_idx = parsed_ep

    if task_id is None or episode_idx is None:
        raise ValueError(f"cannot infer task_id/episode_idx from schedule: case_id={case_id}")

    base_seed = extract_debug_noise_base_seed(schedule)

    out: list[dict[str, Any]] = []
    prev = -1

    for x in raw:
        chunk_idx = int(x["chunk_idx"])
        if chunk_idx != prev + 1:
            raise ValueError(f"schedule is not contiguous: prev={prev}, current={chunk_idx}")

        vlm_bits, action_bits = parse_pair_bits(x)
        if vlm_bits not in BITS_TO_LABEL or action_bits not in BITS_TO_LABEL:
            raise ValueError(f"bad bits at chunk={chunk_idx}: v{vlm_bits}_a{action_bits}")

        noise_seed = int(x.get("noise_seed", debug_noise_seed(base_seed, int(task_id), int(episode_idx), chunk_idx)))

        out.append(
            {
                "chunk_idx": chunk_idx,
                "vlm_a_bits": int(vlm_bits),
                "action_a_bits": int(action_bits),
                "pair": pair_name(vlm_bits, action_bits),
                "vlm_precision": bits_to_precision(vlm_bits),
                "action_precision": bits_to_precision(action_bits),
                "vlm_label_id": int(BITS_TO_LABEL[vlm_bits]),
                "action_label_id": int(BITS_TO_LABEL[action_bits]),
                "noise_seed": noise_seed,
            }
        )
        prev = chunk_idx

    for i, item in enumerate(out):
        if i + 1 < len(out):
            next_vlm_bits = int(out[i + 1]["vlm_a_bits"])
            item["next_vlm_a_bits"] = next_vlm_bits
            item["next_vlm_precision"] = bits_to_precision(next_vlm_bits)
            item["next_vlm_label_id"] = int(BITS_TO_LABEL[next_vlm_bits])
        else:
            item["next_vlm_a_bits"] = -1
            item["next_vlm_precision"] = ""
            item["next_vlm_label_id"] = -1

    return out


def save_selector_state(
    path: Path,
    state: dict[str, np.ndarray],
    *,
    case_id: str,
    task_id: int,
    episode_idx: int,
    chunk_idx: int,
    step_start: int,
    noise_seed: int,
    vlm_a_bits: int,
    action_a_bits: int,
    next_vlm_a_bits: int,
    action_label_id: int,
    next_vlm_label_id: int,
) -> None:
    """Debug-only raw client-side 8-dim state.

    Clean training should use server_state saved in server mid npz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        selector_state=state["selector_state"],
        observation_state=state["observation_state"],
        eef_pos=state["eef_pos"],
        eef_axis_angle=state["eef_axis_angle"],
        gripper_qpos=state["gripper_qpos"],

        case_id=np.asarray(case_id),
        task_id=np.asarray(task_id, dtype=np.int64),
        episode_idx=np.asarray(episode_idx, dtype=np.int64),
        chunk_idx=np.asarray(chunk_idx, dtype=np.int64),
        step_start=np.asarray(step_start, dtype=np.int64),
        noise_seed=np.asarray(noise_seed, dtype=np.int64),

        schedule_vlm_a_bits=np.asarray(vlm_a_bits, dtype=np.int64),
        vlm_a_bits=np.asarray(vlm_a_bits, dtype=np.int64),
        action_a_bits=np.asarray(action_a_bits, dtype=np.int64),

        action_label_id=np.asarray(action_label_id, dtype=np.int64),
        next_vlm_a_bits=np.asarray(next_vlm_a_bits, dtype=np.int64),
        next_vlm_label_id=np.asarray(next_vlm_label_id, dtype=np.int64),

        vlm_precision=np.asarray(bits_to_precision(vlm_a_bits)),
        action_precision=np.asarray(bits_to_precision(action_a_bits)),
        next_vlm_precision=np.asarray(bits_to_precision(next_vlm_a_bits) if next_vlm_a_bits in BITS_TO_LABEL else ""),
    )


def mid_feature_path(case_id: str, chunk_idx: int, vlm_bits: int) -> Path:
    return (
        SERVER_MID_ROOT
        / "cases"
        / case_id
        / "mid"
        / bits_to_precision(vlm_bits)
        / f"chunk{chunk_idx:04d}.npz"
    )


def verify_mid_feature_file(
    *,
    case_id: str,
    chunk_idx: int,
    vlm_bits: int,
    action_bits: int,
) -> Path:
    """Check that server saved the expected selector feature file.

    New server format saves selector_context_tokens / selector_context_mask,
    not vlm_prefix_last_hidden. Old aliases prefix_input_embs / prefix_pad_mask
    are accepted for transition only.
    """
    path = mid_feature_path(case_id, chunk_idx, vlm_bits)

    if not path.exists():
        raise FileNotFoundError(f"server did not save mid feature: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "selector_context_tokens" in data:
            token_key = "selector_context_tokens"
        elif "prefix_input_embs" in data:
            token_key = "prefix_input_embs"
        else:
            raise KeyError(f"missing selector_context_tokens/prefix_input_embs in {path}")

        if "selector_context_mask" in data:
            mask_key = "selector_context_mask"
        elif "prefix_pad_mask" in data:
            mask_key = "prefix_pad_mask"
        else:
            raise KeyError(f"missing selector_context_mask/prefix_pad_mask in {path}")

        if "server_state" not in data and "state" not in data:
            raise KeyError(f"missing server_state/state in {path}")

        tokens = data[token_key]
        mask = data[mask_key]

        if tokens.dtype != np.float16:
            raise TypeError(f"{path} {token_key} dtype={tokens.dtype}, expected float16")
        if tokens.ndim != 2:
            raise RuntimeError(f"{path} {token_key} ndim={tokens.ndim}, expected 2")
        if mask.dtype != np.bool_:
            raise TypeError(f"{path} {mask_key} dtype={mask.dtype}, expected bool")
        if mask.ndim != 1:
            raise RuntimeError(f"{path} {mask_key} ndim={mask.ndim}, expected 1")
        if mask.shape[0] != tokens.shape[0]:
            raise RuntimeError(f"{path} mask length={mask.shape[0]} != tokens T={tokens.shape[0]}")
        if int(mask.sum()) <= 0:
            raise RuntimeError(f"{path} mask has no valid tokens")

        state_key = "server_state" if "server_state" in data else "state"
        state = data[state_key]
        if state.dtype != np.float32:
            raise TypeError(f"{path} {state_key} dtype={state.dtype}, expected float32")
        if state.ndim != 1:
            raise RuntimeError(f"{path} {state_key} ndim={state.ndim}, expected 1")

        saved_vlm = None
        for k in ("current_vlm_a_bits", "vlm_a_bits", "requested_vlm_a_bits"):
            if k in data:
                saved_vlm = int(np.asarray(data[k]).reshape(-1)[0])
                break

        if saved_vlm is not None and saved_vlm != int(vlm_bits):
            raise RuntimeError(
                f"saved vlm bits mismatch in {path}: expected={vlm_bits}, saved={saved_vlm}"
            )

        if "requested_action_a_bits" in data:
            saved_action = int(np.asarray(data["requested_action_a_bits"]).reshape(-1)[0])
            if saved_action != int(action_bits):
                raise RuntimeError(
                    f"saved requested_action_a_bits mismatch in {path}: "
                    f"expected={action_bits}, saved={saved_action}"
                )

    return path


def collect(schedule_path: Path, *, host: str, port: int) -> None:
    schedule = load_json(schedule_path)
    case_id = str(schedule["case_id"])
    task_id = int(schedule["task_id"])
    episode_idx = int(schedule.get("episode_idx", parse_case_id(case_id)[1]))

    if schedule.get("final_success") is False:
        raise RuntimeError(f"schedule final_success is false: {schedule_path}")

    items = schedule_rows(schedule)
    outcome_source_path = resolve_outcome_source(schedule_path)

    bank_path = find_a4_bank(case_id, schedule)
    bank = load_json(bank_path)

    case_root = CLIENT_DATA_ROOT / "cases" / case_id
    state_dir = case_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    client = websocket_policy.WebsocketClientPolicy(host, int(port))

    env, env_task_description = make_env(task_id)
    task_description = str(schedule.get("instruction") or env_task_description)

    counts = Counter()
    done = False
    step = 0
    last_chunk = None
    written_mid_files: list[str] = []
    feature_index_rows: list[dict[str, Any]] = []

    try:
        env.reset()
        obs0 = env.set_init_state(np.asarray(bank["initial_state"]))
        obs = obs0 if isinstance(obs0, dict) else fresh_obs(env)

        dummy = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(NUM_STEPS_WAIT):
            obs, _, done, _ = env_step(env, dummy)
            step += 1
            if done:
                raise RuntimeError("done during initial wait")

        for item in items:
            if done or step >= MAX_STEPS + NUM_STEPS_WAIT:
                break

            chunk_idx = int(item["chunk_idx"])
            best_vlm_bits = int(item["vlm_a_bits"])
            best_action_bits = int(item["action_a_bits"])
            best_next_vlm_bits = int(item["next_vlm_a_bits"])
            best_action_label_id = int(item["action_label_id"])
            best_next_vlm_label_id = int(item["next_vlm_label_id"])
            noise_seed = int(item["noise_seed"])
            outcome_group_id = f"{case_id}__chunk{chunk_idx:04d}"

            element_base, state = make_policy_input(obs, task_description)

            save_selector_state(
                state_dir / f"chunk{chunk_idx:04d}_state.npz",
                state,
                case_id=case_id,
                task_id=task_id,
                episode_idx=episode_idx,
                chunk_idx=chunk_idx,
                step_start=step,
                noise_seed=noise_seed,
                vlm_a_bits=best_vlm_bits,
                action_a_bits=best_action_bits,
                next_vlm_a_bits=best_next_vlm_bits,
                action_label_id=best_action_label_id,
                next_vlm_label_id=best_next_vlm_label_id,
            )

            # Same pre-action observation, three VLM precisions, one dynamic-switching server.
            # Server saves selector context tokens and server_state via debug_collect_tag into:
            #   SERVER_MID_ROOT/cases/<case_id>/mid/w4a4|w4a8|w4a16/chunkXXXX.npz
            # Only the result matching best_vlm_bits is executed in the environment.
            results: dict[int, Any] = {}
            debug_noise = make_debug_noise(noise_seed)

            for collect_vlm_bits in BITS:
                tag = f"{case_id}__midprefix_{bits_to_precision(collect_vlm_bits)}__chunk{chunk_idx:04d}"
                element = dict(element_base)
                element["debug_noise"] = debug_noise.copy()
                element["debug_collect_tag"] = tag
                element["vlm_a_bits"] = int(collect_vlm_bits)
                element["action_a_bits"] = int(best_action_bits)

                results[int(collect_vlm_bits)] = client.infer(element)

            for collect_vlm_bits in BITS:
                collect_vlm_bits = int(collect_vlm_bits)
                mid_path = verify_mid_feature_file(
                    case_id=case_id,
                    chunk_idx=chunk_idx,
                    vlm_bits=collect_vlm_bits,
                    action_bits=best_action_bits,
                )
                written_mid_files.append(str(mid_path))

                feature_index_rows.append(
                    {
                        "case_id": case_id,
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "chunk_idx": int(chunk_idx),

                        # Current VLM bit of this feature file.
                        "current_vlm_a_bits": collect_vlm_bits,
                        "current_vlm_precision": bits_to_precision(collect_vlm_bits),
                        "current_vlm_label_id": int(BITS_TO_LABEL[collect_vlm_bits]),
                        "feature_path": str(mid_path),

                        # Best / anchor path decision at this chunk.
                        "best_path_pair": pair_name(best_vlm_bits, best_action_bits),
                        "best_path_vlm_a_bits": int(best_vlm_bits),
                        "best_path_action_a_bits": int(best_action_bits),
                        "best_path_next_vlm_a_bits": int(best_next_vlm_bits),
                        "best_path_action_label_id": int(best_action_label_id),
                        "best_path_next_vlm_label_id": int(best_next_vlm_label_id),

                        # Pointer to the 9-pair counterfactual outcome table.
                        "outcome_group_id": outcome_group_id,
                        "outcome_source": str(outcome_source_path),
                        "outcome_num_pairs_expected": 9,
                        "outcome_key_fields": [
                            "case_id",
                            "chunk_idx",
                            "candidate_vlm_a_bits",
                            "candidate_action_a_bits",
                        ],
                        "outcome_expected_candidate_pairs": EXPECTED_CANDIDATE_PAIRS,

                        "step_start": int(step),
                        "noise_seed": int(noise_seed),
                        "is_best_path_vlm_branch": bool(collect_vlm_bits == int(best_vlm_bits)),
                    }
                )

            result = results[int(best_vlm_bits)]
            for action in np.asarray(result["actions"][:REPLAN_STEPS]):
                obs, _, done, _ = env_step(env, action)
                step += 1
                if done:
                    break

            counts[pair_name(best_vlm_bits, best_action_bits)] += 1
            last_chunk = chunk_idx

            print(
                f"[COLLECT] {case_id} chunk={chunk_idx:04d} "
                f"exec={pair_name(best_vlm_bits, best_action_bits)} "
                f"next_vlm={best_next_vlm_bits} "
                f"step={step} done={done}",
                flush=True,
            )

        feature_index_path = case_root / "feature_index.jsonl"
        write_jsonl(feature_index_path, feature_index_rows)

        save_json(
            case_root / "state_summary.json",
            {
                "case_id": case_id,
                "task_id": task_id,
                "episode_idx": episode_idx,
                "schedule": str(schedule_path),
                "recovery_schedule": str(schedule_path),
                "trial_summary": str(schedule_path.parent / "trial_summary.jsonl"),
                "candidate_outcomes": str(schedule_path.parent / "candidate_outcomes.jsonl"),
                "outcome_source_used": str(outcome_source_path),
                "feature_index": str(feature_index_path),
                "a4_bank": str(bank_path),

                "success": bool(done),
                "last_chunk": last_chunk,
                "step_end": int(step),
                "counts": dict(counts),

                "client_data_root": str(CLIENT_DATA_ROOT),
                "server_mid_root": str(SERVER_MID_ROOT),
                "state_dir": str(state_dir),
                "num_written_mid_files_checked": len(written_mid_files),

                "mid_dirs": {
                    "w4a4": str(SERVER_MID_ROOT / "cases" / case_id / "mid" / "w4a4"),
                    "w4a8": str(SERVER_MID_ROOT / "cases" / case_id / "mid" / "w4a8"),
                    "w4a16": str(SERVER_MID_ROOT / "cases" / case_id / "mid" / "w4a16"),
                },

                "collection_mode": "mode7_selector_context_tokens_feature_index_best_path_decision_outcome_pointer",

                "feature_semantics": {
                    "server_mid_npz": (
                        "primary clean feature file: selector_context_tokens/prefix_input_embs "
                        "+ selector_context_mask/prefix_pad_mask + server_state "
                        "for each case/chunk/current_vlm_a_bits"
                    ),
                    "feature_index_jsonl": (
                        "index file linking each server feature file to current_vlm_a_bits, "
                        "best-path action bit, best-path next VLM bit, and the outcome group "
                        "containing 9 candidate pair results"
                    ),
                    "outcome_source": (
                        "JSONL table containing 9 precision-pair counterfactual outcomes per anchor chunk; "
                        "merge by case_id + chunk_idx + candidate_vlm_a_bits + candidate_action_a_bits"
                    ),
                    "client_state_npz": (
                        "debug raw 8-dim state only; not primary training state"
                    ),
                },

                "index_schema": {
                    "current_vlm_a_bits": "VLM bits used to produce this server feature file",
                    "best_path_action_a_bits": "action bits used by the validated best/anchor schedule at this chunk",
                    "best_path_next_vlm_a_bits": "VLM bits used by the validated best/anchor schedule at the next chunk; -1 for final chunk",
                    "outcome_group_id": "case/chunk group id for 9-pair outcome lookup",
                    "outcome_key_fields": [
                        "case_id",
                        "chunk_idx",
                        "candidate_vlm_a_bits",
                        "candidate_action_a_bits",
                    ],
                },
            },
        )

    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule_json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    collect(Path(args.schedule_json), host=args.host, port=int(args.port))


if __name__ == "__main__":
    main()

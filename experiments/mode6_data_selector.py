#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as websocket_policy


DATASET_ROOT = Path("/share/chengyuxuan-local/openpi/recovery_data_selector")
A4_BANK_ROOT = Path("/home/chengyuxuan/openpi/experiments/smooth/baseline/trace/action_chunk")

TASK_SUITE_NAME = "libero_10"
SEED = 7
RESIZE_SIZE = 224
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = 520

DEBUG_NOISE_HORIZON = 10
DEBUG_NOISE_DIM = 32

LABEL_MAP = {"w4a4": 0, "w4a8": 1, "w4a16": 2}
PRECISIONS = ("w4a4", "w4a8", "w4a16")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.load(open(path, "r", encoding="utf-8"))


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


def normalize_precision(p: str) -> str:
    p = str(p).lower().strip()
    if p in ("a4", "w4a4"):
        return "w4a4"
    if p in ("a8", "w4a8"):
        return "w4a8"
    if p in ("a16", "w4a16"):
        return "w4a16"
    raise ValueError(f"bad precision: {p}")


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
        "observation_state": observation_state,
        "eef_pos": eef_pos.astype(np.float32),
        "eef_axis_angle": eef_axis_angle.astype(np.float32),
        "gripper_qpos": gripper_qpos.astype(np.float32),
    }
    return element, state, agent, wrist


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
    # New mode7 recovery_schedule.json stores banks here.
    bank_path = (schedule.get("bank_paths") or {}).get("w4a4")
    if bank_path and Path(bank_path).exists():
        return Path(bank_path)

    # Old recovery schedule compatibility.
    src_path = (schedule.get("source") or {}).get("a4_action_json")
    if src_path and Path(src_path).exists():
        return Path(src_path)

    target = f"{case_id}_w4a4_bank.json"
    matches = list(A4_BANK_ROOT.glob(f"*/w4a4/w4a4_action_bank/{target}")) or list(A4_BANK_ROOT.rglob(target))
    if not matches:
        raise FileNotFoundError(f"cannot find A4 bank for {case_id}")
    return sorted(matches, key=lambda p: (0 if "w4a4_action_bank" in str(p) else 1, len(str(p))))[0]


def selector_items(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    # Old recovery schedules used selector_chunk_schedule.
    # New mode7 schedules use chunk_schedule.
    raw = schedule.get("selector_chunk_schedule") or schedule.get("chunk_schedule")
    if not raw:
        raise KeyError("missing selector_chunk_schedule / chunk_schedule")

    out, prev = [], -1
    for x in raw:
        idx = int(x["chunk_idx"])
        if idx != prev + 1:
            raise ValueError(f"schedule is not contiguous: prev={prev}, current={idx}")

        precision = normalize_precision(x["precision"])
        out.append({
            "chunk_idx": idx,
            "precision": precision,
            "label_id": int(x.get("label_id", LABEL_MAP[precision])),
            "noise_seed": int(x["noise_seed"]),
        })
        prev = idx

    return out


def save_state(
    path: Path,
    state: dict[str, np.ndarray],
    *,
    chunk_idx: int,
    label_id: int,
    noise_seed: int,
    step_start: int,
    precision: str,
    next_label_id: int = -1,
    next_precision: str = "",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observation_state=state["observation_state"],
        eef_pos=state["eef_pos"],
        eef_axis_angle=state["eef_axis_angle"],
        gripper_qpos=state["gripper_qpos"],
        chunk_idx=np.asarray(chunk_idx, dtype=np.int64),
        # action precision label for current chunk.
        label_id=np.asarray(label_id, dtype=np.int64),
        action_label_id=np.asarray(label_id, dtype=np.int64),
        precision=np.asarray(str(precision)),
        action_precision=np.asarray(str(precision)),
        # next VLM precision label for chunk k+1; -1 for final chunk.
        next_label_id=np.asarray(next_label_id, dtype=np.int64),
        next_vlm_label_id=np.asarray(next_label_id, dtype=np.int64),
        next_precision=np.asarray(str(next_precision)),
        next_vlm_precision=np.asarray(str(next_precision)),
        noise_seed=np.asarray(noise_seed, dtype=np.int64),
        step_start=np.asarray(step_start, dtype=np.int64),
    )


def collect(schedule_path: Path, *, host: str, ports: str):
    schedule = load_json(schedule_path)
    case_id = str(schedule["case_id"])
    task_id = int(schedule["task_id"])

    if schedule.get("final_success") is False:
        raise RuntimeError(f"schedule final_success is false: {schedule_path}")

    items = selector_items(schedule)

    bank_path = find_a4_bank(case_id, schedule)
    bank = load_json(bank_path)

    case_root = DATASET_ROOT / "cases" / case_id
    pre_dir = case_root / "pre"
    state_dir = case_root / "state"
    pre_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    p4, p8, p16 = [int(x) for x in ports.split(",")]
    clients = {
        "w4a4": websocket_policy.WebsocketClientPolicy(host, p4),
        "w4a8": websocket_policy.WebsocketClientPolicy(host, p8),
        "w4a16": websocket_policy.WebsocketClientPolicy(host, p16),
    }

    env, env_task_description = make_env(task_id)
    task_description = str(schedule.get("instruction") or env_task_description)

    counts = {"w4a4": 0, "w4a8": 0, "w4a16": 0}
    done = False
    step = 0
    last_chunk = None

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

        for item_i, item in enumerate(items):
            if done or step >= MAX_STEPS + NUM_STEPS_WAIT:
                break

            chunk_idx = int(item["chunk_idx"])
            precision = normalize_precision(item["precision"])
            label_id = int(item["label_id"])
            noise_seed = int(item["noise_seed"])

            if item_i + 1 < len(items):
                next_precision = normalize_precision(items[item_i + 1]["precision"])
                next_label_id = int(items[item_i + 1].get("label_id", LABEL_MAP[next_precision]))
            else:
                next_precision = ""
                next_label_id = -1

            element_base, state, agent_img, wrist_img = make_policy_input(obs, task_description)

            save_state(
                state_dir / f"chunk{chunk_idx:04d}_state.npz",
                state,
                chunk_idx=chunk_idx,
                label_id=label_id,
                precision=precision,
                next_label_id=next_label_id,
                next_precision=next_precision,
                noise_seed=noise_seed,
                step_start=step,
            )
            save_png(pre_dir / f"chunk{chunk_idx:04d}_agent.png", agent_img)
            save_png(pre_dir / f"chunk{chunk_idx:04d}_wrist.png", wrist_img)

            # For the same pre-action observation, query all three servers.
            # Each server saves:
            #   cases/<case_id>/mid_prefix/<precision>/chunkXXXX.npz
            # using the model-side midprefix tag parser.
            results = {}
            mid_prefix_expected = {}
            for p in PRECISIONS:
                tag = f"{case_id}__midprefix_{p}__chunk{chunk_idx:04d}"
                element = dict(element_base)
                element["debug_noise"] = make_debug_noise(noise_seed)
                element["debug_collect_tag"] = tag

                results[p] = clients[p].infer(element)

                mid_prefix_expected[p] = str(
                    case_root / "mid_prefix" / p / f"chunk{chunk_idx:04d}.npz"
                )

            # Execute only the oracle/schedule precision. Other two calls are feature collection only.
            result = results[precision]
            for action in np.asarray(result["actions"][:REPLAN_STEPS]):
                obs, _, done, _ = env_step(env, action)
                step += 1
                if done:
                    break

            counts[precision] += 1
            last_chunk = chunk_idx
            print(
                f"[COLLECT] {case_id} chunk={chunk_idx:04d} "
                f"exec={precision} next_vlm_label={next_precision or 'none'} "
                f"step={step} done={done}",
                flush=True,
            )

        save_json(case_root / "state_summary.json", {
            "case_id": case_id,
            "schedule": str(schedule_path),
            "a4_bank": str(bank_path),
            "success": bool(done),
            "last_chunk": last_chunk,
            "step_end": int(step),
            "counts": counts,
            "collection_mode": "multi_precision_mid_prefix_execute_schedule_precision",
            "mid_prefix": {
                "w4a4": str(case_root / "mid_prefix" / "w4a4"),
                "w4a8": str(case_root / "mid_prefix" / "w4a8"),
                "w4a16": str(case_root / "mid_prefix" / "w4a16"),
            },
            "label_semantics": {
                "label_id": "current chunk action precision label from schedule",
                "next_vlm_label_id": "next chunk VLM precision label from schedule; -1 for final chunk",
            },
        })
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule_json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", default="8000,8008,8016")
    args = parser.parse_args()
    collect(Path(args.schedule_json), host=args.host, ports=args.ports)


if __name__ == "__main__":
    main()

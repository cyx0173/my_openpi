#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as websocket_client_policy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def get_obs_element(obs, task_description: str, resize_size: int):
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_raw, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_raw, resize_size, resize_size))

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def make_debug_noise(seed: int, horizon: int, dim: int):
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((int(horizon), int(dim))).astype(np.float32)


def add_noise_from_bank(element: dict, bank: dict | None, chunk_idx: int, args):
    if bank is None:
        return element, {"source": "none", "seed": None, "shape": None}

    chunks = bank.get("chunks", [])
    if chunk_idx < 0 or chunk_idx >= len(chunks):
        return element, {"source": "missing_chunk", "seed": None, "shape": None}

    chunk = chunks[chunk_idx]
    out = dict(element)

    if chunk.get("debug_noise") is not None:
        noise = np.asarray(chunk["debug_noise"], dtype=np.float32)
        out["debug_noise"] = noise.copy()
        return out, {
            "source": "bank_debug_noise",
            "seed": chunk.get("noise_seed"),
            "shape": list(noise.shape),
        }

    if chunk.get("noise_seed") is not None:
        seed = int(chunk["noise_seed"])
        noise = make_debug_noise(seed, args.debug_noise_horizon, args.debug_noise_dim)
        out["debug_noise"] = noise.copy()
        return out, {
            "source": "bank_noise_seed",
            "seed": seed,
            "shape": list(noise.shape),
        }

    return element, {"source": "none", "seed": None, "shape": None}


def case_name(task_id: int, episode_idx: int):
    return f"task{task_id:02d}_ep{episode_idx:03d}"


def candidate_bank_paths(args):
    case = case_name(args.task_id, args.episode_idx)
    root = pathlib.Path(args.bank_dir)

    names = [
        f"{case}_{args.bank_tag}_bank.json",
        f"{case}_w4a16_bank.json",
        f"{case}_w4a4_bank.json",
        f"{case}_bank.json",
        f"{case}.json",
    ]

    roots = [
        root,
        root / "action_bank",
        root / f"{args.bank_tag}_action_bank",
        root / "w4a4_action_bank",
        root / "w4a16_action_bank",
        root.parent / f"{args.bank_tag}_action_bank",
        root.parent / "w4a4_action_bank",
        root.parent / "w4a16_action_bank",
        root.parent / "action_bank",
    ]

    paths = []
    for r in roots:
        for n in names:
            paths.append(r / n)
        paths.extend(sorted(r.glob(f"{case}_*_bank.json")) if r.exists() else [])
    return paths


def load_bank(args):
    if not args.bank_dir:
        return None, None

    for p in candidate_bank_paths(args):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                bank = json.load(f)
            if not isinstance(bank.get("chunks", []), list):
                raise ValueError(f"invalid bank chunks: {p}")
            return bank, p

    raise FileNotFoundError(
        f"Cannot find action bank for {case_name(args.task_id, args.episode_idx)} under {args.bank_dir}"
    )


def get_chunk_actions(chunk: dict, replan_steps: int):
    for key in ("executed_actions", "chosen_actions", "actions", "action_chunk"):
        if key in chunk and chunk[key] is not None:
            return np.asarray(chunk[key][:replan_steps], dtype=np.float32).tolist()
    raise KeyError(f"Cannot find actions in bank chunk. keys={list(chunk.keys())}")


def reset_wait_replay_to_chunk(env, initial_state, bank: dict | None, target_chunk: int, args):
    env.reset()
    obs = env.set_init_state(initial_state)

    done = False
    t = 0
    while t < args.num_steps_wait:
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        t += 1
        if done:
            break

    if bank is None and target_chunk > 0:
        raise ValueError("target_chunk > 0 requires action bank replay.")

    replayed = []

    for chunk_idx in range(target_chunk):
        if done:
            break

        chunks = bank.get("chunks", [])
        if chunk_idx >= len(chunks):
            raise IndexError(f"bank missing chunk {chunk_idx}; target_chunk={target_chunk}")

        actions = get_chunk_actions(chunks[chunk_idx], args.replan_steps)
        executed = 0
        for action in actions:
            if t >= args.num_steps_wait + args.max_steps_run:
                break
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
            t += 1
            executed += 1
            if done:
                break

        replayed.append({
            "chunk_idx": chunk_idx,
            "num_actions": executed,
            "noise_seed": chunks[chunk_idx].get("noise_seed"),
        })

    return obs, done, t, replayed


def infer_actions(client, element, replan_steps: int):
    t0 = time.perf_counter()
    result = client.infer(element)
    dt = time.perf_counter() - t0
    actions = np.asarray(result["actions"][:replan_steps], dtype=np.float64)
    return actions, dt


def diff_actions(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"same_shape": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = np.abs(a - b)
    return {
        "same_shape": True,
        "shape": list(a.shape),
        "max_abs": float(d.max()) if d.size else 0.0,
        "mean_abs": float(d.mean()) if d.size else 0.0,
        "p95_abs": float(np.percentile(d, 95)) if d.size else 0.0,
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port-bypass", type=int, default=8010)
    p.add_argument("--port-fake-native", type=int, default=8011)
    p.add_argument("--port-fake-fp32", type=int, default=8012)

    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--task-id", type=int, default=9)
    p.add_argument("--episode-idx", type=int, default=4)
    p.add_argument("--chunks", default="0,1,2,5")

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--max-steps-run", type=int, default=520)

    p.add_argument("--bank-dir", default="/home/chengyuxuan/openpi/experiments/baseline/320steps/w4a16/w4a4_action_bank")
    p.add_argument("--bank-tag", default="w4a16")

    p.add_argument("--debug-noise-horizon", type=int, default=10)
    p.add_argument("--debug-noise-dim", type=int, default=32)

    p.add_argument("--out", default="/home/chengyuxuan/openpi/experiments/ablation/fake16_action_diff.json")
    return p.parse_args()


def main():
    args = parse_args()
    chunks = [int(x.strip()) for x in args.chunks.split(",") if x.strip()]

    bank, bank_path = load_bank(args)
    print(f"[bank] {bank_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    initial_state = initial_states[args.episode_idx]

    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    clients = {
        "bypass": websocket_client_policy.WebsocketClientPolicy(args.host, args.port_bypass),
        "fake_native": websocket_client_policy.WebsocketClientPolicy(args.host, args.port_fake_native),
        "fake_fp32": websocket_client_policy.WebsocketClientPolicy(args.host, args.port_fake_fp32),
    }

    records = []

    try:
        for chunk_idx in chunks:
            print("=" * 120)
            print(f"[chunk={chunk_idx}]")

            obs, done_before, t, replayed = reset_wait_replay_to_chunk(
                env, initial_state, bank, chunk_idx, args
            )

            control_step = max(0, int(t - args.num_steps_wait))
            element = get_obs_element(obs, task_description, args.resize_size)
            element, noise_meta = add_noise_from_bank(element, bank, chunk_idx, args)

            actions = {}
            infer_times = {}

            for name, client in clients.items():
                a, dt = infer_actions(client, element, args.replan_steps)
                actions[name] = a
                infer_times[name] = dt

            pair_diffs = {
                "bypass_vs_fake_native": diff_actions(actions["bypass"], actions["fake_native"]),
                "bypass_vs_fake_fp32": diff_actions(actions["bypass"], actions["fake_fp32"]),
                "fake_native_vs_fake_fp32": diff_actions(actions["fake_native"], actions["fake_fp32"]),
            }

            for k, v in pair_diffs.items():
                print(f"  {k:<28} max={v.get('max_abs')} mean={v.get('mean_abs')} p95={v.get('p95_abs')}")

            record = {
                "case": case_name(args.task_id, args.episode_idx),
                "task_id": args.task_id,
                "episode_idx": args.episode_idx,
                "chunk_idx": chunk_idx,
                "env_step": int(t),
                "control_step": int(control_step),
                "done_before_chunk": bool(done_before),
                "noise": noise_meta,
                "replayed": replayed,
                "infer_times": infer_times,
                "pair_diffs": pair_diffs,
                "actions": {k: v.tolist() for k, v in actions.items()},
            }
            records.append(record)

    finally:
        env.close()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "bank_path": str(bank_path),
        "records": records,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

    print("=" * 120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

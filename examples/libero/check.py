#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as websocket_policy  # noqa: E402

try:
    import OpenGL.raw.EGL._errors as _egl_errors  # noqa: E402

    _orig_mj_del = None
    _orig_egl_del = None

    def _patch_mujoco_egl_cleanup():
        global _orig_mj_del, _orig_egl_del
        import robosuite.renderers.context.egl_context as _egl_ctx
        import robosuite.utils.binding_utils as _binding

        if _orig_mj_del is not None or _orig_egl_del is not None:
            return

        _orig_mj_del = _binding.MjRenderContext.__del__
        _orig_egl_del = _egl_ctx.EGLGLContext.__del__

        def _safe_mj_del(self):
            try:
                _orig_mj_del(self)
            except _egl_errors.EGLError:
                pass

        def _safe_egl_del(self):
            try:
                _orig_egl_del(self)
            except _egl_errors.EGLError:
                pass

        _binding.MjRenderContext.__del__ = _safe_mj_del
        _egl_ctx.EGLGLContext.__del__ = _safe_egl_del

    _patch_mujoco_egl_cleanup()
except Exception:
    pass


TASK_SUITE_NAME = "libero_10"
SEED = 7
RESIZE_SIZE = 224
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = 520
LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

DEBUG_NOISE_HORIZON = 10
DEBUG_NOISE_DIM = 32

PRECISIONS = ("w4a4", "w4a8", "w4a16")
DEFAULT_ROOT = Path("/home/chengyuxuan/openpi/experiments/selector_dataset")


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_ports(ports: str) -> dict[str, int]:
    xs = [int(x) for x in str(ports).split(",") if x.strip()]
    if len(xs) != 3:
        raise ValueError(f"--ports must be a4,a8,a16, got {ports!r}")
    return {"w4a4": xs[0], "w4a8": xs[1], "w4a16": xs[2]}


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


def make_debug_noise(seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).standard_normal(
        (DEBUG_NOISE_HORIZON, DEBUG_NOISE_DIM)
    ).astype(np.float32)


def debug_noise_seed(task_id: int, episode_idx: int, chunk_idx: int, base_seed: int) -> int:
    return int(base_seed) + int(task_id) * 100000 + int(episode_idx) * 1000 + int(chunk_idx)


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def make_env_and_state(task_id: int, episode_idx: int):
    task_suite = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
    task = task_suite.get_task(int(task_id))
    initial_states = task_suite.get_task_init_states(int(task_id))
    initial_state = np.asarray(initial_states[int(episode_idx)])

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(SEED)
    return env, str(task.language), initial_state


def make_policy_input(obs: dict[str, Any], task_description: str) -> dict[str, Any]:
    # Same preprocessing as normal LIBERO/OpenPI rollout:
    # rotate 180 degrees, then resize_with_pad to 224.
    agent_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    agent = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent_raw, RESIZE_SIZE, RESIZE_SIZE)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_raw, RESIZE_SIZE, RESIZE_SIZE)
    )

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            quat_to_axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
        )
    ).astype(np.float32)

    return {
        "observation/image": agent,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(task_description),
    }


def check_tag_format(tag: str) -> None:
    if re.fullmatch(r"(.+)__chunk(\d+)", tag) is None:
        raise ValueError(f"bad tag format: {tag}")


def mid_path(root: Path, tag: str) -> Path:
    m = re.fullmatch(r"(.+)__chunk(\d+)", tag)
    if m is None:
        raise ValueError(f"bad tag format: {tag}")
    case_tag = m.group(1)
    chunk_idx = int(m.group(2))
    return root / "cases" / case_tag / "mid" / f"chunk{chunk_idx:04d}.npz"


def wait_for_file(path: Path, timeout_s: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(0.1)
    raise TimeoutError(f"mid file not created: {path}")


def load_mid(path: Path) -> dict[str, np.ndarray]:
    x = np.load(path)
    return {
        "hidden": np.asarray(x["vlm_prefix_last_hidden"], dtype=np.float32),
        "mask": np.asarray(x["prefix_pad_mask"]).astype(bool),
        "chunk_idx": np.asarray(x["chunk_idx"]),
    }


def masked_mean(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = hidden[mask].astype(np.float64)
    if valid.size == 0:
        raise ValueError("empty valid tokens")
    return valid.mean(axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return float("nan")
    return float(np.dot(a, b) / den)


def compare_mid(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    ha, hb = a["hidden"], b["hidden"]
    ma, mb = a["mask"], b["mask"]

    out: dict[str, Any] = {
        "shape_a": list(ha.shape),
        "shape_b": list(hb.shape),
        "mask_equal": bool(np.array_equal(ma, mb)),
        "valid_tokens_a": int(ma.sum()),
        "valid_tokens_b": int(mb.sum()),
    }

    if ha.shape != hb.shape:
        out["error"] = "hidden_shape_mismatch"
        return out

    m = ma & mb
    if int(m.sum()) == 0:
        out["error"] = "empty_valid_intersection"
        return out

    va = ha[m].astype(np.float64)
    vb = hb[m].astype(np.float64)
    d = va - vb

    flat_a = va.reshape(-1)
    flat_b = vb.reshape(-1)
    flat_d = d.reshape(-1)

    pa = masked_mean(ha, m)
    pb = masked_mean(hb, m)
    pd = pa - pb

    out.update(
        {
            "token_max_abs": float(np.max(np.abs(flat_d))),
            "token_mean_abs": float(np.mean(np.abs(flat_d))),
            "token_p95_abs": float(np.percentile(np.abs(flat_d), 95)),
            "token_rel_l2": float(np.linalg.norm(flat_d) / (np.linalg.norm(flat_b) + 1e-12)),
            "token_cosine": cosine(flat_a, flat_b),
            "pooled_max_abs": float(np.max(np.abs(pd))),
            "pooled_mean_abs": float(np.mean(np.abs(pd))),
            "pooled_rel_l2": float(np.linalg.norm(pd) / (np.linalg.norm(pb) + 1e-12)),
            "pooled_cosine": cosine(pa, pb),
            "allclose_1e-6": bool(np.allclose(va, vb, atol=1e-6, rtol=0.0)),
            "allclose_1e-4": bool(np.allclose(va, vb, atol=1e-4, rtol=0.0)),
        }
    )
    return out


def cleanup_check_dirs(root: Path, case_id: str) -> None:
    for precision in PRECISIONS:
        d = root / "cases" / f"{case_id}__prefixcheck_{precision}"
        if d.exists():
            shutil.rmtree(d)


def infer_all_servers(
    *,
    clients: dict[str, Any],
    ports: dict[str, int],
    root: Path,
    case_id: str,
    chunk_idx: int,
    element_base: dict[str, Any],
    noise_seed: int,
    tag_key: str,
    timeout_s: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str], dict[str, Any], dict[str, np.ndarray]]:
    mids: dict[str, dict[str, np.ndarray]] = {}
    mid_paths: dict[str, str] = {}
    timing: dict[str, Any] = {}
    actions: dict[str, np.ndarray] = {}

    for precision in PRECISIONS:
        tag = f"{case_id}__prefixcheck_{precision}__chunk{int(chunk_idx):04d}"
        check_tag_format(tag)

        out_path = mid_path(root, tag)
        if out_path.exists():
            out_path.unlink()

        element = dict(element_base)
        element["debug_noise"] = make_debug_noise(noise_seed)
        element[tag_key] = tag

        print(f"[SEND] chunk={chunk_idx:04d} {precision} port={ports[precision]} tag={tag}", flush=True)
        t0 = time.perf_counter()
        result = clients[precision].infer(element)
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0

        timing[precision] = {
            "roundtrip_ms": roundtrip_ms,
            "policy_timing": result.get("policy_timing"),
            "server_timing": result.get("server_timing"),
        }
        actions[precision] = np.asarray(result["actions"][:REPLAN_STEPS], dtype=np.float64)

        wait_for_file(out_path, timeout_s=timeout_s)
        mids[precision] = load_mid(out_path)
        mid_paths[precision] = str(out_path)

        h = mids[precision]["hidden"]
        m = mids[precision]["mask"]
        print(f"[MID]  chunk={chunk_idx:04d} {precision} hidden={h.shape} valid={int(m.sum())}", flush=True)

    return mids, mid_paths, timing, actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=9)
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--ports", required=True, help="a4,a8,a16, e.g. 8024,8032,8040")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="server-side selector dataset root for saved mid")
    ap.add_argument("--rollout-precision", default="w4a16", choices=list(PRECISIONS))
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--tag-key", default="debug_collect_tag", choices=["debug_collect_tag", "debug_tag"])
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument(
        "--out",
        default=None,
        help="default: <root>/prefix_hidden_checks/taskXX_epYYY_all_chunks.json",
    )
    args = ap.parse_args()

    task_id = int(args.task_id)
    episode_idx = int(args.episode_idx)
    case_id = f"task{task_id:02d}_ep{episode_idx:03d}"
    root = Path(args.root)
    ports = parse_ports(args.ports)
    rollout_precision = normalize_precision(args.rollout_precision)

    clients = {
        p: websocket_policy.WebsocketClientPolicy(args.host, ports[p])
        for p in PRECISIONS
    }

    if not args.keep:
        cleanup_check_dirs(root, case_id)

    env, task_description, initial_state = make_env_and_state(task_id, episode_idx)

    chunk_reports: list[dict[str, Any]] = []
    done = False
    step = 0
    chunk_idx = 0

    print("=" * 100)
    print(f"case_id={case_id}")
    print(f"task_description={task_description!r}")
    print(f"ports={ports}")
    print(f"rollout_precision={rollout_precision}")
    print(f"tag_key={args.tag_key}")
    print("=" * 100)

    try:
        env.reset()
        obs = env.set_init_state(initial_state)

        for _ in range(NUM_STEPS_WAIT):
            obs, _, done, _ = env_step(env, LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                raise RuntimeError("done during dummy wait")

        while step < int(args.max_steps) + NUM_STEPS_WAIT:
            if args.max_chunks is not None and chunk_idx >= int(args.max_chunks):
                break
            if done:
                break

            noise_seed = debug_noise_seed(task_id, episode_idx, chunk_idx, args.base_seed)
            element_base = make_policy_input(obs, task_description)

            mids, mid_paths, timing, actions = infer_all_servers(
                clients=clients,
                ports=ports,
                root=root,
                case_id=case_id,
                chunk_idx=chunk_idx,
                element_base=element_base,
                noise_seed=noise_seed,
                tag_key=args.tag_key,
                timeout_s=float(args.timeout_s),
            )

            comparisons = {
                "w4a4_vs_w4a8": compare_mid(mids["w4a4"], mids["w4a8"]),
                "w4a4_vs_w4a16": compare_mid(mids["w4a4"], mids["w4a16"]),
                "w4a8_vs_w4a16": compare_mid(mids["w4a8"], mids["w4a16"]),
            }

            worst_rel_l2 = max(v.get("token_rel_l2", float("inf")) for v in comparisons.values())
            worst_mean_abs = max(v.get("token_mean_abs", float("inf")) for v in comparisons.values())
            min_cos = min(v.get("token_cosine", -1.0) for v in comparisons.values())

            report = {
                "chunk_idx": int(chunk_idx),
                "step_start": int(step),
                "noise_seed": int(noise_seed),
                "rollout_precision": rollout_precision,
                "mid_paths": mid_paths,
                "server_timing": timing,
                "comparisons": comparisons,
                "summary": {
                    "worst_token_rel_l2": float(worst_rel_l2),
                    "worst_token_mean_abs": float(worst_mean_abs),
                    "min_token_cosine": float(min_cos),
                },
            }
            chunk_reports.append(report)

            print(
                f"[CHUNK-SUMMARY] chunk={chunk_idx:04d} "
                f"worst_rel_l2={worst_rel_l2:.6g} "
                f"worst_mean_abs={worst_mean_abs:.6g} "
                f"min_cos={min_cos:.9f}",
                flush=True,
            )

            # Advance environment with one chosen precision. This only defines the visited states.
            # Hidden comparison itself always uses the same obs for all three servers.
            for action in actions[rollout_precision].tolist():
                obs, _, done, _ = env_step(env, action)
                step += 1
                if done:
                    break

            chunk_idx += 1

    finally:
        env.close()

    if chunk_reports:
        worst_rel_l2_all = max(x["summary"]["worst_token_rel_l2"] for x in chunk_reports)
        worst_mean_abs_all = max(x["summary"]["worst_token_mean_abs"] for x in chunk_reports)
        min_cos_all = min(x["summary"]["min_token_cosine"] for x in chunk_reports)
    else:
        worst_rel_l2_all = float("nan")
        worst_mean_abs_all = float("nan")
        min_cos_all = float("nan")

    final_report = {
        "case_id": case_id,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "ports": ports,
        "rollout_precision": rollout_precision,
        "base_seed": int(args.base_seed),
        "tag_key": args.tag_key,
        "final_done": bool(done),
        "final_step": int(step),
        "num_chunks_checked": int(len(chunk_reports)),
        "overall_summary": {
            "worst_token_rel_l2": float(worst_rel_l2_all),
            "worst_token_mean_abs": float(worst_mean_abs_all),
            "min_token_cosine": float(min_cos_all),
        },
        "chunks": chunk_reports,
    }

    out_path = Path(args.out) if args.out else root / "prefix_hidden_checks" / f"{case_id}_all_chunks.json"
    save_json(out_path, final_report)

    print("=" * 100)
    print(f"[DONE] case={case_id} done={done} final_step={step} chunks={len(chunk_reports)}")
    print(f"[OVERALL] worst_token_rel_l2={worst_rel_l2_all:.6g}")
    print(f"[OVERALL] worst_token_mean_abs={worst_mean_abs_all:.6g}")
    print(f"[OVERALL] min_token_cosine={min_cos_all:.9f}")
    print(f"[REPORT] {out_path}")
    print("=" * 100)

    if not args.keep:
        cleanup_check_dirs(root, case_id)


if __name__ == "__main__":
    main()

import os
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

import collections
import dataclasses
import json
import logging
import math
import pathlib
import time
from typing import Any

import numpy as np
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


try:
    import OpenGL.raw.EGL._errors as _egl_errors

    _orig_mj_del = None
    _orig_egl_del = None

    def _patch_mujoco_egl_cleanup():
        global _orig_mj_del, _orig_egl_del
        import robosuite.utils.binding_utils as _binding
        import robosuite.renderers.context.egl_context as _egl_ctx

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


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"

    # Two servers are needed for this file:
    #   1) A16 main/rescue policy
    #   2) injected low-precision policy, default W4A4
    port_a16: int = 8004
    port_inject: int = 8002

    task_suite_name: str = "libero_10"
    task_ids: str = "9"
    episode_start: int = 0
    episode_end: int = 10
    seed: int = 7

    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_steps: int = 520
    success_thresholds: str = "320,420,520"

    main_policy: str = "w4a16"
    inject_policy: str = "w4a4"

    # This is the A16 baseline/action-bank directory.
    # It provides two things:
    #   1) nominal A16 prefix actions before the injected chunk
    #   2) original A16 noise_seed for each chunk
    a16_bank_dir: str = "/home/chengyuxuan/openpi/recovery/mode3/w4a16_action_bank"

    # inject_chunks means: on the nominal A16 trajectory, replace chunk k with one low-precision chunk.
    # k=0: no A16 prefix replay; call injected policy from the initial post-wait state.
    inject_chunks: str = "0,1,2,5,10,20"

    use_debug_noise: bool = True
    send_debug_noise: bool = True
    debug_noise_base_seed: int = 0
    debug_noise_horizon: int = 10
    debug_noise_dim: int = 32

    base_dir: str = "/home/chengyuxuan/openpi/experiments/recovery/mode2_a4_a16"
    skip_existing: bool = True


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _quat2axisangle(quat):
    quat = np.asarray(quat).copy()

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_obs_element(obs, task_description: str, resize_size: int):
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_raw, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_raw, resize_size, resize_size)
    )

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _resolve_task_ids(args: Args, num_tasks_in_suite: int):
    task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    return [x for x in task_ids if 0 <= x < num_tasks_in_suite]


def _resolve_episode_indices(args: Args, num_initial_states: int):
    start = max(0, int(args.episode_start))
    end = min(int(args.episode_end), num_initial_states)
    return list(range(start, end))


def _parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _a16_bank_path(args: Args, task_id: int, episode_idx: int):
    return (
        pathlib.Path(args.a16_bank_dir)
        / f"task{task_id:02d}_ep{episode_idx:03d}_{args.main_policy}_bank.json"
    )


def _load_a16_bank(args: Args, task_id: int, episode_idx: int):
    path = _a16_bank_path(args, task_id, episode_idx)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def _to_action_list(actions, replan_steps: int):
    return np.asarray(actions[:replan_steps]).tolist()


def _fallback_noise_seed(args: Args, task_id: int, episode_idx: int, chunk_idx: int):
    return (
        int(args.debug_noise_base_seed)
        + int(task_id) * 100000
        + int(episode_idx) * 1000
        + int(chunk_idx)
    )


def _noise_seed_from_a16_bank(bank, args: Args, task_id: int, episode_idx: int, chunk_idx: int):
    if chunk_idx < len(bank["chunks"]):
        seed = bank["chunks"][chunk_idx].get("noise_seed")
        if seed is not None:
            return int(seed), "a16_bank"
    return _fallback_noise_seed(args, task_id, episode_idx, chunk_idx), "fallback_formula"


def _make_debug_noise(args: Args, noise_seed: int):
    rng = np.random.default_rng(int(noise_seed))
    noise = rng.standard_normal(
        size=(int(args.debug_noise_horizon), int(args.debug_noise_dim))
    ).astype(np.float32)
    return noise


def _add_debug_noise(element, args: Args, noise_seed: int | None):
    if not args.use_debug_noise or noise_seed is None:
        return element, {"enabled": False, "seed": None, "shape": None}

    noise = _make_debug_noise(args, noise_seed)
    out = dict(element)
    if args.send_debug_noise:
        out["debug_noise"] = noise.copy()

    return out, {
        "enabled": True,
        "sent": bool(args.send_debug_noise),
        "seed": int(noise_seed),
        "shape": list(noise.shape),
    }


def _success_at(success: bool, done_control_step, thresholds):
    return {
        str(h): bool(success and done_control_step is not None and done_control_step <= h)
        for h in thresholds
    }


def _reset_env(env, initial_state, args: Args):
    env.reset()
    obs = env.set_init_state(initial_state)

    done = False
    t = 0
    while t < args.num_steps_wait:
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        t += 1
        if done:
            break

    return obs, done, t


def _replay_a16_prefix(env, a16_bank, inject_chunk: int, args: Args, *, obs, done: bool, t: int):
    """Replay nominal A16 chunks 0..inject_chunk-1 to reach the injection state."""
    prefix_chunks = []

    for chunk_idx in range(inject_chunk):
        if done:
            break

        chunk = a16_bank["chunks"][chunk_idx]
        actions = _to_action_list(chunk["executed_actions"], args.replan_steps)
        step_start = int(t)
        num_actions = 0

        for action in actions:
            if t >= args.max_steps + args.num_steps_wait:
                break
            obs, _, done, _ = env.step(action)
            t += 1
            num_actions += 1
            if done:
                break

        prefix_chunks.append(
            {
                "chunk_idx": int(chunk_idx),
                "step_start": step_start,
                "control_step_start": int(step_start - args.num_steps_wait),
                "noise_seed": chunk.get("noise_seed"),
                "num_actions": int(num_actions),
            }
        )

    return obs, done, t, prefix_chunks


def _run_injected_chunk(
    env,
    task_description: str,
    inject_client,
    a16_bank,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    inject_chunk: int,
    obs,
    done: bool,
    t: int,
):
    """Run exactly one injected low-precision chunk at chunk index inject_chunk."""
    if done:
        return obs, done, t, {
            "chunk_idx": int(inject_chunk),
            "skipped": True,
            "reason": "already_done_before_injection",
        }

    step_start = int(t)
    noise_seed, noise_source = _noise_seed_from_a16_bank(
        a16_bank, args, task_id, episode_idx, inject_chunk
    )

    element = _get_obs_element(obs, task_description, args.resize_size)
    element, noise_meta = _add_debug_noise(element, args, noise_seed)

    infer_start = time.time()
    result = inject_client.infer(element)
    infer_time = time.time() - infer_start

    actions = _to_action_list(result["actions"], args.replan_steps)
    num_actions = 0

    for action in actions:
        if t >= args.max_steps + args.num_steps_wait:
            break
        obs, _, done, _ = env.step(action)
        t += 1
        num_actions += 1
        if done:
            break

    injected = {
        "chunk_idx": int(inject_chunk),
        "policy": str(args.inject_policy),
        "step_start": step_start,
        "control_step_start": int(step_start - args.num_steps_wait),
        "noise_seed": int(noise_seed),
        "noise_source": noise_source,
        "noise_meta": noise_meta,
        "infer_time": float(infer_time),
        "num_actions": int(num_actions),
        "executed_actions": actions,
        "done_after_injection": bool(done),
    }

    logging.info(
        f"        inject {args.inject_policy} chunk={inject_chunk:04d} "
        f"step={step_start:04d} noise_seed={noise_seed} infer={infer_time:.3f}s"
    )

    return obs, done, t, injected


def _run_a16_suffix(
    env,
    task_description: str,
    a16_client,
    a16_bank,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    next_chunk_idx: int,
    obs,
    done: bool,
    t: int,
):
    """After one injected chunk, continue online A16 from chunk inject_chunk+1."""
    chunk_idx = int(next_chunk_idx)
    action_plan = collections.deque()
    chunks = []
    start_time = time.time()

    while t < args.max_steps + args.num_steps_wait:
        if done:
            break

        if not action_plan:
            step_start = int(t)
            noise_seed, noise_source = _noise_seed_from_a16_bank(
                a16_bank, args, task_id, episode_idx, chunk_idx
            )

            element = _get_obs_element(obs, task_description, args.resize_size)
            element, noise_meta = _add_debug_noise(element, args, noise_seed)

            infer_start = time.time()
            result = a16_client.infer(element)
            infer_time = time.time() - infer_start

            actions = _to_action_list(result["actions"], args.replan_steps)
            action_plan.extend(actions)

            chunks.append(
                {
                    "chunk_idx": int(chunk_idx),
                    "policy": str(args.main_policy),
                    "step_start": step_start,
                    "control_step_start": int(step_start - args.num_steps_wait),
                    "noise_seed": int(noise_seed),
                    "noise_source": noise_source,
                    "noise_meta": noise_meta,
                    "infer_time": float(infer_time),
                    "executed_actions": actions,
                }
            )

            logging.info(
                f"        suffix {args.main_policy} chunk={chunk_idx:04d} "
                f"step={step_start:04d} noise_seed={noise_seed} infer={infer_time:.3f}s"
            )

            chunk_idx += 1

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action)
        t += 1

    done_control_step = int(t - args.num_steps_wait) if done else None

    return {
        "success": bool(done),
        "total_steps": int(t),
        "done_control_step": done_control_step,
        "num_a16_suffix_chunks": int(len(chunks)),
        "a16_suffix_time": float(time.time() - start_time),
        "chunks": chunks,
    }


def _run_mode2_episode(
    env,
    task_description: str,
    initial_state,
    a16_client,
    inject_client,
    a16_bank,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    inject_chunk: int,
):
    total_start = time.time()

    obs, done, t = _reset_env(env, initial_state, args)

    prefix_start = time.time()
    obs, done, t, a16_prefix_chunks = _replay_a16_prefix(
        env,
        a16_bank,
        inject_chunk,
        args,
        obs=obs,
        done=done,
        t=t,
    )
    a16_prefix_time = time.time() - prefix_start
    prefix_done_before_injection = bool(done)

    inject_start = time.time()
    obs, done, t, injected_chunk = _run_injected_chunk(
        env,
        task_description,
        inject_client,
        a16_bank,
        args,
        task_id=task_id,
        episode_idx=episode_idx,
        inject_chunk=inject_chunk,
        obs=obs,
        done=done,
        t=t,
    )
    inject_time = time.time() - inject_start
    done_after_injection = bool(done)

    suffix_result = _run_a16_suffix(
        env,
        task_description,
        a16_client,
        a16_bank,
        args,
        task_id=task_id,
        episode_idx=episode_idx,
        next_chunk_idx=inject_chunk + 1,
        obs=obs,
        done=done,
        t=t,
    )

    return {
        "success": bool(suffix_result["success"]),
        "total_steps": int(suffix_result["total_steps"]),
        "done_control_step": suffix_result["done_control_step"],
        "prefix_done_before_injection": prefix_done_before_injection,
        "done_after_injection": done_after_injection,
        "num_a16_prefix_chunks": int(len(a16_prefix_chunks)),
        "num_a16_suffix_chunks": int(suffix_result["num_a16_suffix_chunks"]),
        "a16_prefix_time": float(a16_prefix_time),
        "inject_time": float(inject_time),
        "a16_suffix_time": float(suffix_result["a16_suffix_time"]),
        "total_wall_time": float(time.time() - total_start),
        "a16_prefix_chunks": a16_prefix_chunks,
        "injected_chunk": injected_chunk,
        "a16_suffix_chunks": suffix_result["chunks"],
    }


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    base_dir = pathlib.Path(args.base_dir)
    result_dir = base_dir / f"inject_{args.inject_policy}_into_{args.main_policy}"
    result_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_int_list(args.success_thresholds)
    inject_chunks = _parse_int_list(args.inject_chunks)

    with open(base_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "a16_mainline_one_low_chunk_injection_then_a16_rescue",
                "task_suite_name": args.task_suite_name,
                "task_ids": args.task_ids,
                "episode_start": int(args.episode_start),
                "episode_end": int(args.episode_end),
                "seed": int(args.seed),
                "replan_steps": int(args.replan_steps),
                "num_steps_wait": int(args.num_steps_wait),
                "max_steps": int(args.max_steps),
                "success_thresholds": thresholds,
                "main_policy": args.main_policy,
                "inject_policy": args.inject_policy,
                "a16_bank_dir": args.a16_bank_dir,
                "inject_chunks": inject_chunks,
                "use_debug_noise": bool(args.use_debug_noise),
                "send_debug_noise": bool(args.send_debug_noise),
                "debug_noise_base_seed": int(args.debug_noise_base_seed),
                "debug_noise_shape": [
                    int(args.debug_noise_horizon),
                    int(args.debug_noise_dim),
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logging.info(f"Result dir: {result_dir}")
    logging.info(f"A16 bank dir: {args.a16_bank_dir}")
    logging.info(f"A16 server: ws://{args.host}:{args.port_a16}")
    logging.info(f"Inject server ({args.inject_policy}): ws://{args.host}:{args.port_inject}")

    a16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_a16)
    inject_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_inject)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    selected_task_ids = _resolve_task_ids(args, task_suite.n_tasks)

    logging.info(f"Selected task ids: {selected_task_ids}")

    summary_records = []

    for task_id in tqdm.tqdm(selected_task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        episode_indices = _resolve_episode_indices(args, len(initial_states))

        env = None
        try:
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            for episode_idx in tqdm.tqdm(
                episode_indices,
                desc=f"task{task_id:02d}",
                leave=False,
            ):
                try:
                    a16_bank, a16_bank_path = _load_a16_bank(args, task_id, episode_idx)
                except FileNotFoundError:
                    logging.warning(
                        f"[Skip] missing A16 bank: task{task_id:02d}_ep{episode_idx:03d}"
                    )
                    continue

                initial_state = np.array(
                    initial_states[episode_idx].tolist()
                    if hasattr(initial_states[episode_idx], "tolist")
                    else initial_states[episode_idx]
                )

                for inject_chunk in inject_chunks:
                    if inject_chunk >= len(a16_bank["chunks"]):
                        logging.warning(
                            f"[Skip] inject_chunk={inject_chunk} >= num_a16_bank_chunks={len(a16_bank['chunks'])}"
                        )
                        continue

                    out_path = (
                        result_dir
                        / f"task{task_id:02d}_ep{episode_idx:03d}_inject{inject_chunk:03d}_{args.inject_policy}_then_{args.main_policy}.json"
                    )

                    if out_path.exists() and args.skip_existing:
                        logging.info(f"[Skip] exists: {out_path}")
                        continue

                    logging.info(
                        f"[Mode2] task={task_id:02d} ep={episode_idx:03d} "
                        f"inject_chunk={inject_chunk:03d}: "
                        f"{args.main_policy} prefix -> {args.inject_policy} one chunk -> {args.main_policy} suffix"
                    )

                    result = _run_mode2_episode(
                        env,
                        task_description,
                        initial_state,
                        a16_client,
                        inject_client,
                        a16_bank,
                        args,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        inject_chunk=inject_chunk,
                    )

                    record = {
                        "task_id": int(task_id),
                        "task_description": str(task_description),
                        "episode_idx": int(episode_idx),
                        "seed": int(args.seed),
                        "initial_state_idx": int(episode_idx),
                        "inject_chunk": int(inject_chunk),
                        "main_policy": str(args.main_policy),
                        "inject_policy": str(args.inject_policy),
                        "a16_bank_path": str(a16_bank_path),
                        "a16_bank_success": bool(a16_bank.get("success", False)),
                        "success": bool(result["success"]),
                        "done": bool(result["success"]),
                        "total_steps": int(result["total_steps"]),
                        "done_control_step": result["done_control_step"],
                        "success_at": _success_at(
                            result["success"],
                            result["done_control_step"],
                            thresholds,
                        ),
                        "prefix_done_before_injection": bool(result["prefix_done_before_injection"]),
                        "done_after_injection": bool(result["done_after_injection"]),
                        "num_a16_prefix_chunks": int(result["num_a16_prefix_chunks"]),
                        "num_a16_suffix_chunks": int(result["num_a16_suffix_chunks"]),
                        "replan_steps": int(args.replan_steps),
                        "num_steps_wait": int(args.num_steps_wait),
                        "max_steps": int(args.max_steps),
                        "a16_prefix_time": float(result["a16_prefix_time"]),
                        "inject_time": float(result["inject_time"]),
                        "a16_suffix_time": float(result["a16_suffix_time"]),
                        "total_wall_time": float(result["total_wall_time"]),
                        "a16_prefix_chunks": result["a16_prefix_chunks"],
                        "injected_chunk": result["injected_chunk"],
                        "a16_suffix_chunks": result["a16_suffix_chunks"],
                    }

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)

                    summary_records.append(
                        {
                            "task_id": int(task_id),
                            "episode_idx": int(episode_idx),
                            "inject_chunk": int(inject_chunk),
                            "success": bool(record["success"]),
                            "done_control_step": record["done_control_step"],
                            "success_at": record["success_at"],
                            "prefix_done_before_injection": record["prefix_done_before_injection"],
                            "done_after_injection": record["done_after_injection"],
                            "path": str(out_path),
                        }
                    )

                    logging.info(
                        f"[Saved] {out_path} success={record['success']} "
                        f"done_control_step={record['done_control_step']}"
                    )

        finally:
            if env is not None:
                env.close()

    with open(base_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(eval_libero)

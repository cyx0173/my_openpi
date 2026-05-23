import os
import sys

# Must be set before importing libero / robosuite / mujoco / OpenGL
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

import imageio.v2 as imageio
import numpy as np
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


# Optional: silence EGL cleanup exception at process exit
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

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # Server ports.
    host: str = "127.0.0.1"

    # One 4-model group:
    # FP16 / W4A4 / W4A8 / W4A16
    port_fp16: int = 8000
    port_w4a4: int = 8001
    port_w4a8: int = 8002
    port_w4a16: int = 8003

    # LIBERO.
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7

    # Task / episode selection.
    # task_ids 为空时，使用 [task_id_start, task_id_end)。
    # task_id_end = -1 表示跑到 suite 末尾。
    #
    # 注意：LIBERO task_id 是 0-based。
    # 如果你说“task1”指第一个任务，则这里用 --task-ids 0。
    task_ids: str = ""
    task_id_start: int = 0
    task_id_end: int = -1

    # episode_end = -1 时，使用 [episode_start, episode_start + num_trials_per_task)。
    # 如果显式设置 episode_end，则使用 [episode_start, episode_end)。
    episode_start: int = 0
    episode_end: int = -1

    # 多 worker 并行时，用于区分输出 summary 和 traj 文件名。
    worker_id: int = 0

    # Policy input.
    resize_size: int = 224
    replan_steps: int = 5

    # Debug noise for deterministic diffusion sampling.
    # 每个 chunk 会生成一个固定 debug_noise，并同时传给 FP16/W4A4/W4A8/W4A16。
    # 这样动作差异只来自量化，而不是不同 diffusion noise。
    use_debug_noise: bool = True
    debug_noise_base_seed: int = 700000
    debug_noise_horizon: int = 10
    debug_noise_dim: int = 32

    # Output.
    base_dir: str = "/home/chengyuxuan/openpi/active_quant/mode1_data_select"
    output_subdir: str = "checkpoint_bank/data"
    obs_subdir: str = "checkpoint_bank/obs"
    noise_subdir: str = "checkpoint_bank/noise"
    video_subdir: str = "checkpoint_bank/videos"
    log_subdir: str = "logs/checkpoint_bank"

    # Control.
    skip_existing: bool = True
    save_failed_trajectories: bool = False
    save_video: bool = True


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


def _sanitize_filename(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(",", "")
        .replace(".", "")
        .replace("'", "")
    )[:100]


def _capture_mujoco_state(env) -> dict:
    mj = env.sim.get_state()
    return {
        "time": float(mj.time),
        "qpos": [float(x) for x in mj.qpos],
        "qvel": [float(x) for x in mj.qvel],
    }


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


def _save_chunk_observation(obs_dir: pathlib.Path, sample_id: str, element: dict) -> str:
    obs_dir.mkdir(parents=True, exist_ok=True)
    out_path = obs_dir / f"{sample_id}.npz"

    np.savez_compressed(
        out_path,
        image=element["observation/image"],
        wrist_image=element["observation/wrist_image"],
        state=element["observation/state"],
        prompt=np.array(element["prompt"]),
    )

    return str(out_path)


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
    if args.task_ids.strip():
        task_ids = [
            int(x.strip())
            for x in args.task_ids.split(",")
            if x.strip()
        ]
    else:
        end = num_tasks_in_suite if args.task_id_end < 0 else min(args.task_id_end, num_tasks_in_suite)
        task_ids = list(range(args.task_id_start, end))

    valid = []
    for task_id in task_ids:
        if task_id < 0 or task_id >= num_tasks_in_suite:
            logging.warning(
                f"[Skip] invalid task_id={task_id}, "
                f"valid range=[0,{num_tasks_in_suite})"
            )
            continue
        valid.append(task_id)
    return valid


def _resolve_episode_range(args: Args, num_initial_states: int):
    start = max(0, int(args.episode_start))

    if args.episode_end >= 0:
        end = min(int(args.episode_end), num_initial_states)
    else:
        end = min(start + int(args.num_trials_per_task), num_initial_states)

    if start >= end:
        return []

    return list(range(start, end))


def _make_debug_noise(args: Args, task_id: int, episode_idx: int, chunk_idx: int):
    """
    Generate deterministic diffusion noise for one chunk.

    同一个 task/episode/chunk 会得到同一个 seed/noise；
    四个 policy 使用完全相同的 noise。
    """
    noise_seed = (
        int(args.debug_noise_base_seed)
        + int(task_id) * 100000
        + int(episode_idx) * 1000
        + int(chunk_idx)
    )
    rng = np.random.default_rng(noise_seed)

    noise = rng.standard_normal(
        size=(int(args.debug_noise_horizon), int(args.debug_noise_dim))
    ).astype(np.float32)

    return noise_seed, noise


def _save_debug_noise(noise_dir: pathlib.Path, sample_id: str, noise: np.ndarray) -> str:
    noise_dir.mkdir(parents=True, exist_ok=True)
    out_path = noise_dir / f"{sample_id}_debug_noise.npy"
    np.save(out_path, noise)
    return str(out_path)


def _make_policy_elements_with_optional_noise(
    element: dict,
    *,
    args: Args,
    task_id: int,
    episode_idx: int,
    chunk_idx: int,
    sample_id: str,
    noise_dir: pathlib.Path,
):
    """
    Return:
        element_fp16, element_w4a4, element_w4a8, element_w4a16, debug_noise_meta
    """
    if not args.use_debug_noise:
        debug_noise_meta = {
            "enabled": False,
            "seed": None,
            "shape": None,
            "path": "",
            "base_seed": int(args.debug_noise_base_seed),
        }
        return element, element, element, element, debug_noise_meta

    noise_seed, debug_noise = _make_debug_noise(
        args=args,
        task_id=task_id,
        episode_idx=episode_idx,
        chunk_idx=chunk_idx,
    )
    noise_path = _save_debug_noise(noise_dir, sample_id, debug_noise)

    debug_noise_meta = {
        "enabled": True,
        "seed": int(noise_seed),
        "shape": list(debug_noise.shape),
        "path": noise_path,
        "base_seed": int(args.debug_noise_base_seed),
        "horizon": int(args.debug_noise_horizon),
        "dim": int(args.debug_noise_dim),
        "formula": "base_seed + task_id*100000 + episode_idx*1000 + chunk_idx",
    }

    element_fp16 = dict(element)
    element_w4a4 = dict(element)
    element_w4a8 = dict(element)
    element_w4a16 = dict(element)

    # Important: pass the same debug_noise to all four policies.
    element_fp16["debug_noise"] = debug_noise.copy()
    element_w4a4["debug_noise"] = debug_noise.copy()
    element_w4a8["debug_noise"] = debug_noise.copy()
    element_w4a16["debug_noise"] = debug_noise.copy()

    return element_fp16, element_w4a4, element_w4a8, element_w4a16, debug_noise_meta


def _to_action_list(actions, replan_steps: int):
    return np.asarray(actions[:replan_steps]).tolist()


def _action_stats(fp16_actions, w4a4_actions, w4a8_actions, w4a16_actions):
    fp16 = np.asarray(fp16_actions, dtype=np.float32)

    def _one(name: str, acts):
        arr = np.asarray(acts, dtype=np.float32)
        diff = arr - fp16
        return {
            f"{name}_shape": list(arr.shape),
            f"{name}_mean": float(arr.mean()) if arr.size else None,
            f"{name}_std": float(arr.std()) if arr.size else None,
            f"{name}_abs_max": float(np.abs(arr).max()) if arr.size else None,
            f"{name}_diff_l2_to_fp16": float(np.linalg.norm(diff)) if arr.size else None,
            f"{name}_diff_mean_to_fp16": float(np.abs(diff).mean()) if arr.size else None,
            f"{name}_diff_max_to_fp16": float(np.abs(diff).max()) if arr.size else None,
        }

    stats = {}
    stats.update(_one("w4a4", w4a4_actions))
    stats.update(_one("w4a8", w4a8_actions))
    stats.update(_one("w4a16", w4a16_actions))
    return stats


def _append_current_chunk(
    chunks,
    *,
    chunk_step_start: int,
    current_fp16,
    current_w4a4,
    current_w4a8,
    current_w4a16,
    current_chunk_state,
    current_obs_npz_path: str,
    current_obs_state,
    current_prompt: str,
    current_debug_noise_meta,
):
    if not current_fp16:
        return

    chunk_idx = len(chunks)
    step_end = chunk_step_start + len(current_fp16)

    stats = _action_stats(
        current_fp16,
        current_w4a4,
        current_w4a8,
        current_w4a16,
    )

    chunks.append(
        {
            "chunk_idx": int(chunk_idx),
            "step_start": int(chunk_step_start),
            "step_end": int(step_end),
            "num_actions": int(len(current_fp16)),

            "mujoco_state": current_chunk_state,

            "obs_npz_path": current_obs_npz_path,
            "observation_state": current_obs_state,
            "prompt": current_prompt,

            "debug_noise": current_debug_noise_meta,

            "actions": {
                "fp16": current_fp16,
                "w4a4": current_w4a4,
                "w4a8": current_w4a8,
                "w4a16": current_w4a16,
            },

            # Backward-compatible fields for old Mode 6 recovery code.
            "fp16_actions": current_fp16,
            "w4a4_actions": current_w4a4,
            "w4a8_actions": current_w4a8,
            "w4a16_actions": current_w4a16,

            "action_stats": stats,
        }
    )


def _save_video(video_dir, traj_name, tag, images, success):
    if not images:
        return
    pathlib.Path(video_dir).mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    out_path = pathlib.Path(video_dir) / f"{traj_name}_{tag}_{suffix}.mp4"
    imageio.mimwrite(out_path, images, fps=10)
    logging.info(f"    [Video] {out_path}")


def _run_four_track(
    env,
    task_description: str,
    initial_state,
    fp16_client,
    w4a4_client,
    w4a8_client,
    w4a16_client,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    traj_name: str,
    obs_dir: pathlib.Path,
    noise_dir: pathlib.Path,
):
    env.reset()
    obs = env.set_init_state(initial_state)

    done = False
    t = 0
    max_steps = _MAX_STEPS.get(args.task_suite_name, 520)

    fp16_plan = collections.deque()
    w4a4_plan = collections.deque()
    w4a8_plan = collections.deque()
    w4a16_plan = collections.deque()

    replay_images = []
    chunks = []

    current_fp16 = []
    current_w4a4 = []
    current_w4a8 = []
    current_w4a16 = []

    current_chunk_state = None
    current_obs_npz_path = ""
    current_obs_state = None
    current_prompt = str(task_description)
    current_debug_noise_meta = {
        "enabled": False,
        "seed": None,
        "shape": None,
        "path": "",
    }
    chunk_step_start = args.num_steps_wait

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

            if not fp16_plan:
                # Flush previous chunk.
                _append_current_chunk(
                    chunks,
                    chunk_step_start=chunk_step_start,
                    current_fp16=current_fp16,
                    current_w4a4=current_w4a4,
                    current_w4a8=current_w4a8,
                    current_w4a16=current_w4a16,
                    current_chunk_state=current_chunk_state,
                    current_obs_npz_path=current_obs_npz_path,
                    current_obs_state=current_obs_state,
                    current_prompt=current_prompt,
                    current_debug_noise_meta=current_debug_noise_meta,
                )

                # Current chunk start: save env state and observation.
                chunk_idx = len(chunks)
                chunk_step_start = t
                current_chunk_state = _capture_mujoco_state(env)

                element = _get_obs_element(obs, task_description, args.resize_size)
                sample_id = f"{traj_name}_chunk{chunk_idx:04d}_step{t:04d}"

                current_obs_npz_path = _save_chunk_observation(obs_dir, sample_id, element)
                current_obs_state = np.asarray(
                    element["observation/state"], dtype=np.float32
                ).tolist()
                current_prompt = str(task_description)

                (
                    element_fp16,
                    element_w4a4,
                    element_w4a8,
                    element_w4a16,
                    current_debug_noise_meta,
                ) = _make_policy_elements_with_optional_noise(
                    element,
                    args=args,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    chunk_idx=chunk_idx,
                    sample_id=sample_id,
                    noise_dir=noise_dir,
                )

                current_fp16 = []
                current_w4a4 = []
                current_w4a8 = []
                current_w4a16 = []

                # Same observation + same debug_noise, query four servers.
                fp16_result = fp16_client.infer(element_fp16)
                w4a4_result = w4a4_client.infer(element_w4a4)
                w4a8_result = w4a8_client.infer(element_w4a8)
                w4a16_result = w4a16_client.infer(element_w4a16)

                fp16_plan.extend(_to_action_list(fp16_result["actions"], args.replan_steps))
                w4a4_plan.extend(_to_action_list(w4a4_result["actions"], args.replan_steps))
                w4a8_plan.extend(_to_action_list(w4a8_result["actions"], args.replan_steps))
                w4a16_plan.extend(_to_action_list(w4a16_result["actions"], args.replan_steps))

                noise_seed = current_debug_noise_meta.get("seed", None)
                logging.info(
                    f"    task={task_id} ep={episode_idx} "
                    f"chunk={chunk_idx:04d} step={t:04d} "
                    f"noise_seed={noise_seed} "
                    f"actions: fp16={len(fp16_plan)}, "
                    f"w4a4={len(w4a4_plan)}, "
                    f"w4a8={len(w4a8_plan)}, "
                    f"w4a16={len(w4a16_plan)}"
                )

            fp16_action = fp16_plan.popleft()
            w4a4_action = w4a4_plan.popleft()
            w4a8_action = w4a8_plan.popleft()
            w4a16_action = w4a16_plan.popleft()

            # Important: environment always executes FP16.
            # This guarantees checkpoint bank follows teacher trajectory.
            obs, _, done, _ = env.step(fp16_action)

            current_fp16.append(fp16_action)
            current_w4a4.append(w4a4_action)
            current_w4a8.append(w4a8_action)
            current_w4a16.append(w4a16_action)

            t += 1

            if done:
                _append_current_chunk(
                    chunks,
                    chunk_step_start=chunk_step_start,
                    current_fp16=current_fp16,
                    current_w4a4=current_w4a4,
                    current_w4a8=current_w4a8,
                    current_w4a16=current_w4a16,
                    current_chunk_state=current_chunk_state,
                    current_obs_npz_path=current_obs_npz_path,
                    current_obs_state=current_obs_state,
                    current_prompt=current_prompt,
                    current_debug_noise_meta=current_debug_noise_meta,
                )

                return {
                    "chunks": chunks,
                    "success": True,
                    "total_steps": int(t),
                    "replay_images": replay_images,
                }

        except Exception as e:
            logging.exception(f"  Exception at step {t}: {e}")
            _append_current_chunk(
                chunks,
                chunk_step_start=chunk_step_start,
                current_fp16=current_fp16,
                current_w4a4=current_w4a4,
                current_w4a8=current_w4a8,
                current_w4a16=current_w4a16,
                current_chunk_state=current_chunk_state,
                current_obs_npz_path=current_obs_npz_path,
                current_obs_state=current_obs_state,
                current_prompt=current_prompt,
                current_debug_noise_meta=current_debug_noise_meta,
            )
            break

    _append_current_chunk(
        chunks,
        chunk_step_start=chunk_step_start,
        current_fp16=current_fp16,
        current_w4a4=current_w4a4,
        current_w4a8=current_w4a8,
        current_w4a16=current_w4a16,
        current_chunk_state=current_chunk_state,
        current_obs_npz_path=current_obs_npz_path,
        current_obs_state=current_obs_state,
        current_prompt=current_prompt,
        current_debug_noise_meta=current_debug_noise_meta,
    )

    return {
        "chunks": chunks,
        "success": bool(done),
        "total_steps": int(t),
        "replay_images": replay_images,
    }


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    base_dir = pathlib.Path(args.base_dir)
    output_dir = base_dir / args.output_subdir
    obs_dir = base_dir / args.obs_subdir
    noise_dir = base_dir / args.noise_subdir
    video_dir = base_dir / args.video_subdir
    log_dir = base_dir / args.log_subdir

    output_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Obs dir: {obs_dir}")
    logging.info(f"Noise dir: {noise_dir}")
    logging.info(f"Video dir: {video_dir}")
    logging.info(
        f"Server ports: fp16={args.port_fp16}, "
        f"w4a4={args.port_w4a4}, "
        f"w4a8={args.port_w4a8}, "
        f"w4a16={args.port_w4a16}"
    )
    logging.info(f"worker_id={args.worker_id}")
    logging.info(
        f"debug_noise: use={args.use_debug_noise}, "
        f"base_seed={args.debug_noise_base_seed}, "
        f"shape=({args.debug_noise_horizon}, {args.debug_noise_dim})"
    )

    fp16_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port_fp16
    )
    w4a4_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port_w4a4
    )
    w4a8_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port_w4a8
    )
    w4a16_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port_w4a16
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    selected_task_ids = _resolve_task_ids(args, num_tasks_in_suite)

    logging.info(f"Selected task ids: {selected_task_ids}")
    logging.info(
        f"Episode selection: episode_start={args.episode_start}, "
        f"episode_end={args.episode_end}, "
        f"num_trials_per_task={args.num_trials_per_task}"
    )

    summary = {
        "created_at": time.time(),
        "worker_id": int(args.worker_id),
        "task_suite_name": args.task_suite_name,
        "selected_task_ids": selected_task_ids,
        "episode_start": int(args.episode_start),
        "episode_end": int(args.episode_end),
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "num_trials_per_task": args.num_trials_per_task,
        "debug_noise": {
            "use_debug_noise": bool(args.use_debug_noise),
            "base_seed": int(args.debug_noise_base_seed),
            "horizon": int(args.debug_noise_horizon),
            "dim": int(args.debug_noise_dim),
            "formula": "base_seed + task_id*100000 + episode_idx*1000 + chunk_idx",
            "noise_subdir": args.noise_subdir,
        },
        "ports": {
            "fp16": args.port_fp16,
            "w4a4": args.port_w4a4,
            "w4a8": args.port_w4a8,
            "w4a16": args.port_w4a16,
        },
        "records": [],
    }

    for task_id in tqdm.tqdm(selected_task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        episode_indices = _resolve_episode_range(args, len(initial_states))

        logging.info(f"Task {task_id}: episode_indices={episode_indices}")

        if not episode_indices:
            logging.warning(f"[Skip] no valid episode indices for task_id={task_id}")
            continue

        for episode_idx in tqdm.tqdm(
            episode_indices,
            desc=f"task{task_id:02d}",
            leave=False,
        ):
            init_arr = initial_states[episode_idx]
            initial_state = np.array(
                init_arr.tolist() if hasattr(init_arr, "tolist") else init_arr
            )

            env = None
            try:
                env, task_description = _get_libero_env(
                    task,
                    LIBERO_ENV_RESOLUTION,
                    args.seed,
                )

                task_segment = _sanitize_filename(task_description)

                # Include worker_id to avoid file overwrite between parallel workers.
                traj_name = (
                    f"task{task_id:02d}_ep{episode_idx:03d}"
                    f"_seed{args.seed}_w{args.worker_id}_{task_segment}"
                )
                out_path = output_dir / f"{traj_name}_combined.json"

                if out_path.exists() and args.skip_existing:
                    logging.info(f"[Skip] exists: {out_path}")
                    continue

                logging.info(
                    f"Task {task_id}, episode {episode_idx}, "
                    f"worker {args.worker_id}: {task_description}"
                )

                result = _run_four_track(
                    env,
                    task_description,
                    initial_state,
                    fp16_client,
                    w4a4_client,
                    w4a8_client,
                    w4a16_client,
                    args,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    traj_name=traj_name,
                    obs_dir=obs_dir,
                    noise_dir=noise_dir,
                )

                if (not result["success"]) and (not args.save_failed_trajectories):
                    logging.warning(
                        f"  [Not saved] FP16 trajectory failed: "
                        f"task={task_id}, ep={episode_idx}, worker={args.worker_id}"
                    )
                    continue

                record = {
                    "schema_version": "chunk_bank_v3_task_parallel_debug_noise",
                    "created_at": time.time(),
                    "worker_id": int(args.worker_id),
                    "quant_profile": "staged_b_aggressive_atm_ohb",
                    "task_suite_name": args.task_suite_name,
                    "task_description": task_description,
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "initial_state": initial_state,
                    "success": bool(result["success"]),
                    "total_steps": int(result["total_steps"]),
                    "replan_steps": int(args.replan_steps),
                    "num_steps_wait": int(args.num_steps_wait),
                    "debug_noise": {
                        "use_debug_noise": bool(args.use_debug_noise),
                        "base_seed": int(args.debug_noise_base_seed),
                        "horizon": int(args.debug_noise_horizon),
                        "dim": int(args.debug_noise_dim),
                        "formula": "base_seed + task_id*100000 + episode_idx*1000 + chunk_idx",
                    },
                    "ports": {
                        "fp16": args.port_fp16,
                        "w4a4": args.port_w4a4,
                        "w4a8": args.port_w4a8,
                        "w4a16": args.port_w4a16,
                    },
                    "num_chunks": len(result["chunks"]),
                    "chunks": result["chunks"],
                }

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        record,
                        f,
                        indent=2,
                        cls=_NumpyEncoder,
                        ensure_ascii=False,
                    )

                logging.info(f"    Saved combined chunk bank: {out_path}")

                if args.save_video:
                    _save_video(
                        video_dir,
                        traj_name,
                        "fp16_teacher_combined",
                        result["replay_images"],
                        result["success"],
                    )

                summary["records"].append(
                    {
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "worker_id": int(args.worker_id),
                        "task_description": task_description,
                        "success": bool(result["success"]),
                        "num_chunks": len(result["chunks"]),
                        "path": str(out_path),
                    }
                )

            finally:
                if env is not None:
                    env.close()

    summary_path = output_dir / f"chunk_bank_summary_w{args.worker_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            cls=_NumpyEncoder,
            ensure_ascii=False,
        )

    logging.info(f"Done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(eval_libero)
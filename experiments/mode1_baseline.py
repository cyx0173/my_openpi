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

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 320,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port_w4a4: int = 8000

    task_suite_name: str = "libero_10"
    task_ids: str = "0,1,2,3,4,5,6,7,8,9"
    episode_start: int = 0
    episode_end: int = 10
    seed: int = 7

    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10

    use_debug_noise: bool = True
    debug_noise_base_seed: int = 0
    debug_noise_horizon: int = 10
    debug_noise_dim: int = 32

    base_dir: str = "/home/chengyuxuan/openpi/recovery/mode3"
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


def _make_debug_noise(args: Args, task_id: int, episode_idx: int, chunk_idx: int):
    noise_seed = int(args.debug_noise_base_seed) + int(chunk_idx)

    rng = np.random.default_rng(noise_seed)
    noise = rng.standard_normal(
        size=(int(args.debug_noise_horizon), int(args.debug_noise_dim))
    ).astype(np.float32)

    return noise_seed, noise


def _add_debug_noise(element: dict, args: Args, task_id: int, episode_idx: int, chunk_idx: int):
    if not args.use_debug_noise:
        return element, {
            "enabled": False,
            "seed": None,
            "shape": None,
        }

    noise_seed, debug_noise = _make_debug_noise(args, task_id, episode_idx, chunk_idx)

    out = dict(element)
    out["debug_noise"] = debug_noise.copy()

    return out, {
        "enabled": True,
        "seed": int(noise_seed),
        "shape": list(debug_noise.shape),
    }


def _to_action_list(actions, replan_steps: int):
    return np.asarray(actions[:replan_steps]).tolist()


def _run_w4a4_episode(
    env,
    task_description: str,
    initial_state,
    w4a4_client,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
):
    env.reset()
    obs = env.set_init_state(initial_state)

    done = False
    t = 0
    chunk_idx = 0
    action_plan = collections.deque()
    chunks = []

    max_steps = _MAX_STEPS.get(args.task_suite_name, 320)

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not action_plan:
            step_start = int(t)

            element = _get_obs_element(obs, task_description, args.resize_size)
            element, noise_meta = _add_debug_noise(
                element,
                args=args,
                task_id=task_id,
                episode_idx=episode_idx,
                chunk_idx=chunk_idx,
            )

            result = w4a4_client.infer(element)
            executed_actions = _to_action_list(result["actions"], args.replan_steps)

            action_plan.extend(executed_actions)

            chunks.append(
                {
                    "chunk_idx": int(chunk_idx),
                    "step_start": step_start,
                    "noise_seed": noise_meta["seed"],
                    "executed_actions": executed_actions,
                }
            )

            logging.info(
                f"    task={task_id} ep={episode_idx} "
                f"chunk={chunk_idx:04d} step={step_start:04d} "
                f"noise_seed={noise_meta['seed']}"
            )

            chunk_idx += 1

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action)
        t += 1

        if done:
            break

    return {
        "success": bool(done),
        "total_steps": int(t),
        "num_chunks": int(len(chunks)),
        "chunks": chunks,
    }


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    base_dir = pathlib.Path(args.base_dir)
    bank_dir = base_dir / "w4a4_action_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)

    config_path = base_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_suite_name": args.task_suite_name,
                "task_ids": args.task_ids,
                "episode_start": int(args.episode_start),
                "episode_end": int(args.episode_end),
                "seed": int(args.seed),
                "replan_steps": int(args.replan_steps),
                "num_steps_wait": int(args.num_steps_wait),
                "max_steps": int(_MAX_STEPS.get(args.task_suite_name, 320)),
                "precision": "w4a4",
                "use_debug_noise": bool(args.use_debug_noise),
                "debug_noise_base_seed": int(args.debug_noise_base_seed),
                "debug_noise_shape": [
                    int(args.debug_noise_horizon),
                    int(args.debug_noise_dim),
                ],
                "debug_noise_formula": "base_seed + chunk_idx",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logging.info(f"Bank dir: {bank_dir}")
    logging.info(f"W4A4 server: ws://{args.host}:{args.port_w4a4}")

    w4a4_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port_w4a4
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    selected_task_ids = _resolve_task_ids(args, task_suite.n_tasks)

    logging.info(f"Selected task ids: {selected_task_ids}")

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
                out_path = bank_dir / f"task{task_id:02d}_ep{episode_idx:03d}_w4a4_bank.json"

                if out_path.exists() and args.skip_existing:
                    logging.info(f"[Skip] exists: {out_path}")
                    continue

                initial_state = np.array(
                    initial_states[episode_idx].tolist()
                    if hasattr(initial_states[episode_idx], "tolist")
                    else initial_states[episode_idx]
                )

                result = _run_w4a4_episode(
                    env,
                    task_description,
                    initial_state,
                    w4a4_client,
                    args,
                    task_id=task_id,
                    episode_idx=episode_idx,
                )

                record = {
                    "task_id": int(task_id),
                    "task_description": str(task_description),
                    "episode_idx": int(episode_idx),
                    "seed": int(args.seed),
                    "initial_state_idx": int(episode_idx),
                    "initial_state": initial_state.tolist(),
                    "initial_state_shape": list(initial_state.shape),
                    "initial_state_dtype": str(initial_state.dtype),
                    "success": bool(result["success"]),
                    "total_steps": int(result["total_steps"]),
                    "num_chunks": int(result["num_chunks"]),
                    "replan_steps": int(args.replan_steps),
                    "num_steps_wait": int(args.num_steps_wait),
                    "chunks": result["chunks"],
                }

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)

                logging.info(
                    f"[Saved] {out_path} "
                    f"success={result['success']} chunks={result['num_chunks']}"
                )

        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(eval_libero)
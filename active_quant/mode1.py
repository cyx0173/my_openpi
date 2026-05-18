import os
import sys

# Must be set before importing libero / robosuite / mujoco / OpenGL
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "8"

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
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
    "libero_10": 320,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    port_w4a4: int = 8001
    port_w4a8: int = 8002
    port_w4a16: int = 8003

    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO parameters
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    # Utils
    output_dir: str = "/home/chengyuxuan/openpi/data/mode1/libero_10/data"
    video_dir: str = "/home/chengyuxuan/openpi/data/mode1/libero_10/videos"
    seed: int = 7


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
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
    )


def _capture_mujoco_state(env):
    mj = env.sim.get_state()
    return {
        "time": float(mj.time),
        "qpos": [float(x) for x in mj.qpos],
        "qvel": [float(x) for x in mj.qvel],
    }


def _append_current_chunk(
    chunks,
    chunk_step_start,
    current_fp16,
    current_w4a4,
    current_w4a8,
    current_w4a16,
    current_chunk_state,
):
    if not current_fp16:
        return

    chunks.append(
        {
            "chunk_idx": len(chunks),
            "step_start": chunk_step_start,
            "fp16_actions": current_fp16,
            "w4a4_actions": current_w4a4,
            "w4a8_actions": current_w4a8,
            "w4a16_actions": current_w4a16,
            "mujoco_state": current_chunk_state,
        }
    )


def _save_video(video_dir, traj_name, tag, images, success):
    if not images:
        return
    suffix = "success" if success else "failure"
    out_path = pathlib.Path(video_dir) / f"{traj_name}_{tag}_{suffix}.mp4"
    imageio.mimwrite(out_path, images, fps=10)
    logging.info(f"    [Video] {out_path}")


def _quat2axisangle(quat):
    """
    Copied from robosuite transform_utils.
    """
    quat = np.asarray(quat).copy()

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_obs_element(obs, task_description, resize_size):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


def _get_libero_env(task, resolution, seed):
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


def _to_action_list(actions, replan_steps):
    return np.asarray(actions[:replan_steps]).tolist()

def _run_four_track(
    env,
    task,
    task_description,
    initial_state,
    fp16_client,
    w4a4_client,
    w4a8_client,
    w4a16_client,
    args,
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
    chunk_step_start = args.num_steps_wait

    while t < max_steps + args.num_steps_wait:
        try:
            # Let objects settle.
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            replay_images.append(img)
            if not fp16_plan:
                _append_current_chunk(
                    chunks,
                    chunk_step_start,
                    current_fp16,
                    current_w4a4,
                    current_w4a8,
                    current_w4a16,
                    current_chunk_state,
                )
                current_chunk_state = _capture_mujoco_state(env)
                current_fp16 = []
                current_w4a4 = []
                current_w4a8 = []
                current_w4a16 = []
                chunk_step_start = t
                element = _get_obs_element(obs, task_description, args.resize_size)

                fp16_result = fp16_client.infer(element)
                w4a4_result = w4a4_client.infer(element)
                w4a8_result = w4a8_client.infer(element)
                w4a16_result = w4a16_client.infer(element)

                fp16_plan.extend(_to_action_list(fp16_result["actions"], args.replan_steps))
                w4a4_plan.extend(_to_action_list(w4a4_result["actions"], args.replan_steps))
                w4a8_plan.extend(_to_action_list(w4a8_result["actions"], args.replan_steps))
                w4a16_plan.extend(_to_action_list(w4a16_result["actions"], args.replan_steps))

            fp16_action = fp16_plan.popleft()
            w4a4_action = w4a4_plan.popleft()
            w4a8_action = w4a8_plan.popleft()
            w4a16_action = w4a16_plan.popleft()

            obs, reward, done, info = env.step(fp16_action)

            current_fp16.append(fp16_action)
            current_w4a4.append(w4a4_action)
            current_w4a8.append(w4a8_action)
            current_w4a16.append(w4a16_action)

            t += 1

            if done:
                _append_current_chunk(
                    chunks,
                    chunk_step_start,
                    current_fp16,
                    current_w4a4,
                    current_w4a8,
                    current_w4a16,
                    current_chunk_state,
                )
                return {
                    "chunks": chunks,
                    "success": bool(done),
                    "replay_images": replay_images,
                }

        except Exception as e:
            logging.exception(f"  Exception at step {t}: {e}")
            _append_current_chunk(
                chunks,
                chunk_step_start,
                current_fp16,
                current_w4a4,
                current_w4a8,
                current_w4a16,
                current_chunk_state,
            )
            break

    _append_current_chunk(
        chunks,
        chunk_step_start,
        current_fp16,
        current_w4a4,
        current_w4a8,
        current_w4a16,
        current_chunk_state,
    )

    return {
        "chunks": chunks,
        "success": bool(done),
        "replay_images": replay_images,
    }


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(f"Task suite: {args.task_suite_name}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = pathlib.Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    fp16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    w4a4_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_w4a4)
    w4a8_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_w4a8)
    w4a16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_w4a16)

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        init_arr = initial_states[0]
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
            suffix = "combined"
            out_path = output_dir / f"rollout_{task_segment}_success_{suffix}.json"

            logging.info(f"Task {task_id}: {task_description}")

            result = _run_four_track(
                env,
                task,
                task_description,
                initial_state,
                fp16_client,
                w4a4_client,
                w4a8_client,
                w4a16_client,
                args,
            )

            record = {
                "task_description": task_description,
                "task_id": task_id,
                "episode_idx": 0,
                "initial_state": initial_state,
                "success": bool(result["success"]),
                "replan_steps": args.replan_steps,
                "num_chunks": len(result["chunks"]),
                "chunks": result["chunks"],
            }

            with open(out_path, "w") as f:
                json.dump(record, f, indent=2, cls=_NumpyEncoder)

            logging.info(f"    Saved: {out_path}")

            vid_suffix = f"{suffix}_success" if result["success"] else f"{suffix}_failure"
            _save_video(
                video_dir,
                f"rollout_{task_segment}",
                vid_suffix,
                result["replay_images"],
                result["success"],
            )

        finally:
            if env is not None:
                env.close()

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
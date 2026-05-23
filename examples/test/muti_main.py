import os
import sys

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
import OpenGL.raw.EGL._errors as _egl_errors
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


_orig_mj_del = None
_orig_egl_del = None


def _append_jsonl(path, record):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _patch_mujoco_egl_cleanup():
    global _orig_mj_del, _orig_egl_del

    import robosuite.renderers.context.egl_context as _egl_ctx
    import robosuite.utils.binding_utils as _binding

    def _safe_mj_del(self):
        try:
            _orig_mj_del(self)
        except _egl_errors.EGLError:
            pass
        except AttributeError:
            pass

    def _safe_egl_del(self):
        try:
            _orig_egl_del(self)
        except _egl_errors.EGLError:
            pass
        except AttributeError:
            pass

    _orig_mj_del = _binding.MjRenderContext.__del__
    _orig_egl_del = _egl_ctx.EGLGLContext.__del__
    _binding.MjRenderContext.__del__ = _safe_mj_del
    _egl_ctx.EGLGLContext.__del__ = _safe_egl_del


_patch_mujoco_egl_cleanup()
del _patch_mujoco_egl_cleanup


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def make_seeded_noise(seed: int, action_horizon: int, action_dim: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(action_horizon, action_dim),
    ).astype(np.float32)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Server parameters
    #################################################################################################################
    host: str = "0.0.0.0"

    fp16_port: int = 8000
    w4a8_port: int = 8001
    w4a8_atm_port: int = 8002
    w4a8_atm_ohb_port: int = 8003
    w4a8_ohb_port: int = 8004

    #################################################################################################################
    # Inference parameters
    #################################################################################################################
    resize_size: int = 224
    replan_steps: int = 5
    action_horizon: int = 10
    internal_action_dim: int = 32
    noise_seed_base: int = 12345

    # 环境只执行哪个 server 的 action。建议固定 fp16。
    execute_role: str = "fp16"

    #################################################################################################################
    # LIBERO parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    seed: int = 7

    #################################################################################################################
    # Output
    #################################################################################################################
    video_out_path: str = "data/libero/videos_multi_compare"
    append_action_chunk: str = (
        "/home/chengyuxuan/openpi/lab_track/final/action_chunks/multi_compare_action_chunks.jsonl"
    )


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.append_action_chunk).parent.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    role_to_port = {
        "fp16": args.fp16_port,
        "w4a8": args.w4a8_port,
        "w4a8_atm": args.w4a8_atm_port,
        "w4a8_atm_ohb": args.w4a8_atm_ohb_port,
        "w4a8_ohb": args.w4a8_ohb_port,
    }

    if args.execute_role not in role_to_port:
        raise ValueError(f"execute_role must be one of {list(role_to_port)}, got {args.execute_role}")

    clients = {}
    for role, port in role_to_port.items():
        logging.info(f"Connecting {role} server at ws://{args.host}:{port}")
        clients[role] = _websocket_client_policy.WebsocketClientPolicy(args.host, port)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        chunk_idx = int(t - args.num_steps_wait) // args.replan_steps
                        noise_seed = int(args.noise_seed_base + chunk_idx)
                        noise = make_seeded_noise(
                            noise_seed,
                            args.action_horizon,
                            args.internal_action_dim,
                        )

                        base_element = {
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

                        action_chunks = {}
                        action_arrays = {}

                        # 同一个 observation + 同一个 debug_noise，同时请求所有 server。
                        for role, client in clients.items():
                            element = dict(base_element)
                            element["debug_noise"] = np.array(noise, dtype=np.float32, copy=True)

                            result = client.infer(element)
                            action_chunk = result["actions"]
                            action_array = np.asarray(action_chunk, dtype=np.float64)

                            action_chunks[role] = action_chunk
                            action_arrays[role] = action_array

                        fp16_actions = action_arrays["fp16"]

                        diffs_to_fp16 = {}
                        for role, arr in action_arrays.items():
                            if arr.shape != fp16_actions.shape:
                                raise RuntimeError(
                                    f"Action shape mismatch: fp16={fp16_actions.shape}, "
                                    f"{role}={arr.shape}"
                                )

                            diff = np.abs(arr - fp16_actions)
                            diffs_to_fp16[role] = {
                                "max": float(diff.max()),
                                "mean": float(diff.mean()),
                                "l2": float(np.linalg.norm(diff)),
                            }

                        record = {
                            "task_description": task_description,
                            "task_id": int(task_id),
                            "episode_idx": int(episode_idx),
                            "step": int(t),
                            "chunk_idx": int(chunk_idx),
                            "noise_seed": int(noise_seed),
                            "noise_shape": list(noise.shape),
                            "execute_role": args.execute_role,
                            "ports": {role: int(port) for role, port in role_to_port.items()},
                            "action_shape": list(fp16_actions.shape),
                            "diffs_to_fp16": diffs_to_fp16,
                            "actions": {
                                role: arr.tolist()
                                for role, arr in action_arrays.items()
                            },
                        }

                        _append_jsonl(args.append_action_chunk, record)

                        logging.info(
                            "Multi-compare chunk "
                            f"task={task_id} ep={episode_idx} step={t} chunk={chunk_idx} "
                            f"seed={noise_seed} | "
                            f"w4a8 mean={diffs_to_fp16['w4a8']['mean']:.6f}, "
                            f"atm mean={diffs_to_fp16['w4a8_atm']['mean']:.6f}, "
                            f"atm+ohb mean={diffs_to_fp16['w4a8_atm_ohb']['mean']:.6f}, "
                            f"ohb mean={diffs_to_fp16['w4a8_ohb']['mean']:.6f}"
                        )

                        execute_action_chunk = action_chunks[args.execute_role]

                        assert len(execute_action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but {args.execute_role} only predicts {len(execute_action_chunk)} steps."
                        )

                        # 环境只执行 execute_role，默认 FP16，保证后续 observation 都在同一条轨迹上。
                        action_plan.extend(execute_action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.exception(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"multi_compare_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


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


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
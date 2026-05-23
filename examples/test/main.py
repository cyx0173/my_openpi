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


def _append_action_chunk_jsonl(path, record):
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
    """Generate deterministic diffusion noise from a seed."""
    rng = np.random.default_rng(int(seed))
    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(action_horizon, action_dim),
    ).astype(np.float32)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"

    # FP16 teacher server
    teacher_port: int = 8000

    # W4A8 + ATM student server
    student_port: int = 8001

    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos_ohb_pair"
    append_action_chunk: str = (
        "/home/chengyuxuan/openpi/lab_track/atm_1/ohb_pair_action_chunks.jsonl"
    )

    # PI0.5 internal diffusion noise shape
    action_horizon: int = 10
    internal_action_dim: int = 32

    # Base seed for deterministic diffusion noise
    noise_seed_base: int = 12345

    seed: int = 7


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

    logging.info(
        f"Connecting teacher server at ws://{args.host}:{args.teacher_port}"
    )
    teacher_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.teacher_port,
    )

    logging.info(
        f"Connecting student server at ws://{args.host}:{args.student_port}"
    )
    student_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.student_port,
    )

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
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            img,
                            args.resize_size,
                            args.resize_size,
                        )
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            wrist_img,
                            args.resize_size,
                            args.resize_size,
                        )
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

                        teacher_element = dict(base_element)
                        student_element = dict(base_element)

                        # 两边必须拿到完全相同的 internal diffusion noise
                        teacher_element["debug_noise"] = np.array(
                            noise,
                            dtype=np.float32,
                            copy=True,
                        )
                        student_element["debug_noise"] = np.array(
                            noise,
                            dtype=np.float32,
                            copy=True,
                        )

                        teacher_result = teacher_client.infer(teacher_element)
                        student_result = student_client.infer(student_element)

                        teacher_action_chunk = teacher_result["actions"]
                        student_action_chunk = student_result["actions"]

                        teacher_action_chunk_np = np.asarray(
                            teacher_action_chunk,
                            dtype=np.float64,
                        )
                        student_action_chunk_np = np.asarray(
                            student_action_chunk,
                            dtype=np.float64,
                        )

                        if teacher_action_chunk_np.shape != student_action_chunk_np.shape:
                            raise RuntimeError(
                                "Teacher/student action shape mismatch: "
                                f"{teacher_action_chunk_np.shape} vs "
                                f"{student_action_chunk_np.shape}"
                            )

                        action_diff = np.abs(
                            teacher_action_chunk_np - student_action_chunk_np
                        )

                        _append_action_chunk_jsonl(
                            args.append_action_chunk,
                            {
                                "task_description": task_description,
                                "task_id": int(task_id),
                                "episode_idx": int(episode_idx),
                                "step": int(t),
                                "chunk_idx": int(chunk_idx),
                                "noise_seed": int(noise_seed),
                                "noise_shape": list(noise.shape),
                                "teacher_port": int(args.teacher_port),
                                "student_port": int(args.student_port),
                                "teacher_action_shape": list(
                                    teacher_action_chunk_np.shape
                                ),
                                "student_action_shape": list(
                                    student_action_chunk_np.shape
                                ),
                                "teacher_actions": teacher_action_chunk_np.tolist(),
                                "student_actions": student_action_chunk_np.tolist(),
                                "abs_diff_max": float(action_diff.max()),
                                "abs_diff_mean": float(action_diff.mean()),
                            },
                        )

                        logging.info(
                            "Pair chunk "
                            f"task={task_id} episode={episode_idx} "
                            f"step={t} chunk={chunk_idx} seed={noise_seed} "
                            f"diff_max={float(action_diff.max()):.6f} "
                            f"diff_mean={float(action_diff.mean()):.6f}"
                        )

                        assert len(teacher_action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but teacher policy only predicts {len(teacher_action_chunk)} steps."
                        )

                        # 环境只执行 teacher / FP16 action，保证后续轨迹由 teacher 决定
                        action_plan.extend(teacher_action_chunk[: args.replan_steps])

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
                / f"pair_rollout_{task_segment}_{suffix}.mp4",
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
            f"Current task success rate: "
            f"{float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: "
            f"{float(total_successes) / float(total_episodes)}"
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
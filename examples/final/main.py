import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

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
import time

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


def _append_jsonl(path: str, record: dict):
    if not path:
        return
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_task_ids(task_ids: str, num_tasks: int) -> list[int]:
    if not task_ids.strip():
        return list(range(num_tasks))

    out = []
    for part in task_ids.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))

    out = sorted(set(out))
    for task_id in out:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"Invalid task_id={task_id}, num_tasks={num_tasks}")
    return out


def _safe_name(text: str, max_len: int = 100) -> str:
    text = text.replace("/", "_").replace("\\", "_")
    text = "_".join(text.split())
    return text[:max_len]


def _patch_mujoco_egl_cleanup():
    global _orig_mj_del, _orig_egl_del
    import robosuite.renderers.context.egl_context as _egl_ctx
    import robosuite.utils.binding_utils as _binding

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

    _orig_mj_del = _binding.MjRenderContext.__del__
    _orig_egl_del = _egl_ctx.EGLGLContext.__del__
    _binding.MjRenderContext.__del__ = _safe_mj_del
    _egl_ctx.EGLGLContext.__del__ = _safe_egl_del


_patch_mujoco_egl_cleanup()
del _patch_mujoco_egl_cleanup


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    task_ids: str = ""  # "" means all tasks. Supports "0", "0,1,2", "0-7".
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    seed: int = 7
    trial_start_idx: int = 0

    #################################################################################################################
    # Screening / logging
    #################################################################################################################
    summary_out_path: str = ""
    run_tag: str = "w4a4_screen"
    precision_tag: str = "w4a4"
    layout_tag: str = "naive_vlm_action_selective"

    #################################################################################################################
    # Optional outputs
    #################################################################################################################
    save_video: bool = False
    video_out_path: str = "/share/chengyuxuan-local/openpi/active_quant/w4a4_screen_libero10/videos"

    # 默认不保存 action chunk。快速筛选阶段不要开。
    action_chunk_out_path: str = ""
    save_action_values: bool = False


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_order = _parse_task_ids(args.task_ids, num_tasks_in_suite)

    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Using task ids: {task_order}")
    logging.info(f"Server: ws://{args.host}:{args.port}")
    logging.info(f"Run tag: {args.run_tag}, precision={args.precision_tag}, layout={args.layout_tag}")

    if args.save_video:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.summary_out_path:
        pathlib.Path(args.summary_out_path).parent.mkdir(parents=True, exist_ok=True)

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

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(task_order):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0

        for local_episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            episode_idx = args.trial_start_idx + local_episode_idx
            episode_seed = int(args.seed + task_id * 1000 + episode_idx)
            initial_state_idx = int(episode_idx % len(initial_states))

            logging.info(f"\nTask {task_id}: {task_description}")
            logging.info(f"Starting episode {episode_idx + 1}, seed={episode_seed}, init_state={initial_state_idx}")

            done = False
            exception_msg = None
            t = 0
            chunk_id = 0
            replay_images = []
            episode_start_time = time.time()

            try:
                env.seed(episode_seed)
                env.reset()
                action_plan = collections.deque()

                obs = env.set_init_state(initial_states[initial_state_idx])

                while t < max_steps + args.num_steps_wait:
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

                    if args.save_video:
                        replay_images.append(img)

                    if not action_plan:
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        action_chunk_np = np.asarray(action_chunk)

                        if args.action_chunk_out_path:
                            record = {
                                "timestamp": time.time(),
                                "run_tag": args.run_tag,
                                "precision": args.precision_tag,
                                "layout": args.layout_tag,
                                "suite": args.task_suite_name,
                                "task_id": int(task_id),
                                "task_description": str(task_description),
                                "episode_idx": int(episode_idx),
                                "episode_seed": episode_seed,
                                "initial_state_idx": initial_state_idx,
                                "env_step": int(t),
                                "chunk_id": int(chunk_id),
                                "shape": list(action_chunk_np.shape),
                                "mean": float(action_chunk_np.mean()),
                                "std": float(action_chunk_np.std()),
                                "abs_max": float(np.abs(action_chunk_np).max()),
                            }
                            if args.save_action_values:
                                record["actions"] = action_chunk_np.tolist()
                            _append_jsonl(args.action_chunk_out_path, record)

                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )

                        action_plan.extend(action_chunk[: args.replan_steps])
                        chunk_id += 1

                    action = action_plan.popleft()

                    obs, reward, done, info = env.step(action.tolist())
                    t += 1

                    if done:
                        break

            except Exception as e:
                exception_msg = repr(e)
                logging.error(f"Caught exception: {e}")

            elapsed_sec = time.time() - episode_start_time

            task_episodes += 1
            total_episodes += 1

            if done:
                task_successes += 1
                total_successes += 1

            video_path = ""
            if args.save_video and replay_images:
                suffix = "success" if done else "failure"
                task_segment = _safe_name(task_description)
                video_path = str(
                    pathlib.Path(args.video_out_path)
                    / f"task{task_id:02d}_trial{episode_idx:03d}_seed{episode_seed}_{suffix}_{task_segment}.mp4"
                )
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

            summary_record = {
                "timestamp": time.time(),
                "run_tag": args.run_tag,
                "suite": args.task_suite_name,
                "task_id": int(task_id),
                "task_description": str(task_description),
                "episode_idx": int(episode_idx),
                "episode_seed": episode_seed,
                "initial_state_idx": initial_state_idx,
                "precision": args.precision_tag,
                "layout": args.layout_tag,
                "host": args.host,
                "port": int(args.port),
                "success": bool(done),
                "env_steps": int(t),
                "num_chunks": int(chunk_id),
                "elapsed_sec": float(elapsed_sec),
                "exception": exception_msg,
                "video_path": video_path,
            }
            _append_jsonl(args.summary_out_path, summary_record)

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
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
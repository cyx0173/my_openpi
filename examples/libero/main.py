import os
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "8"

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

import time
import collections
import dataclasses
import logging
import math
import pathlib
import imageio
import json

import numpy as np
import tqdm
import tyro
import OpenGL.raw.EGL._errors as _egl_errors
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


def make_debug_noise(seed: int, action_horizon: int = 10, action_dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((action_horizon, action_dim)).astype(np.float32)


def _parse_int_list(spec: str) -> list[int]:
    spec = str(spec or "").strip()
    if not spec:
        return []

    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Bad range in id list: {part}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))

    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _validate_indices(indices: list[int], total: int, name: str) -> list[int]:
    bad = [x for x in indices if x < 0 or x >= total]
    if bad:
        raise ValueError(f"{name} ids out of range [0, {total}): {bad}")
    return indices


def _select_task_indices(num_tasks: int, args) -> list[int]:
    explicit = _parse_int_list(args.task_ids)
    if explicit:
        return _validate_indices(explicit, num_tasks, "task")

    start = int(args.task_start)
    end = int(args.task_end)
    if end < 0:
        end = num_tasks

    if start < 0 or end < start:
        raise ValueError(f"Bad task range: task_start={start}, task_end={end}")

    return _validate_indices(list(range(start, min(end, num_tasks))), num_tasks, "task")


def _select_episode_indices(num_initial_states: int, args) -> list[int]:
    explicit = _parse_int_list(args.episode_ids)
    if explicit:
        return _validate_indices(explicit, num_initial_states, "episode")

    start = int(args.episode_start)
    end = int(args.episode_end)
    if end < 0:
        end = start + int(args.num_trials_per_task)

    if start < 0 or end < start:
        raise ValueError(f"Bad episode range: episode_start={start}, episode_end={end}")

    return _validate_indices(list(range(start, min(end, num_initial_states))), num_initial_states, "episode")


def _patch_mujoco_egl_cleanup():
    global _orig_mj_del, _orig_egl_del
    import robosuite.utils.binding_utils as _binding
    import robosuite.renderers.context.egl_context as _egl_ctx

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
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    task_start: int = 0
    task_end: int = -1
    task_ids: str = ""

    episode_start: int = 20
    episode_end: int = 21
    episode_ids: str = ""

    base_dir: str = "/home/chengyuxuan/openpi/lab_track/atm_1"
    policy_tag: str = "selector_pref_gate"

    save_video: bool = False
    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_indices = _select_task_indices(num_tasks_in_suite, args)
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Selected task ids: {task_indices}")

    base_dir = pathlib.Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    action_log_path = base_dir / "action_chunks.jsonl"
    episode_log_path = base_dir / "episode_results.jsonl"
    video_out_path = base_dir / "videos"
    if args.save_video:
        video_out_path.mkdir(parents=True, exist_ok=True)

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

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_indices, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        episode_indices = _select_episode_indices(len(initial_states), args)
        logging.info(f"Task {task_id}: selected episode/init-state ids: {episode_indices}")
        if not episode_indices:
            logging.warning(f"Task {task_id}: no selected episodes, skip.")
            continue

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(episode_indices, desc=f"episodes(task={task_id})"):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            success_step = None
            policy_chunk_idx = 0
            replay_images = [] if args.save_video else None

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

                    if args.save_video:
                        replay_images.append(img)

                    if not action_plan:
                        chunk_idx = int(policy_chunk_idx)
                        reset = chunk_idx == 0
                        noise_seed = int(task_id) * 100000 + int(episode_idx) * 1000 + chunk_idx

                        element = {
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
                            "debug_noise": make_debug_noise(noise_seed),
                            "debug_reset": reset,
                        }

                        t0 = time.perf_counter()
                        response = client.infer(element)
                        policy_chunk_idx += 1
                        client_infer_ms = (time.perf_counter() - t0) * 1000.0
                        action_chunk = response["actions"]
                        action_chunk_np = np.asarray(action_chunk)
                        server_timing = response.get("server_timing", {})
                        server_infer_ms = server_timing.get("infer_ms", None)
                        server_infer_str = "None" if server_infer_ms is None else f"{server_infer_ms:.2f}"
                        print(
                            f"server_infer={server_infer_str} ms "
                            f"client_roundtrip={client_infer_ms:.2f} ms",
                            flush=True,
                        )

                        _append_jsonl(
                            action_log_path,
                            {
                                "timestamp": time.time(),
                                "task_description": task_description,
                                "task_id": int(task_id),
                                "episode_idx": int(episode_idx),
                                "step": int(t),
                                "chunk_idx": int(chunk_idx),
                                "env_step_after_wait": int(t - args.num_steps_wait),
                                "policy_tag": args.policy_tag,
                                "shape": list(action_chunk_np.shape),
                                "mean": float(action_chunk_np.mean()),
                                "std": float(action_chunk_np.std()),
                                "abs_max": float(np.abs(action_chunk_np).max()),
                                "actions": action_chunk_np.tolist(),
                            },
                        )

                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    t += 1

                    if done:
                        success_step = int(t)
                        task_successes += 1
                        total_successes += 1
                        break

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            _append_jsonl(
                episode_log_path,
                {
                    "id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "success_step": None if success_step is None else int(success_step),
                },
            )

            if args.save_video:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                video_name = (
                    f"task{int(task_id):02d}_ep{int(episode_idx):03d}_"
                    f"{args.policy_tag}_{suffix}_{task_segment}.mp4"
                )
                imageio.mimwrite(
                    video_out_path / video_name,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
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

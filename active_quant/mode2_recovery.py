import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
import time
from collections import Counter
from typing import Any

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "8")

LIBERO_REPO = "/home/chengyuxuan/openpi/third_party/libero"
if LIBERO_REPO not in sys.path:
    sys.path.insert(0, LIBERO_REPO)

import imageio.v2 as imageio
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


try:
    from robosuite.models.robots.robot_model import create_robot as _create_robot_orig
    import robosuite.models.robots.robot_model

    def _safe_create_robot(robot_name, *args, **kwargs):
        try:
            return _create_robot_orig(robot_name, *args, **kwargs)
        except KeyError:
            import robosuite.models.robots.panda_model as pm
            return pm.Panda(idn=kwargs.get("idn", 0))

    robosuite.models.robots.robot_model.create_robot = _safe_create_robot
    for _mod in list(sys.modules.values()):
        if _mod is not None and hasattr(_mod, "create_robot"):
            _mod.create_robot = _safe_create_robot
except Exception:
    pass


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
PRECISION_ORDER = ["w4a4", "w4a8", "w4a16", "fp16"]

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port_fp16: int = 8000

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    seed: int = 7
    resize_size: int = 224
    replan_steps: int = 5

    bank_base_dir: str = "/home/chengyuxuan/openpi/active_quant/mode1_data_select"
    out_base_dir: str = "/home/chengyuxuan/openpi/active_quant/mode2_recovery_select"
    combined_subdir: str = "checkpoint_bank/data"
    output_subdir: str = "recovery_chunk"
    video_subdir: str = "recovery_chunk/videos"
    obs_subdir: str = "recovery_chunk/obs"
    vlm_hidden_subdir: str = "recovery_chunk/vlm_hidden"

    trajectory_list: str = ""
    task_id: int = -1
    num_shards: int = 1
    shard_id: int = 0

    save_vlm_hidden: bool = True
    vlm_hidden_mode: str = "full_tokens"
    vlm_hidden_dtype: str = "float16"
    vlm_hidden_tag: str = "fp16_recovery"

    worker_id: int = 0
    checkpoint_every: int = 20
    skip_existing: bool = True
    save_video: bool = False
    test_all_precisions: bool = False
    skip_failed_source_trajectories: bool = True


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


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_obs_element(obs, task_description: str, resize_size: int):
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_raw, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_raw, resize_size, resize_size))
    state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _save_obs_npz(obs_dir: pathlib.Path, sample_id: str, obs, task_description: str, resize_size: int) -> str:
    obs_dir.mkdir(parents=True, exist_ok=True)
    element = _get_obs_element(obs, task_description, resize_size)
    out_path = obs_dir / f"{sample_id}.npz"
    np.savez_compressed(
        out_path,
        image=element["observation/image"],
        wrist_image=element["observation/wrist_image"],
        state=element["observation/state"],
        prompt=np.array(element["prompt"]),
    )
    return str(out_path)


def _load_obs_npz_element(obs_npz_path: str, task_description: str):
    d = np.load(obs_npz_path, allow_pickle=True)
    return {
        "observation/image": d["image"],
        "observation/wrist_image": d["wrist_image"],
        "observation/state": d["state"],
        "prompt": str(task_description),
    }


def _state_from_obs_or_npz(obs, obs_npz_path: str, task_description: str, resize_size: int):
    if obs is not None:
        return np.asarray(
            _get_obs_element(obs, task_description, resize_size)["observation/state"],
            dtype=np.float32,
        )

    if not obs_npz_path:
        raise ValueError("obs is None and obs_npz_path is empty; cannot recover observation_state")

    d = np.load(obs_npz_path, allow_pickle=True)
    return np.asarray(d["state"], dtype=np.float32)


def _agentview_image_from_obs_or_npz(obs, obs_npz_path: str):
    if obs is not None:
        return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

    if not obs_npz_path:
        return None

    d = np.load(obs_npz_path, allow_pickle=True)
    if "image" not in d:
        return None

    return np.asarray(d["image"])


def _save_video(video_dir, traj_name: str, tag: str, images, success: bool):
    if not images:
        return
    pathlib.Path(video_dir).mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    out_path = pathlib.Path(video_dir) / f"{traj_name}_{tag}_{suffix}.mp4"
    imageio.mimwrite(out_path, [np.asarray(x) for x in images], fps=10)


def _get_eef_pos(env, obs):
    if obs is not None and "robot0_eef_pos" in obs and obs["robot0_eef_pos"] is not None:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    try:
        return env.sim.data.body_xpos[env.robot0_gripper_body_id].copy()
    except Exception:
        return None


def _get_chunk_actions(chunk: dict, precision: str):
    if "actions" in chunk and precision in chunk["actions"]:
        return chunk["actions"][precision]
    key = f"{precision}_actions"
    if key in chunk:
        return chunk[key]
    raise KeyError(f"Cannot find actions for precision={precision}")


def _restore_env_to_chunk_start(env, initial_state, chunks, chunk_idx: int, num_steps_wait: int):
    """
    Deterministically restore to the start of chunk_idx by replaying the
    original FP16 teacher trajectory from the episode initial state.

    This intentionally does NOT use chunk["mujoco_state"], because partial
    MuJoCo state restoration can miss robosuite/controller/internal states.
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    for _ in range(num_steps_wait):
        try:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        except ValueError:
            return obs, "reset_replay_crashed_wait"

        if done:
            return obs, "reset_replay_done_wait"

    for i in range(chunk_idx):
        for a in _get_chunk_actions(chunks[i], "fp16"):
            try:
                obs, _, done, _ = env.step(a)
            except ValueError:
                return obs, "reset_replay_crashed_fp16"

            if done:
                return obs, "reset_replay_done_fp16"

    return obs, "reset_fp16_replay"


def _execute_chunk(env, actions, images=None):
    steps = 0
    last_obs = None

    for a in actions:
        try:
            obs, _, done, _ = env.step(a)
        except ValueError:
            return {
                "done": False,
                "crash": True,
                "steps_executed": steps,
                "last_obs": last_obs,
            }

        last_obs = obs
        steps += 1

        if images is not None:
            images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

        if done:
            return {
                "done": True,
                "crash": False,
                "steps_executed": steps,
                "last_obs": last_obs,
            }

    return {
        "done": False,
        "crash": False,
        "steps_executed": steps,
        "last_obs": last_obs,
    }


def _load_chunk_debug_noise(chunk: dict):
    info = chunk.get("debug_noise", {})
    path = info.get("path", "")
    if not path:
        return None

    p = pathlib.Path(path)
    if not p.exists():
        logging.warning(f"[debug_noise] missing file: {p}")
        return None

    return np.load(p).astype(np.float32)


def _attach_vlm_feature_request(element: dict, *, save_path: pathlib.Path, tag: str, metadata: dict, args: Args):
    element = dict(element)
    element["__vlm_feature_request__"] = {
        "enabled": True,
        "save_path": str(save_path),
        "tag": tag,
        "mode": args.vlm_hidden_mode,
        "dtype": args.vlm_hidden_dtype,
        "save_embedding_hidden": False,
        "metadata": metadata,
    }
    return element


def _collect_vlm_hidden(
    fp16_client,
    obs,
    *,
    task_description: str,
    args: Args,
    save_path: pathlib.Path,
    metadata: dict,
    obs_npz_path: str,
    debug_noise=None,
):
    if not args.save_vlm_hidden:
        return {}, {}

    if obs is not None:
        element = _get_obs_element(obs, task_description, args.resize_size)
    else:
        element = _load_obs_npz_element(obs_npz_path, task_description)

    if debug_noise is not None:
        element["debug_noise"] = debug_noise

    element = _attach_vlm_feature_request(
        element,
        save_path=save_path,
        tag=args.vlm_hidden_tag,
        metadata=metadata,
        args=args,
    )

    result = fp16_client.infer(element)
    path = result.get("vlm_feature_path", "")
    if not path:
        return {}, {}

    return {"fp16": path}, {"fp16": result.get("vlm_feature_meta", {})}
def _diff_stats(name: str, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.shape != b.shape:
        logging.info(f"[debug-diff] {name}: shape mismatch a={a.shape}, b={b.shape}")
        return

    diff = a - b
    logging.info(
        f"[debug-diff] {name}: "
        f"mean={float(np.abs(diff).mean()):.6f} "
        f"max={float(np.abs(diff).max()):.6f} "
        f"l2={float(np.linalg.norm(diff)):.6f}"
    )


def _load_bank_policy_element(chunk: dict, task_description: str):
    path = chunk.get("obs_npz_path", "")
    if not path:
        return None

    p = pathlib.Path(path)
    if not p.exists():
        logging.warning(f"[debug-bank-obs] missing obs_npz_path: {p}")
        return None

    return _load_obs_npz_element(str(p), task_description)


def _compare_policy_element_to_bank(element: dict, bank_chunk: dict, *, task_description: str, tag: str):
    bank = _load_bank_policy_element(bank_chunk, task_description)
    if bank is None:
        return

    logging.info(
        f"[debug-bank-obs] {tag}: "
        f"bank_obs={bank_chunk.get('obs_npz_path', '')}"
    )

    _diff_stats(f"{tag}/state", element["observation/state"], bank["observation/state"])
    _diff_stats(f"{tag}/image", element["observation/image"], bank["observation/image"])
    _diff_stats(f"{tag}/wrist_image", element["observation/wrist_image"], bank["observation/wrist_image"])


def _compare_noise_to_bank(debug_noise, bank_chunk: dict, *, tag: str):
    info = bank_chunk.get("debug_noise", {})
    path = info.get("path", "")

    if debug_noise is None:
        logging.info(f"[debug-noise] {tag}: online noise is None, bank_path={path}")
        return

    if not path or not pathlib.Path(path).exists():
        logging.info(f"[debug-noise] {tag}: bank noise missing, bank_path={path}")
        return

    bank_noise = np.load(path).astype(np.float32)

    logging.info(
        f"[debug-noise] {tag}: "
        f"seed={info.get('seed', None)} "
        f"path={path}"
    )
    _diff_stats(f"{tag}/noise", debug_noise, bank_noise)

def _run_fp16_recovery(
    env,
    fp16_client,
    task_description: str,
    args: Args,
    *,
    obs,
    chunks,
    start_noise_chunk_idx: int,
    images=None,
    max_steps: int = 600,
    debug_online_vs_bank: bool = False,
):
    if obs is None:
        return {
            "success": False,
            "crash": False,
            "steps_taken": 0,
            "final_obs": None,
            "noise_start_chunk_idx": int(start_noise_chunk_idx),
            "noise_end_chunk_idx": int(start_noise_chunk_idx),
        }

    steps_taken = 0
    done = False
    final_obs = obs
    noise_chunk_idx = int(start_noise_chunk_idx)

    while not done and steps_taken < max_steps:
        if noise_chunk_idx >= len(chunks):
            return {
                "success": False,
                "crash": False,
                "steps_taken": steps_taken,
                "final_obs": final_obs,
                "noise_start_chunk_idx": int(start_noise_chunk_idx),
                "noise_end_chunk_idx": int(noise_chunk_idx),
            }

        try:
            element = _get_obs_element(obs, task_description, args.resize_size)

            debug_noise = _load_chunk_debug_noise(chunks[noise_chunk_idx])
            if debug_noise is not None:
                element["debug_noise"] = debug_noise

            if debug_online_vs_bank:
                _compare_policy_element_to_bank(
                    element,
                    chunks[noise_chunk_idx],
                    task_description=task_description,
                    tag=f"online_recovery/chunk{noise_chunk_idx:04d}",
                )

                _compare_noise_to_bank(
                    debug_noise,
                    chunks[noise_chunk_idx],
                    tag=f"online_recovery/chunk{noise_chunk_idx:04d}",
                )

            # 只 infer 一次：这个 actions 既用于真正执行，也用于 debug 比较
            action_chunk = fp16_client.infer(element)["actions"]
            actions = np.asarray(action_chunk[: args.replan_steps]).tolist()

            if debug_online_vs_bank:
                try:
                    bank_actions = np.asarray(
                        _get_chunk_actions(chunks[noise_chunk_idx], "fp16"),
                        dtype=np.float32,
                    )
                    online_actions = np.asarray(actions, dtype=np.float32)

                    n = min(len(online_actions), len(bank_actions))
                    if n > 0:
                        diff = online_actions[:n] - bank_actions[:n]
                        logging.info(
                            f"[online-vs-bank][fp16-only] noise_chunk={noise_chunk_idx} "
                            f"online_shape={list(online_actions.shape)} "
                            f"bank_shape={list(bank_actions.shape)} "
                            f"diff_mean={float(np.abs(diff).mean()):.6f} "
                            f"diff_max={float(np.abs(diff).max()):.6f} "
                            f"diff_l2={float(np.linalg.norm(diff)):.6f}"
                        )
                except Exception as e:
                    logging.warning(
                        f"[online-vs-bank][fp16-only] compare failed at "
                        f"noise_chunk={noise_chunk_idx}: {e}"
                    )

            noise_chunk_idx += 1

        except Exception as e:
            logging.warning(f"[FP16 recovery] inference failed: {e}")
            break

        for a in actions:
            try:
                obs, _, done, _ = env.step(a)
            except ValueError:
                return {
                    "success": False,
                    "crash": True,
                    "steps_taken": steps_taken,
                    "final_obs": final_obs,
                    "noise_start_chunk_idx": int(start_noise_chunk_idx),
                    "noise_end_chunk_idx": int(noise_chunk_idx),
                }

            final_obs = obs

            if images is not None:
                images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

            steps_taken += 1

            if done:
                return {
                    "success": True,
                    "crash": False,
                    "steps_taken": steps_taken,
                    "final_obs": final_obs,
                    "noise_start_chunk_idx": int(start_noise_chunk_idx),
                    "noise_end_chunk_idx": int(noise_chunk_idx),
                }

            if steps_taken >= max_steps:
                return {
                    "success": False,
                    "crash": False,
                    "steps_taken": steps_taken,
                    "final_obs": final_obs,
                    "noise_start_chunk_idx": int(start_noise_chunk_idx),
                    "noise_end_chunk_idx": int(noise_chunk_idx),
                }

    return {
        "success": bool(done),
        "crash": False,
        "steps_taken": steps_taken,
        "final_obs": final_obs,
        "noise_start_chunk_idx": int(start_noise_chunk_idx),
        "noise_end_chunk_idx": int(noise_chunk_idx),
    }


def _run_fp16_bank_replay_from_chunk(
    env,
    initial_state,
    chunks,
    *,
    chunk_idx: int,
    num_steps_wait: int,
    images=None,
    max_steps: int = 600,
):
    _restore_env_to_chunk_start(env, initial_state, chunks, chunk_idx, num_steps_wait)

    steps_taken = 0
    final_obs = None

    for future_chunk_idx in range(chunk_idx, len(chunks)):
        for a in _get_chunk_actions(chunks[future_chunk_idx], "fp16"):
            if steps_taken >= max_steps:
                return {
                    "success": False,
                    "crash": False,
                    "steps_taken": steps_taken,
                    "final_obs": final_obs,
                    "fallback": "fp16_bank_replay",
                }

            try:
                obs, _, done, _ = env.step(a)
            except ValueError:
                return {
                    "success": False,
                    "crash": True,
                    "steps_taken": steps_taken,
                    "final_obs": final_obs,
                    "fallback": "fp16_bank_replay",
                }

            final_obs = obs
            steps_taken += 1

            if images is not None:
                images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

            if done:
                return {
                    "success": True,
                    "crash": False,
                    "steps_taken": steps_taken,
                    "final_obs": final_obs,
                    "fallback": "fp16_bank_replay",
                }

    return {
        "success": False,
        "crash": False,
        "steps_taken": steps_taken,
        "final_obs": final_obs,
        "fallback": "fp16_bank_replay",
    }


def _build_data_point(
    *,
    traj_name: str,
    task_id: int,
    episode_idx: int,
    task_description: str,
    chunk: dict,
    obs_at_chunk_start,
    obs_npz_path: str,
    restore_method: str,
    required_precision: str,
    vlm_feature_paths: dict,
    vlm_feature_meta: dict,
    debug_noise_meta: dict,
    args: Args,
):
    observation_state = _state_from_obs_or_npz(
        obs_at_chunk_start,
        obs_npz_path or chunk.get("obs_npz_path", ""),
        task_description,
        args.resize_size,
    )
    chunk_idx = int(chunk["chunk_idx"])
    step_start = int(chunk.get("step_start", -1))
    step_end = int(chunk.get("step_end", step_start + int(chunk.get("num_actions", 0))))

    return {
        "schema_version": "chunk_selector_v1",
        "traj_name": traj_name,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "chunk_idx": chunk_idx,
        "step_start": step_start,
        "step_end": step_end,
        "observation_state": observation_state.tolist(),
        "vlm_feature_paths": vlm_feature_paths,
        "vlm_feature_meta": vlm_feature_meta,
        "required_precision": required_precision,
        "task_description": task_description,
        "obs_npz_path": obs_npz_path or chunk.get("obs_npz_path", ""),
        "restore_method": restore_method,
        "debug_noise": debug_noise_meta,
        "source_chunk_bank": {
            "traj_name": traj_name,
            "chunk_idx": chunk_idx,
        },
        "metadata": {
            "precision_order": PRECISION_ORDER,
            "label_definition": "minimum precision whose whole injected chunk followed by FP16 online recovery succeeds",
        },
    }


def _run_chunk_recovery_probe(
    env,
    *,
    initial_state,
    chunks,
    chunk_idx: int,
    task_description: str,
    task_id: int,
    episode_idx: int,
    traj_name: str,
    fp16_client,
    args: Args,
    obs_dir: pathlib.Path,
    vlm_hidden_dir: pathlib.Path,
    video_dir: pathlib.Path | None = None,
):
    chunk = chunks[chunk_idx]
    step_start = int(chunk.get("step_start", args.num_steps_wait))
    fp16_actions = _get_chunk_actions(chunk, "fp16")
    debug_noise = _load_chunk_debug_noise(chunk)
    debug_noise_meta = chunk.get("debug_noise", {})

    obs_for_data_point, restore_method_for_data_point = _restore_env_to_chunk_start(
        env, initial_state, chunks, chunk_idx, args.num_steps_wait
    )

    obs_npz_path = chunk.get("obs_npz_path", "")
    if not obs_npz_path:
        if obs_for_data_point is None:
            raise RuntimeError(
                f"chunk={chunk_idx} has no obs_npz_path, and direct mujoco restore does not return raw obs"
            )
        obs_npz_path = _save_obs_npz(
            obs_dir,
            f"{traj_name}_chunk{chunk_idx:04d}_recovery_obs",
            obs_for_data_point,
            task_description,
            args.resize_size,
        )

    vlm_feature_paths = {}
    vlm_feature_meta = {}
    if args.save_vlm_hidden:
        save_path = vlm_hidden_dir / "fp16" / f"{traj_name}_chunk{chunk_idx:04d}_step{step_start:04d}_fp16_vlm_hidden.npz"
        metadata = {
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "chunk_idx": int(chunk_idx),
            "step_start": int(step_start),
            "traj_name": traj_name,
            "task_description": str(task_description),
            "obs_npz_path": str(obs_npz_path),
            "source": "recovery_chunk_start",
            "debug_noise_path": debug_noise_meta.get("path", ""),
            "debug_noise_seed": debug_noise_meta.get("seed", None),
        }
        vlm_feature_paths, vlm_feature_meta = _collect_vlm_hidden(
            fp16_client,
            obs_for_data_point,
            task_description=task_description,
            args=args,
            save_path=save_path,
            metadata=metadata,
            obs_npz_path=obs_npz_path,
            debug_noise=debug_noise,
        )

    recovery_results = {}
    required_precision = "failed"

    for precision in PRECISION_ORDER:
        obs_at_chunk_start, _ = _restore_env_to_chunk_start(env, initial_state, chunks, chunk_idx, args.num_steps_wait)
        candidate_actions = _get_chunk_actions(chunk, precision)
        eef_before = _get_eef_pos(env, obs_at_chunk_start)
        if args.save_video:
            if obs_at_chunk_start is not None:
                images = [np.ascontiguousarray(obs_at_chunk_start["agentview_image"][::-1, ::-1])]
            else:
                images = []
        else:
            images = None

        inject_result = _execute_chunk(env, candidate_actions, images=images)

        if inject_result["crash"]:
            recovery_results[precision] = {
                "success": False,
                "crash": True,
                "inject_steps": int(inject_result["steps_executed"]),
                "fp16_recovery_steps": 0,
                "total_steps_after_restore": int(inject_result["steps_executed"]),
                "eef_delta": None,
            }
        else:
            if inject_result["done"]:
                recovery_info = {
                    "success": True,
                    "crash": False,
                    "steps_taken": 0,
                    "final_obs": inject_result.get("last_obs"),
                }
            else:
                max_total_steps = _MAX_STEPS.get(args.task_suite_name, 520) + args.num_steps_wait
                max_recovery_steps = max(1, max_total_steps - (step_start + int(inject_result["steps_executed"])))
                recovery_info = _run_fp16_recovery(
                    env,
                    fp16_client,
                    task_description,
                    args,
                    obs=inject_result.get("last_obs"),
                    chunks=chunks,
                    start_noise_chunk_idx=chunk_idx + 1,
                    images=images,
                    max_steps=max_recovery_steps,
                    debug_online_vs_bank=(precision == "fp16"),
                )

                if precision == "fp16" and not bool(recovery_info.get("success", False)):
                    logging.info(
                        f"  chunk={chunk_idx:04d} fp16 online recovery failed; "
                        "fallback to fp16 bank replay"
                    )
                    recovery_info = _run_fp16_bank_replay_from_chunk(
                        env,
                        initial_state,
                        chunks,
                        chunk_idx=chunk_idx,
                        num_steps_wait=args.num_steps_wait,
                        images=images,
                        max_steps=max_total_steps - step_start,
                    )

            final_obs = recovery_info.get("final_obs", inject_result.get("last_obs"))
            eef_after = _get_eef_pos(env, final_obs)
            eef_delta = float(np.linalg.norm(eef_after - eef_before)) if eef_before is not None and eef_after is not None else None
            fp16_ref = np.asarray(fp16_actions, dtype=np.float32)
            cand = np.asarray(candidate_actions, dtype=np.float32)
            recovery_results[precision] = {
                "success": bool(recovery_info["success"]),
                "crash": bool(recovery_info.get("crash", False)),
                "inject_steps": int(inject_result["steps_executed"]),
                "fp16_recovery_steps": int(recovery_info.get("steps_taken", 0)),
                "total_steps_after_restore": int(inject_result["steps_executed"] + recovery_info.get("steps_taken", 0)),
                "eef_delta": eef_delta,
                "recovery_noise_start_chunk_idx": recovery_info.get("noise_start_chunk_idx", None),
                "recovery_noise_end_chunk_idx": recovery_info.get("noise_end_chunk_idx", None),
                "recovery_fallback": recovery_info.get("fallback", ""),
                "chunk_diff_mean_to_fp16": float(np.abs(cand - fp16_ref).mean()) if cand.size else None,
                "chunk_diff_max_to_fp16": float(np.abs(cand - fp16_ref).max()) if cand.size else None,
            }

        logging.info(
            f"  chunk={chunk_idx:04d} {precision}: "
            f"success={recovery_results[precision]['success']}, "
            f"inject_steps={recovery_results[precision]['inject_steps']}, "
            f"recovery_steps={recovery_results[precision]['fp16_recovery_steps']}"
        )

        if args.save_video and video_dir is not None:
            _save_video(video_dir, traj_name, f"chunk{chunk_idx:04d}_{precision}", images, recovery_results[precision]["success"])

        if recovery_results[precision]["success"] and required_precision == "failed":
            required_precision = precision
            if not args.test_all_precisions:
                break

    return _build_data_point(
        traj_name=traj_name,
        task_id=task_id,
        episode_idx=episode_idx,
        task_description=task_description,
        chunk=chunk,
        obs_at_chunk_start=obs_for_data_point,
        obs_npz_path=obs_npz_path,
        restore_method=restore_method_for_data_point,
        required_precision=required_precision,
        vlm_feature_paths=vlm_feature_paths,
        vlm_feature_meta=vlm_feature_meta,
        debug_noise_meta=debug_noise_meta,
        args=args,
    )


def _load_checkpoint(output_dir: pathlib.Path, worker_id: int):
    checkpoint_path = output_dir / f"checkpoint_w{worker_id}.jsonl"
    completed = set()
    all_data = []
    entries = []

    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                completed.add((entry["traj_name"], int(entry["chunk_idx"])))
                all_data.append(entry["data_point"])
                entries.append(entry)

    return completed, all_data, entries


def _save_checkpoint(output_dir: pathlib.Path, new_entries, worker_id: int):
    with open(output_dir / f"checkpoint_w{worker_id}.jsonl", "a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, cls=_NumpyEncoder, ensure_ascii=False) + "\n")


def _save_dataset(all_data, output_dir: pathlib.Path, worker_id: int):
    dataset_path = output_dir / f"active_dataset_w{worker_id}.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for dp in all_data:
            f.write(json.dumps(dp, cls=_NumpyEncoder, ensure_ascii=False) + "\n")
    logging.info(f"[Dataset] saved {len(all_data)} data points to {dataset_path}")


def _save_summary(all_data, output_dir: pathlib.Path, worker_id: int):
    label_counts = Counter(dp["required_precision"] for dp in all_data)
    total = len(all_data)
    summary = {
        "schema_version": "chunk_recovery_summary_v2_vlm",
        "created_at": time.time(),
        "total_data_points": total,
        "label_counts": dict(label_counts),
        "label_rates": {k: v / total if total else 0.0 for k, v in label_counts.items()},
    }
    summary_path = output_dir / f"active_summary_w{worker_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)
    logging.info(f"[Summary] saved to {summary_path}")


def _resolve_combined_paths(args: Args, combined_dir: pathlib.Path):
    if args.trajectory_list.strip():
        paths = [pathlib.Path(p.strip()) for p in args.trajectory_list.split(",") if p.strip()]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(combined_dir.glob("*_combined.json"))

    if args.task_id >= 0:
        prefix = f"task{args.task_id:02d}_"
        paths = [p for p in paths if p.name.startswith(prefix)]

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    return [p for i, p in enumerate(paths) if i % args.num_shards == args.shard_id]


def run_chunk_recovery(args: Args) -> None:
    np.random.seed(args.seed)

    bank_base_dir = pathlib.Path(args.bank_base_dir)
    out_base_dir = pathlib.Path(args.out_base_dir)
    combined_dir = bank_base_dir / args.combined_subdir
    output_dir = out_base_dir / args.output_subdir
    video_dir = out_base_dir / args.video_subdir
    obs_dir = out_base_dir / args.obs_subdir
    vlm_hidden_dir = out_base_dir / args.vlm_hidden_subdir

    for d in (output_dir, video_dir, obs_dir, vlm_hidden_dir):
        d.mkdir(parents=True, exist_ok=True)

    logging.info("Mode 6: Chunk-level recovery probing + VLM hidden at chunk start")
    logging.info(f"Combined dir: {combined_dir}")
    logging.info(f"Output dir:   {output_dir}")
    logging.info(f"VLM hidden:   {args.save_vlm_hidden} -> {vlm_hidden_dir}")
    logging.info(f"FP16 server:  {args.host}:{args.port_fp16}")

    combined_paths = _resolve_combined_paths(args, combined_dir)
    if not combined_paths:
        logging.error("No combined trajectory files found.")
        return

    completed, all_data, checkpoint_entries = _load_checkpoint(output_dir, args.worker_id)
    new_entries = []

    fp16_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_fp16)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()

    for traj_path in tqdm.tqdm(combined_paths):
        traj_name = traj_path.stem.replace("_combined", "")
        out_result_path = output_dir / f"{traj_name}_chunk_recovery.json"

        if out_result_path.exists() and args.skip_existing:
            continue

        with open(traj_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)

        if args.skip_failed_source_trajectories and not bool(traj_data.get("success", False)):
            continue

        task_id = int(traj_data["task_id"])
        episode_idx = int(traj_data.get("episode_idx", 0))
        task_description = traj_data["task_description"]
        chunks = traj_data["chunks"]

        if not chunks:
            continue

        task = task_suite.get_task(task_id)

        if "initial_state" in traj_data:
            initial_state = np.asarray(traj_data["initial_state"])
        else:
            init_arr = task_suite.get_task_init_states(task_id)[episode_idx]
            initial_state = np.asarray(init_arr.tolist() if hasattr(init_arr, "tolist") else init_arr)

        traj_results = [
            {
                "chunk_idx": int(e["chunk_idx"]),
                "required_precision": e["data_point"]["required_precision"],
                "vlm_feature_paths": e["data_point"].get("vlm_feature_paths", {}),
            }
            for e in checkpoint_entries
            if e["traj_name"] == traj_name
        ]

        env = None
        try:
            env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            for chunk_idx in tqdm.tqdm(range(len(chunks)), desc=traj_name, leave=False):
                key = (traj_name, int(chunk_idx))
                if key in completed:
                    continue

                data_point = _run_chunk_recovery_probe(
                    env,
                    initial_state=initial_state,
                    chunks=chunks,
                    chunk_idx=chunk_idx,
                    task_description=task_description,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    traj_name=traj_name,
                    fp16_client=fp16_client,
                    args=args,
                    obs_dir=obs_dir,
                    vlm_hidden_dir=vlm_hidden_dir,
                    video_dir=video_dir if args.save_video else None,
                )

                all_data.append(data_point)
                traj_results.append(
                    {
                        "chunk_idx": int(chunk_idx),
                        "required_precision": data_point["required_precision"],
                        "vlm_feature_paths": data_point.get("vlm_feature_paths", {}),
                    }
                )

                new_entries.append({"traj_name": traj_name, "chunk_idx": int(chunk_idx), "data_point": data_point})
                completed.add(key)

                if len(new_entries) >= args.checkpoint_every:
                    _save_checkpoint(output_dir, new_entries, args.worker_id)
                    new_entries = []

        finally:
            if env is not None:
                env.close()

        with open(out_result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "chunk_recovery_traj_v2_vlm",
                    "traj_name": traj_name,
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "task_description": task_description,
                    "num_chunks": len(chunks),
                    "results": traj_results,
                },
                f,
                indent=2,
                cls=_NumpyEncoder,
                ensure_ascii=False,
            )

    if new_entries:
        _save_checkpoint(output_dir, new_entries, args.worker_id)

    _save_dataset(all_data, output_dir, args.worker_id)
    _save_summary(all_data, output_dir, args.worker_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(run_chunk_recovery)

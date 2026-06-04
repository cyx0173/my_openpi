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
import hashlib
import logging
import math
import pathlib
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
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

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

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 600,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"

    # Kept for compatibility with your old scripts. This is the policy server port.
    port_w4a4: int = 8000

    # Name written into saved bank/trace files. You can set this to w4a8 or w4a16
    # while still using --port-w4a4 as the server port.
    precision: str = "w4a4"

    task_suite_name: str = "libero_10"
    task_ids: str = "0,1,2,3,4,5,6,7,8,9"
    episode_start: int = 0
    episode_end: int = 10
    seed: int = 7

    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10

    use_debug_noise: bool = True
    send_debug_noise: bool = True
    debug_noise_base_seed: int = 0
    debug_noise_horizon: int = 10
    debug_noise_dim: int = 32

    base_dir: str = "/home/chengyuxuan/openpi/experiments/baseline/recovery_base"
    skip_existing: bool = False

    # Save raw MuJoCo env checkpoint at every chunk start.
    # This is additive: it does not change the old bank / trace / image / privileged-state fields.
    save_env_checkpoints: bool = True
    checkpoint_subdir: str = "checkpoints"

    # What to save for diagnosis.
    save_images: bool = True
    save_privileged_state: bool = True
    save_contacts: bool = True

    # Thresholds for transition_features / derived_events.
    gripper_delta_threshold: float = 0.005
    eef_motion_threshold: float = 0.005
    object_motion_threshold: float = 0.010
    object_lift_threshold: float = 0.020
    object_near_eef_threshold: float = 0.080
    object_following_cos_threshold: float = 0.50
    joint_delta_threshold: float = 0.010


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


def _decode_name(name) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="ignore")
    return str(name)


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


def _get_policy_images_and_proprio(obs, task_description: str, resize_size: int):
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_raw, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_raw, resize_size, resize_size)
    )

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    eef_axis_angle = np.asarray(_quat2axisangle(eef_quat), dtype=np.float64)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)

    # Keep multiple scalar summaries because different gripper conventions use
    # different signs. gripper_width_abs is usually the most robust for analysis.
    gripper_width_sum = float(np.sum(gripper_qpos))
    gripper_width_abs = float(np.sum(np.abs(gripper_qpos)))
    gripper_width_norm = float(np.linalg.norm(gripper_qpos))

    state = np.concatenate((eef_pos, eef_axis_angle, gripper_qpos))

    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }

    proprio = {
        "eef_pos": eef_pos.tolist(),
        "eef_quat": eef_quat.tolist(),
        "eef_axis_angle": eef_axis_angle.tolist(),
        "gripper_qpos": gripper_qpos.tolist(),
        "gripper_width": gripper_width_abs,
        "gripper_width_sum": gripper_width_sum,
        "gripper_width_abs": gripper_width_abs,
        "gripper_width_norm": gripper_width_norm,
    }

    return element, img, wrist_img, proprio


def _save_image(arr: np.ndarray, path_base: pathlib.Path) -> str:
    path_base.parent.mkdir(parents=True, exist_ok=True)

    if Image is not None:
        path = path_base.with_suffix(".png")
        Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)
        return str(path)

    # Fallback if pillow is unavailable.
    path = path_base.with_suffix(".npy")
    np.save(path, np.asarray(arr, dtype=np.uint8))
    return str(path)


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



def _get_sim(env):
    """Return the underlying MuJoCo sim object from a LIBERO / robosuite env or wrapper."""
    if hasattr(env, "sim"):
        return env.sim
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        return env.env.sim
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "sim"):
        return env.unwrapped.sim
    raise AttributeError("Cannot find env.sim")


def _iter_env_candidates(env):
    """Iterate wrapper chain: env, env.env, env.env.env, ... without duplicates."""
    out = []
    cur = env
    while cur is not None:
        if not any(id(cur) == id(x) for x in out):
            out.append(cur)
        cur = getattr(cur, "env", None)
    return out


def _array_copy_or_none(x):
    if x is None:
        return None
    try:
        return np.asarray(x, dtype=np.float64).copy()
    except Exception:
        return None


def _env_state_components(env) -> dict[str, np.ndarray]:
    """
    MuJoCo-side raw state components saved at chunk start.

    qacc_warmstart is saved for completeness, but should not be the only hard
    pass/fail criterion for later restore validation.
    """
    sim = _get_sim(env)
    data = sim.data
    out: dict[str, np.ndarray] = {}

    for name in [
        "qpos",
        "qvel",
        "qacc_warmstart",
        "ctrl",
        "mocap_pos",
        "mocap_quat",
        "qfrc_applied",
        "xfrc_applied",
        "act",
        "eq_active",
        "userdata",
    ]:
        if hasattr(data, name):
            arr = _array_copy_or_none(getattr(data, name))
            if arr is not None:
                out[name] = arr.reshape(-1)

    if hasattr(data, "time"):
        out["time"] = np.asarray([float(data.time)], dtype=np.float64)

    return out


def _sha256_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def save_env_checkpoint(
    env,
    path: str | pathlib.Path,
    *,
    task_id: int,
    episode_idx: int,
    chunk_idx: int,
    step: int,
) -> dict[str, Any]:
    """
    Save a raw simulator checkpoint at the current chunk-start state.

    This function is intentionally additive: collection still uses the original
    step-returned obs and original action execution path.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sim = _get_sim(env)
    comps = _env_state_components(env)

    payload: dict[str, Any] = {
        "task_id": np.asarray([int(task_id)], dtype=np.int64),
        "episode_idx": np.asarray([int(episode_idx)], dtype=np.int64),
        "chunk_idx": np.asarray([int(chunk_idx)], dtype=np.int64),
        "step": np.asarray([int(step)], dtype=np.int64),
    }

    for name, arr in comps.items():
        payload[name] = np.asarray(arr, dtype=np.float64).copy()

    if hasattr(sim, "get_state"):
        try:
            st = sim.get_state()
            if hasattr(st, "flatten"):
                payload["sim_state_flat"] = np.asarray(st.flatten(), dtype=np.float64).copy()
        except Exception as e:
            payload["get_state_error"] = np.asarray([str(e)])

    np.savez_compressed(path, **payload)

    meta = {
        "path": str(path),
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "chunk_idx": int(chunk_idx),
        "step": int(step),
        "saved_fields": sorted(list(payload.keys())),
    }

    for name in ["qpos", "qvel", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "sim_state_flat"]:
        if name in payload:
            meta[f"{name}_sha256"] = _sha256_array(payload[name])

    return meta


def _sync_env_bookkeeping_after_restore(env, ckpt):
    """Sync robosuite/LIBERO Python-side time and step counters after set_state."""
    sim = _get_sim(env)
    sim_time = float(sim.data.time)

    step = None
    if "step" in ckpt.files:
        try:
            step = int(np.asarray(ckpt["step"]).reshape(-1)[0])
        except Exception:
            step = None

    for e in _iter_env_candidates(env):
        for attr in ["cur_time", "_cur_time"]:
            if hasattr(e, attr):
                try:
                    setattr(e, attr, sim_time)
                except Exception:
                    pass

        if step is not None:
            for attr in [
                "timestep",
                "_timestep",
                "cur_step",
                "_cur_step",
                "episode_step",
                "_episode_step",
                "_elapsed_steps",
            ]:
                if hasattr(e, attr):
                    try:
                        setattr(e, attr, step)
                    except Exception:
                        pass

        if hasattr(e, "done"):
            try:
                setattr(e, "done", False)
            except Exception:
                pass


def _refresh_env_after_restore(env):
    """Refresh controller / observable caches after sim.set_state."""
    sim = _get_sim(env)

    for e in _iter_env_candidates(env):
        robots = getattr(e, "robots", None)
        if robots is not None:
            for robot in robots:
                if hasattr(robot, "update_sim"):
                    try:
                        robot.update_sim(sim)
                    except Exception:
                        pass

                if hasattr(robot, "update"):
                    try:
                        try:
                            robot.update(force=True)
                        except TypeError:
                            robot.update()
                    except Exception:
                        pass

                controller = getattr(robot, "controller", None)
                if controller is not None and hasattr(controller, "update"):
                    try:
                        controller.update()
                    except Exception:
                        pass

        for method_name in ["_update_observables", "_update_observable_sensor"]:
            if hasattr(e, method_name):
                try:
                    method = getattr(e, method_name)
                    try:
                        method(force=True)
                    except TypeError:
                        method()
                except Exception:
                    pass


def load_env_checkpoint(env, path: str | pathlib.Path) -> dict[str, Any]:
    """
    Restore checkpoint saved by save_env_checkpoint.

    This is not used during collection, but is kept here so future rescue code can
    import the same helper and load checkpoints consistently.
    """
    path = pathlib.Path(path)
    sim = _get_sim(env)
    data = sim.data
    ckpt = np.load(path, allow_pickle=False)

    loaded_with = "direct_arrays"
    if "sim_state_flat" in ckpt.files and hasattr(sim, "set_state_from_flattened"):
        sim.set_state_from_flattened(ckpt["sim_state_flat"])
        loaded_with = "set_state_from_flattened"
    else:
        if "qpos" in ckpt.files:
            data.qpos[:] = ckpt["qpos"]
        if "qvel" in ckpt.files:
            data.qvel[:] = ckpt["qvel"]

    for name in [
        "ctrl",
        "mocap_pos",
        "mocap_quat",
        "qfrc_applied",
        "xfrc_applied",
        "act",
        "eq_active",
        "userdata",
    ]:
        if name in ckpt.files and hasattr(data, name):
            try:
                target = getattr(data, name)
                arr = ckpt[name]
                if target.shape == arr.shape:
                    target[:] = arr
            except Exception:
                pass

    if "time" in ckpt.files and hasattr(data, "time"):
        data.time = float(np.asarray(ckpt["time"]).reshape(-1)[0])

    sim.forward()

    if "qacc_warmstart" in ckpt.files and hasattr(data, "qacc_warmstart"):
        try:
            arr = ckpt["qacc_warmstart"]
            if data.qacc_warmstart.shape == arr.shape:
                data.qacc_warmstart[:] = arr
        except Exception:
            pass

    _sync_env_bookkeeping_after_restore(env, ckpt)
    _refresh_env_after_restore(env)

    return {
        "path": str(path),
        "loaded_with": loaded_with,
        "step": int(np.asarray(ckpt["step"]).reshape(-1)[0]) if "step" in ckpt.files else None,
        "chunk_idx": int(np.asarray(ckpt["chunk_idx"]).reshape(-1)[0]) if "chunk_idx" in ckpt.files else None,
    }


def _resolve_task_ids(args: Args, num_tasks_in_suite: int):
    task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    return [x for x in task_ids if 0 <= x < num_tasks_in_suite]


def _resolve_episode_indices(args: Args, num_initial_states: int):
    start = max(0, int(args.episode_start))
    end = min(int(args.episode_end), num_initial_states)
    return list(range(start, end))


def _make_debug_noise(args: Args, task_id: int, episode_idx: int, chunk_idx: int):
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


def _add_debug_noise(element: dict, args: Args, task_id: int, episode_idx: int, chunk_idx: int):
    if not args.use_debug_noise:
        return element, {"enabled": False, "sent": False, "seed": None, "shape": None}

    noise_seed, debug_noise = _make_debug_noise(args, task_id, episode_idx, chunk_idx)

    out = dict(element)
    if args.send_debug_noise:
        out["debug_noise"] = debug_noise.copy()

    return out, {
        "enabled": True,
        "sent": bool(args.send_debug_noise),
        "seed": int(noise_seed),
        "shape": list(debug_noise.shape),
    }


def _to_action_list(actions, replan_steps: int):
    return np.asarray(actions[:replan_steps]).tolist()


def _is_robot_name(name: str) -> bool:
    low = name.lower()
    return (
        low.startswith("robot0")
        or "robot0" in low
        or low.startswith("panda")
        or "gripper" in low
        or "eef" in low
    )


def _body_names_for_trace(env) -> list[str]:
    names = [_decode_name(x) for x in getattr(env.sim.model, "body_names", [])]
    out = []
    for name in names:
        low = name.lower()
        if low in {"world", "worldbody"}:
            continue
        if _is_robot_name(name):
            continue
        out.append(name)
    return sorted(set(out))


def _joint_names_for_trace(env) -> list[str]:
    names = [_decode_name(x) for x in getattr(env.sim.model, "joint_names", [])]
    out = []
    for name in names:
        if _is_robot_name(name):
            continue
        out.append(name)
    return sorted(set(out))


def _get_joint_qpos(env, joint_name: str):
    try:
        qpos = env.sim.data.get_joint_qpos(joint_name)
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return float(arr[0])
        return arr.tolist()
    except Exception:
        return None


def _collect_contacts(env) -> list[list[str]]:
    contacts: list[list[str]] = []
    try:
        data = env.sim.data
        model = env.sim.model
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1 = model.geom_id2name(int(c.geom1))
            g2 = model.geom_id2name(int(c.geom2))
            if g1 is None:
                g1 = str(int(c.geom1))
            if g2 is None:
                g2 = str(int(c.geom2))
            contacts.append([str(g1), str(g2)])
    except Exception:
        pass
    return contacts


def _state_snapshot(env, obs, *, body_names: list[str], joint_names: list[str], save_contacts: bool) -> dict[str, Any]:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)

    gripper_width_sum = float(np.sum(gripper_qpos))
    gripper_width_abs = float(np.sum(np.abs(gripper_qpos)))
    gripper_width_norm = float(np.linalg.norm(gripper_qpos))

    objects: dict[str, Any] = {}
    for name in body_names:
        try:
            body_id = env.sim.model.body_name2id(name)
            pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64)
            quat = np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float64)
            objects[name] = {
                "pos": pos.tolist(),
                "quat": quat.tolist(),
                "z": float(pos[2]),
                "dist_to_eef": float(np.linalg.norm(pos - eef_pos)),
            }
        except Exception:
            continue

    joints: dict[str, Any] = {}
    for name in joint_names:
        qpos = _get_joint_qpos(env, name)
        if qpos is not None:
            joints[name] = qpos

    snap = {
        "eef_pos": eef_pos.tolist(),
        "eef_quat": eef_quat.tolist(),
        "eef_axis_angle": _quat2axisangle(eef_quat).tolist(),
        "gripper_qpos": gripper_qpos.tolist(),
        "gripper_width": gripper_width_abs,
        "gripper_width_sum": gripper_width_sum,
        "gripper_width_abs": gripper_width_abs,
        "gripper_width_norm": gripper_width_norm,
        "objects": objects,
        "joints": joints,
    }

    if save_contacts:
        snap["contacts"] = _collect_contacts(env)

    return snap


def _joint_value_as_array(x) -> np.ndarray:
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64).reshape(-1)
    return np.asarray([x], dtype=np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _transition_features(start: dict[str, Any], end: dict[str, Any], args: Args):
    eef_start = np.asarray(start["eef_pos"], dtype=np.float64)
    eef_end = np.asarray(end["eef_pos"], dtype=np.float64)
    eef_delta = eef_end - eef_start
    eef_motion = float(np.linalg.norm(eef_delta))

    gripper_delta = float(end["gripper_width"] - start["gripper_width"])

    object_motions: dict[str, float] = {}
    object_z_deltas: dict[str, float] = {}
    object_follow_cos: dict[str, float] = {}

    moved_objects: list[str] = []
    lifted_objects: list[str] = []
    following_objects: list[str] = []

    closest_object_start = None
    closest_object_end = None
    closest_dist_start = None
    closest_dist_end = None

    for name, sobj in start.get("objects", {}).items():
        eobj = end.get("objects", {}).get(name)
        if eobj is None:
            continue

        spos = np.asarray(sobj["pos"], dtype=np.float64)
        epos = np.asarray(eobj["pos"], dtype=np.float64)
        odelta = epos - spos

        motion = float(np.linalg.norm(odelta))
        z_delta = float(eobj["z"] - sobj["z"])
        cos = _cosine(eef_delta, odelta)

        object_motions[name] = motion
        object_z_deltas[name] = z_delta
        object_follow_cos[name] = cos

        if motion >= args.object_motion_threshold:
            moved_objects.append(name)
        if z_delta >= args.object_lift_threshold:
            lifted_objects.append(name)

        near_eef = min(float(sobj["dist_to_eef"]), float(eobj["dist_to_eef"])) <= args.object_near_eef_threshold
        if (
            near_eef
            and motion >= args.object_motion_threshold
            and eef_motion >= args.eef_motion_threshold
            and cos >= args.object_following_cos_threshold
        ):
            following_objects.append(name)

        ds = float(sobj["dist_to_eef"])
        de = float(eobj["dist_to_eef"])
        if closest_dist_start is None or ds < closest_dist_start:
            closest_dist_start = ds
            closest_object_start = name
        if closest_dist_end is None or de < closest_dist_end:
            closest_dist_end = de
            closest_object_end = name

    joint_deltas: dict[str, float] = {}
    for name, sj in start.get("joints", {}).items():
        ej = end.get("joints", {}).get(name)
        if ej is None:
            continue
        sarr = _joint_value_as_array(sj)
        earr = _joint_value_as_array(ej)
        if sarr.shape != earr.shape:
            continue
        joint_deltas[name] = float(np.linalg.norm(earr - sarr))

    max_object_motion = max(object_motions.values(), default=0.0)
    max_object_z_delta = max(object_z_deltas.values(), default=0.0)
    max_joint_delta = max(joint_deltas.values(), default=0.0)

    transition = {
        "gripper_delta": gripper_delta,
        "eef_motion": eef_motion,
        "max_object_motion": float(max_object_motion),
        "max_object_z_delta": float(max_object_z_delta),
        "max_joint_delta": float(max_joint_delta),
        "closest_object_start": closest_object_start,
        "closest_object_dist_start": closest_dist_start,
        "closest_object_end": closest_object_end,
        "closest_object_dist_end": closest_dist_end,
        "moved_objects": sorted(moved_objects),
        "lifted_objects": sorted(lifted_objects),
        "following_objects": sorted(following_objects),
        "object_motions": object_motions,
        "object_z_deltas": object_z_deltas,
        "object_follow_cos": object_follow_cos,
        "joint_deltas": joint_deltas,
    }

    derived = {
        "gripper_closing": bool(gripper_delta <= -args.gripper_delta_threshold),
        "gripper_opening": bool(gripper_delta >= args.gripper_delta_threshold),
        "large_eef_motion": bool(eef_motion >= args.eef_motion_threshold),
        "large_object_motion": bool(max_object_motion >= args.object_motion_threshold),
        "object_lifted": bool(len(lifted_objects) > 0),
        "object_near_gripper": bool(
            (closest_dist_start is not None and closest_dist_start <= args.object_near_eef_threshold)
            or (closest_dist_end is not None and closest_dist_end <= args.object_near_eef_threshold)
        ),
        "object_following_eef": bool(len(following_objects) > 0),
        "large_joint_motion": bool(max_joint_delta >= args.joint_delta_threshold),
    }

    return transition, derived


def _run_policy_episode(
    env,
    task_description: str,
    initial_state,
    policy_client,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    episode_image_dir: pathlib.Path,
    episode_checkpoint_dir: pathlib.Path | None = None,
):
    env.reset()
    obs = env.set_init_state(initial_state)

    body_names = _body_names_for_trace(env)
    joint_names = _joint_names_for_trace(env)

    done = False
    t = 0
    chunk_idx = 0
    bank_chunks = []
    trace_chunks = []

    max_steps = _MAX_STEPS.get(args.task_suite_name, 600)

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            if done:
                break
            continue

        step_start = int(t)

        checkpoint_meta = None
        if args.save_env_checkpoints and episode_checkpoint_dir is not None:
            checkpoint_meta = save_env_checkpoint(
                env,
                episode_checkpoint_dir / f"chunk{chunk_idx:04d}_start.npz",
                task_id=task_id,
                episode_idx=episode_idx,
                chunk_idx=chunk_idx,
                step=step_start,
            )

        element, agent_img, wrist_img, proprio = _get_policy_images_and_proprio(
            obs, task_description, args.resize_size
        )

        agent_image_path = None
        wrist_image_path = None
        if args.save_images:
            agent_image_path = _save_image(
                agent_img,
                episode_image_dir / f"agent_chunk{chunk_idx:04d}",
            )
            wrist_image_path = _save_image(
                wrist_img,
                episode_image_dir / f"wrist_chunk{chunk_idx:04d}",
            )

        privileged_start = None
        if args.save_privileged_state:
            privileged_start = _state_snapshot(
                env,
                obs,
                body_names=body_names,
                joint_names=joint_names,
                save_contacts=args.save_contacts,
            )

        element, noise_meta = _add_debug_noise(
            element,
            args=args,
            task_id=task_id,
            episode_idx=episode_idx,
            chunk_idx=chunk_idx,
        )

        result = policy_client.infer(element)
        planned_actions = _to_action_list(result["actions"], args.replan_steps)

        executed_actions = []
        for action in planned_actions:
            if t >= max_steps + args.num_steps_wait:
                break
            obs, _, done, _ = env.step(action)
            t += 1
            executed_actions.append(action)
            if done:
                break

        step_end = int(t)

        privileged_end = None
        transition = {}
        derived = {}
        if args.save_privileged_state:
            privileged_end = _state_snapshot(
                env,
                obs,
                body_names=body_names,
                joint_names=joint_names,
                save_contacts=args.save_contacts,
            )
            transition, derived = _transition_features(privileged_start, privileged_end, args)

        bank_chunks.append(
            {
                "chunk_idx": int(chunk_idx),
                "step_start": step_start,
                "step_end": step_end,
                "noise_seed": noise_meta["seed"],
                "env_checkpoint": checkpoint_meta,
                "num_actions_planned": int(len(planned_actions)),
                "num_actions_executed": int(len(executed_actions)),
                "executed_actions": executed_actions,
            }
        )

        trace_chunk = {
            "chunk_idx": int(chunk_idx),
            "step_start": step_start,
            "step_end": step_end,
            "noise_seed": noise_meta["seed"],
            "env_checkpoint": checkpoint_meta,
            "num_actions_planned": int(len(planned_actions)),
            "num_actions_executed": int(len(executed_actions)),
            "policy_input_start": {
                "agent_image_path": agent_image_path,
                "wrist_image_path": wrist_image_path,
                "proprio": proprio,
                "task_description": str(task_description),
            },
            "transition_features": transition,
            "derived_events": derived,
        }

        if args.save_privileged_state:
            trace_chunk["privileged_start"] = privileged_start
            trace_chunk["privileged_end"] = privileged_end

        trace_chunks.append(trace_chunk)

        logging.info(
            f"    task={task_id} ep={episode_idx} "
            f"chunk={chunk_idx:04d} step={step_start:04d}->{step_end:04d} "
            f"noise_seed={noise_meta['seed']} done={done}"
        )

        chunk_idx += 1
        if done:
            break

    return {
        "success": bool(done),
        "total_steps": int(t),
        "num_chunks": int(len(bank_chunks)),
        "bank_chunks": bank_chunks,
        "trace_chunks": trace_chunks,
        "body_names_collected": body_names,
        "joint_names_collected": joint_names,
    }


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    precision = str(args.precision).lower().strip()
    base_dir = pathlib.Path(args.base_dir)

    bank_dir = base_dir / f"{precision}_action_bank"
    trace_dir = base_dir / f"{precision}_trace"
    image_root = base_dir / "images"
    checkpoint_root = base_dir / str(args.checkpoint_subdir)

    bank_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        image_root.mkdir(parents=True, exist_ok=True)
    if args.save_env_checkpoints:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

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
                "max_steps": int(_MAX_STEPS.get(args.task_suite_name, 600)),
                "precision": precision,
                "server_port": int(args.port_w4a4),
                "save_images": bool(args.save_images),
                "save_privileged_state": bool(args.save_privileged_state),
                "save_contacts": bool(args.save_contacts),
                "save_env_checkpoints": bool(args.save_env_checkpoints),
                "checkpoint_subdir": str(args.checkpoint_subdir),
                "use_debug_noise": bool(args.use_debug_noise),
                "send_debug_noise": bool(args.send_debug_noise),
                "debug_noise_base_seed": int(args.debug_noise_base_seed),
                "debug_noise_shape": [int(args.debug_noise_horizon), int(args.debug_noise_dim)],
                "debug_noise_formula": "base_seed + task_id * 100000 + episode_idx * 1000 + chunk_idx",
                "thresholds": {
                    "gripper_delta_threshold": float(args.gripper_delta_threshold),
                    "eef_motion_threshold": float(args.eef_motion_threshold),
                    "object_motion_threshold": float(args.object_motion_threshold),
                    "object_lift_threshold": float(args.object_lift_threshold),
                    "object_near_eef_threshold": float(args.object_near_eef_threshold),
                    "object_following_cos_threshold": float(args.object_following_cos_threshold),
                    "joint_delta_threshold": float(args.joint_delta_threshold),
                },
                "output_dirs": {
                    "bank_dir": str(bank_dir),
                    "trace_dir": str(trace_dir),
                    "image_root": str(image_root),
                    "checkpoint_root": str(checkpoint_root),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logging.info(f"Base dir: {base_dir}")
    logging.info(f"Bank dir: {bank_dir}")
    logging.info(f"Trace dir: {trace_dir}")
    logging.info(f"{precision} server: ws://{args.host}:{args.port_w4a4}")

    policy_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_w4a4)

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

            for episode_idx in tqdm.tqdm(episode_indices, desc=f"task{task_id:02d}", leave=False):
                stem = f"task{task_id:02d}_ep{episode_idx:03d}_{precision}"
                bank_path = bank_dir / f"{stem}_bank.json"
                trace_path = trace_dir / f"{stem}_trace.json"
                episode_image_dir = image_root / stem
                episode_checkpoint_dir = checkpoint_root / stem if args.save_env_checkpoints else None

                if bank_path.exists() and trace_path.exists() and args.skip_existing:
                    logging.info(f"[Skip] exists: {bank_path} and {trace_path}")
                    continue

                initial_state = np.array(
                    initial_states[episode_idx].tolist()
                    if hasattr(initial_states[episode_idx], "tolist")
                    else initial_states[episode_idx]
                )

                result = _run_policy_episode(
                    env,
                    task_description,
                    initial_state,
                    policy_client,
                    args,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    episode_image_dir=episode_image_dir,
                    episode_checkpoint_dir=episode_checkpoint_dir,
                )

                common_meta = {
                    "task_id": int(task_id),
                    "task_description": str(task_description),
                    "episode_idx": int(episode_idx),
                    "seed": int(args.seed),
                    "initial_state_idx": int(episode_idx),
                    "initial_state": initial_state.tolist(),
                    "initial_state_shape": list(initial_state.shape),
                    "initial_state_dtype": str(initial_state.dtype),
                    "precision": precision,
                    "success": bool(result["success"]),
                    "total_steps": int(result["total_steps"]),
                    "num_chunks": int(result["num_chunks"]),
                    "replan_steps": int(args.replan_steps),
                    "num_steps_wait": int(args.num_steps_wait),
                }

                bank_record = {
                    **common_meta,
                    "trace_json": str(trace_path),
                    "chunks": result["bank_chunks"],
                }

                trace_record = {
                    **common_meta,
                    "action_json": str(bank_path),
                    "body_names_collected": result["body_names_collected"],
                    "joint_names_collected": result["joint_names_collected"],
                    "notes": {
                        "policy_input_start": "selector-usable input only; do not include privileged object states in selector training",
                        "privileged_start_end": "offline diagnosis only; contains simulator state",
                    },
                    "chunks": result["trace_chunks"],
                }

                with open(bank_path, "w", encoding="utf-8") as f:
                    json.dump(bank_record, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)

                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(trace_record, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)

                logging.info(f"[Saved] {bank_path} success={result['success']} chunks={result['num_chunks']}")
                logging.info(f"[Saved] {trace_path}")

        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(eval_libero)

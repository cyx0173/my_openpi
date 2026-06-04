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
import re
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

# Fixed debug-noise settings. These are intentionally not CLI args.
USE_DEBUG_NOISE = True
SEND_DEBUG_NOISE = True
DEBUG_NOISE_BASE_SEED = 0
DEBUG_NOISE_HORIZON = 10
DEBUG_NOISE_DIM = 32

BANK_RE = re.compile(r"task(?P<task_id>\d+)_ep(?P<ep>\d+)_.*?_bank\.json$")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"

    # Historical name. In this script it means the main/rescue server port.
    # For the new W4A4-main rescue experiment, this should be the W4A4 server port.
    port_a16: int = 8002

    # Inject/rescue server port. For example: W4A16 port=8000 or W4A8 port=8001.
    port_inject: int = 8000

    task_suite_name: str = "libero_10"
    task_ids: str = "all"

    seed: int = 7
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_steps: int = 520
    success_thresholds: str = "320,420,520"

    # New direct entry.
    # Pass one W4A4 action bank:
    #   --args.action-chunk-json /.../task08_ep002_w4a4_bank.json
    # It also accepts a flat list json whose items contain action_chunk_json.
    action_chunk_json: str = ""

    # Backward-compatible old entries. These are ignored when action_chunk_json is set.
    candidate_dir: str = (
        "/home/chengyuxuan/openpi/experiments/baseline/data/"
        "classified_from_action_index/recovery_candidate"
    )
    action_index_dir: str = (
        "/home/chengyuxuan/openpi/experiments/baseline/data/action_index_by_task"
    )

    # New recovery default:
    #   W4A4 failed mainline -> one W4A16/W4A8 injected chunk -> W4A4 suffix.
    main_policy: str = "w4a4"
    inject_policy: str = "w4a16"
    inject_chunks: str = "all"

    base_dir: str = "/home/chengyuxuan/openpi/experiments/mode3/recovery"
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


def _parse_task_id_filter(task_ids: str, num_tasks_in_suite: int) -> set[int]:
    raw = str(task_ids).strip().lower()

    if raw in {"", "all", "*"}:
        return set(range(num_tasks_in_suite))

    out: set[int] = set()
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        task_id = int(x)
        if 0 <= task_id < num_tasks_in_suite:
            out.add(task_id)

    return out


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_bank_json(path_like: str | pathlib.Path) -> tuple[dict[str, Any], pathlib.Path]:
    path = pathlib.Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"action bank json does not exist: {path}")

    obj = _read_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict action bank json: {path}")
    if not isinstance(obj.get("chunks"), list):
        raise ValueError(f"Action bank missing list field 'chunks': {path}")

    return obj, path


def _parse_task_ep_from_bank_path(path: pathlib.Path) -> tuple[int, int]:
    m = BANK_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse task_id/episode from action bank filename: {path.name}")
    return int(m.group("task_id")), int(m.group("ep"))


def _candidate_from_action_bank(path_like: str | pathlib.Path) -> dict[str, Any]:
    bank, path = _load_bank_json(path_like)
    fallback_task_id, fallback_ep = _parse_task_ep_from_bank_path(path)

    task_id = int(bank.get("task_id", fallback_task_id))
    episode_idx = int(
        bank.get(
            "episode_idx",
            bank.get("task_ep", bank.get("eps_id", bank.get("initial_state_idx", fallback_ep))),
        )
    )

    return {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "action_json": str(path),
        "candidate_file": str(path),
    }


def _normalise_flat_candidate_item(raw: dict[str, Any], candidate_file: pathlib.Path) -> dict[str, Any]:
    action_json = raw.get("action_chunk_json") or raw.get("action_json") or raw.get("w4a4_action_json")
    if action_json is None:
        raise KeyError(f"Missing action_chunk_json/action_json/w4a4_action_json in {candidate_file}: {raw}")

    item = _candidate_from_action_bank(str(action_json))

    # Prefer explicit metadata from the flat list if present.
    if "task_id" in raw:
        item["task_id"] = int(raw["task_id"])
    if "eps_id" in raw:
        item["episode_idx"] = int(raw["eps_id"])
    elif "task_ep" in raw:
        item["episode_idx"] = int(raw["task_ep"])
    elif "ep_id" in raw:
        item["episode_idx"] = int(raw["ep_id"])

    item["candidate_file"] = str(candidate_file)
    return item


def _candidate_items_from_direct_action_chunk(args: Args) -> dict[int, dict[int, dict[str, Any]]]:
    candidate_path = pathlib.Path(args.action_chunk_json).expanduser().resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"action_chunk_json does not exist: {candidate_path}")

    obj = _read_json(candidate_path)
    out: dict[int, dict[int, dict[str, Any]]] = {}

    # Case 1: action_chunk_json is one action bank json.
    if isinstance(obj, dict) and isinstance(obj.get("chunks"), list):
        item = _candidate_from_action_bank(candidate_path)
        out.setdefault(item["task_id"], {})[item["episode_idx"]] = item
        return out

    # Case 2: action_chunk_json is the merged failed-episode list.
    if isinstance(obj, list):
        for raw in obj:
            if not isinstance(raw, dict):
                raise ValueError(f"Expected dict item in list json: {candidate_path}")
            item = _normalise_flat_candidate_item(raw, candidate_path)
            out.setdefault(item["task_id"], {})[item["episode_idx"]] = item
        return out

    # Case 3: action_chunk_json is task index json with episodes.
    if isinstance(obj, dict) and isinstance(obj.get("episodes"), list):
        parent_task_id = obj.get("task_id")
        for ep in obj["episodes"]:
            raw = dict(ep)
            if parent_task_id is not None:
                raw.setdefault("task_id", parent_task_id)
            item = _normalise_flat_candidate_item(raw, candidate_path)
            out.setdefault(item["task_id"], {})[item["episode_idx"]] = item
        return out

    raise ValueError(
        "Unsupported --args.action-chunk-json input. Expected one action bank dict, "
        f"a flat list json, or a dict with episodes list: {candidate_path}"
    )


def _load_candidate_items(args: Args) -> dict[int, dict[int, dict[str, Any]]]:
    """Load candidate episode items.

    Preferred mode:
      --args.action-chunk-json /path/to/taskXX_epYYY_w4a4_bank.json

    Backward-compatible mode:
      --args.candidate-dir /path/to/candidate_dir_or_json
    """
    if str(args.action_chunk_json).strip():
        out = _candidate_items_from_direct_action_chunk(args)
        if not out:
            logging.warning(f"No candidate episodes loaded from: {args.action_chunk_json}")
        return out

    candidate_path = pathlib.Path(args.candidate_dir).resolve()
    out: dict[int, dict[int, dict[str, Any]]] = {}

    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate path does not exist: {candidate_path}")

    if candidate_path.is_file():
        json_files = [candidate_path]
    elif candidate_path.is_dir():
        json_files = []
        for path in candidate_path.glob("*.json"):
            try:
                int(path.stem)
            except ValueError:
                continue
            json_files.append(path)
        json_files = sorted(json_files, key=lambda p: int(p.stem))
    else:
        raise ValueError(f"candidate path is neither file nor directory: {candidate_path}")

    for path in json_files:
        obj = _read_json(path)
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict) and isinstance(obj.get("episodes"), list):
            parent_task_id = obj.get("task_id")
            items = []
            for ep in obj["episodes"]:
                item = dict(ep)
                if parent_task_id is not None:
                    item.setdefault("task_id", parent_task_id)
                items.append(item)
        else:
            raise ValueError(f"Expected list JSON or dict with episodes list: {path}")

        for raw in items:
            item = _normalise_flat_candidate_item(raw, path)
            out.setdefault(item["task_id"], {})[item["episode_idx"]] = item

    if not out:
        logging.warning(f"No candidate episodes loaded from: {candidate_path}")
    return out


def _load_action_index_task(args: Args, task_id: int) -> dict[str, Any]:
    path = pathlib.Path(args.action_index_dir).resolve() / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"action index not found: {path}")

    obj = _read_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON in action index: {path}")

    got_task_id = int(obj.get("task_id", task_id))
    if got_task_id != int(task_id):
        raise ValueError(f"task_id mismatch in action index: expected {task_id}, got {got_task_id}: {path}")

    return obj


def _get_policy_item_from_action_index(
    args: Args,
    task_id: int,
    episode_idx: int,
    policy: str,
) -> dict[str, Any]:
    index = _load_action_index_task(args, task_id)

    for ep in index.get("episodes", []):
        if int(ep.get("episode", -1)) != int(episode_idx):
            continue

        item = ep.get(policy)
        if not isinstance(item, dict):
            raise KeyError(f"policy={policy} missing for task={task_id} ep={episode_idx}")

        action_json = item.get("json")
        if action_json is None:
            raise FileNotFoundError(f"policy={policy} json is None for task={task_id} ep={episode_idx}")

        return item

    raise KeyError(f"episode={episode_idx} not found for task={task_id}")


def _load_main_policy_bank(args: Args, task_id: int, episode_idx: int):
    item = _get_policy_item_from_action_index(
        args=args,
        task_id=task_id,
        episode_idx=episode_idx,
        policy=args.main_policy,
    )
    return _load_bank_json(str(item["json"]))


def _resolve_inject_chunks(raw: str, num_chunks: int) -> list[int]:
    raw_l = str(raw).strip().lower()

    if raw_l in {"", "all", "*"}:
        return list(range(num_chunks))

    chunks: list[int] = []
    for x in raw_l.split(","):
        x = x.strip()
        if not x:
            continue
        chunk_idx = int(x)
        if 0 <= chunk_idx < num_chunks:
            chunks.append(chunk_idx)

    return sorted(set(chunks))


def _get_obs_element(obs, task_description: str, resize_size: int):
    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_raw, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_raw, resize_size, resize_size))

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
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _to_action_list(actions, replan_steps: int):
    return np.asarray(actions[:replan_steps]).tolist()


def _fallback_noise_seed(task_id: int, episode_idx: int, chunk_idx: int):
    return int(DEBUG_NOISE_BASE_SEED) + int(task_id) * 100000 + int(episode_idx) * 1000 + int(chunk_idx)


def _noise_seed_from_main_bank(bank, task_id: int, episode_idx: int, chunk_idx: int):
    chunks = bank.get("chunks", [])
    if chunk_idx < len(chunks):
        seed = chunks[chunk_idx].get("noise_seed")
        if seed is not None:
            return int(seed), "main_bank"
    return _fallback_noise_seed(task_id, episode_idx, chunk_idx), "fallback_formula"


def _make_debug_noise(noise_seed: int):
    rng = np.random.default_rng(int(noise_seed))
    return rng.standard_normal(size=(int(DEBUG_NOISE_HORIZON), int(DEBUG_NOISE_DIM))).astype(np.float32)


def _add_debug_noise(element, noise_seed: int | None):
    if not USE_DEBUG_NOISE or noise_seed is None:
        return element

    out = dict(element)
    if SEND_DEBUG_NOISE:
        out["debug_noise"] = _make_debug_noise(noise_seed).copy()
    return out


def _success_at(success: bool, total_env_steps: int, thresholds: list[int]):
    return {str(h): bool(success and total_env_steps <= int(h)) for h in thresholds}


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


def _replay_main_prefix(env, main_bank, inject_chunk: int, args: Args, *, obs, done: bool, t: int):
    """Replay main_policy chunks 0..inject_chunk-1 to reach the injection state."""
    for chunk_idx in range(inject_chunk):
        if done:
            break

        chunk = main_bank["chunks"][chunk_idx]
        actions = _to_action_list(chunk["executed_actions"], args.replan_steps)

        for action in actions:
            if t >= args.max_steps + args.num_steps_wait:
                break
            obs, _, done, _ = env.step(action)
            t += 1
            if done:
                break

    return obs, done, t


def _run_injected_chunk(
    env,
    task_description: str,
    inject_client,
    main_bank,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    inject_chunk: int,
    obs,
    done: bool,
    t: int,
):
    """Run exactly one injected precision chunk at inject_chunk."""
    if done:
        return obs, done, t

    noise_seed, _ = _noise_seed_from_main_bank(main_bank, task_id, episode_idx, inject_chunk)
    element = _get_obs_element(obs, task_description, args.resize_size)
    element = _add_debug_noise(element, noise_seed)

    result = inject_client.infer(element)
    actions = _to_action_list(result["actions"], args.replan_steps)

    for action in actions:
        if t >= args.max_steps + args.num_steps_wait:
            break
        obs, _, done, _ = env.step(action)
        t += 1
        if done:
            break

    return obs, done, t


def _run_main_suffix(
    env,
    task_description: str,
    main_client,
    main_bank,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    next_chunk_idx: int,
    obs,
    done: bool,
    t: int,
):
    """Continue online main_policy from chunk inject_chunk+1."""
    chunk_idx = int(next_chunk_idx)
    action_plan = collections.deque()

    while t < args.max_steps + args.num_steps_wait:
        if done:
            break

        if not action_plan:
            noise_seed, _ = _noise_seed_from_main_bank(main_bank, task_id, episode_idx, chunk_idx)
            element = _get_obs_element(obs, task_description, args.resize_size)
            element = _add_debug_noise(element, noise_seed)

            result = main_client.infer(element)
            action_plan.extend(_to_action_list(result["actions"], args.replan_steps))
            chunk_idx += 1

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action)
        t += 1

    done_control_step = int(t - args.num_steps_wait) if done else None
    total_env_steps = int(done_control_step) if done_control_step is not None else int(args.max_steps)

    return {
        "success": bool(done),
        "total_env_steps": total_env_steps,
    }


def _run_recovery_trial(
    env,
    task_description: str,
    initial_state,
    main_client,
    inject_client,
    main_bank,
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    inject_chunk: int,
):
    obs, done, t = _reset_env(env, initial_state, args)

    obs, done, t = _replay_main_prefix(
        env,
        main_bank,
        inject_chunk,
        args,
        obs=obs,
        done=done,
        t=t,
    )

    obs, done, t = _run_injected_chunk(
        env,
        task_description,
        inject_client,
        main_bank,
        args,
        task_id=task_id,
        episode_idx=episode_idx,
        inject_chunk=inject_chunk,
        obs=obs,
        done=done,
        t=t,
    )

    return _run_main_suffix(
        env,
        task_description,
        main_client,
        main_bank,
        args,
        task_id=task_id,
        episode_idx=episode_idx,
        next_chunk_idx=inject_chunk + 1,
        obs=obs,
        done=done,
        t=t,
    )


def _load_main_bank_for_episode(args: Args, candidate_item: dict[str, Any], task_id: int, episode_idx: int):
    if str(args.action_chunk_json).strip():
        return _load_bank_json(candidate_item["action_json"])
    return _load_main_policy_bank(args, task_id, episode_idx)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    base_dir = pathlib.Path(args.base_dir)
    result_dir = base_dir / f"labels_{args.inject_policy}_into_{args.main_policy}"
    result_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_int_list(args.success_thresholds)

    with (base_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "w4a4_failed_mainline_one_chunk_rescue",
                "task_suite_name": args.task_suite_name,
                "task_ids": args.task_ids,
                "seed": int(args.seed),
                "replan_steps": int(args.replan_steps),
                "num_steps_wait": int(args.num_steps_wait),
                "max_steps": int(args.max_steps),
                "success_thresholds": thresholds,
                "action_chunk_json": args.action_chunk_json,
                "candidate_dir": args.candidate_dir,
                "action_index_dir": args.action_index_dir,
                "main_policy": args.main_policy,
                "inject_policy": args.inject_policy,
                "inject_chunks": args.inject_chunks,
                "debug_noise": {
                    "use": bool(USE_DEBUG_NOISE),
                    "send": bool(SEND_DEBUG_NOISE),
                    "base_seed": int(DEBUG_NOISE_BASE_SEED),
                    "shape": [int(DEBUG_NOISE_HORIZON), int(DEBUG_NOISE_DIM)],
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logging.info(f"Result dir: {result_dir}")
    if args.action_chunk_json.strip():
        logging.info(f"Direct action_chunk_json: {args.action_chunk_json}")
    else:
        logging.info(f"Candidate path: {args.candidate_dir}")
        logging.info(f"Action index dir: {args.action_index_dir}")
    logging.info(f"Main server ({args.main_policy}): ws://{args.host}:{args.port_a16}")
    logging.info(f"Inject server ({args.inject_policy}): ws://{args.host}:{args.port_inject}")

    main_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_a16)
    inject_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port_inject)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    task_id_filter = _parse_task_id_filter(args.task_ids, task_suite.n_tasks)
    candidate_items = _load_candidate_items(args)
    selected_task_ids = sorted(task_id for task_id in candidate_items.keys() if task_id in task_id_filter)

    logging.info(f"Selected task ids: {selected_task_ids}")

    summary_records = []

    for task_id in tqdm.tqdm(selected_task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        episode_indices = [
            ep for ep in sorted(candidate_items.get(task_id, {}).keys())
            if 0 <= ep < len(initial_states)
        ]

        logging.info(f"[Task {task_id:02d}] candidate episodes={episode_indices}")

        env = None
        try:
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            for episode_idx in tqdm.tqdm(episode_indices, desc=f"task{task_id:02d}", leave=False):
                candidate_item = candidate_items[task_id][episode_idx]

                try:
                    main_bank, main_bank_path = _load_main_bank_for_episode(
                        args,
                        candidate_item,
                        task_id,
                        episode_idx,
                    )
                except (FileNotFoundError, KeyError, ValueError) as e:
                    logging.warning(f"[Skip] missing main bank: task{task_id:02d}_ep{episode_idx:03d}: {e}")
                    continue

                if "chunks" not in main_bank or not isinstance(main_bank["chunks"], list):
                    logging.warning(f"[Skip] invalid main bank chunks: {main_bank_path}")
                    continue

                inject_chunks = _resolve_inject_chunks(args.inject_chunks, len(main_bank["chunks"]))
                if not inject_chunks:
                    logging.warning(f"[Skip] no valid inject chunks for task{task_id:02d}_ep{episode_idx:03d}")
                    continue

                initial_state = np.array(
                    initial_states[episode_idx].tolist()
                    if hasattr(initial_states[episode_idx], "tolist")
                    else initial_states[episode_idx]
                )

                episode_result = {
                    "task_id": int(task_id),
                    "task_ep": int(episode_idx),
                    "task_description": str(task_description),
                    "seed": int(args.seed),
                    "main_action_json": str(main_bank_path),
                    "chunks": [],
                }

                out_path = result_dir / (
                    f"task{task_id:02d}_ep{episode_idx:03d}_{args.inject_policy}_into_{args.main_policy}.json"
                )

                if out_path.exists() and args.skip_existing:
                    logging.info(f"[Skip] exists: {out_path}")
                    saved = _read_json(out_path)
                    summary_records.append(
                        {
                            "task_id": int(task_id),
                            "task_ep": int(episode_idx),
                            "num_chunks": len(saved.get("chunks", [])),
                            "path": str(out_path),
                            "skipped_existing": True,
                        }
                    )
                    continue

                logging.info(
                    f"[Episode] task={task_id:02d} ep={episode_idx:03d} "
                    f"main_action_json={main_bank_path} num_chunks={len(main_bank['chunks'])}"
                )

                for inject_chunk in inject_chunks:
                    logging.info(
                        f"[Recovery] task={task_id:02d} ep={episode_idx:03d} "
                        f"inject_chunk={inject_chunk:03d}: "
                        f"{args.main_policy} prefix -> {args.inject_policy} one chunk -> {args.main_policy} suffix"
                    )

                    result = _run_recovery_trial(
                        env,
                        task_description,
                        initial_state,
                        main_client,
                        inject_client,
                        main_bank,
                        args,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        inject_chunk=inject_chunk,
                    )

                    total_env_steps = int(result["total_env_steps"])
                    success = bool(result["success"])

                    episode_result["chunks"].append(
                        {
                            "chunk": int(inject_chunk),
                            "rescue_tag": str(args.inject_policy),
                            "success": success,
                            "total_env_steps": total_env_steps,
                            "success_at": _success_at(success, total_env_steps, thresholds),
                        }
                    )

                    logging.info(
                        f"[Result] task={task_id:02d} ep={episode_idx:03d} "
                        f"chunk={inject_chunk:03d} success={success} total_env_steps={total_env_steps}"
                    )

                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(episode_result, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)

                summary_records.append(
                    {
                        "task_id": int(task_id),
                        "task_ep": int(episode_idx),
                        "num_chunks": len(episode_result["chunks"]),
                        "num_success": sum(1 for x in episode_result["chunks"] if x["success"]),
                        "num_fail": sum(1 for x in episode_result["chunks"] if not x["success"]),
                        "path": str(out_path),
                    }
                )
                logging.info(f"[Saved] {out_path}")

        finally:
            if env is not None:
                env.close()

    with (base_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)

    logging.info(f"[Saved] summary={base_dir / 'summary.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tyro.cli(eval_libero)

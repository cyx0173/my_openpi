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

# Keep recovery budget consistent with the W4A4 bank collection script:
# total budget = max_steps + num_steps_wait = 320 action steps + 10 wait steps for libero_10.
_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 320,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # Server parameters
    host: str = "0.0.0.0"
    port_w4a8: int = 8000
    port_w4a16: int = 8001

    # LIBERO parameters
    task_suite_name: str = "libero_10"
    task_ids: str = "8,9"
    episode_start: int = 0
    episode_end: int = 10
    seed: int = 7

    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10

    # Debug-noise parameters.
    # For chunks already present in the W4A4 bank, this script reuses the stored
    # bank["chunks"][chunk_idx]["noise_seed"] instead of recomputing another formula.
    use_debug_noise: bool = True
    debug_noise_base_seed: int = 700000
    debug_noise_horizon: int = 10
    debug_noise_dim: int = 32

    # Output directory: oracle labels are written here.
    base_dir: str = "/home/chengyuxuan/openpi/recovery/mode2"

    # Bank directory parent: W4A4 bank is read from bank_base_dir / "w4a4_action_bank".
    # If empty, this falls back to base_dir.
    bank_base_dir: str = "/home/chengyuxuan/openpi/recovery/mode1"

    output_name: str = "oracle_labels.jsonl"
    skip_success_bank: bool = True
    overwrite_output: bool = True

    # Debug logging. This prints each recovery inference chunk, so logs are verbose but traceable.
    log_every_recovery_chunk: bool = True


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


def _append_jsonl(path: pathlib.Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, cls=_NumpyEncoder) + "\n")


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
    text = str(args.task_ids).strip().lower()
    if text in {"all", "*"}:
        return list(range(num_tasks_in_suite))

    task_ids = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(item))

    invalid = [x for x in task_ids if x < 0 or x >= num_tasks_in_suite]
    if invalid:
        raise ValueError(
            f"Invalid task ids {invalid}; suite has task ids 0..{num_tasks_in_suite - 1}"
        )
    return task_ids


def _resolve_episode_indices(args: Args, num_initial_states: int):
    start = max(0, int(args.episode_start))
    end = min(int(args.episode_end), num_initial_states)
    if end < start:
        return []
    return list(range(start, end))


def _make_debug_noise_from_seed(args: Args, noise_seed: int):
    rng = np.random.default_rng(int(noise_seed))
    return rng.standard_normal(
        size=(int(args.debug_noise_horizon), int(args.debug_noise_dim))
    ).astype(np.float32)


def _fallback_noise_seed(args: Args, chunk_idx: int):
    # Match the collection script's formula: base_seed + chunk_idx.
    return int(args.debug_noise_base_seed) + int(chunk_idx)


def _get_noise_seed_from_bank(args: Args, bank_chunks, chunk_idx: int):
    """
    Prefer the exact noise seed stored in the W4A4 bank.

    If recovery runs beyond the stored bank chunks, fall back to the same formula
    used by the collection script: base_seed + chunk_idx.
    """
    chunk_idx = int(chunk_idx)
    if 0 <= chunk_idx < len(bank_chunks):
        seed = bank_chunks[chunk_idx].get("noise_seed", None)
        if seed is not None:
            return int(seed)
    return _fallback_noise_seed(args, chunk_idx)


def _add_debug_noise_with_seed(element: dict, args: Args, noise_seed: int):
    if not args.use_debug_noise:
        return element, None

    debug_noise = _make_debug_noise_from_seed(args, int(noise_seed))
    out = dict(element)
    out["debug_noise"] = debug_noise.copy()
    return out, int(noise_seed)


def _to_action_list(actions, replan_steps: int):
    actions_np = np.asarray(actions)
    if len(actions_np) < int(replan_steps):
        raise ValueError(
            f"Policy returned {len(actions_np)} actions, but replan_steps={replan_steps}"
        )
    return actions_np[: int(replan_steps)].tolist()


def _replay_w4a4_prefix(env, obs, bank_chunks, switch_chunk: int):
    """
    Replay W4A4 chunks [0, switch_chunk) exactly.

    switch_chunk = 0:
        replay nothing; recover immediately after wait steps.
    switch_chunk = 1:
        replay chunk 0 completely; recover from chunk 1 boundary.
    switch_chunk = k:
        replay chunks 0..k-1 completely.
    """
    t_after_wait = 0
    replayed_chunks = 0
    max_prefix_chunks = min(int(switch_chunk), len(bank_chunks))

    for chunk_idx in range(max_prefix_chunks):
        actions = bank_chunks[chunk_idx]["executed_actions"]
        for action in actions:
            obs, _, done, _ = env.step(action)
            t_after_wait += 1
            if done:
                return obs, True, chunk_idx + 1, t_after_wait
        replayed_chunks += 1

    return obs, False, replayed_chunks, t_after_wait


def _run_recovery_from_bank(
    env,
    task_description: str,
    initial_state,
    bank: dict,
    recovery_client,
    args: Args,
    *,
    switch_chunk: int,
    recovery_precision: str,
):
    task_id = int(bank["task_id"])
    episode_idx = int(bank["episode_idx"])
    bank_chunks = bank["chunks"]

    env.reset()
    obs = env.set_init_state(initial_state)

    done = False
    t = 0
    max_steps = _MAX_STEPS.get(args.task_suite_name, 320)
    total_step_budget = max_steps + int(args.num_steps_wait)

    # Match W4A4 collection: wait num_steps_wait simulator steps first.
    while t < int(args.num_steps_wait):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        t += 1
        if done:
            return {
                "success": True,
                "total_steps": int(t),
                "num_chunks": 0,
                "replayed_chunks": 0,
                "replayed_steps": 0,
                "recovery_chunks": 0,
                "final_by": "dummy_wait",
                "first_recovery_noise_seed": None,
                "last_recovery_noise_seed": None,
            }

    # Replay W4A4 prefix exactly until the chosen chunk boundary.
    obs, done, replayed_chunks, replayed_steps = _replay_w4a4_prefix(
        env,
        obs,
        bank_chunks,
        switch_chunk,
    )
    t += replayed_steps

    if done:
        return {
            "success": True,
            "total_steps": int(t),
            "num_chunks": int(replayed_chunks),
            "replayed_chunks": int(replayed_chunks),
            "replayed_steps": int(replayed_steps),
            "recovery_chunks": 0,
            "final_by": "w4a4_prefix",
            "first_recovery_noise_seed": None,
            "last_recovery_noise_seed": None,
        }

    action_plan = collections.deque()
    chunk_idx = int(switch_chunk)
    recovery_chunks = 0
    first_recovery_noise_seed = None
    last_recovery_noise_seed = None

    while t < total_step_budget:
        if not action_plan:
            noise_seed = _get_noise_seed_from_bank(args, bank_chunks, chunk_idx)
            if first_recovery_noise_seed is None:
                first_recovery_noise_seed = int(noise_seed)
            last_recovery_noise_seed = int(noise_seed)

            if args.log_every_recovery_chunk:
                logging.info(
                    f"[{recovery_precision.upper()}-CHUNK] "
                    f"task={task_id} ep={episode_idx} "
                    f"switch={switch_chunk} infer_chunk={chunk_idx} "
                    f"t={t}/{total_step_budget} noise_seed={noise_seed}"
                )

            element = _get_obs_element(obs, task_description, args.resize_size)
            element, _ = _add_debug_noise_with_seed(
                element,
                args=args,
                noise_seed=noise_seed,
            )

            result = recovery_client.infer(element)
            action_plan.extend(_to_action_list(result["actions"], args.replan_steps))

            chunk_idx += 1
            recovery_chunks += 1

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action)
        t += 1
        if done:
            break

    final_by = recovery_precision if done else "timeout"

    return {
        "success": bool(done),
        "total_steps": int(t),
        "num_chunks": int(chunk_idx),
        "replayed_chunks": int(replayed_chunks),
        "replayed_steps": int(replayed_steps),
        "recovery_chunks": int(recovery_chunks),
        "final_by": final_by,
        "first_recovery_noise_seed": first_recovery_noise_seed,
        "last_recovery_noise_seed": last_recovery_noise_seed,
    }


def _load_bank(bank_path: pathlib.Path):
    with open(bank_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bank_step_start(bank: dict, switch_chunk: int):
    chunks = bank.get("chunks", [])
    if 0 <= int(switch_chunk) < len(chunks):
        return chunks[int(switch_chunk)].get("step_start", None)
    return None


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    out_base_dir = pathlib.Path(args.base_dir)
    bank_base_dir = pathlib.Path(args.bank_base_dir) if args.bank_base_dir else out_base_dir

    bank_dir = bank_base_dir / "w4a4_action_bank"
    out_path = out_base_dir / args.output_name
    out_base_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite_output and out_path.exists():
        out_path.unlink()
        logging.info(f"[Init] Removed existing output: {out_path}")

    logging.info(f"Bank dir: {bank_dir}")
    logging.info(f"Output: {out_path}")
    logging.info(f"Max steps: {_MAX_STEPS.get(args.task_suite_name, 320)} + wait {args.num_steps_wait}")
    logging.info("Noise: reuse bank chunk noise_seed; fallback=base_seed+chunk_idx")
    logging.info(f"W4A8 server: ws://{args.host}:{args.port_w4a8}")
    logging.info(f"W4A16 server: ws://{args.host}:{args.port_w4a16}")

    if not bank_dir.exists():
        logging.warning(f"[Warning] Bank dir does not exist: {bank_dir}")

    w4a8_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port_w4a8
    )
    w4a16_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port_w4a16
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    selected_task_ids = _resolve_task_ids(args, task_suite.n_tasks)
    logging.info(f"Selected task ids: {selected_task_ids}")

    label_counter = collections.Counter()
    missing_banks = 0
    skipped_success_banks = 0
    processed_switches = 0

    for task_id in tqdm.tqdm(selected_task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        episode_indices = _resolve_episode_indices(args, len(initial_states))

        env = None
        try:
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            logging.info(
                f"[TASK] task={task_id} episodes={episode_indices} desc={task_description}"
            )

            for episode_idx in tqdm.tqdm(
                episode_indices,
                desc=f"task{task_id:02d}",
                leave=False,
            ):
                bank_path = bank_dir / f"task{task_id:02d}_ep{episode_idx:03d}_w4a4_bank.json"

                if not bank_path.exists():
                    missing_banks += 1
                    logging.warning(f"[Skip] missing bank: {bank_path}")
                    continue

                bank = _load_bank(bank_path)
                bank_success = bool(bank.get("success", False))
                bank_num_chunks = int(bank.get("num_chunks", len(bank.get("chunks", []))))
                bank_total_steps = int(bank.get("total_steps", -1))

                if bank_success and args.skip_success_bank:
                    skipped_success_banks += 1
                    logging.info(
                        f"[Skip] W4A4 success bank: task={task_id} ep={episode_idx} "
                        f"steps={bank_total_steps} chunks={bank_num_chunks} path={bank_path}"
                    )
                    continue

                initial_state = np.array(
                    initial_states[episode_idx].tolist()
                    if hasattr(initial_states[episode_idx], "tolist")
                    else initial_states[episode_idx]
                )

                logging.info(
                    f"[EPISODE] task={task_id} ep={episode_idx} "
                    f"w4a4_success={bank_success} w4a4_steps={bank_total_steps} "
                    f"num_chunks={bank_num_chunks}"
                )

                for switch_chunk in tqdm.tqdm(
                    range(bank_num_chunks),
                    desc=f"task{task_id:02d}_ep{episode_idx:03d}_chunks",
                    leave=False,
                ):
                    noise_seed_at_switch = _get_noise_seed_from_bank(
                        args,
                        bank["chunks"],
                        switch_chunk,
                    )
                    step_start = _bank_step_start(bank, switch_chunk)

                    logging.info(
                        f"[SWITCH] task={task_id} ep={episode_idx} "
                        f"switch_chunk={switch_chunk}/{bank_num_chunks - 1} "
                        f"bank_step_start={step_start} noise_seed={noise_seed_at_switch}"
                    )

                    w4a8_result = _run_recovery_from_bank(
                        env,
                        task_description,
                        initial_state,
                        bank,
                        w4a8_client,
                        args,
                        switch_chunk=switch_chunk,
                        recovery_precision="w4a8",
                    )

                    logging.info(
                        f"[A8-RESULT] task={task_id} ep={episode_idx} switch={switch_chunk} "
                        f"success={w4a8_result['success']} "
                        f"final_by={w4a8_result['final_by']} "
                        f"steps={w4a8_result['total_steps']} "
                        f"replayed_chunks={w4a8_result['replayed_chunks']} "
                        f"recovery_chunks={w4a8_result['recovery_chunks']}"
                    )

                    w4a16_result = None
                    if w4a8_result["success"]:
                        if w4a8_result["final_by"] == "w4a4_prefix":
                            oracle_label = "w4a4_prefix"
                        elif w4a8_result["final_by"] == "dummy_wait":
                            oracle_label = "dummy_wait"
                        else:
                            oracle_label = "w4a8"
                    else:
                        w4a16_result = _run_recovery_from_bank(
                            env,
                            task_description,
                            initial_state,
                            bank,
                            w4a16_client,
                            args,
                            switch_chunk=switch_chunk,
                            recovery_precision="w4a16",
                        )

                        logging.info(
                            f"[A16-RESULT] task={task_id} ep={episode_idx} switch={switch_chunk} "
                            f"success={w4a16_result['success']} "
                            f"final_by={w4a16_result['final_by']} "
                            f"steps={w4a16_result['total_steps']} "
                            f"replayed_chunks={w4a16_result['replayed_chunks']} "
                            f"recovery_chunks={w4a16_result['recovery_chunks']}"
                        )

                        if w4a16_result["success"]:
                            if w4a16_result["final_by"] == "w4a4_prefix":
                                oracle_label = "w4a4_prefix"
                            elif w4a16_result["final_by"] == "dummy_wait":
                                oracle_label = "dummy_wait"
                            else:
                                oracle_label = "w4a16"
                        else:
                            oracle_label = "unrecoverable"

                    record = {
                        "task_id": int(task_id),
                        "task_description": str(task_description),
                        "episode_idx": int(episode_idx),
                        "switch_chunk": int(switch_chunk),
                        "bank_step_start": None if step_start is None else int(step_start),
                        "noise_seed": int(noise_seed_at_switch),
                        "oracle_label": oracle_label,
                        "w4a4_bank_success": bank_success,
                        "w4a4_bank_total_steps": bank_total_steps,
                        "w4a4_bank_num_chunks": bank_num_chunks,
                        "w4a8_success": bool(w4a8_result["success"]),
                        "w4a8_final_by": w4a8_result["final_by"],
                        "w4a8_steps": int(w4a8_result["total_steps"]),
                        "w4a8_replayed_chunks": int(w4a8_result["replayed_chunks"]),
                        "w4a8_replayed_steps": int(w4a8_result["replayed_steps"]),
                        "w4a8_recovery_chunks": int(w4a8_result["recovery_chunks"]),
                        "w4a8_first_recovery_noise_seed": w4a8_result["first_recovery_noise_seed"],
                        "w4a8_last_recovery_noise_seed": w4a8_result["last_recovery_noise_seed"],
                        "w4a16_success": None if w4a16_result is None else bool(w4a16_result["success"]),
                        "w4a16_final_by": None if w4a16_result is None else w4a16_result["final_by"],
                        "w4a16_steps": None if w4a16_result is None else int(w4a16_result["total_steps"]),
                        "w4a16_replayed_chunks": None if w4a16_result is None else int(w4a16_result["replayed_chunks"]),
                        "w4a16_replayed_steps": None if w4a16_result is None else int(w4a16_result["replayed_steps"]),
                        "w4a16_recovery_chunks": None if w4a16_result is None else int(w4a16_result["recovery_chunks"]),
                        "w4a16_first_recovery_noise_seed": None if w4a16_result is None else w4a16_result["first_recovery_noise_seed"],
                        "w4a16_last_recovery_noise_seed": None if w4a16_result is None else w4a16_result["last_recovery_noise_seed"],
                    }

                    _append_jsonl(out_path, record)
                    label_counter[oracle_label] += 1
                    processed_switches += 1

                    logging.info(
                        f"[LABEL] task={task_id} ep={episode_idx} switch={switch_chunk} "
                        f"label={oracle_label} "
                        f"a8={w4a8_result['success']} "
                        f"a16={None if w4a16_result is None else w4a16_result['success']}"
                    )

        finally:
            if env is not None:
                env.close()

    logging.info("=" * 80)
    logging.info(f"Done. Labels saved to: {out_path}")
    logging.info(f"Processed switches: {processed_switches}")
    logging.info(f"Missing banks: {missing_banks}")
    logging.info(f"Skipped successful W4A4 banks: {skipped_success_banks}")
    logging.info(f"Label counts: {dict(label_counter)}")
    logging.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    tyro.cli(eval_libero)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RestoredChunkStart:
    chunk_idx: int
    step_start: int
    checkpoint_path: str
    obs: dict
    bank_chunk: dict[str, Any]
    restore_report: dict[str, Any]


def find_bank_chunk(bank_record: dict[str, Any], chunk_idx: int) -> dict[str, Any]:
    """
    Find one chunk record from action bank by chunk_idx.
    """
    for ch in bank_record.get("chunks", []):
        if int(ch.get("chunk_idx", -1)) == int(chunk_idx):
            return ch
    raise KeyError(f"Cannot find chunk_idx={chunk_idx} in bank_record")


def get_current_obs_after_restore(env) -> dict:
    """
    Get fresh observation from restored env state.

    This should be used after load_env_checkpoint(...).
    Do not use trace-saved policy_input_start as main rollout input.
    """

    # Refresh robot / controller / observable cache if helper exists.
    # In collection script we already have _refresh_env_after_restore(env).
    if "_refresh_env_after_restore" in globals():
        try:
            _refresh_env_after_restore(env)
        except Exception:
            pass

    # Try wrapper chain: env, env.env, env.env.env, ...
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))

        if hasattr(cur, "_get_observations"):
            try:
                try:
                    obs = cur._get_observations(force_update=True)
                except TypeError:
                    obs = cur._get_observations()

                if isinstance(obs, dict):
                    return obs
            except Exception:
                pass

        cur = getattr(cur, "env", None)

    raise RuntimeError("Cannot get fresh observation after checkpoint restore")


def restore_env_to_chunk_start(
    *,
    env,
    bank_record: dict[str, Any],
    chunk_idx: int,
    initial_state: np.ndarray | None = None,
) -> RestoredChunkStart:
    """
    Restore env to the start of a collected chunk.

    This is the first building block of rescue runner.

    It intentionally does NOT:
      - replay old prefix actions
      - call policy
      - execute any action
      - use trace saved input as rollout input

    It only does:
      1. env.reset()
      2. env.set_init_state(initial_state)
      3. load_env_checkpoint(...)
      4. get fresh obs from restored env
    """

    bank_chunk = find_bank_chunk(bank_record, chunk_idx)

    ckpt_meta = bank_chunk.get("env_checkpoint")
    if not isinstance(ckpt_meta, dict):
        raise ValueError(
            f"chunk {chunk_idx} has no env_checkpoint. "
            "This bank was probably collected before per-chunk checkpoint support."
        )

    checkpoint_path = ckpt_meta.get("path")
    if not checkpoint_path:
        raise ValueError(f"chunk {chunk_idx} env_checkpoint has no path")

    checkpoint_path = str(checkpoint_path)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    if initial_state is None:
        if "initial_state" not in bank_record:
            raise ValueError("initial_state not provided and not found in bank_record")
        initial_state = np.asarray(bank_record["initial_state"])

    # Important:
    # We still call reset + set_init_state before loading checkpoint.
    # This constructs the correct task scene / object layout / wrappers.
    env.reset()
    env.set_init_state(initial_state)

    # load_env_checkpoint is the helper from collection script.
    # It should:
    #   sim.set_state_from_flattened(...)
    #   restore ctrl/mocap/time/...
    #   sim.forward()
    #   restore qacc_warmstart
    #   sync bookkeeping
    #   refresh controller cache
    restore_report = load_env_checkpoint(env, checkpoint_path)

    obs = get_current_obs_after_restore(env)

    step_start = int(bank_chunk.get("step_start", ckpt_meta.get("step", -1)))
    if step_start < 0:
        raise ValueError(f"Cannot determine step_start for chunk {chunk_idx}")

    return RestoredChunkStart(
        chunk_idx=int(chunk_idx),
        step_start=step_start,
        checkpoint_path=checkpoint_path,
        obs=obs,
        bank_chunk=bank_chunk,
        restore_report=restore_report,
    )
'''
1. env.reset()
2. env.set_init_state(initial_state)
3. load_env_checkpoint(chunkXXXX_start.npz)
end reset核心逻辑，后续会执行policy并step
'''
'''
bank_record = load_json(bank_path)
initial_state = np.asarray(bank_record["initial_state"])

restored = restore_env_to_chunk_start(
    env=env,
    bank_record=bank_record,
    chunk_idx=37,
    initial_state=initial_state,
)

obs = restored.obs
t = restored.step_start
chunk_idx = restored.chunk_idx

element = _get_policy_images_and_proprio(obs, task_description, resize_size)
noise = make_debug_noise(task_id, episode_idx, chunk_idx)
result = policy_client.infer(element)
env.step(action)


'''
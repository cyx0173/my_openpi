#!/usr/bin/env python3
"""
07_greedy_recovery_runner.py

Final greedy recovery runner skeleton + core algorithm.

This file contains the completed greedy/search/cache logic.  The only parts that
must be connected to your OpenPI/LIBERO runtime are the methods in RuntimeAdapter.

Core rules implemented here:
  1. restore uses checkpoint/cache, not old action replay as the main path.
  2. rollout actions are generated online by policy inference.
  3. noise_seed is always read from the original A4 action bank for the same chunk.
  4. baseline start validation uses full-episode env_success.
  5. greedy downgrade uses target_stage_state only.
  6. final validation uses full-episode env_success.
  7. final output is only chunk-level schedule: chunk_idx / precision / noise_seed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


PRECISION_COST = {
    "w4a4": 1,
    "w4a8": 2,
    "w4a16": 4,
}

PRECISIONS = ("w4a4", "w4a8", "w4a16")

# Recovery should follow the same episode step budget used by the original
# baseline collection.  For LIBERO-10 this is 520 policy/env steps after the
# initial waiting steps, not 600.
_RECOVERY_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# =============================================================================
# Basic IO / JSON helpers
# =============================================================================

def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=indent)


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def to_jsonable(x: Any) -> Any:
    """Best-effort conversion for numpy / torch objects without importing them."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # numpy scalar / array
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass
    if hasattr(x, "tolist") and callable(getattr(x, "tolist")):
        try:
            return x.tolist()
        except Exception:
            pass

    # fallback
    return str(x)


def unique_keep_order(xs: list[Any]) -> list[Any]:
    out = []
    seen = set()
    for x in xs:
        if x is None:
            continue
        key = str(x)
        if key not in seen:
            out.append(x)
            seen.add(key)
    return out


def stable_hash(obj: Any) -> str:
    raw = json.dumps(to_jsonable(obj), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def debug_log(cfg: Any, msg: str) -> None:
    """Lightweight debug printer. Output goes to the per-job log file in client.sh."""
    if cfg is not None and getattr(cfg, "debug_print", False):
        print(msg, flush=True)


# =============================================================================
# Action bank helpers
# =============================================================================

def find_bank_chunk(bank_record: dict[str, Any], chunk_idx: int) -> dict[str, Any]:
    for ch in bank_record.get("chunks", []) or []:
        if int(ch.get("chunk_idx", -1)) == int(chunk_idx):
            return ch
    raise KeyError(f"Cannot find chunk_idx={chunk_idx} in bank_record")


def get_bank_chunk_optional(bank_record: dict[str, Any], chunk_idx: int) -> Optional[dict[str, Any]]:
    try:
        return find_bank_chunk(bank_record, chunk_idx)
    except Exception:
        return None


def _read_noise_seed_from_chunk(ch: dict[str, Any]) -> Optional[int]:
    for key in ("noise_seed", "debug_noise_seed", "seed"):
        if key in ch and ch[key] is not None:
            return int(ch[key])
    # Some banks may put metadata under nested keys.
    for meta_key in ("noise", "debug_noise", "policy_noise"):
        meta = ch.get(meta_key)
        if isinstance(meta, dict):
            for key in ("noise_seed", "debug_noise_seed", "seed"):
                if key in meta and meta[key] is not None:
                    return int(meta[key])
    return None


def get_noise_seed_from_bank(bank_record: dict[str, Any], chunk_idx: int) -> Optional[int]:
    ch = get_bank_chunk_optional(bank_record, chunk_idx)
    if ch is None:
        return None
    return _read_noise_seed_from_chunk(ch)


def infer_noise_base_seed_from_bank(bank_record: dict[str, Any], job: dict[str, Any]) -> Optional[int]:
    """Infer collector debug_noise_base_seed from existing bank chunks.

    Collector formula:
      seed = base_seed + task_id * 100000 + episode_idx * 1000 + chunk_idx

    This lets recovery continue beyond the original A4 bank length while still
    using the same deterministic noise convention as the original collection.
    """
    task_id = int(job.get("task_id", bank_record.get("task_id", 0)) or 0)
    episode_idx = int(job.get("episode_idx", bank_record.get("episode_idx", 0)) or 0)

    bases = []
    for ch in bank_record.get("chunks", []) or []:
        if "chunk_idx" not in ch:
            continue
        seed = _read_noise_seed_from_chunk(ch)
        if seed is None:
            continue
        c = int(ch["chunk_idx"])
        bases.append(int(seed) - task_id * 100000 - episode_idx * 1000 - c)

    if not bases:
        return None

    # If there are multiple due to corrupted data, use the most frequent one.
    counts: dict[int, int] = {}
    for b in bases:
        counts[b] = counts.get(b, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def fallback_noise_seed(job: dict[str, Any], chunk_idx: int, base_seed: int) -> int:
    """Deterministic fallback using the collector's debug-noise formula."""
    task_id = int(job.get("task_id", 0) or 0)
    episode_idx = int(job.get("episode_idx", 0) or 0)
    return int(base_seed) + task_id * 100000 + episode_idx * 1000 + int(chunk_idx)


def get_noise_seed_for_chunk(
    *,
    bank_record: dict[str, Any],
    chunk_idx: int,
    job: dict[str, Any],
    cfg: Optional["GreedyConfig"] = None,
) -> tuple[int, bool]:
    """Return (noise_seed, used_fallback).

    Default behavior is strict: if the original A4 bank does not contain a seed,
    raise an error.  Fallback is allowed only when cfg.allow_noise_seed_fallback=True.
    """
    seed = get_noise_seed_from_bank(bank_record, chunk_idx)
    if seed is not None:
        return int(seed), False

    # Important for recovery: A8/A16 may run more chunks than the original failed
    # A4 bank.  In that case the bank has no chunk record, but the noise seed is
    # still well-defined by the collector formula.  Infer base_seed from existing
    # chunks and extend the same formula.
    inferred_base_seed = infer_noise_base_seed_from_bank(bank_record, job)
    if inferred_base_seed is not None:
        seed = fallback_noise_seed(job, chunk_idx, inferred_base_seed)
        if cfg is not None and getattr(cfg, "debug_noise_extend", False):
            debug_log(
                cfg,
                f"[Noise扩展] 原始A4没有 chunk {chunk_idx}，按采集公式补 seed："
                f"base={inferred_base_seed} seed={seed}",
            )
        return seed, True

    if cfg is not None and cfg.allow_noise_seed_fallback:
        seed = fallback_noise_seed(job, chunk_idx, cfg.fallback_noise_base_seed)
        debug_log(
            cfg,
            f"[NOISE_FALLBACK] chunk={chunk_idx} using explicit base_seed="
            f"{cfg.fallback_noise_base_seed} seed={seed}",
        )
        return seed, True

    raise ValueError(
        f"chunk {chunk_idx} has no noise_seed/debug_noise_seed/seed in original A4 bank "
        "and base_seed could not be inferred from existing chunks."
    )


def get_initial_state_from_bank(bank_record: dict[str, Any]) -> Any:
    if "initial_state" not in bank_record:
        raise KeyError("bank_record has no initial_state")
    return bank_record["initial_state"]


def original_bank_checkpoint_path(bank_record: dict[str, Any], chunk_idx: int) -> Optional[str]:
    ch = get_bank_chunk_optional(bank_record, chunk_idx)
    if not ch:
        return None
    meta = ch.get("env_checkpoint")
    if not isinstance(meta, dict):
        return None
    p = meta.get("path")
    return str(p) if p else None


# =============================================================================
# Runtime adapter: connect these methods to your existing OpenPI/LIBERO code
# =============================================================================

@dataclass
class RestoredChunkStart:
    chunk_idx: int
    step_start: int
    checkpoint_path: str
    obs: dict[str, Any]
    bank_chunk: dict[str, Any]
    restore_report: dict[str, Any]


class RuntimeAdapter:
    """
    Connect this class to your existing environment and policy clients.

    Required runtime behavior:
      - reset env and set initial state
      - save/load env checkpoint
      - get fresh obs after restore/step
      - get privileged raw state
      - infer action chunk with a specified precision and noise_seed
      - step an action chunk
      - report env_success

    The algorithm code below does not assume a particular OpenPI import path.
    """

    def reset_to_initial_state(self, initial_state: Any) -> None:
        raise NotImplementedError

    def save_env_checkpoint(self, path: str | Path) -> dict[str, Any]:
        raise NotImplementedError

    def load_env_checkpoint(self, path: str | Path) -> dict[str, Any]:
        raise NotImplementedError

    def get_current_obs(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_privileged_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def env_success(self) -> bool:
        raise NotImplementedError

    def env_timeout(self) -> bool:
        return False

    def current_step(self) -> Optional[int]:
        return None

    def infer_action_chunk(
        self,
        *,
        precision: str,
        obs: dict[str, Any],
        task_description: str,
        noise_seed: int,
        chunk_idx: int,
    ) -> Any:
        raise NotImplementedError

    def step_action_chunk(self, action_chunk: Any) -> dict[str, Any]:
        """
        Execute a policy action chunk in env.

        Return may include any extra info; not required by the core algorithm.
        """
        raise NotImplementedError


# =============================================================================
# OpenPI / LIBERO runtime implementation
# =============================================================================

def _import_runtime_modules():
    """Import runtime helpers from files in the same directory.

    mode4_collect.py provides checkpoint save/load helpers.
    mode3_collector.py provides the older collector helpers; we keep it as fallback.
    """
    try:
        import mode4_collect as mode4
    except Exception as e:
        raise ImportError(
            "Cannot import mode4_collect.py. Put mode4_collect.py in the same directory "
            "as mode5_recovery.py, or add that directory to PYTHONPATH."
        ) from e

    try:
        import mode3_collector as mode3
    except Exception:
        mode3 = None

    return mode4, mode3


def _parse_chunk_idx_from_checkpoint_path(path: str | Path) -> int:
    name = Path(path).name
    m = re.search(r"chunk[_]?(\d+)_start\.npz|chunk(\d+)_start\.npz", name)
    if not m:
        raise ValueError(f"Cannot parse chunk_idx from checkpoint path: {path}")
    return int(next(g for g in m.groups() if g is not None))


def _get_obs_after_restore_like(env, mode4) -> dict[str, Any]:
    """Get fresh observation from env / wrapper chain."""
    if hasattr(mode4, "_refresh_env_after_restore"):
        try:
            mode4._refresh_env_after_restore(env)
        except Exception:
            pass

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

    raise RuntimeError("Cannot get fresh observation from env")


def _add_debug_noise_by_seed(
    element: dict[str, Any],
    *,
    noise_seed: int,
    horizon: int,
    dim: int,
    send_debug_noise: bool = True,
) -> dict[str, Any]:
    out = dict(element)
    if send_debug_noise:
        rng = np.random.default_rng(int(noise_seed))
        debug_noise = rng.standard_normal(size=(int(horizon), int(dim))).astype(np.float32)
        out["debug_noise"] = debug_noise
    return out


class OpenPIRuntimeAdapter(RuntimeAdapter):
    """Concrete RuntimeAdapter for LIBERO + OpenPI websocket policy servers."""

    def __init__(self, job: dict[str, Any], args: argparse.Namespace):
        self.job = job
        self.args = args
        self.mode4, self.mode3 = _import_runtime_modules()

        self.task_id = int(job["task_id"])
        self.episode_idx = int(job["episode_idx"])
        self.task_suite_name = str(getattr(args, "task_suite_name", "libero_10"))
        self.seed = int(getattr(args, "seed", 7))
        self.resize_size = int(getattr(args, "resize_size", 224))
        self.replan_steps = int(getattr(args, "replan_steps", 5))
        self.num_steps_wait = int(getattr(args, "num_steps_wait", 10))
        self.max_steps = int(
            getattr(args, "max_env_steps", None)
            or _RECOVERY_MAX_STEPS.get(self.task_suite_name, 600)
        )
        self.debug_noise_horizon = int(getattr(args, "debug_noise_horizon", 10))
        self.debug_noise_dim = int(getattr(args, "debug_noise_dim", 32))
        self.send_debug_noise = bool(getattr(args, "send_debug_noise", True))
        self.save_contacts = bool(getattr(args, "save_contacts", True))

        benchmark_dict = self.mode4.benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.task_suite_name]()
        self.task = self.task_suite.get_task(self.task_id)

        self.env, self.task_description = self.mode4._get_libero_env(
            self.task,
            self.mode4.LIBERO_ENV_RESOLUTION,
            self.seed,
        )
        self.task_description = str(job.get("instruction") or self.task_description)

        host = str(getattr(args, "policy_host", "127.0.0.1"))
        self.policy_clients = {
            "w4a4": self.mode4._websocket_client_policy.WebsocketClientPolicy(host, int(args.port_a4)),
            "w4a8": self.mode4._websocket_client_policy.WebsocketClientPolicy(host, int(args.port_a8)),
            "w4a16": self.mode4._websocket_client_policy.WebsocketClientPolicy(host, int(args.port_a16)),
        }

        # If URL args are provided, parse host/port from URL-like strings.
        # WebsocketClientPolicy itself takes host, port.
        for precision, url_arg in [
            ("w4a4", getattr(args, "url_a4", None)),
            ("w4a8", getattr(args, "url_a8", None)),
            ("w4a16", getattr(args, "url_a16", None)),
        ]:
            if url_arg:
                from urllib.parse import urlparse
                u = urlparse(str(url_arg))
                h = u.hostname or host
                p = u.port
                if p is None:
                    raise ValueError(f"{precision} URL has no port: {url_arg}")
                self.policy_clients[precision] = self.mode4._websocket_client_policy.WebsocketClientPolicy(h, int(p))

        self.body_names: list[str] = []
        self.joint_names: list[str] = []
        self.obs: Optional[dict[str, Any]] = None
        self.done: bool = False
        self.t: int = 0

    def _refresh_names(self) -> None:
        self.body_names = self.mode4._body_names_for_trace(self.env)
        self.joint_names = self.mode4._joint_names_for_trace(self.env)

    def reset_to_initial_state(self, initial_state: Any) -> None:
        initial_state = np.asarray(initial_state)
        self.env.reset()
        self.obs = self.env.set_init_state(initial_state)
        self.done = False
        self.t = 0
        self._refresh_names()

        # Match the collector: first wait num_steps_wait dummy actions, then chunk 0 starts.
        while self.t < self.num_steps_wait:
            self.obs, _, self.done, _ = self.env.step(self.mode4.LIBERO_DUMMY_ACTION)
            self.t += 1
            if self.done:
                break

    def save_env_checkpoint(self, path: str | Path) -> dict[str, Any]:
        chunk_idx = _parse_chunk_idx_from_checkpoint_path(path)
        return self.mode4.save_env_checkpoint(
            self.env,
            path,
            task_id=self.task_id,
            episode_idx=self.episode_idx,
            chunk_idx=chunk_idx,
            step=int(self.t),
        )

    def load_env_checkpoint(self, path: str | Path) -> dict[str, Any]:
        report = self.mode4.load_env_checkpoint(self.env, path)
        step = report.get("step")
        if step is not None:
            self.t = int(step)
        self.done = False
        self.obs = _get_obs_after_restore_like(self.env, self.mode4)
        self._refresh_names()
        return report

    def get_current_obs(self) -> dict[str, Any]:
        if self.obs is None:
            self.obs = _get_obs_after_restore_like(self.env, self.mode4)
        return self.obs

    def get_privileged_state(self) -> dict[str, Any]:
        obs = self.get_current_obs()
        if not self.body_names or not self.joint_names:
            self._refresh_names()
        return self.mode4._state_snapshot(
            self.env,
            obs,
            body_names=self.body_names,
            joint_names=self.joint_names,
            save_contacts=self.save_contacts,
        )

    def env_success(self) -> bool:
        return bool(self.done)

    def infer_action_chunk(
        self,
        *,
        precision: str,
        obs: dict[str, Any],
        task_description: str,
        noise_seed: int,
        chunk_idx: int,
    ) -> Any:
        if precision not in self.policy_clients:
            raise ValueError(f"Unknown precision for policy client: {precision}")

        element, _agent_img, _wrist_img, _proprio = self.mode4._get_policy_images_and_proprio(
            obs,
            task_description or self.task_description,
            self.resize_size,
        )
        element = _add_debug_noise_by_seed(
            element,
            noise_seed=int(noise_seed),
            horizon=self.debug_noise_horizon,
            dim=self.debug_noise_dim,
            send_debug_noise=self.send_debug_noise,
        )
        result = self.policy_clients[precision].infer(element)
        return self.mode4._to_action_list(result["actions"], self.replan_steps)

    def env_timeout(self) -> bool:
        return bool(self.t >= self.max_steps + self.num_steps_wait)

    def current_step(self) -> Optional[int]:
        return int(self.t)

    def step_action_chunk(self, action_chunk: Any) -> dict[str, Any]:
        actions = np.asarray(action_chunk, dtype=np.float64).tolist()
        executed = []
        timeout = False
        for action in actions[: self.replan_steps]:
            if self.env_timeout():
                timeout = True
                break
            self.obs, _reward, self.done, info = self.env.step(action)
            self.t += 1
            executed.append(action)
            if self.done:
                break
        if self.env_timeout() and not self.done:
            timeout = True
        return {
            "step": int(self.t),
            "done": bool(self.done),
            "timeout": bool(timeout),
            "num_actions_executed": int(len(executed)),
        }



def policy_endpoints_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Return policy server endpoints for A4/A8/A16.

    Recovery uses three precision policies:
      - w4a4: build restore cache, target-after fallback, and final A4 suffix
      - w4a8: baseline and greedy candidates
      - w4a16: baseline and greedy candidates

    You can either pass explicit URLs or pass host + three ports.
    """
    host = getattr(args, "policy_host", "127.0.0.1")
    return {
        "w4a4": getattr(args, "url_a4", None) or f"http://{host}:{int(args.port_a4)}",
        "w4a8": getattr(args, "url_a8", None) or f"http://{host}:{int(args.port_a8)}",
        "w4a16": getattr(args, "url_a16", None) or f"http://{host}:{int(args.port_a16)}",
    }


def build_runtime(job: dict[str, Any], args: argparse.Namespace) -> RuntimeAdapter:
    """Build the concrete LIBERO/OpenPI runtime."""
    return OpenPIRuntimeAdapter(job, args)


# =============================================================================
# Restore cache
# =============================================================================

@dataclass
class RestoreCachePaths:
    case_dir: Path
    manifest_path: Path
    prefix_chunks_path: Path
    env_states_dir: Path


def restore_cache_paths(cache_root: str | Path, case_id: str) -> RestoreCachePaths:
    case_dir = Path(cache_root) / case_id
    return RestoreCachePaths(
        case_dir=case_dir,
        manifest_path=case_dir / "manifest.json",
        prefix_chunks_path=case_dir / "prefix_chunks.json",
        env_states_dir=case_dir / "env_states",
    )


def candidate_start_chunks(job: dict[str, Any]) -> list[int]:
    rs = job.get("recovery_start") or {}
    xs = [rs.get("preferred_chunk")] + list(rs.get("candidate_chunks") or [])
    return [int(x) for x in unique_keep_order(xs)]


def preferred_chunk(job: dict[str, Any]) -> Optional[int]:
    rs = job.get("recovery_start") or {}
    x = rs.get("preferred_chunk")
    return int(x) if x is not None else None


def ordered_start_chunks(job: dict[str, Any], cfg: GreedyConfig) -> list[dict[str, Any]]:
    """Return ordered starts with role annotations.

    In fallback mode:
      first = preferred_chunk
      rest = candidate chunks, used only if preferred fails.
    In exhaustive mode:
      same order, but caller may continue after success.
    """
    pref = preferred_chunk(job)
    starts = candidate_start_chunks(job)
    out = []
    for i, c in enumerate(starts):
        role = "preferred" if pref is not None and int(c) == int(pref) else "fallback"
        out.append({"chunk": int(c), "role": role, "index": int(i)})
    return out


def live_schedule_path(cfg: GreedyConfig, case_id: str) -> Path:
    return Path(cfg.output_root) / str(case_id) / "live_greedy_schedule.json"


def live_decisions_path(cfg: GreedyConfig, case_id: str) -> Path:
    return Path(cfg.output_root) / str(case_id) / "live_greedy_decisions.jsonl"


def save_live_greedy_schedule(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    cfg: GreedyConfig,
    start_chunk: int,
    suffix_precision: str,
    accepted_prefix: dict[int, str],
    last_decision: Optional[dict[str, Any]] = None,
) -> None:
    """Persist the current greedy decisions immediately.

    This is intentionally a *live / partial* file.  Final correctness is still
    decided by recovery_schedule.json after final validation.
    """
    case_id = str(job.get("case_id"))
    rows = []
    for c, precision in sorted((int(k), v) for k, v in accepted_prefix.items()):
        try:
            noise_seed, used_fallback = get_noise_seed_for_chunk(
                bank_record=bank_record,
                chunk_idx=int(c),
                job=job,
                cfg=cfg,
            )
        except Exception:
            noise_seed, used_fallback = None, False
        row = {
            "chunk_idx": int(c),
            "precision": str(precision),
            "fixed": True,
        }
        if noise_seed is not None:
            row["noise_seed"] = int(noise_seed)
        if used_fallback:
            row["noise_seed_fallback"] = True
        rows.append(row)

    record = {
        "case_id": case_id,
        "task_id": job.get("task_id"),
        "episode_idx": job.get("episode_idx"),
        "instruction": job.get("instruction"),
        "status": "greedy_in_progress",
        "note": "Partial schedule written after every irreversible greedy decision; final schedule is recovery_schedule.json.",
        "start_chunk": int(start_chunk),
        "suffix_precision_for_unfixed_chunks": str(suffix_precision),
        "num_fixed_chunks": int(len(rows)),
        "last_decision": last_decision,
        "fixed_chunk_schedule": rows,
    }
    save_json(live_schedule_path(cfg, case_id), record)

    if last_decision is not None:
        p = live_decisions_path(cfg, case_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(last_decision), ensure_ascii=False) + "\n")


def checkpoint_name(chunk_idx: int) -> str:
    return f"chunk_{int(chunk_idx):04d}_start.npz"


def has_original_checkpoints(bank_record: dict[str, Any], chunks_to_check: list[int]) -> bool:
    for c in chunks_to_check:
        p = original_bank_checkpoint_path(bank_record, c)
        if not p or not Path(p).exists():
            return False
    return True


def build_manifest_from_original_bank(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    chunks_to_cache: list[int],
    cache_paths: RestoreCachePaths,
    cfg: "GreedyConfig",
) -> dict[str, Any]:
    cached: dict[str, Any] = {}
    for c in chunks_to_cache:
        ch = find_bank_chunk(bank_record, c)
        p = original_bank_checkpoint_path(bank_record, c)
        if not p:
            raise ValueError(f"original bank chunk {c} has no env_checkpoint")
        noise_seed, used_fallback = get_noise_seed_for_chunk(
            bank_record=bank_record,
            chunk_idx=c,
            job=job,
            cfg=cfg,
        )
        cached[str(c)] = {
            "chunk_idx": int(c),
            "step_start": int(ch.get("step_start", ch.get("env_checkpoint", {}).get("step", -1))),
            "state_path": str(p),
            "noise_seed": noise_seed,
            "noise_seed_fallback": used_fallback,
            "source": "original_bank_env_checkpoint",
        }

    manifest = {
        "case_id": job["case_id"],
        "mode": "original_bank_checkpoint",
        "a4_action_json": job["a4_action_json"],
        "candidate_chunks": chunks_to_cache,
        "cached_chunks": cached,
    }
    cache_paths.case_dir.mkdir(parents=True, exist_ok=True)
    save_json(cache_paths.manifest_path, manifest)
    debug_log(cfg, f"[建缓存] 完成: {cache_paths.manifest_path}")
    return manifest


def build_restore_cache_online(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    runtime: RuntimeAdapter,
    cache_paths: RestoreCachePaths,
    cfg: "GreedyConfig",
) -> dict[str, Any]:
    chunks_to_cache = candidate_start_chunks(job)
    if not chunks_to_cache:
        raise ValueError("No candidate chunks in recovery job")

    max_chunk = max(chunks_to_cache)
    chunks_to_cache_set = set(int(x) for x in chunks_to_cache)

    cache_paths.env_states_dir.mkdir(parents=True, exist_ok=True)

    initial_state = get_initial_state_from_bank(bank_record)
    runtime.reset_to_initial_state(initial_state)

    prefix_chunks: list[dict[str, Any]] = []
    cached: dict[str, Any] = {}

    task_description = job.get("instruction", "")

    debug_log(cfg, f"[建缓存] 在线 A4 从 chunk 0 跑到候选最大起点 chunk {max_chunk}")

    for chunk_idx in range(0, max_chunk + 1):
        # Save checkpoint at chunk start if it is a candidate or chunk 0.
        if chunk_idx in chunks_to_cache_set or chunk_idx == 0:
            state_path = cache_paths.env_states_dir / checkpoint_name(chunk_idx)
            save_report = runtime.save_env_checkpoint(state_path)
            bank_ch = get_bank_chunk_optional(bank_record, chunk_idx) or {}
            if bank_ch:
                noise_seed, used_fallback = get_noise_seed_for_chunk(
                    bank_record=bank_record,
                    chunk_idx=chunk_idx,
                    job=job,
                    cfg=cfg,
                )
            else:
                noise_seed, used_fallback = None, False

            cached[str(chunk_idx)] = {
                "chunk_idx": int(chunk_idx),
                "step_start": int(bank_ch.get("step_start", -1)),
                "state_path": str(state_path),
                "noise_seed": noise_seed,
                "noise_seed_fallback": used_fallback,
                "save_report": save_report,
                "source": "online_a4_prefix_cache",
            }
            debug_log(cfg, f"[建缓存] 保存 chunk {chunk_idx} 起点状态: {state_path}")

        # If max candidate is chunk K, we only need state at K start.
        if chunk_idx == max_chunk:
            break

        bank_ch = find_bank_chunk(bank_record, chunk_idx)
        noise_seed, used_fallback = get_noise_seed_for_chunk(
            bank_record=bank_record,
            chunk_idx=chunk_idx,
            job=job,
            cfg=cfg,
        )

        obs = runtime.get_current_obs()
        privileged_start = runtime.get_privileged_state()

        action = runtime.infer_action_chunk(
            precision="w4a4",
            obs=obs,
            task_description=task_description,
            noise_seed=noise_seed,
            chunk_idx=chunk_idx,
        )
        step_info = runtime.step_action_chunk(action)
        privileged_end = runtime.get_privileged_state()

        prefix_chunks.append({
            "chunk_idx": int(chunk_idx),
            "precision": "w4a4",
            "noise_seed": int(noise_seed),
            "step_start": bank_ch.get("step_start"),
            "step_end": bank_ch.get("step_end"),
            "action": to_jsonable(action),
            "step_info": to_jsonable(step_info),
            "privileged_start": to_jsonable(privileged_start),
            "privileged_end": to_jsonable(privileged_end),
        })

    prefix_record = {
        "case_id": job["case_id"],
        "initial_state": to_jsonable(initial_state),
        "max_cached_start_chunk": int(max_chunk),
        "chunks": prefix_chunks,
    }
    save_json(cache_paths.prefix_chunks_path, prefix_record)

    manifest = {
        "case_id": job["case_id"],
        "mode": "online_a4_prefix_cache",
        "a4_action_json": job["a4_action_json"],
        "candidate_chunks": chunks_to_cache,
        "cached_chunks": cached,
        "prefix_chunks_path": str(cache_paths.prefix_chunks_path),
    }
    save_json(cache_paths.manifest_path, manifest)
    return manifest


def build_or_load_restore_cache(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    runtime: RuntimeAdapter,
    cfg: "GreedyConfig",
) -> dict[str, Any]:
    paths = restore_cache_paths(cfg.restore_cache_root, job["case_id"])
    chunks_to_cache = candidate_start_chunks(job)

    # Hard rule: if original A4 bank already has valid checkpoints for all
    # candidate chunks, use them directly.  Do not online-rebuild cache even when
    # --rebuild-restore-cache is passed, because original checkpoints best
    # preserve the original A4 chunk-start states.
    debug_log(cfg, f"[恢复缓存] case={job.get('case_id')} 候选起点={chunks_to_cache}")

    if cfg.use_original_bank_checkpoints and has_original_checkpoints(bank_record, chunks_to_cache):
        debug_log(cfg, "[恢复缓存] 原始 A4 bank 里已有 checkpoint，直接使用")
        return build_manifest_from_original_bank(
            job=job,
            bank_record=bank_record,
            chunks_to_cache=chunks_to_cache,
            cache_paths=paths,
            cfg=cfg,
        )

    if paths.manifest_path.exists() and not cfg.rebuild_restore_cache:
        debug_log(cfg, f"[恢复缓存] 读取已有 cache: {paths.manifest_path}")
        return load_json(paths.manifest_path)

    debug_log(cfg, f"[恢复缓存] 没有可用 checkpoint，开始在线跑 A4 prefix 建 cache: {paths.case_dir}")
    return build_restore_cache_online(
        job=job,
        bank_record=bank_record,
        runtime=runtime,
        cache_paths=paths,
        cfg=cfg,
    )


def restore_env_to_chunk_start_from_cache(
    *,
    runtime: RuntimeAdapter,
    bank_record: dict[str, Any],
    manifest: dict[str, Any],
    chunk_idx: int,
) -> RestoredChunkStart:
    cached = (manifest.get("cached_chunks") or {}).get(str(int(chunk_idx)))
    if not isinstance(cached, dict):
        raise KeyError(f"chunk {chunk_idx} not found in restore cache manifest")

    checkpoint_path = cached.get("state_path")
    if not checkpoint_path:
        raise ValueError(f"cache chunk {chunk_idx} has no state_path")
    checkpoint_path = str(checkpoint_path)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    initial_state = get_initial_state_from_bank(bank_record)

    # Important: construct correct scene/wrappers before checkpoint restore.
    runtime.reset_to_initial_state(initial_state)
    restore_report = runtime.load_env_checkpoint(checkpoint_path)
    obs = runtime.get_current_obs()

    bank_chunk = find_bank_chunk(bank_record, chunk_idx)
    step_start = int(bank_chunk.get("step_start", cached.get("step_start", -1)))
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


# =============================================================================
# Target matcher
# =============================================================================

def scalar_or_none(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def float_list_or_none(x: Any) -> Optional[list[float]]:
    if not isinstance(x, list):
        return None
    out = []
    for v in x:
        sv = scalar_or_none(v)
        if sv is None:
            return None
        out.append(sv)
    return out


def l2_or_none(a: Any, b: Any) -> Optional[float]:
    av = float_list_or_none(a)
    bv = float_list_or_none(b)
    if av is None or bv is None or len(av) != len(bv):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv)))


def obj_pos(raw_state: dict[str, Any], obj: str) -> Optional[list[float]]:
    return ((raw_state.get("objects") or {}).get(obj) or {}).get("pos")


def obj_z(raw_state: dict[str, Any], obj: str) -> Optional[float]:
    o = ((raw_state.get("objects") or {}).get(obj) or {})
    if "z" in o:
        return scalar_or_none(o.get("z"))
    p = float_list_or_none(o.get("pos"))
    return p[2] if p and len(p) >= 3 else None


def joint_raw_value(raw_state: dict[str, Any], j: str) -> Any:
    return (raw_state.get("joints") or {}).get(j)


def is_free_joint_value(x: Any) -> bool:
    xs = float_list_or_none(x)
    return bool(xs is not None and len(xs) >= 7)


@dataclass
class TargetSpec:
    objects: list[str]
    joints: list[str]
    start_raw_state: dict[str, Any]
    target_raw_state: dict[str, Any]


def build_target_spec(
    *,
    start_raw_state: dict[str, Any],
    target_raw_state: dict[str, Any],
    obj_delta_threshold: float,
    z_delta_threshold: float,
    joint_delta_threshold: float,
    max_objects: int,
) -> TargetSpec:
    start_objects = start_raw_state.get("objects") or {}
    target_objects = target_raw_state.get("objects") or {}
    rows = []
    for obj in sorted(set(start_objects) & set(target_objects)):
        dp = l2_or_none(obj_pos(start_raw_state, obj), obj_pos(target_raw_state, obj))
        z0 = obj_z(start_raw_state, obj)
        z1 = obj_z(target_raw_state, obj)
        dz = None if z0 is None or z1 is None else abs(z1 - z0)
        score = max(dp or 0.0, dz or 0.0)
        if (dp is not None and dp >= obj_delta_threshold) or (dz is not None and dz >= z_delta_threshold):
            rows.append((score, obj))
    rows.sort(reverse=True)
    objects = [obj for _, obj in rows[:max_objects]]

    start_joints = start_raw_state.get("joints") or {}
    target_joints = target_raw_state.get("joints") or {}
    joints = []
    for j in sorted(set(start_joints) & set(target_joints)):
        # free joints duplicate object pose; skip by default.
        if is_free_joint_value(start_joints.get(j)) or is_free_joint_value(target_joints.get(j)):
            continue
        d = l2_or_none(start_joints.get(j), target_joints.get(j))
        if d is None:
            s0 = scalar_or_none(start_joints.get(j))
            s1 = scalar_or_none(target_joints.get(j))
            d = None if s0 is None or s1 is None else abs(s1 - s0)
        if d is not None and d >= joint_delta_threshold:
            joints.append(j)

    return TargetSpec(
        objects=objects,
        joints=joints,
        start_raw_state=start_raw_state,
        target_raw_state=target_raw_state,
    )


def target_distance(
    raw_state: dict[str, Any],
    spec: TargetSpec,
    obj_tol: float,
    z_tol: float,
    joint_tol: float,
) -> dict[str, Any]:
    failures = []
    scores = []

    for obj in spec.objects:
        dp = l2_or_none(obj_pos(raw_state, obj), obj_pos(spec.target_raw_state, obj))
        z0 = obj_z(raw_state, obj)
        z1 = obj_z(spec.target_raw_state, obj)
        dz = None if z0 is None or z1 is None else abs(z0 - z1)

        if dp is None or dp > obj_tol:
            failures.append(f"objects.{obj}.pos")
        if dz is None or dz > z_tol:
            failures.append(f"objects.{obj}.z")

        if dp is not None:
            scores.append(dp / max(obj_tol, 1e-12))
        if dz is not None:
            scores.append(dz / max(z_tol, 1e-12))

    for j in spec.joints:
        cur = joint_raw_value(raw_state, j)
        tgt = joint_raw_value(spec.target_raw_state, j)
        d = l2_or_none(cur, tgt)
        if d is None:
            c = scalar_or_none(cur)
            t = scalar_or_none(tgt)
            d = None if c is None or t is None else abs(c - t)

        if d is None or d > joint_tol:
            failures.append(f"joints.{j}")
        if d is not None:
            scores.append(d / max(joint_tol, 1e-12))

    score = None if not scores else sum(scores) / len(scores)
    matched = bool((spec.objects or spec.joints) and not failures and score is not None and score <= 1.0)
    return {
        "matched": matched,
        "score": score,
        "failures": failures,
        "objects": spec.objects,
        "joints": spec.joints,
    }


# =============================================================================
# Rollout / Greedy algorithm
# =============================================================================

@dataclass
class GreedyConfig:
    restore_cache_root: str
    output_root: str

    use_original_bank_checkpoints: bool = True
    rebuild_restore_cache: bool = False

    max_episode_chunks: int = 140
    max_stage_chunks: int = 80

    obj_delta_threshold: float = 0.05
    z_delta_threshold: float = 0.03
    joint_delta_threshold: float = 0.03
    target_obj_tol: float = 0.08
    target_z_tol: float = 0.04
    target_joint_tol: float = 0.05
    target_max_objects: int = 4

    no_improve_patience: int = 12
    min_chunks_before_fast_fail: int = 6
    enable_fast_fail: bool = False

    trial_cache_enabled: bool = True

    allow_noise_seed_fallback: bool = False
    fallback_noise_base_seed: int = 0

    debug_print: bool = True
    debug_noise_extend: bool = False

    # "fallback": preferred_chunk is the main start.  Stop the case once it
    # succeeds.  candidate_chunks are only fallback rollback points.
    # "exhaustive": old behavior, try all candidate starts and keep best.
    candidate_mode: str = "fallback"


@dataclass
class RolloutResult:
    mode: str
    start_chunk: int
    stage_success: bool
    final_success: bool
    stage_reached_chunk: Optional[int]
    final_chunk: Optional[int]
    chunk_schedule: list[dict[str, Any]]
    cost: dict[str, Any]
    target_best_score: Optional[float]
    target_last_score: Optional[float]
    target_best_report: Optional[dict[str, Any]]
    target_last_report: Optional[dict[str, Any]]


def compute_cost(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"w4a4": 0, "w4a8": 0, "w4a16": 0}
    total = 0
    for row in schedule:
        p = str(row["precision"])
        counts[p] = counts.get(p, 0) + 1
        total += PRECISION_COST[p]
    return {
        "a4": PRECISION_COST["w4a4"],
        "a8": PRECISION_COST["w4a8"],
        "a16": PRECISION_COST["w4a16"],
        "total": int(total),
        "num_a4": int(counts.get("w4a4", 0)),
        "num_a8": int(counts.get("w4a8", 0)),
        "num_a16": int(counts.get("w4a16", 0)),
    }


def schedule_precision(accepted_prefix: dict[int, str], suffix_precision: str, chunk_idx: int) -> str:
    return accepted_prefix.get(int(chunk_idx), suffix_precision)


def trial_key(start_chunk: int, accepted_prefix: dict[int, str], suffix_precision: str, mode: str) -> str:
    payload = {
        "mode": mode,
        "start_chunk": int(start_chunk),
        "suffix_precision": suffix_precision,
        "accepted_prefix": sorted((int(k), v) for k, v in accepted_prefix.items()),
    }
    return stable_hash(payload)


def rollout(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    manifest: dict[str, Any],
    runtime: RuntimeAdapter,
    cfg: GreedyConfig,
    start_chunk: int,
    accepted_prefix: dict[int, str],
    suffix_precision: str,
    mode: str,  # "target" or "full"
    target_spec: Optional[TargetSpec] = None,
) -> RolloutResult:
    debug_log(
        cfg,
        f"[Rollout开始] 模式={mode} start={start_chunk} 默认后缀={suffix_precision} "
        f"已固定chunk数={len(accepted_prefix)}",
    )
    restored = restore_env_to_chunk_start_from_cache(
        runtime=runtime,
        bank_record=bank_record,
        manifest=manifest,
        chunk_idx=start_chunk,
    )

    task_description = job.get("instruction", "")
    chunk_schedule: list[dict[str, Any]] = []

    stage_success = False
    final_success = False
    stage_reached_chunk: Optional[int] = None
    final_chunk: Optional[int] = None

    best_score: Optional[float] = None
    last_score: Optional[float] = None
    best_report: Optional[dict[str, Any]] = None
    last_report: Optional[dict[str, Any]] = None
    no_improve = 0

    max_chunks = cfg.max_stage_chunks if mode == "target" else cfg.max_episode_chunks

    # Also cap by remaining env steps.  Example for LIBERO-10:
    #   max_steps=520, num_steps_wait=10, replan_steps=5
    #   chunk 0 starts at step 10
    #   valid full-episode chunks are 0..103, i.e. 104 chunks total.
    # If we restore at start_chunk=48, the remaining full-episode budget is
    #   chunks 48..103 inclusive, not another 104 chunks.
    cur_step = runtime.current_step()
    max_env_steps = getattr(runtime, "max_steps", None)
    num_steps_wait = int(getattr(runtime, "num_steps_wait", 0) or 0)
    replan_steps = int(getattr(runtime, "replan_steps", 5) or 5)
    if cur_step is not None and max_env_steps is not None:
        end_step_exclusive = int(max_env_steps) + int(num_steps_wait)
        remaining_steps = max(0, end_step_exclusive - int(cur_step))
        max_chunks_by_step = int(math.ceil(remaining_steps / max(replan_steps, 1)))
        old_max_chunks = int(max_chunks)
        max_chunks = min(int(max_chunks), max_chunks_by_step)
        debug_log(
            cfg,
            f"[Rollout预算] 当前step={cur_step}，episode结束step={end_step_exclusive}，"
            f"剩余steps={remaining_steps}，最多还能跑chunk数={max_chunks}",
        )

    obs = restored.obs

    for offset in range(max_chunks):
        chunk_idx = int(start_chunk) + offset

        precision = schedule_precision(accepted_prefix, suffix_precision, chunk_idx)
        if precision not in PRECISION_COST:
            raise ValueError(f"unknown precision: {precision}")

        noise_seed, used_fallback = get_noise_seed_for_chunk(
            bank_record=bank_record,
            chunk_idx=chunk_idx,
            job=job,
            cfg=cfg,
        )

        action = runtime.infer_action_chunk(
            precision=precision,
            obs=obs,
            task_description=task_description,
            noise_seed=int(noise_seed),
            chunk_idx=chunk_idx,
        )
        runtime.step_action_chunk(action)

        raw_state = runtime.get_privileged_state()

        chunk_schedule.append({
            "chunk_idx": int(chunk_idx),
            "precision": precision,
            "noise_seed": int(noise_seed),
        })

        if target_spec is not None:
            td = target_distance(
                raw_state,
                target_spec,
                obj_tol=cfg.target_obj_tol,
                z_tol=cfg.target_z_tol,
                joint_tol=cfg.target_joint_tol,
            )
            last_report = td
            last_score = td.get("score")
            if last_score is not None and (best_score is None or last_score < best_score):
                best_score = last_score
                best_report = td
                no_improve = 0
            else:
                no_improve += 1

            if td.get("matched", False) and not stage_success:
                stage_success = True
                stage_reached_chunk = int(chunk_idx)
                if mode == "target":
                    break

            # Fast fail only for stage-local trials.
            if (
                cfg.enable_fast_fail
                and mode == "target"
                and offset >= cfg.min_chunks_before_fast_fail
                and no_improve >= cfg.no_improve_patience
                and not stage_success
            ):
                debug_log(
                    cfg,
                    f"[阶段提前停止] target score 连续没有改善；chunk={chunk_idx} "
                    f"best_score={best_score} last_score={last_score}",
                )
                break

        if runtime.env_success():
            final_success = True
            final_chunk = int(chunk_idx)
            if mode == "full":
                # In full mode, final success is enough to stop.
                break

        if runtime.env_timeout():
            debug_log(
                cfg,
                f"[Rollout停止] 达到episode步数上限：模式={mode} chunk={chunk_idx} step={runtime.current_step()}",
            )
            break

        # refresh obs after step
        obs = runtime.get_current_obs()

    result_cost = compute_cost(chunk_schedule)
    debug_log(
        cfg,
        f"[Rollout结束] 模式={mode} start={start_chunk} 默认后缀={suffix_precision} "
        f"阶段成功={stage_success} 阶段chunk={stage_reached_chunk} "
        f"最终成功={final_success} 最终chunk={final_chunk} cost={result_cost}",
    )

    return RolloutResult(
        mode=mode,
        start_chunk=int(start_chunk),
        stage_success=stage_success,
        final_success=final_success,
        stage_reached_chunk=stage_reached_chunk,
        final_chunk=final_chunk,
        chunk_schedule=chunk_schedule,
        cost=result_cost,
        target_best_score=best_score,
        target_last_score=last_score,
        target_best_report=best_report,
        target_last_report=last_report,
    )


@dataclass
class Baseline:
    start_chunk: int
    accepted_prefix: dict[int, str]
    suffix_precision: str
    result: RolloutResult
    target_spec: TargetSpec


def find_safe_baseline_by_env_success(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    manifest: dict[str, Any],
    runtime: RuntimeAdapter,
    cfg: GreedyConfig,
    start_chunk: int,
) -> Optional[Baseline]:
    # Build target spec from restored start raw_state and target raw_state.
    restore_env_to_chunk_start_from_cache(
        runtime=runtime,
        bank_record=bank_record,
        manifest=manifest,
        chunk_idx=start_chunk,
    )
    start_raw_state = runtime.get_privileged_state()
    target_spec = build_target_spec(
        start_raw_state=start_raw_state,
        target_raw_state=job["target_stage_state"]["raw_state"],
        obj_delta_threshold=cfg.obj_delta_threshold,
        z_delta_threshold=cfg.z_delta_threshold,
        joint_delta_threshold=cfg.joint_delta_threshold,
        max_objects=cfg.target_max_objects,
    )

    debug_log(
        cfg,
        f"[基线验证] start_chunk={start_chunk}；阶段目标对象={target_spec.objects}；目标关节={target_spec.joints}",
    )

    # Baseline is a rescue certificate, so it should NOT use A4.
    # A4 is only introduced later in greedy lowering.
    for suffix_precision in ("w4a8", "w4a16"):
        debug_log(cfg, f"[基线验证] 从 chunk {start_chunk} 开始，后面全用 {suffix_precision}，跑完整任务")
        result = rollout(
            job=job,
            bank_record=bank_record,
            manifest=manifest,
            runtime=runtime,
            cfg=cfg,
            start_chunk=start_chunk,
            accepted_prefix={},
            suffix_precision=suffix_precision,
            mode="full",
            target_spec=target_spec,
        )
        debug_log(
            cfg,
            f"[基线结果] start_chunk={start_chunk} suffix={suffix_precision} "
            f"阶段成功={result.stage_success} 最终成功={result.final_success} "
            f"阶段到达chunk={result.stage_reached_chunk} 最终chunk={result.final_chunk} cost={result.cost}",
        )
        if result.final_success:
            debug_log(cfg, f"[基线成功] 选择 safe_suffix={suffix_precision}，接下来在 greedy 里尝试插入 A4 降成本")
            return Baseline(
                start_chunk=int(start_chunk),
                accepted_prefix={},
                suffix_precision=suffix_precision,
                result=result,
                target_spec=target_spec,
            )

    debug_log(cfg, f"[基线失败] start_chunk={start_chunk}：A8 和 A16 都不能完整成功")
    return None


def rollout_to_target_cached(
    *,
    cache: dict[str, RolloutResult],
    job: dict[str, Any],
    bank_record: dict[str, Any],
    manifest: dict[str, Any],
    runtime: RuntimeAdapter,
    cfg: GreedyConfig,
    start_chunk: int,
    accepted_prefix: dict[int, str],
    suffix_precision: str,
    target_spec: TargetSpec,
) -> RolloutResult:
    key = trial_key(start_chunk, accepted_prefix, suffix_precision, mode="target")
    if cfg.trial_cache_enabled and key in cache:
        return cache[key]

    result = rollout(
        job=job,
        bank_record=bank_record,
        manifest=manifest,
        runtime=runtime,
        cfg=cfg,
        start_chunk=start_chunk,
        accepted_prefix=accepted_prefix,
        suffix_precision=suffix_precision,
        mode="target",
        target_spec=target_spec,
    )
    if cfg.trial_cache_enabled:
        cache[key] = result
    return result


@dataclass
class StageSchedule:
    start_chunk: int
    accepted_prefix: dict[int, str]
    suffix_precision: str
    stage_result: RolloutResult
    target_spec: TargetSpec
    greedy_events: list[dict[str, Any]]


def safe_suffix_greedy_to_target(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    manifest: dict[str, Any],
    runtime: RuntimeAdapter,
    cfg: GreedyConfig,
    baseline: Baseline,
) -> StageSchedule:
    accepted_prefix: dict[int, str] = dict(baseline.accepted_prefix)
    suffix_precision = baseline.suffix_precision
    start_chunk = int(baseline.start_chunk)
    target_spec = baseline.target_spec
    cache: dict[str, RolloutResult] = {}
    greedy_events: list[dict[str, Any]] = []

    # Greedy chunk order comes from the successful baseline rollout up to target.
    if baseline.result.stage_reached_chunk is not None:
        end_chunk = int(baseline.result.stage_reached_chunk)
    elif baseline.result.final_chunk is not None:
        # If target matcher missed but full episode succeeded, use final_chunk as fallback.
        end_chunk = int(baseline.result.final_chunk)
    else:
        end_chunk = start_chunk + cfg.max_stage_chunks - 1

    chunk_order = list(range(start_chunk, end_chunk + 1))

    current_result: Optional[RolloutResult] = None

    debug_log(
        cfg,
        f"[贪心开始] start={start_chunk}，当前安全后缀={suffix_precision}；"
        f"只优化到阶段目标chunk范围：{chunk_order[0] if chunk_order else None}..{chunk_order[-1] if chunk_order else None}；"
        f"提前失败停止={'开' if cfg.enable_fast_fail else '关'}",
    )

    for chunk_idx in chunk_order:
        debug_log(cfg, f"[贪心] 正在决定 chunk {chunk_idx}，当前安全后缀={suffix_precision}")
        if suffix_precision == "w4a8":
            # Try current chunk as A4, suffix remains A8.
            trial_prefix = dict(accepted_prefix)
            trial_prefix[int(chunk_idx)] = "w4a4"
            result = rollout_to_target_cached(
                cache=cache,
                job=job,
                bank_record=bank_record,
                manifest=manifest,
                runtime=runtime,
                cfg=cfg,
                start_chunk=start_chunk,
                accepted_prefix=trial_prefix,
                suffix_precision="w4a8",
                target_spec=target_spec,
            )
            debug_log(
                cfg,
                f"[贪心尝试] chunk {chunk_idx}: 改成 A4，后面接 A8；"
                f"阶段成功={result.stage_success} score={result.target_best_score}",
            )
            if result.stage_success:
                debug_log(cfg, f"[贪心接受] chunk {chunk_idx} 使用 A4")
                accepted_prefix = trial_prefix
                current_result = result
                save_live_greedy_schedule(
                    job=job,
                    bank_record=bank_record,
                    cfg=cfg,
                    start_chunk=start_chunk,
                    suffix_precision=suffix_precision,
                    accepted_prefix=accepted_prefix,
                    last_decision={
                        "chunk_idx": int(chunk_idx),
                        "decision": "accept",
                        "precision": "w4a4",
                        "suffix_precision": suffix_precision,
                        "stage_success": True,
                        "target_best_score": result.target_best_score,
                    },
                )
            else:
                debug_log(cfg, f"[贪心保持] chunk {chunk_idx} 保持 A8")
                accepted_prefix[int(chunk_idx)] = "w4a8"
                save_live_greedy_schedule(
                    job=job,
                    bank_record=bank_record,
                    cfg=cfg,
                    start_chunk=start_chunk,
                    suffix_precision=suffix_precision,
                    accepted_prefix=accepted_prefix,
                    last_decision={
                        "chunk_idx": int(chunk_idx),
                        "decision": "keep",
                        "precision": "w4a8",
                        "suffix_precision": suffix_precision,
                        "stage_success": False,
                        "target_best_score": result.target_best_score,
                    },
                )

        elif suffix_precision == "w4a16":
            # 1) current A4 + suffix A8
            trial_prefix = dict(accepted_prefix)
            trial_prefix[int(chunk_idx)] = "w4a4"
            result = rollout_to_target_cached(
                cache=cache,
                job=job,
                bank_record=bank_record,
                manifest=manifest,
                runtime=runtime,
                cfg=cfg,
                start_chunk=start_chunk,
                accepted_prefix=trial_prefix,
                suffix_precision="w4a8",
                target_spec=target_spec,
            )
            debug_log(
                cfg,
                f"[贪心尝试] chunk {chunk_idx}: 改成 A4，后面接 A8；"
                f"阶段成功={result.stage_success} score={result.target_best_score}",
            )
            if result.stage_success:
                debug_log(cfg, f"[贪心接受] chunk {chunk_idx} 使用 A4，并且安全后缀从 A16 降到 A8")
                greedy_events.append({
                    "event": "suffix_downgrade",
                    "chunk_idx": int(chunk_idx),
                    "from": "w4a16",
                    "to": "w4a8",
                    "target_best_score": result.target_best_score,
                    "target_last_report": result.target_last_report,
                })
                accepted_prefix = trial_prefix
                suffix_precision = "w4a8"
                current_result = result
                save_live_greedy_schedule(
                    job=job,
                    bank_record=bank_record,
                    cfg=cfg,
                    start_chunk=start_chunk,
                    suffix_precision=suffix_precision,
                    accepted_prefix=accepted_prefix,
                    last_decision={
                        "chunk_idx": int(chunk_idx),
                        "decision": "accept",
                        "precision": "w4a4",
                        "suffix_precision": suffix_precision,
                        "suffix_downgrade": "w4a16->w4a8",
                        "stage_success": True,
                        "target_best_score": result.target_best_score,
                    },
                )
                continue

            # 2) current A4 + suffix A16
            trial_prefix = dict(accepted_prefix)
            trial_prefix[int(chunk_idx)] = "w4a4"
            result = rollout_to_target_cached(
                cache=cache,
                job=job,
                bank_record=bank_record,
                manifest=manifest,
                runtime=runtime,
                cfg=cfg,
                start_chunk=start_chunk,
                accepted_prefix=trial_prefix,
                suffix_precision="w4a16",
                target_spec=target_spec,
            )
            debug_log(
                cfg,
                f"[贪心尝试] chunk {chunk_idx}: 改成 A4，后面接 A16；"
                f"阶段成功={result.stage_success} score={result.target_best_score}",
            )
            if result.stage_success:
                debug_log(cfg, f"[贪心接受] chunk {chunk_idx} 使用 A4，后缀仍保持 A16")
                accepted_prefix = trial_prefix
                current_result = result
                save_live_greedy_schedule(
                    job=job,
                    bank_record=bank_record,
                    cfg=cfg,
                    start_chunk=start_chunk,
                    suffix_precision=suffix_precision,
                    accepted_prefix=accepted_prefix,
                    last_decision={
                        "chunk_idx": int(chunk_idx),
                        "decision": "accept",
                        "precision": "w4a4",
                        "suffix_precision": suffix_precision,
                        "stage_success": True,
                        "target_best_score": result.target_best_score,
                    },
                )
                continue

            # 3) current A8 + suffix A16
            trial_prefix = dict(accepted_prefix)
            trial_prefix[int(chunk_idx)] = "w4a8"
            result = rollout_to_target_cached(
                cache=cache,
                job=job,
                bank_record=bank_record,
                manifest=manifest,
                runtime=runtime,
                cfg=cfg,
                start_chunk=start_chunk,
                accepted_prefix=trial_prefix,
                suffix_precision="w4a16",
                target_spec=target_spec,
            )
            debug_log(
                cfg,
                f"[贪心尝试] chunk {chunk_idx}: 改成 A8，后面接 A16；"
                f"阶段成功={result.stage_success} score={result.target_best_score}",
            )
            if result.stage_success:
                debug_log(cfg, f"[贪心接受] chunk {chunk_idx} 使用 A8，后缀仍保持 A16")
                accepted_prefix = trial_prefix
                current_result = result
                save_live_greedy_schedule(
                    job=job,
                    bank_record=bank_record,
                    cfg=cfg,
                    start_chunk=start_chunk,
                    suffix_precision=suffix_precision,
                    accepted_prefix=accepted_prefix,
                    last_decision={
                        "chunk_idx": int(chunk_idx),
                        "decision": "accept",
                        "precision": "w4a8",
                        "suffix_precision": suffix_precision,
                        "stage_success": True,
                        "target_best_score": result.target_best_score,
                    },
                )
                continue

            # 4) keep current as A16, no rollout needed.
            debug_log(cfg, f"[贪心保持] chunk {chunk_idx} 保持 A16")
            accepted_prefix[int(chunk_idx)] = "w4a16"
            save_live_greedy_schedule(
                job=job,
                bank_record=bank_record,
                cfg=cfg,
                start_chunk=start_chunk,
                suffix_precision=suffix_precision,
                accepted_prefix=accepted_prefix,
                last_decision={
                    "chunk_idx": int(chunk_idx),
                    "decision": "keep",
                    "precision": "w4a16",
                    "suffix_precision": suffix_precision,
                    "stage_success": False,
                    "target_best_score": result.target_best_score,
                },
            )

        else:
            raise ValueError(f"unsupported suffix_precision: {suffix_precision}")

    # Produce final optimized stage rollout. This fixes chunk_schedule up to target.
    debug_log(
        cfg,
        f"[贪心阶段复跑] 用已确定的局部 schedule 再跑到阶段目标；"
        f"start={start_chunk} 后缀={suffix_precision} 已固定chunk数={len(accepted_prefix)}",
    )
    final_stage_result = rollout(
        job=job,
        bank_record=bank_record,
        manifest=manifest,
        runtime=runtime,
        cfg=cfg,
        start_chunk=start_chunk,
        accepted_prefix=accepted_prefix,
        suffix_precision=suffix_precision,
        mode="target",
        target_spec=target_spec,
    )

    return StageSchedule(
        start_chunk=start_chunk,
        accepted_prefix=accepted_prefix,
        suffix_precision=suffix_precision,
        stage_result=final_stage_result,
        target_spec=target_spec,
        greedy_events=greedy_events,
    )


def final_validate_by_env_success(
    *,
    job: dict[str, Any],
    bank_record: dict[str, Any],
    manifest: dict[str, Any],
    runtime: RuntimeAdapter,
    cfg: GreedyConfig,
    stage_schedule: StageSchedule,
) -> RolloutResult:
    # Use optimized stage chunk_schedule as fixed prefix.  After target reached,
    # default suffix is A4 for the rest of episode.
    debug_log(
        cfg,
        f"[最终验证] 用贪心得到的 schedule 从 chunk {stage_schedule.start_chunk} 跑完整任务；"
        f"阶段后默认使用 A4",
    )
    fixed_prefix: dict[int, str] = {}
    for row in stage_schedule.stage_result.chunk_schedule:
        fixed_prefix[int(row["chunk_idx"])] = str(row["precision"])

    return rollout(
        job=job,
        bank_record=bank_record,
        manifest=manifest,
        runtime=runtime,
        cfg=cfg,
        start_chunk=stage_schedule.start_chunk,
        accepted_prefix=fixed_prefix,
        suffix_precision="w4a4",
        mode="full",
        target_spec=stage_schedule.target_spec,
    )


def result_better(a: RolloutResult, b: Optional[RolloutResult]) -> bool:
    if b is None:
        return True

    # final_success first
    if a.final_success != b.final_success:
        return bool(a.final_success and not b.final_success)

    # stage_success second
    if a.stage_success != b.stage_success:
        return bool(a.stage_success and not b.stage_success)

    ca, cb = a.cost, b.cost
    if ca["total"] != cb["total"]:
        return ca["total"] < cb["total"]
    if ca["num_a16"] != cb["num_a16"]:
        return ca["num_a16"] < cb["num_a16"]
    if ca["num_a8"] != cb["num_a8"]:
        return ca["num_a8"] < cb["num_a8"]

    # Tie: later start chunk = shorter rescue prefix.
    return a.start_chunk > b.start_chunk


def make_recovery_schedule_record(
    *,
    job: dict[str, Any],
    result: RolloutResult,
) -> dict[str, Any]:
    return {
        "case_id": job.get("case_id"),
        "task_id": job.get("task_id"),
        "episode_idx": job.get("episode_idx"),
        "instruction": job.get("instruction"),
        "chosen_start_chunk": int(result.start_chunk),
        "prefix_end_chunk": int(result.start_chunk) - 1,
        "stage_success": bool(result.stage_success),
        "final_success": bool(result.final_success),
        "cost": result.cost,
        "chunk_schedule": result.chunk_schedule,
    }


def run_greedy_recovery_job(job_path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    job = load_json(job_path)
    bank_record = load_json(job["a4_action_json"])

    cfg = GreedyConfig(
        restore_cache_root=args.restore_cache_root,
        output_root=args.out,
        use_original_bank_checkpoints=not args.no_original_bank_checkpoints,
        rebuild_restore_cache=args.rebuild_restore_cache,
        max_episode_chunks=args.max_episode_chunks,
        max_stage_chunks=args.max_stage_chunks,
        obj_delta_threshold=args.obj_delta_threshold,
        z_delta_threshold=args.z_delta_threshold,
        joint_delta_threshold=args.joint_delta_threshold,
        target_obj_tol=args.target_obj_tol,
        target_z_tol=args.target_z_tol,
        target_joint_tol=args.target_joint_tol,
        target_max_objects=args.target_max_objects,
        no_improve_patience=args.no_improve_patience,
        min_chunks_before_fast_fail=args.min_chunks_before_fast_fail,
        enable_fast_fail=args.enable_fast_fail,
        trial_cache_enabled=not args.no_trial_cache,
        allow_noise_seed_fallback=args.allow_noise_seed_fallback,
        fallback_noise_base_seed=args.fallback_noise_base_seed,
        debug_print=args.debug_print,
        debug_noise_extend=args.debug_noise_extend,
        candidate_mode=args.candidate_mode,
    )

    runtime = build_runtime(job, args)

    debug_log(cfg, f"[任务开始] case={job.get('case_id')} job={job_path}")
    debug_log(cfg, f"[端口] A4={getattr(args, 'port_a4', None)} A8={getattr(args, 'port_a8', None)} A16={getattr(args, 'port_a16', None)}")

    manifest = build_or_load_restore_cache(
        job=job,
        bank_record=bank_record,
        runtime=runtime,
        cfg=cfg,
    )

    best_result: Optional[RolloutResult] = None
    trial_summaries: list[dict[str, Any]] = []

    starts = ordered_start_chunks(job, cfg)
    debug_log(
        cfg,
        f"[起点策略] candidate_mode={cfg.candidate_mode}；"
        f"先跑 preferred，只有 preferred 失败才回滚 candidate"
        if cfg.candidate_mode == "fallback"
        else f"[起点策略] candidate_mode={cfg.candidate_mode}；会遍历所有 candidate 并比较 cost",
    )

    for start_meta in starts:
        start_chunk = int(start_meta["chunk"])
        start_role = str(start_meta["role"])

        debug_log(
            cfg,
            f"[候选起点] 开始尝试 start_chunk={start_chunk} role={start_role}"
        )

        baseline = find_safe_baseline_by_env_success(
            job=job,
            bank_record=bank_record,
            manifest=manifest,
            runtime=runtime,
            cfg=cfg,
            start_chunk=int(start_chunk),
        )

        if baseline is None:
            debug_log(cfg, f"[候选起点] start_chunk={start_chunk} 失败：A8/A16 都不能完整救回任务")
            trial_summaries.append({
                "start_chunk": int(start_chunk),
                "start_role": start_role,
                "baseline_found": False,
                "fallback_reason": "baseline_failed",
            })
            continue

        stage_schedule = safe_suffix_greedy_to_target(
            job=job,
            bank_record=bank_record,
            manifest=manifest,
            runtime=runtime,
            cfg=cfg,
            baseline=baseline,
        )

        final_result = final_validate_by_env_success(
            job=job,
            bank_record=bank_record,
            manifest=manifest,
            runtime=runtime,
            cfg=cfg,
            stage_schedule=stage_schedule,
        )

        trial_summaries.append({
            "start_chunk": int(start_chunk),
            "start_role": start_role,
            "baseline_found": True,
            "baseline_suffix_precision": baseline.suffix_precision,
            "baseline_final_success": baseline.result.final_success,
            "stage_success": stage_schedule.stage_result.stage_success,
            "stage_target_best_score": stage_schedule.stage_result.target_best_score,
            "stage_target_last_report": stage_schedule.stage_result.target_last_report,
            "greedy_events": stage_schedule.greedy_events,
            "final_success": final_result.final_success,
            "cost": final_result.cost,
        })

        debug_log(
            cfg,
            f"[候选起点完成] start_chunk={start_chunk} role={start_role} "
            f"阶段成功={final_result.stage_success} 最终成功={final_result.final_success} cost={final_result.cost}",
        )

        if final_result.final_success:
            if result_better(final_result, best_result):
                best_result = final_result
            if cfg.candidate_mode == "fallback":
                debug_log(
                    cfg,
                    f"[起点策略] start_chunk={start_chunk} 已最终成功；"
                    "fallback 模式下不再尝试其他 candidate，当前 case 结束搜索",
                )
                break
            debug_log(cfg, f"[当前最优] 更新为 start_chunk={start_chunk} cost={final_result.cost}")
            continue

        # Final validation failed.  In fallback mode, only then do we allow rollback
        # to later candidate entries.
        debug_log(
            cfg,
            f"[候选起点失败] start_chunk={start_chunk} baseline 能救，但优化后最终验证失败；"
            "如果还有 fallback candidate，将继续回滚尝试",
        )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if best_result is None:
        record = {
            "case_id": job.get("case_id"),
            "task_id": job.get("task_id"),
            "episode_idx": job.get("episode_idx"),
            "instruction": job.get("instruction"),
            "success": False,
            "reason": "no_start_chunk_found_safe_baseline",
            "trials": trial_summaries,
        }
    else:
        record = make_recovery_schedule_record(job=job, result=best_result)
        record["trials"] = trial_summaries if args.save_trial_summary else []

    schedule_path = out_root / job["case_id"] / "recovery_schedule.json"
    save_json(schedule_path, record)
    debug_log(cfg, f"[保存结果] recovery_schedule 已保存: {schedule_path}")

    if args.save_trial_summary:
        save_jsonl(out_root / job["case_id"] / "trial_summary.jsonl", trial_summaries)

    return record


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Greedy recovery runner: build restore cache, search schedule, output chunk-level schedule.")

    ap.add_argument("--job", type=str, required=True, help="minimal recovery_job.json")
    ap.add_argument("--out", type=str, required=True, help="output root for recovery_schedule.json")
    ap.add_argument("--restore-cache-root", type=str, required=True)

    ap.add_argument("--rebuild-restore-cache", action="store_true")
    ap.add_argument("--no-original-bank-checkpoints", action="store_true")

    ap.add_argument("--max-episode-chunks", type=int, default=140)
    ap.add_argument("--max-stage-chunks", type=int, default=80)

    ap.add_argument("--obj-delta-threshold", type=float, default=0.05)
    ap.add_argument("--z-delta-threshold", type=float, default=0.03)
    ap.add_argument("--joint-delta-threshold", type=float, default=0.03)

    ap.add_argument("--target-obj-tol", type=float, default=0.08)
    ap.add_argument("--target-z-tol", type=float, default=0.04)
    ap.add_argument("--target-joint-tol", type=float, default=0.05)
    ap.add_argument("--target-max-objects", type=int, default=4)

    ap.add_argument("--no-improve-patience", type=int, default=12)
    ap.add_argument("--min-chunks-before-fast-fail", type=int, default=6)
    ap.add_argument(
        "--enable-fast-fail",
        action="store_true",
        help="Enable early stop for target rollout when target score does not improve. Default is disabled for safer greedy labels.",
    )

    ap.add_argument("--no-trial-cache", action="store_true")
    ap.add_argument("--save-trial-summary", action="store_true")

    ap.add_argument("--allow-noise-seed-fallback", action="store_true")
    ap.add_argument("--fallback-noise-base-seed", type=int, default=0)
    ap.add_argument("--candidate-mode", choices=["fallback", "exhaustive"], default="fallback")
    ap.add_argument("--quiet-recovery", dest="debug_print", action="store_false")
    ap.set_defaults(debug_print=True)

    # By default do not print every inferred noise seed beyond the original A4 bank,
    # because it can spam the log. Enable only when debugging seed extension.
    ap.add_argument("--debug-noise-extend", dest="debug_noise_extend", action="store_true")
    ap.set_defaults(debug_noise_extend=False)

    # Policy server endpoints. Defaults follow the common three-server setup:
    #   8002 = w4a4, 8003 = w4a8, 8004 = w4a16
    # If your ports differ, pass them explicitly.
    ap.add_argument("--policy-host", type=str, default="127.0.0.1")
    ap.add_argument("--port-a4", type=int, default=8002)
    ap.add_argument("--port-a8", type=int, default=8003)
    ap.add_argument("--port-a16", type=int, default=8004)
    ap.add_argument("--url-a4", type=str, default=None)
    ap.add_argument("--url-a8", type=str, default=None)
    ap.add_argument("--url-a16", type=str, default=None)

    # LIBERO/OpenPI runtime args.
    ap.add_argument("--task-suite-name", type=str, default="libero_10")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--max-env-steps", type=int, default=None)
    ap.add_argument("--debug-noise-horizon", type=int, default=10)
    ap.add_argument("--debug-noise-dim", type=int, default=32)
    ap.add_argument("--no-send-debug-noise", dest="send_debug_noise", action="store_false")
    ap.set_defaults(send_debug_noise=True)
    ap.add_argument("--no-save-contacts", dest="save_contacts", action="store_false")
    ap.set_defaults(save_contacts=True)

    # Runtime-specific args can be added here in your repo.
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    record = run_greedy_recovery_job(args.job, args)
    print(json.dumps({
        "case_id": record.get("case_id"),
        "final_success": record.get("final_success", record.get("success")),
        "stage_success": record.get("stage_success"),
        "chosen_start_chunk": record.get("chosen_start_chunk"),
        "cost": record.get("cost"),
        "schedule_path": str(Path(args.out) / str(record.get("case_id")) / "recovery_schedule.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ppo_precision_model import ActorCriticPrecisionSelector
from ppo_precision_algo import PPOConfig, load_trajectory_files, ppo_update


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def load_cache_meta(cache_root: str | Path) -> dict[str, Any]:
    meta_path = Path(cache_root) / "meta.json"
    if not meta_path.exists():
        # Some earlier cache versions only have index.json. Use user-provided defaults if missing.
        return {}
    return json.loads(meta_path.read_text())


def build_model(args: argparse.Namespace) -> ActorCriticPrecisionSelector:
    meta = load_cache_meta(args.cache_root)
    hidden_dim = int(args.hidden_dim or meta.get("hidden_dim", 2048))
    state_dim = int(args.state_dim or meta.get("state_dim", 32))
    model = ActorCriticPrecisionSelector(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        selector_dim=args.selector_dim,
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
    )
    if args.init_a16_bias:
        model.init_policy_bias_to_decision(decision_id=8, boost=args.init_a16_bias_boost)
    return model


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, args: argparse.Namespace, iteration: int, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "model": model.state_dict(),
        "args": vars(args),
        "iteration": int(iteration),
        "stats": stats,
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    torch.save(obj, path)


def worker_main(args: argparse.Namespace) -> None:
    from ppo_precision_rollout_adapter import collect_rollouts

    device = torch.device(f"cuda:{args.local_gpu}" if torch.cuda.is_available() and args.local_gpu >= 0 else "cpu")
    model = build_model(args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    trajectories, stats = collect_rollouts(
        model=model,
        args=args,
        device=device,
        worker_id=args.worker_id,
        num_episodes=args.episodes_per_worker,
    )
    out = {
        "trajectories": trajectories,
        "stats": stats,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
    }
    out_path = Path(args.worker_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    stats_path = out_path.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))


def learner_main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.learner_gpu}" if torch.cuda.is_available() and args.learner_gpu >= 0 else "cpu")
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt and not args.no_resume_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt.get("iteration", 0)) + 1
    else:
        start_iter = 1

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    latest_ckpt = out_dir / "latest.pt"
    save_checkpoint(latest_ckpt, model, optimizer, args, start_iter - 1, {"init": True})

    best_success = -1.0
    best_cost = float("inf")
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]

    ppo_cfg = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        norm_adv=not args.no_norm_adv,
    )

    history_path = out_dir / "history.jsonl"
    for it in range(start_iter, args.num_iterations + 1):
        t0 = time.time()
        iter_dir = out_dir / "rollouts" / f"iter_{it:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # Save current policy for workers.
        rollout_ckpt = iter_dir / "policy.pt"
        save_checkpoint(rollout_ckpt, model, None, args, it, {"rollout": True})

        procs = []
        worker_files = []
        for wid in range(args.num_workers):
            gpu = gpu_ids[wid % len(gpu_ids)]
            out_file = iter_dir / f"worker_{wid:03d}.pt"
            worker_files.append(out_file)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--role", "worker",
                "--checkpoint", str(rollout_ckpt),
                "--worker-out", str(out_file),
                "--worker-id", str(wid),
                "--local-gpu", str(gpu),
            ] + args._passthrough_args
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "")
            procs.append(subprocess.Popen(cmd, env=env))
        failed = []
        for p in procs:
            ret = p.wait()
            if ret != 0:
                failed.append(ret)
        if failed:
            raise RuntimeError(f"{len(failed)} rollout workers failed: {failed}")

        batch = load_trajectory_files(worker_files)
        model.train()
        update_stats = ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            cfg=ppo_cfg,
            amp=args.amp,
        )

        # Summarize rollout stats.
        all_stats = []
        for wf in worker_files:
            jp = wf.with_suffix(".json")
            if jp.exists():
                all_stats.extend(json.loads(jp.read_text()))
        n_eps = max(1, len(all_stats))
        success_rate = sum(1 for s in all_stats if s.get("success")) / n_eps
        avg_cost = sum(float(s.get("total_precision_cost", 0.0)) for s in all_stats) / n_eps
        avg_steps = sum(float(s.get("total_steps", 0.0)) for s in all_stats) / n_eps
        avg_return = sum(float(s.get("episode_return", 0.0)) for s in all_stats) / n_eps
        precision_hist = [0] * 9
        for s in all_stats:
            h = s.get("precision_hist", [0] * 9)
            for i, v in enumerate(h[:9]):
                precision_hist[i] += int(v)

        summary = {
            "iter": it,
            "episodes": n_eps,
            "steps": int(batch["_num_steps"].item()),
            "success_rate": success_rate,
            "avg_cost": avg_cost,
            "avg_steps": avg_steps,
            "avg_return": avg_return,
            "precision_hist": precision_hist,
            "update": update_stats,
            "elapsed_s": time.time() - t0,
        }
        with history_path.open("a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        save_checkpoint(latest_ckpt, model, optimizer, args, it, summary)
        # Success-first checkpointing: success primary, cost secondary.
        better = False
        if success_rate > best_success + args.best_success_eps:
            better = True
        elif abs(success_rate - best_success) <= args.best_success_eps and avg_cost < best_cost:
            better = True
        if better:
            best_success = success_rate
            best_cost = avg_cost
            save_checkpoint(out_dir / "best.pt", model, optimizer, args, it, summary)
        print(
            f"[ITER {it:04d}] eps={n_eps} succ={success_rate:.3f} cost={avg_cost:.2f} "
            f"steps={avg_steps:.1f} ret={avg_return:.2f} loss={update_stats['loss']:.4f} "
            f"ent={update_stats['entropy']:.3f} hist={precision_hist} elapsed={summary['elapsed_s']:.1f}s",
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["learner", "worker"], default="learner")
    p.add_argument("--cache-root", default="/share/chengyuxuan-local/openpi/selector/selector_merged_cache_fp16_state32")
    p.add_argument("--out-dir", default="/home/chengyuxuan/openpi/experiments/selector_dataset/selector_runs/ppo_precision_success_first")
    p.add_argument("--resume", default="")
    p.add_argument("--no-resume-optimizer", action="store_true")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--worker-out", default="")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--local-gpu", type=int, default=0)
    p.add_argument("--learner-gpu", type=int, default=0)
    p.add_argument("--gpu-ids", default="0")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--episodes-per-worker", type=int, default=4)
    p.add_argument("--num-iterations", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=0)
    p.add_argument("--state-dim", type=int, default=0)
    p.add_argument("--selector-dim", type=int, default=256)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--init-a16-bias", action="store_true")
    p.add_argument("--init-a16-bias-boost", type=float, default=6.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--no-norm-adv", action="store_true")
    p.add_argument("--amp", action="store_true")
    # Reward / environment args are passed through to adapter.
    p.add_argument("--initial-vlm-bits", type=int, default=16)
    p.add_argument("--success-reward", type=float, default=1000.0)
    p.add_argument("--failure-penalty", type=float, default=1000.0)
    p.add_argument("--lambda-cost", type=float, default=1.0)
    p.add_argument("--lambda-step", type=float, default=0.1)
    p.add_argument("--progress-weight", type=float, default=0.0)
    p.add_argument("--action-cost-values", type=parse_floats, default=[1.0, 2.0, 4.0])
    p.add_argument("--vlm-cost-values", type=parse_floats, default=[1.0, 2.0, 4.0])
    p.add_argument("--action-cost-weight", type=float, default=1.0)
    p.add_argument("--vlm-cost-weight", type=float, default=1.0)
    p.add_argument("--max-precision-cost", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--max-chunks", type=int, default=200)
    p.add_argument("--task-start", type=int, default=0)
    p.add_argument("--task-end", type=int, default=9)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--deterministic-rollout", action="store_true")
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--best-success-eps", type=float, default=1e-6)
    return p


def main() -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    # Preserve all args for worker subprocesses. Remove role/checkpoint/worker specific args.
    passthrough = []
    skip_next = False
    worker_keys = {"--role", "--checkpoint", "--worker-out", "--worker-id", "--local-gpu"}
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in worker_keys:
            i += 2
            continue
        passthrough.append(a)
        i += 1
    args._passthrough_args = passthrough
    if args.role == "worker":
        worker_main(args)
    else:
        learner_main(args)


if __name__ == "__main__":
    main()

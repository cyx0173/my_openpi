#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

BITS = (4, 8, 16)
PAIR_RE = re.compile(r"^v(4|8|16)_a(4|8|16)$")
SELECTOR_RE = re.compile(r"\[SELECTOR\](.*)$")
FIELD_RE = re.compile(r"(current_vlm_a_bits|selected_pair|action_a_bits|next_vlm_a_bits)=([^\s]+)")
BIT_COST = {4: 1.0, 8: 2.0, 16: 4.0}
DEFAULT_PATTERNS = ("*.log", "*.txt", "*.out")


def pct(n: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * n / total


def parse_pair(pair: str) -> tuple[int, int]:
    m = PAIR_RE.match(pair)
    if not m:
        raise ValueError(pair)
    return int(m.group(1)), int(m.group(2))


def runtime_cost(current_vlm_bits: int, action_bits: int) -> float:
    return 3.0 * BIT_COST[current_vlm_bits] + BIT_COST[action_bits]


def find_scope_start(lines: list[str], scope: str) -> int:
    if scope == "all":
        return 0
    if scope == "last_connection":
        marks = [i for i, x in enumerate(lines) if "Connection from" in x or "connection open" in x]
        return marks[-1] if marks else 0
    if scope == "last_server_session":
        marks = [i for i, x in enumerate(lines) if "Loading model" in x or "server listening" in x]
        return marks[-1] if marks else 0
    raise ValueError(scope)


def parse_selector_file(path: Path, scope: str) -> tuple[list[dict[str, Any]], int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = find_scope_start(lines, scope)
    rows: list[dict[str, Any]] = []

    for line_no, line in enumerate(lines[start:], start=start + 1):
        m = SELECTOR_RE.search(line)
        if m is None:
            continue
        fields = dict(FIELD_RE.findall(m.group(1)))
        if not {"current_vlm_a_bits", "selected_pair", "action_a_bits", "next_vlm_a_bits"}.issubset(fields):
            continue
        try:
            pair = str(fields["selected_pair"])
            pair_next_vlm, pair_action = parse_pair(pair)
            rows.append({
                "file": str(path),
                "line_no": int(line_no),
                "current_vlm_a_bits": int(fields["current_vlm_a_bits"]),
                "selected_pair": pair,
                "action_a_bits": int(fields["action_a_bits"]),
                "next_vlm_a_bits": int(fields["next_vlm_a_bits"]),
                "pair_next_vlm_bits": int(pair_next_vlm),
                "pair_action_bits": int(pair_action),
            })
        except Exception:
            pass

    return rows, len(lines)


def transition_check(rows: list[dict[str, Any]]) -> tuple[int, int]:
    checked = matched = 0
    for a, b in zip(rows, rows[1:]):
        checked += 1
        if a["next_vlm_a_bits"] == b["current_vlm_a_bits"]:
            matched += 1
    return matched, checked


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    current = Counter(r["current_vlm_a_bits"] for r in rows)
    action = Counter(r["action_a_bits"] for r in rows)
    pair = Counter(r["selected_pair"] for r in rows)
    matched, checked = transition_check(rows)

    avg_cost = 0.0
    if rows:
        avg_cost = sum(runtime_cost(r["current_vlm_a_bits"], r["action_a_bits"]) for r in rows) / total

    return {
        "total": int(total),
        "avg_runtime_cost": float(avg_cost),
        "current": current,
        "action": action,
        "pair": pair,
        "transition_matched": int(matched),
        "transition_checked": int(checked),
    }


def fmt_bits(counter: Counter[int], total: int) -> str:
    return "  ".join(
        f"A{b}:{int(counter.get(b, 0))}({pct(int(counter.get(b, 0)), total):.2f}%)"
        for b in BITS
    )


def fmt_pair(counter: Counter[str], total: int) -> str:
    if total <= 0:
        return ""
    parts = []
    for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        parts.append(f"{k}:{int(v)}({pct(int(v), total):.2f}%)")
    return "  ".join(parts)


def find_log_files(log_dir: Path, recursive: bool, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(log_dir.rglob(pat) if recursive else log_dir.glob(pat))
    out = []
    seen = set()
    for p in sorted(files):
        if not p.is_file():
            continue
        if p.name in {"summary.txt", "summary.json"}:
            continue
        if p.suffix.lower() in {".jsonl", ".json"}:
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--scope", choices=["last_server_session", "last_connection", "all"], default="last_server_session")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--pattern", action="append", default=None, help="can be repeated; default: *.log, *.txt, *.out")
    ap.add_argument("--summary-name", default="summary.txt")
    ap.add_argument("--json-name", default="summary.json")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    patterns = args.pattern if args.pattern else list(DEFAULT_PATTERNS)
    files = find_log_files(log_dir, args.recursive, patterns)

    all_rows: list[dict[str, Any]] = []
    per_file = []
    skipped = []

    for f in files:
        rows, n_lines = parse_selector_file(f, args.scope)
        if not rows:
            skipped.append(str(f))
            continue
        all_rows.extend(rows)
        s = summarize_rows(rows)
        per_file.append({
            "file": str(f),
            "lines": int(n_lines),
            "summary": s,
        })

    overall = summarize_rows(all_rows)

    summary_path = log_dir / args.summary_name
    json_path = log_dir / args.json_name

    lines: list[str] = []
    lines.append("Selector precision summary")
    lines.append(f"log_dir: {log_dir}")
    lines.append(f"scope: {args.scope}")
    lines.append(f"files_used: {len(per_file)}")
    lines.append(f"files_skipped_no_selector: {len(skipped)}")
    lines.append("")

    lines.append("[OVERALL]")
    total = overall["total"]
    lines.append(f"selector_decisions: {total}")
    lines.append(f"avg_runtime_cost(current_vlm+action): {overall['avg_runtime_cost']:.3f}")
    lines.append(f"current_vlm_a_bits: {fmt_bits(overall['current'], total)}")
    lines.append(f"action_a_bits:      {fmt_bits(overall['action'], total)}")
    lines.append(f"selected_pair:      {fmt_pair(overall['pair'], total)}")
    if overall["transition_checked"] > 0:
        lines.append(
            f"next_to_current_check: {overall['transition_matched']}/{overall['transition_checked']} "
            f"({pct(overall['transition_matched'], overall['transition_checked']):.2f}%)"
        )
    lines.append("")

    lines.append("[PER_FILE]")
    for item in per_file:
        f = item["file"]
        s = item["summary"]
        total = s["total"]
        rel = str(Path(f).relative_to(log_dir)) if Path(f).is_relative_to(log_dir) else f
        lines.append(f"- {rel}")
        lines.append(f"  selector_decisions: {total}")
        lines.append(f"  avg_runtime_cost: {s['avg_runtime_cost']:.3f}")
        lines.append(f"  current_vlm_a_bits: {fmt_bits(s['current'], total)}")
        lines.append(f"  action_a_bits:      {fmt_bits(s['action'], total)}")
        lines.append(f"  selected_pair:      {fmt_pair(s['pair'], total)}")
        if s["transition_checked"] > 0:
            lines.append(
                f"  next_to_current_check: {s['transition_matched']}/{s['transition_checked']} "
                f"({pct(s['transition_matched'], s['transition_checked']):.2f}%)"
            )

    if skipped:
        lines.append("")
        lines.append("[SKIPPED_NO_SELECTOR]")
        for f in skipped:
            try:
                lines.append(str(Path(f).relative_to(log_dir)))
            except Exception:
                lines.append(f)

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_payload = {
        "log_dir": str(log_dir),
        "scope": args.scope,
        "files_used": len(per_file),
        "files_skipped_no_selector": len(skipped),
        "overall": {
            "selector_decisions": overall["total"],
            "avg_runtime_cost": overall["avg_runtime_cost"],
            "current_vlm_a_bits": dict(overall["current"]),
            "action_a_bits": dict(overall["action"]),
            "selected_pair": dict(overall["pair"]),
            "transition_matched": overall["transition_matched"],
            "transition_checked": overall["transition_checked"],
        },
        "per_file": [
            {
                "file": x["file"],
                "selector_decisions": x["summary"]["total"],
                "avg_runtime_cost": x["summary"]["avg_runtime_cost"],
                "current_vlm_a_bits": dict(x["summary"]["current"]),
                "action_a_bits": dict(x["summary"]["action"]),
                "selected_pair": dict(x["summary"]["pair"]),
                "transition_matched": x["summary"]["transition_matched"],
                "transition_checked": x["summary"]["transition_checked"],
            }
            for x in per_file
        ],
        "skipped_no_selector": skipped,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"saved: {summary_path}")
    print(f"saved: {json_path}")
    print(f"files_used={len(per_file)} selector_decisions={overall['total']}")


if __name__ == "__main__":
    main()

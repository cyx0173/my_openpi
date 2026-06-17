#!/usr/bin/env python3
"""Generate one TXT summary for selector_test/action_chunk results.

Output format follows the comparison txt style:
  [OVERALL COMPARISON]
  [PER-TASK ACCURACY COMPARISON]
  [PER-TASK SUCCESS@THRESHOLD COMPARISON]

Default input:
  /home/chengyuxuan/openpi/selector_test/action_chunk

Default output:
  /home/chengyuxuan/openpi/selector_test/action_chunk/summary.txt
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path("/home/chengyuxuan/openpi/selector_test/action_chunk")
if not ROOT.exists():
    ROOT = Path("selector_test/action_chunk")

OUT_TXT = ROOT / "summary.txt"
THRESHOLDS = (320, 420, 520)
EXPECTED_PER_TASK: int | None = None
DEFAULT_METHOD = "action_chunk"

SUCCESS_KEYS = ("success", "succeeded", "is_success")
STEP_KEYS = (
    "success_step",
    "steps",
    "step",
    "num_steps",
    "episode_steps",
    "final_step",
    "done_step",
    "total_steps",
    "length",
)
EP_KEYS = ("episode_idx", "episode_id", "episode")
TASK_ID_KEYS = ("task_id", "task_idx")
TASK_NAME_KEYS = ("task_name", "task")
METHOD_KEYS = ("precision", "method", "mode", "config", "setting", "policy", "name")

TASK_NAME_TO_ID = {
    "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 0,
    "put_both_the_cream_cheese_box_and_the_butter_in_the_basket": 1,
    "turn_on_the_stove_and_put_the_moka_pot_on_it": 2,
    "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": 3,
    "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": 4,
    "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": 5,
    "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": 6,
    "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 7,
    "put_both_moka_pots_on_the_stove": 8,
    "put_the_yellow_and_white_mug_in_the_microwave_and_close_it": 9,
}


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        return extract_records(obj)
    except json.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def extract_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("episodes", "results", "records", "data", "items", "eval_results"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if any(k in obj for k in SUCCESS_KEYS):
            return [obj]
    return []


def get_bool(record: dict[str, Any]) -> bool | None:
    for key in SUCCESS_KEYS:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "1", "yes", "success", "succeeded"}:
                return True
            if s in {"false", "0", "no", "fail", "failed"}:
                return False
    return None


def get_int(record: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key not in record or record[key] is None:
            continue
        try:
            return int(record[key])
        except Exception:
            return None
    return None


def infer_task_from_path(path: Path) -> tuple[int | None, str]:
    rel = path.relative_to(ROOT)
    if not rel.parts:
        name = path.parent.name
    else:
        name = rel.parts[0]
    if name in TASK_NAME_TO_ID:
        return TASK_NAME_TO_ID[name], name
    m = re.match(r"^(?:task[_-]?)?(\d+)[_-](.+)$", name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, name


def infer_method(path: Path, record: dict[str, Any]) -> str:
    for key in METHOD_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            s = value.strip()
            if re.fullmatch(r"w\d+a\d+", s.lower()):
                return s.lower()
            if s.lower() in {"w4a4", "w4a8", "w4a16", "fp16", "bf16", "a4", "a8", "a16"}:
                return s.lower()

    for part in path.parts:
        low = part.lower()
        m = re.search(r"w\d+a\d+", low)
        if m:
            return m.group(0)
        if low in {"fp16", "bf16", "a4", "a8", "a16"}:
            return low
    return DEFAULT_METHOD


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(ROOT.rglob("episode_results.jsonl"))
    if not files:
        files = sorted(list(ROOT.rglob("*.jsonl")) + list(ROOT.rglob("*.json")))
    for path in files:
        if path.name == OUT_TXT.name or "_summary" in path.parts:
            continue
        records = read_records(path)
        if not records:
            continue
        path_task_id, path_task_name = infer_task_from_path(path)
        for rec in records:
            success = get_bool(rec)
            if success is None:
                continue
            task_id = get_int(rec, TASK_ID_KEYS)
            task_name = None
            for key in TASK_NAME_KEYS:
                if isinstance(rec.get(key), str) and rec[key].strip():
                    task_name = rec[key].strip()
                    break
            rows.append(
                {
                    "method": infer_method(path, rec),
                    "task_id": task_id if task_id is not None else path_task_id,
                    "task_name": task_name if task_name is not None else path_task_name,
                    "episode": get_int(rec, EP_KEYS),
                    "success": success,
                    "steps": get_int(rec, STEP_KEYS),
                    "source": str(path),
                }
            )
    return rows


def avg(values: list[int]) -> float | None:
    values = [v for v in values if isinstance(v, int)]
    if not values:
        return None
    return sum(values) / len(values)


def pct(x: float | None) -> str:
    if x is None or math.isnan(x):
        return "-"
    return f"{x:.2f}"


def num(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def acc(found: int, success: int) -> float:
    return 100.0 * success / found if found else 0.0


def success_at(rows: list[dict[str, Any]], threshold: int) -> float:
    found = len(rows)
    if found == 0:
        return 0.0
    ok = 0
    for r in rows:
        step = r.get("steps")
        if r["success"] and isinstance(step, int) and step <= threshold:
            ok += 1
    return 100.0 * ok / found


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    found = len(rows)
    success = sum(1 for r in rows if r["success"])
    expected = EXPECTED_PER_TASK if EXPECTED_PER_TASK is not None else found
    missing = max(0, expected - found)
    steps_all = [r["steps"] for r in rows if isinstance(r.get("steps"), int)]
    steps_success = [r["steps"] for r in rows if r["success"] and isinstance(r.get("steps"), int)]
    out = {
        "found": found,
        "missing": missing,
        "success": success,
        "acc": acc(found, success),
        "avg_steps": avg(steps_all),
        "avg_success_steps": avg(steps_success),
    }
    for t in THRESHOLDS:
        out[f"acc@{t}"] = success_at(rows, t)
    return out


def task_sort_key(task: tuple[int | None, str]) -> tuple[int, str]:
    task_id, task_name = task
    if task_id is None:
        return (10**9, task_name)
    return (task_id, task_name)


def method_sort_key(method: str) -> tuple[int, str]:
    order = {"w4a4": 0, "w4a8": 1, "w4a16": 2, "fp16": 3, DEFAULT_METHOD: 99}
    return (order.get(method, 50), method)


def fmt_cell(value: Any, width: int) -> str:
    return str(value).rjust(width)


def render_table(headers: list[str], rows: list[list[Any]], widths: list[int]) -> str:
    lines = []
    lines.append(" | ".join(fmt_cell(h, w) for h, w in zip(headers, widths)))
    lines.append("-" * len(lines[-1]))
    for row in rows:
        lines.append(" | ".join(fmt_cell(v, w) for v, w in zip(row, widths)))
    return "\n".join(lines)


def build_report(rows: list[dict[str, Any]]) -> str:
    methods = sorted({r["method"] for r in rows}, key=method_sort_key)
    tasks = sorted({(r["task_id"], r["task_name"]) for r in rows}, key=task_sort_key)

    by_method = {m: [r for r in rows if r["method"] == m] for m in methods}
    by_task_method = {
        (task_id, task_name, m): [r for r in rows if r["task_id"] == task_id and r["task_name"] == task_name and r["method"] == m]
        for task_id, task_name in tasks
        for m in methods
    }

    parts: list[str] = []
    line = "=" * 120

    parts.append(line)
    parts.append("[OVERALL COMPARISON]")
    parts.append(line)
    headers = ["precision", "found", "missing", "success", "acc(%)", "avg_steps", "avg_success_steps"]
    headers += [f"acc@{t}(%)" for t in THRESHOLDS]
    widths = [18, 18, 18, 18, 18, 18, 18] + [18] * len(THRESHOLDS)
    table_rows = []
    for m in methods:
        s = summarize(by_method[m])
        table_rows.append(
            [m, s["found"], s["missing"], s["success"], pct(s["acc"]), pct(s["avg_steps"]), pct(s["avg_success_steps"])]
            + [pct(s[f"acc@{t}"]) for t in THRESHOLDS]
        )
    parts.append(render_table(headers, table_rows, widths))

    parts.append("")
    parts.append(line)
    parts.append("[PER-TASK ACCURACY COMPARISON]")
    parts.append(line)
    headers = ["task_id", "task_name"]
    widths = [18, 70]
    for m in methods:
        headers += [f"{m}_found", f"{m}_acc(%)", f"{m}_avg_steps"]
        widths += [18, 18, 18]
    table_rows = []
    for task_id, task_name in tasks:
        row = [task_id if task_id is not None else "-", task_name]
        for m in methods:
            s = summarize(by_task_method[(task_id, task_name, m)])
            row += [s["found"], pct(s["acc"]), pct(s["avg_steps"])]
        table_rows.append(row)
    parts.append(render_table(headers, table_rows, widths))

    parts.append("")
    parts.append(line)
    parts.append("[PER-TASK SUCCESS@THRESHOLD COMPARISON]")
    parts.append(line)
    parts.append("")
    for t in THRESHOLDS:
        parts.append(f"[success@{t}]")
        headers = ["task_id", "task_name"] + [f"{m}_acc@{t}(%)" for m in methods]
        widths = [18, 70] + [18] * len(methods)
        table_rows = []
        for task_id, task_name in tasks:
            row = [task_id if task_id is not None else "-", task_name]
            for m in methods:
                row.append(pct(success_at(by_task_method[(task_id, task_name, m)], t)))
            table_rows.append(row)
        parts.append(render_table(headers, table_rows, widths))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    rows = collect_rows()
    if not rows:
        raise SystemExit(f"No records found under {ROOT}")
    report = build_report(rows)
    OUT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved TXT: {OUT_TXT}")


if __name__ == "__main__":
    main()

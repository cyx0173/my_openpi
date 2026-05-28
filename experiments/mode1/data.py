import json
import pathlib
import re
from collections import defaultdict


# 只改这里：你的 client log 目录。
ROOT = pathlib.Path("/home/chengyuxuan/openpi/experiments/mode1/logs/client")

HORIZONS = [320, 420, 520, 600]
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
PRECISION_ORDER = ["fp16", "w4a16", "w4a8", "w4a4"]

SAVED_RE = re.compile(
    r"\[Saved\]\s+(?P<path>\S+)\s+success=(?P<success>True|False)\s+chunks=(?P<chunks>\d+)"
)
BANK_RE = re.compile(r"task(?P<task_id>\d+)_ep(?P<ep>\d+)_(?P<bank_precision>.+?)_bank\.json$")


def parse_log_name(path: pathlib.Path):
    """Use log filename as the source of truth.

    Example:
      w4a8_put_both_moka_pots_on_the_stove.log
        -> precision=w4a8
        -> task_name=put_both_moka_pots_on_the_stove
    """
    stem = path.stem
    for p in PRECISION_ORDER:
        if stem == p:
            return p, "unknown_task"
        if stem.startswith(p + "_"):
            return p, stem[len(p) + 1:]
    return "unknown", stem


def read_bank_json(bank_path: pathlib.Path):
    try:
        if bank_path.exists():
            with open(bank_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def parse_logs():
    records = {}
    unmatched_logs = []

    for log_path in sorted(ROOT.rglob("*.log")):
        log_precision, task_name = parse_log_name(log_path)
        saved_count = 0

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = SAVED_RE.search(line)
                if not m:
                    continue

                bank_path = pathlib.Path(m.group("path"))
                bm = BANK_RE.search(bank_path.name)
                if not bm:
                    continue

                saved_count += 1

                task_id = int(bm.group("task_id"))
                ep = int(bm.group("ep"))
                bank_precision = bm.group("bank_precision").lower()

                # 关键修正：precision 以 log 文件名前缀为准，不以 bank 文件名为准。
                # 因为你的 w4a8/w4a16 log 里可能仍然保存成 taskXX_epYYY_w4a4_bank.json。
                precision = log_precision if log_precision != "unknown" else bank_precision

                success = m.group("success") == "True"
                chunks = int(m.group("chunks"))
                steps = chunks * REPLAN_STEPS

                bank = read_bank_json(bank_path)
                if bank is not None:
                    # success/chunks 用 json 里的真实值覆盖 log。
                    success = bool(bank.get("success", success))
                    chunks = int(bank.get("num_chunks", chunks))

                    total_steps = bank.get("total_steps", None)
                    wait = int(bank.get("num_steps_wait", NUM_STEPS_WAIT))
                    if total_steps is not None:
                        steps = max(0, int(total_steps) - wait)
                    else:
                        steps = chunks * REPLAN_STEPS

                # 同一个 precision/task/episode 如果 log 里重复出现，保留最后一次 Saved。
                key = (precision, task_name, task_id, ep)
                records[key] = {
                    "precision": precision,
                    "task_name": task_name,
                    "task_id": task_id,
                    "ep": ep,
                    "success": success,
                    "steps": steps,
                    "chunks": chunks,
                    "bank_precision": bank_precision,
                    "bank_path": str(bank_path),
                    "log_path": str(log_path),
                }

        if saved_count == 0:
            unmatched_logs.append(str(log_path))

    return list(records.values()), unmatched_logs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return f"{100 * x:5.1f}%"


def num(x):
    return "   - " if x is None else f"{x:6.1f}"


def ordered_precisions(records):
    ps = sorted({r["precision"] for r in records})
    return [p for p in PRECISION_ORDER if p in ps] + [p for p in ps if p not in PRECISION_ORDER]


def print_stats(title, records):
    print("\n" + title)
    print("-" * 130)
    print("precision   n   " + "  ".join(f"SR@{h}" for h in HORIZONS) + "   avg_done  avg_steps  avg_chunks")
    print("-" * 130)

    for p in ordered_precisions(records):
        xs = [r for r in records if r["precision"] == p]
        if not xs:
            continue

        n = len(xs)
        rates = []
        for h in HORIZONS:
            ok = [r for r in xs if r["success"] and r["steps"] <= h]
            rates.append(len(ok) / n)

        done_steps = [r["steps"] for r in xs if r["success"]]
        all_steps = [r["steps"] for r in xs]
        chunks = [r["chunks"] for r in xs]

        print(
            f"{p:<9} {n:3d} "
            + "  ".join(pct(x) for x in rates)
            + f"   {num(mean(done_steps))}  {num(mean(all_steps))}  {num(mean(chunks))}"
        )


def main():
    records, unmatched_logs = parse_logs()

    if not records:
        print(f"No [Saved] records found under: {ROOT}")
        return

    print("=" * 130)
    print(f"ROOT = {ROOT}")
    print(f"records = {len(records)}")
    print("重要：precision 以 log 文件名前缀为准，例如 w4a8_xxx.log -> w4a8。")
    print("如果 [Saved] 指向的 bank json 存在，会读取 json 里的 success/total_steps；否则用 chunks*5 估计 steps。")
    print("=" * 130)

    # 检查 log precision 和 bank filename precision 是否不一致。
    mismatches = [r for r in records if r["bank_precision"] != r["precision"]]
    if mismatches:
        print(f"\n[INFO] log precision != bank filename precision: {len(mismatches)} records")
        print("       这是正常的，如果你的 w4a8/w4a16 log 里保存的 bank 文件名仍然带 w4a4。")
        for r in mismatches[:10]:
            print(
                f"       log={r['precision']:<5} bank_file={r['bank_precision']:<5} "
                f"task={r['task_name']} ep={r['ep']:03d}"
            )

    print_stats("[ALL TASKS]", records)

    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_name"]].append(r)

    print("\n\n[PER TASK]")
    for task_name in sorted(by_task.keys()):
        print("\n" + "=" * 130)
        print(task_name)
        print_stats("", by_task[task_name])

    print("\n\n[FAIL CASES BY TASK / PRECISION]")
    for task_name in sorted(by_task.keys()):
        xs = by_task[task_name]
        fails = [r for r in xs if not r["success"]]
        if not fails:
            continue

        print("\n" + task_name)
        for r in sorted(fails, key=lambda x: (x["precision"], x["ep"])):
            print(f"  {r['precision']:<6} ep={r['ep']:03d} steps={r['steps']:4d} chunks={r['chunks']:3d}")

    if unmatched_logs:
        print("\n\n[LOGS WITHOUT SAVED RECORDS]")
        for p in unmatched_logs[:50]:
            print("  " + p)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import pathlib
import re
from collections import defaultdict

BASE = pathlib.Path("/home/chengyuxuan/openpi/experiments/mode1")

# [{steps, precision, task_name, episode_idx, success, chunks}]
records = []

# 使用 rglob 自动递归查找所有 logs/client/*.log，完美规避目录层级数错的问题
for log_file in sorted(BASE.rglob("logs/client/*.log")):
    parts = log_file.parts
    
    # 1. 提取 steps (例如从 '320steps' 中拿到 '320')
    try:
        steps_idx = next(i for i, p in enumerate(parts) if "steps" in p)
        steps = parts[steps_idx].replace("steps", "")
    except StopIteration:
        steps = "unknown"

    # 2. 从文件名解析 precision 和 task_name
    # 文件名形如: w4a4_put_both_stoves.log 
    stem = log_file.stem
    prec_task = stem.split("_", 1)
    precision = prec_task[0]                     # e.g., 'w4a4'
    task_name = prec_task[1] if len(prec_task) > 1 else stem # e.g., 'put_both_stoves'

    # 3. 按行解析日志内部的 [Saved] 标志
    try:
        log_content = log_file.read_text(encoding="utf-8", errors="ignore")
        for line in log_content.splitlines():
            if "[Saved]" not in line:
                continue
                
            # 使用更宽泛的正则匹配：支持 task 后面跟数字或字符，完美适配各类命名
            m = re.search(r"task(\w+)_ep(\d+).*success=(True|False).*chunks=(\d+)", line)
            if not m:
                continue
                
            records.append({
                "steps": steps,
                "precision": precision,
                "task_name": task_name, # 直接使用文件名里的任务名，更直观
                "episode_idx": int(m.group(2)),
                "success": m.group(3) == "True",
                "chunks": int(m.group(4)),
            })
    except Exception as e:
        print(f"⚠️ 读取文件失败 {log_file.name}: {e}")

if not records:
    print(f"❌ 未能从日志文件中解析到任何包含 '[Saved] ... success=...' 的有效数据。")
    print("请确认你的 .log 文件中是否已经打印了评测完成的 [Saved] 行。")
    exit(1)

# ==================== 汇总 ====================
# key: (steps, precision, task_name)
agg = defaultdict(lambda: {"success": 0, "total": 0, "total_chunks": 0})
for r in records:
    k = (r["steps"], r["precision"], r["task_name"])
    agg[k]["total"] += 1
    agg[k]["total_chunks"] += r["chunks"]
    if r["success"]:
        agg[k]["success"] += 1

# 打印表格
all_keys = sorted(agg, key=lambda x: (x[0], x[1], x[2]))

print()
print("=" * 90)
print(f"{'Steps':6s}  {'Precision':9s}  {'Task Name':30s}  {'Success Rate':18s}  {'Avg Chunks':12s}")
print("-" * 90)
for k in all_keys:
    steps, precision, task_name = k
    s = agg[k]
    rate = s["success"] / s["total"] * 100
    avg_chunks = s["total_chunks"] / s["total"]
    # 限制任务名打印长度防止错位
    display_name = task_name[:30]
    print(f"{steps:6s}  {precision:9s}  {display_name:30s}  "
          f"{rate:5.1f}% ({s['success']}/{s['total']})  {avg_chunks:5.1f}")

print("=" * 90)

# 按 steps × precision 合计整体表现
overall = defaultdict(lambda: {"success": 0, "total": 0, "total_chunks": 0})
for k in all_keys:
    steps, precision = k[0], k[1]
    s = agg[k]
    overall[(steps, precision)]["total"] += s["total"]
    overall[(steps, precision)]["success"] += s["success"]
    overall[(steps, precision)]["total_chunks"] += s["total_chunks"]

print()
print("Overall by Steps × Precision:")
print("-" * 65)
for (steps, precision), s in sorted(overall.items(), key=lambda x: (x[0][0], x[0][1])):
    rate = s["success"] / s["total"] * 100
    avg_chunks = s["total_chunks"] / s["total"]
    print(f"  {steps}steps / {precision:8s}: {rate:5.1f}% ({s['success']}/{s['total']}), "
          f"avg_chunks={avg_chunks:.1f}, total_eps={s['total']}")
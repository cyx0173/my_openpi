cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

RUN_DIR=/home/chengyuxuan/openpi/experiments/selector/query_dual_head_v2_current_labels
mkdir -p "$RUN_DIR"

nohup python src/openpi/models_pytorch/selector/train.py \
  --task-json /share/chengyuxuan-local/openpi/recovery_data_selector/task_schedule.json \
  --root /share/chengyuxuan-local/openpi/recovery_data_selector \
  --out-dir "$RUN_DIR" \
  --batch-size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --num-workers 4 \
  --selector-dim 512 \
  --require-all-precisions \
  > "$RUN_DIR/train.log" 2>&1 &

echo $! > "$RUN_DIR/train.pid"
echo "pid=$(cat $RUN_DIR/train.pid)"
echo "log=$RUN_DIR/train.log"
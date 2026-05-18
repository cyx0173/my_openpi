#!/bin/bash
# Mode 5: FP16 + W4A4 + W4A8 + W4A16 Quad-Server Launcher

OPENPI_DIR="/home/chengyuxuan/openpi"
mkdir -p "$OPENPI_DIR/logs/gpu/mode5"

# FP16 Server - Port 8000 - GPU 1
OPENPI_QUANT_MODE=0 CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py \
    --port 8000 \
    > "$OPENPI_DIR/logs/gpu/mode5/device0.log" 2>&1 &

# W4A4 Server - Port 8001 - GPU 0
OPENPI_QUANT_MODE=1 CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py \
    --port 8001 \
    --quantize \
    > "$OPENPI_DIR/logs/gpu/mode5/device1.log" 2>&1 &

# W4A8 Server - Port 8002 - GPU 2
OPENPI_QUANT_MODE=2 CUDA_VISIBLE_DEVICES=2 python scripts/serve_policy.py \
    --port 8002 \
    --quantize \
    > "$OPENPI_DIR/logs/gpu/mode5/device2.log" 2>&1 &

# W4A16 Server - Port 8003 - GPU 3
OPENPI_QUANT_MODE=3 CUDA_VISIBLE_DEVICES=3 python scripts/serve_policy.py \
    --port 8003 \
    --quantize \
    > "$OPENPI_DIR/logs/gpu/mode5/device3.log" 2>&1 &


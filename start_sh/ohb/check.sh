env \
-u OPENPI_ATM_CAPTURE_TAG \
-u OPENPI_ATM_CAPTURE_PATH \
-u OPENPI_OHB_CAPTURE_TAG \
-u OPENPI_OHB_CAPTURE_PATH \
-u OPENPI_ATM_ALPHA_ONES \
-u OPENPI_ATM_ALPHA_HALF \
-u OPENPI_OHB_BETA_ONES \
-u OPENPI_OHB_BETA_CONSTANT \
OPENPI_QUANT_MODE=2 \
OPENPI_ATM_ENABLE=1 \
OPENPI_ATM_SCOPE=gemma_expert \
OPENPI_ATM_ALPHA_PATH=/home/chengyuxuan/openpi/lab_track/atm_1/pi05_atm_alpha_w4a8.json \
OPENPI_OHB_ENABLE=1 \
OPENPI_OHB_SCOPE=gemma_expert \
OPENPI_OHB_BETA_PATH=/home/chengyuxuan/openpi/lab_track/atm_1/pi05_ohb_beta_w4a8_atm.json \
CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py --port 8001 --quantize
#!/bin/bash
# ============================================================
# run_single_rpi5.sh  —  Raspberry Pi 5 CPU-only LOSO fine-tuning
#
# Target hardware:
#   Raspberry Pi 5, ARM Cortex-A76 x4, 8 GB LPDDR4X
#
# Usage:
#   bash run_single_rpi5.sh <dataset> [checkpoint] [epochs] [hop_length] [conda_env] [test_set_dir] [context_length] [batch_size]
#
# Examples:
#   bash run_single_rpi5.sh wisdm
#   bash run_single_rpi5.sh wisdm tspulse_dual_domain_pretrained_120_full_dataset_bestckpt 100 30 naman_tspulse "" 120 256
#
# If you are not using conda on the Pi:
#   CONDA_ENV= bash run_single_rpi5.sh wisdm
#   PYTHON_CMD=/path/to/venv/bin/python CONDA_ENV= bash run_single_rpi5.sh wisdm
#
# Live log:
#   tail -f results_rpi5/logs_<checkpoint>_ep<N>_<ctx>_bs<B>/<dataset>.log
# ============================================================

set -o pipefail

DATASET=${1:?"Usage: bash run_single_rpi5.sh <dataset> [checkpoint] [epochs] [hop] [conda_env] [test_set_dir] [context_length] [batch_size]"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT=${2:-"$SCRIPT_DIR/checkpoints/tspulse_dual_domain_pretrained_120_full_dataset_bestckpt"}
EPOCHS=${3:-100}
HOP_LENGTH=${4:-30}             # 30 = 75% overlap with context_length=120
if [[ $# -ge 5 ]]; then
    CONDA_ENV="$5"
else
    CONDA_ENV="${CONDA_ENV-naman_tspulse}"
fi
TEST_SET_DIR=${6:-"$(cd "$(dirname "${BASH_SOURCE[0]}")"/test_set && pwd)"}
CONTEXT_LENGTH=${7:-120}
BATCH_SIZE=${8:-256}            # Conservative default for Pi 5 8 GB RAM

DEVICE="cpu"
PATIENCE=15
OUTPUT_DIR="results_rpi5"
PYTHON_CMD=${PYTHON_CMD:-python}

CHECKPOINT_NAME=$(basename "$CHECKPOINT")
RESULT_SUBDIR="${CHECKPOINT_NAME}_rpi5_ep${EPOCHS}_ctx${CONTEXT_LENGTH}_bs${BATCH_SIZE}"
LOG_DIR="$SCRIPT_DIR/$OUTPUT_DIR/logs_${CHECKPOINT_NAME}_ep${EPOCHS}_${CONTEXT_LENGTH}_bs${BATCH_SIZE}"

mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR/$RESULT_SUBDIR"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/${DATASET}.log"
RESULT_FILE="$SCRIPT_DIR/$OUTPUT_DIR/$RESULT_SUBDIR/ibm_${DATASET}.csv"

# Keep CPU threading matched to the four Cortex-A76 cores.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SCRIPT_DIR/granite-tsfm:${PYTHONPATH:-}"

echo "=============================================="
echo "  hartpulse.py — Raspberry Pi 5 fine-tuning"
echo "=============================================="
echo "  Dataset     : $DATASET"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Device      : $DEVICE"
echo "  Epochs      : $EPOCHS"
echo "  Context len : $CONTEXT_LENGTH"
echo "  Hop length  : $HOP_LENGTH"
echo "  Batch size  : $BATCH_SIZE"
echo "  Patience    : $PATIENCE"
echo "  Conda env   : ${CONDA_ENV:-none}"
echo "  Python cmd  : $PYTHON_CMD"
echo "  Test set dir: $TEST_SET_DIR"
echo "  Log file    : $LOG_FILE"
echo "  Result file : $RESULT_FILE"
echo "=============================================="
echo ""

# Resume-friendly: skip if result CSV already exists.
if [[ -f "$RESULT_FILE" ]]; then
    echo "[SKIP] $DATASET — result already exists at $RESULT_FILE"
    echo "       (delete it to re-run)"
    exit 0
fi

START_TIME=$(date +%s)

if [[ -n "$CONDA_ENV" ]]; then
    RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_CMD")
else
    RUNNER=("$PYTHON_CMD")
fi

"${RUNNER[@]}" -u "$SCRIPT_DIR/hartpulse.py" \
    --dataset_name    "$DATASET" \
    --checkpoint_path "$CHECKPOINT" \
    --device          "$DEVICE" \
    --epochs          "$EPOCHS" \
    --hop_length      "$HOP_LENGTH" \
    --context_length  "$CONTEXT_LENGTH" \
    --batch_size      "$BATCH_SIZE" \
    --patience        "$PATIENCE" \
    --test_set_dir    "$TEST_SET_DIR" \
    --output_dir      "$SCRIPT_DIR/$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=============================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  DONE  $DATASET   time: ${MINS}m ${SECS}s"
    echo "  Result: $RESULT_FILE"
else
    echo "  FAIL  $DATASET   time: ${MINS}m ${SECS}s   (exit $EXIT_CODE)"
fi
echo "  Log   : $LOG_FILE"
echo "=============================================="

exit $EXIT_CODE

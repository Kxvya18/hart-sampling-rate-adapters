#!/bin/bash
# ============================================================
# run_all_rpi5.sh  —  Sequential Raspberry Pi 5 CPU benchmark
#
# Usage:
#   bash run_all_rpi5.sh [checkpoint] [epochs] [hop_length] [conda_env] [test_set_dir] [context_length] [batch_size]
# ============================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT=${1:-"$SCRIPT_DIR/checkpoints/tspulse_dual_domain_pretrained_120_full_dataset_bestckpt"}
EPOCHS=${2:-100}
HOP_LENGTH=${3:-30}
if [[ $# -ge 4 ]]; then
    CONDA_ENV="$4"
else
    CONDA_ENV="${CONDA_ENV-naman_tspulse}"
fi
TEST_SET_DIR=${5:-"$SCRIPT_DIR/test_set"}
CONTEXT_LENGTH=${6:-120}
BATCH_SIZE=${7:-256}

DATASETS=(
    capture24
    daphnet
    dsads
    harsense
    harth
    hhar
    hugadb
    ku_har
    mhealth
    motion_sense
    opportunity
    realworld
    uci_har
    usc_had
    w_har
    wisdm
    wisdm_19_phone
    wisdm_19_watch
)

TOTAL=${#DATASETS[@]}
PASS=0
FAIL=0
FAILED_LIST=()
START_TIME=$(date +%s)

echo "=============================================="
echo "  hartpulse.py — Raspberry Pi 5 full run"
echo "=============================================="
echo "  Checkpoint  : $CHECKPOINT"
echo "  Epochs      : $EPOCHS"
echo "  Hop length  : $HOP_LENGTH"
echo "  Context len : $CONTEXT_LENGTH"
echo "  Batch size  : $BATCH_SIZE"
echo "  Conda env   : ${CONDA_ENV:-none}"
echo "  Test set dir: $TEST_SET_DIR"
echo "  Datasets    : $TOTAL"
echo "=============================================="
echo ""

IDX=0
for DATASET in "${DATASETS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "[$IDX/$TOTAL] Running $DATASET"
    if "$SCRIPT_DIR/run_single_rpi5.sh" \
        "$DATASET" \
        "$CHECKPOINT" \
        "$EPOCHS" \
        "$HOP_LENGTH" \
        "$CONDA_ENV" \
        "$TEST_SET_DIR" \
        "$CONTEXT_LENGTH" \
        "$BATCH_SIZE"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_LIST+=("$DATASET")
    fi
done

ELAPSED=$(( $(date +%s) - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=============================================="
echo "  Raspberry Pi 5 run complete"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
echo "  Time   : ${MINS}m ${SECS}s"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed datasets: ${FAILED_LIST[*]}"
fi
echo "=============================================="

[[ $FAIL -eq 0 ]]

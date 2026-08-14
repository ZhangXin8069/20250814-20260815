#!/bin/bash
# ==============================================================================
# LQCD Agent Validation Pipeline — Auto-Fix Loop
#
# Runs the complete pipeline with up to MAX_RETRIES attempts.
# On failure, runs auto_fix.py to parse errors and apply fixes.
# All output is tee'd to run.log.
# ==============================================================================

set -euo pipefail

MAX_RETRIES=5
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${RUN_DIR}/run.log"
PYTHON="$(which python3 || which python)"

echo "============================================================" | tee -a "$LOG_FILE"
echo "LQCD Agent Validation Pipeline" | tee -a "$LOG_FILE"
echo "Run directory: ${RUN_DIR}" | tee -a "$LOG_FILE"
echo "Max retries: ${MAX_RETRIES}" | tee -a "$LOG_FILE"
echo "Python: ${PYTHON}" | tee -a "$LOG_FILE"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# ── Environment Setup ────────────────────────────────────────────────────────

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# Add project root to PYTHONPATH
export PYTHONPATH="/root/lattice-pdf:${PYTHONPATH:-}"

# ── Run Pipeline with Retry Loop ─────────────────────────────────────────────

EXIT_CODE=0
for attempt in $(seq 0 $((MAX_RETRIES - 1))); do
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "ATTEMPT $((attempt + 1)) / ${MAX_RETRIES}" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    if ${PYTHON} "${RUN_DIR}/run_pipeline.py" \
        --run-dir "${RUN_DIR}" \
        --attempt "${attempt}" \
        2>&1 | tee -a "$LOG_FILE"; then

        echo "" | tee -a "$LOG_FILE"
        echo "============================================================" | tee -a "$LOG_FILE"
        echo "SUCCESS on attempt $((attempt + 1))!" | tee -a "$LOG_FILE"
        echo "============================================================" | tee -a "$LOG_FILE"
        EXIT_CODE=0
        break
    else
        EXIT_CODE=$?
        echo "" | tee -a "$LOG_FILE"
        echo "Attempt $((attempt + 1)) failed with exit code ${EXIT_CODE}" | tee -a "$LOG_FILE"

        if [ $attempt -lt $((MAX_RETRIES - 1)) ]; then
            echo "Running auto-fix..." | tee -a "$LOG_FILE"
            ${PYTHON} "${RUN_DIR}/auto_fix.py" \
                --run-dir "${RUN_DIR}" \
                --attempt "${attempt}" \
                --log "run.log" \
                2>&1 | tee -a "$LOG_FILE"

            echo "Auto-fix complete. Retrying..." | tee -a "$LOG_FILE"
        else
            echo "Max retries exhausted." | tee -a "$LOG_FILE"
        fi
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PIPELINE COMPLETE" | tee -a "$LOG_FILE"
echo "Exit code: ${EXIT_CODE}" | tee -a "$LOG_FILE"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# List output files
echo "" | tee -a "$LOG_FILE"
echo "Output files:" | tee -a "$LOG_FILE"
find "${RUN_DIR}" -type f \( -name "*.png" -o -name "*.json" -o -name "*.npz" -o -name "*.h5" -o -name "*.yaml" -o -name "*.md" \) \
    -not -path "*/.claude/*" \
    -exec ls -lh {} \; 2>/dev/null | tee -a "$LOG_FILE"

exit ${EXIT_CODE}

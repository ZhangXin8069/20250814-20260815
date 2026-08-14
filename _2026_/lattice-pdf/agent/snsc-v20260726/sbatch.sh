#!/bin/bash
# ==============================================================================
# SNSC Validation Pipeline — Slurm Submission
#
# One-click submission for the integrated validation pipeline on the cluster.
# Runs: 2pt distillation → OPE loading → huangcl analysis → final report
#
# The working directory (#SBATCH --chdir) is set to this script's directory
# so the pipeline finds run_config.json and outputs go under output_*/
#
# Usage:
#     cd /root/lattice-pdf/agent/snsc-v20260726 && sbatch sbatch.sh
# ==============================================================================

#SBATCH --job-name=snsc-v20260726
#SBATCH --output=logs/snsc_%j.out
#SBATCH --error=logs/snsc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --chdir=$(dirname "$(realpath "$0")")

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "SNSC Validation Pipeline — snsc-v20260726"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Job Name:    ${SLURM_JOB_NAME:-local}"
echo "Nodes:       ${SLURM_JOB_NUM_NODES:-1}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-8}"
echo "Working dir: ${SCRIPT_DIR}"
echo "Started:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Threading limits
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Python setup — activate conda environment
if [ -f /public/home/zhangxin/miniconda3/etc/profile.d/conda.sh ]; then
    source /public/home/zhangxin/miniconda3/etc/profile.d/conda.sh
    conda activate zhangxin-snsc 2>/dev/null || \
        echo "[WARNING] Could not activate conda env 'zhangxin-snsc', using system Python"
fi

# Python path
PYTHON="${CONDA_PREFIX:-/usr}/bin/python3"
echo "Python: ${PYTHON}"
${PYTHON} --version

# Add project root to PYTHONPATH
export PYTHONPATH="/public/home/zhangxin/lattice-pdf:${PYTHONPATH:-}"

# ── Run Pipeline ──────────────────────────────────────────────────────────────
echo ""
echo "Starting pipeline..."
echo ""

cd "${SCRIPT_DIR}"

${PYTHON} run_pipeline.py \
    2>&1 | tee -a "${LOG_DIR}/snsc_${SLURM_JOB_ID:-local}.pipeline.log"

EXIT_CODE=${PIPESTATUS[0]}

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Pipeline finished with exit code: ${EXIT_CODE}"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# List outputs
LATEST=$(ls -dt output_*/ 2>/dev/null | head -1)
if [ -n "${LATEST}" ]; then
    echo ""
    echo "Latest output: ${LATEST}"
    echo "  Data:  $(ls ${LATEST}/data/conf_*/ 2>/dev/null | wc -l) items"
    echo "  Plots: $(ls ${LATEST}/plots/*.png 2>/dev/null | wc -l) images"
    ls -lh ${LATEST}/plots/*.png 2>/dev/null || true
    echo ""
    echo "Log: ${LATEST}/run.log"
    echo "Report: ${LATEST}/final_report.md"
fi

exit ${EXIT_CODE}

#!/bin/bash
# ============================================================================
# Slurm 提交脚本 — 格点 QCD 质子非极化胶子 PDF 计算 (snsc/main.py)
# ============================================================================
#
# GPU 分区: nv100-ins, nv100-sug, dgx2, na100-ins, na100-sug, gpu-debug,
#            na100-40g, na800-sug, na800-pcie, h20-nettr
# CPU 分区: cpu6248R, cpueicc, i72c512g
#
# 分析模式:
#   --analysis-type pdf        胶子 PDF 全流程 (13 步可选)
#   --analysis-type proton-2pt 质子 2pt 蒸馏 (复现 donghx)
#   --analysis-type 2pt        2pt IOG 有效质量分析
#
# Python 环境: miniconda3, conda env "zhangxin-snsc"
#
# 用法:
#   sbatch sbatch.sh                                          # 默认参数 (L32x64, conf=20000)
#   sbatch --export=ALL,STEPS="1,2,3,6" sbatch.sh             # 覆盖步骤
#   sbatch --export=ALL,XP="cupy",PARTITION="na100-sug" sbatch.sh  # GPU 模式
#   sbatch --export=ALL,ANALYSIS_TYPE="proton-2pt" sbatch.sh   # 质子 2pt 模式
#   sbatch --export=ALL,ENSEMBLE="L24x72",CONF_START=46000 sbatch.sh  # L24x72 系综
# ============================================================================

# ============================================================================
# Slurm 作业配置 — 默认单 CPU 环境
# ============================================================================

#SBATCH --job-name=gluon_pdf
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --chdir=/public/home/zhangxin/lattice-pdf/snsc
#SBATCH --output=logs/gluon_pdf_%j.out
#SBATCH --error=logs/gluon_pdf_%j.err

# ============================================================================
# 环境设置
# ============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

source /public/home/zhangxin/miniconda3/etc/profile.d/conda.sh && conda activate zhangxin-snsc

# ============================================================================
# 路径配置
# ============================================================================

mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="output_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/data" "${OUTPUT_DIR}/plots"

# ============================================================================
# 参数配置 — 通过环境变量覆盖默认值
# ============================================================================

# ── 分析类型 ──
# pdf = 胶子 PDF 全流程, proton-2pt = 质子 2pt 蒸馏, 2pt = IOG 有效质量
ANALYSIS_TYPE="${ANALYSIS_TYPE:-pdf}"

# ── 后端选择 ──
# numpy (CPU) 或 cupy (GPU)
XP="${XP:-numpy}"

# ── 步骤选择 (仅 pdf 模式) ──
STEPS="${STEPS:-all}"

# ── 系综预设 ──
# L24x72, L32x64, L32x96, L36x108, L48x96, L48x144, L64x128
ENSEMBLE="${ENSEMBLE:-L32x64}"

# ── 数据类型 ──
DTYPE="${DTYPE:-complex128}"

# ── 格点参数 (不使用系综预设时) ──
NT="${NT:-64}"
NX="${NX:-32}"
NEV="${NEV:-100}"
NEV1="${NEV1:-100}"
DELTA_Z="${DELTA_Z:-15}"
Z_DIR="${Z_DIR:-2}"

# ── 动量 ──
PX="${PX:-0}"
PY="${PY:-0}"
PZ="${PZ:-6}"
PZ_LIST="${PZ_LIST:-}"

# ── 组态 ──
CONF_START="${CONF_START:-20000}"
CONF_STEP="${CONF_STEP:-50}"
CONF_NUM="${CONF_NUM:-1}"

# ── 动量涂抹 ──
MOM_SMEAR="${MOM_SMEAR:-3}"
MOM_SMEAR_PHASE="${MOM_SMEAR_PHASE:--3}"

# ── 规范组态文件 (ILDG 二进制, pdf 模式必需) ──
GAUGE_FILE="${GAUGE_FILE:-}"

# ── 匹配参数 ──
ALPHA_S="${ALPHA_S:-0.2}"
MU_OVER_PZ="${MU_OVER_PZ:-1.0}"

# ── 读取路径 (已有数据) ──
READ_2PT_DIR="${READ_2PT_DIR:-}"
READ_3PT_DIR="${READ_3PT_DIR:-}"
READ_VVV_DIR="${READ_VVV_DIR:-}"
READ_OPE_DIR="${READ_OPE_DIR:-}"

# ── 生成路径 (写入) ──
GEN_2PT_DIR="${GEN_2PT_DIR:-}"
GEN_3PT_DIR="${GEN_3PT_DIR:-}"
GEN_VVV_DIR="${GEN_VVV_DIR:-}"
GEN_OPE_DIR="${GEN_OPE_DIR:-}"

# ── 质子 2pt / 2pt 专用 ──
ELEMENT="${ELEMENT:-_Cg5g4}"
ALTTC="${ALTTC:-0.1053}"
MEFF_TYPE="${MEFF_TYPE:-cosh}"
HADRON="${HADRON:-pion}"
TSEP="${TSEP:-36}"

# ── 作图控制 ──
NO_PLOT="${NO_PLOT:-0}"

# ============================================================================
# 构建命令行参数
# ============================================================================

ARGS=(
    --analysis-type="${ANALYSIS_TYPE}"
    --xp="${XP}"
    --dtype="${DTYPE}"
    --output-dir="${OUTPUT_DIR}"
    --Px="${PX}"
    --Py="${PY}"
    --Pz="${PZ}"
    --conf-start="${CONF_START}"
    --conf-step="${CONF_STEP}"
    --conf-num="${CONF_NUM}"
    --delta-z="${DELTA_Z}"
    --z-dir="${Z_DIR}"
    --alpha-s="${ALPHA_S}"
    --mu-over-pz="${MU_OVER_PZ}"
)

# 步骤 (仅 pdf 模式)
if [ "${ANALYSIS_TYPE}" = "pdf" ]; then
    ARGS+=(--steps="${STEPS}")
fi

# 系综预设
if [ -n "${ENSEMBLE}" ]; then
    ARGS+=(--ensemble="${ENSEMBLE}")
fi

# 格点参数 (不使用系综时手动指定)
if [ -z "${ENSEMBLE}" ]; then
    ARGS+=(
        --Nt="${NT}" --Nx="${NX}"
        --Nev="${NEV}" --Nev1="${NEV1}"
        --mom-smear="${MOM_SMEAR}"
        --mom-smear-phase="${MOM_SMEAR_PHASE}"
    )
fi

# 动量扫描
if [ -n "${PZ_LIST}" ]; then
    ARGS+=(--Pz-list="${PZ_LIST}")
fi

# 规范组态
if [ -n "${GAUGE_FILE}" ]; then
    ARGS+=(--gauge-file="${GAUGE_FILE}")
fi

# 读取路径
[ -n "${READ_2PT_DIR}" ] && ARGS+=(--read-2pt-dir="${READ_2PT_DIR}")
[ -n "${READ_3PT_DIR}" ] && ARGS+=(--read-3pt-dir="${READ_3PT_DIR}")
[ -n "${READ_VVV_DIR}" ] && ARGS+=(--read-VVV-dir="${READ_VVV_DIR}")
[ -n "${READ_OPE_DIR}" ] && ARGS+=(--read-ope-dir="${READ_OPE_DIR}")

# 生成路径
[ -n "${GEN_2PT_DIR}" ] && ARGS+=(--gen-2pt-dir="${GEN_2PT_DIR}")
[ -n "${GEN_3PT_DIR}" ] && ARGS+=(--gen-3pt-dir="${GEN_3PT_DIR}")
[ -n "${GEN_VVV_DIR}" ] && ARGS+=(--gen-VVV-dir="${GEN_VVV_DIR}")
[ -n "${GEN_OPE_DIR}" ] && ARGS+=(--gen-ope-dir="${GEN_OPE_DIR}")

# 质子 2pt / 2pt 专用参数
ARGS+=(
    --element="${ELEMENT}"
    --alttc="${ALTTC}"
    --meff-type="${MEFF_TYPE}"
    --hadron="${HADRON}"
    --tsep="${TSEP}"
)

# 禁用作图
if [ "${NO_PLOT}" = "1" ]; then
    ARGS+=(--no-plot)
fi

# ============================================================================
# 打印运行信息
# ============================================================================

{
    echo "=============================================="
    echo "  格点 QCD 非极化胶子 PDF 计算"
    echo "  (snsc/main.py — 统一流水线)"
    echo "  开始时间: $(date)"
    echo "  节点: $(hostname)"
    echo "  conda: zhangxin-snsc"
    echo "  ------------------------------------"
    echo "  模式: ${ANALYSIS_TYPE}"
    echo "  后端: ${XP}"
    echo "  系综: ${ENSEMBLE}"
    echo "  组态: start=${CONF_START}, step=${CONF_STEP}, N=${CONF_NUM}"
    echo "  动量: Pz=${PZ}, Pz_list=${PZ_LIST}"
    [ -n "${GAUGE_FILE}" ]     && echo "  规范组态: ${GAUGE_FILE}"
    [ -n "${READ_2PT_DIR}" ]   && echo "  Read 2pt:  ${READ_2PT_DIR}"
    [ -n "${GEN_2PT_DIR}" ]    && echo "  Gen  2pt:  ${GEN_2PT_DIR}"
    [ -n "${READ_OPE_DIR}" ]   && echo "  Read OPE:  ${READ_OPE_DIR}"
    [ -n "${GEN_OPE_DIR}" ]    && echo "  Gen  OPE:  ${GEN_OPE_DIR}"
    echo "  输出目录: ${OUTPUT_DIR}"
    echo "  命令行: python -u main.py ${ARGS[*]}"
    echo "=============================================="
} | tee -a "${LOG_FILE}"

# ============================================================================
# 运行主脚本
# ============================================================================

python -u "./main.py" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# 运行后摘要
# ============================================================================

{
    echo "=============================================="
    echo "  结束时间: $(date)"
    echo "  退出码: ${EXIT_CODE}"
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "  状态: ✓ 成功"
    else
        echo "  状态: ✗ 失败 (退出码 ${EXIT_CODE})"
    fi
    echo "  输出目录: ${OUTPUT_DIR}"
    echo ""
    echo "  输出文件清单:"
    find "${OUTPUT_DIR}" -type f -exec ls -lh {} \; 2>/dev/null | awk '{print "    " $NF " (" $5 ")"}'
    echo "=============================================="
} | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}

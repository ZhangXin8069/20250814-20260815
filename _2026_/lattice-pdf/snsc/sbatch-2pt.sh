#!/bin/bash
# ============================================================================
# Slurm 提交脚本 — 质子 2pt 关联函数蒸馏计算
# ============================================================================
#
# 目标: 完全复现 examples/donghx/2pt_proton_Cg5gmu_L24x72_mom2_zdir_dcu.py
#
# 功能: 从本征矢量 + Perambulator 出发,
#       通过蒸馏框架计算质子 2pt 关联函数 C₂(t_snk, t_src),
#       包含动量涂抹、VVV Baryon Block 构造、Wick 收缩、
#       宇称投影和边界条件符号修正。
#       提取有效质量 (cosh 方法), 与参考数据对比,
#       生成出版级作图。
#
# 输出文件 (命名与参考代码完全一致):
#   Raw 收缩矩阵:
#     twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase{mom_smear}{element}_contract_conf{conf_id}.npy
#     形状: (Nt, Nt, 4, 4)
#
#   Parity 投影后:
#     twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase{mom_smear}{element}_nopol_ss_conf{conf_id}.npy
#     形状: (Nt, Nt)
#
#   有效质量:
#     meff_Pz{Pz}_conf{conf_id}.npz  (含 meff_gev, C2pt_1d, meff_ref, C2pt_ref)
#
#   对比报告:
#     compare_Pz{Pz}_conf{conf_id}.npz  (含 gen_raw, gen_pp, 各指标)
#
#   VVV 缓存:
#     VVV_Nev1{Nev1}_Px{Px}Py{Py}Pz{Pz}_conf{conf_id}.npy
#
# 作图输出 (plots/ 目录):
#   - C2pt_meff_Pz{Pz}_{element}_conf{conf_id}.pdf  (2pt 关联函数 + 有效质量)
#   - C2pt_matrix_Pz{Pz}_{element}_conf{conf_id}.pdf (2pt 矩阵热力图)
#   - raw_Pz{Pz}_scatter_comparison.pdf              (raw 散点对比)
#   - pp_Pz{Pz}_scatter_comparison.pdf               (pp 散点对比)
#   - meff_all_Pz_{element}_conf{conf_id}.pdf        (多动量有效质量汇总)
#
# Python 环境: miniconda3, conda env "zhangxin-snsc"
#
# 用法:
#   sbatch sbatch-2pt.sh                                          # 默认参数 (L24x72, conf=46000)
#   sbatch --export=ALL,CONF_ID="46000" sbatch-2pt.sh              # 指定组态
#   sbatch --export=ALL,PZ_LIST="-2,-3" sbatch-2pt.sh              # 指定动量
#   sbatch --export=ALL,ELEMENT="_Cg5g3" sbatch-2pt.sh             # 指定插值算符
#   sbatch --export=ALL,ENSEMBLE="L32x96" sbatch-2pt.sh            # 指定系综
#   sbatch --export=ALL,CONF_START=10050,CONF_STEP=50,CONF_NUM=3 sbatch-2pt.sh  # 多组态
# ============================================================================

# ============================================================================
# Slurm 作业配置 — 单 CPU 环境
# ============================================================================

#SBATCH --job-name=proton_2pt
#SBATCH --partition=cpu6248R,cpueicc,i72c512g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --chdir=/public/home/zhangxin/lattice-pdf/snsc
#SBATCH --output=logs/proton_2pt_%j.out
#SBATCH --error=logs/proton_2pt_%j.err

# ============================================================================
# 环境设置 — 限制多线程, 防止 BLAS/LAPACK 多线程干扰
# ============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# 激活 conda 环境
source /public/home/zhangxin/miniconda3/etc/profile.d/conda.sh && conda activate zhangxin-snsc

# ============================================================================
# 路径配置 (工作目录由 #SBATCH --chdir 设为 snsc/)
# ============================================================================

# 日志与输出目录
mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="output_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/data" "${OUTPUT_DIR}/plots"

# ============================================================================
# 参数配置 — 通过环境变量覆盖默认值
# ============================================================================

# ── 系综预设 (控制格点参数 + 数据路径) ──
# 可选: L24x72, L32x64, L32x96, L36x108, L48x96, L48x144, L64x128
ENSEMBLE="${ENSEMBLE:-L24x72}"

# ── 格点几何 (若不使用系综预设, 则需手动设置) ──
NX="${NX:-24}"
NT="${NT:-72}"

# ── 本征矢量 ──
NEV="${NEV:-100}"
NEV1="${NEV1:-100}"

# ── 组态选择 ──
# 单组态模式
CONF_ID="${CONF_ID:-46000}"
CONF_START="${CONF_START:-${CONF_ID}}"
# 多组态模式 (扫描)
CONF_STEP="${CONF_STEP:-1}"
CONF_NUM="${CONF_NUM:-1}"

# ── 动量 ──
PX="${PX:-0}"
PY="${PY:-0}"
# PZ: 单动量 (当 PZ_LIST 未设置时使用)
PZ="${PZ:--2}"
# PZ_LIST: 动量扫描列表 (逗号分隔, 覆盖 PZ)
# 默认扫描 5 个 boost 动量 (与 donghx 参考代码一致)
PZ_LIST="${PZ_LIST:--2,-3,-4,-5,-6}"

# ── 动量涂抹 ──
# L24x72: mom_smear=-2, phase=2
# L32x64: mom_smear=3, phase=-3
# L32x96: mom_smear=2, phase=-2
MOM_SMEAR="${MOM_SMEAR:--2}"
MOM_SMEAR_PHASE="${MOM_SMEAR_PHASE:-2}"

# ── 插值算符 ──
# _Cg5g4 = Cγ₅γ₄ (production default)
# _Cg5g3 = Cγ₅γ₃
# _Cg5   = Cγ₅
ELEMENT="${ELEMENT:-_Cg5g4}"

# ── 格距 (用于有效质量物理单位转换) ──
# L24x72: 0.1053 fm (a⁻¹≈1.874 GeV)
# L32x96: 0.083 fm  (a⁻¹≈2.377 GeV)
ALTTC="${ALTTC:-0.1053}"

# ── 数据路径 (覆盖系综预设的默认路径) ──
# 本征矢量 (仅读取)
EIG_DIR="${EIG_DIR:-}"
# Perambulator (仅读取)
PERAM_U_DIR="${PERAM_U_DIR:-}"
# 参考数据 (已有 donghx 结果, 用于对比验证)
CORR_NUCL_DIR="${CORR_NUCL_DIR:-}"

# ── 作图控制 ──
NO_PLOT="${NO_PLOT:-0}"

# ── 数据类型 ──
DTYPE="${DTYPE:-complex128}"

# ============================================================================
# 构建命令行参数
# ============================================================================

ARGS=(
    --analysis-type=proton-2pt
    --xp=numpy
    --dtype="${DTYPE}"
    --output-dir="${OUTPUT_DIR}"
    --Nt="${NT}"
    --Nx="${NX}"
    --Nev="${NEV}"
    --Nev1="${NEV1}"
    --conf-start="${CONF_START}"
    --conf-step="${CONF_STEP}"
    --conf-num="${CONF_NUM}"
    --Px="${PX}"
    --Py="${PY}"
    --Pz="${PZ}"
    --mom-smear="${MOM_SMEAR}"
    --mom-smear-phase="${MOM_SMEAR_PHASE}"
    --element="${ELEMENT}"
    --alttc="${ALTTC}"
)

# 系综预设 (控制 Nt/Nx/Nev/Nev1/mom_smear 及数据路径)
if [ -n "${ENSEMBLE}" ]; then
    ARGS+=(--ensemble="${ENSEMBLE}")
fi

# Pz-list (动量扫描)
if [ -n "${PZ_LIST}" ]; then
    ARGS+=(--Pz-list="${PZ_LIST}")
fi

# 数据路径 (仅在用户显式设置时覆盖系综默认值)
[ -n "${EIG_DIR}" ]        && ARGS+=(--eig-dir="${EIG_DIR}")
[ -n "${PERAM_U_DIR}" ]    && ARGS+=(--peram-u-dir="${PERAM_U_DIR}")
[ -n "${CORR_NUCL_DIR}" ]  && ARGS+=(--corr-nucl-dir="${CORR_NUCL_DIR}")

# 禁用作图
if [ "${NO_PLOT}" = "1" ]; then
    ARGS+=(--no-plot)
fi

# ============================================================================
# 打印运行信息
# ============================================================================

{
    echo "=============================================="
    echo "  质子 2pt 关联函数蒸馏计算"
    echo "  (复现 2pt_proton_Cg5gmu_L24x72_mom2_zdir_dcu.py)"
    echo "  开始时间: $(date)"
    echo "  节点: $(hostname)"
    echo "  conda: zhangxin-snsc"
    echo "  ------------------------------------"
    echo "  系综: ${ENSEMBLE}"
    echo "  格点: ${NT}×${NX}³, Nev=${NEV}, Nev1=${NEV1}"
    echo "  组态: start=${CONF_START}, step=${CONF_STEP}, N=${CONF_NUM}"
    echo "  动量: Pz_list=${PZ_LIST}"
    echo "  涂抹: mom_smear=${MOM_SMEAR}, phase=${MOM_SMEAR_PHASE}"
    echo "  插值: ${ELEMENT}"
    echo "  格距: a=${ALTTC} fm"
    [ -n "${EIG_DIR}" ]        && echo "  本征矢量: ${EIG_DIR}"
    [ -n "${PERAM_U_DIR}" ]    && echo "  Peramb:    ${PERAM_U_DIR}"
    [ -n "${CORR_NUCL_DIR}" ]  && echo "  参考对比:  ${CORR_NUCL_DIR}"
    echo "  生成输出:  ${OUTPUT_DIR}/data/"
    echo "  作图输出:  ${OUTPUT_DIR}/plots/"
    echo "  运行日志:  ${LOG_FILE}"
    echo "  命令行: python -u main.py ${ARGS[*]}"
    echo "=============================================="
} | tee -a "${LOG_FILE}"

# ============================================================================
# 运行主脚本 (无缓冲输出, 便于日志实时查看)
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
    echo "  数据: ${OUTPUT_DIR}/data/"
    echo "  作图: ${OUTPUT_DIR}/plots/"
    echo "  日志: ${OUTPUT_DIR}"
    echo ""
    echo "  输出文件清单:"
    find "${OUTPUT_DIR}" -type f -exec ls -lh {} \; 2>/dev/null | awk '{print "    " $NF " (" $5 ")"}'
    echo "=============================================="
} | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}

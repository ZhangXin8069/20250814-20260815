# AGENTS.md — examples/zengch

ZengCH 的 GPD/PDF 数据拟合分析脚本（`fit_*`、`matching_*`、`C_pt_load*`、`hB_data*` 等）。

## 功能

| 脚本类别 | 用途 |
|---|---|
| `C_pt_load*.py` | 加载 2pt/3pt 关联函数数据（`data_chose.py` 选择数据集，`data_chose_for_draw.py` 用于画图） |
| `fit_2pt.py` / `fit_E0.py` | 2pt 关联函数与基态能量拟合 |
| `fit_ratio*.py` / `fit_hR*.py` | 3pt/2pt 比率拟合（Feynman-Hellmann 变体见 `fit_ratio_FeynmenHellman*.py`） |
| `fit_zr*.py` / `fit_pz_a_extrapolatiing.py` | z/R 与 Pz 外推 |
| `matching*.py` | 轻锥 PDF 匹配（`matching_MPI.py` MPI 版，`matching_cc.py` 协变组合） |
| `hB_data*.py` | 匹配所需强子矩阵元数据预处理 |

## 关键约定

- **数据与工具在集群**：`sys.path` 指向 `/public/group/imp/zengch/LQCD/input_file`、`/public/group/imp/zengch/LQCD/tool`、`/public/home/zengch/All_TMD_dependence`，本机不解析。
- **拟合**：`iminuit`（`Minuit` + `LeastSquares` cost）；`pdb` 调试。
- 数据名形如 `L24x72_dhxmeang1_pz2`（dhx 数据均值化、pz 动量）。
- `picture/` 为输出图目录（含 `par_distribution/` 等子目录，图片不入库）。

## 验证

无测试框架；运行结果与图供人工核验（`picture/`）。

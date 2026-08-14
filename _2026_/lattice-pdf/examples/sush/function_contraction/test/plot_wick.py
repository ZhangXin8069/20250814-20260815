# 引入对应的包，地址：/public/home/sush/distillation/function_contraction
import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction.corr_wick import *

# 输入需要计算的算符, "|" 为分割符，每一个具体的强子算符都需要用 "|" 分开
sink_operators = ['|', 'd^d', 'gamma_5', 'u', '|'] # , 'u', 'u', 'gamma_proton_sink', 'd'
source_operators = ['|', 'u^d', 'gamma_5', 'd', '|'] # , 'u^d', 'd^d', 'gamma_proton_source', 'u^d', 'd^d', 'gamma_pion_source', 'u'
curr_operators = [] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
A = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    # curr_operators = [''],
    Cpt = '2pt',
    curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
    )

# 画图
fig, ax = plot_figure_wick(A, diagram_index=0, Cpt = '2pt')
fig.savefig('/public/home/sush/distillation/function_contraction/test/figure/wick_contraction_diagram_pipi_2pt.pdf', dpi=300, bbox_inches='tight')

sink_operators = ['|', 'd^d', 'gamma_5', 'u', '|'] # , 'u', 'u', 'gamma_proton_sink', 'd'
source_operators = ['|', 'u^d', 'gamma_5', 'd', '|'] # , 'u^d', 'd^d', 'gamma_proton_source', 'u^d', 'd^d', 'gamma_pion_source', 'u'
curr_operators = ['u^d', 'gamma_curr', 'u'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
A = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    # curr_operators = [''],
    Cpt = '3pt',
    curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
    )

# 画图
fig, ax = plot_figure_wick(A, diagram_index=0, Cpt = '3pt')
fig.savefig('/public/home/sush/distillation/function_contraction/test/figure/wick_contraction_diagram_pipi_3pt.pdf', dpi=300, bbox_inches='tight')

sink_operators = ['|', 'd^d', 'gamma_5', 'u', '|', '|', 'u', 'u', 'gamma_7', 'd', '|'] # , 'u', 'u', 'gamma_proton_sink', 'd'
source_operators = ['|', 'd^d', 'gamma_7', 'u^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'] # , 'u^d', 'd^d', 'gamma_proton_source', 'u^d', 'd^d', 'gamma_pion_source', 'u'
curr_operators = ['|', 'u^d', 'gamma_4', 'u', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
A = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    # curr_operators = [''],
    Cpt = '2pt',
    curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
    )

# 画图
fig, ax = plot_figure_wick(A, diagram_index=0, Cpt = '2pt')
fig.savefig('/public/home/sush/distillation/function_contraction/test/figure/wick_contraction_diagram.pdf', dpi=300, bbox_inches='tight')


sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 
curr_operators = ['|', 'u^d', 'gamma_C1', 'u', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
wick_diag = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '3pt',
    curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
    )

# 画图
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 创建一个 PdfPages 对象，指定输出的 PDF 文件名
with PdfPages('/public/home/sush/distillation/function_contraction/test/figure/wick_contraction_diagram_PP_3pt.pdf') as pdf:
    for i in range(len(wick_diag['result_indx'])):
        # 创建一个新图形
        fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '3pt')
        
        # 将当前图形保存到 PDF（新的一页）
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
        # 关闭图形以释放内存（重要！）
        plt.close(fig)

sink_operators = ['|', 'd^d', 'gamma_P', 'u', '|']
source_operators = ['|', 'u^d', 'gamma_K', 's', '|'] 
curr_operators = ['|', 'l^d', 'gamma_C1', 'l', '|', '|', 's^d', 'gamma_C1', 'l', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
wick_diag = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '3pt',
    curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
    )

# 画图
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 创建一个 PdfPages 对象，指定输出的 PDF 文件名
with PdfPages('/public/home/sush/distillation/function_contraction/test/figure/wick_contraction_diagram_Kpi.pdf') as pdf:
    for j in range(len(wick_diag)):
        for i in range(len(wick_diag[j]['result_indx'])):
            # 创建一个新图形
            fig, ax = plot_figure_wick(wick_diag[j], diagram_index=i, Cpt = '3pt')
            
            # 将当前图形保存到 PDF（新的一页）
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            
            # 关闭图形以释放内存（重要！）
            plt.close(fig)

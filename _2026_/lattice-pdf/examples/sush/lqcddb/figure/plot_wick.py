# 引入对应的包，地址：/public/home/sush/distillation/function_contraction
from lqcddb import wick_contraction, plot_figure_wick, identify_equivalent_diagrams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#****************************************************NN 2pt*************************************
#Group 0
sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 

wick_diag_1 = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '2pt',
    curr_operators = [],
    )

#Group 1
sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

wick_diag_2= wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '2pt',
    curr_operators = [],
    )



# #Group 2
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
# source_operators = [-1.0, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

# wick_diag_3 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 3
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

# wick_diag_4 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 4
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 

# wick_diag_5 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 5
# sink_operators = [-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 

# wick_diag_6 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 6
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 

# wick_diag_7 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 7
sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 
source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

wick_diag_8 = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '2pt',
    curr_operators = [],
    )
print(wick_diag_8)
# #Group 8
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 
# source_operators = [-1, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

# wick_diag_9 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 9
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

# wick_diag_10 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 10
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

# wick_diag_11 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 11
# sink_operators = [-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

# wick_diag_12 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 12
# sink_operators = [-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = [-1, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

# wick_diag_13 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 13
# sink_operators = [-1, '|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

# wick_diag_14 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 14
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = [-1, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

# wick_diag_15 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 15
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

# wick_diag_16 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# wick_diag = [wick_diag_1, wick_diag_2, wick_diag_3, wick_diag_4, wick_diag_5, wick_diag_6, wick_diag_7, wick_diag_8, wick_diag_9, wick_diag_10, wick_diag_11, wick_diag_12, wick_diag_13, wick_diag_14, wick_diag_15, wick_diag_16]
# print(identify_equivalent_diagrams(*wick_diag))
# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# num = 0
# with PdfPages('/public/home/sush/distillation/0v2b/figure/NN_2pt.pdf') as pdf:
#     for j in range(len(wick_diag)):
#         for i in range(len(wick_diag[j]['result_indx'])):
#             # if num == 30 or num == 44 or num == 78 or num == 91:
#             # 创建一个新图形
#             fig, ax = plot_figure_wick(wick_diag[j], diagram_index=i, Cpt = '2pt')
#             fig.text(0.98, 0.02, f"{num}", ha='right', va='bottom', fontsize=12)
            
#             # 将当前图形保存到 PDF（新的一页）
#             pdf.savefig(fig, dpi=300, bbox_inches='tight')
            
#             # 关闭图形以释放内存（重要！）
#             plt.close(fig)
                
#             num += 1

# #****************************************************Np 2pt*************************************

# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_1 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# wick_diag = [wick_diag_1]
# print(identify_equivalent_diagrams(wick_diag_1))
# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# num = 0
# with PdfPages('/public/home/sush/distillation/0v2b/figure/Np_2pt.pdf') as pdf:
#     for j in range(len(wick_diag)):
#         for i in range(len(wick_diag[j]['result_indx'])):
#             # 创建一个新图形
#             fig, ax = plot_figure_wick(wick_diag[j], diagram_index=i, Cpt = '2pt')
#             fig.text(0.98, 0.02, f"{num}", ha='right', va='bottom', fontsize=12)
            
#             # 将当前图形保存到 PDF（新的一页）
#             pdf.savefig(fig, dpi=300, bbox_inches='tight')
            
#             # 关闭图形以释放内存（重要！）
#             plt.close(fig)
            
#             num += 1

# #****************************************************3pt*************************************
# curr_operators = ['|', 'u^d', 'gamma_mu', 'd', '|']

# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_1 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '3pt',
#     curr_operators = curr_operators,
#     )

# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_2 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '3pt',
#     curr_operators = curr_operators,
#     )

# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_3 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '3pt',
#     curr_operators = curr_operators,
#     )

# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_4 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '3pt',
#     curr_operators = curr_operators,
#     )

# wick_diag = [wick_diag_1, wick_diag_2, wick_diag_3, wick_diag_4]
# print(identify_equivalent_diagrams(wick_diag_1, wick_diag_2, wick_diag_3, wick_diag_4))
# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# num = 0
# with PdfPages('/public/home/sush/distillation/0v2b/figure/Np_I-1.5-I-0.5_2pt.pdf') as pdf:
#     for j in range(len(wick_diag)):
#         for i in range(len(wick_diag[j]['result_indx'])):
#             # 创建一个新图形
#             fig, ax = plot_figure_wick(wick_diag[j], diagram_index=i, Cpt = '3pt')
#             fig.text(0.98, 0.02, f"{num}", ha='right', va='bottom', fontsize=12)
            
#             # 将当前图形保存到 PDF（新的一页）
#             pdf.savefig(fig, dpi=300, bbox_inches='tight')
            
#             # 关闭图形以释放内存（重要！）
#             plt.close(fig)
            
#             num += 1

#****************************************************2pt*************************************
# # neutron pi+ neutron pi+
# # 输入需要计算的算符, "|" 为分割符，每一个具体的强子算符都需要用 "|" 分开
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_5', 'd', '|', '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_4', 'u', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     # curr_operators = [''],
#     Cpt = '2pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/neutron_pi+_neutron_pi+.new.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)

# # neutron pi+ proton pi0d
# # 输入需要计算的算符, "|" 为分割符，每一个具体的强子算符都需要用 "|" 分开
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'd^d', 'gamma_5', 'd', '|', '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_4', 'u', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     # curr_operators = [''],
#     Cpt = '2pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/neutron_pi+_proton_pi0d.new.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)

# # proton pi0u proton pi0u
# # 输入需要计算的算符, "|" 为分割符，每一个具体的强子算符都需要用 "|" 分开
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_5', 'u', '|', '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_4', 'u', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     # curr_operators = [''],
#     Cpt = '2pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/proton_pi0u_proton_pi0u.new.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)

# # proton pi0d proton pi0d
# # 输入需要计算的算符, "|" 为分割符，每一个具体的强子算符都需要用 "|" 分开
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'd^d', 'gamma_5', 'd', '|', '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_4', 'u', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     # curr_operators = [''],
#     Cpt = '2pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/proton_pi0d_proton_pi0d.new.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)

# # proton pi0d neutron pi+
# # 输入需要计算的算符, "|" 为分割符，每一个具体的强子算符都需要用 "|" 分开
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_5', 'd', '|', '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_4', 'u', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     # curr_operators = [''],
#     Cpt = '2pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/proton_pi0d_neutron_pi+.new.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)
        
# #**********************************************3pt**********************************************
# # proton J_\nu J_\mu neutron pi-
# # 输入需要计算的算符, "|" 为分割符，每一个具体的强子算符都需要用 "|" 分开
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
# # sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

# source_operators = ['|', 'd^d', 'gamma_5', 'u', '|', '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_C1', 'd', '|', '|', 'u^d', 'gamma_C2', 'd', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '3pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/proton_J_nu_J_mu_neutron_pi-.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '3pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)

# # neutron J_\mu neutron pi-
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|'] 
# # sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

# source_operators = ['|', 'd^d', 'gamma_5', 'u', '|', '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_C1', 'd', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '3pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/neutron_J_mu_neutron_pi-.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '3pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)
        
# # proton J_\mu neutron
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
# # sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 
# curr_operators = ['|', 'u^d', 'gamma_C2', 'd', '|'] #, 'u^d', 'gamma_curr_2', 'u' #, 'u^d', 'gamma_curr_1', 'd'

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '3pt',
#     curr_operators = curr_operators, # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 画图

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/proton_J_mu_neutron.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '3pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)


# # proton proton
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|'] 
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [], # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/proton_proton.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)
        
# # neutron neutron
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|'] 

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [], # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/neutron_neutron.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)
        
# # neutron pi- neutron pi-
# source_operators = ['|', 'd^d', 'gamma_5', 'u', '|', '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|',] 

# # 调用wick收缩，输出一个字典，其中 'result_indx' 为收缩指标 'result_name' 为收缩的参量名称（仅供参考，具体需要配合收缩图像进行分析） 'result_sign'为收缩后的系数，一般会差一个 -1 ！
# wick_diag = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [], # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
#     )

# # 创建一个 PdfPages 对象，指定输出的 PDF 文件名
# with PdfPages('/public/home/sush/distillation/0v2b/figure/neutron_pi-_neutron_pi-.pdf') as pdf:
#     for i in range(len(wick_diag['result_indx'])):
#         # 创建一个新图形
#         fig, ax = plot_figure_wick(wick_diag, diagram_index=i, Cpt = '2pt')
        
#         # 将当前图形保存到 PDF（新的一页）
#         pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
#         # 关闭图形以释放内存（重要！）
#         plt.close(fig)
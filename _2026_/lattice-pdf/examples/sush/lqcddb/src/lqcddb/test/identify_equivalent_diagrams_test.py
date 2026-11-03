# 引入对应的包，地址：/public/home/sush/distillation/function_contraction
from lqcddb import wick_contraction, identify_equivalent_diagrams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# #****************************************************NN 2pt*************************************
# #Group 0
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 

# wick_diag_1 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

#Group 1
sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

wick_diag_2= wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '2pt',
    curr_operators = [],
    )

#Group 2
sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
source_operators = [-1.0, '|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

wick_diag_3 = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '2pt',
    curr_operators = [],
    )

#Group 3
sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|'] 
source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 

wick_diag_4 = wick_contraction(
    sink_operators = sink_operators, 
    source_operators = source_operators, 
    Cpt = '2pt',
    curr_operators = [],
    )

# #Group 4
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 

# wick_diag_5 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 5
# sink_operators = [-1, '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 

# wick_diag_6 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 6
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|'] 

# wick_diag_7 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 7
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_8 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 8
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 
# source_operators = [-1, '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

# wick_diag_9 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 9
# sink_operators = ['|', 'u', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

# wick_diag_10 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 10
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_11 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 11
# sink_operators = [-1, '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|', '|', 'd^d', 'gamma_5', 'u', '|'] 

# wick_diag_12 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 12
# sink_operators = [-1, '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = [-1, '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

# wick_diag_13 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 13
# sink_operators = [-1, '|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

# wick_diag_14 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

# #Group 14
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = [-1, '|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'd^d', 'gamma_5', 'd', '|'] 

# wick_diag_15 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

#Group 15
# sink_operators = ['|', 'd', 'u', 'gamma_7', 'd', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 
# source_operators = ['|', 'u^d', 'gamma_7', 'd^d', 'd^d', '|', '|', 'u^d', 'gamma_5', 'u', '|'] 

# wick_diag_16 = wick_contraction(
#     sink_operators = sink_operators, 
#     source_operators = source_operators, 
#     Cpt = '2pt',
#     curr_operators = [],
#     )

B = [wick_diag_2, wick_diag_3, wick_diag_4]
A = identify_equivalent_diagrams(*B)

print(f"Number of equivalence groups: {len(A)}")
for g_idx, group in enumerate(A):
    print(f"\nGroup {g_idx}:")
    for (d_idx, diag_idx, coeff) in group:
        print(f"  dict={d_idx:2d}, diagram={diag_idx}, coefficient={coeff:+.0f}   (result_sign={B[d_idx]['result_sign'][diag_idx]:+.0f})")
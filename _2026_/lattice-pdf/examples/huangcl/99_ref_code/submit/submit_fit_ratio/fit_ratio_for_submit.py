#!/public/home/zengch/anaconda3/bin/python3
import sys, fileinput
sys.path.append('/public/group/imp/zengch/LQCD/renorma')
from fit_ratio import c0_vs_z

# 默认值
n_remove, t_sep_list = None, []

for line in fileinput.input():
    tmp = line.split()
    if not tmp or tmp[0].startswith('#'):   # 跳过空行/注释
        continue
    if tmp[0] == 'n_remove':
        n_remove = int(tmp[1])
    if tmp[0] == 't_sep_list':
        t_sep_list = [int(x) for x in tmp[1:]]

if n_remove is None or not t_sep_list:
    sys.exit('input_file 必须包含 n_remove 和 t_sep_list')

#data_name = 'L24x72_dhxmeang1_pz6'
#data_name = 'L32x64_C32P29_sxmtest_pz0'
data_name = 'L32x64_E32P29_sxmtest_pz0'
#data_name = 'L32x96_dhx_pz6'
#data_name = 'L48x144_pz0'
#data_name = 'L48x144_dhx_pz6'
#data_name = 'L32x64_C32P23_pz3'
c0_vs_z(n_remove, t_sep_list, data_name, 'boot')
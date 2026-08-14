#!/usr/bin/env python3
import sys
from pathlib import Path

def usage():
    sys.exit('用法: python make_inputs_nab.py n a b c\n'
             '生成 input_file_n_a_b .. input_file_n_a_c')

if len(sys.argv) != 5:
    usage()

n, a, b, c = map(int, sys.argv[1:])
if b > c:
    usage()

for k in range(b, c + 1):
    fname = f'input_file_{n}_{a}_{k}'
    with open(fname, 'w') as f:
        f.write(f'n_remove   {n}\n')
        t_list = ' '.join(map(str, range(a, k + 1)))
        f.write(f't_sep_list {t_list}\n')
    print(f'已生成 {fname}')
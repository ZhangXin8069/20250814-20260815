import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from lqcddb import conjugate_operator, wick_contraction

# pion
sink_operator = ['|', 'u^d', 'gamma_5', 'd', '|']
source_operator = conjugate_operator(sink_operator)
print(source_operator)

A = wick_contraction(
    sink_operators = sink_operator, 
    source_operators = source_operator, 
    # curr_operators = [''],
    Cpt = '2pt',
    curr_operators = [], # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
)
print(A)

# proton
sink_operator = ['|', 'u', 'u', 'C * gamma_5', 'd', '|']
source_operator = conjugate_operator(sink_operator)
print(source_operator)

A = wick_contraction(
    sink_operators = sink_operator, 
    source_operators = source_operator, 
    # curr_operators = [''],
    Cpt = '2pt',
    curr_operators = [], # 'u^d', 'gamma_curr', 'u', 'u^d', 'gamma_curr_2', 'u', 'u^d', 'gamma_curr_3', 'u'
)

print(A)
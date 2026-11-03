import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import conjugate_operator

# pion
sink_operator = ['|', 'u^d', 'gamma_5', 'd', '|']
source_operator = conjugate_operator(sink_operator)
print(source_operator)

# proton
sink_operator = ['|', 'u', 'u', 'C * gamma_5', 'd', '|']
source_operator = conjugate_operator(sink_operator)
print(source_operator)


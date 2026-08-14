#!/bin/bash

# 参数设置
n_set=(3)
a_set=(8)
b_set=(20)



#n_set=(2 3)
#a_set=(7 8 9 10)
#b_set=(12 14 16)


echo "开始生成文件..."
echo "n_set: ${n_set[@]}"
echo "a_set: ${a_set[@]}"
echo "b_set: ${b_set[@]}"
echo "将生成 $(( ${#n_set[@]} * ${#a_set[@]} * ${#b_set[@]} )) 个文件"
echo

# 生成所有组合的文件
for n in "${n_set[@]}"; do
    for a in "${a_set[@]}"; do
        for b in "${b_set[@]}"; do
            fname="input_file_${n}_${a}_${b}"
            
            # 生成文件内容
            {
                echo "n_remove   $n"
                seq_list=$(seq $a $b | tr '\n' ' ')
                echo "t_sep_list $seq_list"
            } > "$fname"
            
            echo "已生成 $fname"
        done
    done
done

echo
echo "完成! 共生成 $(( ${#n_set[@]} * ${#a_set[@]} * ${#b_set[@]} )) 个文件"
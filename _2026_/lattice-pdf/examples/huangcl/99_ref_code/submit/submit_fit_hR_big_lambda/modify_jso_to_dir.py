import json
import os
import glob
import re

# ===== 配置区 =====
new_mu = 2.0                     # 要修改的 mu 的值，写 2.0，不要写 2
fit_change = 0                   # lambda 外推拟合改变范围（正数为加，负数为减）
new_append_note = "_Nremovenew8"        # 替换 append_note 字段的值为这个字符串 syslamm1, syszs
new_zs_input = 0.3              # 修改 zs_input 字段的值 0.3 -> 0.2
target_dir = "sys_err_for_lambda_extra_submit"   # 保存修改后文件的目录
# =================

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 获取当前目录下所有 .json 文件
json_files = glob.glob("*.json")

if not json_files:
    print("当前目录下没有找到 .json 文件。")
else:
    for filename in json_files:
        try:
            # 读取原始 JSON 文件
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 修改 mu
            if "mu" in data:
                data["mu"] = new_mu
                print(f"{filename}: mu -> {new_mu}")

            # 修改 append_note 字段的值
            if "append_note" in data:
                original_append = data["append_note"]
                data["append_note"] = new_append_note
                print(f"{filename}: append_note {original_append} -> {data['append_note']}")

            # 修改 zs_input 字段的值
            if "zs_input" in data:
                original_zs = data["zs_input"]
                data["zs_input"] = new_zs_input
                print(f"{filename}: zs_input {original_zs} -> {data['zs_input']}")

            # 修改 lamb_do_set
            if "lamb_do_set" in data and isinstance(data["lamb_do_set"], list):
                original = data["lamb_do_set"]
                data["lamb_do_set"] = [x + fit_change for x in original]
                print(f"{filename}: lamb_do_set {original} -> {data['lamb_do_set']}")

            # 修改 lamb_up_set
            if "lamb_up_set" in data and isinstance(data["lamb_up_set"], list):
                original = data["lamb_up_set"]
                data["lamb_up_set"] = [x + fit_change for x in original]
                print(f"{filename}: lamb_up_set {original} -> {data['lamb_up_set']}")
            
            # ========== 新增：同步修改 lambda_extra，同加 fit_change ==========
            if "lambda_extra" in data and isinstance(data["lambda_extra"], (int, float)):
                orig_extra = data["lambda_extra"]
                data["lambda_extra"] = orig_extra + fit_change
                print(f"{filename}: lambda_extra {orig_extra} -> {data['lambda_extra']}")

            # 构造目标文件路径
            target_path = os.path.join(target_dir, filename)

            # 先以缩进 2 生成字符串（会强制数组换行）
            json_str = json.dumps(data, indent=2, ensure_ascii=False)

            # 将多行数组（不包含嵌套）合并为单行，正确处理空字符串元素
            def compact_array(match):
                content = match.group(1)
                lines = content.split('\n')
                elements = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.endswith(','):
                        line = line[:-1]
                    elements.append(line)
                return '[' + ', '.join(elements) + ']'

            # 使用正则替换简单数组
            json_str = re.sub(
                r'\[\s*\n(.*?)\n\s*\]',
                compact_array,
                json_str,
                flags=re.DOTALL
            )

            # 写入文件
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

            print(f"已保存修改后的文件: {target_path}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print("所有文件处理完成。")
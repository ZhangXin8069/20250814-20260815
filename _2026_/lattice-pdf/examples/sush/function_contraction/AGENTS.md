# AGENTS.md — examples/sush/function_contraction

核心规则（唯一一条）：**只允许修改 `test/` 目录**。所有新脚本、图、输出都放那里。

上游原版在集群 `/public/home/sush/distillation/`；本目录为导入用扁平模块（`sys.path.insert` 方式）。新工作优先用正式包 `../lqcddb/`。

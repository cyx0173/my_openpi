你现在帮我完整调研 OpenPI PI0.5 PyTorch 模型结构和量化覆盖范围。请不要先修改核心逻辑，只做代码阅读、运行 introspection、输出报告和必要的 debug 脚本。

背景：
我现在使用的是 OpenPI 的 PI0.5 PyTorch 模型，入口大概是 scripts/serve_policy.py，通过 train_config.model.load_pytorch(train_config, weight_path) 加载模型。模型类里有 PI0Pytorch，里面包含：
- paligemma_with_expert
- action_in_proj
- action_out_proj
- pi0.5 下的 time_mlp_in / time_mlp_out
- paligemma_with_expert.paligemma
- paligemma_with_expert.gemma_expert

我现在有几个核心问题：
1. PI0.5 完整模型结构到底是什么？
2. 所谓 VLM 部分到底是哪些 module？
3. 所谓 action expert / action head 到底是哪些 module？
4. 为什么 enable_openpi_duquant_all_linears(model) 后会显示替换了 289 个 Linear？
5. 这 289 个 Linear 分别来自哪些模块？哪些属于 VLM，哪些属于 action expert，哪些属于 action projection / time MLP？
6. 当前量化是否覆盖了 action_in_proj / action_out_proj / time_mlp_in / time_mlp_out？
7. 当前量化是否覆盖了 gemma_expert 里的 q_proj/k_proj/v_proj/o_proj 和 gate_proj/up_proj/down_proj？
8. 当前量化是否覆盖了 paligemma 主干里的 vision tower、language model、multimodal projector？
9. 我们现在做的 ATM/OHB 到底作用在哪些层？这些层和被量化的 Linear 有什么关系？

请按下面步骤调研。

一、源码定位
请找到并阅读以下定义的位置，列出文件路径和关键类/函数：
- PI0Pytorch
- PaliGemmaWithExpertModel
- paligemma_with_expert.paligemma 的具体结构
- paligemma_with_expert.gemma_expert 的具体结构
- action_in_proj / action_out_proj / time_mlp_in / time_mlp_out 的定义和使用位置
- sample_actions / denoise_step / embed_prefix / embed_suffix 或类似函数
- enable_openpi_duquant_all_linears
- DuQuant Linear wrapper 类
- ATM/OHB 当前挂载逻辑相关文件：pi05_atm.py, pi05_ohb.py, modeling_gemma.py 中 eager_attention_forward

二、写一个 introspection 脚本
请新增一个脚本，例如：
scripts/inspect_pi05_model_structure.py

要求：
1. 尽量复用 scripts/serve_policy.py 的模型加载逻辑，确保加载的是同一个 checkpoint 和同一个 train_config。
2. 支持参数：
   --quantize  是否执行 enable_openpi_duquant_all_linears(model)
   --out-dir   输出目录
3. 加载模型后输出以下文件：
   A. full_named_modules.txt
      每一行：
      name | class | extra info
   B. linear_layers_before_or_after_quant.csv
      每一行至少包含：
      name, class_name, in_features, out_features, bias, branch, suffix, is_quantized
   C. module_tree_summary.txt
      按高层模块统计 module 数量和 Linear 数量
   D. quantized_layers.txt
      只列出被 DuQuant/Quant wrapper 替换的层名、原始 shape、所属 branch
   E. action_related_modules.txt
      列出所有名字里包含 action / time_mlp / state_proj / gemma_expert 的 module
   F. atm_ohb_target_layers.txt
      列出当前 ATM/OHB 会匹配到的 self_attn 层，包含 name、num_heads、head_dim、q/k/v/o proj 类型

三、branch 分类规则
请在脚本里给每个 Linear 自动分类 branch，规则先按名字匹配：
- branch = "action_expert" if name.startswith("paligemma_with_expert.gemma_expert.")
- branch = "vlm_paligemma" if name.startswith("paligemma_with_expert.paligemma.")
- branch = "action_projection" if name in 或包含：
  action_in_proj, action_out_proj, time_mlp_in, time_mlp_out, state_proj, action_time_mlp_in, action_time_mlp_out
- branch = "vision_tower" if name 里包含 vision_tower / vision_model / siglip / image
- branch = "language_model" if name 里包含 language_model / gemma.model.layers
- branch = "multimodal_projector" if name 里包含 projector / multi_modal_projector
- branch = "other" otherwise

如果实际名字和这些规则不一致，请根据真实 named_modules 结果修正，并在报告里说明真实命名。

四、统计内容
请输出清晰统计表：

1. 总 module 数
2. 总 nn.Linear 数
3. quantize 前 Linear 数
4. quantize 后被替换的 Quant/DuQuant Linear 数
5. 按 branch 统计：
   - vlm_paligemma 有多少 Linear
   - action_expert 有多少 Linear
   - action_projection 有多少 Linear
   - vision_tower 有多少 Linear
   - language_model 有多少 Linear
   - multimodal_projector 有多少 Linear
   - other 有多少 Linear

6. 按 suffix 统计：
   - q_proj
   - k_proj
   - v_proj
   - o_proj
   - out_proj
   - gate_proj
   - up_proj
   - down_proj
   - fc1
   - fc2
   - action_in_proj
   - action_out_proj
   - time_mlp_in
   - time_mlp_out
   - linear
   - others

7. 按 shape 统计 Top 30：
   例如 Linear(1152->1152): count
   Linear(2048->2048): count
   等

五、解释 289 个 Linear 的来源
我之前看到量化日志类似：
[OPENPI-DUQUANT] Matched nn.Linear layers: 289
suffix breakdown:
  k_proj: 45
  v_proj: 45
  q_proj: 45
  out_proj: 27
  fc1: 27
  fc2: 27
  o_proj: 18
  gate_proj: 18
  up_proj: 18
  down_proj: 18
  linear: 1

请你基于真实 named_modules，把这 289 个拆开解释：
- 哪些 45 个 q/k/v 来自哪里？为什么是 45，不是 18？
- 哪些 27 个 out_proj / fc1 / fc2 来自哪里？
- 哪些 18 个 o_proj / gate_proj / up_proj / down_proj 是不是正好来自 gemma_expert 18 层？
- 那个 linear: 1 是谁？是不是 action_in_proj 或 projector？
- action_out_proj 是否被量化？如果没有，为什么没有？
- time_mlp_in / time_mlp_out 是否被量化？如果没有，为什么没有？
- vision tower 里的 Linear 是否被量化？如果有，请列出来；如果没有，也请说明依据。

六、VLM 和 action expert 的边界解释
请根据源码和 named_modules 解释：

1. paligemma_with_expert.paligemma 是什么？
   - 包含视觉编码器吗？
   - 包含语言模型吗？
   - 图像 token 和语言 token 是在哪里处理的？
   - 它在 sample_actions 中主要负责 prefix KV cache 吗？

2. paligemma_with_expert.gemma_expert 是什么？
   - 是否是 action expert？
   - 有多少层？
   - 每层包含什么 attention 和 MLP？
   - 它在 denoise_step 中如何接收 state/action/time token？
   - 为什么我们把 ATM/OHB 只挂在 gemma_expert.model.layers.*.self_attn 上？

3. action_in_proj / action_out_proj / time_mlp 是什么？
   - action_in_proj 把什么维度映射到什么维度？
   - action_out_proj 把什么映射回 action_dim？
   - time_mlp_in/out 在 PI0.5 里干什么？
   - 这些算不算 action head 的一部分？

七、ATM/OHB 目标层核对
请确认当前 ATM/OHB 匹配函数实际匹配到哪些层：
- paligemma_with_expert.gemma_expert.model.layers.0.self_attn
...
- paligemma_with_expert.gemma_expert.model.layers.17.self_attn

请输出每层：
name
num_heads
head_dim
q_proj type
k_proj type
v_proj type
o_proj type
是否已经被量化

八、最终输出一个 Markdown 报告
请生成：
lab_track/model_structure/pi05_structure_report.md

报告结构：
1. Executive summary
2. PI0.5 high-level structure
3. VLM 部分有哪些
4. Action expert / action head 部分有哪些
5. Linear 层总数和 289 个量化层来源
6. 当前量化覆盖范围
7. action_in_proj / action_out_proj / time_mlp 是否被量化
8. ATM/OHB 作用层和原因
9. 对后续实验的建议：
   - 哪些层可以量化
   - 哪些层建议先不量化
   - 如果要做 W4A4/W4A8/W4A16，应该分别校准哪些 alpha/beta

请尽量用真实代码和 introspection 输出支撑，不要凭空猜测。
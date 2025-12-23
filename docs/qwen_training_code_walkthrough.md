# Megatron-LM 中 Qwen 的训练实现（从源码走读）

面向第一次接触 Megatron-LM 源码的工程师：这份文档**不从架构/ModuleSpec 讲起**，而是按“训练入口 → 模型构建 → Qwen 差异落点 → tokenizer/数据 → checkpoint 转换”的顺序，直接指到仓库里的具体代码位置，帮助你快速建立“Qwen 在 Megatron 里到底是怎么训练起来的”的心智模型。

> 重要事实（先讲结论）  
> 在 Megatron-LM 里，**Qwen2/2.5/3（dense decoder-only LM）通常不会对应一个独立的 `QwenModel` 类**。绝大多数情况下，它被当成“Llama-like 的 GPT decoder-only”来训练：训练 loop 和 `GPTModel` 是通用的，Qwen 的差异主要通过一组参数（RMSNorm、SwiGLU、GQA、RoPE base、QKV bias、Qwen3 的 QK LayerNorm 等）落地到通用实现上，再通过 checkpoint 转换把 HF 权重映射到 Megatron 格式。

---

## 1. 训练是从哪里开始的：`pretrain_gpt.py`

Megatron 的“预训练/继续预训练/SFT（文本 LM）”入口通常是 `pretrain_gpt.py`，Qwen 也一样。

- `pretrain_gpt.py#L252`：脚本最后调用 `megatron.training.pretrain(...)` 启动训练流程。
- `pretrain_gpt.py#L122`：`forward_step(...)` 是每个 step 的前向入口（取 batch、喂给 model、返回 loss closure）。

源码上的关键点：

1) **训练 loop 不因为 Qwen 有分支**：Qwen/Llama/GPT 都走同一份 `pretrain_gpt.py`。  
2) Qwen 的差异不在这里，而在“模型怎么 build、tokenizer 怎么 build、checkpoint 怎么转换”。

相关代码：
- `pretrain_gpt.py#L122`
- `pretrain_gpt.py#L252`

---

## 2. 模型是怎么 build 出来的：`model_provider` → `gpt_builder` → `GPTModel`

### 2.1 `model_provider.py`：只负责调用 builder

`model_provider.py` 的 `model_provider(...)` 基本就是一个 “builder dispatcher”：

- `model_provider.py#L24`：定义 `model_provider(...)`
- `model_provider.py#L67`：最后调用 `model_builder(args, ...)` 返回模型

它不会因为 Qwen 做结构性分支，你可以把它理解成“训练入口统一，模型构建可替换”。

相关代码：
- `model_provider.py#L24`
- `model_provider.py#L67`

### 2.2 `gpt_builders.py`：决定用哪个 GPT layer spec，并构造 `GPTModel`

真正把“命令行参数”变成“一个可训练的 decoder-only Transformer”的地方在 `gpt_builders.py`：

- `gpt_builders.py#L24`：`gpt_builder(args, pre_process, post_process, ...)`
- `gpt_builders.py#L80`：构造 `GPTModel(...)` 并把关键开关塞进去

尤其是这些参数最终会影响 Qwen 的行为（RoPE、是否 untie embedding/output、是否走特定 attention/MLP 组合等）：

- `position_embedding_type=args.position_embedding_type`
- `rotary_percent=args.rotary_percent`
- `rotary_base=args.rotary_base`
- `share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights`

相关代码：
- `gpt_builders.py#L24`
- `gpt_builders.py#L80`

---

## 3. Qwen 的“适配”具体落在了哪些源码点？

下面按 Qwen 系列常见差异点（RoPE base / only-QKV-bias / QK LayerNorm / tokenizer 模板）逐个定位到代码。

### 3.1 RoPE：`--rotary-base 1000000` 为什么能生效？

Qwen2.x/2.5/3 很常见的配置是 `--position-embedding-type rope` 且 `--rotary-base 1000000`（也就是 HF 里的 `rope_theta`）。

这条参数最终落在：

- `megatron/core/models/gpt/gpt_model.py#L152`：当 `position_embedding_type == 'rope'` 时构造 `RotaryEmbedding(...)`  
- `megatron/core/models/common/embeddings/rotary_pos_embedding.py#L56`：`RotaryEmbedding.__init__` 里用 `rotary_base` 计算 `inv_freq`

也就是说，Megatron 并没有“Qwen 专用 RoPE”；它是通用 RoPE，Qwen 只需要把 `rotary_base` 配对齐即可。

相关代码：
- `megatron/core/models/gpt/gpt_model.py#L152`
- `megatron/core/models/common/embeddings/rotary_pos_embedding.py#L56`

### 3.2 Qwen2.5：只给 QKV 加 bias（`--disable-bias-linear` + `--add-qkv-bias`）怎么实现？

Qwen2.5 的典型做法是：**大部分线性层没有 bias，但 QKV projection 有 bias**。Megatron 用两个配置开关来表达：

- `megatron/core/transformer/transformer_config.py#L147`：`add_bias_linear`（默认 True）
- `megatron/core/transformer/transformer_config.py#L151`：`add_qkv_bias`（默认 False）

真正“只让 QKV 有 bias”的落点在 SelfAttention 构造 QKV 线性层时：

- `megatron/core/transformer/attention.py#L1046`：`bias=self.config.add_bias_linear or self.config.add_qkv_bias`

因此你在启动参数里写：

- `--disable-bias-linear` → `add_bias_linear=False`：关闭全局线性层 bias  
- `--add-qkv-bias` → `add_qkv_bias=True`：仅 QKV bias 重新打开  

就能得到 Qwen2.5 的 “only-QKV-bias”。

相关代码：
- `megatron/core/transformer/transformer_config.py#L147`
- `megatron/core/transformer/attention.py#L1046`

### 3.3 Qwen3：QK LayerNorm（`--qk-layernorm`）怎么落到 attention 里？

Qwen3 常见会对 Q/K embedding 额外做一次 norm（很多资料称 QK LayerNorm）。

Megatron 对应的配置字段是：

- `megatron/core/transformer/transformer_config.py#L192`：`qk_layernorm: bool`

SelfAttention 初始化时会根据 submodules 配置创建 `q_layernorm` / `k_layernorm`：

- `megatron/core/transformer/attention.py#L1053` 到 `megatron/core/transformer/attention.py#L1071`

你不需要先理解 ModuleSpec 的细节也能抓住关键点：当 `--qk-layernorm` 打开后，attention 子模块里会真实存在 `self.q_layernorm/self.k_layernorm`，从而在 forward 里对 Q/K 做额外归一化（这也是 Qwen3 相比 Qwen2.5 更“独特”的一处）。

相关代码：
- `megatron/core/transformer/transformer_config.py#L192`
- `megatron/core/transformer/attention.py#L1053`

---

## 4. Qwen 训练里 tokenizer 的实现与坑位

### 4.1 预训练/继续预训练：基本用 `HuggingFaceTokenizer`

训练时 tokenizer 的 build 入口在：

- `megatron/training/tokenizer/tokenizer.py#L21`：`build_tokenizer(args, ...)`
- `megatron/training/tokenizer/tokenizer.py#L49`：`args.tokenizer_type == 'HuggingFaceTokenizer'` 分支

并且会计算 `padded_vocab_size`（让 vocab size 能被 TP 相关约束整除）：

- `megatron/training/tokenizer/tokenizer.py#L110`：若没从 checkpoint 读到 padded vocab，就会调用 `_vocab_size_with_padding(...)`

工程上这会直接影响 “embedding/output 的 shape 是否能对上 HF checkpoint”，所以很多 Qwen 配置脚本会显式设置：

- `--padded-vocab-size ...` 或 `--make-vocab-size-divisible-by ...`

相关代码：
- `megatron/training/tokenizer/tokenizer.py#L21`
- `megatron/training/tokenizer/tokenizer.py#L49`
- `megatron/training/tokenizer/tokenizer.py#L110`

### 4.2 SFT/RL（对话数据）：Qwen 的 ChatML 模板在哪里？

如果你做 SFT/RL，需要把 conversation 格式化成 tokens，Qwen2.x/2.5 常用的 ChatML 模板 `<|im_start|>...<|im_end|>` 在这里：

- `megatron/training/tokenizer/multimodal_tokenizer.py#L164`：支持 `prompt_format in ("qwen2p0", "qwen2p5")`  
- `megatron/training/tokenizer/multimodal_tokenizer.py#L169`：使用 `qwen2p0_custom_template`  
- `megatron/training/tokenizer/multimodal_tokenizer.py#L173`：给出默认 system message（“You are Qwen...”）

这块属于“为了对齐 Qwen Instruct 数据格式”的明确适配代码路径。

相关代码：
- `megatron/training/tokenizer/multimodal_tokenizer.py#L164`

---

## 5. 从 HuggingFace Qwen checkpoint 开始训练：转换脚本里的 Qwen 分支

很多团队训练 Qwen 的起点是 HF checkpoint（继续预训练 / SFT / RLHF）。Megatron 支持把 HF checkpoint 转成 Megatron 格式，而这一步也是“适配工作量”的集中地。

在 `tools/checkpoint/loader_llama_mistral.py` 里，Qwen2.5 被作为一种 `--model-size` 支持，并且会为转换过程构造一个“形状对齐”的 Megatron(legacy) model args：

- `tools/checkpoint/loader_llama_mistral.py#L469`：如果是 `qwen2.5`，强制 `margs.tokenizer_type = "HuggingFaceTokenizer"`
- `tools/checkpoint/loader_llama_mistral.py#L471`：并设置 `margs.add_qkv_bias = True`（这是对齐 Qwen2.5 的关键）

相关代码：
- `tools/checkpoint/loader_llama_mistral.py#L469`

---

## 6. 仓库里现成的 Qwen 配置脚本（最直观的“Qwen 训练参数长什么样”）

你可以直接把这些脚本当成“Qwen 适配 checklist”：

### 6.1 Qwen2.5-7B Instruct（dense）关键参数

`examples/post_training/modelopt/conf/Qwen/Qwen2.5-7B-Instruct.sh#L10` 中最关键的是：

- `--disable-bias-linear` + `--add-qkv-bias`（only QKV bias）
- `--normalization RMSNorm`
- `--swiglu`
- `--group-query-attention --num-query-groups 4`
- `--position-embedding-type rope --rotary-base 1000000`
- `--untie-embeddings-and-output-weights`（是否 untie 要跟具体 checkpoint 对齐）

相关代码：
- `examples/post_training/modelopt/conf/Qwen/Qwen2.5-7B-Instruct.sh#L10`

### 6.2 Qwen3-8B（dense）关键参数

`examples/post_training/modelopt/conf/Qwen/Qwen3-8B.sh#L10` 相比 Qwen2.5 的“新增重点”是：

- `--qk-layernorm`（Qwen3 的关键差异）

相关代码：
- `examples/post_training/modelopt/conf/Qwen/Qwen3-8B.sh#L10`

### 6.3 Qwen2.5-32B 的 RL 示例里有 tokenizer/pad 的工程坑

`examples/rl/model_configs/qwen_2p5_32b.sh#L56` 提到：

- “原始 Qwen tokenizer 的 padding_id 有坑，unsloth tokenizer 修复了”

这类问题在 RL/SFT 比预训练更常见（因为会强依赖对话模板/特殊 token），建议你把它当成排查 checklist 的一项。

相关代码：
- `examples/rl/model_configs/qwen_2p5_32b.sh#L56`

---

## 7. 额外路径：直接用 HuggingFace 的 Qwen2ForCausalLM（不是主流预训练路径，但仓库里确实有）

如果你想把 HF 模型本体包一层塞进 Megatron（而不是用 Megatron Core 的 `GPTModel`），仓库里有一个 Qwen2 的 HF wrapper：

- `megatron/core/models/huggingface/module.py#L62`：从 HF config 识别 model_type，遇到 qwen 返回 `"qwen"`
- `megatron/core/models/huggingface/module.py#L82`：`build_hf_model(...)` 会构造 `QwenHuggingFaceModel`
- `megatron/core/models/huggingface/qwen_model.py#L21`：`QwenHuggingFaceModel` 内部加载 `transformers.models.qwen2.Qwen2ForCausalLM`
- `megatron/core/models/huggingface/qwen_model.py#L39`：forward 使用 `inputs_embeds` 走 HF 模型

这条路更像“训练框架用 Megatron、模型实现用 transformers”，与“用 Megatron Core 训练 Qwen-like GPT”是两条不同路线。

相关代码：
- `megatron/core/models/huggingface/module.py#L62`
- `megatron/core/models/huggingface/qwen_model.py#L21`

---

## 8. 你要“从头适配一个新的 Qwen 变体”，通常最先改哪里？（实战顺序）

1) **先不改核心模型代码，先用参数对齐跑通**：以 `examples/post_training/modelopt/conf/Qwen/*.sh` 为模板，把 RMSNorm/SwiGLU/GQA/RoPE base/QKV bias/QK LayerNorm 这些对齐。  
2) **需要 HF → Megatron checkpoint 转换时**：在 `tools/checkpoint/loader_llama_mistral.py` 增加你的 `--model-size` 分支（类似 `tools/checkpoint/loader_llama_mistral.py#L469`），把对齐结构必须的开关（尤其 `add_qkv_bias`、tie/untie、rope base）设对。  
3) **做对话 SFT/RL 时**：确保对话模板一致（Qwen2/2.5 常见 ChatML），对照 `megatron/training/tokenizer/multimodal_tokenizer.py#L164`。  
4) **遇到 padding_id/special tokens 的坑**：优先在 tokenizer 选择与配置层解决（可参考 RL 示例里使用 `unsloth/Qwen2.5-32B` 的做法：`examples/rl/model_configs/qwen_2p5_32b.sh#L56`）。  
5) **只有当你的 Qwen 变体引入 Megatron 还没有的算子/层行为**（例如全新 attention/norm 逻辑），才需要进入 `megatron/core/transformer/...` 做代码级实现。


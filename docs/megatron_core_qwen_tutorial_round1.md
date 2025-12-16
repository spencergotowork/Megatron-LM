# Megatron-Core / Megatron-LM 教程（第 1 轮）：核心架构与 Qwen 适配路径

> 目标读者：有分布式训练经验的 AI Infra 工程师；跳过 TP/PP/DP 基础概念，聚焦 Megatron-Core 的实现落点与 Qwen 适配。
>
> 代码基于本仓库（Megatron-LM 主线），重点路径可直接点击打开查看源码。

---

## 0) 安装与最小可运行（Megatron-Core + Megatron-LM）

仓库 README 推荐方式（适合在 NGC PyTorch 容器里）：

```bash
pip install --no-build-isolation megatron-core[mlm,dev]
pip install --no-build-isolation .[mlm,dev]
```

关键依赖：
- Transformer Engine（TE）：用于 `--transformer-impl transformer_engine`、FP8、fused attention/norm/linear
- FlashAttention（可选）：用于 `--attention-backend flash` 或 `--use-flash-attn`

---

## 1) “简单 GPT”最小例子（直接看 MCore 训练 loop）

使用仓库自带的纯 MCore 演示脚本：`examples/run_simple_mcore_train_loop.py`

```bash
torchrun --standalone --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

你应该重点看它做了三件事（都在 `examples/run_simple_mcore_train_loop.py`）：
1) `parallel_state.initialize_model_parallel(tp, pp)` 初始化 MCore 进程组  
2) `TransformerConfig(...)` + `GPTModel(config=..., transformer_layer_spec=...)` 组装模型  
3) `get_forward_backward_func()` + `finalize_model_grads()` 跑 microbatch pipeline

这基本就是你之后把 Qwen 跑起来时，Megatron-LM “训练壳”在做的事情（只是封装得更完整）。

---

## 2) Megatron-Core 核心架构（你要真正掌握的“骨架”）

### 2.1 `TransformerConfig`：把“模型结构 + 并行策略 + 内核选择”收敛成一个 dataclass

入口：`megatron/core/transformer/transformer_config.py`

把它当作唯一真相源：训练脚本参数最终都会被折叠进这个 config（由 `megatron/training/arguments.py` 里的 `core_transformer_config_from_args` 等函数完成，见 `gpt_builders.py`）。

对 Qwen 适配最关键的字段（都在 `TransformerConfig` 里能找到对应项）：
- GQA：`num_query_groups`
- Norm：`normalization`（`RMSNorm` / `LayerNorm`）
- MLP：`gated_linear_unit` + `activation_func`（SwiGLU）
- Attention backend：`attention_backend`（`flash/fused/unfused/local/auto`）
- RoPE：`rotary_base/rotary_percent/rotary_interleaved/rope_scaling/...`
  - 模型侧通过 `position_embedding_type` 选择 `rope/yarn/mrope/none`（见 `megatron/core/models/gpt/gpt_model.py`）

### 2.2 `ModuleSpec`：Megatron-Core “可组合 Transformer”的关键抽象

入口：`megatron/core/transformer/spec_utils.py`

`ModuleSpec` 的作用：把“TransformerLayer 里每个子模块（norm/attention/mlp/…）用什么实现”做成可注入配置，并用 `build_module()` 动态实例化。

在 GPT 上的落点非常清晰：
- 层/块 spec 定义：`megatron/core/models/gpt/gpt_layer_specs.py`
- 组装模型时选择 spec：`gpt_builders.py`（支持 `--spec <module> <fn>` 覆盖默认 spec）
- 模型使用 spec：`megatron/core/models/gpt/gpt_model.py` → `TransformerBlock(spec=...)`

### 2.3 源码阅读路径（按目录抓关键实现）

- **TP（张量并行算子与通信）**：`megatron/core/tensor_parallel/`
  - 高频：`ColumnParallelLinear` / `RowParallelLinear`、以及 sequence-parallel mappings（如 `gather_from_sequence_parallel_region`）
- **PP（pipeline 调度）**：`megatron/core/pipeline_parallel/`
  - `schedules.py`、`p2p_communication.py` 是理解 1F1B / 交错流水的核心
  - 层偏移/分层逻辑：`megatron/core/transformer/transformer_layer.py` 的 `get_transformer_layer_offset()`
- **DP/optimizer/梯度收敛**：`megatron/core/distributed/`
  - `distributed_data_parallel.py`、`finalize_model_grads.py`、以及 `fsdp/` 子目录
- **Transformer 子模块实现**：`megatron/core/transformer/`
  - Attention：`megatron/core/transformer/attention.py`（GQA 分区、TE/FlashAttn 路由、RoPE 融合等关键逻辑）
  - MLP：`megatron/core/transformer/mlp.py`（SwiGLU fusion 也在这里）
  - Norm：`megatron/core/extensions/transformer_engine.py`（TE LayerNorm/RMSNorm）、`megatron/core/transformer/torch_norm.py`（torch fallback）
- **RoPE / YaRN**：
  - RoPE：`megatron/core/models/common/embeddings/rotary_pos_embedding.py`
  - YaRN：`megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`
  - `GPTModel` 里通过 `position_embedding_type` 选择 `rope/yarn/mrope`：`megatron/core/models/gpt/gpt_model.py`

---

## 3) TE（Transformer Engine）在 Megatron 里的真实用法（FP8/BF16/融合）

Megatron 通过 **SpecProvider** 把 TE 组件“插入”到 Transformer layer 里：
- TE spec provider：`megatron/core/extensions/transformer_engine_spec_provider.py`
- TE 模块封装：`megatron/core/extensions/transformer_engine.py`
- GPT 的 TE layer spec：`megatron/core/models/gpt/gpt_layer_specs.py` 的 `get_gpt_layer_with_transformer_engine_spec(...)`

训练脚本启用方式：

```bash
--transformer-impl transformer_engine \
--bf16 \
--attention-backend flash
```

FP8 常见组合（参数在 `megatron/training/arguments.py`，如 `--fp8-format/--fp8-recipe/...`）：

```bash
--fp8-format e4m3fn \
--fp8-recipe delayed
```

---

## 4) Sequence Parallel & Context Parallel：Megatron 的实现细节（高频踩坑点）

### 4.1 Sequence Parallel（SP）

开关：`--sequence-parallel`（定义在 `megatron/training/arguments.py`）

关键实现点：
- SP scatter/gather primitive：`megatron/core/tensor_parallel/mappings.py`
- 模型上批量打标/开关：`megatron/core/transformer/utils.py` 的 `set_model_to_sequence_parallel(...)`
- TE Linear/Norm 会携带 `sequence_parallel` 属性（`megatron/core/extensions/transformer_engine.py`）

注意：
- **MoE + TP 通常强依赖 SP**（仓库 README 也强调了 EP+TP 的约束）。

### 4.2 Context Parallel（CP）

开关：`--context-parallel-size`、`--cp-comm-type`（参数在 `megatron/training/arguments.py`）

两个必须看懂的落点：
1) **数据切分发生在训练脚本**：`pretrain_gpt.py` 的 `get_batch()` 会调用 `get_batch_on_this_cp_rank()`（来自 `megatron/training/utils.py`）
2) **RoPE/position embedding 会按 CP rank 切片**：`megatron/core/models/common/embeddings/rotary_pos_embedding.py` 里通过 `get_pos_emb_on_this_cp_rank(...)` 取本 rank 的那段

CP 还有一系列形状/整除约束（如 seq_len 与 cp_size 的关系）会在 args 校验中触发；跑长上下文时要优先对齐这些约束。

---

## 5) Qwen 适配：优先走“官方支持路径”（本仓库已包含）

### 5.1 HF → Megatron-Core checkpoint 转换（Qwen2.5：现成 loader）

本仓库已经把 Qwen2.5 当作 “Llama-like” 支持进了转换器：
- 转换入口：`tools/checkpoint/convert.py`
- HF loader：`tools/checkpoint/loader_llama_mistral.py`（`--model-size qwen2.5` 分支）
- 官方示例命令：`examples/rl/README.md`

可直接复制：

```bash
TP=8
HF_FORMAT_DIR=<PATH_TO_HF_DIR>
MEGATRON_FORMAT_DIR=<PATH_TO_MCORE_CKPT_DIR>
TOKENIZER_MODEL=$HF_FORMAT_DIR

python ./tools/checkpoint/convert.py \
  --bf16 \
  --model-type GPT \
  --loader llama_mistral \
  --saver core \
  --target-tensor-parallel-size ${TP} \
  --checkpoint-type hf \
  --load-dir ${HF_FORMAT_DIR} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --model-size qwen2.5 \
  --loader-transformer-impl transformer_engine \
  --make-vocab-size-divisible-by 128
```

你需要知道这个转换做了什么（为什么能转 Qwen2.5）：
- 仍然落到 GPT/MCore 这套权重 schema（由 `tools/checkpoint/saver_core.py` + `schema_core.py` 决定）
- loader 把 HF 的权重映射为 Megatron 的（含合并/重排等），细节在 `tools/checkpoint/loader_llama_mistral.py`

### 5.2 训练/继续预训练：本质还是 `pretrain_gpt.py` + “Qwen 风格参数”

Megatron-LM 默认就是用 MCore（`--use-mcore-models` 已 deprecated，见 `megatron/training/arguments.py`），所以你只要**别加** `--use-legacy-models` 就是在跑 MCore GPTModel。

Qwen2.5 的典型参数形状本仓库也给了参考（例如 `examples/rl/model_configs/qwen_2p5_32b.sh`），你至少需要对齐：
- `--normalization RMSNorm`
- `--swiglu`
- `--group-query-attention --num-query-groups <G>`
- `--position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0`
- **QKV bias**：`--disable-bias-linear --add-qkv-bias`

继续预训练模板（路径替换为你的）：

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 1 \
  --transformer-impl transformer_engine \
  --bf16 \
  --attention-backend flash \
  --sequence-parallel \
  --use-distributed-optimizer \
  --ckpt-format torch_dist \
  --load <MEGATRON_FORMAT_DIR> \
  --use-checkpoint-args \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model <TOKENIZER_DIR_OR_NAME> \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --disable-bias-linear \
  --add-qkv-bias \
  --group-query-attention \
  --num-query-groups 8 \
  --position-embedding-type rope \
  --rotary-base 1000000 \
  --rotary-percent 1.0 \
  --swiglu \
  --data-path <IDX_PREFIX_OR_BLEND_CFG> \
  --split 949,50,1 \
  --micro-batch-size 1 \
  --global-batch-size 1024 \
  --lr 1e-4 \
  --train-iters 1000 \
  --log-interval 10 \
  --save-interval 1000 \
  --save <SAVE_DIR>
```

数据部分后续会用 `tools/preprocess_data.py` 生成 `.bin/.idx`，然后 `--data-path` 指向 prefix 或 blended 配置；`pretrain_gpt.py` 内部使用 `BlendedMegatronDatasetBuilder` 组装（见 `megatron/core/datasets/`）。

### 5.3 Qwen tokenizer 的坑（本仓库已明确提示）

`examples/rl/model_configs/qwen_2p5_32b.sh` 明确提示：
> “Original Qwen model uses a wrong padding_id token. unsloth tokenizer fixes it.”

继续预训练/pack 数据时务必检查：
- `pad_token_id` 是否正确（否则 loss_mask / padding 行为会异常）
- 必要时直接用类似 `unsloth/Qwen2.5-XXB` 的 tokenizer（如脚本所示）

---

## 6) 如果你要“从零自定义适配 Qwen”（官方支持不够 / 你要上 Qwen3/MoE）

在 Megatron-Core 里，Qwen（Dense）一般不需要“写一个新模型类”，更推荐两条路：

### 路线 A（推荐）：不改模型类，只写/改 spec + args→config 映射

1) 复制/改造 GPT layer spec：从 `megatron/core/models/gpt/gpt_layer_specs.py` 入手  
2) 用 `--spec` 注入你自己的 spec（参数定义在 `megatron/training/arguments.py`）：

```bash
--spec my_qwen_specs qwen_layer_spec
```

其中 `qwen_layer_spec()` 返回一个 `ModuleSpec`（参考 `get_gpt_layer_with_transformer_engine_spec` 的写法）。

你要改的点对应源码位置：
- GQA/注意力形状与分区：`megatron/core/transformer/attention.py`
- RoPE/YaRN：`megatron/core/models/common/embeddings/rotary_pos_embedding.py` + `.../yarn_rotary_pos_embedding.py`
- SwiGLU：`megatron/core/transformer/mlp.py`
- RMSNorm：TE 路径 `megatron/core/extensions/transformer_engine.py`，torch fallback `megatron/core/transformer/torch_norm.py`
- Flash attention 路由：`megatron/core/transformer/attention.py`（结合 `--attention-backend`）

### 路线 B：确实要加“新模型”（比如 Qwen3 MoE/特殊 block）

- 参考 GPT 组装方式：`gpt_builders.py` + `megatron/core/models/gpt/gpt_model.py`
- 新增 builder 并复用入口模式：`model_provider.py`
- MoE 优先读：`megatron/core/transformer/moe/README.md`，以及 GPT 的 MoE spec：`megatron/core/models/gpt/moe_module_specs.py`

---

## 7) 多节点启动（torchrun / slurm 模板）

torchrun 多节点最小模板：

```bash
torchrun \
  --nnodes=$NNODES --node_rank=$NODE_RANK \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  pretrain_gpt.py <你的参数...>
```

slurm 风格可以直接参考仓库脚本（multimodal qwen 示例）：`examples/multimodal/nvlm/pretrain_qwen20_72b_internvit_6b.sh`

---

## 下一步（建议你给我这些信息，我可以按你的目标生成可直接跑的命令）

请告诉我：
1) 目标型号：`Qwen2.5-*` / `Qwen3-*`，Dense 还是 MoE  
2) 目标 GPU：单机/多机，GPU 型号与数量，网络（IB/以太）  
3) 目标 `seq_len`：8k/32k/128k，是否需要 CP  
4) 训练目标：从 HF checkpoint 继续预训练 / 从零预训练

我会给你：
- checkpoint 转换命令（含并行度选择与权重映射校验要点）
- 训练命令（TP/PP/CP/EP + TE/FP8 + recompute + distributed optimizer 组合）
- 常见坑排雷清单（OOM、parallelism mismatch、load fail、tokenizer/pad、长上下文约束）


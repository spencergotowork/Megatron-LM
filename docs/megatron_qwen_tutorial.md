# Megatron-LM 深入教程：从核心架构到 Qwen 适配

本文档为有分布式训练经验的 AI Infra 工程师提供 Megatron-LM 的深入指导，重点介绍如何将 Qwen 系列模型适配到 Megatron 中进行高效分布式预训练。

---

## 目录

- [第一部分：Megatron-Core 核心架构](#第一部分megatron-core-核心架构)
- [第二部分：Qwen 官方支持情况](#第二部分qwen-官方支持情况)
- [第三部分：Qwen 模型配置详解](#第三部分qwen-模型配置详解)
- [第四部分：model_provider 和 gpt_builder](#第四部分model_provider-和-gpt_builder)
- [第五部分：Qwen 适配完整指南](#第五部分qwen-适配完整指南)
- [第六部分：数据准备](#第六部分数据准备)
- [第七部分：常见优化选项](#第七部分常见优化选项)
- [第八部分：多节点启动](#第八部分多节点启动)
- [第九部分：调试和常见问题](#第九部分调试和常见问题)

---

## 第一部分：Megatron-Core 核心架构

### 1.1 整体架构概览

```
megatron/
├── core/                          # 核心库（推荐使用）
│   ├── models/                    # 模型实现
│   │   ├── gpt/                   # GPT 模型 ← Qwen 基于此
│   │   │   ├── gpt_model.py       # GPTModel 类
│   │   │   └── gpt_layer_specs.py # 层规范定义
│   │   └── common/embeddings/     # 共享嵌入（RoPE 等）
│   ├── transformer/               # Transformer 核心组件
│   │   ├── transformer_config.py  # TransformerConfig
│   │   ├── attention.py           # 注意力机制
│   │   ├── mlp.py                 # MLP 层
│   │   └── spec_utils.py          # ModuleSpec
│   ├── tensor_parallel/           # 张量并行
│   ├── pipeline_parallel/         # 流水线并行
│   └── distributed/               # 分布式训练
├── training/                      # 训练参考实现
└── legacy/                        # 旧版（不推荐）
```

### 1.2 TransformerConfig：中心配置类

`TransformerConfig` 是所有 Transformer 模型的配置中心，继承自 `ModelParallelConfig`。

**关键配置字段**（`megatron/core/transformer/transformer_config.py:34`）：

```python
@dataclass
class TransformerConfig(ModelParallelConfig):
    # 模型架构
    num_layers: int = 0                          # Transformer 层数
    hidden_size: int = 0                         # 隐藏层维度
    num_attention_heads: int = 0                 # 注意力头数
    num_query_groups: Optional[int] = None       # GQA 的 KV 头数（None=MHA）
    ffn_hidden_size: Optional[int] = None        # FFN 隐藏层（默认 4*hidden_size）
    kv_channels: Optional[int] = None            # KV 投影维度

    # 归一化
    normalization: str = "LayerNorm"             # "LayerNorm" 或 "RMSNorm"
    layernorm_epsilon: float = 1e-5

    # 激活函数
    gated_linear_unit: bool = False              # SwiGLU
    activation_func: Callable = F.gelu           # 激活函数

    # 偏置
    add_bias_linear: bool = True                 # 线性层偏置
    add_qkv_bias: bool = False                   # 仅 QKV 偏置（Qwen2.5）

    # 位置编码 - RoPE 相关
    rotary_interleaved: bool = False             # RoPE 交错模式

    # QK LayerNorm（Qwen3 需要）
    qk_layernorm: bool = False

    # MoE 相关
    num_moe_experts: Optional[int] = None
    moe_router_topk: int = 2
    moe_grouped_gemm: bool = False

    # 混合精度
    fp8: Optional[str] = None                    # FP8 训练
    bf16: bool = False
    fp16: bool = False
```

**Qwen 系列的关键配置**：
- Qwen2.5: `normalization="RMSNorm"`, `gated_linear_unit=True` (SwiGLU), `add_bias_linear=False`, `add_qkv_bias=True`
- Qwen3: 额外增加 `qk_layernorm=True`

### 1.3 ModuleSpec：模块规范系统

`ModuleSpec` 是 Megatron-Core 的核心设计模式，用于声明式地定义模块结构（`megatron/core/transformer/spec_utils.py:9`）：

```python
@dataclass
class ModuleSpec:
    module: Union[Tuple, type]      # 模块类或导入路径
    params: dict = field(default_factory=lambda: {})  # 初始化参数
    submodules: type = None         # 子模块规范
```

**使用示例**：

```python
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules

# 定义注意力层规范
attention_spec = ModuleSpec(
    module=SelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=SelfAttentionSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=DotProductAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=IdentityOp,  # 或 RMSNorm for Qwen3
        k_layernorm=IdentityOp,
    ),
)

# 构建模块
attention = build_module(attention_spec, config=config, layer_number=1)
```

### 1.4 GPT Layer Specs：层规范工厂

`gpt_layer_specs.py` 提供了预定义的层规范工厂函数（`megatron/core/models/gpt/gpt_layer_specs.py`）：

```python
# 1. Transformer Engine 后端（推荐，支持 FP8）
def get_gpt_layer_with_transformer_engine_spec(
    num_experts=None,
    moe_grouped_gemm=False,
    qk_layernorm=False,          # Qwen3 设置为 True
    multi_latent_attention=False, # MLA (DeepSeek)
    moe_use_legacy_grouped_gemm=False,
    qk_l2_norm=False,
    use_te_op_fuser=False,
    use_kitchen=False,
    use_te_activation_func=False,
) -> ModuleSpec

# 2. 纯 PyTorch 后端
def get_gpt_layer_local_spec(
    num_experts=None,
    qk_layernorm=False,
    normalization="LayerNorm",    # Qwen 用 "RMSNorm"
    ...
) -> ModuleSpec

# 3. MoE 模型的 block spec
def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    ...
) -> TransformerBlockSubmodules
```

**TransformerLayer 子模块结构**（`megatron/core/transformer/transformer_layer.py`）：

```python
@dataclass
class TransformerLayerSubmodules:
    input_layernorm: type = None          # 输入归一化
    self_attention: ModuleSpec = None     # 自注意力
    self_attn_bda: type = None            # bias-dropout-add
    pre_mlp_layernorm: type = None        # MLP 前归一化
    mlp: ModuleSpec = None                # MLP/MoE
    mlp_bda: type = None
```

---

## 第二部分：Qwen 官方支持情况

### 2.1 当前支持状态

**好消息**：Megatron-LM 已经原生支持 Qwen 系列模型！

文档明确说明（`docs/llama_mistral.md:421`）：
> "Many models such as Yi-34B and **Qwen2.x** use the Llama architecture and may be converted from HuggingFace to Megatron using the commands in Llama-3.x."

**已有的 Qwen 配置文件**：
- `examples/rl/model_configs/qwen3_8b.sh` - Qwen3 8B RL 训练配置
- `examples/rl/model_configs/qwen_2p5_32b.sh` - Qwen2.5 32B 配置
- `examples/post_training/modelopt/conf/Qwen/` - 后训练配置

### 2.2 Qwen 与 Llama 架构差异

| 特性 | Llama | Qwen2.5 | Qwen3 |
|------|-------|---------|-------|
| Normalization | RMSNorm | RMSNorm | RMSNorm |
| Activation | SwiGLU | SwiGLU | SwiGLU |
| Attention | GQA | GQA | GQA |
| Position Embedding | RoPE | RoPE | RoPE |
| Bias in Linear | 无 | 无 | 无 |
| QKV Bias | 无 | **有** | **有** |
| QK LayerNorm | 无 | 无 | **有** |
| Vocab Size | 128256 | 151936/152064 | 151936 |
| RoPE Base | 500000 | 1000000 | 1000000 |

### 2.3 Checkpoint 转换

使用 `tools/checkpoint/convert.py`，通过 `llama_mistral` loader 转换（`examples/rl/README.md:36`）：

```bash
# Qwen2.5 HF -> Megatron
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

**支持的 model-size 值**（查看 `tools/checkpoint/loader_llama_mistral.py`）：
- `qwen2.5` - Qwen2.5 系列
- `llama3` - Llama3 系列
- `mistral` - Mistral 系列

---

## 第三部分：Qwen 模型配置详解

### 3.1 Qwen2.5-7B 完整参数配置

从 `examples/post_training/modelopt/conf/Qwen/Qwen2.5-7B-Instruct.sh` 提取：

```bash
MODEL_ARGS=" \
    # 精度和基础设置
    --bf16 \
    --use-mcore-models \
    --transformer-impl transformer_engine \

    # 模型架构
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --kv-channels 128 \
    --max-position-embeddings 32768 \

    # GQA
    --group-query-attention \
    --num-query-groups 4 \

    # 归一化和激活
    --normalization RMSNorm \
    --swiglu \

    # Bias 设置 - Qwen2.5 特有
    --disable-bias-linear \
    --add-qkv-bias \

    # RoPE
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --no-rope-fusion \

    # 词表
    --tokenizer-type HuggingFaceTokenizer \
    --padded-vocab-size 152064 \
    --make-vocab-size-divisible-by 1 \

    # 其他
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --no-bias-swiglu-fusion \
"
```

### 3.2 Qwen3-8B 完整参数配置

从 `examples/rl/model_configs/qwen3_8b.sh` 提取：

```bash
MODEL_OPTIONS="\
    # 模型架构
    --num-layers 36 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --max-position-embeddings 40960 \

    # GQA
    --group-query-attention \
    --num-query-groups 8 \

    # 归一化
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \

    # Qwen3 特有：QK LayerNorm
    --qk-layernorm \

    # 激活和 Bias
    --swiglu \
    --disable-bias-linear \

    # RoPE
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --use-rotary-position-embeddings \

    # 词表
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3-8B \
    --vocab-size 151936 \
    --make-vocab-size-divisible-by 128 \

    # 其他
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
"
```

### 3.3 Qwen 各版本配置速查表

| 参数 | Qwen2.5-7B | Qwen2.5-32B | Qwen3-8B | Qwen3-32B |
|------|------------|-------------|----------|-----------|
| num_layers | 28 | 64 | 36 | 64 |
| hidden_size | 3584 | 5120 | 4096 | 5120 |
| ffn_hidden_size | 18944 | 27648 | 12288 | 25600 |
| num_attention_heads | 28 | 40 | 32 | 40 |
| num_query_groups | 4 | 8 | 8 | 8 |
| kv_channels | 128 | 128 | 128 | 128 |
| vocab_size | 152064 | 152064 | 151936 | 151936 |
| rotary_base | 1000000 | 1000000 | 1000000 | 1000000 |
| qk_layernorm | No | No | **Yes** | **Yes** |
| add_qkv_bias | **Yes** | **Yes** | No | No |

### 3.4 代码层面的配置映射

```python
# 从命令行参数创建 TransformerConfig
# megatron/training/arguments.py -> core_transformer_config_from_args()

from megatron.training.arguments import core_transformer_config_from_args

def get_qwen25_config(args):
    """Qwen2.5 配置示例"""
    config = core_transformer_config_from_args(args)

    # 验证关键配置
    assert config.normalization == "RMSNorm"
    assert config.gated_linear_unit == True  # --swiglu
    assert config.add_bias_linear == False   # --disable-bias-linear
    assert config.add_qkv_bias == True       # --add-qkv-bias

    return config
```

---

## 第四部分：model_provider 和 gpt_builder

### 4.1 model_provider 函数

`model_provider.py` 是模型构建的入口点：

```python
# model_provider.py:24
def model_provider(
    model_builder: Callable,        # 构建函数，如 gpt_builder
    pre_process=True,               # 是否包含 embedding（PP 首 rank）
    post_process=True,              # 是否包含 output layer（PP 末 rank）
    vp_stage: Optional[int] = None, # 虚拟流水线阶段
    config=None,
    pg_collection=None,
) -> Union[GPTModel, MambaModel]:
    ...
```

### 4.2 gpt_builder 详解

`gpt_builders.py` 展示了如何构建 GPT 模型：

```python
# gpt_builders.py:24
def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    # 1. 加载配置
    if config is None:
        config = core_transformer_config_from_args(args)

    # 2. 选择层规范
    use_te = args.transformer_impl == "transformer_engine"

    if args.num_experts:
        # MoE 模型使用 block spec
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=use_te,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            vp_stage=vp_stage,
        )
    else:
        # Dense 模型使用 layer spec
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            qk_layernorm=args.qk_layernorm,  # Qwen3 需要
            ...
        )

    # 3. 构建模型
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        ...
    )

    return model
```

---

## 第五部分：Qwen 适配完整指南

### 5.1 方案一：使用官方支持（推荐）

#### Step 1: 下载 HuggingFace Checkpoint

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen2.5-7b-hf

# 或使用 Python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="./qwen2.5-7b-hf")
```

#### Step 2: 转换到 Megatron 格式

```bash
TP=2  # 根据你的 GPU 数量调整
PP=1

python ./tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size ${PP} \
    --checkpoint-type hf \
    --load-dir ./qwen2.5-7b-hf \
    --save-dir ./qwen2.5-7b-megatron \
    --tokenizer-model ./qwen2.5-7b-hf \
    --model-size qwen2.5 \
    --loader-transformer-impl transformer_engine \
    --make-vocab-size-divisible-by 128
```

#### Step 3: 编写训练脚本

创建 `train_qwen.sh`：

```bash
#!/bin/bash

# 基础设置
CHECKPOINT_PATH="./qwen2.5-7b-megatron"
TOKENIZER_MODEL="./qwen2.5-7b-hf"
DATA_PATH="your_data_prefix"
TP=2
PP=1

# Qwen2.5-7B 模型参数
COMMON_ARGS="\
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --bf16 \
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --kv-channels 128 \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --group-query-attention \
    --num-query-groups 4 \
    --normalization RMSNorm \
    --swiglu \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --padded-vocab-size 152064 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
"

# 训练参数
TRAINING_ARGS="\
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --train-iters 10000 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-warmup-iters 100 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
"

# 优化参数
OPTIMIZATION_ARGS="\
    --use-distributed-optimizer \
    --sequence-parallel \
    --recompute-granularity selective \
    --recompute-modules core_attn \
"

# 数据参数
DATA_ARGS="\
    --data-path ${DATA_PATH} \
    --split 99,1,0 \
"

# Checkpoint 参数
CHECKPOINT_ARGS="\
    --load ${CHECKPOINT_PATH} \
    --save ${CHECKPOINT_PATH} \
    --save-interval 500 \
    --ckpt-format torch_dist \
    --finetune \
"

torchrun --nproc_per_node=2 \
    pretrain_gpt.py \
    ${COMMON_ARGS} \
    ${TRAINING_ARGS} \
    ${OPTIMIZATION_ARGS} \
    ${DATA_ARGS} \
    ${CHECKPOINT_ARGS}
```

### 5.2 方案二：自定义 Qwen Builder

如果你需要自定义 Qwen 的实现，可以创建专门的 builder：

```python
# qwen_builder.py
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_local_spec,
)
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

def qwen_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Qwen 专用 builder"""

    if config is None:
        config = core_transformer_config_from_args(args)

    # 验证 Qwen 必需的配置
    assert config.normalization == "RMSNorm", "Qwen requires RMSNorm"
    assert config.gated_linear_unit, "Qwen requires SwiGLU"

    use_te = args.transformer_impl == "transformer_engine"

    # 获取层规范
    if use_te:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts,
            moe_grouped_gemm=args.moe_grouped_gemm,
            qk_layernorm=args.qk_layernorm,  # Qwen3 需要 True
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        )
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=args.num_experts,
            qk_layernorm=args.qk_layernorm,
            normalization=args.normalization,
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

    return model
```

然后在 `pretrain_gpt.py` 中使用：

```python
from qwen_builder import qwen_builder

pretrain(
    train_valid_test_datasets_provider,
    partial(model_provider, qwen_builder),  # 使用自定义 builder
    ModelType.encoder_or_decoder,
    forward_step,
)
```

### 5.3 Qwen MoE 模型适配

对于 Qwen MoE 模型，需要额外配置：

```bash
# Qwen MoE 特有参数
MOE_ARGS="\
    --num-experts 64 \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --expert-model-parallel-size 8 \
"
```

---

## 第六部分：数据准备

### 6.1 创建 Indexed Dataset

```bash
# 1. 准备 JSONL 格式数据
# data.jsonl 每行格式: {"text": "..."}

# 2. 使用 preprocess_data.py 转换
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix my_dataset \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ./qwen2.5-7b-hf \
    --workers 32 \
    --append-eod

# 生成文件:
# my_dataset_text_document.bin
# my_dataset_text_document.idx
```

### 6.2 Blended Dataset 配置

```bash
# 多数据集混合
DATA_PATH="\
    0.7 dataset1_text_document \
    0.2 dataset2_text_document \
    0.1 dataset3_text_document \
"

# 或使用配置文件
--data-path-config data_blend.yaml
```

### 6.3 SFT 数据格式

```bash
# SFT 模式
--sft \
--data-path sft_data.jsonl \

# sft_data.jsonl 格式:
# {"input": "user message", "output": "assistant response"}
```

---

## 第七部分：常见优化选项

### 7.1 Selective Recompute（选择性重计算）

```bash
--recompute-granularity selective \
--recompute-modules core_attn \  # 只重计算注意力
```

可选模块：
- `core_attn` - 核心注意力（推荐）
- `moe_act` - MoE 激活函数
- `layernorm` - LayerNorm
- `mlp` - MLP 层
- `moe` - 完整 MoE 层
- `shared_experts` - 共享专家

### 7.2 Distributed Optimizer

```bash
--use-distributed-optimizer \  # ZeRO-1 风格
--overlap-grad-reduce \        # 梯度通信重叠
--overlap-param-gather \       # 参数聚合重叠
```

### 7.3 Sequence Parallel

```bash
--sequence-parallel \  # 与 TP 配合使用，减少激活内存
```

### 7.4 FP8 训练（需要 Hopper GPU）

```bash
--fp8 e4m3 \
--fp8-recipe tensorwise \
--transformer-impl transformer_engine \
```

### 7.5 Context Parallel（超长序列）

```bash
--context-parallel-size 2 \
--cp-comm-type p2p \  # 或 all_gather, a2a
```

### 7.6 Flash Attention

```bash
--attention-backend flash \
```

---

## 第八部分：多节点启动

### 8.1 torchrun 方式

```bash
# 节点 0
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    pretrain_gpt.py \
    ${ALL_ARGS}

# 节点 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    pretrain_gpt.py \
    ${ALL_ARGS}
```

### 8.2 SLURM 方式

```bash
#!/bin/bash
#SBATCH --job-name=qwen_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00

srun --container-image=nvcr.io/nvidia/pytorch:25.01-py3 \
    bash -c "
    torchrun \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
        --master_port=29500 \
        pretrain_gpt.py \
        ${ALL_ARGS}
    "
```

### 8.3 并行度配置建议

| 模型规模 | TP | PP | DP | 推荐 GPU 数 |
|---------|----|----|----|-----------|
| 7B | 1-2 | 1 | N | 8+ |
| 32B | 4-8 | 1-2 | N | 32+ |
| 72B | 8 | 2-4 | N | 64+ |
| 405B | 8 | 8-16 | N | 512+ |

---

## 第九部分：调试和常见问题

### 9.1 OOM 排查

```bash
# 1. 减小 micro-batch-size
--micro-batch-size 1

# 2. 启用重计算
--recompute-granularity selective

# 3. 检查 TP/PP 配置是否合理
# 对于 7B 模型，TP=2 通常足够

# 4. 使用 distributed optimizer
--use-distributed-optimizer

# 5. 减小序列长度
--seq-length 2048
```

### 9.2 Checkpoint 加载失败

```bash
# 检查 TP/PP 是否匹配
--tensor-model-parallel-size ${TP}  # 必须与转换时一致
--pipeline-model-parallel-size ${PP}

# 使用 checkpoint args
--use-checkpoint-args
--no-load-optim  # 微调时不加载优化器状态
--no-load-rng

# 检查 checkpoint 格式
--ckpt-format torch_dist  # 或 torch
```

### 9.3 数值差异检查

```python
# 对比 Megatron 和 HuggingFace 输出
# examples/inference/llama_mistral/huggingface_reference.py
python examples/inference/llama_mistral/huggingface_reference.py \
    --model_path ./qwen2.5-7b-hf \
    --prompt "Hello, world!"
```

### 9.4 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `AssertionError: num_attention_heads must be divisible by tensor_model_parallel_size` | TP 配置不合理 | 调整 TP 使其能整除 attention heads |
| `RuntimeError: CUDA out of memory` | 显存不足 | 减小 batch size，启用重计算 |
| `KeyError: 'model.language_model.encoder.layers.0.self_attention.query_key_value.weight'` | Checkpoint 格式不匹配 | 检查转换参数，确保 TP/PP 一致 |
| `ValueError: Tokenizer not found` | Tokenizer 路径错误 | 检查 `--tokenizer-model` 路径 |

### 9.5 性能调优检查清单

- [ ] 使用 `--transformer-impl transformer_engine`
- [ ] 启用 `--sequence-parallel`
- [ ] 启用 `--use-distributed-optimizer`
- [ ] 配置合理的 `--recompute-granularity selective`
- [ ] 使用 `--attention-backend flash`
- [ ] 检查 `--micro-batch-size` 是否最大化利用显存
- [ ] 启用 `--overlap-grad-reduce` 和 `--overlap-param-gather`

---

## 附录 A：完整训练脚本示例

### Qwen2.5-7B 继续预训练

```bash
#!/bin/bash

set -e

# 环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

# 路径配置
MEGATRON_PATH="/path/to/Megatron-LM"
CHECKPOINT_PATH="/path/to/qwen2.5-7b-megatron"
TOKENIZER_MODEL="/path/to/qwen2.5-7b-hf"
DATA_PATH="/path/to/data_text_document"
OUTPUT_PATH="/path/to/output"

# 并行配置
TP=2
PP=1
WORLD_SIZE=8

# 训练配置
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
SEQ_LENGTH=4096
TRAIN_ITERS=10000

cd ${MEGATRON_PATH}

torchrun \
    --nproc_per_node=${WORLD_SIZE} \
    pretrain_gpt.py \
    \
    `# 并行配置` \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    \
    `# 模型配置 - Qwen2.5-7B` \
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --kv-channels 128 \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings 32768 \
    --group-query-attention \
    --num-query-groups 4 \
    --normalization RMSNorm \
    --swiglu \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    \
    `# Tokenizer` \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --padded-vocab-size 152064 \
    --make-vocab-size-divisible-by 1 \
    \
    `# 训练参数` \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-warmup-iters 100 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    \
    `# 优化` \
    --use-distributed-optimizer \
    --sequence-parallel \
    --recompute-granularity selective \
    --recompute-modules core_attn \
    --attention-backend flash \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    \
    `# 数据` \
    --data-path ${DATA_PATH} \
    --split 99,1,0 \
    \
    `# Checkpoint` \
    --load ${CHECKPOINT_PATH} \
    --save ${OUTPUT_PATH} \
    --save-interval 500 \
    --ckpt-format torch_dist \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    \
    `# 日志` \
    --log-interval 10 \
    --tensorboard-dir ${OUTPUT_PATH}/tensorboard \
    --log-throughput
```

---

## 附录 B：参考资源

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-Core 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
- [Qwen 模型系列](https://huggingface.co/Qwen)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

---

*文档生成时间：2025年12月*
*基于 Megatron-LM 主分支最新代码*

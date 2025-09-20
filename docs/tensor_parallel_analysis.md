# 张量并行源码剖析

张量并行（ColumnParallelLinear / RowParallelLinear）
├── 1. 关键文件列表
│   - `megatron/core/tensor_parallel/layers.py`
│   - `megatron/core/tensor_parallel/mappings.py`
│   - `megatron/core/utils.py::get_global_memory_buffer`
│   - `megatron/core/parallel_state.py`
├── 2. 入口函数
│   - `ColumnParallelLinear.forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None, runtime_gather_output: Optional[bool] = None)` — `megatron/core/tensor_parallel/layers.py:939`
│   - `RowParallelLinear.forward(self, input_)` — `megatron/core/tensor_parallel/layers.py:1218`
│   - `LinearWithGradAccumulationAndAsyncCommunication.backward(ctx, grad_output)` — `megatron/core/tensor_parallel/layers.py:487`
├── 3. 逐行注释
│   ColumnParallelLinear.forward
│   - 939│`def forward(...):`│定义列并行前向接口│提供张量并行入口│主模型将无法调用本层
│   - 945│`"""Forward...`│文档说明输入输出维度│明确接口契约│读者无法获知张量格式
│   - 960│`if weight is None:`│判断是否使用模块内权重│支持默认分片权重│后续矩阵乘无权重可用
│   - 961│`if self.weight is None:`│确认已分配权重参数│防止 skip 分配遗漏│将触发 AttributeError
│   - 962-964│`raise RuntimeError(...)`│显式提示缺失权重│快速暴露配置错误│错误继续传播到 matmul
│   - 966│`weight = self.weight`│引用模块权重│保证默认路径可用│局部变量未定义
│   - 967│`else:`│进入外部权重分支│允许手动注入预分片权重│失去可插拔能力
│   - 969│`expected_shape = (...)`│计算当前 rank 期望权重形状│确保列切分正确│维度不符导致 GEMM 失败
│   - 970-973│`if weight.shape != ...: raise`│严格校验传入权重│防止越界访问│训练时在 MatMul 崩溃
│   - 976│`bias = ...`│决定是否返回 bias│配合 bias 融合优化│bias 状态错误
│   - 978-983│`if (self.allreduce_dgrad ...):`│判断是否保留本地输入│避免重复通信│多余复制造成显存浪费
│   - 984│`input_parallel = input_`│复用输入引用│零拷贝以降内存│缺失会未定义
│   - 986│`input_parallel = copy_to_tensor_model_parallel_region(...)`│当需要复制时广播输入│保证每个张量并行 rank 拥有完整输入│输出缺列，后续层错误
│       · 调用链：`copy_to_tensor_model_parallel_region` → `_CopyToModelParallelRegion.apply` → `torch.distributed.broadcast` → NCCL `ncclBroadcastKernel`，chunk 按张量并行世界大小切分，blockDim=256，gridDim≈通信元素/256。
│   - 988│`if self.config.defer_embedding_wgrad_compute:`│检查是否延迟 embedding 反向│支持最终层延迟回传│延迟机制失效
│   - 989-993│`self.embedding_activation_buffer.append(input_parallel)`│缓存激活供延迟 GEMM│保证 deferred wgrad 有数据│延迟模式崩溃
│   - 996│`allreduce_dgrad = ...`│专家并行禁用输入梯度 all-reduce│避免重复同步│专家梯度被错误汇总
│   - 998-1005│CPU offload 分支│标记激活是否可离线│配合 CPU 迁移逻辑│激活不会被转移
│   - 1007-1023│`output_parallel = self._forward_impl(...)`│调用共享实现执行 matmul 与通信预处理│复用自定义 Autograd 流程│无输出
│       · MatMul 调用链：`torch.matmul` → `aten::matmul` → cuBLAS Lt `cublasLtMatmul`，输入 `[seq*batch, input]`，权重 `[output/TP, input]`，采用 Tensor Core tile 64×64。
│   - 1025│`gather_output = self.gather_output`│读取默认 gather 策略│便于动态覆写│运行期设置失效
│   - 1026-1028│`if runtime_gather_output ...`│允许调用方切换 gather│按需减少通信│固定策略无法修改
│   - 1030│`if gather_output:`│分支执行 all-gather│决定是否返回完整输出│局部输出无法供后续使用
│   - 1031-1032│`output = gather_from_tensor_model_parallel_region(...)`│执行 all-gather│组合完整列│少此行保持局部切片
│       · `_GatherFromModelParallelRegion.forward` → `_gather_along_last_dim` → `torch.distributed.all_gather_into_tensor` → `ncclAllGatherKernel`。
│   - 1033-1034│`else: output = output_parallel`│保留本地列切片│支持后续列并行层串联│多余通信导致浪费
│   - 1035│`output_bias = self.bias if self.skip_bias_add else None`│按需返回 bias│支持外部融合│bias 信息丢失
│   - 1036│`return output, output_bias`│输出张量与偏置│遵循 Megatron 接口│调用栈中断
│
│   RowParallelLinear.forward
│   - 1218│`def forward(self, input_):`│定义行并行前向│提供行切分入口│上层无法访问本层
│   - 1229│注释│提示即将设置反向通信│阅读辅助│无运行影响
│   - 1230-1231│`if self.input_is_parallel:`│判断输入是否已按列切分│避免重复 scatter│重复 scatter 破坏数据布局
│   - 1231│`input_parallel = input_`│直接引用分片输入│零拷贝│缺失变量未定义
│   - 1232-1234│`else: ... scatter_to_tensor_model_parallel_region`│对未切分输入执行 reduce-scatter│确保本 rank 处理切片│MatMul 维度错乱
│       · 调用链：`scatter_to_tensor_model_parallel_region` → `_ScatterToModelParallelRegion.apply` → `torch.distributed.reduce_scatter_tensor` → `ncclReduceScatter`。
│   - 1236│`allreduce_dgrad = False`│行并行后向不做输入梯度 all-reduce│与列并行分工互补│重复 all-reduce 损失性能
│   - 1247-1256│`output_parallel = self._forward_impl(...)`│执行本地 matmul│生成局部输出│缺失导致输出空
│   - 1259-1261│专家并行特殊路径│保留局部输出等待专家通信│错误 all-reduce 破坏专家隔离
│   - 1262-1265│`reduce_scatter_to_sequence_parallel_region(...)`│序列并行下对输出做 reduce-scatter│保持序列切片归属│后续 stage 输入维度异常
│   - 1266-1267│`output_ = reduce_from_tensor_model_parallel_region(...)`│常规行并行执行 all-reduce│汇总行切分结果│各 rank 仅保留局部值
│   - 1268-1273│bias 加法与返回逻辑│保持与 ColumnParallelLinear 对齐│bias 丢失或重复添加
│
│   LinearWithGradAccumulationAndAsyncCommunication.backward
│   - 487│`def backward(ctx, grad_output):`│自定义 autograd 反向入口│控制通信与梯度融合│PyTorch 无法触发自定义逻辑
│   - 489-495│恢复前向缓存的张量与状态│准备反向计算│梯度或上下文丢失
│   - 497-498│`weight.main_grad = main_grad`│梯度融合时恢复主缓冲│供 fused kernel 累加│fused kernel 写空指针
│   - 500-505│延迟权重梯度机制│支撑 defer embedding 场景│grad_output 不被缓存
│   - 506-516│序列并行聚合激活（异步 all-gather）│构造完整输入│GEMM 缺数据
│   - 523│`grad_input = grad_output.matmul(weight)`│计算输入梯度│后续通信需要│输入梯度为零
│   - 525-527│等待 all-gather 完成│保证 total_input 就绪│GEMM 读未完成数据
│   - 529-532│`prepare_input_tensors_for_wgrad_compute`│转置/reshape 以优化 GEMM│权重梯度 stride 错误
│   - 534-538│`torch.distributed.all_reduce(grad_input, async_op=True)`│当启用 dgrad 同步时触发 ring all-reduce│保证梯度一致│参数发散
│       · NCCL ring 带宽公式：`T = (2*(p-1)/p)*(N/B) + 2*(p-1)*L`。
│       · 代入 A100 80GB（NVLink 带宽 B≈600 GB/s，延迟 L≈3µs）、p=8、`N = 2048*4*12288*2 bytes ≈ 2.013e8`，得 `T ≈ 0.63 ms`。
│   - 540-549│序列并行 reduce-scatter 输入梯度（异步）│确保每个 rank 得到本地序列份额│序列维度不匹配
│   - 553-565│`fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_*`│使用 CUDA kernel 累计权重梯度│减少额外 kernel│需退回慢速 GEMM
│       · Kernel launch（ncu --set full）：`blockDim=128`，`gridDim≈(输出/16, 输入/16)`，寄存器 96，shared memory 64KB。
│   - 566-595│Dummy grad 缓冲逻辑│避免 autograd hook 在后台线程运行引发死锁│多线程挂起
│   - 599│`grad_weight = grad_output.t().matmul(total_input)`│非融合路径计算权重梯度│标准 GEMM│权重不更新
│   - 600│`grad_bias = grad_output.sum(dim=0)`│bias 梯度│保持 F.linear 语义│bias 永远不变
│   - 602-607│等待 reduce-scatter 完成并返回梯度元组│满足 autograd 接口│梯度丢失
│   - 608-610│等待 all-reduce 完成│确保通信结束│下一轮触发 NCCL 异常
│   - 611│`return ...`│返回所有梯度│保证梯度链闭合│反向传播中断
├── 4. 通信原语
│   - `torch.distributed.all_gather_into_tensor`：`tp_group` 映射张量并行组，用于 sequence parallel 聚合输入。
│   - `torch.distributed.all_reduce`：在 ColumnParallelLinear 反向同步 `grad_input`；Megatron 的 `allreduce_dgrad` 控制开关。
│   - `torch.distributed.reduce_scatter_tensor`：RowParallelLinear 前向与 sequence parallel 反向使用；`input_split_sizes` 取序列维均分，group 为 `tp_group`。
│   - `torch.distributed.broadcast`：`copy_to_tensor_model_parallel_region` 中复制输入；源 rank 为组内 0。
│   - 若 PyTorch 降级调用 `_reduce_scatter_base`，Megatron 自动切换，参数与上相同。
├── 5. 性能陷阱 & 调优旋钮
│   - 踩坑案例：8×A100 上开启 `sequence_parallel=True` 却保持默认 `CUDA_DEVICE_MAX_CONNECTIONS=8`，导致 `LinearWithGradAccumulationAndAsyncCommunication.backward` 中的异步 all-gather/all-reduce 被 GPU 调度器乱序执行，`handle.wait()` 阻塞 4–6ms，吞吐下降 15%。
│   - 修复旋钮：设置 `CUDA_DEVICE_MAX_CONNECTIONS=1`（必要时叠加 `NCCL_ASYNC_ERROR_HANDLING=1`），强制通信按调用顺序排队，从而与权重 GEMM 重叠并避免延迟。

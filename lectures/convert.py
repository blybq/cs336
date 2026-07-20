import ast
import json
import os
import re

# Reference map for variables passed to link()
reference_map = {
    "deepseek_v3_2_2025": "[DeepSeek-V3.2 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2512.02556)",
    "adagrad_2011": "[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (Duchi et al., 2011)](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)",
    "nemotron_3_super_2026": "[Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning (NVIDIA, 2026)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)",
    "Tokenizer": "[Tokenizer](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py#L256)",
}

# Translate slide texts
text_translation_map = {
    "Announcements:": "### 课程公告",
    "- Join the CS336 slack": "- 加入 CS336 Slack 频道",
    "- Sign up on Modal with your **Stanford** email": "- 使用你的 **Stanford** 邮箱注册并加入 Modal 算力平台",
    "- Read the [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)": "- 阅读 [AI 政策指南](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)",
    "- Read the [cluster guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)": "- 阅读 [集群使用指南](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)",
    "Marin 1e23 FLOPs run finished and [matched forecasts](https://x.com/WilliamBarrHeld/status/2039373983632814318)!": "Marin $10^{23}$ FLOPs 的模型预训练已顺利完成，并且[非常完美地符合我们的预测损失](https://x.com/WilliamBarrHeld/status/2039373983632814318)！",
    "Last lecture: overview, tokenization": "上一讲内容回顾：课程概述与分词 (Tokenization)。",
    "Today: resource accounting (systems)": "今天主题：**系统底座的资源核算 (Resource Accounting)**。",
    "Recall: what's the best model one can train given fixed resources (compute, memory)?": "> **核心出发点**：在硬件资源（算力、显存）固定的情况下，如何训练出最好的模型？",
    "In other words: maximize (computational) **efficiency**.": "> 也就是说，要最大化**计算效率**。",
    "Prerequisite: understand the resources (compute, memory) for a given computation.": "> **前提条件**：准确分析和核算给定计算任务的资源消耗。",
    "What knowledge to take away from this lecture:": "本讲的核心知识：",
    "- Mechanics: straightforward (PyTorch semantics)": "- **机制 (Mechanics)**：基础操作（PyTorch 语法）",
    "- Mindset: resource accounting (remember to do it)": "- **心态 (Mindset)**：学会进行资源核算，凡事量化分析",
    "- Intuitions: get a sense of how resources are spent, no ML magic today": "- **直觉 (Intuitions)**：对资源如何被消耗有一个大致的概念（这里没有 ML 魔法，只有 Napkin Math 物理账本）",
    "**Question**: How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?": "1. **问题**：在 1024 张 H100 GPU 上，训练一个 70B 参数的模型（在 15T Token 上预训练）需要多久？",
    "**Question**: What's the largest model that can you can train on 8 H100s using AdamW?": "2. **问题**：在 8 张 H100 (80GB) GPU 上，使用 AdamW 优化器，能训练的最大模型参数量是多少？",
    "Caveat: activations are not accounted for (depends on batch size and sequence length), so this is an upper bound.": "说明：这里未计算激活值显存（它取决于批大小和序列长度），因此这仅是模型参数量的上限。",
    "This is a rough back-of-the-envelope calculation.": "这是一个非常粗略的估算（Napkin Math）。",
    "But it gives you the flavor of napkin math one can quickly do to get a sense of resources.": "但它能让你体会到通过物理账本快速估算资源占用和训练耗时的方法。",
    "Tensors are the basic building block for storing everything:": "张量（Tensor）是存储一切的底层基本构建模块：",
    "- data": "- 数据 (Data)",
    "- parameters": "- 参数 (Parameters)",
    "- gradients": "- 梯度 (Gradients)",
    "- optimizer state": "- 优化器状态 (Optimizer state)",
    "- activations": "- 激活值 (Activations)",
    "Example: parameters of the DeepSeek v3.2 model ": "例如：DeepSeek v3.2 模型的参数。",
    "Each tensor has a rank, which is the number of dimensions.": "每个张量都有一个秩（Rank），即它的维度数量。",
    "In Transformers, will see tensors of rank 4:": "在 Transformer 中，我们经常会看到秩为 4 的张量：",
    "Elements of tensors are generally floating point numbers.": "张量的元素通常是浮点数。",
    "## fp32": "## fp32 (单精度)",
    "The fp32 data type (also known as float32 or single precision) is the default.": "fp32 数据类型（也称为 float32 或单精度）是默认的格式。",
    "Traditionally, in scientific computing, fp32 is the baseline; you could use double precision (fp64) in some cases.": "传统上，在科学计算中，fp32 是基线，甚至在某些情况下会使用双精度（fp64）。",
    "In deep learning, you can be a lot sloppier.": "但在深度学习中，我们可以对精度“粗心”得多。",
    "Let's examine memory usage of these tensors.": "让我们看看这些张量的显存占用情况。",
    "Memory is determined by the (i) number of values and (ii) data type of each value.": "显存占用是由（1）数值的个数 和（2）每个数值的数据类型 共同决定的。",
    "One matrix in the feedforward layer of GPT-3:": "GPT-3 前馈层（FFN）中的一个矩阵的大小：",
    "## fp16": "## fp16 (半精度)",
    "The fp16 data type (also known as float16 or half precision) cuts down the memory.": "fp16 数据类型（也称为 float16 或半精度）可以将内存减半。",
    "However, the dynamic range (especially for small numbers) isn't great.": "然而，fp16 的动态范围（尤其是针对极小数值）并不够大。",
    "If this happens when you train, you can get instability.": "如果在训练过程中发生这种情况，很容易导致数值不稳定（训练崩溃）。",
    "## bf16": "## bf16 (Bfloat16)",
    "Google Brain developed brain floating point (bf16) in 2018 to address this issue.": "Google Brain 研发的 bfloat16 格式，每个数值占用 2 字节。它具有与 fp32 相同的动态范围，但尾数精度较低。在大型模型训练中，bf16 可以免去 fp16 易发生下溢的梯度缩放操作。",
    "bf16 uses the same memory as fp16 but has the same dynamic range as fp32!": "bf16 与 fp16 占用相同的内存，但具有与 fp32 相同的动态范围！",
    "The only catch is that the resolution is worse, but this matters less for deep learning.": "唯一的妥协是其精度分辨率稍差，但这在深度学习中往往不是问题。",
    "## Mixed precision": "## 混合精度 (Mixed precision)",
    "Implications on training:": "这对训练的影响：",
    "- Training with fp32 works, but requires lots of memory.": "- 用 fp32 训练最稳定，但需要非常庞大的显存。",
    "- Training with fp16 and even bf16 is risky, and you can get instability.": "- 全程使用 fp16 甚至 bf16 存在极高的不稳定性风险。",
    "Solution: mixed precision training ": "解决方案：混合精度训练 (Mixed precision training)",
    "- Use bf16 for parameters, activations, and gradients": "- 参数、激活值和梯度使用 bf16 存储与计算",
    "- Use fp32 for optimizer states": "- 优化器状态（一阶、二阶动量）使用 fp32 存储与累加",
    "Pytorch has an automatic mixed precision (AMP) library. ": "PyTorch 提供了自动混合精度 (AMP) 库。",
    "Tries to cast things into bf16 when safe (matmuls, not exp).": "它会在安全的情况下自动将操作（例如矩阵乘法而非指数操作）转换为 bf16 计算。",
    "## fp8": "## fp8",
    "In 2022, fp8 was standardized, motivated by machine learning workloads [primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).": "2022年，受机器学习工作负载的推动，fp8 得到了标准化。",
    "H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]).": "H100 GPU 支持两种 FP8 变体：E4M3（动态范围 [-448, 448]）和 E5M2（动态范围 [-57344, 57344]）。",
    "Reference: ": "参考资料：",
    "## fp4": "## fp4",
    "In 2025, NVIDIA developed [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)": "2025年，NVIDIA 推出了针对超高效率推理的 [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。",
    "Only 4 bits per value!": "每个值仅占用 4 比特！",
    "Values: -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6": "可选数值：-6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6",
    "Use a separate scale factor per block, so actually get more dynamic range (but just can't vary freely from neighbors).": "在每个块内使用独立的缩放因子，从而在小范围内获得更高的动态范围（但无法脱离块内邻居随意变化）。",
    "Nemotron 3 Super was trained in NVFP4 ": "Nemotron 3 Super 是使用 NVFP4 进行训练的。",
    "Some of this is done in NVIDIA libraries outside of user control.": "其中许多底层的量化操作都是由 NVIDIA 底层库在后台完成的，对普通用户是不透明的。",
    "By default, tensors are stored in CPU memory.": "默认情况下，张量是存储在 CPU 内存中的。",
    "However, what about GPUs?": "然而，对于 GPU 呢？",
    "In order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory.": "为了利用 GPU 庞大的并行计算能力，我们需要将它们移动至 GPU 显存中。",
    "Or create the tensor directly on the GPU:": "或者直接在 GPU 上创建张量：",
    "Einops is a library for manipulating tensors where dimensions are named.": "Einops 是一个极其强大且直观的张量操作库，允许我们直接给维度命名。",
    "It is inspired by Einstein summation notation (Einstein, 1916).": "它的设计灵感来源于爱因斯坦求和约定 (Einstein, 1916)。",
    "Traditional PyTorch code:": "传统的 PyTorch 代码：",
    "Easy to mess up the dimensions (what is -2, -1?)...": "传统的写法非常容易搞错维度（到底 -2 和 -1 代表什么维度？）……",
    "Einsum is generalized matrix multiplication with good bookkeeping.": "Einsum 是包含维度簿记的泛化矩阵乘法。",
    "Let's try a more complex example...": "让我们来看一个更复杂的例子……",
    "Dimensions that are not named in the output are summed over.": "在输出部分未被命名的维度，将在计算时被自动求和缩并。",
    "You can reduce a single tensor via some operation (e.g., sum, mean, max, min).": "你可以通过某些操作（如求和、均值、最大值、最小值）来对单个张量的某些维度进行规约（Reduce）。",
    "Sometimes, a dimension represents two dimensions": "有时，一个物理维度实际上代表了两个逻辑维度。",
    "...and you want to operate on one of them.": "……而你希望只对其中一个维度进行操作。",
    "...where `total_hidden` is a flattened representation of `heads * hidden1`": "……其中 `total_hidden` 是 `heads * hidden1` 平铺展开后的表示。",
    "Having gone through all the operations, let us examine their computational cost.": "了解了这些基础操作后，接下来让我们深入核算它们的计算开销（FLOPs）。",
    "A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y).": "浮点运算次数 (FLOP) 是指最基础的浮点数学运算（如一次加法 $x+y$ 或一次乘法 $x \\cdot y$）。",
    "Two terribly confusing acronyms (pronounced the same!):": "两个非常容易混淆的英文缩写（读音完全相同）：",
    "- FLOPs: floating-point operations (measure of computation done)": "- **FLOPs**：浮点运算次数，是衡量执行的**总计算量**的度量。",
    "- FLOP/s: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.": "- **FLOP/s**（或 FLOPS）：每秒浮点运算次数，用于衡量**硬件计算速度**的性能指标。",
    "## Intuitions": "## 直觉感受",
    "Training GPT-3 (2020) took 3.14e23 FLOPs. ": "训练 GPT-3 (2020) 消耗了约 $3.14 \\times 10^{23}$ FLOPs 的总计算量。",
    "Training GPT-4 (2023) is speculated to take 2e25 FLOPs. ": "据估算，训练 GPT-4 (2023) 消耗了约 $2 \\times 10^{25}$ FLOPs。",
    "H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without ": "H100 标称的半精度稀疏峰值性能为 1979 TFLOPS，无稀疏时减半。",
    "8 H100s for 2 weeks:": "用 8 张 H100 GPU 连续训练两星期能提供的最大算力容量：",
    "## Linear model": "## 线性模型 (Linear Model)",
    "How many FLOPs is this matmul?": "该矩阵乘法（matmul）共包含多少次浮点运算？",
    "We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.": "对于每一个 $(i, j, k)$ 元组，我们需要执行一次乘法 and 一次加法。",
    "We can also time this operation to see how long it takes.": "我们也可以对这个操作进行测速。",
    "The actual FLOP/s of this operation:": "该操作的实际每秒浮点运算速度 (FLOP/s)：",
    "Each GPU has a specification sheet that provides the peak performance.": "每款 GPU 都有对应的规格表来提供它的峰值理论性能指标。",
    "- Example: ": "- 例如：",
    "Note that the FLOP/s depends heavily on the data type!": "需要注意的是，FLOP/s 在极大程度上取决于所使用的数据类型精度！",
    "## Model FLOPs utilization (MFU)": "## 模型算力利用率 (Model FLOPs Utilization, MFU)",
    "Definition: MFU = (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]": "定义：$\\text{MFU} = \\frac{\\text{实际吞吐 FLOP/s}}{\\text{硬件峰值 FLOP/s}}$（不含通信和框架冗余）。",
    "Usually, MFU of ≥ 0.5 is quite good!": "在实际大规模模型预训练中，MFU $\\ge 0.5$ 就已经是非常不错的硬件利用效率了！",
    "But why is MFU not closer to 1?": "为什么 MFU 难以接近 1.0？",
    "To answer this question, we need to look more closely at how computations are done on GPUs...": "为了回答这个问题，我们需要更深入地了解数据是如何在 GPU 的硬件层面上流转和计算的……",
    "How to compute a thing:": "硬件执行一次计算的基本步骤：",
    "1. Send inputs from memory to accelerator": "1. 将输入数据从全局显存 (HBM) 加载至计算核心 (Core / Register)",
    "2. Perform computation": "2. 核心执行实际的浮点计算",
    "3. Send outputs from accelerator to memory": "3. 将计算完成的输出写回全局显存 (HBM)",
    "How long does this take?": "整个过程需要多少时间？",
    "Depends on two things:": "这取决于两个关键的硬件指标：",
    "1. Accelerator speed (FLOP/s)": "1. 运算核心的速度 (FLOP/s)",
    "2. Memory bandwidth (bytes/s)": "2. 全局显存的带宽 (Bytes/s)",
    "Assume we can overlap communication and computation perfectly.": "假设我们能将数据传输与浮点计算完美地重叠进行。",
    "What is the bottleneck?": "什么是瓶颈所在？",
    "- Memory-bound: communication time > computation time": "- **显存带宽受限 (Memory-bound)**：传输数据所花的时间长于核心计算的时间。",
    "- Compute-bound: computation time > communication time": "- **算力受限 (Compute-bound)**：核心计算的时间长于数据传输的时间。",
    "In this case, ReLU is memory-bound.": "在这种情况下，单独的 ReLU 操作显然是显存带宽受限（Memory-bound）的。",
    "Alternative way to see this:": "另一种等价的分辨方法：",
    "Accelerator intensity: how much work can the accelerator do per byte transferred?": "硬件计算强度 (Accelerator Intensity)：硬件每秒传输一个字节的数据，计算核心理论上能做多少次计算？",
    "Arithmetic intensity: how much actual work per byte for this workload?": "算法的算术强度 (Arithmetic Intensity)：当前算子中，平均每传输一个字节的数据，实际执行了多少次浮点运算？",
    "In general, we'll find ourselves memory bound.": "你会发现，在许多元素级操作中，我们都处于显存受限（Memory-bound）状态。",
    "Can we increase arithmetic intensity?": "我们能设法提高算术强度吗？",
    "Note that GeLU does more work than ReLU per byte moved, so it has higher arithmetic intensity.": "我们注意到，由于 GeLU 包含复杂的 tanh/立方项计算，它在每移动一个字节的数据时执行了更多计算，因而其算术强度要显著高于 ReLU。",
    "But still memory-bound!": "但它依然处于显存受限阶段！",
    "In other words, ReLU is not faster than GeLU (when doing things in an isolated way).": "换言之，单独执行时，ReLU 并不比 GeLU 跑得更快（因为瓶颈全都在读写显存上）。",
    "Memory-bound!": "显存带宽受限！",
    "Finally, compute-bound!": "终于，进入算力受限（Compute-bound）状态！",
    "As long as we have large matrices, we're compute-bound (saturating the accelerator).": "只要矩阵维度足够大，乘法操作就能彻底掩盖显存传输开销，进入算力受限状态（榨干 GPU 算力）。",
    "Training Transformers involves big matrix multiplications.": "这就是为什么在训练 Transformer 时，我们主要处于算力受限（矩阵乘法为主），这很有利于压榨硬件性能。",
    "Matrix-vector product is what happens during inference, which is why inference is memory-bound.": "而在大模型推理（生成 Token）时，由于一次只处理一个 Token，矩阵乘法退化为矩阵-向量乘法，这也正是为什么大模型推理极度依赖显存带宽。",
    "Note: arithmetic/accelerator intensity also depends on the precision (bf16 versus fp32).": "注：算术强度与硬件计算强度的权衡，同样也高度依赖于我们选用的数值精度类型（例如 bf16 的硬件计算强度明显高于 fp32）。",
    "We can visualize the relationship between arithmetic intensity and performance using roofline plots.": "我们可以利用 Roofline 模型（屋顶图）非常直观地展现算法算术强度与硬件实际性能之间的关系。",
    "- Each slice on the x-axis is a particular computation (with some arithmetic intensity)": "- 横坐标 $x$ 代表算法的算术强度（每字节传输对应的计算次数）",
    "- Each piecewise linear function corresponds to a particular hardware": "- 折线图代表特定硬件在当前算术强度下能发挥的最大实际性能",
    "- Kink is the accelerator intensity (transition from memory-bound to compute-bound)": "- 折弯处的拐点（Kink）正是硬件计算强度（标志着从显存受限向算力受限的过渡）",
    "We can now relate this back to MFU:": "此时，我们将这与 MFU 关联起来：",
    "So far, we've constructed tensors and passed them through operations (forward).": "到目前为止，我们已经创建了各种张量并让它们执行前向传播。",
    "Now, we're going to compute the gradient (backward).": "现在，我们要开始计算梯度，即执行反向传播 (backward)。",
    "As a simple example, let's consider the simple linear model:": "作为一个极其简单的例子，让我们考查一个一维的线性模型：",
    "y = 0.5 (x * w - 5)^2": "$$y = 0.5 (x \\cdot w - 5)^2$$",
    "Forward pass: compute loss": "前向传播：计算 Loss",
    "Backward pass: compute gradients": "反向传播：计算梯度",
    "Let us count the FLOPs for computing gradients.": "接下来，我们来精确计算求解梯度所需的 FLOPs 开销。",
    "Define a simplified model (2-layer linear network):": "定义一个简化的 2 层线性网络模型：",
    "## Zoom in on one layer": "## 聚焦于单个层级",
    "Let's focus on the second layer (h2 = h1 @ w2)": "我们重点分析第二层：$h_2 = h_1 W_2$",
    "**Forward pass**: Recall the number of forward FLOPs: ": "**前向传播**：回想一下前向矩阵乘法的 FLOPs 数量：",
    "**Backward pass**: How many FLOPs is running the backward pass?": "**反向传播**：执行反向梯度计算需要多少 FLOPs？",
    "We need to compute:": "我们需要计算：",
    "- h1.grad = d loss / d h1": "- 相对输入激活值的梯度 `h1.grad`（$\\frac{\\partial L}{\\partial h_1}$）",
    "- w2.grad = d loss / d w2": "- 相对权重参数的梯度 `w2.grad`（$\\frac{\\partial L}{\\partial W_2}$）",
    "Note that the backward pass is 2x more expensive than the forward pass.": "我们非常清楚地看到，单个层的反向传播计算开销恰好是前向传播的 2 倍。",
    "## Consider all layers": "## 考虑网络的所有层",
    "This was just for w2, need to apply it to all parameters in the network.": "上面只考查了 $W_2$，反向传播必须贯穿整个神经网络的所有参数。",
    "Putting it together:": "总结前向与反向的算力配比：",
    "- Forward pass: 2 (# data points) (# parameters) FLOPs": "- **前向传播**：$2 \\times \\text{数据样本数} \\times \\text{模型参数量}$ FLOPs",
    "- Backward pass: 4 (# data points) (# parameters) FLOPs": "- **反向传播**：$4 \\times \\text{数据样本数} \\times \\text{模型参数量}$ FLOPs",
    "- Total: 6 (# data points) (# parameters) FLOPs": "- **单步训练总计**：$6 \\times \\text{数据样本数} \\times \\text{模型参数量}$ FLOPs",
    "This is for multilayer perceptrons (MLPs)": "这个极其著名的 “6 倍参数量” 经验法则同样也对多层感知机（MLP）和 Transformer 在短上下文下非常适用。",
    "...but it turns out to be a good approximation for Transformers for short context lengths as well.": "……并且它也给后面的作业中预估大规模预训练算力开销提供了最核心的理论依据。",
    "Consider a deep network with L layers and D-dimensional inputs, activations, and outputs.": "考查一个具有 $L$ 层，且输入、输出及中间激活值均为 $D$ 维的深度 MLP 网络模型。",
    "Recall our deep network.": "回顾我们刚刚定义的深度神经网络。",
    "Let's define the AdaGrad optimizer": "让我们定义一个 AdaGrad 优化器（作为实现定制优化器的展示）：",
    "- momentum = SGD + exponential averaging of grad": "- 动量法 (Momentum) = SGD + 梯度的一阶指数移动平均",
    "- AdaGrad = SGD + averaging by grad^2": "- AdaGrad = SGD + 累加梯度历史平方和进行自适应缩放",
    "- RMSProp = AdaGrad but with exponential averaging of grad^2": "- RMSProp = AdaGrad + 梯度的二阶指数移动平均",
    "- Adam = RMSProp + momentum": "- Adam = RMSProp + Momentum（结合一阶与二阶动量，当下大模型最常用的优化器）",
    "AdaGrad ": "AdaGrad 论文参考：",
    "## Memory": "## 显存核算 (Memory Accounting)",
    "It is customary to use fp32 for stability (accumulating averages over powers over many steps).": "为了数值更新稳定性，优化器的状态（如平方梯度和、动量）通常必须使用高精度 (fp32) 存储。",
    "Optimizer state memory:": "优化器状态的显存占用：",
    "- AdaGrad: 4 bytes/parameter for storing second moments": "- AdaGrad：每个参数 4 字节（仅需存储二阶动量值 $g^2$）",
    "- Adam: 8 bytes/parameter for storing first and second moments": "- Adam：每个参数 8 字节（需要存储一阶动量 $m$ 和二阶动量 $v$）",
    "## Compute (for one training step)": "## 单步训练计算开销",
    "## Transformers": "## 在 Transformer 中的资源核算",
    "The accounting for a Transformer is more complicated, but the same idea.": "Transformer 里的资源账本核算要复杂一些（需要考虑多头注意力、KV 缓存等），但核心方法完全一致。",
    "Assignment 1 will ask you to do that.": "作业 1 将要求你亲手完成 Transformer 的资源核算。",
    "Blog post describing memory usage for Transformer training ": "有关 Transformer 训练显存分析的优秀博客：",
    "Blog post describing FLOPs for a Transformer: ": "有关 Transformer 训练算力 FLOPs 计算的优秀博客：",
    "Large batch sizes: improve training stability": "大批大小（Large batch sizes）能够平滑梯度，提升分布式训练的稳定性。",
    "However, activation memory scales with batch size, so might run out.": "然而，前向中间激活值的显存随 Batch size 线性增加，很容易导致显存溢出（OOM）。",
    "Gradient accumulation:": "**梯度累加 (Gradient Accumulation)** 的工作原理：",
    "- Compute gradient on micro batches": "- 在更小的 Micro batch 上分别执行前向和反向，但不执行权重更新；",
    "- Accumulate the gradients (don't zero it out)": "- 将梯度累加在参数的 `.grad` 中；",
    "- Every batch_size / micro_batch_size steps, update the parameters and zero out the gradients": "- 每隔若干步后（当有效 batch size 到达预期目标），让优化器步进更新一次权重，随后清空梯度。",
    "For training, we need to store the activations of all layers": "在进行模型训练时，反向传播必须使用前向的所有激活值以求解梯度，因而需将其保留在显存中。",
    "For inference, we don't compute gradients, so we only need to store the current layer's activations.": "而在模型推理时不需要求解梯度，因而前向之后可以直接释放前面的激活值，仅保留当前层级的信息即可。",
    "The memory usage is": "此时的显存开销对比：",
    "Can we reduce this?": "我们能进一步压缩前向激活值占用的显存吗？",
    "Activation checkpointing = gradient checkpointing = rematerialization": "**激活值检查点 (Activation Checkpointing)**（又称梯度检查点或重算机制）：",
    "Key idea:": "核心思想：",
    "- Forward pass: keep only activations at subset of layers": "- 前向传播：只保存极少数“检查点”层（如每段 Block 的输入）的激活值，丢弃中间结果；",
    "- Backward pass: recompute the missing activations from the last checkpoint": "- 反向传播：当某层计算梯度需要中间激活值时，从最近的检查点层出发重新运行一次前向重算（Rematerialization），复原所需激活值。",
    "Philosophy: tradeoff memory for compute": "其核心本质是以少量的“重算时间”来换取海量的“显存空间”。",
    "Can we reduce this even more, especially for deep networks (large L)?": "对于超深的网络（$L$ 很大），我们能继续减少激活显存吗？",
    "How frequently to checkpoint?": "我们应该以多大的频率（间隔）设置检查点？",
    "- If store each layer's activations, then activation memory is O(L) and no recomputation.": "- **保存所有层**：激活显存为 $O(L)$，无重算开销。",
    "- If store no activations, then activation memory is O(1) and compute is O(L^2) (recompute from the start for each layer).": "- **完全不保存**：激活显存为 $O(1)$，但重算时间复杂度退化至 $O(L^2)$。",
    "- If store every sqrt(L) layers, then activation memory is O(sqrt(L)) and O(L) recomputation.": "- **每隔 $\\sqrt{L}$ 层保存一次**：实现完美折中，激活显存降低至 $O(\\sqrt{L})$，重算开销仅为 $O(L)$。",
    "Summary:": "## 第二讲总结",
    "- Everything is operations on tensors (parameters, gradients, activations, optimizer states, data)": "- 深度学习底层的计算全部都是围绕张量进行的（参数、梯度、中间激活、优化器状态、训练数据）。",
    "- einops: better way to think about tensor operations": "- **Einops** 库提供了一种更健壮、不易出错且更清晰的张量维度管理和变换方式。",
    "- 6 (# data points) (# parameters) FLOPs per training step": "- 每次梯度更新需要执行大约 **$6 \times N \times D$** 次浮点运算（FLOPs）。",
    "- Arithmetic intensity / roofline analysis: compute-bound or memory-bound?": "- 通过**算术强度与 Roofline 拓扑分析**，我们可以判断硬件当前的运行状态究竟是受限于显存带宽还是计算算力。",
    "- Matrix multiplications are compute-bound, elementwise operations are memory-bound": "- 大型矩阵乘法往往是**算力受限 (Compute-bound)** 的；而逐元素（Element-wise）操作或矩阵-向量乘法往往是**显存带宽受限 (Memory-bound)** 的。",
    "- Gradient accumulation, activation checkpointing: reduce memory to use bigger batch sizes": "- 我们可以通过**梯度累加**和**激活值检查点**这两大常用机制，以微小的时间牺牲，成倍降低训练时的显存峰值要求，从而能够训练更大规模的模型。",

    # LECTURE 07 SPECIFIC
    "Last week: parallelism within a single GPU": "上一讲主题：单块 GPU 内部的计算与内存并行优化。",
    "This week: parallelism across multiple GPUs": "本讲主题：跨多块 GPU 和多计算节点的分布式并行训练。",
    "In both cases, **compute** (arithmetic logic units) is far from inputs/outputs (**data**).": "在这两种情况下，核心的**计算单元** (ALU / Tensor Core) 距离**数据源** (显存/内存) 都显得相当遥远。",
    "Unifying theme: orchestrate computation to avoid data transfer bottlenecks": "核心思想：精心编排计算与传输的重叠，最大限度避免数据传输成为计算瓶颈。",
    "Generalized hierarchy:": "分布式架构中的数据层级关系：",
    "- Single node, single GPU: L1 cache / shared memory (fastest)": "- 单节点、单 GPU 内部：L1 缓存 / 共享内存 (极快)",
    "- Single node, single GPU: HBM": "- 单节点、单 GPU 显存：HBM 显存",
    "- Single node, multi-GPU: NVLink/NVSwitch": "- 单节点、多 GPU 之间：NVLink/NVSwitch 通信总线",
    "- Multi-node, multi-GPU: Infiniband/Ethernet (slowest)": "- 多节点、多 GPU 之间：Infiniband / 以太网网络连接 (最慢)",
    "Last week: reduce memory accesses via fusion/tiling": "单 GPU 层面：利用算子融合与 Tiling 机制减少多余的显存读写。",
    "This week: reduce communication across GPUs/nodes via replication/sharding": "多 GPU 层面：利用参数复制、分片等策略减少节点间的通信开销。",
    "Why do multi-GPU?": "为什么我们需要采用多 GPU 并行？",
    "1. Your parameters (optimizer state + gradients + activations) don't fit on a single GPU.": "1. **显存装不下**：随着模型增大，参数、优化器状态、梯度和中间激活值超出了单张 GPU 的显存容量。",
    "2. You want to use more GPUs (more FLOPs) to train faster.": "2. **算力不够快**：希望联合更多 GPU (获取更大算力 FLOPs)，从而缩短训练时间。",
    "### Part 1: building blocks of distributed communication/computation": "### 第一部分：分布式通信与计算的基本构建块",
    "### Part 2: distributed training": "### 第二部分：分布式训练并行策略",
    "Walk through bare-bones implementations of each strategy on deep MLPs.": "我们将通过多层感知机 (MLP) 的最小化代码实现，逐个解剖不同的并行策略。",
    "Recall that MLPs are the compute bottleneck in Transformers, so this is representative.": "这极具代表性，因为 MLP 是 Transformer 模型中最主要的计算开销之一。",
    "What's missing?": "本讲暂未涵盖的议题：",
    "- Communication/computation overlap": "- 通信与计算的深度重叠优化",
    "- More general models (with attention, etc.)": "- 更为复杂的注意力机制并行等",
    "- Other forms of parallelism (e.g., sequence parallelism, expert parallelism, combinations)": "- 其他高级并行形式（如序列并行、专家并行以及混合并行等）",
    "- Jax/TPUs: just define the model, the sharding strategy, and the Jax compiler handles the rest ": "- Jax/TPU 并行：只需在模型中定义张量的分片方式，底层的编译器将自动生成通信拓扑。",
    "- But we're doing PyTorch so you can see how one builds up from the primitives": "- 但在 PyTorch 中，我们需要手动调用分布式原语，这非常有助于深刻理解底层机制。",
    "### Summary": "### 第七讲总结",
    "- Many ways to parallelize: data (batch), tensor/expert (width), pipeline (depth), sequence (length)": "- 分布式并行有多种拆分维度：数据并行 (拆分 Batch)、张量/专家并行 (拆分宽度/通道数)、流水线并行 (拆分层深)、序列并行 (拆分序列长度)",
    "- Data parallelism: DDP (all-reduce), FSDP/ZeRO (all-gather + reduce-scatter)": "- **数据并行**：DDP（借助 All-Reduce 同步梯度）以及 FSDP/ZeRO（结合 All-Gather 与 Reduce-Scatter 消除多余的显存持有）",
    "- Tensor parallelism: requires very fast interconnects (e.g., NVLink)": "- **张量并行**：将单层拆分到不同 GPU，由于每层均需同步激活值，因而极度依赖极高速的卡间带宽 (如 NVLink)",
    "- Pipeline parallelism: can work with slow interconnects, but need to work to reduce pipeline bubbles": "- **流水线并行**：将不同层部署到不同 GPU 顺次计算，对网络通信带宽要求低，但必须合理排布流水线以减小空闲泡泡 (Bubble)",
    "- Can **re-compute** or store in **memory** or store in another GPUs memory and **communicate**": "- 系统设计的永恒权衡：是用**重算 (Recompute)**、**显存存储 (Memory)** 还是**跨卡通信 (Communicate)** 来解决局部硬件存储限制",
    "- Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure": "- 尽管硬件网络在不断加速，但模型规模的膨胀使得这些多层次 of 分布式并行架构始终是前沿训练的必修课。",
    "**Collective operations** are the conceptual primitives used for distributed programming ": "**集体通信操作 (Collective Operations)** 是分布式并行编程中最底层的概念基石。",
    "- These are classic in the parallel programming literature from the 1980s.": "- 这些操作早在 1980 年代的并行机集群设计文献中就已经成为经典。",
    "- *Collective* means that you specify a general communication pattern across many devices.": "- **集体 (Collective)** 意味着你需要在一个通信组内的多台设备间指定一种统一的通信拓扑。",
    "- This can be better/faster than managing point-to-point communication yourself.": "- 相比由用户自己维护繁琐的卡对卡点对点通信，集体通信库往往能提供更为极致的网络拓扑性能优化。",
    "**Setup**:": "**分布式设置**：",
    "- **Rank**: a particular device/GPU (e.g., 0, 1, 2, 3)": "- **Rank**：标识特定的 GPU 设备编号（例如 0, 1, 2, 3 等）",
    "- **World size**: total number of devices (e.g., 4)": "- **World size**：当前通信组内的 GPU 总卡数（例如 4）",
    "Operations:": "主要的通信操作包括：",
    "- Broadcast, scatter, gather, reduce (foundations)": "- Broadcast (广播)、Scatter (分发)、Gather (收集)、Reduce (规约) 等基础操作",
    "- All-gather, reduce-scatter, all-reduce (workhorse)": "- All-Gather、Reduce-Scatter、All-Reduce 等分布式训练的核心顶梁柱原语",
    "- All-to-all (for MoEs)": "- All-to-All (常用于混合专家模型 MoE 中路由数据)",
    "**Broadcast**: copy from rank 0 to all ranks": "**Broadcast (广播)**：将 Rank 0 卡上的数据完整复制到所有 Rank 卡上。",
    "Minor use case: rank 0 loads initial checkpoint and broadcasts to all ranks": "常见用例：Rank 0 负责从磁盘读取初始化检查点，然后 Broadcast 给其余 worker 同步参数状态。",
    "**Scatter** tensor on rank 0 to all ranks": "**Scatter (分发)**：将 Rank 0 卡上的一个大张量按维度均匀切分，并分发到各个 Rank 卡上。",
    "Note: stepping stone to understanding reduce-scatter": "注：这对于理解 Reduce-Scatter 很有帮助。",
    "**Gather** pieces from all ranks to rank 0 (opposite of scatter)": "**Gather (收集)**：将各个 Rank 卡上的小张量拼接，收集到 Rank 0 上形成一个大张量（Scatter 的反向操作）。",
    "Note: stepping stone to understanding all-gather": "注：这对于理解 All-Gather 很有帮助。",
    "**Reduce** pieces from all ranks to rank 0, applying some operation (e.g., sum, min, max)": "**Reduce (规约)**：对所有 Rank 卡上的数据对应位置应用某种数学规约操作（如求和、求极值），最后只把结果保存在 Rank 0 上。",
    "Note: stepping stone to understanding all-reduce": "注：这对于理解 All-Reduce 很有帮助。",
    "**All-gather**: perform gather to all ranks, not just rank 0": "**All-Gather**：在各个 Rank 卡上独立执行 Gather，使得最后所有 Rank 卡都拥有完全拼接后的完整大张量。",
    "Use case: each rank holds parameter shard, gather to get full parameters for forward pass": "典型用例：在 ZeRO/FSDP 中，各卡在平时只保存一份参数切片，前向计算时通过 All-Gather 收集并恢复成完整参数。",
    "**Reduce-scatter**: perform reduce on each dimension, scatter results": "**Reduce-Scatter**：对数据对应位置执行数学规约，随后将规约结果按 Rank 维度切分分发到各个 Rank 上。",
    "Use case: after backward pass, sum gradients from different data shards, but distribute storage": "典型用例：在反向传播计算出梯度后，通过 Reduce-Scatter 进行梯度均值规约，并让各卡只存储梯度的一部分切片，从而节省显存。",
    "**All-reduce** = reduce-scatter + all-gather": "**All-Reduce**：对所有卡上的数据应用数学规约，并将完整规约结果输出到所有卡上（等价于先 Reduce-Scatter 再 All-Gather）。",
    "Use case: after backward pass, sum gradients from different data shards, but replicate full parameters": "典型用例：在传统数据并行 (DDP) 中，反向传播后通过 All-Reduce 同步各卡的梯度，随后在所有卡上重复更新相同的完整参数。",
    "Breaking all-reduce into reduce-scatter + all-gather allows for flexibility (e.g., ZeRO/FSDP)": "将 All-Reduce 拆解为 Reduce-Scatter 与 All-Gather，极大促进了 ZeRO/FSDP 等显存友好型数据并行的发展。",
    "**All-to-all**: each rank sends each other rank some tensor (most general)": "**All-to-all**：最通用的多对多通信，每个 Rank 向所有其他 Rank 各自发送特定的张量分片。",
    "Notes:": "要点说明：",
    "- Useful for MoEs: each rank has split of data and subset of experts; need to route data to experts": "- 它是混合专家模型 (MoE) 的核心通信管道：每张卡持有不同的样本批次，通过 All-to-All 将不同的 Token 路由发送到特定的专家 (Expert) 卡上进行处理。",
    "- For balanced splits, all-to-all looks like transpose": "- 在数据均衡分布时，All-to-All 通信在逻辑上非常类似矩阵的转置。",
    "- Also handles unbalanced splits (but want splits to be as balanced as possible)": "- 它也能够处理不均衡的数据分片（但通常由于硬件负载考虑，应尽量做到样本均衡分配）。",
    "Way to remember the terminology:": "如何快速记忆这些名词原语：",
    "- Reduce: performs some associative/commutative operation (sum, min, max)": "- **Reduce**：表示对数据应用结合律/交换律的规约运算（求和、最小值、最大值）。",
    "- Scatter is inverse of gather": "- **Scatter (分发)** 与 **Gather (收集)** 互为逆操作。",
    "- All: means destination is all devices": "- **All-** 前缀：意味着最终数据接收端是所有参与的计算设备，而网络拓扑效率更高。",
    "Classic (in the home):": "经典拓扑（家用/个人工作站环境）：",
    "- GPUs on same node communicate via a PCI(e) bus (v7.0, 16 lanes => 242 GB/s) ": "- 同节点内的多张 GPU 通过 PCI(e) 总线完成通信（PCIe 7.0 x16 单向带宽可达 242 GB/s）。",
    "- GPUs on different nodes communicate via Ethernet (~200 MB/s)": "- 跨节点多卡之间使用千兆/万兆以太网进行连接，这往往会带来毁灭性的网络延迟和带宽限制 (~200 MB/s)。",
    "Modern (in the data center):": "现代拓扑（数据中心高性能集群环境）：",
    "Typical setup:": "典型多卡网络架构设计：",
    "- 8 GPUs per node, connected by NVLink to an NVSwitch (B200s' NVLink 5.0 gets 1.8 TB/s; HBM was 8 TB/s)": "- **单节点 8 卡**：通过板载 NVLink 高速总线直接互连到 NVSwitch 交换芯片（B200 对应的 NVLink 5.0 可提供高达 1.8 TB/s 的卡间双向带宽，作为对比，HBM 显存带宽约为 8 TB/s）。",
    "- 256 nodes per pod, connected by Infiniband (via PCIe -> HCA / Infiniband NIC -> Infiniband cable) (~0.05 TB/s)": "- **单 Pod 内 256 个节点**：由 PCIe 扩展出专用网卡 (Infiniband NIC/HCA)，通过 Infiniband 交换机互连，节点间跨网带宽约可达 ~0.05 TB/s。",
    "- N pods per cluster / datacenter, connected by Ethernet (via PCIe -> CPU)": "- **集群/数据中心多 Pod**：采用常规光纤以太网完成超大规模的连接。",
    "Bypassing the CPU:": "绕过 CPU 参与的数据传输：",
    "- Ethernet requires passing through the CPU (copying data to kernel socket buffer, build TCP packets, copy to NIC ring buffer)": "- 传统的以太网传输需要操作系统 CPU 的频繁干预（需要多次拷贝数据至内核 Socket 缓冲区，建立 TCP 协议栈并打包，最后发送至网卡发送环缓冲区）。",
    "- Remote Direct Memory Access (RDMA): allows one GPU to directly read/write another GPU's memory without involving the CPU": "- **远程直接内存访问 (RDMA)** 机制允许一张 GPU 绕过 CPU 控制直接读取或写入另一台机器上 GPU 的显存空间。",
    "- Infiniband supports RDMA, but standard Ethernet does not": "- Infiniband 网络天生完美支持 RDMA；而标准商业以太网往往不支持。",
    "Advancements:": "最新技术演进：",
    "- GB200/GB300 NVL72: 8 GPUs per tray, 9 trays per rack -> 72 GPUs in one NVLink domain": "- **GB200/GB300 NVL72 柜机**：每盘包含 8 颗 GPU，单个机架放入 9 盘，形成由 72 颗 GPU 直接构成的巨大统一 NVLink 域。",
    "- RDMA over Converged Ethernet (RoCE): Ethernet bypasses CPU, similar but cheaper/weaker than Infiniband, used by Meta": "- **RoCE 技术**：在常规以太网上承载 RDMA 流量，相比 Infiniband 成本更低，在 Meta 的超大规模集群中得到了极其广泛的应用。",
    "### NVIDIA Collective Communication Library (NCCL)": "### NVIDIA 集合通信库 (NCCL)",
    "NCCL translates collective operations into low-level packets that are sent between GPUs. ": "NCCL 负责将顶层的 Collective 集合通信原语（如 All-Reduce）转化为底层硬件网络的数据包进行传输。",
    "- Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe)": "- 自动感知系统的底层拓扑（有多少张卡、多少个交换机、走 NVLink 还是走 PCIe）。",
    "- Optimizes the path between GPUs": "- 自动匹配并优化跨卡数据流的最佳路径。",
    "- Launches GPU kernels to send/receive data": "- 直接调度定制化的 GPU CUDA Kernel 负责极速收发数据，免去 CPU 开销。",
    "PyTorch distributed library (`torch.distributed`) ": "PyTorch 分布式框架 (`torch.distributed`)",
    "- Provides clean interface for collective operations (e.g., `all_gather_into_tensor`)": "- 提供了极为整洁的集合通信 API 接口（例如 `all_gather_into_tensor`）。",
    "- Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)": "- 支持对接多种底层硬件后端：gloo (支持 CPU 分布式通信) 和 nccl (支持 GPU 高速通信)。",
    "- Also supports higher-level algorithms (e.g., `FullyShardedDataParallel`) [not used in this course]": "- 也封装了例如 FSDP 等的高阶接口（本课程暂不涉及，我们从底层写起）。",
    "Let's walk through some examples.": "让我们来看几个实际运行的例子。",
    "Indeed, all-reduce = reduce-scatter + all-gather!": "我们可以非常直观地验证：All-Reduce 的最终结果确实等于先 Reduce-Scatter 再 All-Gather！",
    "How fast does communication happen?": "集群中的卡间通信到底能有多快？",
    "References:": "网络性能测试参考资料：",
    "Sharding strategy: each rank gets a slice of the data": "切分策略：数据切片分布在各卡上，各卡模型与参数完全一致。",
    "Notes:": "要点说明：",
    "- Losses are different across ranks (computed on local data)": "- **Loss 计算独立**：由于各卡处理的样本（Data shard）不同，前向传播得到的局部 Loss 也完全不同。",
    "- Gradients are all-reduced to be the same across ranks": "- **梯度 All-Reduce 同步**：反向传播后，需要对各卡的梯度进行 All-Reduce 并广播，以保证梯度完全一致。",
    "- Therefore, parameters remain the same across ranks": "- **参数完全镜像**：梯度一致，加上优化器以相同步长推进，保证了各卡在每次更新后权重完全相同。",
    "Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory": "下节预告：FSDP/ZeRO 并行，使用 All-Gather 和 Reduce-Scatter 消除重复持有完整模型参数的显存开销。",
    "Sharding strategy: each rank gets part of each layer, transfer all data/activations": "切分策略：每一层的参数矩阵被按维度切分到不同的卡上，每次计算时需要卡间通信同步激活值。",
    "Sharding strategy: each rank gets subset of layers, transfer all data/activations": "切分策略：将网络深度层均匀分发到不同的 GPU 上，顺次执行前向与反向传输。",
    "Not handled: overlapping communication/computation to eliminate pipeline bubbles": "说明：此处未加入计算与通信的异步重叠（Overlapping），因此无法完全消除流水线并行中的气泡延迟（Pipeline Bubble）。",
}

comments_map = {
    # Comments in lecture 02
    "Memory accounting": "显存账本核算",
    "Compute accounting": "算力账本核算",
    "Memory and compute accounting for training": "训练过程中的显存与算力核算",
    "More memory optimizations": "更多显存控制优化",
    "parameters (2), gradients (2), optimizer state (4 + 4)": "参数(2), 梯度(2), 优化器状态(4 + 4)",
    "Float is 4 bytes": "float32 占用 4 字节",
    "Default type": "默认数据类型",
    "Underflow!": "数值下溢！",
    "No underflow!": "未发生下溢！",
    "Use bf16 for parameters, activations, and gradients": "参数、激活值和梯度使用 bf16",
    "Use fp32 for optimizer states": "优化器状态使用 fp32",
    "Tries to cast things into bf16 when safe (matmuls, not exp).": "安全时自动转为 bf16（如矩阵乘法，非指数项）",
    "H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]).": "H100 支持两种 FP8 变体：E4M3 和 E5M2",
    "Only 4 bits per value!": "每个值仅 4 比特！",
    "Values: -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6": "可选值列表",
    "By default, tensors are stored in CPU memory.": "默认情况下，张量保存在 CPU 内存中",
    "Or create the tensor directly on the GPU:": "或者直接在 GPU 上创建张量：",
    "rank 1 tensor (vector)": "秩为 1 的张量（向量）",
    "rank 2 tensor (matrix)": "秩为 2 的张量（矩阵）",
    "rank 3 tensor": "秩为 3 的张量",
    "Batch size": "批大小",
    "Sequence length": "序列长度",
    "Number of heads": "注意力头数",
    "Hidden dimension per head": "每个头的隐藏层维度",
    "batch seq hidden": "批大小 序列长度 隐藏维度",
    "batch seq seq": "批大小 序列长度 序列长度",
    "seq1 hidden": "序列1 隐藏维度",
    "hidden seq2": "隐藏维度 序列2",
    "Old way": "传统 PyTorch 写法",
    "New (einops) way": "使用 Einops 写法",
    "batch seq1 hidden": "批大小 序列1 隐藏维度",
    "batch seq2 hidden": "批大小 序列2 隐藏维度",
    "Or can use `...` to represent broadcasting over any number of dimensions": "或者可以使用 `...` 表示广播任意数量的维度",
    "seq total_hidden": "序列 隐藏维度总和",
    "hidden1 hidden2": "隐藏维度1 隐藏维度2",
    "Break up `total_hidden` into two dimensions (`heads` and `hidden1`": "将 total_hidden 拆分为两个维度（heads 和 hidden1）",
    "Break up total_hidden into two dimensions (heads and hidden1)": "将 total_hidden 拆分为两个维度（heads 和 hidden1）",
    "Perform the transformation by `w`": "通过 w 矩阵执行线性变换",
    "Perform the transformation by w": "通过 w 矩阵执行线性变换",
    "Combine `heads` and `hidden2` back together": "将 heads 和 hidden2 重新合并",
    "Combine heads and hidden2 back together": "将 heads 和 hidden2 重新合并",
    "Number of points": "样本数量",
    "Dimension of each point": "样本维度",
    "Number of outputs": "输出维度",
    "We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.": "对每个 (i, j, k) 三元组，有一次乘法和一次加法",
    "actual FLOP/s of this operation:": "该操作的实际 FLOP/s",
    "MFU = min(1, arithmetic-intensity / accelerator-intensity)": "MFU 计算公式",
    "Read x, write y (bf16 is 2 bytes/float)": "读取 x, 写入 y (bf16 为 2 字节)",
    "n comparisons": "n 次比较",
    "Accelerator intensity: how much work can the accelerator do per byte transferred?": "硬件计算强度：每传输一个字节硬件能执行多少次计算",
    "Arithmetic intensity: how much actual work per byte for this workload?": "算法算术强度：对该任务，平均每字节实际执行多少次计算",
    "tanh can be approximated in various ways (e.g., polynomials)": "tanh 可以通过多项式等多种方式进行近似",
    "Read x, read w, write y": "读取 x, 读取 w, 写入 y",
    "n multiplications, n-1 additions": "n 次乘法，n-1 次加法",
    "n dot-products": "n 次点积",
    "n^2 dot products": "n^2 次点积",
    "Forward pass: compute loss": "前向传播：计算损失值",
    "Backward pass: compute gradients": "反向传播：计算梯度",
    "Define a simplified model (2-layer linear network):": "定义简化的模型（2 层线性网络）：",
    "Forward pass": "前向传播",
    "Backward pass": "反向传播",
    "For debugging": "仅供调试检查使用",
    "Let's focus on the second layer (h2 = h1 @ w2)": "重点分析第二层",
    "Forward pass: Recall the number of forward FLOPs:": "前向传播：回想前向矩阵乘法的 FLOPs 数量",
    "Backward pass: How many FLOPs is running the backward pass?": "反向传播：反向梯度计算的 FLOPs",
    "Define the network": "定义网络结构",
    "Dimensionality of input, activations, and output": "输入、激活和输出的维度",
    "Number of layers": "网络层数",
    "Run the model on a batch of data": "在一个 batch 的数据上运行模型",
    "Batch size": "批大小",
    "Simple block that applies a linear transformation followed by a ReLU nonlinearity.": "线性变换加 ReLU 激活的简单模块",
    "Linear": "线性变换",
    "Activation": "激活函数",
    "Map `dim`-vector to a `dim`-vector.": "将 dim 维向量映射到 dim 维向量",
    "Map dim-vector to a dim-vector.": "将 dim 维向量映射到 dim 维向量",
    "Apply all the layers sequentially": "顺次应用所有层",
    "Recall our deep network.": "回顾深度网络",
    "Let's define the AdaGrad optimizer": "定义 AdaGrad 优化器",
    "Compute gradients": "计算梯度",
    "Take a step": "优化器参数步进更新",
    "Free up the memory": "清空/释放梯度以节省内存",
    "Putting it all together": "将显存各项开销相加",
    "6 (# data points) (# parameters) FLOPs per training step": "每次训练包含的 FLOPs 开销",
    "Blog post describing memory usage for Transformer training": "介绍 Transformer 训练显存分析的优秀博客",
    "Blog post describing FLOPs for a Transformer:": "介绍 Transformer 训练算力 FLOPs 计算的博客",
    "Optimizer state": "优化器状态",
    "Get squared gradients": "获取平方梯度累积值",
    "Update optimizer state": "更新优化器内部状态",
    "Update parameters": "步进更新参数",
    "True linear function with weights (0, 1, 2, ..., D-1)": "真实线性权重",
    "Data loader that generates (x, y) pairs": "自动生成 (x, y) 的数据生成器",
    "Define the model and optimizer": "定义网络与优化器",
    "Train!": "开始训练循环！",
    "Get data": "提取数据 batch",
    "Update parameters": "更新模型参数",
    "However, activation memory scales with batch size, so might run out.": "激活值显存随批大小线性增加，容易溢出",
    "Gradient accumulation:": "梯度累加操作：",
    "Every batch_size / micro_batch_size steps, update the parameters and zero out the gradients": "满累积步数后，更新参数并清空梯度",
    "For training, we need to store the activations of all layers": "训练时必须存储所有层的前向激活值",
    "For inference, we don't compute gradients, so we only need to store the current layer's activations.": "推理时无需梯度，只需保留当前层的激活值",
    "Define the model with checkpointing": "定义带检查点优化的模型",
    "How frequently to checkpoint?": "设置检查点的频率多大合适？",
    "Same as DeepNetwork, but with activation checkpointing.": "与 DeepNetwork 相同，但引入了激活值检查点",
    "KEY: only store activations at checkpoints, recompute the rest": "核心：仅在检查点处保留激活值，其余计算通过重算复原",
    "Return the peak FLOP/s for device operating on dtype.": "返回特定精度的峰值理论算力",
    "No CUDA device available, so can't get FLOP/s": "无 CUDA 设备可用，返回 1",
    "Return the number of seconds required to perform func.": "返回执行 func 所需的平均秒数",
    "Wait until previous CUDA threads are done": "同步等待之前的 CUDA 线程执行完毕",
    "Time the operation num_trials times": "重复执行 num_trials 次以统计耗时",

    # Comments in lecture 07
    "In both cases, compute is far from inputs/outputs.": "在这两种情况下，计算单元均距离数据源十分遥远",
    "Single node, single GPU: L1 cache / shared memory (fastest)": "单节点、单 GPU 内部：L1 / 共享内存 (最快)",
    "Single node, single GPU: HBM": "单节点、单 GPU 显存：HBM 显存",
    "Single node, multi-GPU: NVLink/NVSwitch": "单节点、多 GPU 之间：NVLink 通信",
    "Multi-node, multi-GPU: Infiniband/Ethernet (slowest)": "多节点、多 GPU 之间：Infiniband 连接 (最慢)",
    "Why do multi-GPU?": "为什么我们需要采用多 GPU 并行？",
    "When you execute this lecture directly (python lecture_07.py), it uses multiprocessing, which produces output from each process (below).": "直接执行 python 脚本时会启动多进程，产生不同进程的输出。",
    "However, when you trace this lecture (python -m edtrace.execute -m lecture_07), we turn off multiprocessing.": "但当我们 trace 这门课时，会临时停用多进程以方便追踪。",
    "Walk through bare-bones implementations of each strategy on deep MLPs.": "通过深层 MLP 对各个并行策略进行最小化代码展示。",
    "Recall that MLPs are the compute bottleneck in Transformers, so this is representative.": "这是极具代表性的，因为 MLP 是 Transformer 模型中最主要的计算开销。",
    "Data parallelism: DDP (all-reduce), FSDP/ZeRO (all-gather + reduce-scatter)": "数据并行：DDP (All-Reduce) 以及 FSDP/ZeRO (All-Gather + Reduce-Scatter)",
    "Tensor parallelism: requires very fast interconnects (e.g., NVLink)": "张量并行：需要极高速的卡间网络 (如 NVLink)",
    "Pipeline parallelism: can work with slow interconnects, but need to work to reduce pipeline bubbles": "流水线并行：能在低带宽网络下运行，但需解决流水线气泡问题",
    "Can re-compute or store in memory or store in another GPUs memory and communicate": "系统设计的永恒权衡：用重算、内存保存还是跨卡通信",
    "Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure": "硬件虽快，但模型的膨胀使层级式分布式架构永远是前沿标配",
    "Collective operations are the conceptual primitives used for distributed programming": "集体通信原语是分布式并行编程的基石",
    "Rank: a particular device/GPU (e.g., 0, 1, 2, 3)": "Rank：标识通信组内的 GPU 卡号",
    "World size: total number of devices (e.g., 4)": "World size：通信组内的 GPU 卡数",
    "Operations:": "通信操作列表",
    "Broadcast, scatter, gather, reduce (foundations)": "基础原语",
    "All-gather, reduce-scatter, all-reduce (workhorse)": "分布式训练的顶梁柱",
    "All-to-all (for MoEs)": "MoE 专属的 All-to-All 路由原语",
    "Broadcast: copy from rank 0 to all ranks": "广播 (Broadcast)：将 Rank 0 的数据同步至所有 Rank",
    "Input": "输入数据",
    "Output": "输出数据",
    "Minor use case: rank 0 loads initial checkpoint and broadcasts to all ranks": "常见用例：Rank 0 读取初始检查点并广播到所有进程",
    "Scatter tensor on rank 0 to all ranks": "分发 (Scatter)：将大张量均匀切分并分发给所有 Rank",
    "Note: stepping stone to understanding reduce-scatter": "它是理解 Reduce-Scatter 的基石",
    "Gather pieces from all ranks to rank 0 (opposite of scatter)": "收集 (Gather)：将所有 Rank 上的分片拼回 Rank 0",
    "Note: stepping stone to understanding all-gather": "它是理解 All-Gather 的基石",
    "Reduce pieces from all ranks to rank 0, applying some operation (e.g., sum, min, max)": "规约 (Reduce)：对所有 Rank 的数据进行聚合运算并保存在 Rank 0",
    "Note: stepping stone to understanding all-reduce": "它是理解 All-Reduce 的基石",
    "All-gather: perform gather to all ranks, not just rank 0": "全收集 (All-Gather)：类似于 Gather 但让所有 Rank 卡均获得拼接大张量",
    "Reduce-scatter: perform reduce on each dimension, scatter results": "规约分发 (Reduce-Scatter)：对维度对应位置执行规约后均匀分发至各个 Rank",
    "All-reduce = reduce-scatter + all-gather": "全规约 (All-Reduce)：在所有卡上完成均值同步",
    "All-to-all: each rank sends each other rank some tensor (most general)": "多对多通信 (All-to-All)：每卡对其他卡各自发送特定数据切片",
    "Useful for MoEs: each rank has split of data and subset of experts; need to route data to experts": "MoE 核心：每卡拥有不同数据分片和专家子集，通过其实现 Token 的专家路由",
    "For balanced splits, all-to-all looks like transpose": "样本均衡分布时，All-to-All 类似于张量转置操作",
    "Also handles unbalanced splits (but want splits to be as balanced as possible)": "也支持非均衡分片，但通常出于均衡负载考虑应尽量样本分配均匀",
    "Way to remember the terminology:": "如何记忆分布式原语的名词：",
    "Reduce: performs some associative/commutative operation (sum, min, max)": "Reduce：应用结合律与交换律的规约计算",
    "Scatter is inverse of gather": "Scatter (分发) 与 Gather (收集) 为逆操作",
    "All: means destination is all devices": "All- 前缀：意味着数据最终同步分发到所有参与计算的卡上",
    "Modern (in the data center):": "现代拓扑（数据中心高性能集群环境）：",
    "Typical setup:": "典型分布式架构拓扑：",
    "GB200/GB300 NVL72: 8 GPUs per tray, 9 trays per rack -> 72 GPUs in one NVLink domain": "GB200/GB300 NVL72 柜机：每盘 8 颗 GPU，共 72 颗 GPU 直接构建单 NVLink 通信域",
    "RDMA over Converged Ethernet (RoCE): Ethernet bypasses CPU, similar but cheaper/weaker than Infiniband, used by Meta": "RoCE 技术：商业以太网上承载 RDMA 流量，便宜且能绕过 CPU",
    "Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe)": "自动感知系统底层网络拓扑结构",
    "Optimizes the path between GPUs": "自动调配最佳的卡间通信流路径",
    "Launches GPU kernels to send/receive data": "调度专用的 CUDA Kernel 自动收发 data，不耗费 CPU",
    "Provides clean interface for collective operations (e.g., all_gather_into_tensor)": "提供了整洁的集体通信 API（例如 `all_gather_into_tensor`）",
    "Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)": "支持多种网络通信后端：gloo (支持 CPU) 以及 nccl (支持 GPU)",
    "Also supports higher-level algorithms (e.g., FullyShardedDataParallel) [not used in this course]": "也封装了诸如 FSDP 的高阶接口（本课程为了教学底层全部从零实现）",
    "Let's walk through some examples.": "让我们来看几个分布式程序运行的实例",
    "All-reduce (dist = torch.distributed)": "集合通信 All-Reduce 同步示例",
    "dist.barrier()  # Waits for all processes to get to this point": "同步栅栏，确保所有分布式进程均到达此点",
    "Both input and output": "作为输入，同时也作为规约结果 of 输出张量",
    "Modifies tensor in place": "就地 (in-place) 修改张量内容",
    "Input": "输入数据",
    "Allocate output": "为输出分配显存空间",
    "Input is the output of reduce-scatter": "将 Reduce-Scatter 的规约结果作为 All-Gather 的输入",
    "How fast does communication happen?": "通信传输速度到底有多快？",
    "Warmup": "通信预热操作",
    "Wait for CUDA kernels to finish": "强行同步，等待 CUDA 卡上所有的内核执行完毕",
    "Wait for all the processes to get here": "barrier 强行同步所有进程，确保起始计时点对齐",
    "Measure the effective bandwidth": "计算网络有效带宽",
    "2x because send + receive, world_size-1 steps in all-reduce": "因为包含发与收的双向通信，且 all-reduce 需要折合 world_size-1 步",
    "Create input and outputs": "创建输入输出",
    "Each rank has a matrix": "各个 rank 各自拥有一份大矩阵",
    "all-reduce = reduce-scatter + all-gather": "全规约 (All-Reduce) = 规约分发 + 全收集",
    "all-reduce moves 2x the data in 2x the time compared to reduce-scatter, so similar bandwidth": "相比 reduce-scatter，all-reduce 移动 2 倍数据、耗费 2 倍时间，因而带宽表现类似",
    "Sharding strategy: each rank gets a slice of the data": "数据并行切分策略：数据按 Batch 均匀分片分布在各卡上",
    "Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory": "下一单元预告：FSDP/ZeRO 消除重复持有完整模型参数 of 显存开销",
    "Get the slice of data for this rank (in practice, each rank should load only its own data)": "提取本卡所分到的数据分片（生产中通常由 Dataloader 各自读取）",
    "Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)": "初始化 MLP 参数（数据并行中各卡均独立初始化相同的完整参数）",
    "Each rank has own optimizer state": "各卡独自维护各自的优化器状态 (如动量)",
    "Sync gradients across workers (ONLY difference between standard training and DDP)": "跨 worker 同步梯度（这是数据并行训练 DDP 与普通单卡训练的唯一差别！）",
    "Sharding strategy: each rank gets part of each layer, transfer all data/activations": "张量并行切分策略：每层的参数按维度被分片，计算时同步激活值",
    "All ranks get the data (batch_size x num_dim)": "所有 worker 卡都会被分配到完整的完整 batch 数据",
    "Shard num_dim": "对通道隐藏维度进行均匀切分",
    "Create model (each rank gets 1/world_size of the parameters)": "创建模型，各个 worker 只存模型参数矩阵的 world_size 分之一",
    "Compute activations (batch_size x local_num_dim)": "计算本卡局部前向输出的激活值",
    "Note: this is only on a slice of the parameters": "注：此矩阵乘法只涉及切分参数的计算",
    "Allocate memory for activations (world_size x batch_size x local_num_dim)": "为 Gather 后的全量中间激活分片预先分配空间",
    "Send activations via all gather": "卡间通过 All-Gather 同步完整激活值",
    "Concatenate them to get batch_size x num_dim": "顺次拼接获得完整前向激活矩阵",
    "Backward pass: homework exercise": "反向传播：留作课后作业",
    "Sharding strategy: each rank gets subset of layers, transfer all data/activations": "流水线并行切分策略：把神经网络深度层均匀分发到不同的 GPU 上",
    "Split up layers": "顺次切分层深",
    "Each rank gets a subset of layers": "各个 worker 各自被分配局部的一部分连续网络层",
    "Break up into micro batches to minimize the bubble": "拆分成 micro batch 隐藏流水线气泡以最大化硬件并发",
    "Get activations from previous rank": "如果不是首卡，则通过 dist.recv 接收前一个 worker 卡发过来的前向激活值",
    "Compute layers assigned to this rank": "执行本 worker 负责的那一部分局部层的前向计算",
    "Send to the next rank": "如果不是末卡，则通过 dist.send 发送前向结果至下一个 worker 卡",
    "Not handled: overlapping communication/computation to eliminate pipeline bubbles": "注：此处未做计算/通信重叠优化，因而无法消除 Pipeline Bubble 泡泡",
    "Backward pass: homework exercise": "反向传播：留作课后作业",
    "Initializes the distributed environment (called at start of process).": "初始化分布式进程组环境",
    "Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)": "设置主节点 Rank 0 的地址与端口以供分布式协商通信",
    "Cleans up the distributed environment (called at end of process).": "释放销毁分布式进程组通信环境",
    "Context manager that temporarily disables distributed functions (replaces with no-ops).": "临时禁用分布式通信函数的上下文管理器（全替换为空操作 no-ops）",
    "Launches world_size processes that each calls func on world_size, args, kwargs.": "并行启动 world_size 个子进程顺次运行 func 函数",
    "This is the normal code path for multiprocessing": "多进程环境下多卡并发计算的通用流程",
    "If we're being traced (inside edtrace), just run the function directly.": "当遇到 edtrace 调试追踪时，退化为单卡单进程直行测试",
    "Create parameters and put them on the rank-th GPU.": "创建参数，并自动移动至指定 Rank 卡显存中",
    "For reproducibility": "设置随机种子以保证实验可复现",
    "Return a / b and throw an error if there's a remainder.": "整除除法，如果不能整除则直接报错",
    "Return the number of seconds required to perform func.": "返回执行 func 函数所需的平均秒数时间",
}

def clean_and_translate_comments(line):
    # Strip any trailing \n
    line = line.rstrip('\n')
    # Remove @inspect / @stepover / @clear annotations from the line
    line = re.sub(r'\s*@inspect\s+[\w\.\,\(\)\-\s]+', '', line)
    line = re.sub(r'\s*@inspect.*$', '', line)
    line = re.sub(r'\s*@stepover', '', line)
    line = re.sub(r'\s*@clear.*$', '', line)
    
    # Also clean up empty comments if they resulted from cleaning inspect
    line = re.sub(r'\s*#\s*$', '', line)
    
    if "#" in line:
        code_part, comment_part = line.split("#", 1)
        comment_part_stripped = comment_part.strip()
        if comment_part_stripped in comments_map:
            comment_part = " " + comments_map[comment_part_stripped]
        else:
            for eng, chi in comments_map.items():
                if eng in comment_part_stripped:
                    comment_part_stripped = comment_part_stripped.replace(eng, chi)
            comment_part = " " + comment_part_stripped
        line = code_part + "#" + comment_part
    return line


def translate_text(t):
    return text_translation_map.get(t, t)

# Evaluate argument value from AST node
def get_arg_val(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return reference_map.get(node.id, node.id)
    else:
        return ast.unparse(node)

# Convert slide statement (text/image/link call) to markdown representation
def parse_slide_statement(node):
    if isinstance(node, ast.Expr):
        val = node.value
        # Handle tuple of calls
        if isinstance(val, ast.Tuple):
            parts = []
            for elt in val.elts:
                parts.append(parse_call(elt))
            return "".join(parts)
        elif isinstance(val, ast.Call):
            return parse_call(val)
    return None

def parse_call(call_node):
    if not isinstance(call_node, ast.Call):
        return ""
    if not isinstance(call_node.func, ast.Name):
        return ""
    
    func_name = call_node.func.id
    if func_name not in ('text', 'image', 'link', 'article_link', 'post_link', 'video_link'):
        return ""
        
    args = [get_arg_val(arg) for arg in call_node.args]
    kwargs = {}
    for kw in call_node.keywords:
        kwargs[kw.arg] = get_arg_val(kw.value)
        
    if func_name == "text":
        return translate_text(args[0])
    elif func_name == "image":
        src = args[0]
        width = kwargs.get("width")
        if width:
            return f'<img src="{src}" width="{width}" />'
        return f'<img src="{src}" />'
    elif func_name == "link":
        if len(args) == 1:
            val = args[0]
            if val.startswith("http"):
                return f"[{val}]({val})"
            return val # reference_map value
        else:
            title = kwargs.get("title", "")
            url = kwargs.get("url", "")
            if len(args) >= 2:
                pass
            return f"[{title}]({url})"
    elif func_name == "article_link":
        return f" [相关文章]({args[0]})"
    elif func_name == "post_link":
        return f" [相关帖子]({args[0]})"
    elif func_name == "video_link":
        return f" [相关视频]({args[0]})"
    return ""

def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()
    
    tree = ast.parse("".join(file_lines))
    
    # Identify imports
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append("".join(file_lines[node.lineno-1 : node.end_lineno]))
            
    # Group imports cell
    cells = []
    if imports:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line for imp in imports for line in imp.splitlines(keepends=True)]
        })
        
    nodes_by_name = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes_by_name[node.name] = node
        elif isinstance(node, ast.ClassDef):
            nodes_by_name[node.name] = node
            
    def is_slide_statement(stmt):
        if not isinstance(stmt, ast.Expr):
            return False
        val = stmt.value
        if isinstance(val, ast.Tuple):
            return all(isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name) and elt.func.id in ('text', 'image', 'link', 'article_link', 'post_link', 'video_link') for elt in val.elts)
        if isinstance(val, ast.Call):
            return isinstance(val.func, ast.Name) and val.func.id in ('text', 'image', 'link', 'article_link', 'post_link', 'video_link')
        return False
        
    current_markdown = []
    current_code = []
    
    def flush_markdown():
        if current_markdown:
            markdown_content = "\n\n".join(current_markdown)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in markdown_content.splitlines()]
            })
            current_markdown.clear()
            
    def flush_code():
        if current_code:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [clean_and_translate_comments(line) + "\n" for line in current_code]
            })
            current_code.clear()
            
    def add_markdown(text):
        if text.strip():
            flush_code()
            current_markdown.append(text)
            
    def add_code_line(line):
        flush_markdown()
        current_code.append(line)
        
    def dedent_line(line, num_spaces=4):
        if line.startswith(" " * num_spaces):
            return line[num_spaces:]
        return line.lstrip(" ") if line.strip() == "" else line

    processed_functions = set()
    
    def trace_function(func_node):
        processed_functions.add(func_node.name)
        for stmt in func_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in nodes_by_name and isinstance(nodes_by_name[stmt.value.func.id], ast.FunctionDef):
                called_func = stmt.value.func.id
                if called_func not in processed_functions:
                    flush_markdown()
                    flush_code()
                    trace_function(nodes_by_name[called_func])
                continue
                
            if is_slide_statement(stmt):
                md = parse_slide_statement(stmt)
                if md:
                    add_markdown(md)
            else:
                stmt_lines = file_lines[stmt.lineno-1 : stmt.end_lineno]
                for l in stmt_lines:
                    add_code_line(dedent_line(l.rstrip('\n'), 4))
                    
    if "main" in nodes_by_name:
        trace_function(nodes_by_name["main"])
        flush_markdown()
        flush_code()
        
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name == "main" or node.name in processed_functions:
                continue
            node_lines = file_lines[node.lineno-1 : node.end_lineno]
            cleaned_node_lines = [clean_and_translate_comments(l) + "\n" for l in node_lines]
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cleaned_node_lines
            })
            
    if "lecture_02" in file_path:
        cells.insert(1, {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CS336: 从头开始构建语言模型 (2026春季)\n",
                "\n",
                "# 第二讲：资源核算 (Resource Accounting)\n"
            ]
        })
    elif "lecture_07" in file_path:
        cells.insert(1, {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CS336: 从头开始构建语言模型 (2026春季)\n",
                "\n",
                "# 第七讲：分布式并行训练 (Parallelism)\n"
            ]
        })
        
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return notebook

def main():
    lectures_dir = "/home/blybq/code-project/cs336/lectures"
    
    for filename in ("lecture_02.py", "lecture_07.py"):
        py_path = os.path.join(lectures_dir, filename)
        ipynb_path = py_path.replace(".py", ".ipynb")
        
        print(f"Processing {py_path}...")
        nb = process_file(py_path)
        
        json_str = json.dumps(nb, indent=1, ensure_ascii=False)
        try:
            verified_nb = json.loads(json_str)
            print(f"Verification successful for {ipynb_path}")
        except Exception as e:
            print(f"Verification FAILED for {ipynb_path}: {e}")
            continue
            
        with open(ipynb_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Wrote notebook to {ipynb_path}")

if __name__ == "__main__":
    main()

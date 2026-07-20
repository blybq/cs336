# 第 5 讲：图形处理器 (GPUs)

CS336
Tatsu H

---

## 第 1 页 (Page 1)

# 第 5 讲
## GPUs

CS336  
Tatsu H  

---

## 第 2 页 (Page 2)

### 大纲与目标 (Outline and goals)
- **让 CUDA 和 GPU 不再神秘 (Make CUDA and GPUs less magic)**
- **理解 GPU 变慢的场景 (Understand when GPUs get slow)**
- **理解如何编写快速算法 (Understand how to make fast algorithms)**

---

## 第 3 页 (Page 3)

### 在开始之前 (Before we start..)
我们要对以下几项优秀的内容来源致以诚挚的感谢：
- Horace He 的个人博客
- CUDA Mode 工作组
- TPU（以及现在的 GPU）书籍 (!!)
- 其它资料来源，包括：[nichijou.co](https://nichijou.co/)，[jonathan-hui.medium.com](https://jonathan-hui.medium.com/)

---

## 第 4 页 (Page 4)

### 今日内容组织 (Organization today)
- **第一部分**：深入探究 GPU —— 它们是如何工作的以及重要组成部分
- **第二部分**：理解 GPU 的性能表现
- **第三部分**：融会贯通 —— 剖析 FlashAttention 机制

---

## 第 5 页 (Page 5)

### 引入主题：计算量带来可预测的性能提升 (Setting the stage: compute leads to predictable perf)
- 通常情况下，计算量为语言模型带来可预测的性能增益。
- 仅靠更快的硬件、更好的利用率、以及更优的并行手段即可推动技术进步（就目前而言）。

*(参考文献：Kaplan et al. Neural Scaling Laws)*

---

## 第 6 页 (Page 6)

### 我们如何获得算力规模的扩充？早期——唐纳德缩放定律 (How do we get compute scaling? Early on – Dennard scaling)
- 传统的芯片缩放模式（Dennard 缩放定律）在 1980 至 2000 年代就已经走到尽头。
- 我们该如何继续满足大语言模型（LLMs）对算力无止境的胃口？

---

## 第 7 页 (Page 7)

### 并行缩放仍在继续 (Parallel scaling continues)
- 基于 GPU 的并行算力在过去 10 年里提升了超过 1000 倍。
- **没有 GPU 算力的提升，就没有 LLM 的规模扩展。**

*(引自 Bill Dally，HotChips 主题演讲)*

---

## 第 8 页 (Page 8)

### GPU 与 CPU 有什么区别？ (How is a GPU different from a CPU?)
- **CPU** 针对少数且高速执行的线程进行优化；而 **GPU** 针对海量的并发线程进行优化。
- GPU 拥有很多个微型计算单元（ALUs）。对分支结构（control、cache）的支持则少得多。
- **CPU 优化延迟**（确保每个线程能以最快速度完成）。
- **GPU 优化吞吐量**（追求在单位时间内处理的总数据量）。

---

## 第 9 页 (Page 9)

### GPU 内部剖析：执行单元 (Anatomy of a GPU (execution units))
- 每个流式多处理器（SM）中包含许多个流处理器（SPs），它们能并行地执行各个“线程 (threads)”。
- GPU 拥有许多流式多处理器（SMs），它们能独立地去执行各个“线程块 (blocks)”（即分配的工作任务）。

---

## 第 10 页 (Page 10)

### GPU 内部剖析：内存 (Anatomy of a GPU (memory))
- **内存距离 SM 越近，其存取速度就越快** —— L1 缓存与共享内存（shared memory）位于流式多处理器（SM）的内部。L2 缓存集成在芯片上，而全局内存（Global memory / HBM）则是 GPU 旁边的内存颗粒。
- SRAM（共享/缓存内存）极其昂贵（贵约 100 倍），但比 DRAM（全局内存）快大约 8 倍。

#### 各种内存访问延迟（TABLE IV）
- **全局内存 (Global memory)**：290 周期 (cycles)
- **L2 缓存 (L2 cache)**：200 周期 (cycles)
- **L1 缓存 (L1 cache)**：33 周期 (cycles)
- **共享内存 (Shared Memory (ld/st))**：(23/19) 周期 (cycles)

---

## 第 11 页 (Page 11)

### GPU 执行模型 (Execution model of a GPU)
GPU 执行模型中有三个核心角色：
- **线程 (Threads)**：线程在并行计算中“实际干活” —— 所有线程执行相同的指令，但接受不同的输入（单指令多线程 SIMT 模式）。
- **线程块 (Blocks)**：线程块是线程的分组。每个线程块在流式多处理器（SM）上运行，并拥有它专属的共享内存（shared memory）。
- **线程束 (Warp)**：线程束由 32 个连续编号的线程组成。它们总是作为一个整体进行协同执行。

---

## 第 12 页 (Page 12)

### GPU 内存模型 (Memory model of a GPU)
- 每个线程均能访问其专属的寄存器（register），以及当前线程块内的共享内存。
- **跨越线程块的数据交互需要通过全局内存（global memory）进行读写，这非常慢。**

#### 设备端代码（Device code）可以：
- 读写线程专属的寄存器 (registers)
- 读写线程专属的局部内存 (local memory)
- 读写线程块专属的共享内存 (shared memory)
- 读写网格（grid）专属的全局内存 (global memory)
- 只读网格（grid）专属的常量内存 (constant memory)

#### 主机端代码（Host code）可以：
- 在主机与网格全局内存及常量内存之间进行数据流传输

---

## 第 13 页 (Page 13)

### 支线讨论：TPU 怎么样？ (Side thread – What about TPUs?)
- 从宏观上看，GPU、TPU 以及许多其他加速器都是相似的。
- **核心架构**：轻量级控制系统、巨大且高速的矩阵乘法单元（matmul unit）、高速内存。
- **不同点**：加速器之间的互连组网方式（将在并行计算章节展开）；没有线程束（Warp）的概念，只有线程块（Blocks）—— 这在矩阵乘法与非矩阵乘法任务中存在折衷。
- **GPU 拥有更多的 SM，TPU 拥有更少的 Tensor Core (但二者的 matmul 性能接近)。**

---

## 第 14 页 (Page 14)

### 支线讨论：TPU 与 GPU 对应关系

| GPU | TPU | 说明 |
| :--- | :--- | :--- |
| 流式多处理器 (SM) | Tensor Core (张量核心) | 包含其他计算单元的核“单元” |
| 线程束调度器 (Warp Scheduler) | VPU 槽 (slots) | SIMD 向量算术单元 |
| CUDA Core | VPU ALU | SIMD ALU 算术逻辑单元 |
| SMEM (L1 缓存) | VMEM | 片上高速缓存内存 |
| Tensor Core | MXU | 矩阵乘法单元 |
| HBM (即 GMEM) | HBM | 高带宽大容量内存 |

#### 硬件规模对比（NVIDIA H100 vs Google TPU v5p）
- **SM / Tensor Core 数量**：132 vs 2
- **线程束调度器 / VPU 槽数**：528 vs 8
- **SMEM / VMEM 大小**：32MB vs 128MB
- **寄存器 / 向量寄存器大小**：32MB vs 256KB
- **Tensor Core / MXU 数量**：528 vs 8

*(数据源自：[jax-ml.github.io/scaling-book/gpus](https://jax-ml.github.io/scaling-book/gpus/))*

---

## 第 15 页 (Page 15)

### GPU 模型的优势 (Strengths of the GPU model)
- 易于扩展困难的任务负载（仅需增加更多的 SM 即可）。
- 由于采用单指令多线程（SIMT）模型，编程相对简单。
- 线程非常“轻量”，能够实现无摩擦的暂停与启动。

---

## 第 16 页 (Page 16)

### GPU 作为高速矩阵乘法器 (GPUs as fast matrix multipliers)
- 在 NVIDIA GPU 的早期，研究人员通过可编程着色器（Programmable Shaders）以奇技淫巧去实现矩阵乘法（matmul）。

---

## 第 17 页 (Page 17)

### 矩阵乘法硬件的发展带来了专属高速的体验 (New matmul hardware means matmuls are fast and special)
- 张量核心（Tensor Cores，从 V、T 系列 GPU 起引入）是专属的硬件级矩阵乘法电路。
- **矩阵乘法的计算速度比其它普通浮点计算操作要快 10 倍以上！**

---

## 第 18 页 (Page 18)

### 算力增长速度远超内存读写速度的增长 (Compute scaling is faster than memory scaling)
- 算力（FLOPS）扩展速度显著快于内存带宽，我们很难保证有充足的数据持续喂给计算单元！
- [理解内存墙问题 (RiseLab)](https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8)
- 硬件 FLOPs 增长率：20 年增长 60000 倍 ($\sim 3.0\times$ 每 2 年)
- DRAM 带宽增长率：20 年增长 100 倍 ($\sim 1.6\times$ 每 2 年)
- 互连带宽增长率：20 年增长 30 倍 ($\sim 1.4\times$ 每 2 年)

---

## 第 19 页 (Page 19)

### 第一部分回顾：GPU 是什么以及它们是如何工作的
- **GPU 具有极高的并行度**：相同的指令被分发给海量的工作单元同步执行。
- **计算能力（特别是矩阵乘法）的增长速度快于内存存取。**
- 我们**必须尊重并合理利用内存层级结构**，才能让代码飞速运行。

---

## 第 20 页 (Page 20)

## 第二部分：优化 GPU 上的机器学习负载 (Part 2: Making ML workloads fast on a GPU)
即使对于像方阵乘法（Square Matmul）这样看似简单的操作，GPU 上的性能表现也可以非常复杂。

---

## 第 21 页 (Page 21)

### 是什么决定了 ML 任务的运行速度？
#### Roofline 模型 (The roofline model)
- **本节核心内容：我们该如何避免陷入内存受限（memory bound）的瓶颈？**

---

## 第 22 页 (Page 22)

### 我们该如何加速 GPU 的运行？
1. **控制分支控制分流** (Control divergence，这并非内存瓶颈)。
2. **低精度计算** (Low precision computation)。
3. **算子融合** (Operator fusion)。
4. **重计算 / 重算** (Recomputation)。
5. **内存合并访问** (Coalescing memory)。
6. **分块 / 瓦片化** (Tiling)。

---

## 第 23 页 (Page 23)

### 优化 1：控制分支分流 (Control divergence - 非内存瓶颈)
- GPU 按照单指令多线程（SIMT）模型运转 —— 同一个线程束中的每一个线程都必须在同一时刻执行完全相同的指令。
- 允许使用条件分支语句（if-else），但它们会因 GPU 执行机制而引入极大的执行开销。

---

## 第 24 页 (Page 24)

### 优化 2：低精度计算 (Low precision computation)
- **数据位数越少，需要移动和搬运的数据量就越小。**

---

## 第 25 页 (Page 25)

### 低精度能显著改善计算强度 (Low precision improves arithmetic intensity)
以长度为 $n$ 的向量进行逐元素 ReLU 操作 ($x = \max(0, x)$) 为例：

#### Float 32 情况：
- **内存读写**：1 次读取 ($x$)，如果 $x < 0$ 则有 1 次写入。Float 32 每次操作需要移动 4 字节。
- **计算操作**：1 次比较操作，1 次 FLOP。
- **计算强度**：8 字节 / FLOP。

#### Float 16 情况：
- **内存读写**：1 次读取 ($x$)，如果 $x < 0$ 则有 1 次写入。Float 16 每次操作需要移动 2 字节。
- **计算操作**：1 次比较操作，1 次 FLOP。
- **计算强度**：4 字节 / FLOP。

---

## 第 26 页 (Page 26)

### 低精度推动矩阵乘法大幅提速 (Low precision drives faster matrix multiplies)
现代 GPU 中的绝大多数计算任务，都是通过低精度或混合精度（Mixed Precision）操作在张量核心（Tensor Cores）上完成提速的。

---

## 第 27 页 (Page 27)

### 低精度的前沿发展 (Frontiers in low precision)
- **超低精度（FP8）**带来的不同取舍。
- 引入**多重缩放因子 MXFP8 (Blackwell 架构)**：
  - 采用 E4M3（保留更多尾数位）和更多的缩放因子以保持精度。
  - 缩放因子本身也是 FP8 (E8M0) 类型，且每 32 个元素共享 1 个缩放因子。
  - 在此架构下，矩阵转置操作变得不再平凡（nontrivial）！

---

## 第 28 页 (Page 28)

### 实际开发中的 MXFP8 训练 (MXFP8 training in practice)
在实践中，并非所有的权重都采用 MXFP8 格式，转置操作也需要独立进行量化。

*(数据源自：[arXiv:2506.08027](https://arxiv.org/html/2506.08027v2))*

---

## 第 29 页 (Page 29)

### 低精度的前沿发展 (续)
#### MXFP4：
- 这代表了你能用极小位数表示的所有可能值！每 16 个值共享 1 个缩放因子，使用 E4M3 作为缩放因子的数据格式。

---

## 第 30 页 (Page 30)

### 优化 3：算子融合 (Operator fusion)
- 我们可以将 GPU 想象成一座工厂 —— 所有的计算输入来自仓库（内存），然后搬运至厂房（计算单元）进行加工处理。
- 厂房的加工速度（算力）在不断攀升，但搬运速度（内存）却没有跟上。

*(图表来自 Horace He 的个人博客 [horace.io](https://horace.io/brrr_intro.html))*

---

## 第 31 页 (Page 31)

### 算子融合减少内存读写 (Operator fusion to minimize memory access)
当我们需要执行多项计算任务时，频繁在仓库和厂房之间搬运中间产物会显得十分愚蠢。
- **非融合方案（Naive / Non-fused）**：多次计算，多次往返全局内存。
- **融合方案（Fused Kernel）**：数据加载一次，在片上完成全部计算后一次性写回。

---

## 第 32 页 (Page 32)

### 算子融合实例：正弦与余弦计算 (Example – sines and cosines)
计算 $\sin^2 x + \cos^2 x$ 如果以最直白的方式书写，将会在 GPU 上发起 5 次 CUDA Kernel 的调用，产生大量的中间数据搬运。

*(数据源自：[towardsdatascience.com](https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26))*

---

## 第 33 页 (Page 33)

### 融合示例 (Fusion example)
上述的 5 个逐元素算子可以被完全融合成 1 个 CUDA Kernel 调用。  
这类“简单的”算子融合现在能够通过编译器（如 `torch.compile`）自动完成。

---

## 第 34 页 (Page 34)

### 优化 4：重计算 (Trick 3: recomputation)
- 在反向传播（Backpropagation）算法中，我们需要在内存中保留前向传播计算出的激活值（图中黄色节点），并在后向计算时计算雅可比矩阵（图中绿色节点）。

*(引自 Stanford cs221)*

---

## 第 35 页 (Page 35)

### 保存（及调用）激活值可能会极其昂贵！ (Storing (and retrieving) activations can be expensive!)
假设我们将 3 个 Sigmoid 函数层层叠加：
- 每次前向计算和后向计算会频繁对中间层产生的激活值进行读写。
- 这会导致 8 次内存存取操作，计算强度极低，造成严重的性能损耗。

*(数据源自：[PyTorch Dev-Discuss](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467))*

---

## 第 36 页 (Page 36)

### 丢弃激活值，需要时重新计算！ (Throw away the activations, re-compute them!)
丢弃部分前向传播的激活值并在反向传播时重算，可以让整体的内存存取次数降低至原先的 5/8，在许多场景下这反而是最优的性能选择！

---

## 第 37 页 (Page 37)

### 优化 5：内存合并存取与 DRAM (Memory coalescing and DRAM)
DRAM（全局内存）是以“突发模式 (burst mode)”进行数据读取的 —— 每次读取操作，实际上都会捎带返回许多字节！
- 内存地址空间被划分为不同的突发块（Burst Sections）。
- 无论何时访问其中的某一个地址，该分块内的所有其它地址数据都会被一同送达处理器。
- 在实际应用中，我们拥有至少 4GB 的地址空间，每次突发获取的数据块大小往往在 128 字节或更多。

*(引自 [CSDN 博文](https://blog.csdn.net/xll_bit/article/details/117702476) 及 [YouTube 视频](https://www.youtube.com/watch?v=9BjVUmaXaCQ))*

---

## 第 38 页 (Page 38)

### 内存合并访问 (Memory coalescing)
- 当同一个线程束（Warp）内的所有 32 个线程的访存请求正好落入同一个内存突发块（Burst Section）内时，访存就被成功合并（coalesced）。
- 提示：一个线程束的 32 个线程是在流式多处理器（SM）中协同执行并同步发出访存请求的。

---

## 第 39 页 (Page 39)

### 矩阵乘法中的合并访问 (Coalescing for matrix multiplication)
对于按行优先（Row-Major）存储的矩阵：
- 如果线程束中的线程沿着矩阵行方向进行非合并访存，将会引发巨大的带宽浪费，因为每次移步都需要从全局内存中加载全新的突发块。

---

## 第 40 页 (Page 40)

### 优化 6：分块技术 (Trick 5: tiling)
**分块（Tiling）是合并与重新排列线程的访存顺序，以此最大程度减少全局内存读取的核心思想。**

- 再次回到矩阵乘法中：
- 简单的乘法计算会导致访存请求杂乱无序且不符合合并访问规范，导致相同的数据（如 $M_{0,0}$ 和 $N_{1,0}$）被重复地从全局内存读取多次。

---

## 第 41 页 (Page 41)

### 分块 —— 在共享内存中存储并复用数据 (Tiling – store and reuse information in shared memory)
将大矩阵切分为更小的“方块 (tiles)”，并在计算时整体一次性读取入共享内存中。  
按照分阶段（Phases）运行矩阵乘法：
1. 将矩阵 $M_{0,0}$ 和 $N_{0,0}$ 切片加载入共享内存（SHM）。
2. 计算生成结果矩阵 $P$ 的部分累加和（完成一个分块）。
3. 接着将 $M_{0,0}$ 和 $N_{2,0}$ 切片加载入共享内存。
4. 依次类推……
- **优势**：重复读取的数据直接在片上共享内存内完成，避免了对慢速全局内存的多次请求。

---

## 第 42 页 (Page 42)

### 分块背后的数学 (Tiling math)
- **非分块矩阵乘法**：每个输入元素必须从慢速全局内存中重复读取 $N$ 次。
- **分块矩阵乘法**：如果分块尺寸为 $T$，每个输入元素只需从全局内存读取 $N/T$ 次，在共享内存上读取 $T$ 次。这为全局内存存取带来了 $T$ 倍的庞大降幅！

---

## 第 43 页 (Page 43)

### 分块可能引入的复杂情况 (Complexities with tiling)
- 分块的方块大小（Tile sizes）可能无法整除矩阵的维度，这会导致“尾部块占不满（tile quantization）”而导致硬件利用率降低。
- 影响分块选择的关键要素：
  - 合并访存的物理限制
  - 共享内存空间的总容量
  - 矩阵维度的可除性

*(引自 [NVIDIA 开发者指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#tile-quant))*

---

## 第 44 页 (Page 44)

### 分块复杂性之二 —— 内存对齐 (Complexities with tiling 2 – memory alignment)
- 内存是以突发数据包（Bursts）为边界进行传输的。
- 只有当分块的内存边界恰好与矩阵在内存中的对齐突发相吻合时，数据加载效率才最高。
- 如果维度没有对齐，可能需要通过向矩阵添加填充（Padding）以辅助实现对齐存取。

---

## 第 45 页 (Page 45)

### 融会贯通：理解一个“矩阵神秘现象” (Putting it together: understanding a matrix mystery)
Andrej Karpathy 在推特上分享道：
> “迄今为止对 nanoGPT 最具戏剧性的优化（实现约 25% 的提速）其实非常简单：仅仅将词表大小从 50257 增大到 50304（最接近 64 的整数倍）。这虽然多算了一些无用维度，但成功让 GPU 走入了一条完全不同的、占用率（occupancy）高得多的算子通道。玩 2 的幂次时请千万保持谨慎。”

**为什么矩阵变大了，运行速度反倒变快了？**

*(本节内容深入剖析：[thonking.ai](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications))*

---

## 第 46 页 (Page 46)

### 矩阵神秘现象 (Matrix mystery)
我们已经理解了其中蕴含的原理（计算强度、分块对齐）。让我们做进一步研究……

---

## 第 47 页 (Page 47)

### 第一部分：分块对齐 (Part 1: tiling)
分块是否能与内存的物理边界合理对齐，会对最终的计算效率产生决定性的影响。

---

## 第 48 页 (Page 48)

### 第二部分：波次量化问题 (Part 2: wave quantization)
为什么会出现周期性的速度波动？
- 例如在 1792 和 1793 的尺寸切换间会出现极大的性能差异。
- **原因分析**：假定使用 $256 \times 128$ 的分块尺寸：
  - 1792 尺寸下：有 $\frac{1792}{256} \times \frac{1792}{128} = 7 \times 14 = 98$ 个任务块。
  - 1793 尺寸下：则需要 $8 \times 15 = 120$ 个任务块。
- A100 GPU 拥有 108 个流式多处理器（SMs），它能在一个计算波次（Wave）中处理 98 个块，但面临 120 个块时，就必须启动第二波次（Wave 2），让大部分 SM 闲置以等待剩下的 12 个块完成。

---

## 第 49 页 (Page 49)

### 第二部分回顾：ML 负载加速秘籍 (Recap of part 2: making ML workloads go fast)
- **减少内存读写**：
  - 合并访存 (Coalescing)
  - 算子融合 (Fusion)
- **将数据置于片上高速共享内存中**：
  - 分块 (Tiling)
- **在内存存取与计算精度/计算量之间进行置换**：
  - 量化 (Quantization)
  - 重计算 (Recomputation)

---

## 第 50 页 (Page 50)

## 第三部分：Flash Attention 注意力机制解析 (Part 3: Understanding Flash Attention)
由 Dao et al. 提出的 Flash Attention 机制极大地加速了注意力机制的计算，这背后的核心原理是什么？

---

## 第 51 页 (Page 51)

### 注意力计算流程回顾 (Recap of attention computation)
标准的注意力机制计算包含 3 个大矩阵乘法（分别对应 $K, Q, V$），并在中间穿插 Softmax 归一化。

---

## 第 52 页 (Page 52)

### 注意力分块之一：对 KQV 矩阵乘法进行分块 (Tiling part 1: tiling for the KQV matrix multiply)
- 论文中的图 1 实质上就是针对注意力机制中矩阵乘法的分块存取流程。
- **但关键在于，我们应该如何对夹在中间的 Softmax 实施分块计算？**

---

## 第 53 页 (Page 53)

### 注意力分块之二：Softmax 的渐进式增量计算 (Tiling part 2: incremental computation of the softmax)
为了能够以方块（Tile-by-Tile）为单位渐进式地计算 Softmax，我们需要采用以下方案：
- **在线 Softmax 算法（Online Softmax）**：在迭代过程中动态更新当前观测到的最大值，并通过数学变形展开为伸缩级数形式进行局部累加。这使我们可以在不需要完整观测所有数据前，就以分块形式完成 Softmax 归一化的中间计算。

*(参考文献：Mikailov and Gimelshein 2018)*

---

## 第 54 页 (Page 54)

### 融会贯通：Flash Attention 的前向传播流程 (Putting it all together – the forward pass of flash attention)
从 Dao 2023 论文中，我们能清晰地提炼出三个优化支柱：
1. **分块计算**：针对中间乘积项（$S$）实施 Tile-wise 级的分块计算。
2. **算子融合**：将指数运算与其它计算操作融为一体。
3. **在线 Softmax**：通过伸缩级数技巧，分块渐进式地完成 Softmax 归一化。
- *(注：本讲中我们不涉及反向传播，但在反向时同样是按分块动态重计算激活值。)*

---

## 第 55 页 (Page 55)

### 全课总结 (Recap for the whole lecture)
- **硬件的物理限制决定了规模的上线**：低层级的技术对齐，决定了哪些架构在大规模训练中可行，哪些不可行。
- 现代以 GPU 为基石的算力生态，强力约束着我们必须深入思考 **“矩阵乘法”** 与 **“数据搬运”**。
- 精准尊重并响应 GPU 的硬件本性（内存合并访问、分块复用、算子融合）是获取顶级模型运行效率的唯一通道。

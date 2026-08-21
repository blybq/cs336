# CS336 课程笔记：10. 大模型推理与加速优化（Model Inference）

## 1. 推理场景与核心评估指标
大模型推理是在模型参数固定（Frozen）的前提下，根据用户的输入 Prompt 实时生成 Response 的过程。相比于一次性的预训练成本，推理成本是随着用户请求规模和调用频次成正比持续产生的运营成本。

### 核心评估指标
1. **首字延迟（Time to First Token, TTFT / $T_{\text{ttft}}$）**：
   - 指用户发送 Prompt 到系统输出第一个 Token 的等待时间。它主要由**预填充阶段（Prefill Phase）**决定。在聊天或代码补全等交互式场景中，TTFT 是决定用户体验的首要指标。
2. **字间延迟（Inter-Token Latency）**：
   - 指在首字输出后，后续生成每个 Token 的平均时间（倒数即为单个请求的生成速率 Tokens Per Second）。它主要由**解码阶段（Decoding Phase）**决定。通常需要快于人类的阅读速度（如 > 20 Tokens/s）。
3. **吞吐量（Throughput）**：
   - 指系统平均每秒能生成的 Token 总数。对于离线批量任务（如数据清洗、自动化评测），系统吞吐量是核心关注指标。**注意：高吞吐量不等于低延迟**。为了最大化利用硬件资源，我们往往会增大 Batch Size，这会极大地提高系统总吞吐量，但也会成倍增加单个请求的字间延迟。

---

## 2. 算术强度与硬件瓶颈（计算受限 vs. 内存受限）
要透彻理解为什么大模型推理极慢且昂贵，必须引入**算术强度（Arithmetic Intensity）**和**屋顶模型（Roofline Model）**。

### 2.1 算术强度定义
$$\text{算术强度} = \frac{\text{浮点运算次数 (FLOPs)}}{\text{内存访问字节数 (Bytes)}}$$

以最基础的矩阵-向量乘法为例：输入向量 $X \in \mathbb{R}^{1 \times D}$ 与权重矩阵 $W \in \mathbb{R}^{D \times F}$ 相乘。使用半精度精度（FP16/BF16，每个元素占用 2 字节）：
- **浮点计算量**：乘法与加法算作 2 次 FLOPs，因此总 FLOPs 为 $2 \times D \times F$。
- **内存读取量**：需要从全局显存（HBM）读取权重矩阵 $W$ 和向量 $X$。权重大小为 $2 \times D \times F$ 字节，向量大小为 $2 \times D$ 字节。
- **内存写入量**：输出向量大小为 $2 \times F$ 字节。
- **总算术强度**：
  $$\text{Intensity} = \frac{2DF}{2DF + 2D + 2F} \approx 1 \quad (\text{当 } D, F \text{ 很大时})$$

**结论**：在单批次（Batch Size $B = 1$）自回归解码时，每个步骤的矩阵-向量乘法算术强度极低，接近于 1。这意味着每进行 1 次浮点运算，就需要从显存中搬运约 1 字节的数据。

### 2.2 硬件瓶颈与 Roofline Model 划分（以 H100 GPU 为例）
- **H100 Tensor Core 峰值计算性能**：$\approx 989 \times 10^{12}$ FLOPs/s (BF16)
- **H100 HBM3 显存带宽**：$\approx 3.2 \times 10^{12}$ Bytes/s
- **硬件分界线强度（Hardware Intensity Bound）**：
  $$\text{Hardware Intensity Bound} = \frac{\text{峰值算力}}{\text{内存带宽}} = \frac{989 \times 10^{12} \text{ FLOPs/s}}{3.2 \times 10^{12} \text{ Bytes/s}} \approx 309 \text{ FLOPs/Byte}$$

- **屋顶模型判断准则**：
  - **计算受限（Compute-Bound）**：若算法的 $\text{算术强度} > 309$，GPU 的计算单元能被充分填满，实际性能受限于 GPU 算力极限。
  - **内存受限（Memory-Bound）**：若算法的 $\text{算术强度} < 309$，GPU 计算单元大部分时间在闲置等待数据搬运，实际性能受限于显存带宽限制。
  - **推理痛点**：在 $B=1$ 的自回归解码中，算术强度约为 1，远远低于 309 的硬件分界线。因此，**大模型解码阶段是极其典型的内存受限（Memory-Bound）任务**。

---

## 3. 自回归推理的双阶段工作负载与 KV Cache

自回归语言模型（Autoregressive LLM）生成文本时，经历两个完全不同的计算阶段。

### 3.1 预填充阶段（Prefill Phase）
- **任务**：一次性处理用户输入的全部 Prompt 序列（长度为 $S$），计算所有 Input Tokens 的表征，并输出第一个生成的 Token。
- **特征**：由于可以并行计算 Prompt 中 $S$ 个 Token 的交叉注意力，且 Batch Size 在序列长度维度上等效于 $S$，因此其算术强度正比于 $S$。该阶段属于**计算受限**，GPU 算力利用率极高。

### 3.2 解码生成阶段（Decoding Phase）
- **任务**：逐个 Token 自回归地生成后续内容。第 $t$ 步的输入仅为第 $t-1$ 步生成的**单个 Token**。
- **特征**：这是极度**内存受限**的。每一生成步中，GPU 必须把巨大的模型参数（如 70B 模型在半精度下约 140GB）从 HBM 完整加载到 SRAM 缓存中，而仅仅是为了给这一个 Token 的向量做一次前向传播。

### 3.3 KV Cache 详解与推导
为了避免每次生成新 Token 时，都重复计算从第 1 个 Token 到当前 Token 的 Key 和 Value 向量（如果不做缓存，第 $T$ 步计算复杂度将呈 $O(T^2)$ 爆炸式增长），我们采用 **KV Cache（键值缓存）** 技术。即在预填充和解码过程中，将历史 Token 的 Key 和 Value 向量持久保存在显存中，每次新生成步仅需计算最新 Token 的 Key/Value 并拼接到缓存后。

#### KV Cache 显存占用公式的逐步推导：
我们来推导存储整个 KV Cache 所需的显存字节数：

1. **单个 Token、单个 Layer、单个 Attention Head** 对应的 Key 和 Value 向量维度均为 $d_{\text{head}}$。
2. 因此，需要存储的浮点数数量为：
   $$2 \times d_{\text{head}}$$
   *(其中 $2$ 代表 Key 向量和 Value 向量各一个)*
3. 假设使用半精度（FP16 或 BF16）格式存储，每个浮点数占用 2 字节（Bytes）。那么，对应的显存字节数为：
   $$2 \times 2 \times d_{\text{head}} \text{ bytes}$$
   *(前一个 $2$ 表示 K 和 V，后一个 $2$ 表示每个半精度数值占 2 字节)*
4. 若每一层包含 $n_{\text{heads}}$ 个注意力头（更准确地说是 KV 头，在 GQA 中 $n_{\text{heads}} = n_{\text{KV\_heads}}$），则单层所需的显存为：
   $$2 \times 2 \times n_{\text{heads}} \times d_{\text{head}} \text{ bytes}$$
5. 若模型总共包含 $n_{\text{layers}}$ 个 Transformer 层，则单个 Token 在模型所有层中累积的 KV 显存为：
   $$2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \text{ bytes}$$
6. 随着生成的进行，当序列总长度达到 $L$（包含 Prompt 长度和已生成的 Token 数）时，该请求累积的 KV 缓存为：
   $$2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L \text{ bytes}$$
7. 当系统并发处理批次大小为 $n_{\text{batch}}$ 的请求时，总的 KV Cache 显存占用公式为：
   $$\text{Memory}_{\text{KVCache}} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L \times n_{\text{batch}} \text{ bytes}$$

#### 物理含义清单：
- **第一个 `2`**：分别对应 Key Cache 和 Value Cache。
- **第二个 `2`**：半精度浮点格式（FP16/BF16）占用的字节数（2 Bytes）。如果使用 FP8，该项将变为 1；如果使用 INT4 量化，该项将降为 0.5。
- **$n_{\text{layers}}$**：Transformer 模型中包含 Multi-Head Attention 的网络总层数。
- **$n_{\text{heads}}$**：注意力头数。注意在 GQA 中，该项为实际的 KV 组数 $n_{\text{KV\_heads}}$，比 Query 头数少得多。
- **$d_{\text{head}}$**：注意力头的投影特征维度（通常为 $\text{hidden\_size} / \text{num\_attention\_heads}$）。
- **$L$**：当前上下文的序列总长度。
- **$n_{\text{batch}}$**：当前批次中并发的请求总数。

**注意：Attention 层的内存受限死结**
虽然随着并发 $n_{\text{batch}}$ 增大，MLP 层可以通过共享权重矩阵使算术强度线性增长（逐渐转化为计算受限）；然而，在 Attention 层中，每个请求都需要读取自己独占的、互不共享的 KV Cache 显存。读取 KV 缓存的显存带宽消耗与 $n_{\text{batch}}$ 呈同步等比例上升，导致分母的访存量和分子的计算量被完全抵消。因此，**Attention 层的解码计算无论如何增大批次，都无法摆脱内存受限的死结**。

---

## 4. PagedAttention 机制与虚拟内存类比
传统的 LLM 推理框架（如早期的 Hugging Face）在为请求分配 KV Cache 显存时，面临着严重的内存碎片和利用率低下问题。

### 4.1 传统连续显存分配的缺陷
由于模型在生成前无法预测最终的输出 Token 长度，推理引擎只能保守地为每个并发请求预先分配一段**连续的显存空间**，其大小等于模型的最大生成上限（如 $L_{\text{max}} = 2048$）。这带来了三个显存黑洞：
1. **预留显存浪费（Reservation Waste）**：为可能生成的未来 Token 预留的空间，但在生成结束前一直处于闲置状态。
2. **内部碎片（Internal Fragmentation）**：请求实际在生成了 100 个 Token 之后就提前终止了，但被系统强行占用了 2048 大小的显存，多余的 1948 个位置的空间被彻底浪费。
3. **外部碎片（External Fragmentation）**：不同请求动态启动和结束，导致显存空间被切碎成不连续的小片段，无法分配给新的大批次请求。

### 4.2 PagedAttention 的虚拟内存映射方案
vLLM 借鉴了操作系统中的**虚拟内存分页管理（Virtual Memory Paging）**思想，提出了 **PagedAttention**。

![PagedAttention 逻辑块与物理块非连续页映射关系](images/paged_attention_mapping.drawio.png)

- **工作机制**：
  1. **固定大小物理页**：将每个请求的 KV Cache 划分为不连续的固定大小的物理页块（Physical Blocks，例如每个 Block 默认包含 16 个 Token 的 KV 向量）。
  2. **页表映射**：维护一个逻辑到物理的**页表（Page Table）**。对于一个请求，其逻辑序列中的连续 Token，被映射到物理显存中任意分散、不连续的物理 Block 上。
  3. **按需动态分配**：当一个 Block（16 个 Token）被填满时，系统才向显存申请分配下一个物理 Block。
- **消除碎片与共享（Copy-on-Write）**：
  - PagedAttention 彻底消除了 Reservation 浪费和内部碎片（除了最后一个 Block 可能会有极少字节的未装满浪费），显存利用率飙升至 96% 以上，从而可以将并发 Batch Size 扩大数倍。
  - **写时复制（Copy-on-Write）**：在处理多路分支生成（如 Parallel Sampling 或 Beam Search 共享同一个 System Prompt 前缀）时，页表可以让多个逻辑 Block 共享指向同一个物理 Page 内存。仅当某个分支开始分叉、生成特有 Token 时，系统才复制该 Page，从而极大地节省了显存。

---

## 5. 投机解码（Speculative Decoding）与数学原理

为了突破自回归解码受限于内存带宽的瓶颈，DeepMind 等机构提出了**投机解码（Speculative Decoding）**。它巧妙地利用了**“验证（Prefill）一段序列的速度远快于自回归逐字生成该序列的速度”**这一硬件不对称特性。

### 5.1 基本工作流程
1. **草稿阶段（Drafting）**：使用一个极其轻量、推理极快的**小草稿模型（Draft Model, $p$）**，自回归地串行向前预测 $K$ 个候选 Token，形成一个草稿序列：
   $$\tilde{X} = (\tilde{x}_1, \tilde{x}_2, \dots, \tilde{x}_K)$$
   *(由于草稿模型参数量极小，自回归推理耗时极短)*
2. **验证阶段（Verification）**：将这 $K$ 个草稿 Token 一次性打包送入**目标大模型（Target Model, $q$）**进行单次前向传播。
   *(因为是一次性输入 $K$ 个 Token，此时大模型处于计算受限的 Prefill 状态，运行速度非常快)*
3. **拒绝采样验证**：大模型会输出每个位置的概率分布 $q(x_i | x_{<i})$。我们利用修正拒绝采样算法（Rejection Sampling），来决定接受或拒绝草稿模型的推荐。

### 5.2 拒绝采样的数学概率接受法则
为了确保投机解码的输出概率分布与**直接使用大模型采样的分布完全一致（数学上无偏，保证生成质量完全不降级）**，我们使用如下接受概率公式：

对于第 $i$ 个草稿 Token $\tilde{x}_i$（$i = 1, \dots, K$）：
1. **接受概率**：
   $$\alpha_i = \min\left(1, \frac{q(\tilde{x}_i | x_{<i})}{p(\tilde{x}_i | x_{<i})}\right)$$
   我们生成一个均匀分布随机数 $u \sim \text{Uniform}(0, 1)$。若 $u < \alpha_i$，则接受该 Token 并将其固定在生成序列中，继续验证第 $i+1$ 个。
2. **拒绝与重采样**：
   一旦在第 $j$ 步发生拒绝（即 $u \ge \alpha_j$），则放弃 $\tilde{x}_j$ 及其后面的所有草稿 Token。大模型在这一步会重新采样，采样的修正概率分布为：
   $$q'(x) = \frac{\max\left(0, \quad q(x | x_{<j}) - p(x | x_{<j})\right)}{\sum_y \max\left(0, \quad q(y | x_{<j}) - p(y | x_{<j})\right)}$$
   用 $q'(x)$ 采样出的 Token 作为第 $j$ 个正式生成的 Token，然后这一轮验证提前结束，重新开启下一轮草稿。

#### 数学无偏性直观理解：
如果目标模型对某个 Token 的倾向性（$q$）高于草稿模型（$p$），那么这个 Token 在草稿模型中一旦产生，我们就应该 $100\%$ 接受它；如果目标模型对其认可度低（$q < p$），我们则按比例打折接受它。如果拒绝了，我们将草稿模型过高估计的概率部分切除掉（即 $q - p$ 差值），重构分布并归一化，这样采样出来的样本依然完美服从 $q$。
通过这种投机机制，在每一步大模型前向传播中，我们可能一次性接受多个 Token（平均每次可验证接受 $2 \sim 3$ 个 Token），从而带来 $2 \times \sim 3 \times$ 的端到端推理提速。

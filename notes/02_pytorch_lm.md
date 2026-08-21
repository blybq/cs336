# CS336 课程笔记 02：利用 PyTorch 搭建大模型 (Building LLMs with PyTorch)

---

## 1. 资源估算与“信封背面”计算 (Resource Estimation)

在大模型研发中，算力与显存资源是极其昂贵的瓶颈。在动工训练前，我们必须通过简单的数学公式在草稿纸上（即“信封背面”）估算出模型训练的时间和显存消耗，以确保项目可行。

### 1.1 训练时间估算

#### 训练总计算量经验公式
$$\text{训练总 FLOPs} = 6 \times N \times T$$

* **$\text{训练总 FLOPs}$ (浮点运算总次数)**：训练整个模型所需的全部乘法和加法运算次数的总和。
* **$6$ (常数系数)**：对于 Transformer 模型中的每个权重参数，在前向传播中，处理每个 Token 需要进行 $2$ 次 FLOPs（一次乘法，一次加法，合称乘加运算 MAC）。在反向传播中，我们需要计算对输入的梯度（耗费 $2$ 次 FLOPs）以及对权重的梯度（耗费 $2$ 次 FLOPs）。因此，**反向传播的计算量恰好是前向传播的 2 倍**。前向与反向相加：$2 + 4 = 6$ 次浮点运算。
* **$N$ (参数量)**：模型的总权重参数量（不包含 Embedding 层和最后一层分类头，因为它们是稀疏/离散计算，但通常可以用模型总参数量来做近似估计，例如 $7 \times 10^9$）。
* **$T$ (Token 总数)**：整个训练数据集中的 Token 数量（例如 $15 \times 10^{12}$，即 15 万亿）。

---

#### 🌟 案例实战分析
我们计划训练一个 **7B (70亿参数)** 的模型，训练数据集包含 **15T (15万亿)** 个 Token，使用 **1024 张 H100 GPU** 组成集群，估计需要多少天？

1. **计算总 FLOPs**：
   $$\text{总 FLOPs} = 6 \times (7 \times 10^9) \times (15 \times 10^{12}) = 6.3 \times 10^{23} \text{ FLOPs}$$

2. **硬件实际算力计算**：
   * 单张 H100 GPU 的理论 FP16/BF16 密集计算峰值性能为 **989 TFLOPS**（即每秒 $989 \times 10^{12}$ 次浮点运算）。
   * **模型 FLOPs 利用率 (MFU, Model FLOPs Utilization)**：
     在大规模多卡训练中，由于 GPU 之间需要进行大量的网络梯度通信（All-Reduce 等）、数据搬运以及算子切换，GPU 绝不可能达到 100% 的理论峰值。工业界优秀的工程实现能达到约 **50% 的 MFU**。
   * 单卡每日实际 FLOPs 产出：
     $$\text{每日单卡 FLOPs} = 989 \times 10^{12} \text{ FLOPs/s} \times 86400 \text{ s/day} \times 0.5 \approx 4.27 \times 10^{19} \text{ FLOPs/day}$$
     * $86400$：一天的总秒数（$24 \text{ 小时} \times 60 \text{ 分} \times 60 \text{ 秒}$）。
     * $0.5$：MFU 利用率。
   * 1024 张 H100 每日总 FLOPs 产出：
     $$\text{每日集群总 FLOPs} = 1024 \times 4.27 \times 10^{19} \approx 4.37 \times 10^{22} \text{ FLOPs/day}$$

3. **估算所需训练天数**：
   $$\text{训练天数} = \frac{6.3 \times 10^{23} \text{ FLOPs}}{4.37 \times 10^{22} \text{ FLOPs/day}} \approx 144 \text{ 天}$$

---

### 1.2 GPU 显存容量估算与模型承载极限

在单张具有 80GB HBM 显存的 H100 显卡上，如果我们使用 FP32（单精度）主权重 + AdamW 优化器进行标准训练，单卡最大能承载多大的模型？

#### 静态显存分析 (每个参数所占字节)
1. **参数权重 (Weights)**：$4$ 字节 (Bytes) —— 存储在 FP32 格式下。
2. **梯度 (Gradients)**：$4$ 字节 —— 反向传播计算所得的梯度。
3. **AdamW 优化器状态 (Optimizer States)**：每个参数对应 **$8$ 字节**。
   * AdamW 需要为每个参数维护两个状态：**一阶动量（Momentum）**和**二阶方差（Variance）**。
   * 每一个状态在 FP32 下都占用 4 字节，共计 $4 \times 2 = 8$ 字节。
4. **静态显存总计**：每个参数需要 **$16$ 字节** 显存。
   $$4 \text{ (权重)} + 4 \text{ (梯度)} + 8 \text{ (优化器状态)} = 16 \text{ 字节/参数}$$

#### 极限参数量计算
$$\text{极限参数量} = \frac{80 \times 10^9 \text{ 字节 (80GB)}}{16 \text{ 字节/参数}} = 5 \times 10^9 \text{ 参数 (5B)}$$

> [!CAUTION]
> **真实工程警告**：
> 5B 是绝对的理论极限。因为在前向传播过程中，我们还必须存储每一层计算产生的**激活值（Activations）**以供反向传播求导使用。激活值所占显存与 Batch Size、序列长度呈线性正相关。因此，在单张 80GB 显卡上，实际能够跑起来的最大模型通常只能在 **4B 左右**，否则就会遭遇 OOM (Out Of Memory) 崩溃。

---

## 2. 精度与数据类型 (Precision & Datatypes)

数据精度直接决定了模型的显存占用、计算速度（通过 Tensor Cores 硬件加速）以及训练稳定性。

![大模型浮点数精度格式对比](images/float_formats_comparison.drawio.png)

### 2.1 常用精度对比

| 数据类型 | 符号位 (Sign) | 指数位 (Exponent) | 尾数位 (Mantissa) | 总位数 (Bits) | 特点与在大模型中的应用 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FP32** | 1 | 8 | 23 | 32 | **单精度**。动态范围大且极其精确。但在大模型训练中，仅用于保存**主副本权重（Master Weights）**和**优化器状态（Optimizer States）**，以防微小的梯度更新被舍入误差抹去。 |
| **FP16** | 1 | 5 | 10 | 16 | **半精度**。虽然省空间，但由于指数位仅 5 位，其最大能表示的数仅为 65504，且无法表示小于 $6.1 \times 10^{-5}$ 的数值，**极其容易发生下溢（Underflow）和上溢（Overflow）**，导致大模型训练瞬间崩溃。目前已被 LLM 预训练淘汰。 |
| **BF16** | 1 | 8 | 7 | 16 | **Brain Float 16**。Google 专门为深度学习设计。它的指数位与 FP32 相同（同为 8 位，意味着动态范围完全一致，绝不下溢），虽然牺牲了尾数精度，但**极其稳定，是现代大模型训练的标配**。 |
| **FP8** | 1 | 4/5 | 3/2 | 8 | H100 硬件级支持。分为 E4M3（注重精度，前向使用）与 E5M2（注重范围，反向使用）。通过极度复杂的动态缩放（Dynamic Scaling）来最大化榨取 GPU 的 FLOPS。 |

### 2.2 混合精度训练 (Mixed Precision Training)
在前向和反向传播中，矩阵乘法在 BF16/FP16 中执行，使计算速度翻倍；但在梯度更新和优化器步骤中，必须将其累加回 FP32 格式的“主权重”中，以防止微小的更新值因精度不够而归零。

---

## 3. PyTorch 张量与显存机制 (Tensor & Strides)

### 3.1 张量底层原理与步长 (Strides)

#### 通俗科普：什么是 Strides（步长）？
在计算机内存中，所有的物理存储器在物理上都是**扁平的一维连续空间**。但是，我们在 PyTorch 中使用的张量（Tensor）往往是多维的（比如形状为 `[Batch, Head, Seq, Dim]` 的四维张量）。
PyTorch 是如何用一维内存表达多维结构的？答案就是 **Strides（步长）**。
每一个 Tensor 都有两个核心部分：
1. **一维数据存储区 (Storage)**：连续的一段物理内存，按扁平化顺序存储所有数值。
2. **元数据 (Metadata)**：包括张量的 `shape` (逻辑形状) 以及 `strides` (步长)。
**步长**指的是：在张量的某一个逻辑维度上向前移动 1 步时，在底层物理一维存储中需要跳过多少个元素。

#### 物理存储偏移量公式
$$\text{Offset} = \sum_{i=0}^{d-1} \text{index}_i \times \text{stride}_i$$

* **$\text{Offset}$**：目标元素在一维扁平物理内存中的索引位置（从 0 开始）。
* **$\text{index}_i$**：目标元素在第 $i$ 个逻辑维度上的索引。
* **$\text{stride}_i$**：在第 $i$ 个逻辑维度上的步长（即该维度改变 1 时，物理内存中跨越的元素个数）。
* **$d$**：张量的维度总数（Rank）。
* **$\sum$ (求和符号)**：将各个维度上的（索引 $\times$ 步长）乘积全部相加，得到最终的一维物理内存偏移量。

---

#### 🌟 步长计算实例
假设有一个形状为 `[2, 3]` 的二维张量 `T`：
```python
T = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])
```
* 底层扁平物理存储为：`[10, 20, 30, 40, 50, 60]`
* 它的 `shape` 是 `(2, 3)`
* 它的 `strides` 是 `(3, 1)`：
  * 要在第 0 维（行）移动 1 步（比如从 `T[0, 0]` 变到 `T[1, 0]`），需要跳过 3 个元素；
  * 要在第 1 维（列）移动 1 步（比如从 `T[0, 0]` 变到 `T[0, 1]`），需要跳过 1 个元素。
* 验证公式：`T[1, 2]` 的逻辑值是 `60`。
  $$\text{Offset} = (1 \times 3) + (2 \times 1) = 5$$
  在一维物理存储中，索引为 5 的元素确实是 `60`。

---

### 3.2 视图 (Views) 与连续性 (Contiguity)
* **视图操作**：`transpose()`, `t()`, `slice`，以及 Einops 中的 `rearrange`。
  这些操作**在底层完全不进行任何内存拷贝**！它们仅仅改变了元数据中的 `shape` 和 `strides`，这使得这些操作在 PyTorch 中是极其高效且“免费”的。
* **连续张量 (Contiguous Tensor)**：
  如果一个 Tensor 在物理内存中的一维排列顺序，与我们按行优先顺序（C-contiguous）逻辑读取的顺序完全一致，则该 Tensor 是连续的。
  * **转置导致的非连续**：如果我们将上述形状为 `(2, 3)` 的张量转置，新张量的形状为 `(3, 2)`，步长变为了 `(1, 3)`。此时，由于逻辑上的“下一行”在物理内存中只需要移动 1 个元素，这打破了行优先的连续存储逻辑，该张量就变成了**非连续（Non-contiguous）**的。
  * **连续性报错**：在非连续张量上直接调用 `.view()` 会抛出报错。因为 `.view()` 要求逻辑维度改变必须对应扁平内存的物理切割。为了解决这一问题，必须先调用 `.contiguous()`。该操作会在内存中分配一块全新的连续空间，并将数据拷贝过去，使得步长恢复正常。

---

### 3.3 Einsum (爱因斯坦求和约定) 与 Einops
大模型代码中充斥着复杂的张量维度变换（如注意力头合并与拆分）。为了避免使用极易出错且可读性极差的 `permute`、`reshape` 和硬编码索引，现代 LLM 代码推荐使用 **Einsum** 和 **Einops**。

#### Einsum 规则
1. 在输入维度和输出维度中，用英文字母给每个维度命名。
2. 出现在输入中但**未出现在输出中**的字母维度，会在矩阵相乘后被自动相加求和（降维 / Reduce）。
3. 使用 `...`（省略号）可以表示任意多个前置维度（如 Batch Size，使得操作可以自动广播）。

#### 示例：计算自注意力中 $Q \cdot K^T$
```python
# Q 的形状为 [Batch, Head, Seq_Q, Dim] -> 'bhqd'
# K 的形状为 [Batch, Head, Seq_K, Dim] -> 'bhkd'
# 我们希望在 'd' 维度进行点积，输出 [Batch, Head, Seq_Q, Seq_K] -> 'bhqk'
attn_weights = torch.einsum('bhqd, bhkd -> bhqk', Q, K)
```

---

## 4. 计算开销的矩阵分析 (FLOPs of Matrix Multiplications)

在大模型中，线性投影层（Linear Layer）本质上就是矩阵乘法。我们需要能够精确量化每次矩阵乘法的 FLOPs 数量。

### 4.1 矩阵乘法计算量公式
若计算矩阵 $A \in \mathbb{R}^{M \times N}$ 与矩阵 $B \in \mathbb{R}^{N \times K}$ 的乘积：
$$\text{FLOPs} = 2 \times M \times N \times K$$

* **$M, N, K$**：参与相乘的两个矩阵的逻辑维度尺寸。
* **$2$ (常数系数)**：输出矩阵的形状为 $M \times K$，即共有 $M \times K$ 个待求元素。对于其中的每一个元素，都是由 $A$ 的某一行（长度为 $N$）与 $B$ 的某一列（长度为 $N$）进行点积所得。这个点积包含 $N$ 次乘法运算与 $N - 1$ 次加法运算（近似为 $N$ 次加法），合称 $2N$ 次浮点运算。因此，总计算量为 $2 \times M \times N \times K$。

---

### 4.2 线性层前向与反向计算量分解

设输入张量 $X \in \mathbb{R}^{B \times D}$（其中 $B$ 为 Batch 中的 Token 总数，$D$ 为输入通道维度），线性层权重为 $W \in \mathbb{R}^{D \times K}$（输出维度为 $K$），参数量 $N_{\text{param}} = D \times K$。

#### 1. 前向传播 (Forward)：$Y = XW$
* 矩阵乘法维度：$(B \times D) \times (D \times K)$
* 计算量：
  $$\text{Forward FLOPs} = 2 \times B \times D \times K = 2 \times B \times N_{\text{param}}$$
  **含义**：前向传播计算量等于 $2 \times \text{Tokens数} \times \text{参数量}$。

#### 2. 反向传播 (Backward)
反向传播时，线性层需要计算两个梯度：
* **对权重 $W$ 的梯度**：$\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}$
  * 矩阵乘法维度：$(D \times B) \times (B \times K)$
  * 计算量：
    $$\text{Grad W FLOPs} = 2 \times D \times B \times K = 2 \times B \times N_{\text{param}}$$
* **对输入 $X$ 的梯度**：$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$
  * 矩阵乘法维度：$(B \times K) \times (K \times D)$
  * 计算量：
    $$\text{Grad X FLOPs} = 2 \times B \times K \times D = 2 \times B \times N_{\text{param}}$$
* **反向传播总计算量**：
  $$\text{Backward FLOPs} = \text{Grad W FLOPs} + \text{Grad X FLOPs} = 4 \times B \times N_{\text{param}}$$

#### 3. 前反向总计
$$\text{Total FLOPs} = \text{Forward FLOPs} + \text{Backward FLOPs} = 6 \times B \times N_{\text{param}}$$
这正是第一章“训练时间估算经验公式”中系数 **$6$** 的底层矩阵数学来源。

---

## 5. 通俗科普：因果掩码 (Causal Masking)

### 5.1 为什么需要因果掩码？
大语言模型通常是**自回归（Autoregressive）**的。在生成文本时，模型是根据已经生成的上文，来预测下一个 Token（例如，已知“我喜欢吃”，预测下一个词是“苹果”）。
在训练阶段，为了提高效率，我们采用**Teacher Forcing**机制：将整句话（如“我喜欢吃苹果”）一次性输入给模型。
但是，Transformer 的注意力机制允许当前 Token 与全句所有的 Token 发生交互。如果让“喜欢”直接注意到它后面的“苹果”，模型就会发生“作弊（信息泄漏）”——它不需要学习上文的逻辑，直接抄后面的答案就行了。
为了防止这种未来的信息泄漏，我们必须在计算注意力时使用**因果掩码（Causal Mask）**。

---

### 5.2 因果掩码的数学机制

设输入序列的 Token 长度为 $L$。在自注意力机制中，Query 向量与 Key 向量相乘，得到一个大小为 $L \times L$ 的原始注意力得分矩阵 $A$：
$$A_{i, j} = \frac{Q_i K_j^T}{\sqrt{d_k}}$$

为了实现因果关系，我们定义一个掩码矩阵 $M \in \mathbb{R}^{L \times L}$（一个上三角矩阵）：
$$M_{i, j} = \begin{cases} 
0 & i \ge j \quad (\text{当前位置或历史位置，允许关注}) \\
-\infty & i < j \quad (\text{未来位置，强行屏蔽})
\end{cases}$$

将掩码矩阵加到原始注意力得分上：
$$\text{Masked Attention Logits}_{i, j} = A_{i, j} + M_{i, j}$$

接下来，我们对得分应用 Softmax 激活函数来归一化注意力权重：
$$\text{Softmax}(\text{Masked Attention Logits})_{i, j} = \frac{e^{A_{i, j} + M_{i, j}}}{\sum_{k=1}^L e^{A_{i, k} + M_{i, k}}}$$

当 $i < j$（即目标是未来的 Token）时，由于 $M_{i, j} = -\infty$：
$$e^{A_{i, j} + (-\infty)} = e^{-\infty} = 0$$

因此，未来的 Token 获得的注意力权重被数学上**强行归零**：
$$\text{Softmax}(\text{Masked Attention Logits})_{i, j} = 0 \quad (\text{当 } j > i)$$
这样，第 $i$ 个 Token 就只能感知到第 $0 \sim i$ 个 Token 的信息，彻底断绝了对未来信息的窥探。

---

### 5.3 PyTorch 实现方式
在 PyTorch 中，我们可以利用 `torch.triu`（提取矩阵的上三角部分）来快速创建一个布尔掩码，然后使用 `.masked_fill` 将上三角部分填充为极小的负数（如 `-1e9` 或 `-inf`）：

```python
L = 5  # 序列长度
# 创建一个上三角全 1 矩阵（不包含主对角线，diagonal=1）
# 形状为 [L, L] 的矩阵，上三角部分为 True，其余为 False
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()

# 模拟自注意力得分矩阵
attn_scores = torch.randn(L, L)

# 将未来位置填充为极小值（-inf）
masked_scores = attn_scores.masked_fill(mask, float('-inf'))
```
填充后的 `masked_scores` 经过 `softmax` 之后，上三角的所有值都会变成标准的 0.0。

---

## 6. 模型初始化与训练机制

### 6.1 参数初始化与方差控制
* **方差膨胀问题**：若使用标准正态分布 $\mathcal{N}(0, 1)$ 初始化隐藏层权重 $W$，由于输出是输入与权重的点积，输出特征的方差会随着输入通道数 $D_{\text{in}}$ 的增加而呈线性膨胀。在深层网络中，这会导致前向传播值和反向梯度迅速爆炸。
* **Xavier/He 初始化**：将权重初始化为方差与 $1/D_{\text{in}}$ 成正比的分布，从而在数学上将隐藏状态的方差稳定在 1。
* **截断正态分布 (Truncated Normal)**：为了防止随机初始化时产生极端偏离均值的权重值，通常在 $[-2, 2]$ 标准差范围内截断。

### 6.2 优化器状态的显存占用
对于包含 $N$ 个参数的模型，在训练时的静态显存占用：
* **SGD**：$4N$ 字节（仅存储权重，梯度在反向传播计算完后可直接覆盖）。
* **带有动量的 SGD**：$8N$ 字节（权重 4 字节 + 动量缓存 4 字节）。
* **AdamW**：$16N$ 字节（权重 4 字节 + 梯度 4 字节 + 一阶动量 $m$ 4 字节 + 二阶方差 $v$ 4 字节）。**这导致在大模型训练中，优化器本身吃掉了绝大部分显存。**

### 6.3 激活值重计算 (Activation Checkpointing)

#### 通俗科普：为什么需要重计算？
在前向传播中，每一层计算所得的中间结果（即“激活值”）在反向传播中都需要被用来计算偏导数。
这意味着在进行反向传播前，我们必须在 GPU 显存中保存**所有层**在前向传播中产生的所有激活值。随着网络层数越来越深，激活值占用的显存会呈线性增长，成为训练时的显存第一大杀手（往往远超模型参数自身）。
**重计算（Activation Checkpointing）**提供了一个巧妙的思路：**“用计算时间换显存空间”**。
我们不在显存中保存所有中间层的激活值，而是仅在一些关键节点（比如每 4 个 Transformer Block 的边界）保存激活值（这些保存点称为 Checkpoints）。
当反向传播计算到中间无激活值的层时，GPU 会从最近的前一个 Checkpoint 出发，**重新跑一次局部的快速前向传播**，当场把缺失的激活值计算出来，用完后立即释放。
* **代价**：增加了大约 **33% (1/3)** 的额外前向计算耗时。
* **收益**：激活值显存占用大幅度降低，使我们能够在相同的 GPU 硬件上训练参数量大数倍的模型。

### 6.4 训练容灾与磁盘检查点 (Disk Checkpointing)
* **背景科普**：在前面的 `6.3` 中，我们学到了 *激活值检查点 (Activation Checkpointing)*，这是一种“用计算换显存”的显存优化技术。而在工业界，**磁盘检查点 (Disk Checkpointing / Model Resumption)** 则是完全不同的概念，它是用于**训练容灾和断点续传**的存储备份技术。
* **设计直觉**：大模型训练通常运行在成百上千个 GPU 节点上，连续训练数周甚至数月。在这个过程中，由于硬件老化、掉卡、网络闪断或节点崩溃，训练任务随时可能异常终止。为了避免从零开始训练，必须定期（如每 500 步或 1000 步）将训练状态持久化保存到磁盘上。
* **需要保存的完整训练状态**：
  1. **模型权重 (Model Weights)**：当前的神经网络模型参数（FP32 或混合精度下的主权重）。
  2. **优化器状态 (Optimizer States)**：以 AdamW 为例，需要保存一阶动量 $m_t$ 和二阶动量 $v_t$。正如我们在 `6.2` 中算过的，优化器状态占用的存储空间是模型本身的两倍！如果不保存优化器状态，强行从模型权重恢复，训练动力学会彻底打乱。
  3. **当前迭代步数与数据加载器状态 (Iteration Index & DataLoader States)**：数据加载器的游标（Offset）必须保存，以保证恢复训练时，模型能从上次中断的那个 Token 开始继续喂数据，不重复读取已经见过的语料，也不漏过未见过的语料。

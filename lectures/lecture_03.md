# CS336 课程讲义

## 第 1 页 (Page 1)

# 第 3 讲：关于语言模型架构与超参数你所不想知道的一切

**CS336**
**主讲人：Tatsu H**

---

## 第 2 页 (Page 2)

### 大纲与目标

* **现代 Transformer 快速回顾**（即你所实现的部分）
* **大多数大型语言模型有哪些共同点？**
* **架构与训练过程中常见的变体有哪些？**

> **今日主题**：最好的学习方式是亲自动手实践，第二好的学习方式是尝试从他人的经验中学习。

---

## 第 3 页 (Page 3)

### 起点：“原始” Transformer

**回顾：标准 Transformer 中的选择**

* **位置嵌入（Position embedding）**：正弦与余弦
  $$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
  $$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
* **前馈网络（FFN）**：ReLU
  $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
* **归一化类型（Norm type）**：后归一化（post-norm），LayerNorm

---

## 第 4 页 (Page 4)

### 你所实现的——简单、现代的变体

**主要差异：**
* **LayerNorm 位于模块的前面**（前归一化 pre-norm）
* **旋转位置嵌入（RoPE）**
* **前馈层使用 SwiGLU**，而非 ReLU
* **线性层（及 LayerNorm）没有偏置**（常数）项

> 我们为什么要选择这些？你应该选择什么？

---

## 第 5 页 (Page 5)

### 我们应该如何看待架构？

* 众多的架构。光是在 2024-2025 年之间……
* 发布了超过 **19 个新的稠密（dense）模型**，其中许多都带有微小的架构调整。

---

## 第 6 页 (Page 6)

### 我们应该如何看待架构？ (续)

* 今年不可能发布那么多大语言模型（LLM），对吧？
  *(注：技术报告包含 LLaMA 4, gpt-oss, DeepSeek-V3.2, MiniMax M2, Kimi K2, Intern-S1, GLM-4.7, 步神 Step-3 等)*

---

## 第 7 页 (Page 7)

### 让我们来看看数据（关于稠密架构）

从现有的许多其他模型（和论文）中学习。我们将详细讨论许多主要的架构和超参数变体：
* 所有这些模型有什么共同点？
* 哪些部分有所变化？
* 我们可以从中学习到什么？

---

## 第 8 页 (Page 8)

### 我们将要涵盖哪些内容？

* **常见的架构变体**
  * 激活函数、FFN
  * 注意力变体
  * 位置嵌入
* **重要或不重要的超参数**
  * 什么是 $ff\_dim$？多头维度之和总是等于 $model\_dim$ 吗？
  * 词表大小（vocab size）是多少？
* **稳定性技巧**

---

## 第 9 页 (Page 9)

### 架构变体……

让我们思考核心架构部分。

**高层视角：**
* **“类 LLaMA”架构占据主导地位**
* **历年趋势**（QK-norm、混合注意力）

---

## 第 10 页 (Page 10)

### 前归一化（Pre-norm）与后归一化（Post-norm）

**大家一致同意的一点（在 2024 年）**

* 设置 LayerNorm 使其不影响主要的残差信号路径（如左图所示）。
* 几乎所有现代语言模型都使用**前归一化（pre-norm）**，而 BERT 使用的是后归一化（post-norm）。
* 一个有些滑稽的例外是 **OPT350M**，我不知道为什么它是后归一化。

*(插图引自 Xiong et al. 2020)*

---

## 第 11 页 (Page 11)

### 前归一化 vs 后归一化：数据

*(插图引自 Xiong et al. 2020 以及 Salazar & Nguyen 2019，展示了 English-Vietnamese BLEU 分数及 Validation Loss 的对比)*

---

## 第 12 页 (Page 12)

### 前归一化 vs 后归一化：物理解释？

* **梯度衰减** [Xiong 2020] 与 **梯度尖峰** [Salazar & Nguyen]
* 最初宣称的优势：**消除学习率预热期（warmup）**。
* 如今的实际作用：**大网络训练的稳定性和使用更大的学习率（LR）**。

---

## 第 13 页 (Page 13)

### 新事物——“双”归一化（Double Norm）或非残差后归一化

* 如果在残差流中放入 LayerNorm 效果不好……为什么不在残差流之外进行后归一化呢？
* **最近的模型**：Grok, Gemma 2。Olmo 2 仅执行非残差后归一化。

---
### 💡 核心机制沉淀：Pre-norm 与 Post-norm 的拓扑结构、梯度传播与演进

#### 1. 概念澄清：归一化层在残差结构中的具体位置
在传统单层前馈网络（如经典 CNN/MLP）中，归一化常置于激活函数前（`Linear -> BatchNorm -> ReLU`）。
但在 **Transformer 残差架构** 中，**前归一化 (Pre-LN)** 与 **后归一化 (Post-LN)** 的核心区别在于 **归一化层与残差加法分支的相对拓扑位置**：

| 架构类型 | 数学公式 | 归一化层位置 | 残差主干道（Residual Stream）状态 |
| :--- | :--- | :--- | :--- |
| **后归一化 (Post-LN)**<br>*(原始 Transformer, BERT)* | $$x_{l+1} = \text{Norm}(x_l + F(x_l))$$ | **残差相加之后** | **残差流被 Norm 截断污染**，主干信号必须逐层经过 Norm |
| **前归一化 (Pre-LN)**<br>*(GPT-2/3/4, LLaMA, 现代主流)* | $$x_{l+1} = x_l + F(\text{Norm}(x_l))$$ | **子模块计算之前（仅在分支内）** | **残差流保持纯净（Clean Residual）**，主干道是直通恒等连接 |

> 📌 **核心辨析**：Pre-LN 不是在子层内部的激活函数前加 Norm，而是在进入整个 Attention 或 FFN 子层前对残差输入做一次 Norm，计算完毕后直接加回残差流，**残差主干道上无任何算子阻挡**。

#### 2. 为什么现代大模型普遍选择 Pre-LN？
核心原因在于**深层网络的训练稳定性与梯度反向传播机制**：
- **无损梯度高速公路（Clean Gradient Flow）**：
  - Pre-LN 输出展开为 $x_L = x_0 + \sum_{l=0}^{L-1} F(\text{Norm}(x_l))$，损失对浅层输入的导数包含恒等矩阵项 $\mathbf{I}$：
    $$\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \sum_{l=0}^{L-1} \frac{\partial F(\text{Norm}(x_l))}{\partial x_0}$$
    深层损失信号可无衰减直通回传至底层，彻底杜绝梯度弥散。
  - Post-LN 反向传播每一层必须连乘 $\frac{\partial \text{Norm}}{\partial x}$，浅层梯度呈指数级衰减，且初始化时顶层极易发生严重的梯度尖峰（Gradient Spikes）。
- **对超参数与深度的鲁棒性**：
  - Post-LN 训练非常脆弱，极度依赖漫长精细的学习率预热（Warmup），难以拓展至深层（如上百层）；
  - Pre-LN 极大地降低了对 Warmup 的依赖，允许使用更大更激进的学习率，支撑了千亿甚至万亿参数超深大模型的稳定收敛。

#### 3. 前沿演进：Pre-LN 的局限与双归一化 (Double Norm)
- **Pre-LN 的潜在问题**：随着网络加深，残差流 $x_l$ 的方差随层数累积增长，导致深层子模块分支 $F(\text{Norm}(x_l))$ 对残差流的相对更新幅度被逐渐稀释（Representation Collapse）。
- **双归一化 / 非残差后归一化 (Double Norm)**：
  $$x_{l+1} = x_l + \text{Norm}_{\text{out}}(F(\text{Norm}_{\text{in}}(x_l)))$$
  - 在子层输入前做 Pre-Norm，在子层输出后、加回残差流**之前**再做一次 Norm；
  - **残差主干道依然保持直通**，既享受了 Pre-LN 的稳定无损梯度，又约束了每一层分支更新的幅度（被 Gemma 2、Grok、OLMo 2 等前沿模型广泛采纳）。
---

## 第 14 页 (Page 14)

### LayerNorm 对比 RMSNorm

* **原始 Transformer：LayerNorm** —— 在 $d_{model}$ 上对均值和方差进行归一化：
  $$y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \odot \gamma + \beta$$
  *代表模型：GPT3/2/1, OPT, GPT-J, BLOOM*
* **许多现代语言模型：RMSNorm** —— 不减去均值，也不加偏置项：
  $$y = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
  *代表模型：LLaMA 家族, PaLM, Chinchilla, T5*

---

## 第 15 页 (Page 15)

### 为什么使用 RMSNorm？

**现代的合理解释——它更快（且效果同样好）：**
* **操作更少**（无均值计算）
* **参数更少**（无需存储偏置项）

**这个解释合理吗？**
* 矩阵乘法占了 FLOPs（和内存）的绝大部分（约 99.8%）。
* 归一化操作仅占 0.17% FLOPs。

*(数据引自 Ivanov et al. 2023)*

---

## 第 16 页 (Page 16)

### 为什么使用 RMSNorm (2)

> **重要教训**：FLOPs 不等于实际运行时间！（我们稍后将对此进行更详细的讨论）

* 由于**数据传输（Memory Bandwidth / Data Movement）**的重要性，RMSNorm 依然非常关键。虽然 FLOPs 占比极低，但它属于元素级操作，受限于内存带宽，RMSNorm 减少了内存读写。

---

## 第 17 页 (Page 17)

### RMSNorm - 验证

在学术论文中已经看到了 RMSNorm 带来的实际运行时间（以及令人惊讶的性能）提升。

*(数据引自 Narang et al. 2020)*

---
### 💡 核心机制沉淀：RMSNorm 的性能本质——“FLOPs 伪命题”与“访存带宽减负”

#### 1. 核心论点澄清：为什么说“FLOPs 不等于实际耗时”？
在 GPU 硬件物理执行中，算子有着根本性的瓶颈划分（如 Lecture 02 中的 Roofline 拓扑）：
- **算力受限 (Compute-bound，如 GEMM 矩阵乘法)**：占总 FLOPs 的 **99.8%**，Tensor Core 处于满载状态，耗时由总计算次数决定。
- **访存受限 (Memory-bound / Element-wise，如 LayerNorm/RMSNorm)**：仅占总 FLOPs 的 **0.17%**。虽然算术计算极其简单，但计算前必须把张量从全局显存 (HBM) 搬进芯片寄存器，算完再写回显存。**耗时 90% 以上都在等待显存总线搬运数据，GPU 算力核心大部分时间处于空闲发呆状态**。

#### 2. RMSNorm 为什么能带来 7%~15% 的真实端到端加速？
RMSNorm 的加速本质**绝非省了几个加减法 FLOPs，而是减少了对全局显存的数据搬运趟数 (Memory Passes)**：

| 归一化类型 | 计算公式 | 显存扫描趟数 (Memory Passes) | 访存与参数开销 |
| :--- | :--- | :--- | :--- |
| **标准 LayerNorm** | $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$ | **2 趟扫描 (Two-pass)**：<br>1. 先读一遍 $x$ 算均值 $\mu = \mathbb{E}[x]$ 并暂存；<br>2. 再从显存读一遍 $x$ 结合均值算方差 $\sigma^2$ 并完成归一化。 | 两次扫描大张量，显存带宽开销大；需额外加载偏置参数 $\beta$。 |
| **简化版 RMSNorm** | $y = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$ | **1 趟扫描 (One-pass)**：<br>彻底舍弃均值 $\mu$ 项，在流式读取 $x$ 的同时累加平方和并直接完成缩放归一化。 | **减少了一整趟对整个隐藏层张量的显存读写**；无偏置参数 $\beta$。 |

> 📌 **本质结论**：
> RMSNorm 丢弃均值中心化（Zero-centering），使底层 CUDA Kernel 能将两趟显存访问熔断为**单趟流式访存 (One-pass)**，从根本上缓解了 Memory-bound 瓶颈，因而在大模型训练与推理中取得了扎实的物理加速。
---

## 第 18 页 (Page 18)

### 更一般地：舍弃偏置项

大多数现代 Transformer 没有偏置项。

* **原始 Transformer**：
  $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
* **大多数现代实现（若非门控形式）**：
  $$\text{FFN}(x) = \sigma(xW_1)W_2$$

**原因**：内存考量（与 RMSNorm 类似）以及优化过程的稳定性。

---

## 第 19 页 (Page 19)

### LayerNorm：总结

* **基本上每个人都执行非残差归一化（通常是 pre-norm）**
  * **直觉**：保留残差连接的优良特性。
  * **观察**：更优的梯度传播，更少的尖峰。
  * 有些人在残差流之外添加了第二个归一化。
* **大多数人使用 RMSNorm**
  * 在实践中，效果与 LayerNorm 一样好。
  * 但是，需要移动的参数和张量数据更少，从而节省了实际运行时间。
  * 人们更普遍地舍弃偏置项，因为计算/参数的权衡并不划算。

---

## 第 20 页 (Page 20)

### 激活函数

**激活函数的庞大家族：**
ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU

> 这些是什么？人们在用什么？这重要吗？

---

## 第 21 页 (Page 21)

### 几种常见的激活函数

* **ReLU**
  $$\text{FF}(x) = \max(0, xW_1)W_2$$
  *代表模型：原始 Transformer, T5, Gopher, Chinchilla, OPT*
* **GeLU**
  $$\text{FF}(x) = \text{GELU}(xW_1)W_2$$
  $$\text{GELU}(x) := x\Phi(x)$$
  *代表模型：GPT-1/2/3, GPT-J, GPT-Neox, BLOOM*
* **SwiGLU / GeGLU** (见下一页)
  *代表模型：LLaMA, PaLM, T5 v1.1, 2023 年后的大多数模型*

---

## 第 22 页 (Page 22)

### 门控激活函数 (*GLU)

GLU 修改了前馈层的“第一部分”。
$$\text{FF}(x) = \max(0, xW_1)W_2$$

为了代替经典的 线性 + ReLU，我们用一个逐元素相乘的线性项来增强它：
$$\max(0, xW_1) \to \max(0, xW_1) \otimes (xV)$$

这得到了门控变体（ReGLU）——请注意，我们引入了一个额外的参数矩阵（V）：
$$\text{FF}_{\text{ReGLU}}(x) = (\max(0, xW_1) \otimes xV) W_2$$

---

## 第 23 页 (Page 23)

### 标准前馈层的门控变体

* **GeGLU**
  $$\text{FFN}_{\text{GEGLU}}(x, W, V, W_2) = (\text{GELU}(xW) \otimes xV)W_2$$
  *著名模型：T5 v1.1, mT5, LaMDA, Phi3, Gemma 2, Gemma 3, Gemma 4*
* **SwiGLU** (Swish 为 $x \cdot \text{sigmoid}(x)$)
  $$\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV)W_2$$
  *著名模型：LLaMA 1/2/3, PaLM, Mistral, OlMo, 以及 2023 年后的大多数模型*

> **注意**：门控模型通常将前馈维度 $d_{ff}$ 缩小约 2/3，以保持参数量相当。

---

## 第 24 页 (Page 24)

### 门控线性单元有效吗？

是的，表现相当一致地更优。

*(参考 Shazeer 2020)*

---

## 第 25 页 (Page 25)

### 门控线性单元有效吗 (2)？

是的，其他工作也证实了 Shazeer 2020 的结论。

*(参考 Narang et al. 2020)*

---

## 第 26 页 (Page 26)

### 门控与激活函数总结

* 不同模型之间有许多变体（ReLU, GeLU, *GLU）。
* 对于一个能工作的模型，*GLU 并不是必需的（例如 GPT-3 依然工作良好），但在目前极少见到其他选择。
  * 一些特立独行的模型：Nemotron 340B (使用平方 ReLU)。
* 现有证据一致表明，Swi/GeGLU 带来了稳定且明显的性能增益。

---
### 💡 核心机制沉淀：前馈网络 (FFN) 结构辨析与门控激活函数 (*GLU) 底层原理

#### 1. 符号与运算澄清：$\otimes$ 与 $\odot$ 的真实含义
- 课件中的 $\otimes$ 与经典文献中的 $\odot$ 在此处均指 **逐元素相乘（Hadamard Product / Element-wise Multiplication，即 Python/PyTorch 中的 `*` 运算符）**，**绝非**升维的高阶张量外积（Kronecker Product）。
- **计算定义**：对于同形状向量 $a, b \in \mathbb{R}^{d_{ff}}$，其结果向量的每个分量为 $(a \odot b)_i = a_i \cdot b_i$。

#### 2. FFN 结构辨析：为什么公式中包含 $W_1$ 与 $W_2$？
- **激活函数本身**：如 $\text{ReLU}(z)=\max(0, z)$、$\text{GELU}(z)$，是纯数学、**无任何可学习参数**的逐元素非线性算子。
- **前馈网络子模块 ($\text{FFN}(x)$)**：指的是由两个线性投影层夹着一个无参激活层构成的两层 MLP Block：
  $$\text{FFN}(x) = \sigma(x W_1) W_2$$
  - $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$：升维投影矩阵（Up-projection）；
  - $\sigma(\cdot)$：无参中间激活函数；
  - $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$：降维投影矩阵（Down-projection）。

#### 3. 为什么现代大模型普遍采用门控激活单元 (*GLU)？三大核心直觉
在标准 FFN 基础上，GLU 引入了第二个线性变换分支 $V \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$，构成双分支逐元素点乘结构：
$$\text{FFN}_{\text{GLU}}(x) = \Big(\underbrace{\sigma(x W_1)}_{\text{门控分支 (Gate)}} \odot \underbrace{(x V)}_{\text{特征候选分支 (Value)}}\Big) W_2$$

1. **物理直觉：动态特征“软开关”（Soft Gating）**
   - 分支 $x V$ 提取当前 Token 的候选语义特征；
   - 分支 $\sigma(x W_1)$ 输出每个通道的放行系数（0 表示抑制关闭，大数值表示增强放行）；
   - 逐元素相乘赋予了模型根据上下文动态路由、按需过滤特征通道的能力。
2. **数学直觉：双线性特征交叉（Bilinear Interaction）**
   - 传统 MLP 仅有一阶线性输入；
   - 门控乘积 $\sigma(x W_1) \odot (x V)$ 引入了输入 $x$ 的**二阶多项式交互项**，在不增加网络深度的情况下成倍拓宽了高维特征空间的几何划分与表达能力。
3. **优化直觉：反向传播的“梯度直通通道”（Gradient Highway）**
   - 根据乘积求导法则 $\frac{\partial (a \odot b)}{\partial x} = \frac{\partial a}{\partial x} \odot b + a \odot \frac{\partial b}{\partial x}$；
   - 线性分支 $b = x V$ 的导数为常数矩阵 $V$，为梯度提供了一条**绕过非线性饱和区（如 ReLU 死区）的直通线性梯度通路**，极大改善了深层网络的优化稳定性。

#### 4. 参数量平衡技巧（2/3 维度缩放法则）
- 传统 FFN 包含 2 个矩阵（$W_1, W_2$），总参数量为 $2 \times d_{\text{model}} \times d_{ff} = 8 d_{\text{model}}^2$（当标准 $d_{ff} = 4 d_{\text{model}}$ 时）；
- GLU 变体包含 3 个矩阵（$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$），为保持总参数量与算力开销相当，工业界统一将前馈维度缩小至原来的 **2/3**：
  $$d_{ff} \approx \frac{8}{3} d_{\text{model}} \approx 2.66 d_{\text{model}}$$
---

## 第 27 页 (Page 27)

### 串行与并行层

* 标准的 Transformer 块是**串行**的——它们先计算注意力，然后再计算 MLP。
* 我们能把 Transformer 块并行化吗？

---

## 第 28 页 (Page 28)

### 并行层

少数模型（GPT-J, PaLM, GPT-NeoX）使用并行层。最初在 GPT-J 中提出。

* **串行形式**：
  $$y = x + \text{MLP}(\text{LayerNorm}(x + \text{Attention}(\text{LayerNorm}(x))))$$
* **并行形式**：
  $$y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))$$

在大规模训练中，由于 MLP 和注意力的输入矩阵乘法可以融合，并行形式使训练速度提高了大约 15%。消融实验显示在 8B 规模下有轻微的质量下降，但在 62B 规模下没有质量下降，因此我们外推在 540B 规模下并行层的影响在质量上应当是中性的。

* **最近的模型**：Cohere Command A, Falcon 2 11B, Command R+

---
### 💡 核心机制沉淀：Transformer 块与 Attention 层的张量维度流转对齐

#### 1. 概念澄清：中间注意力权重 vs 最终输出张量
- **中间注意力权重矩阵 $A$**：由 $\text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$ 计算得出，形状确实为 **$k \times k$**（$k$ 为序列长度，表示 Token 间的注意力分布）。
- **加权聚合与投影输出 $\text{Attention}(X)$**：$A$ 必须进一步与 Value 矩阵相乘并经过输出投影层：
  $$\text{Attention}(X) = \Big(\underbrace{A}_{k \times k} \cdot \underbrace{V}_{k \times d}\Big) \cdot \underbrace{W^O}_{d \times d} \quad \Longrightarrow \quad \mathbf{k \times d}$$
  经过矩阵乘法 $(k \times k) \times (k \times d)$，每个 Token 加权汇聚整个序列的特征向量，输出维度严格还原为 **$k \times d$**。

#### 2. 并行/串行残差块的维度一致性
在公式 $y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))$ 中：
- 输入残差流 $x \in \mathbb{R}^{k \times d}$
- $\text{Attention}(\text{LN}(x)) \in \mathbb{R}^{k \times d}$
- $\text{MLP}(\text{LN}(x)) \in \mathbb{R}^{k \times d}$（经 $k \times d \to k \times d_{ff} \to k \times d$ 升降维）
- 三者形状完全一致（均为 $k \times d$），可直接进行逐元素残差相加。
---

## 第 29 页 (Page 29)

### 架构总结

* **前归一化 vs 后归一化**：几乎所有人都在使用非残差归一化（除 OPT350M 外），这很可能有充分的理由。
* **LayerNorm vs RMSNorm**：RMSNorm 具有明显的计算优势，有时甚至性能更好。
* **门控（Gating）**：门控（GLU）现在已成为行业共识。
* **串行 vs 并行层**：目前大多数模型仍然使用串行层。

---

## 第 30 页 (Page 30)

### 位置嵌入的众多变体

* **正弦位置嵌入（Sine embeddings）**：添加正弦和余弦，能够实现局部化定位。
  $$\text{Embed}(x, i) = v_x + PE_{pos}$$
  *代表模型：原始 Transformer*
* **绝对位置嵌入（Absolute embeddings）**：在嵌入中直接加入位置向量。
  $$\text{Embed}(x, i) = v_x + u_i$$
  *代表模型：GPT-1/2/3, OPT*
* **相对位置嵌入（Relative embeddings）**：在注意力计算中加入相对距离向量。
  $$e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}}$$
  *代表模型：T5, Gopher, Chinchilla*
* **RoPE 旋转位置嵌入** (见下一页)
  *代表模型：GPT-J, PaLM, LLaMA, 以及大多数 2024 年以后的模型*

> 💡 **演进思考（为什么早期 GPT 采用绝对可学习位置嵌入？）**：
> 原始正弦编码直接与词向量相加后，在注意力点积中会产生混杂的“语义内容-绝对位置”交叉项，未能实现纯粹的相对不变性；且在早期固定短上下文（512~2048）场景下，可学习嵌入不仅参数开销微乎其微（<0.02%），而且具备纯数据驱动的高拟合自由度。直到现代超长上下文需求爆发、绝对嵌入暴露出“无法外推”的硬伤后，才全面演进到了兼具纯净相对性与外推能力的 **RoPE 旋转位置编码**。

---

## 第 31 页 (Page 31)

### RoPE：旋转位置嵌入

**高层思考过程：相对位置嵌入应该是某种 $f(x, i)$，使得它们的内积只依赖于相对距离：**
$$\langle f(x, i), f(y, j) \rangle = g(x, y, i - j)$$

也就是说，注意力函数只能依赖于相对位置 $(i-j)$。现有的嵌入方式是如何未能实现这一目标的？
* **正弦**：包含各种非相对的交叉项。
  $$\langle \text{Embed}(x, i), \text{Embed}(y, i) \rangle = \langle v_x, v_y \rangle + \langle PE_i, v_y \rangle ...$$
* **绝对**：显然不是相对的。
* **相对嵌入**：$e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}}$ 不是内积形式。

---

## 第 32 页 (Page 32)

### RoPE：旋转位置嵌入 (设计理念)

*我们如何解决这个问题？*
* 我们希望我们的嵌入对绝对位置具有不变性。
* 我们知道内积在任意旋转下都是不变量。

*(图形解释：)*
* **位置无关的嵌入**（“we” 与 “know” 向量夹角固定）
* **嵌入 “we know that”**：将 “we” 旋转 0 个单位，将 “know” 旋转 1 个单位。
* **嵌入 “of course we know”**：将 “we” 旋转 2 个单位，将 “know” 旋转 3 个单位。

---

## 第 33 页 (Page 33)

### RoPE：旋转位置嵌入 (实现形式)

*旋转有许多种，我们选择哪一种？*
* 只需两两配对坐标，并在二维空间中进行旋转（数学动机：复数乘法即旋转）。
* *Gemma 4 替代方案：仅对前 2 个维度进行旋转。*

*(插图引自 Su et al. 2021)*

---

## 第 34 页 (Page 34)

### RoPE 的实际数学原理

乘以正弦与余弦矩阵：
$$f_{\{q,k\}}(x_m, m) = R^d_{\Theta, m} W_{\{q,k\}} x_m$$

与正弦位置嵌入的差异：**它是乘性的，非加性，且没有交叉项。**

---

## 第 35 页 (Page 35)

### RoPE 的代码实现

```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

# Flash attention 要求输入形状为：
# batch_size x seq_length x head_dim x hidden_dim
# 因此我们只需要保持原始形状：
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

# 获取 RoPE 的 cos/sin 矩阵
cos, sin = self.rotary_emb(value_states, position_ids)
# 乘以 query/key 输入
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

> **注意**：需要在每次注意力操作时都应用位置嵌入，以确保位置不变性。

---

## 第 36 页 (Page 36)

### 超参数

你可能在 NLP 课程（如 CS224n）中遇到过的 Transformer 超参数问题：
* 与隐藏层大小相比，前馈层（FFN）应该大多少？
* 应该设置多少个头，并且头数是否总是能整除隐藏层大小？
* 词表大小（vocab size）应该设置为什么？

以及其他模型扩展问题：
* 人们会对这些巨大的语言模型进行正则化吗？
* 人们是如何扩展这些模型的——是很深还是极宽？

---

## 第 37 页 (Page 37)

### 令人惊讶（？）的共识超参数 1

**前馈层与模型维度的比例。**
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

有两个相关的维度——前馈维度 $d_{ff}$ 和模型维度 $d_{model}$。它们之间的比例应该是什么？
$$d_{ff} = 4 d_{model}$$
这几乎总是正确的。只有少数几个例外。

---

## 第 38 页 (Page 38)

### 例外 #1——GLU 变体

请记住，GLU 变体的计算流需要两个输入矩阵，因此维度大约缩小为原来的 2/3。这意味着大多数 GLU 变体具有 $d_{ff} = \frac{8}{3} d_{model} \approx 2.66 d_{model}$。

| 模型 (Model) | $d_{ff}/d_{model}$ 比例 |
| :--- | :---: |
| PaLM | 4.0 |
| Mistral 7B | 3.5 |
| LLaMA-2 70B | 3.5 |
| LLaMA 70B | 2.68 |
| Qwen 14B | 2.67 |
| DeepSeek 67B | 2.68 |
| Yi 34B | 2.85 |
| T5 v1.1 | 2.5 |

---

## 第 39 页 (Page 39)

### 例外 #2——T5

大多数大语言模型的超参数设置都相当保守。一个例外是 **T5** [Raffel et al. 2020]，它有一些非常大胆的设置。对于其 11B 模型：
$$d_{ff} = 65,536$$
$$d_{model} = 1024$$
达到了令人震惊的 **64 倍乘数**！

> “对于 11B 模型，我们使用 $d_{ff} = 65,536$ 和 128 头注意力。我们选择扩大 $d_{ff}$ 是因为现代加速器（如 TPU）在进行大型稠密矩阵乘法时效率最高。”

*其他最近的例外：Gemma 2 (8倍), SmolLM/Gemma 3/Gemma 4 (4倍, 使用 GLU)*

---

## 第 40 页 (Page 40)

### 为什么是这个乘数范围？

经验上，在前馈比例（$d_{ff}/d_{model}$）为 **1 到 10 之间**存在一个“盆地”，此时该超参数接近最优。

*(图表来自 Kaplan et al. 2020，展示了在 50M 参数模型上，Loss 增加与前馈层比例的关系)*

---

## 第 41 页 (Page 41)

### 我们能从模型维度超参数中学到什么？

* 默认的选择 $d_{ff} = 4d_{model}$ 和 $d_{ff} = 2.66d_{model}$ 几乎在所有现代 LLM 中都表现良好。
* 但 T5 表明，即使是 $d_{ff} = 64d_{model}$ 这样激进的选择也能奏效。超参数的选择并非一成不变。
* 话虽如此，T5 拥有一个后续模型（T5 v1.1），它在 GeGLU 上使用了更标准的 2.5 倍乘数，因此 64 倍乘数很可能是次优的。

---

## 第 42 页 (Page 42)

### 令人惊讶（？）的共识超参数 2

**头维度 (Head-dim) * 头数 (num-heads) 与模型维度 (model-dim) 的比例。**

*多头自注意力在计算上是高效的：*
* 尽管我们计算了 $h$ 个注意力头，但在计算上它其实并没有更昂贵。
* 我们计算 $XQ \in \mathbb{R}^{n \times d}$，然后将其重塑为 $\mathbb{R}^{n \times h \times d/h}$。（对 $XK$ 和 $XV$ 也是如此。）
* 这并不是绝对的：我们可以让头维度大小大于 model-dim / num-heads。但大多数模型都遵循 $h \times d_h = d_{model}$ 这个指导原则（即比例为 1）。

---

## 第 43 页 (Page 43)

### 头数是多少，模型维度又是多少？

该超参数的一些示例：

| 模型 (Model) | 头数 (Num heads) | 头维度 (Head dim) | 模型维度 (Model dim) | 比例 (Ratio) |
| :--- | :---: | :---: | :---: | :---: |
| GPT3 | 96 | 128 | 12,288 | 1.0 |
| T5 | 128 | 128 | 1,024 | 16.0 |
| T5 v1.1 | 64 | 64 | 4,096 | 1.0 |
| LaMDA | 128 | 128 | 8,192 | 2.0 |
| PaLM | 48 | 258 | 18,432 | 1.48 |
| LLaMA2 | 64 | 128 | 8,192 | 1.0 |
| Qwen 3.5 (27B) | 24 | 256 | 5,120 | 1.2 |

*大多数模型的比例在 1 左右——值得注意的例外主要来自 Google 的一些模型。*

---

## 第 44 页 (Page 44)

### 纵横比（Aspect ratios）

我的模型应该设计得更深还是更宽？多深，多宽？

| 模型 (Model) | $d_{model}/n_{layer}$ 比例 |
| :--- | :---: |
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM (540B) | 156 |
| GPT3/OPT/Mistral/Qwen/OLMo 3 | 128 |
| LLaMA / LLaMA2 | 102 |
| Gemma 3 | 87 |
| Gemma 4 | 61 |
| T5 (11B) | 33 |

> **存在黄金分割点？** 大部分大模型的比例在 100 到 200 之间。

---

## 第 45 页 (Page 45)

### 关于纵横比的考量

> **极深的模型更难并行化，并且具有更高的延迟。**

因为层与层之间是串行计算的，必须等待前一层的输出。而增加宽度（width）则非常容易在成百上千个设备上进行并行化。

*(参考 Tay et al. 2021)*

---

## 第 46 页 (Page 46)

### 关于纵横比扩展的证据

在一定范围内，不同纵横比的架构在给定相同 FLOPs 预算下可以达到非常相似的性能性能。

*(图表来自 Kaplan et al. 2020 以及 Tay et al. 2021)*

---

## 第 47 页 (Page 47)

### 典型的词表大小是多少？

* **单语言模型** —— 30k-50k 词表：
  * *原始 Transformer*: 37,000
  * *GPT*: 40,257
  * *GPT2/3*: 50,257
  * *T5/T5v1.1*: 32,128
  * *LLaMA*: 32,000
* **多语言 / 生产系统** —— 100k-250k：
  * *mT5*: 250,000
  * *PaLM*: 256,000
  * *GPT4*: 100,276
  * *Gemma 4*: 262,144
  * *DeepSeek*: 100,000
  * *Qwen 15B*: 152,064
  * *Yi*: 64,000

> **结论**：单语言词表不需要太大，但多语言词表必须很大，以确保对各种语言的压缩率。

---

## 第 48 页 (Page 48)

### Dropout 与其他正则化

**我们在预训练期间需要正则化吗？**

* **反对正则化的论点：**
  * 数据量巨大（数万亿 token），远超参数数量。
  * SGD 只对语料库进行单次遍历（one-pass），模型几乎不可能“死记硬背”所有样本。
* 既然如此，为什么人们在实践中还要使用正则化呢？

---

## 第 49 页 (Page 49)

### 实践中的 Dropout 与权重衰减

*多数情况下，开源模型的论文并不使用或不讨论 dropout。*

| 模型 | Dropout* | 权重衰减 (Weight decay) |
| :--- | :---: | :---: |
| 原始 Transformer | 0.1 | 0 |
| GPT2 | 0.1 | 0.1 |
| T5 | 0.1 | 0 |
| GPT3 | 0.1 | 0.1 |
| T5 v1.1 | 0 | 0 |
| PaLM | 0 | 可变 |
| OPT | 0.1 | 0.1 |
| LLaMA | 0 | 0.1 |
| Qwen 14B | 0.1 | 0.1 |

* 许多较旧的模型在预训练期间使用了 dropout。
* **较新的模型**（除 Qwen 外）仅依赖权重衰减（weight decay），不使用 dropout。

---

## 第 50 页 (Page 50)

### 为什么要对大语言模型进行权重衰减？

[Andriushchenko et al. 2023] 对 LLM 中的权重衰减有一些有趣的观察：
* 这并不是为了控制过拟合（因为是一次性通过数据）。
* 权重衰减与学习率（余弦调度）相互作用，能改变有效步长，从而有利于优化的稳定性。

---

## 第 51 页 (Page 51)

### 总结：超参数

* **前馈层 (Feedforward)**：4 倍的经验法则（GLU 为 8/3 倍）是标准配置。
* **头维度 (Head dim)**：头维度 * 头数 = 模型维度 D 是标准配置，但很少经过严格验证。
* **纵横比 (Aspect ratio)**：合理取值的范围很宽（100-200）。系统的考量（例如延迟和并行化难度）决定了该取值。
* **正则化 (Regularization)**：我们仍然会使用权重衰减，但其作用主要体现在优化动力学上。

---

## 第 52 页 (Page 52)

### 稳定性技巧

最近，网络训练稳定性受到了广泛关注。

> **不要训练呈现出像蓝色曲线那样波动的模型！**

*(折线图展示了不稳定的训练 Loss 尖峰，而稳定的曲线如橙色 OlMo 2 曲线所示)*

---

## 第 53 页 (Page 53)

### 问题出在哪里？警惕 Softmax！

**Softmax 操作可能会导致行为异常（由于指数运算或除以零）。**

---

## 第 54 页 (Page 54)

### 输出 Softmax 稳定性——“z-loss”

回顾 Softmax 计算：
$$\log(P(x)) = U_r(x) - \log(Z(x))$$
$$Z(x) = \sum_{r'=1}^{|V|} e^{U_{r'}(x)}$$

辅助损失设计 (鼓励 $\log(Z)$ 接近 0)：
$$L = \sum_i \left[ \log(P(x_i)) - \alpha \log^2(Z(x_i)) \right]$$

*这对稳定性非常有用！PaLM 使用了这一 “z-loss” 技巧：*
> “我们额外使用了 $z\_loss = 10^{-4} \cdot \log^2 Z$ 的辅助损失。我们发现这能提高训练稳定性。”

*其他例子：Baichuan 2 (2023), DCLM (2024), OLMo 2 (2025), OLMo 3 (2025)*

---

## 第 55 页 (Page 55)

### 注意力 Softmax 稳定性——“QK norm”

* Query 和 Key 在进入 Softmax 操作之前要先进行 Layer (RMS) 归一化，防止注意力权重中的点积值过大导致 softmax 溢出或梯度消失。
* **其他例子**：DCLM, OLMo2, Gemma 2, Qwen3, OLMo 3, Gemma 4。
* 最初源于视觉和多模态模型 [Dehghani et al. 2023, Idefics, Chameleon]。

---

## 第 56 页 (Page 56)

### Logit 软截断 (Logit soft-capping)

通过 Tanh 函数将 logit 限制在某个最大值范围内：
$$\text{logits} \gets \text{soft\_cap} \times \tanh\left(\frac{\text{logits}}{\text{soft\_cap}}\right)$$

> “我们在自注意力层将 soft_cap 参数设置为 50.0，在最终输出层设置为 30.0。”

这能有效防止 logit 数值爆炸，但可能在某些硬件上会带来实际运行性能（速度）的问题。

---

## 第 57 页 (Page 57)

### 注意力头

除了少数次要例外，大多数模型几乎不怎么改动注意力头：
* **GQA / MQA**：通过减少 Key/Value 的头数来降低推理成本（共享 KV cache 内存）。
* **稀疏或滑动窗口注意力**（GPT4/Mistral）：限制注意力模式以降低长文本的计算和显存成本。
* **奇特的 SSM 相关技术**（Jamba, Falcon 3, Qwen 3.5 等）：见下一讲！

---

## 第 58 页 (Page 58)

### GQA/MQA——降低注意力头成本

计算开销分析：
* **总算术运算量**：$O(bnd^2)$
* **总内存访问量**：$O(bnd + bhn^2 + d^2)$
* **算术强度（计算/内存比）** 很高：$O\left(\left(\frac{1}{k} + \frac{1}{bn}\right)^{-1}\right)$ —— 我们可以让 GPU 保持高速运转。

*(其中：d = 隐藏维度，b = 批大小，n = 序列长度，h = 头数，k = 头维度)*

---

## 第 59 页 (Page 59)

### GQA/MQA——增量文本生成的情况

* **关键差异**：增量生成过程（decode）无法并行化——必须一步一步地进行。
* 在这种情况下，我们需要通过“**KV cache**”增量地重新计算/更新注意力，避免重复计算历史 token。

---

## 第 60 页 (Page 60)

### GQA/MQA——增量生成的算术强度

* **总算术运算量**：$O(bnd^2)$
* **总内存访问量**：$O(bn^2d + nd^2)$
* **算术强度非常糟糕**：$O\left(\left(\frac{n}{d} + \frac{1}{b}\right)^{-1}\right)$ —— 此时需要很大的批大小 (b)，或者短序列长度 (n) 和大模型维度 (d) 才能稍微缓解。
* *有没有办法绕过这个限制？因为 $n/d$ 项很难降低。*

---

## 第 61 页 (Page 61)

### MQA——仅仅减少 key 和 value 的维度

* **核心思想**：保留多个 query 头，但对于 key 和 value 仅使用一个头（所有 query 头共享相同的 KV 对）。
* 这使得我们需要移动到内存（KV Cache）中和取出的张量数据大为减少。
* **总内存访问量**：$O(bnd + bn^2k + nd^2)$
* **算术强度**：$O\left(\left(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b}\right)^{-1}\right)$

*(插图引自 fireworks.ai 博客)*

---

## 第 62 页 (Page 62)

### 其他扩展——GQA (Grouped-Query Attention)

* 不彻底走到只有 1 个 KV 头的极端（MQA）——而是使用较少的 KV 头组（介于 MHA 和 MQA 之间）。
* 一个简单的控制表达能力（Key-Query 比例）与推理效率的旋钮。
* **更近期的进展**：DeepSeek-V2 提出的 **MLA (Multi-head Latent Attention)**，通过低秩投影压缩 KV cache 维度。

---

## 第 63 页 (Page 63)

### MQA 会带来性能损伤吗？

* 有时会。使用 MQA 有轻微的困惑度（PPL）损失 [Shazeer 2019]。
* 使用 GQA 时几乎没有性能损失，但能大幅提高推理吞吐量 [Ainslie 2023]。

---

## 第 64 页 (Page 64)

### 稀疏/滑动窗口注意力

* 关注整个上下文可能非常昂贵（二次方复杂度）。
* 构建稀疏/结构化注意力，以微弱的表达能力下降来权衡运行时间（如 GPT-3, GPT OSS, Gemma 4 中的稀疏设计）。

*(插图来自 Child et al. 2019)*

---

## 第 65 页 (Page 65)

### 当前的标配技巧——交替使用全局和局部注意力

* **来自 Cohere Command A** —— 每 4 层中有一层是全局注意力，其余为滑动窗口注意力（SWA）。
* 通过无位置编码（NoPE）传递长程信息，通过 RoPE + SWA 传递短程信息。
* **其他模型** —— LLaMA 4, Gemma 3, Gemma 4, OLMo 3 均采用了 SWA + 全局 RoPE 交替结构。

---

## 第 66 页 (Page 66)

### 其他最近的交替注意力示例

* Gemma 4, Olmo 3, Qwen 3.5 / Qwen 3 Next 的注意力结构设计。

---

## 第 67 页 (Page 67)

### 回顾与结论

* 在大型语言模型之间，Transformer 的许多方面（架构、超参数）基本是相通的。
* **主要区别点**：
  * 位置嵌入（Position embeddings）
  * 激活函数（Activations）
  * 分词（Tokenization）

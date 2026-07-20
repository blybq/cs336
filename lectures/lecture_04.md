# 第 4 讲：注意力机制替代方案与混合专家模型 (Attention Alternatives and Mixtures of Experts)

CS336
Tatsu H

---

## 第 1 页 (Page 1)

# 第 4 讲
## 注意力机制替代方案与混合专家模型

CS336  
Tatsu H  

---

## 第 2 页 (Page 2)

### 注意力机制替代方案 (Attention alternatives)
随着上下文窗口的增大，注意力机制的开销也随之增加……我们该如何控制这些开销？  
[了解增加LLM上下文窗口的影响 (meibel.ai)](https://www.meibel.ai/post/understanding-the-impact-of-increasing-llm-context-windows)

---

## 第 3 页 (Page 3)

### “基础”工具箱 (The ‘basic’ toolkit)
- 结合局部（Local）与全局（Global）注意力机制
- 系统工程优化 (Systems engineering)

**但如果我们想要更彻底、潜在收益更大的提升呢？**

---

## 第 4 页 (Page 4)

### 线性注意力机制 (Linear attention)
考虑常见的注意力机制操作：
$$Q \in \mathbb{R}^{n \times d_k}, \quad K \in \mathbb{R}^{n \times d_k}, \quad V \in \mathbb{R}^{n \times d_v}$$

$$\operatorname{Attn}(Q, K, V) = \rho(Q K^\top) V$$

由于存在 $Q K^\top$ 的计算，其复杂度是二次方的（即 $\mathcal{O}(n^2 d_k)$）。当 $\rho$ 为恒等映射（Identity）时，我们能做得更好吗？
$$(Q K^\top) V = Q (K^\top V)$$

虽然这看起来非常简单，但却出乎意料地重要。我们成功将计算复杂度从 $\mathcal{O}(n^2 d_k + n^2 d_v)$ 降低到了 $\mathcal{O}(2 n d_v d_k)$。

*(参考文献：Shen et al. 2018，Katharopoulos 2020（核函数版本）。这也与快速权重程序设计器 (fast weight programmers) 等概念相关。)*

---

## 第 5 页 (Page 5)

### 线性注意力机制的循环形式 (Recurrent form of linear attention)
回想在纯线性注意力中，我们对计算顺序重新排列：
$$(Q K^\top) V = Q (K^\top V)$$

这虽然是线性时间复杂度（非常棒），但更妙的是，它看起来非常像一个循环神经网络（RNN）：
$$S_t = S_{t-1} + k_t v_t^\top \quad \text{和} \quad y_t = q_t^\top S_t$$

这种“对偶性”（Duality）使我们能够利用并行的二次方形式进行高效训练，并利用串行的线性形式进行高效推理。

*(注意：如果用 $\gamma$ 对 $S_{t-1}$ 进行加权，就会得到 RetNet。)*

---

## 第 6 页 (Page 6)

### Minimax M1
Minimax M1（以及 minimax-text-01）采用了一种 7:1 的混合线性注意力机制（即 7 层线性注意力层结合 1 层全注意力层）。  
整体性能非常强劲，且在上下文长度上展现出线性缩放的特性。

---

## 第 7 页 (Page 7)

### 从线性注意力到 Mamba-2 (From linear attention to Mamba-2)
让我们对线性注意力做一些泛化，加入位置权重（per-position weights）：
- 线性注意力：  
  $$S_t = S_{t-1} + k_t v_t^\top \quad \text{和} \quad y_t = q_t^\top S_t$$
- Mamba-2：  
  $$S_t = \gamma_t S_{t-1} + k_t v_t^\top \quad \text{和} \quad y_t = q_t^\top S_t + v_t^\top D \quad \text{其中} \quad \gamma_t = f(x_t)$$

对此有更多的理论支撑与论证（可以去阅读 Mamba-2 的论文），但在机制上，我们能够通过门控机制（gating）让线性注意力具备更强的表达能力（门控是个好东西！）。  
这也同样保持了对偶属性（可以并行计算 $\gamma$，然后应用对偶性）。

---

## 第 8 页 (Page 8)

### Nemotron 3
采用了 Mamba-Attention 混合架构（比例大概在 3:1 左右）——与其它同类模型相比，其性能表现相当或甚至更优。

---

## 第 9 页 (Page 9)

### 门控 Delta 网络及相关架构 (Gated delta net (and friends))
让我们进一步做泛化——对输入进行门控，并选择性地抹去（erase）状态。
- Mamba-2：  
  $$S_t = \gamma_t S_{t-1} + k_t v_t^\top \quad \text{和} \quad y_t = q_t^\top S_t + v_t^\top D \quad \text{其中} \quad \gamma_t = f(x_t)$$
- Gated Delta Net：  
  $$S_t = \gamma_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top \quad \text{和} \quad y_t = q_t^\top S_t \quad \text{其中} \quad \gamma_t = f(x_t), \beta_t = f(x_t)$$

门控 Delta 网络加入了一个“无输入操作”门控 ($\beta = 0$)，并抹去在当前键（Key）方向上的任何内容（即通过 $I - \beta_t k_t k_t^\top$ 过滤）。  
这与多种快速权重编程（Fast Weight Programming）以及测试时训练（Test Time Training）思想有非常紧密的联系。

---

## 第 10 页 (Page 10)

### Qwen 3.5 / Qwen Next
最新的 Qwen 模型是 3:1 的 GDN (Gated DeltaNet) 与 Attention 混合架构。  
同样地，它们展现出了相当不错的性能，且具有良好的推理特性。

---

## 第 11 页 (Page 11)

### 混合架构的性能 (Hybrid performance)
虽然目前还没有太多受控的消融实验（controlled ablations），但已有部分证据表明，在较低的混合比例下模型能够实现较低的损失（loss）。

---

## 第 12 页 (Page 12)

### 混合架构的替代方案：稀疏自适应 (Alternative to hybrids: sparse adaptation)
与其关注每个 Token（attending to every token），不如进行稀疏注意力机制优化（DSA，DeepSeek Sparse Attention）。  
索引器（indexer）可以做得非常轻量，从而带来显著的效率收益。  
这可以在密集短上下文预训练之后进行“事后”（post hoc）自适应调整。

---

## 第 13 页 (Page 13)

### DSA – Deepseek Sparse Attention (v3.2, GLM5)
*(介绍 DeepSeek 稀疏注意力机制在 v3.2 和 GLM5 中的应用与基准评估)*

---

## 第 14 页 (Page 14)

### 混合专家模型 (Mixture of experts)
- GPT-4 (?)
- Grok
- DeepSeek-V3 技术报告
- Llama 4
- OLMoE

---

## 第 15 页 (Page 15)

### 什么是 MoE？ (What’s a MoE?)
- 用多个大型前馈网络（FFN，称为专家 Expert）和一个选择器层（Router）来替代传统的前馈传播层。
- 你可以在不影响计算量（FLOPs）的前提下增加专家的数量。

*(参考文献：Fedus et al. 2022)*

---

## 第 16 页 (Page 16)

### 为什么 MoE 越来越流行？ (Why are MoEs getting popular?)
- **相同的计算量下，参数量越大模型表现越好。**

*(参考文献：Fedus et al. 2022)*

---

## 第 17 页 (Page 17)

### 为什么 MoE 越来越流行？ (续)
- **训练 MoE 的速度更快。**

*(参考文献：OLMoE)*

---

## 第 18 页 (Page 18)

### 为什么 MoE 越来越流行？ (续)
- **与稠密（Dense）等效模型相比极具竞争力。**

---

## 第 19 页 (Page 19)

### 为什么 MoE 越来越流行？ (续)
- **可并行部署在多台设备上。**

---

## 第 20 页 (Page 20)

### 一些西方推出的 MoE 结果 (Some MoE results – from the west)
- MoE 模型占据了开源模型中性能最高的那一梯队，且推理速度非常快。
- 比较模型：Llama 4 (Maverick), Gemini 2.0 Flash, DeepSeek v3.1, GPT-4o 等。

---

## 第 21 页 (Page 21)

### 早先国内团队的 MoE 结果 – Qwen (Earlier MoE results from Chinese groups – Qwen)
中国的大模型公司在较小规模模型上也开展了相当多的 MoE 相关工作。  
- 涉及模型如：Qwen1.5-7B, Qwen1.5-MoE-A2.7B, DeepSeekMoE 16B 等。

---

## 第 22 页 (Page 22)

### 早先国内团队的 MoE 结果 – DeepSeek (Earlier MoE results from Chinese groups - DeepSeek)
关于 MoE 模型也有一些很好的近期消融实验工作，结果表明 MoE 架构通常具有很好的效果。

---

## 第 23 页 (Page 23)

### 近期的 MoE 成果 – DeepSeek v3 (Recent MoE results – DeepSeek v3)
*(展示了 DeepSeek-V3 的性能指标，例如 MMLU-Pro、GPQA、MATH 500、AIME 2024 等指标)*

---

## 第 24 页 (Page 24)

### 为什么以前 MoE 没有那么流行？ (Why haven’t MoEs been more popular?)
- **基础设施非常复杂**：在多节点环境下的优势更明显，但实现难度高。
- **训练目标有些偏启发式**（有时不太稳定）。

*(参考文献：Zoph et al. 2022, Fedus et al. 2022)*

---

## 第 25 页 (Page 25)

### 常见的 MoE 架构长什么样 (What MoEs generally look like)
- **典型做法**：用 MoE 层替代 MLP / FFN 层。
- **较少见做法**：对注意力头（Attention Heads）使用 MoE。

*(参考文献：ModuleFormer, JetMoE)*

---

## 第 26 页 (Page 26)

### MoE – 变化体现在哪里？ (MoE – what varies?)
- 路由函数 (Routing function)
- 专家大小 (Expert sizes)
- 训练目标 (Training objectives)

---

## 第 27 页 (Page 27)

### 路由函数概览 (Routing function - overview)
许多路由算法最终都归结为“选择前 K 个 (choose top k)”：
- **Token 选择专家** (Token chooses expert)
- **专家选择 Token** (Expert chooses token)
- **通过优化进行全局路由** (Global routing via optimization)

*(参考文献：Fedus et al. 2022)*

---

## 第 28 页 (Page 28)

### 路由类型 (Routing type)
几乎所有的 MoE 都采用标准的“Token选择 Top-K (token choice topk)”路由。这里展示了一些近期的消融实验结果。

---

## 第 29 页 (Page 29)

### 常见路由变体详解 (Common routing variants in detail)
- **Top-k 路由** (在绝大多数 MoE 中使用)：
  - Switch Transformer ($k=1$)
  - GShard ($k=2$), Grok ($k=2$), Mixtral ($k=2$), Qwen ($k=4$), DBRX ($k=4$), DeepSeek ($k=7$)
- **哈希路由** (Hashing)：
  - 常用的 baseline 方案。

*(参考文献：Fedus et al. 2022)*

---

## 第 30 页 (Page 30)

### 其它路由方法 (Other routing methods)
- **通过强化学习（RL）来学习路由路径**：曾在一些极早期工作中使用（如 Bengio 2013），但现在已不常用。
- **求解匹配问题**（BASE 路由）：
  - 将路由视为线性分配（Linear Assignment）问题，在 Clark ‘22 等论文中有所应用。

*(参考文献：Fedus et al. 2022)*

---

## 第 31 页 (Page 31)

### Top-K 路由详解 (Top-K routing in detail)
大多数论文都采用经典且古老的 top-k 路由。它是如何工作的？
- 输入表示为 $\mathbf{h}_t^l$：
  $$\mathbf{h}_t^l = \sum_{i=1}^N \left( g_{i,t} \operatorname{FFN}_i \left( \mathbf{u}_t^l \right) \right) + \mathbf{u}_t^l$$
- 门控系数 $g_{i,t}$ 计算公式：
  $$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} | 1 \le j \le N\}, K), \\ 0, & \text{otherwise} \end{cases}$$
  $$s_{i,t} = \operatorname{Softmax}_i \left( \mathbf{u}_t^{l\top} \mathbf{e}_i^l \right)$$
- 门控参数由一个逻辑回归器（logistic regressor）选择。
- 这是 DeepSeek (V1-2) 所采用的路由器架构（Grok, Qwen 也采用这种方式）。
- Mixtral, DBRX, DeepSeek v3 则是在 TopK 选择之后再进行 softmax 计算。

*(参考文献：Dai et al. 2024)*

---

## 第 32 页 (Page 32)

### DeepSeek 及其他国内语言模型近期的路由变体 (Recent variations from DeepSeek and other Chinese LMs)
- **核心思想**：采用更小、数量更多的专家，配合少量始终处于开启状态的共享专家（Shared Experts）。
- （该方案应用在 DeepSeek / Qwen 模型中，最初起源于 DeepSpeed MoE）。

---

## 第 33 页 (Page 33)

### DeepSeek 论文中的多项消融实验 (Various ablations from the DeepSeek paper)
实验表明，更多的专家以及共享专家设计通常都有助于提升模型的整体性能。

---

## 第 34 页 (Page 34)

### OLMoE 中的消融实验 (Ablations from OlMoE)
- 细粒度专家带来了性能增益，但共享专家没有带来额外收益。

---

## 第 35 页 (Page 35)

### 近期 MoE 的专家路由配置汇总 (Expert routing setups for recent MoEs)

| 模型 | 总路由专家数 (Routed) | 激活专家数 (Active) | 共享专家数 (Shared) | 细粒度专家比例 (Fine-grained ratio) |
| :--- | :---: | :---: | :---: | :---: |
| GShard | 2048 | 2 | 0 | - |
| Switch Transformer | 64 | 1 | 0 | - |
| ST-MoE | 64 | 2 | 0 | - |
| Mixtral | 8 | 2 | 0 | - |
| DBRX | 16 | 4 | 0 | - |
| Grok | 8 | 2 | 0 | - |
| DeepSeek v1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1/8 |
| DeepSeek v3 | 256 | 8 | 1 | 1/14 |
| OLMoE | 64 | 8 | 0 | 1/8 |
| MiniMax | 32 | 2 | 0 | ~1/4 |
| Llama 4 (maverick) | 128 | 1 | 1 | 1/2 |

---

## 第 36 页 (Page 36)

### 我们如何训练 MoE？ (How do we train MoEs?)
- **主要挑战**：为了训练效率，我们需要稀疏性……但是稀疏门控的决策过程是**不可微**的！
- **解决方案？**
  1. 强化学习（RL）以优化门控策略。
  2. 随机扰动（Stochastic perturbations）。
  3. 启发式“负载均衡”损失（Heuristic ‘balancing’ losses）。
- 猜猜大家在实际应用中用的是哪一种？

---

## 第 37 页 (Page 37)

### 面向 MoE 的强化学习 (RL for MoEs)
- 基于 REINFORCE 算法的强化学习确实有效，但相比其他方案并没有明显优势，因此没有绝对胜出。
- 强化学习是“正确的数学解法”，但由于梯度方差大且实现极其复杂，因而并未被广泛使用。

*(参考文献：REINFORCE baseline 方案, Clark et al. 2020)*

---

## 第 38 页 (Page 38)

### 随机近似 (Stochastic approximations)
$$G(x) = \operatorname{Softmax}(\operatorname{KeepTopK}(H(x), k))$$
$$H(x)_i = (x \cdot W_g)_i + \operatorname{StandardNormal}() \cdot \operatorname{Softplus}((x \cdot W_{noise})_i)$$
$$\operatorname{KeepTopK}(v, k)_i = \begin{cases} v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v, \\ -\infty & \text{otherwise}. \end{cases}$$

- 路由决策是通过高斯扰动进行随机化处理的。
- 1. 这能自然而然地训练出更具鲁棒性的专家模型。
- 2. Softmax 的引入使得模型能够学会如何对前 K 个专家进行排序。

*(参考文献：Shazeer et al. 2017)*

---

## 第 39 页 (Page 39)

### 随机近似（续）
```python
if is_training:
    # 增加噪声以促进跨专家的探索
    router_logits += mtf.random_uniform(shape=router_logits.shape, minval=1-eps, maxval=1+eps)
# 将输入转化为 float32 进行 softmax 以保证数值稳定性
router_logits = mtf.to_float32(router_logits)
# 每个 token 被分发到各个专家的概率
router_probs = mtf.softmax(router_logits, axis=-1)
```
- Fedus et al. 2022 中引入了**随机抖动（Stochastic jitter）**。这种方法通过均匀的乘性扰动来减少专家模型的脆性，不过后来在 Zoph et al. 2022 中被移除了。

---

## 第 40 页 (Page 40)

### 启发式负载均衡损失 (Heuristic balancing losses)
另一个关键问题是——为了提高系统效率，我们需要尽量均匀地使用各个专家。
$$\operatorname{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i$$
其中 $f_i$ 是分发到专家 $i$ 的 Token 比例：
$$f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbb{1}\{\operatorname{argmax} p(x) = i\}$$
而 $P_i$ 是分配给专家 $i$ 的路由器概率分量：
$$P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x)$$

损失对 $p_i(x)$ 的导数为 $\frac{\alpha N}{T^2} \sum \mathbb{1}_{\operatorname{argmax} p(x)=i}$，这意味着越频繁被使用的专家，受到的下调惩罚越重。

*(参考文献：Switch Transformer [Fedus et al. 2022])*

---

## 第 41 页 (Page 41)

### 来自 DeepSeek (v1-2) 的负载均衡实例 (Example from deepseek (v1-2))
- **每个专家的负载均衡（Per-expert balancing）**：与 Switch Transformer 一致。
  $$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i$$
  $$f_i = \frac{N'}{K' T} \sum_{t=1}^T \mathbb{1}(\text{Token } t \text{ 选择 Expert } i)$$
  $$P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}$$
- **每个设备的负载均衡（Per-device balancing）**：将上述目标按设备（Device）进行聚合。
  $$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^D f'_i P'_i$$
  $$f'_i = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j \quad \text{和} \quad P'_i = \sum_{j \in \mathcal{E}_i} P_j$$

---

## 第 42 页 (Page 42)

### DeepSeek V3 的变体：专家偏置机制 (DeepSeek v3 variation – per-expert biases)
为每个专家设置一个偏置（使得某些专家更容易分到 Token），并使用在线学习（online learning）调整偏置。
$$g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk}(\{s_{j,t} + b_j \vert 1 \le j \le N_r\}, K_r), \\ 0, & \text{otherwise.} \end{cases}$$

他们称这套机制为 **“无辅助损失负载均衡” (auxiliary loss free balancing)** （不过实际上它并非完全不包含辅助损失……）

**互补的序列级辅助损失（Complementary Sequence-Wise Auxiliary Loss）：**
$$\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i$$
$$f_i = \frac{N_r}{K_r T} \sum_{t=1}^T \mathbb{1}(s_{i,t} \in \operatorname{Topk}(\{s_{j,t} \vert 1 \le j \le N_r\}, K_r))$$
$$s'_{i,t} = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}} \quad \text{和} \quad P_i = \frac{1}{T} \sum_{t=1}^T s'_{i,t}$$

---

## 第 43 页 (Page 43)

### 去除负载均衡损失会发生什么？ (What happens when removing load balancing losses?)
实验结果表明，在完全不添加负载均衡损失（LBL）的情况下，模型的专家分配会在训练早期极快地收敛到少数几个专家（形成垄断），导致系统资源严重倾斜，而加入负载均衡损失则能引导专家分配走向平稳。

---

## 第 44 页 (Page 44)

### 训练 MoE —— 系统层面 (Training MoEs – the systems side)
- MoE 在并行化上表现良好——因为每个前馈网络（FFN / Expert）可以独立装载在单台设备上。
- 这为训练带来了额外的并行化维度（如专家并行 Expert Parallelism）。

---

## 第 45 页 (Page 45)

### 训练 MoE —— 系统层面（续）
- MoE 路由虽然允许高度并行化，但也会引入额外的计算与调度复杂度。
- 现代大模型开源库（如 MegaBlocks）通常会使用更聪明的**块稀疏矩阵乘法**（Block-Sparse MMs）来优化整体的计算吞吐。

---

## 第 46 页 (Page 46)

### MoE 并行化与架构调整 (MoE parallelism and architecture modifications)
- **来自 Nemotron 3 的新思路**：通过对激活值进行下投影（down-projecting）来减少集群节点间的通信开销。

---

## 第 47 页 (Page 47)

### 有趣的边缘问题——MoE 模型的随机性 (Fun side issue – stochasticity of MoE models)
MoE 模型可能会比普通模型引入额外的随机性。

**为什么 MoE 会有额外的随机性？**
路由中的 **Token 丢弃 (Token dropping)** 是在批次（Batch）级别发生的——这意味着**其他人的查询（Queries）有可能会挤掉属于你的 Token！**

---

## 第 48 页 (Page 48)

### MoE 的稳定性问题 (Issues with MoEs - stability)
- **解决方案**：仅在专家路由器（Expert Router）上使用 Float 32 进行计算（有时会搭配辅助 $z$-loss 进行约束）。
  $$L_z(x) = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^N e^{x_j^{(i)}} \right)^2$$

*(参考文献：Zoph 2022)*

---

## 第 49 页 (Page 49)

### 路由器的 Z-loss 稳定性 (Z-loss stability for the router)
消融实验对比了添加路由器 $z$-loss（权重为 0.001）与不添加的情况。去除 $z$-loss 会导致训练过程中出现严重的损失尖峰（instability）。

---

## 第 50 页 (Page 50)

### MoE 模型微调问题 (Issues with MoEs – fine-tuning)
- 稀疏的 MoE 容易在较小规模的微调数据上发生**过拟合**。
- **Zoph et al. 解决方案**：仅微调非 MoE 部分的 MLP 层。
- **DeepSeek 解决方案**：使用大量的微调数据（例如 1.4M 的有监督微调数据 SFT）。

---

## 第 51 页 (Page 51)

### 其它训练方法——向上回收 (Other training methods - upcycling)
我们是否可以利用预训练好的稠密语言模型（Dense LM）来初始化一个 MoE 模型？

---

## 第 52 页 (Page 52)

### 向上回收案例——MiniCPM (Upcycling example - MiniCPM)
- 采用了 MiniCPM 模型进行实验（配置为 Topk=2, 8 个专家，对应约 4B 激活参数）。
- 这种简单的 MoE 架构在约 520B Token 的持续训练下，相较于基准 Dense 模型展示出了显著的性能提升。

---

## 第 53 页 (Page 53)

### 向上回收案例——Qwen MoE (Upcycling example – Qwen MoE)
- Qwen MoE：基于 Qwen 1.8B 模型初始化，配置为 Top-k=4，总共 60 个专家，其中 4 个为共享专家。
- 架构设计上与 DeepSeekMoE 相似，这也是最早被证实能够成功通过 Upcycling 获得成效的模型之一。

---

## 第 54 页 (Page 54)

### DeepSeek MoE v1-v2-v3 架构演进
让我们以 DeepSeek MoE 的架构细节来收尾。

#### V1 版本 (16B 总参数，单 Token 激活 2.8B)：
- 采用标准的 Top-K 路由。
- 架构设计：2 个共享专家 + 64 个细粒度专家（激活其中 4 个）。
- 采用标准的辅助损失负载均衡机制（专家级 + 设备级）。

---

## 第 55 页 (Page 55)

### DeepSeek MoE v2
#### V2 版本 (236B 总参数，单 Token 激活 21B)：
- 架构设计：2 个共享专家 + 160 个细粒度专家（共激活 6 个）。
- **新增特性**：
  - Top-M 设备路由。
  - 通信负载均衡损失——平衡各个设备的流入和流出通信量。

---

## 第 56 页 (Page 56)

### DeepSeek MoE v3
#### V3 版本 (671B 总参数，单 Token 激活 37B)：
- 架构设计：1 个共享专家 + 258 个细粒度专家（共激活 8 个）。
- **新增特性**：
  - Sigmoid + Softmax 结合的 Top-K 和 Top-M 路由。
  - 无辅助损失负载均衡（Aux-loss-free）与序列级辅助损失（Seq-wise Aux）。

---

## 第 57 页 (Page 57)

### 彩蛋：构建 DeepSeek MoE v3 还需要什么？ (Bonus: What else do you need to make DeepSeek MoE v3?)
#### MLA：多头潜变量注意力机制 (Multihead Latent Attention)
- **核心思想**：将 Query (Q)、Key (K)、Value (V) 投影表达为低维“潜变量 (latent)”激活值的函数。

---

## 第 58 页 (Page 58)

### 构建 DeepSeek MoE v3 还需要什么？（续）
- **核心公式**：
  $$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t, \quad \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$$
- **收益**：在进行 KV 缓存时，我们只需要存储 $\mathbf{c}_t^{KV}$，它能比传统形式小得多。其中 $W^{UK}$ 可以融合进 Q 投影矩阵。
  $$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t, \quad \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q$$
- **复杂度**：RoPE（旋转位置编码）会与 MLA 式的缓存产生冲突。
  - 没有 RoPE 时：$\langle Q, K \rangle = \langle h W^Q, W^{UK} c_t^{KV} \rangle = \langle h W^Q W^{UK}, c_t^{KV} \rangle$
  - 引入 RoPE 后：$\langle Q R_q, R_k K \rangle = \langle h W^Q R_q, R_k W^{UK} c_t^{KV} \rangle$ 导致两边位置矩阵不匹配。
  - **解决方案**：预留少量不进行潜变量投影的 Key 维度用于直接旋转计算。

---

## 第 59 页 (Page 59)

### 构建 DeepSeek MoE v3 还需要什么？（续）
#### MTP：多 Token 预测机制 (Multi-Token Prediction)
- 采用一些轻量级的小模型来同时预测未来的多个步骤。
- 比较架构：DeepSeek V3 相比于 EAGLE 架构（但 DeepSeek V3 目前仅进行向前 1 个 Token 的预测）。
- 公式：
  $$\mathbf{h}_i^{\prime k} = M_k[\operatorname{RMSNorm}(\mathbf{h}_i^{k-1}); \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]$$
  $$\mathbf{h}_{1:T-k}^k = \operatorname{TRM}_k(\mathbf{h}_{1:T-k}^{\prime k})$$
  $$p_{i+k+1}^k = \operatorname{OutHead}(\mathbf{h}_i^k)$$

---

## 第 60 页 (Page 60)

### MoE 总结 (MoE summary)
- ❖ MoE 能够有效利用**稀疏性**——并不是所有的输入都需要激活模型全部的参数。
- ❖ 离散的路由选择极其困难，但 Top-K 启发式机制在实践中非常有效。
- ❖ 现阶段有大量的实验数据表明，MoE 机制高度可行，且拥有极高的高性价比。

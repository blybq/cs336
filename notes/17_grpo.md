# CS336 课程笔记 17：深度解析 GRPO 机制与代码实现

本节是强化学习（RL）在语言模型中应用的技术深挖课。我们从马尔可夫决策过程（MDP）的建模出发，推导策略梯度定理与基线校正的方差控制原理，对比经典 PPO 的显存瓶颈，深入剖析 **GRPO（Group Relative Policy Optimization）** 算法的每一步数学设计，并探讨其在大模型（如 DeepSeek-R1-Zero）中的实战应用与代码实现。

---

## 一、 语言模型中的强化学习建模

在大语言模型结果奖励（Outcome-based Reward）的设定中，强化学习的要素定义如下：
* **状态 $s$**：当前的输入提示（Prompt） $q$ 以及模型目前已经生成的 Token 序列：$s_t = [q, y_{<t}]$。
* **动作 $a$**：模型在当前状态下生成的下一个 Token：$a_t = y_t \in \mathcal{V}$。
* **状态转移动态**：确定性转移。将动作直接拼接到状态后作为下一阶段的状态：$s_{t+1} = [s_t, a_t]$。没有机器人控制领域常见的物理环境噪声。
* **奖励 $r(s, a)$**：响应质量的评估函数。通常是**结果验证奖励（Verifiable Reward）**，如判定数学题最终答案是否正确，或运行测试用例检验代码是否通过。

---

## 二、 策略梯度定理与基线校正

### 1. 为什么需要策略梯度（Policy Gradient）？
在深度学习中，普通的反向传播（Backpropagation）要求从损失函数到模型输出之间的整个计算路径都是连续可导的。
然而，在大模型强化学习中，有两个致命的不可导环节：
1. **动作采样的离散性**：模型通过从概率分布 $\pi_\theta(y_t \mid s_t)$ 中自回归采样出一个个离散的 Token $y_t$。采样操作（如 multinomial sampling）是断裂、不可导的。
2. **奖励函数的非微分性**：奖励函数 $r(q, y)$（如运行代码的编译器、检查最终答案的判题脚本）是一个离散的黑盒，我们根本无法求取 $r(q, y)$ 关于模型参数 $\theta$ 的导数。

**策略梯度（Policy Gradient）** 的核心作用就在于：它通过数学恒等变形，将对“期望奖励”的梯度求导，转换成了对“策略对数概率 $\log \pi_\theta(y|q)$”的求导。这样，我们**完全不需要对离散采样过程和黑盒奖励函数求导**，只需对模型概率这部分连续可导的路径进行常规反向传播即可。

### 2. 策略梯度公式推导
设 $y$ 为生成的完整响应序列（包含所有动作），$q$ 为输入提示，奖励为 $r(q, y)$。模型的目标是最大化期望奖励：
$$\text{obj}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, y \sim \pi_\theta(\cdot|q)} [r(q, y)] = \sum_{q, y} P(q) \pi_\theta(y|q) r(q, y)$$
对参数 $\theta$ 求梯度，并利用对数微分恒等式 $\nabla_\theta f(\theta) = f(\theta) \nabla_\theta \log f(\theta)$：
$$\nabla_\theta \text{obj}(\theta) = \sum_{q, y} P(q) \left( \nabla_\theta \pi_\theta(y|q) \right) r(q, y)$$
$$\nabla_\theta \text{obj}(\theta) = \sum_{q, y} P(q) \pi_\theta(y|q) \nabla_\theta \log \pi_\theta(y|q) r(q, y) = \mathbb{E}_{q \sim \mathcal{D}, y \sim \pi_\theta(\cdot|q)} \left[ r(q, y) \nabla_\theta \log \pi_\theta(y|q) \right]$$

### 3. 基线校正 (Baseline Correction)
朴素策略梯度的方差极大。为了减小方差，我们可以减去任何**只与状态相关、与动作无关的基线函数 $b(q)$**，这不会改变梯度的期望（无偏性）：
$$\nabla_\theta \text{obj}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, y \sim \pi_\theta(\cdot|q)} \left[ (r(q, y) - b(q)) \nabla_\theta \log \pi_\theta(y|q) \right]$$
理想的基线 $b(q)$ 是状态的值函数 $V(q) = \mathbb{E}_{y \sim \pi} [r(q, y)]$，二者相减的差值 $r(q, y) - V(q)$ 即为**优势值（Advantage）**，代表生成回答优于平均水平的程度。

---

## 三、 经典 Actor-Critic PPO 及其显存痛点

### 1. PPO 的双网络架构
在经典的 PPO 算法中，优势估计是通过 Actor-Critic 架构实现的：
* **Actor 模型**：参数为 $\theta$，即当前的策略网络 $\pi_\theta(y|q)$，负责输出 Token 概率并采样动作。
* **Critic 模型**：参数为 $\phi$，即价值网络 $V_\phi(s)$，负责拟合每个状态下的期望总回报，用作基线 $b(q)$。

### 2. Critic 为什么是显存怪兽？
在训练过程中，价值网络（Critic）需要精确估计长序列中每个步骤的价值，这要求 Critic 必须具备与 Actor 相当的参数量。
* **参数与优化器翻倍**：训练时，GPU 显存中必须同时常驻 Actor 和 Critic 两个超大模型（通常是数百亿参数），且两者各自需要维护一套庞大的 Adam 优化器状态。
* **激活值堆积**：由于两个模型都需要在反向传播中计算梯度，前向传播中产生的所有中间隐藏层激活值都会常驻显存，使得本来就很紧张的 GPU 显存面临极大的 OOM 风险，严重制约了训练时的最大序列长度与 Batch Size。

---

## 四、 GRPO 核心工作流与数学详解

**GRPO** 彻底颠覆了 Actor-Critic 架构。它的核心思想是：**放弃 Critic 模型，转而通过在当前策略下为同一个提示采样一组响应，并利用组内的相对表现来自适应地估计优势值。**

![DeepSeek GRPO 组内优势相对奖励反馈环流程](images/grpo_flowchart.drawio.png)

### 1. 组内相对优势（Group Relative Advantage）计算
对于给定的提示 $q$，我们使用当前的策略网络采样生成 $G$ 个不同的回答：$\{y_1, y_2, \dots, y_G\}$。使用规则或模型对其进行判分，得到一组奖励值 $\{r_1, r_2, \dots, r_G\}$。
第 $i$ 个响应的优势值 $A_i$ 计算为：
$$A_i = \frac{r_i - \text{Mean}(r)}{\text{Std}(r) + \epsilon}$$

#### 🌟 基线减法与奖励为零时的参数更新惩罚机制
* **课堂互动问答**：如果我们将错误回答的奖励设为 0，那么模型在计算梯度时会不会因为乘积为 0 而无法更新、无法远离错误路径？
* **核心点拨**：不会。在策略梯度和 GRPO 中，优势函数 $A_i$ 采用了**基线对比（Baseline Subtraction）**机制。在 GRPO 中，组内优势为 $A_i = \frac{r_i - \text{Mean}(r)}{\text{Std}(r) + \epsilon}$。如果这组样本的平均奖励（Mean）是 0.5，而当前错误样本得到了 0 分，那么它的优势值 $A_i$ 将是负值。在梯度下降中，负的优势值会驱动概率权重反向更新，将模型参数**强力推离（push away）**这个低分行为，使其概率下降。这就是基线减法机制带来的数学优雅性。

#### 为什么这能省去 Critic 显存？
* **动态经验基线**：组内奖励均值 $\text{Mean}(r) = \frac{1}{G}\sum r_i$ 在数学上就是该状态下当前策略平均预期奖励的无偏经验估计，**完美起到了 Critic 值函数 $V(q)$ 的基线校正作用**。
* **自适应缩放**：标准差 $\text{Std}(r)$ 自动对优势进行了尺度缩放，消除了奖励尺度的敏感性。
* **对比结果**：仅通过组内对比，我们在计算优势时**不需要任何独立的 Critic 价值网络**。这直接消除了 Critic 模型及其优化器、激活值占用的全部 GPU 显存（节省约 50% 显存），极大地简化了系统设计，允许在单张 GPU 上训练超长序列的思维链。

---

### 2. GRPO 损失函数符号详解
GRPO 的损失函数定义为：
$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(y_i|q)}{\pi_{\text{old}}(y_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(y_i|q)}{\pi_{\text{old}}(y_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

#### 每个符号的物理含义：
* **$\theta$**：当前正在被优化和更新的策略网络的参数。
* **$G$**：组大小（Group Size）。即同一提示采样的样本数（通常为 $8 \sim 64$）。
* **$y_i$**：当前组内第 $i$ 个采样的响应序列（包含思维标记与最终答案）。
* **$q$**：输入的 Prompt。
* **$\pi_\theta(y_i|q)$**：当前策略网络输出响应 $y_i$ 的概率。
* **$\pi_{\text{old}}(y_i|q)$**：在本次策略迭代优化开始前，旧策略网络生成该响应的概率。
* **$\frac{\pi_\theta(y_i|q)}{\pi_{\text{old}}(y_i|q)}$**：重要性采样比率（Ratio）。表示在参数更新后，产生该响应的概率发生的相对变化。
* **$A_i$**：第 $i$ 个响应的组内相对优势值。
* **$\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$**：裁剪操作。限制 Ratio 的范围在 $[1-\epsilon, 1+\epsilon]$（例如 $\epsilon = 0.2$），防止新策略概率偏离旧策略过远，保证算法收敛。
* **$\beta$**：KL 散度正则化系数。
* **$\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$**：当前策略 $\pi_\theta$ 与参考策略 $\pi_{\text{ref}}$ 之间的 KL 散度，防止策略漂移和模式坍塌。其在代码中常采用低方差的单样本估计形式：
  $$\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \approx \frac{\pi_{\text{ref}}(y_i|q)}{\pi_\theta(y_i|q)} - \log \frac{\pi_{\text{ref}}(y_i|q)}{\pi_\theta(y_i|q)} - 1$$

---

### 3. 重要性采样比与 Stop Gradient 细节
在实际代码编写中，同一批采样轨迹会在内循环中执行多次（例如 10 次）小步梯度更新。
* **`.detach()` 的关键作用**：分子 $\pi_\theta$ 随参数更新而改变，但分母 $\pi_{\text{old}}$ 对应的概率张量必须在计算图中**断开梯度（Stop Gradient）**：
  ```python
  # PyTorch 伪代码
  ratio = torch.exp(log_prob_theta - log_prob_old.detach())
  ```
  如果忘记对分母执行 `.detach()`，分母将依然参与梯度回传，在内循环的第一步更新后，分子分母的梯度会直接抵消，导致内循环后续更新失效。

### 4. DeepSeek-R1-Zero 中的应用
DeepSeek-R1-Zero 展现出了在纯强化学习（无需冷启动 SFT）下大模型自发涌现推理能力的奇迹。
* **为什么用 GRPO**：由于推理链（思维链 CoT）长度不可控（经常达到数千甚至上万 Token），常规 Actor-Critic PPO 在此上下文下的显存开销完全不可接受。GRPO 是支持 R1-Zero 极长上下文强化学习的工程基石。
* **准确度奖励 (Accuracy Reward)**：检查最终答案是否完全符合数学规范或通过测试用例，提供硬性逻辑约束。
* **格式奖励 (Format Reward)**：强制模型将思考过程写在 `<think>` 和 `</think>` 中，最终答案写在 `<answer>` 和 `</answer>` 中。格式奖励非常微弱，但正是这一格式约束，成功拉开了模型结构化思维涌现的序幕。

---

## 五、 玩具排序任务（Sorting Task）代码设计权衡

我们在代码实践中，针对一个“对 $N$ 个数字进行升序排列”的玩具任务设计强化学习环境。

### 1. 稀疏奖励 vs. 稠密奖励
* **稀疏二元奖励（Sparse Binary Reward）**：排序完全正确给 1 分，否则 0 分。
  * *缺点*：如果 $N \ge 4$ 且模型初始为随机权重，采样到完全正确序列的概率是 $\frac{1}{N!}$，奖励极度稀疏，模型极易因没有正反馈梯度而陷入死锁。
* **稠密部分得分（Dense Reward）**：
  * *方案 A（位置对齐数）*：统计与标靶位置完全吻合的数字个数。
  * *方案 B（相邻有序对数量）*：若数字包含在输入中给分，且每对相邻数字呈递增趋势给分。
  * *利用漏洞（Reward Hacking）*：方案 B 存在漏洞，模型可以通过重复输出相同数字（如 `[2, 2, 2, 2]`）来低成本刷满相邻对分数。因此设计奖励时，必须加入**多样性/唯一性约束**。

### 2. 累积 KL 与分步 KL 散度的实现
为了让新策略在更新中不偏离预训练参考模型 $\pi_{\text{ref}}$，我们计算每个 Token 的相对熵。
```python
# 低方差 KL 散度估计器实现
ratio_ref = torch.exp(log_prob_ref - log_prob_theta)
kl_loss = ratio_ref - torch.log(ratio_ref) - 1.0
total_loss = policy_loss + beta * kl_loss.mean()
```
该 KL 估计器在期望意义上等价于标准的 KL 散度，但在蒙特卡洛采样下方差显著更小。

---

## 六、 推理阶段与系统架构工程瓶颈

虽然 GRPO 消除了 Critic 模型，将显存开销降低了近 50%，但在工业界大规模生产部署中，强化学习的训练效率仍然受到**推理与训练动态切换**的严重制约。

1. **Inference vs. Training 的计算特点冲突**：
   * **Inference 阶段（Rollout 采样）**：自回归生成，具有高内存带宽受限（KV Cache 管理、小 Batch、单 Token 串行读取）的特征。
   * **Training 阶段（反向传播）**：大 Batch、算术吞吐受限（Compute-bound），需要最大化 GPU 核心张量运算利用率。
2. **两阶段架构解耦（Inference/Training Worker Split）**：
   前沿团队使用独立的 GPU 集群运行 vLLM（专门负责生成 Rollouts 并将 KV Cache 压缩到极致），然后将生成的 Token 序列与对数概率通过网络（或 Shared Memory）发送给负责 PyTorch 梯度的 Training Workers。
3. **KV Cache 与 Padding 瓶颈**：
   推理大模型的思维链（COT）长度波动极大（有的问题思考 200 Token，有的思考 8000 Token）。在组内并行计算优势时，不同序列长度不均会导致大量的 Padding 浪费。因此，系统必须实现类似 **FlashAttention / Variable-Length Packing** 的机制，动态拼接非均匀批次，以维持高效运行。


## 七、 课堂互动 Q&A 录

* **Q：在实现作业时，我们需要手写反向传播求导，还是可以直接使用 PyTorch 的 autograd？**
  * **A**：课件中展现的矩阵计算维度分解（如计算前向 $2MNK$ FLOPs）完全是为了做系统层面的算力开销估算。在作业中，直接调用 PyTorch 的 `loss.backward()` 即可，不需要手动计算复杂的导数图。
* **Q：在策略梯度算法中，我们能使用一个冻结的初始模型作为基线函数 $b(s)$ 吗？**
  * **A**：可以。但在使用冻结的参考模型计算基线输出时，必须确保其在前向传播中被移出计算图（即调用 `.detach()`），不允许让基线模型搜集梯度，否则会导致双模型的联合求导使训练过程迅速崩溃。
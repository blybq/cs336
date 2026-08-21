# CS336 课程笔记 16：详解大模型强化学习（RL）算法

在对齐（Alignment）与后训练（Post-training）阶段，传统的偏好微调（如 RLHF / DPO）面临**过度优化（Overoptimization / Goodhart's Law）**和**概率校准度降低（Overconfidence）**的瓶颈。这促使行业转向**基于规则或可验证奖励（Verifiable Rewards）的强化学习**。本节详细拆解大模型强化学习从经典 PPO 到 GRPO 的演进，并对比分析 DeepSeek-R1、Kimi k1.5 和 Qwen 等顶尖推理大模型的设计配方。

---

## 一、 经典强化学习算法回顾与痛点

### 1. 策略梯度与反向传播的权衡 (Why Policy Gradient?)

#### ① 为什么我们不能使用普通的反向传播（Backpropagation）直接优化奖励？
在监督微调（SFT）中，模型优化的目标是让生成的词概率逼近固定的目标 Token（即 Label）。在从 logits 到交叉熵损失函数的计算过程中，每一个操作都是连续且可导的，因此梯度可以顺畅地从损失函数反向传播回模型参数。
然而，在强化学习（RL）场景下：
1. **生成过程是离散的**：模型通过在每个 Token 步上进行离散采样（如 multinomial 采样或 argmax 选择）来生成响应序列 $y$。这种离散采样操作是**不可导**的（无法计算梯度）。
2. **奖励函数通常是黑盒或不可导的**：奖励函数 $r(x, y)$ 可以是运行 Python 代码的编译器、验证数学公式的正则表达式、或者人类的判决。这些奖励函数通常是离散的选择（如对/错，即 1/0），根本无法写出关于模型参数 $\theta$ 的导数形式。
因此，我们无法通过标准的反向传播算法，将奖励的误差直接传回网络。

#### ② 策略梯度定理（Policy Gradient Theorem）
策略梯度（以 REINFORCE 算法为例）通过数学转换绕过了这一不可导屏障。
设响应序列 $y$ 在策略 $\pi_\theta(y|x)$ 下生成的概率为 $P(y|x) = \pi_\theta(y|x)$，奖励为 $r(x, y)$。我们希望最大化期望奖励：
$$\text{obj}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [r(x, y)]$$
我们对其求梯度：
$$\nabla_\theta \text{obj}(\theta) = \nabla_\theta \sum_{y} \pi_\theta(y|x) r(x, y) = \sum_{y} \nabla_\theta \pi_\theta(y|x) r(x, y)$$
利用对数微分恒等式（Log-derivative Trick）$\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$，上式可重写为：
$$\nabla_\theta \text{obj}(\theta) = \sum_{y} \pi_\theta(y|x) \nabla_\theta \log \pi_\theta(y|x) r(x, y) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \nabla_\theta \log \pi_\theta(y|x) \right]$$
* **结论**：虽然奖励函数 $r(x, y)$ 不可导，但策略概率的对数项 $\log \pi_\theta(y|x)$ 关于参数 $\theta$ 是**完全可导**的。我们只需要让模型在当前策略下采样生成响应，根据得到的标量奖励值 $r(x, y)$，来加权调整对应生成序列的对数似然度即可。这便是策略梯度的精妙所在。

为了降低策略梯度的方差，通常引入一个与当前动作无关的基线值 $b$ 进行校正：
$$\nabla_\theta \text{obj}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ (r(x, y) - b) \nabla_\theta \log \pi_\theta(y|x) \right]$$

---

### 2. 经典 PPO 算法与 Critic 模型的开销
PPO (Proximal Policy Optimization) 允许利用旧策略 $\pi_{\text{old}}$ 采样的轨迹进行多次小步梯度更新，通过重要性采样与裁剪（Clip）机制维持新旧策略的接近度。

#### ① Actor 与 Critic 模型
在经典的 Actor-Critic 框架中，训练涉及两个核心神经网络：
* **Actor 模型（策略网络）**：参数为 $\theta$，负责接收提示 $x$ 并生成动作（自回归产生文本 $y$）。
* **Critic 模型（价值网络）**：参数为 $\phi$，负责接收当前状态 $s$（如前文和当前生成的 Token 历史），并预测从该状态出发能获得的长期累积奖励期望值 $V_\phi(s)$。

#### ② 为什么 Critic 模型是极度消耗显存的（Memory-heavy）？
在标准 PPO 中，为了获得准确的基线值 $b(s)$ 来计算优势估计（Advantage） $\hat{A}_t = G_t - V_\phi(s_t)$，Critic 模型通常必须和 Actor 模型具有**相似的参数规模和模型结构**（否则小模型无法理解长文本中的复杂逻辑，会导致价值估计出现巨大方差）。
1. **显存直接翻倍**：我们在训练时，必须在 GPU 显存中同时加载 Actor 和 Critic 两套巨大的模型，并为它们各自维护优化器状态（如 Adam 的一阶、二阶动量）。
2. **激活值开销巨大**：在反向传播计算梯度时，Actor 和 Critic 在前向传播过程中产生的所有中间激活值（Activation）必须同时常驻显存，极易触发 Out of Memory (OOM)。
3. **系统复杂度高**：由于两套网络输入相同但输出维度不同（Actor 输出 Vocabulary 概率分布，Critic 输出单个标量得分），在并行化训练（如 3D Parallelism）时，两套网络的切分与通信同步极其繁琐。

---

## 二、 GRPO（Group Relative Policy Optimization）算法详解

为了彻底消除 PPO 中极其沉重的 Critic 网络，DeepSeek 团队提出了 **GRPO**，通过组内相对比较来自适应地估计优势值。

### 1. GRPO 的组内相对优势估计机制
对于给定的输入提示（Prompt） $q$，GRPO 不再依赖一个独立的 Critic 网络来预测绝对期望价值，而是直接利用当前的策略网络 $\pi_\theta$ 为 $q$ 采样生成一组响应（共 $G$ 个）：
$$\{y_1, y_2, \dots, y_G\}$$
每个响应 $y_i$ 经过外部奖励函数（例如数学判定规则或编译器测试）得到一个标量得分 $r_i$。
第 $i$ 个响应的相对优势值 $A_i$ 直接通过组内的得分均值和标准差进行归一化计算：
$$A_i = \frac{r_i - \text{Mean}(\{r_1, \dots, r_G\})}{\text{Std}(\{r_1, \dots, r_G\}) + \epsilon}$$
* **自适应基线的精妙性**：这组样本的均值 $\text{Mean}(r)$ 自动且动态地充当了当前状态的“值函数 $V(s)$”经验估计。
  * 如果题目很简单，所有响应的分数都很高，只有获得满分的响应才能得到正的相对优势 $A_i > 0$；
  * 如果题目极难，所有响应得分都很低，即使只做对了一小步的稍微好一点的响应，也能因高于平均值而获得正向激励。
* **显存与计算节省**：由于完全不需要 Critic 网络，GRPO 直接释放了近 50% 的 GPU 显存开销，消除了 Critic 的反向传播与同步瓶颈，使得训练更长上下文的思维链（CoT）大为可行。

---

### 2. GRPO 损失函数与符号拆解
GRPO 的完整单步优化损失函数如下：
$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(y_i|q)}{\pi_{\text{old}}(y_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(y_i|q)}{\pi_{\text{old}}(y_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

#### 符号物理含义详细拆解：
* **$\theta$**：待优化的当前策略网络参数。
* **$G$**：组大小（Group Size），即对同一个提示采样的响应个数（通常取 $G = 8 \sim 64$）。
* **$q$**：输入给模型的 Prompt。
* **$y_i$**：组内第 $i$ 个采样的响应序列（包含思维链和最终答案）。
* **$\pi_\theta(y_i|q)$**：当前策略网络参数 $\theta$ 生成该响应序列的概率（通常为序列中所有 Token 生成概率的乘积）。
* **$\pi_{\text{old}}(y_i|q)$**：在本次训练迭代开始前，旧策略网络生成该响应序列的概率。
* **$\frac{\pi_\theta(y_i|q)}{\pi_{\text{old}}(y_i|q)}$**：重要性采样比率（Ratio）。衡量由于参数更新导致该动作生成概率的变化程度。
* **$A_i$**：第 $i$ 个响应的组内相对优势值。
* **$\text{clip}(\text{Ratio}, 1-\epsilon, 1+\epsilon)$**：裁剪函数。如果 Ratio 偏离 $[1-\epsilon, 1+\epsilon]$ 区间（例如 $\epsilon = 0.2$），则强行截断，防止由于重要性采样失效导致策略发生剧烈且有害的更新。
* **$\beta$**：KL 散度惩罚系数，控制模型偏离初始状态的程度。
* **$\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$**：当前策略 $\pi_\theta$ 与冷启动参考模型 $\pi_{\text{ref}}$ 之间的 KL 散度，防止模型在强化学习的探索中发生模式坍塌。

---

## 三、 DeepSeek-R1-Zero 的强化实战与奖励设计

### 1. 为什么 DeepSeek-R1-Zero 选择 GRPO？
DeepSeek-R1-Zero 探索的是在**没有任何人类标注 SFT 数据冷启动**的情况下，仅通过纯强化学习（RL）能否让模型自主涌现出强大的逻辑推理和自我纠错能力。
由于模型需要生成极其冗长、结构复杂的思维链（Thinking Trajectory，有时长达数千甚至上万 Token），在极长上下文下：
1. PPO 的 Critic 网络将占用极其恐怖的显存，导致 Batch Size 严重受限。
2. GRPO 的无 Critic 架构极好地解决了这一痛点，支持大规模的组并发采样（如 $G=64$），使得模型在超长推理链的训练中依然保持极高的吞吐率。

### 2. 规则驱动的双重奖励函数设计

为了让 R1-Zero 在没有人类示范的前提下学会推理，DeepSeek 设计了两个纯规则的奖励函数，避免了神经网络奖励模型被 Hack 的风险：

#### ① 准确度奖励 (Accuracy Rewards)
* **机制**：对生成的回答进行硬性的逻辑判定。
* **例**：对于数学题，提取模型输出的最终数值，直接与标准答案匹配；对于代码题，把模型生成的代码直接扔进编译器运行，看是否能通过所有的 Test Cases。对则给 1 分，错则给 0 分。

#### ② 格式奖励 (Format Rewards)
* **为什么需要格式奖励**：在纯强化早期，模型可能会生成一片混乱的行文。我们需要引导模型把思考过程和最终答案进行结构化拆分。
* **机制**：如果模型能够严格把它的“思维链推理过程”包裹在 `<think>` 和 `</think>` 标签内，并且把最终的“干净答案”包裹在 `<answer>` 和 `</answer>` 标签内，就给予正向的格式得分（例如 0.1 分）。
* **结果**：通过这一简单的格式奖励，R1-Zero 成功涌现出了长序列思维链（CoT）推理能力，并自发学会了在 `<think>` 标签内进行自我纠错、重新审题和逻辑回溯。

---

## 四、 主流推理大模型（Reasoning LLMs）设计配方对比

| 设计维度 | DeepSeek-R1 (R1-Zero / R1) | Kimi k1.5 (Moonshot) | Qwen 2.5-Math (RL) |
| :--- | :--- | :--- | :--- |
| **基础算法** | **GRPO** (去 Critic 架构，极大节省显存) | **类 DPO 闭式方程平方损失** + 组内基线校正 | **GRPO** (去 Critic 架构，高吞吐并行) |
| **冷启动策略** | **R1-Zero**: 0 样本，从 Base 模型直接强化。<br>**R1**: 引入少量精心过滤的 CoT SFT 数据进行冷启动，绕过 R1-Zero 早期不可读的混乱阶段。 | **SFT 困难样本筛选**：只选取在 SFT 模型下多次生成失败但至少成功一次的适中难度问题进行强化。 | **小样本激活动作**：使用几千条高质量可验证推理案例进行强化，即取得巨大推理涨幅。 |
| **奖励函数设计** | **准确度奖励** (正确与否硬判定) + **格式奖励** (限制在 `<think>` 标签内) + **语言一致性奖励** (防止中英多语言混杂)。 | **准确度奖励** + **长度压缩奖励**（通过 $\lambda$ 因子动态调整批次内长度区间，惩罚无意义的冗长）。 | **准确度奖励** + **思考标签规范奖励**。 |
| **系统架构细节** | * 证明在线 MCTS 与 PRM（过程奖励模型）在大模型通用强化学习中表现不及端到端结果奖励（Outcome-based Reward）。<br>* 发现蒸馏超大模型（R1）的 CoT 数据可以使小模型直接获得巨幅数学提升。 | * **系统解耦**：将 vLLM 推理服务器与 RL 梯度工作站解耦，通过 NCCL 消息传递模型权重。<br>* **显存垃圾回收**：每次迭代彻底销毁并重新初始化 vLLM 以应对 KV Cache 严重不均。 | * 发现**通用 RL 对推理能力的负迁移**：优化通用指令跟随的强化学习步可能会导致数学/STEM 性能轻微退化，必须小心平衡。 |

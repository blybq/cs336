# CS336 课程笔记 15：监督微调（SFT）与人类反馈强化学习（RLHF）

大语言模型的开发分为预训练（构建通用常识、代码和推理基底能力）和后训练（Steering/后对齐，使模型符合人类交互规范）。本节详细分析了监督微调（SFT）的数据构建范式、潜在失败模式，以及人类反馈强化学习（RLHF）的数学原理，包括经典的 Bradley-Terry 模型、PPO 目标函数和 DPO（直接偏好优化）的数学推导。

---

## 一、 监督微调 (Supervised Fine-Tuning - SFT)

SFT 的核心是通过最大似然估计（MLE）模仿专家的演示数据，将自回归预测的能力定向到“指令-回复”的对话格局中。

### 1. 三种 SFT 数据构建范式
1. **基准任务聚合（如 FLAN）**：将传统 NLP 学术数据集（如问答、分类、摘要）利用模板改写为“指令-回复”对。
   * *缺点*：文风极其机械、刻板，与用户日常对聊天机器人的真实诉求差距较大。
2. **AI 合成数据（如 Stanford Alpaca）**：使用 **Self-Instruct** 技术，设计少量种子模板提示 GPT-3.5/4 批量生成指令和对应的回答。
   * *优点*：快速获得大量对话流，成本极低。
   * *缺点*：如果大模型盲目模仿超越自身底层容量的知识表达，容易退化出“徒有虚表”的风格（满嘴漂亮套话但事实性一塌糊涂）。
3. **人工专家编写（如 Open Assistant）**：雇佣专业外包或社区志愿者手工编写高质量对话。
   * *优点*：信息密度高、含有真实的参考引用、代码规范度高。
   * *缺点*：极其昂贵，标注员时间压力大，极难规模化。

### 2. SFT 阶段的关键陷阱：John Schulman 幻觉假说

在 SFT 阶段，有两个极其显著的隐式陷阱：
* **长度偏见 (Length Bias)**：无论是人类评估还是 AI 评判（LLM-as-a-judge），都更倾向于给篇幅冗长、排版精美、带有多级列表的回答打高分。这导致 SFT 模型容易被诱导为“废话连篇”却缺乏实质内容的状态。
* **幻觉的隐式诱导（John Schulman 的假说）**：
  OpenAI 的联合创始人 John Schulman 提出过一个重要假说，阐述了 SFT 是如何强迫模型产生幻觉的。
  在大规模预训练中，模型只见过一部分事实性知识（即它“知道”的边界）。但在 SFT 阶段，微调数据中往往包含大量深度事实性问题（例如要求给出冷门的物理常数、复杂的代码库用法等）。
  如果 SFT 的目标标签强行写了这些模型在预训练时没有掌握的知识，在 MLE 损失函数的驱使下，**模型会因为“没有完美背诵出这个它不知道的事实”而被惩罚（梯度更新）**。
  为了在损失函数上拿到低分，模型被迫学会了一种投机策略：**不管自己底层是否知道这些知识，都必须用绝对自信的语气，在规定的格式位置编造出看似合理的词汇组合（即幻觉）**。 SFT 实际上是在“训练”模型去撒谎和谄媚。

#### 解决方案
1. 确保 SFT 数据的知识范围与模型预训练时掌握的知识库相适配。
2. 在 SFT 数据中加入诚实的否定回答，例如“我不知道”、“我无法查证此信息”，让模型学会表达不确定性，而不是强行编造。

### 3. 预训练与 SFT 的融合：衰减阶段混合 (Decay Stage Mixing)
在传统的二阶段训练中，SFT 独立于预训练进行。但最新工业界实践（如 MiniCPM）发现，如果在预训练后期（如学习率退火/衰减阶段）直接将高质量的 SFT 数据混合到预训练语料中共同训练，能显著改善对齐质量，防止灾难性遗忘，并让模型展现出更稳健的对齐表现。

---

## 二、 人类反馈强化学习 (RLHF)

### 1. 范式转变：从模仿到最大化期望奖励
SFT 的本质是“模仿”，如果专家数据本身有局限，模型的天花板就会被锁死。而在强化学习（RL）框架下，我们将模型响应看作是决策序列，其目标是寻找一个策略 $\pi(y|x)$，使得奖励模型（Reward Model）的期望得分最大化：
$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} [r(x, y)]$$
* *核心动机*：人类去评判两个回答哪个更好（成对偏好判定，Pairwise Preference），其难度和成本远低于去写出一份完美的回答。RLHF 能够利用这种相对偏好，不断将模型的生成拉向期望的方向。

### 2. Bradley-Terry (BT) 人类偏好模型

#### ① 数学公式定义
为了构建评估回答好坏的奖励模型（Reward Model），我们假设人类的成对偏好服从 Bradley-Terry 模型。
给定提示 $x$，模型生成了两个不同的响应 $y_w$（优选，Winning）和 $y_l$（次选，Losing）。人类选择 $y_w$ 优于 $y_l$ 的概率建模为：
$$P(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) = \frac{1}{1 + e^{-\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)}}$$
其中 $r_\phi(x, y)$ 是我们训练的标量奖励模型，而 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 Sigmoid 函数。

#### ② Sigmoid 函数的作用与人类决策建模
* **概率映射与平滑性**：Sigmoid 函数将两个响应奖励值的差值 $z = r_\phi(x, y_w) - r_\phi(x, y_l)$ 映射到区间 $(0, 1)$。
  * 当 $r_\phi(x, y_w) \gg r_\phi(x, y_l)$ 时，$z \to +\infty$，概率接近 1。说明优劣分明，人类必定选择 $y_w$。
  * 当 $r_\phi(x, y_w) \approx r_\phi(x, y_l)$ 时，$z \to 0$，概率接近 0.5。说明两者水平相当，人类的偏好选择类似于随机抛硬币。
* **人类偏好的非绝对性**：在现实中，人类做出选择往往不是绝对理性的。两个响应即使有微弱的质量差异，不同的人由于关注点不同（如排版、语气等），也可能投出相反的票。BT 模型借助 Sigmoid 的平滑过渡，完美拟合了这种包含噪声和随机性的人类决策特性，使奖励差值大小与人类选择的确定性直接挂钩。

#### ③ 奖励模型损失函数
通过最小化负对数似然（交叉熵损失），优化奖励模型参数 $\phi$：
$$\mathcal{L}_R(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]$$

### 3. 经典的 PPO 强化学习目标函数
InstructGPT 使用 Proximal Policy Optimization (PPO) 算法，联合优化策略网络 $\pi_\theta$，其目标函数为：
$$\text{obj}(\theta) = \mathbb{E}_{(x, y) \sim D_{\pi_\theta}} \left[ r_\phi(x, y) \right] - \beta \mathbb{D}_{\text{KL}}\left(\pi_\theta(y|x) \parallel \pi_{\text{SFT}}(y|x)\right) + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \pi_\theta(x) \right]$$

* **物理含义**：
  1. **第一项（期望奖励）**：引导模型输出高奖励（更讨喜、更准确）的内容。
  2. **第二项（KL 散度约束）**：惩罚偏离 SFT 基线的行为，防止发生 **Reward Hacking**（模型利用奖励模型的漏洞刷高分，例如写出一堆无逻辑但奖励模型极喜欢的敏感词汇）。
  3. **第三项（预训练语言模型损失）**：在 RL 微调中掺入预训练损失，确保模型对齐的同时，不退化基本的语言能力和常识。

---

## 三、 DPO (Direct Preference Optimization) 数学推导

经典的 PPO 框架需要同时在显存中放置 4 个超大模型：Actor（待优化的策略 $\pi_\theta$）、Critic（价值网络 $V$）、Reference（冻结的 SFT 基准 $\pi_{\text{ref}}$）和 Reward（奖励模型 $r_\phi$）。其在线 Rollout 采样计算不稳定，超参数极难调节。
DPO 提出了一种巧妙的闭式解代换方法，**直接利用策略概率的比值代替显式奖励模型**，省去了 Critic 和 Reward 模型。

### 1. 步骤一：求解 KL 约束下的最优策略
忽略预训练损失项，对于任意给定的提示 $x$，RLHF 的数学优化目标可以表示为：
$$\max_{\pi} \mathbb{E}_{y \sim \pi(y|x)} [r(x, y)] - \beta \mathbb{D}_{\text{KL}}(\pi(y|x) \parallel \pi_{\text{ref}}(y|x))$$
将期望和 KL 散度展开：
$$\max_{\pi} \sum_{y} \pi(y|x) r(x, y) - \beta \sum_{y} \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$$
将常数因子 $\beta$ 提取，并合并对数项：
$$\Rightarrow \max_{\pi} \beta \sum_{y} \pi(y|x) \left[ \frac{r(x, y)}{\beta} - \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$
$$\Rightarrow \max_{\pi} \beta \sum_{y} \pi(y|x) \left[ \log \left( \exp\left(\frac{r(x, y)}{\beta}\right) \right) - \log \pi(y|x) + \log \pi_{\text{ref}}(y|x) \right]$$
$$\Rightarrow \max_{\pi} \beta \sum_{y} \pi(y|x) \left[ \log \left( \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right) \right) - \log \pi(y|x) \right]$$
由于括号内第一项求和不一定为 1，为了构建一个合法的概率分布，我们定义一个归一化配分函数（Partition Function） $Z(x)$：
$$Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$
利用 $Z(x)$ 构造最优策略分布 $\pi^*(y|x)$：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$
代回优化式：
$$\Rightarrow \max_{\pi} \beta \sum_{y} \pi(y|x) \left[ \log \left( Z(x) \pi^*(y|x) \right) - \log \pi(y|x) \right]$$
$$\Rightarrow \max_{\pi} \beta \sum_{y} \pi(y|x) \left[ \log \pi^*(y|x) - \log \pi(y|x) + \log Z(x) \right]$$
由于 $\sum_y \pi(y|x) = 1$，且 $Z(x)$ 与待优化的 $\pi$ 无关，我们可以将 $\log Z(x)$ 提到求和符号外面：
$$\Rightarrow \max_{\pi} \beta \left( - \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi^*(y|x)} + \log Z(x) \right)$$
$$\Rightarrow \max_{\pi} \beta \left( - \mathbb{D}_{\text{KL}}(\pi(y|x) \parallel \pi^*(y|x)) + \log Z(x) \right)$$
因为 KL 散度在两分布完全一致时取得最小值 0，所以当策略 $\pi(y|x)$ 恰好等于最优策略 $\pi^*(y|x)$ 时，整个目标函数取得最大值。即最优策略的闭式解为：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right) \quad \text{--- (式 1)}$$

---

### 2. 步骤二：用最优策略表达隐含奖励 (Implicit Reward)
我们可以利用上面的最优策略闭式解，将未知的奖励函数 $r(x, y)$ 用策略的概率表达出来。
对式 1 两边取自然对数：
$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{r(x, y)}{\beta} - \log Z(x)$$
移项并整理，可得隐含的真实奖励函数表达式：
$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x) \quad \text{--- (式 2)}$$
这是一个极其关键的发现：**隐含的奖励值其实就是最优策略较 SFT 基准的 log 增长率加上一个与动作无关的归一化常数**。

---

### 3. 步骤三：消去配分函数 Z(x)
现在，我们将式 2 代入 Bradley-Terry 偏好模型中：
$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$
由于 $r(x, y_w) - r(x, y_l)$ 中包含两项，我们将其作差展开：
$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta \log Z(x) \right)$$
注意！**由于两项中的 $\beta \log Z(x)$ 完全相同且只与状态 $x$ 相关，在相减中它们被极其完美地抵消了！**
$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$
因此，偏好概率可以重写为：
$$P(y_w \succ y_l \mid x) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

---

### 4. 步骤四：定义 DPO 损失函数
既然我们无法直接得到完美的策略 $\pi^*$，但我们拥有可优化的参数化策略模型 $\pi_\theta$。我们将 $\pi^*$ 用 $\pi_\theta$ 代替，直接在偏好对数据集 $\mathcal{D}$ 上最小化负对数似然损失：
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

### 5. 结论
通过这套推导，我们证明了**显式的奖励模型 $r_\phi(x, y)$ 在数学上被当前策略与参考策略的对数似然概率比之差完全取代**。
DPO 将复杂的强化学习策略搜索问题转变为在偏好数据集上对两个策略网络进行概率比值调节的简单监督二分类任务，从根本上消除了在线采样的不稳定性与多网络并存的显存瓶颈。


### 4.5 DPO 梯度的隐含误差加权机制 (Gradient Weighting)
* **物理意义解析**：DPO 损失函数的梯度在数学上带有一个非常优美的自动加权系数。
* **梯度权重公式**：
  $$\nabla_\theta \mathcal{L}_{\text{DPO}} \propto -\beta \cdot \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) \cdot \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]$$
  其中 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是模型隐含的奖励值。
* **加权机制点拨**：当模型判断错误，即分配给错误选项 $y_l$ 的奖励显著高于正确选项 $y_w$ 时，Sigmoid 项的值会接近 1，从而提供极大的梯度权重，强力纠正模型的参数；而当模型已经学会了正确偏好（$y_w$ 概率远高于 $y_l$）时，Sigmoid 项的值会趋近于 0，停止对参数进行无谓的更新。这在数学上等价于引入了隐式的**预测误差加权机制**，极大地提升了优化稳定性。
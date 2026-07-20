# 第 15 讲：“预训练之后”（中期/后期训练） (Lecture 15: "After Pretraining" (Mid/Posttraining))

## 第 1 页 (Page 1)

# 第 15 讲：“预训练之后”（中期/后期训练）

**CS336**

---

## 第 2 页 (Page 2)

### 课程至今的进展

我们现在已经涵盖了预训练，这让你能够得到类似 GPT-3 的模型。但我们如何才能得到 InstructGPT 呢？

---

## 第 3 页 (Page 3)

### 指令遵循是一种非凡的控制形式

*（图中的文本示例来自 [Bubeck et al 2023]）*

---

## 第 4 页 (Page 4)

### 今日目标：对语言模型（LM）输出进行更好、更紧密的控制

预训练数据并不完全是我们想要的（但它具有可扩展性）……

> **我们能否收集我们所期望的行为数据并训练语言模型？**

1. 这些数据看起来像什么？
2. 我们如何最好地利用这些数据？
3. 我们需要为此进行规模扩展（scale）吗？

---

## 第 5 页 (Page 5)

### 注意事项：后期训练（post-training）的信息相当稀疏！

**在 ChatGPT 竞争前时期的丰富信息**
* **Stiennon 2020 (RLHF 论文)** —— 包含标注指南等。
* **Bai 2022 (Anthropic HH)** —— 关于如何设置安全标注等的细节。

**现代时期的匮乏信息……**
* **开源**：大量的蒸馏（distillation），在开源版本说明中并不太公开。
* **闭源**：数据的“秘方”。

---

## 第 6 页 (Page 6)

### 今日课程的定位

**标准方法** —— 模仿学习（SFT）后接强化学习（基于人类反馈的强化学习，即 “RL” HF）

[Ouyang 2022] 第一部分：**“监督微调”（Supervised Finetuning, SFT）** 部分。

---

## 第 7 页 (Page 7)

### SFT 的要素有哪些？

* **训练数据**
* **方法**

---

## 第 8 页 (Page 8)

### 训练数据

我们已经见过一些主要的指令数据类型……

让我们来讨论关于指令微调数据集的另外两个细节：
1. 这些数据集里面实际装了什么？
2. 在构建“高性能”指令微调数据时，什么才是关键？

---

## 第 9 页 (Page 9)

### SFT 数据的演进（在开源世界中）

* **FLAN**
* **Self-instruct**
* **Alpaca**
* **ShareGPT/Vicuna**
* **Open Assistant**
* **WizardLM**
* **Tulu3**
* **Nemotron 工具使用等**

---

## 第 10 页 (Page 10)

### FLAN —— 随机示例

| 提示词 (Prompt) | 期望输出 (Target) |
| :--- | :--- |
| Stephanie - 能否把附件搞定并签字？我需要先签缩写，但它需要 Brad Richter 签字。谢谢。请为此邮件写一个主题。 | Ronald Chisholm 的意向书 (LOI) |
| Ahold 将向 Permira 出售西班牙业务 (AP) AP - 荷兰超市零售商 Ahold 意图精简全球业务并减少债务，周日表示将以约 3.6849 亿美元的价格将其在西班牙的股份出售给 Permira 基金。这段文本是关于什么的？选项：- 世界 - 体育 - 商业 - 科学/技术 | 商业 |
| 为本文写一段摘要：沿着绿意盎然的街道走过典型的荷兰梯形山墙建筑，成片的番红花，偶尔有骑行者安静地掠过，阳光在运河上跳跃，我想不出比海牙更令人愉快地度过春天周末的地方了。[...] 荷兰最珍贵的殖民地显然是从海牙（又称“印尼遗孀”）进行管理的。夏天：Veenkade 上的 Grapes & Olives 和 Café de Biebon 将在翻新后的运河上提供游船，并计划有音乐 and 乘船游览，包括参观 Kompaan 啤酒厂（我可以为该产品做担保）。我的艺术饕餮之旅在 Panorama 结束，这是一个由 Hendrik Mesdag 绘制的、海牙海滨度假胜地 Scheveningen 的 14 英尺高圆形全景画，令人叹为观止。我坐了 15 分钟的有轨电车到达海滩，在港口边的 Catch by Simonis 享用扇贝和鳎鱼。海牙吸引了一位新粉丝。英国航空（0844 493 0787, www.ba.com）提供从鹿特丹海牙机场起飞的航班（半小时出租车车程），往返 100 英镑起。关于 Remco Dörr 的导览，请发送电子邮件至 remco.dorr@hotmail.com。Carlton Ambassador 酒店（00 31 70 363 0363, www.carlton.nl/ambassador）的双人间价格为 84 英镑起（仅限房间）。更多关于荷兰旅游的信息，请访问 www.holland.com。 | 荷兰城市中最不为人知的是海牙，在 1806 年之前它只是一个村庄。它的成长归功于拿破仑的兄弟路易·波拿巴，他曾统治这里。这座城市拥有丰富的艺术财富，包括维米尔的《戴珍珠耳环的少女》。 |
| 这里有一些关于一家餐馆的数据：name = Aromi, eatType = coffee shop, food = English, customer rating = 5 out of 5, area = city centre. 写一句话，包含以下关于餐馆的数据。 | 在市中心有一家评分为 5 分（满分 5 分）的咖啡店，名为 Aromi，供应英式食物。 |

---

## 第 11 页 (Page 11)

### Alpaca —— 随机示例

| 提示词 (Prompt) | 期望输出 (Target) |
| :--- | :--- |
| 给出三个保持健康的建议。 | 1. 饮食均衡，确保包含丰富的水果和蔬菜。<br>2. 定期运动，保持身体活跃和强壮。<br>3. 保证充足的睡眠，并维持规律的作息时间。 |
| “算法”是什么意思？ | 算法是用于解决问题或达到目标的一步一步的指令集。它是将复杂任务分解为更小、更易于管理的步骤的过程。算法可用于在几秒钟内找到从一个地方到另一个地方的最短路径、对数据进行排序或从巨大的数据库中渲染数据。 |
| 寻找列表中的平均数。 | ```python\ndef avg_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums)\n\nprint(avg_list([4, 3, 6, 10, 8])) # 输出 6.4\n``` |

---

## 第 12 页 (Page 12)

### OpenAssistant —— 随机示例

| 提示词 (Prompt) | 期望输出 (Target) |
| :--- | :--- |
| 你能简要介绍一下“买方垄断 (monopsony)”在经济学中的相关性吗？请使用劳动力市场中潜在买方垄断的例子，并引用相关的研究。 | “买方垄断”是指市场上只有某一种商品或服务的单一买方的市场结构。在经济学中，这个词在劳动力市场中特别相关，因为买方垄断雇主对员工的工资和工作条件具有显著的控制力。[...] 总的来说，买方垄断的概念对于理解劳动力市场的动态以及市场力量对工人的影响至关重要。需要进一步的研究来了解买方垄断在经济中的程度和影响，并制定政策来解决这一问题。参考文献：Bivens, J., & Mishel, L. (2013). The Pay of Corporate Executives and Financial Professionals as Evidence of Rents in Top 1 Percent Incomes. Journal of Economic Perspectives, 27(3), 57-78. |
| 我的孩子需要为小学做一个科学项目，有什么既便宜又有趣的科学项目可以做？ | 小学阶段的孩子可以做很多既便宜又有趣的科学项目！这里有几个点子：1. 自制熔岩灯：在干净的塑料瓶里装入水和植物油，滴入几滴食用色素，然后扔进一片泡腾片。化学反应会使气泡像熔岩灯一样上升和下降。[...] 7. 制作太阳能烤箱：使用披萨盒、铝箔和塑料薄膜制作一个太阳能烤箱，可以烤棉花糖夹心饼或其他小点心。8. 探索磁铁：使用磁铁和各种物品，让你的孩子进行磁性实验，看看哪些物品会被磁铁吸引，哪些不会。这只是给你一个起步的点子！还有很多可以使用家用物品和一点点创意来完成的便宜又有趣的科学项目。 |

---

## 第 13 页 (Page 13)

### Nemotron-SFT-OpenCode-v1

| 提示词 (Prompt) | 期望输出 (Target) |
| :--- | :--- |
| 在 JavaScript 中处理 Promise 时，使用 `async/await` 和 `.then()` 有哪些主要区别？在可读性和错误处理方面，什么时候应该首选其中一种方法？ | `{"role": "assistant", "content": "I'll analyze the key differences between async/await and .then() for handling promises in JavaScript based on the provided AGENTS.md instructions.", "tool_calls": [ { "id": "call_132a55728f6c498fbeac3b95", "type": "function", "function": { "name": "skill", "arguments": "{\"name\":\"bash-skills\"}" } } ]}` |
| 我正在构建一个简单的计算器应用，并且需要决定所有计算是使用整数还是浮点数。每种方法的权衡是什么？对于一个处理基本算术运算、对初学者友好的计算器，你推荐哪一种？ | `"role": "assistant", "content": "I'll help you decide between integers and floating-point numbers for your calculator app. Let me analyze the trade-offs and provide a recommendation based on the requirements in AGENTS.md.", "tool_calls": [{"id": "call_b06182c280514fc7bd1d3ae4","type": "function", "function": {"name": "todowrite","arguments": "{\"todos\":[{\"content\":\"Review the AGENTS.md file for calculator data typerequirements\",\"id\":\"1\",\"priority\":\"high\",\"status\":\"in_progress\"},{\"content\":\"Analyze trade-offs between integers and floating-point numbers\",\"id\":\"2\",\"priority\":\"high\",\"status\":\"pending\"},{\"content\":\"Check available skills for relevant guidance\",\"id\":\"3\",\"priority\":\"medium\",\"status\":\"pending\"},{\"content\":\"Provide recommendation for beginner-friendly calculator\",\"id\":\"4\",\"priority\":\"high\",\"status\":\"pending\"}]}"}}]` |

---

## 第 14 页 (Page 14)

### 这些数据集在哪些方面存在差异？

* **“话痨程度”（Chattiness）** —— FLAN 数据集通常是有效的数据，但人们不想与一个自然语言处理（NLP）基准测试对话。后来的数据集开始倾向于更长、更详细的回答。
* **细节程度（Detail）** —— OASST 数据集在各种事实性知识方面介绍得更加详细。正如我们所看到的，这可能既是优点也是缺点。
* **工具使用（Tool use）** —— 过去一两年的 SFT 也越来越倾向于工具使用、代理型（agentic）下游应用。

---

## 第 15 页 (Page 15)

### 我们在这些数据集中注意到了什么？

这些数据集在很多方面都存在差异……
* **长度和项目符号（列表样式）**（风格变体）
* **参考文献以及其他复杂的知识**

*不那么明显但很重要的方面：*
* **规模 (Scale)**
* **安全性 (Safety)**

> **这些因素是如何影响模型的？**

---

## 第 16 页 (Page 16)

### 数据 and 模型中的风格变体

模型在回答长度上差异巨大……

*(图表展示了不同指令数据集的实例数量、轮数、提示词平均长度以及完成文本平均长度等。)*

---

## 第 17 页 (Page 17)

### 当使用偏好进行评估时，风格至关重要

我们看到了非常显著的长度效应（在人类评估和基于 GPT 的评估中均是如此）。

*(图表展示了对列表（%）的偏好和对更长输出（%）的偏好。)*

---

## 第 18 页 (Page 18)

### 那基准测试（Benchmarks）呢？

这些因素（大多）与其他基准测试的表现不太相关。

*(图表展示了不同指令微调数据集的对比，显示它们可以在不同方面表现突出，而混合数据通常平均表现最好。)*

---

## 第 19 页 (Page 19)

### 参考文献、复杂知识与事实性

考虑这个来自 Open Assistant 的例子：

> **问**：你能简要介绍一下“买方垄断”在经济学中的相关性吗？请使用劳动力市场中潜在买方垄断的例子，并引用相关的研究。
> 
> **答**：“买方垄断”是指……。参考文献：Bivens, J., & Mishel, L. (2013). The Pay of Corporate Executives and Financial Professionals as Evidence of Rents in Top 1 Percent Incomes. Journal of Economic Perspectives, 27(3), 57-78.

这个例子教会了模型什么？
1. 教会模型关于 Bivens J & Mishel, L 的具体知识
2. 教会模型在被要求时输出引用文献

> （但是通过什么机制？模型真的了解这些引用吗？）

---

## 第 20 页 (Page 20)

### 知识提取与对齐

> **民间传闻（Folklore）**：在模型“不知道的事实”上对其进行微调，会导致模型产生幻觉。

*(图表展示了幻觉与行为克隆（Behavior Cloning），以及在未知事实（Train Unknown）上训练时出现的过拟合现象。)*

---

## 第 21 页 (Page 21)

### 知识提取与对齐的启示

1. 你可能**不想**在长尾知识上进行微调，即使那是语言模型的应用场景。
2. 原则上，“强化学习（RL）” 风格的正确性反馈可能会有所帮助。
3. 语言模型中的知识存储与提取是非常混乱且微妙的。

---

## 第 22 页 (Page 22)

### 安全性

语言模型被广泛部署给终端用户，需要一些安全控制。

* **错误信息（Misinformation）** —— 参见 [Goldstein+ 2023]
* **诈骗与垃圾邮件（Scams and spam）** —— 参见 [Kang+ 2023]

---

## 第 23 页 (Page 23)

### 现实中的安全 SFT

具体细节非常稀疏。

* **样本数量**：在 Llama 2 中大约为几千个。
* 安全训练数据的质量和设计对性能有深远的影响。通过广泛的消融实验，我们发现质量比数量更关键。

---

## 第 24 页 (Page 24)

### 细节最丰富的管线

**Tülü 3: Pushing Frontiers in Open Language Model Post-Training**

* 如图表所示，安全与反合规（Safety & Non-Compliance）的数据量（例如 Tülü 3 CoCoNot、WildJailbreak、WildGuardMix）在几万量级。

---

## 第 25 页 (Page 25)

### 主要的安全方法 —— 从用户中提取场景

* **WildChat**：野外 100 万真实 ChatGPT 交互日志。
* 如图展示了 WildTeaming 框架的两个步骤：**MINE**（从真实世界用户编写 of 越狱尝试中自动挖掘越狱策略）与 **COMPOSE**（将越狱策略组合成多样化的对抗性攻击）。

---

## 第 26 页 (Page 26)

### 仅需少量数据即可进行安全微调

通过约 500 个样本就可以对安全性产生显著改善。
> 加入 500 个 Alpaca 风格的示例就能促使模型遵循安全指南（例如针对恶意指令、仇恨言论或 Anthropic HH 数据集）。

---

## 第 27 页 (Page 27)

### 总结 —— SFT 数据

1. **指令微调（SFT）在仅提取预训练行为而非添加新行为时效果最好**。
2. 添加（即使是事实正确的！）数据有时会带来损害。
3. **少量的特定类型行为数据**（安全性、指令遵循、风格）就能带来巨大差异，但要覆盖长尾分布仍能从更多数据中受益。

---

## 第 28 页 (Page 28)

### 如何进行微调

只需进行梯度下降……

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

在许多学术环境中，这基本上就是全部了。**但是如果……**
> 你拥有海量的算力和数据，并且希望扩大指令微调的规模？

---

## 第 29 页 (Page 29)

### 将指令微调转化为预训练

我们能否以某种方式将指令微调数据转化为预训练数据？

以下（越来越受欢迎的）想法给出了肯定的回答：
1. **在网页/预训练数据上进行预训练**
2. **将指令微调数据混入预训练中**
3. **进行一次实际的（但很短的）指令微调**

> 这使你可以在不发生灾难性遗忘的情况下扩大指令微调的规模。

---

## 第 30 页 (Page 30)

### “中期训练” / “两阶段训练” (Midtraining / Two-phase training)

这一配方是许多大语言模型公司之间的常识（但未被记录在案）。

* **稳定阶段的数据混合（Data Mixture of Stable Stage）**：Dolma、CommonCrawl、Code Pretrain、Open Web Math 等。
* **衰减阶段的数据混合（Data Mixture of Decay Stage）**：融入了各种 SFT/数学/代码数据。
* 目前被大多数模型广泛采用……在最近的中文源模型（如 miniCPM, jetMoE）中得到了公开宣传。

---

## 第 31 页 (Page 31)

### RLHF 的第二部分

**标准方法** —— 模仿学习（SFT）后接强化学习（基于人类反馈的强化学习，即 “RL” HF）

[Ouyang 2022] 第二部分：**“强化学习”（RL）** 部分。

---

## 第 32 页 (Page 32)

### 从模仿到优化

#### 模仿学习 (SFT)
拟合 $\hat{p}(y|x) \approx p^*(y|x)$ 以匹配某些参考分布 $p^*(y|x)$
* 纯生成式建模的视角。
* 需要来自参考策略的样本。

#### 优化 (RLHF)
寻找 $\hat{p}(y|x)$ 以最大化期望回报：
$$\max_{p} \mathbb{E}_{y \sim p} [R(y, x)]$$
以获取奖励 $R(y,x)$
* 最大化我们可以测量的某些奖励函数。
* 语言模型是策略（policies），而不是某种分布的模型。

---

## 第 33 页 (Page 33)

### 为什么优化？G-V 差距 (G-V gap)

人们在语言模型输出中，并不总是写出他们最偏好的东西……

*(图表引自：Zhang et al – “Benchmarking Large Language Models for News Summarization” 2023)*

---

## 第 34 页 (Page 34)

### 概述

我们将涵盖 RLHF 的三个方面：

* **数据**
  * 人们如何收集 RLHF data？
  * 有哪些需要担心的事情？
* **我们如何进行 RLHF？**
  * PPO
  * DPO
* **RLHF 有哪些副作用？**

---

## 第 35 页 (Page 35)

### RLHF 数据

什么是成对偏好反馈（pairwise feedback）？我们如何获得（高质量的）成对偏好反馈？

*(图示展示了收集比较数据并训练奖励模型，然后使用强化学习针对奖励模型优化策略的过程。)*

---

## 第 36 页 (Page 36)

### RLHF 与数据 —— 标准设置

标准的“成对反馈（pairwise feedback）”设置：

*(图示展示了一个 MTurk 界面，标注员需要在两个 AI 回答中做出选择：“回答 1 更好”、“回答 1 仅稍微好一点”、“回答 2 仅稍微好一点”、“回答 2 更好”。)*

---

## 第 37 页 (Page 37)

### RLHF 与数据 —— instruct GPT guideline

展示了 API 提示词分布上标注指令的摘录，要求回答必须满足 **helpful (有帮助的)**, **truthful (诚实的)**, 和 **harmless (无害的)**。

---

## 第 38 页 (Page 38)

### 另一个旧示例 —— bard annotations

展示了旧的谷歌 Bard 标注指南，其中包含对“帮助程度”（Helpfulness）和“表达呈现”（Presentation）的评分标准。

---

## 第 39 页 (Page 39)

### 现代数据标注员分布

*注意：许多平台，此处仅展示了一家（如 Outlier, ScaleAI）*

*(图表展示了标注员的年龄分布、最高学历分布以及专业领域分布，其中 Bachelor 占 44%，Master 占 32%，学科以语言、创意写作等为主。)*

---

## 第 40 页 (Page 40)

### 报酬的巨大差异

“专家标注（expert annotation）”有很大的增长。
在 Project Stagecraft 项目中，自由职业者被支付至少每小时 50 美元来创建用于 OpenAI ChatGPT 的材料，以理解各种职业。

---

## 第 41 页 (Page 41)

### RLHF 与数据 —— 众包

**众包的复杂性：**
* 很难获得真正高质量、可验证的标注员。
* 很难让他们去真正检查正确性。
* 必须警惕标注员使用 AI……

---

## 第 42 页 (Page 42)

### RLHF 与数据 —— 众包伦理

大规模的数据收集可能会带来严重的伦理问题。
* 例如：媒体报道“OpenAI 雇佣肯尼亚工人以低于每小时 2 美元的报酬来让 ChatGPT 减少毒性”。
* “美国已经拥有了一个 AI 底层阶级”。

---

## 第 43 页 (Page 43)

### RLHF 与数据 —— 人口统计特征

RLHF 的标注员分布可能会显著改变模型的行为。

*(图表引自 [Santurkar+ 2023]，说明了标注员人口统计学特征如宗教信仰、性别、年龄等对模型对齐结果的影响。)*

---

## 第 44 页 (Page 44)

### RLHF 与风格 —— 标注员极其关键

这对于很多标注员而言都是成立的 —— 但并非所有标注员都是如此。

*(图表引自 [Hosking, Blunsom, Bartolo 2024]，展示了众包标注与“专家”标注之间的错误率差异。标注员倾向于低估自信输出中的不一致性或事实错误。)*

---

## 第 45 页 (Page 45)

### RLHF 与数据 —— LM-generated

GPT-4 是一个令人惊讶地优秀的成对偏好反馈系统。
* 在系统级别上具有近乎完美的秩相关性（Spearman Correlation: 0.98, $R^2 = 0.87$）。
* 偏好一致性接近人类标注员之间的水平。

---

## 第 46 页 (Page 46)

### RLHF 与数据 —— LM-generated (续)

对于跟踪前沿的人来说 —— AI 反馈经常被用于 RLHF（例如 Ultrafeedback, Zephyr 7b, Tulu3）。
常用于 Olmo, Zephyr 等模型。

---

## 第 47 页 (Page 47)

### RLHF 与数据 —— 自训练 (Self-training)

*(图示展示了宪法 AI (Constitutional AI, Bai et al.) 的架构：利用有帮助的 RLHF 模型生成对“红队（Red Teaming）”提示词的回答，然后通过批判（Critique）和修改（Revision）进行监督训练，继而进行偏好训练。)*

---

## 第 48 页 (Page 48)

### RLHF 与风格 —— 长度效应

长度效应是 RLHF 的一个非常显著的结果。

*(图表引自 [Chen et al 2024] 和 [Singhal et al 2024]，展示了 SFT 回答（59个 token）与 RLHF 回答（243个 token）的对比：输出相似，但后者要长得多，细节也更多。)*

---

## 第 49 页 (Page 49)

### 我们如何进行 RLHF？

We now have a (high quality) pairwise feedback data collection pipeline?
How do we adapt the model to make use of pairwise feedback?
* **第一部分：PPO** —— 原始的且非常微妙（finicky）的方法（简短版本）。
* **第二部分：DPO** —— 新的、非常易于使用的方法。

---

## 第 50 页 (Page 50)

### 从模仿到优化

*(重新展示了 SFT 与 RLHF 的公式对比。)*

---

## 第 51 页 (Page 51)

### 语言建模中的 PPO

**来自 InstructGPT**

在 RL 训练中最大化如下组合目标函数：
$$\text{objective}(\phi) = \mathbb{E}_{(x, y) \sim D_{\pi_\phi^{\text{RL}}}} \left[ r_\theta(x, y) - \beta \log\left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right) \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} [\log(\pi_\phi^{\text{RL}}(x))]$$

其中 $\pi_\phi^{\text{RL}}$ 是学到的 RL 策略，$\pi^{\text{SFT}}$ 是监督微调的模型，$D_{\text{pretrain}}$ 是预训练数据分布。
> ……这看起来非常无害。

---

## 第 52 页 (Page 52)

### 更多来自 Stiennon 的细节与背景

**来自 "Learning to summarize from human feedback”**
* **奖励模型（Reward models）**：损失函数为：
  $$\text{loss}(r_\theta) = -\mathbb{E}_{(x, y_0, y_1, i) \sim D} [\log(\sigma(r_\theta(x, y_i) - r_\theta(x, y_{1-i})))]$$
* **人类反馈策略（Human feedback policies）**：完整奖励 $R$ 可以写为：
  $$R(x, y) = r_\theta(x, y) - \beta \log[\pi_\phi^{\text{RL}}(y|x) / \pi^{\text{SFT}}(y|x)]$$

---

## 第 53 页 (Page 53)

### PPO —— 在概念层面上

A brief (and high level) intro to the various ideas in PPO..

* **尝试 1：策略梯度 (Policy gradients)**（方差太大）
  $$\nabla_\theta \mathbb{E}_{p_\theta} [R(z)] = \mathbb{E}_{p_\theta} [R(z) \nabla_\theta \log p_\theta(z)]$$
* **尝试 2：TRPO**（在当前策略周围线性化问题）
* **尝试 3：PPO**（在某些 $\epsilon$ 处对比例进行剪切/裁剪）
  $$L(s, a, \theta_k, \theta) = \min\left( \frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a), \, \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}, 1-\epsilon, 1+\epsilon\right) A^{\pi_{\theta_k}}(s, a) \right)$$

---

## 第 54 页 (Page 54)

### 我们能摆脱 PPO 吗？

> **许多人的疑问**：我们能否避免做任何“RL”？（即同策略 on-policy RL 算法）

一些人们想到的合理做法：
* **使用控制 Token 训练模型** —— （在成对数据上进行 SFT，在选择的回答前加上 `[GOOD]`，在未选的回答前加上 `[BAD]`）。
* **仅在偏好的输出上训练模型**。
* **训练一个奖励模型，获得 LM 输出，在偏好的输出上进行训练**。
* **训练一个奖励模型，获得 1024 个 LM 输出，选取最好的一个**。

---

## 第 55 页 (Page 55)

### DPO —— 无痛的 RLHF？

尝试通过以下方式简化 PPO：
* **去掉奖励模型**
* **去掉任何同策略（on-policy）的操作**（Rollouts, 外循环等）

**取而代之的是：**
* 在好的内容的对数损失（log-loss）上进行梯度上升。
* 在坏的内容上进行负梯度步（并进行适当的加权）。

*(对比了传统的 RLHF 与 DPO 的简化路径。)*

---

## 第 56 页 (Page 56)

### DPO —— 从 RLHF 公式推导

我们的目标是优化：
$$\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)} [r_\phi(x,y)] - \beta \mathbb{D}_{\text{KL}} [\pi_\theta(y|x) \parallel \pi_{\text{ref}}(y|x)]$$

假设策略 $\pi$ 是所有策略的集合（非参数化假设）。最大化者为：
$$\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

求解“隐含奖励（implied reward）”：
$$r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$
*(这也是 kimi-think 论文中使用的等价性。)*

---

## 第 57 页 (Page 57)

### DPO 推导 2

我们现在可以通过 Stiennon 目标函数，将隐含奖励作为一个奖励模型来进行优化：
$$\text{loss}(r_\theta) = -\mathbb{E}_{(x, y_0, y_1, i) \sim D} [\log(\sigma(r_\theta(x, y_i) - r_\theta(x, y_{1-i})))] \quad \text{且} \quad r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

这给出了 DPO 目标函数：
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

**关键步骤有哪些？**
1. 做出非参数化假设（以闭式解联系 $\pi_\theta$ 和 $r$）。
2. 通过策略参数化奖励 $r$。
3. 使用监督损失优化奖励（这反过来优化了策略）。

> 概念上，这是在非参数假设和替代参数化下，对成对奖励进行的极大似然估计（MLE）。

---

## 第 58 页 (Page 58)

### DPO 更新与组件

从某种意义上说，它简化为“对好的内容进行正梯度更新，对坏的内容进行负梯度更新”。
$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{当奖励估计错误时权重更高}} \left[ \underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{增加 } y_w \text{ 的似然}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{减少 } y_l \text{ 的似然}} \right] \right]$$
> （由隐含奖励模型的“预测误差”进行缩放）

---

## 第 59 页 (Page 59)

### DPO in LLaMA (and other open models)

**DPO + 用于后期训练（post-training）的专家迭代（Expert iteration）**

*(图示展示了从收集的提示词生成 K 个候选回答，进行拒绝采样（Rejection Sampling），得到 SFT 数据，训练 SFT 模型，然后再进行 DPO 训练并迭代得到最终模型的流程。)*

---

## 第 60 页 (Page 60)

### 变体

有很多变体（太多变体了），但也许 Tulu 3 论文中值得注意的有这两个：
* **SimPO (无需参考模型)**
  $$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$
* **长度归一化的 DPO (Length normalized DPO)**

---

## 第 61 页 (Page 61)

### 但 PPO 也可以（而且有时更好？）

强化学习（RL）相关经验工作的微妙之处……
许多结果都高度依赖于具体的实验设置。

*(图表对比了在 Tulu 3 框架中，PPO 和各种 DPO 变体在不同超参数和基准测试上的表现，显示 PPO 依然非常强大。)*

---

## 第 62 页 (Page 62)

### RLHF 中需要注意的事项

* **过度优化 / 对奖励的过拟合 (Overoptimization / overfitting on the reward)**
* **模式崩塌 / 熵减少 (Mode collapse / entropy)**

---

## 第 63 页 (Page 63)

### 需要注意的事项 —— 过度优化

在很多不同的 RLHF 风格优化器中：针对奖励进行优化，超过一定点就会发生过拟合。
> 这适用于人类偏好（左）和嘈杂的语言模型偏好（中），但不适用于无噪的语言模型偏好（right）。

---

## 第 64 页 (Page 64)

### 需要注意的事项 —— mode collapse

RLHF 使得模型不再是“概率模型” —— 默认情况下没有校准（no calibration）。

*(图表展示了经过 RLHF 校准的 MMLU 曲线，说明了模型输出概率的校准问题以及熵的变化。)*

---

## 第 65 页 (Page 65)

### 课程回顾

**RLHF 回顾：**
1. **RLHF 的数据收集也十分困难**！存在许多混淆因素。
2. **RLHF 算法比 SFT 稍微复杂一些** —— 特别是 PPO。
3. **要注意过度优化奖励所带来的影响**。

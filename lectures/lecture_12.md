# 第 12 讲：模型评估 (Evaluation)

> **核心议题**：给定一个训练好的语言模型，我们该如何科学、客观、全面地衡量它到底有“多好”？

- **已学内容体系**：我们已经系统讲解了语言模型训练的方方面面（模型架构、优化算法、系统并行、扩展定律）。

- **关键缺失拼图**：模型究竟应该在**什么样的数据**上进行训练？

- **数据塑造能力**：训练数据直接决定了模型的行为特征与能力边界（代码生成？多语言翻译？生物 DNA 序列建模？）。

- **逻辑先决条件**：在深入探讨数据工程之前，我们必须首先明确：**我们期望模型展现出怎样的具体能力与行为？**



### 什么是评估 (Evaluation)？

> **评估的核心问题**：给定一个训练好的模型，它究竟有“**多好**”？



### 评估的表象与实质

表面上看，大模型评估似乎只是一个标准化的机械流程：

1. **准备测试提示**：定义一批测试提示词 (Prompts)

2. **模型生成**：输入模型并收集输出回复 (Responses)

3. **计算准确率**：对照标准答案计算准确率或得分 (Accuracy)

但实际上，**模型评估是一个极其深刻、复杂且影响深远的前沿课题……**

……正是评估标准与基准排行榜的演进，直接引导并塑造了整个人工智能行业的技术发展路线。

> **评估的核心挑战**：如何将人类期望的<font color="red">抽象概念 (Abstract Construct，如“聪明”、“有用”、“安全”)</font> 转化为计算机可精确计算的<font color="blue">具体量化指标 (Concrete Metric)</font>？

#### 维度 1：基准测试得分 (Benchmark Performance)
如果一个模型在各类标准化基准题库上得分很高，它就是个好模型吗？

[Artificial Analysis](https://artificialanalysis.ai/)

<img src="images/artificial-analysis.png" width="800" />

#### 维度 2：性价比与推理成本 (Cost-Efficiency)
如果模型不仅能力出众，而且单次调用的推理 Token 开销极低（极具性价比），它是个好模型吗？

<img src="images/artificial-analysis-cost.png" width="800" />

#### 维度 3：人类主观偏好 (Human Preference)
如果真实用户在双盲盲测中更喜欢它的回复风格与回答质量，它是个好模型吗？

[Arena AI (formerly Chatbot Arena)](https://arena.ai/leaderboard)

<img src="images/lmarena-leaderboard.png" width="400" />

#### 维度 4：真实市场占有率 (Market Adoption)
如果全球开发者和企业用脚投票，频繁调用 API 并愿意真金白银为其付费，它是个好模型吗？

[OpenRouter](https://openrouter.ai/rankings)

<img src="images/openrouter.png" width="600" />



- **回顾概率本质**：语言模型在数学上是定义在 Token 序列上的联合概率分布 **$p(x)$**。

- **困惑度 (Perplexity, PPL)**：定义为 $(1/p(D))^{1/|D|}$，衡量概率模型 $p$ 为测试集 $D$ 赋予的高概率程度（困惑度越低，预测越准）。

> 💡 **数学本质与物理直觉解析（困惑度与交叉熵）**：
> - **联合概率连乘**：由概率链式法则，$p(D) = \prod_{i=1}^{|D|} p(x_i \mid x_{<i})$，本质上是测试集中所有 Token 自回归预测概率的累积连乘积。
> - **单 Token 几何平均的倒数**：表达式 $(1/p(D))^{1/|D|} = \left(\prod p_i\right)^{-1/|D|} = 1/\bar{p}_{\text{geom}}$，在数学上恰好是**每个 Token 预测概率的几何平均值的倒数**（有效消除了文本长短对联合概率连乘趋零的影响）。
> - **取对数严格等价于交叉熵**：对 PPL 取对数即可得到 $\log(\text{PPL}) = -\frac{1}{|D|} \sum \log p(x_i \mid x_{<i}) = \mathcal{L}_{CE}$，即 $\text{PPL} = \exp(\mathcal{L}_{CE})$。最小化交叉熵损失在数学上严格等同于压低困惑度。
> - **物理含义（等效单选分支数）**：PPL 直观反映了模型预测下一个 Token 时的迷茫程度，等价于在一个包含多少个选项的均等随机骰子中做选择（如 PPL=10 相当于十选一猜测）；PPL 越低，模型越笃定准确。

- 预训练的目标函数正是最小化训练集上的困惑度（等价于最小化交叉熵损失）。

- 最直观的评估手段：在未见过的独立测试集切片上直接测量模型的困惑度。

- 这也是传统统计语言模型与早期 NLP 研究中的标准黄金指标。

经典的语言模型基准数据集：

- **Penn Treebank (PTB)**：华尔街日报金融语料

- **WikiText-103**：维基百科高质量长文章集合

- **One Billion Word Benchmark (1BW)**：欧洲议会、联合国与国际新闻的大型语料库

**经典同分布评估范式 (In-Distribution Evaluation)**：在同一数据集的 Train 切片上训练，在 Test 切片上验证。

经典 CNN+LSTM 架构在 1BW 十亿词基准上的困惑度演进 (51.3 → 30.0)[https://arxiv.org/abs/1602.02410](https://arxiv.org/abs/1602.02410)



### GPT-2 开启的零样本/跨分布评估 (Out-of-Distribution)

- 在大规模开放网页语料 WebText（40GB，来自 Reddit 社区高赞外链）上进行通用预训练

- **零样本评估**：不经过任何微调，直接在 PTB、1BW 等标准数据集上测试困惑度

<img src="images/gpt2-perplexity.png" width="800" />

- 发现规律：在小规模数据集 (PTB) 上跨领域泛化优异；但在海量专有语料 (1BW) 上，依然不如直接在该语料内部训练的模型



### “困惑度即一切 (Perplexity is all you need)” 假说

- 设真实世界的信息分布为 $t$，语言模型学习到的分布为 $p$。

- 理论最优困惑度下界为信息熵 $H(t)$，当且仅当模型完全捕捉真实分布 $p = t$ 时取得。

- 如果 $p = t$，则一切现实世界任务皆可迎刃而解：只需推导条件概率 $p(\text{答案} \mid \text{问题})$。

- 因此，只要把困惑度推向理论极致，就必然能够通往通用人工智能 (AGI)。



### 困惑度指标的局限性

- 示例句子：*“Stanford was founded in 1885”*

- 困惑度会对序列中的**每一个 Token** 进行严格惩罚，但很多虚词（如 *was*、*in*）的预测并不影响对核心事实的掌握。

- **改进方案**：引入条件困惑度 $p(\text{回复} \mid \text{提示})^{1/|\text{回复}|}$，只针对生成的核心答案进行惩罚。



### 伪装成下游任务的困惑度测试

- 完形填空任务 (Cloze Task)：**LAMBADA**（测试长文本上下文下最后一个单词的预测能力）[https://arxiv.org/abs/1606.06031](https://arxiv.org/abs/1606.06031)

<img src="images/lambada.png" width="700" />

- 多项选择句子逻辑补全：**HellaSwag**（测试日常常识推理）[https://arxiv.org/pdf/1905.07830](https://arxiv.org/pdf/1905.07830)

<img src="images/hellaswag.png" width="500" />

> 💡 **深度解析：从“条件似然打分”到“现代自由生成/CoT”的评测演进**：
> 1. **为什么叫“伪装成下游任务的困惑度测试”？**
>    - **本质机制**：此阶段模型**并未进行开放式自由生成**，而是将分类/完形填空转化为**条件对数似然打分** $\log P(\text{Target} \mid \text{Prompt})$。对于单向自回归（Causal LM）模型，待测词/选项必须拼接在 Prompt 末端，模型仅跑一次前向传播（Forward Pass）计算目标词序列的 Softmax 概率乘积（累加交叉熵），选出让困惑度最低（最不感惊讶）的选项。
>    - **若待填空的词在“句子中间”如何评测？**
>      - **单向自回归特性决定其不能直接看后文**（不同于 BERT 这种天然能看双向上下文的掩码语言模型 MLM）。
>      - **方案 A（整句/后半句似然比对）**：分别将不同候选词填入该位置组成完整句子，计算整个后半句在模型下的条件概率 $\log P(\text{待测词}+\text{后文} \mid \text{前文})$，比较哪一个候选词引导的完整句子困惑度更低。
>      - **方案 B（Prompt 改写为末尾生成）**：将挖空题重构为问答式 Prompt（如：`"Complete the sentence: 'The ___ sat on the mat.' The missing word is:"`），使待测词在逻辑上重新变成序列末尾。
>    - **选择题两大似然打分法**：① **句子补全似然**：将每个选项分别拼到题干末尾，对比各选项对应 Token 串的条件对数似然；② **选项字母探测**：输入 `"Question... Options... Answer:"`，只取紧随其后首个 Token 位置上 `'A'`, `'B'`, `'C'`, `'D'` 的 Logits 概率峰值。
> 2. **为什么现代评测全面转向“自由生成与思维链 (CoT)”？**
>    - **释放推理潜力**：数学/代码等复杂任务极度依赖思维链（CoT）提供的思考缓冲空间（Scratchpad），强制单步预测会压抑模型推理能力。
>    - **黑盒 API 现实**：大多数商用大模型 API 不开放全词表 Logits/Logprobs，只能作为黑盒输入 Prompt 并抓取文本输出。
>    - **贴近真实交互**：真实用户均为开放式对话，自由生成更贴近真实体验。
> 3. **为何似然打分在今天依然不可替代？**
>    - **基座模型 (Base Model) 评估刚需**：未经过 SFT/RLHF 的 Base 模型不会遵循复杂指令生成格式，必须依赖似然打分探测纯预训练知识储备。
>    - **极高吞吐与零解析误差**：似然打分只需单次并行前向传播（毫秒级，无需循环自回归采样），且 100% 客观确定，不存在自由生成中正则匹配失败（Format/Parsing Error）的痛点。

> **⚠️ 警示（如果你负责维护困惑度评测榜单）**：

- 参赛者提交模型 `LM`，评测平台调用 `log_prob = LM(test_data)` 计算对数似然。

- 你必须保证模型输出的概率分布在数学上严格归一（总和为 1，防止通过异常缩放 Logits 恶意作弊）。

- 对于真实下游任务，直接让模型自回归生成 `response = LM(prompt)` 并校验结果准确率更加安全稳健。



### 过滤阶段小结

- **困惑度的重要价值**：在底层模型开发中极其关键（能展现出极为平滑的 Scaling Laws 幂律曲线）。

- **现实需求**：我们需要更丰富多元、贴近人类复杂生产生活真实场景的评测基准……



## 1. 考试类基准 (Exam Benchmarks)

- **学科与难度可控**：可以灵活覆盖特定学科领域并严格划分难度梯度。

- **客观易判分**：标准答案无歧义，自动化批改效率极高。

### MMLU (大规模多任务语言理解基准)[mmlu_2021 (Berkeley)](https://arxiv.org/pdf/2009.03300.pdf)

- 包含 57 个学科领域（数学、物理、法律、医学、哲学等）的单选题。

- “由高校师生从公开网络考试题库中收集整理”。

- 实质：虽然名字叫语言理解，但本质上主要考核模型的**人类世界百科知识储备**。

- 经典评测方式：使用少样本提示 (Few-shot Prompting) 进行测试。

> 💡 **历史背景与核心用途（为什么成熟模型测 MMLU 仍用 Few-shot？）**：
> 1. **核心场景：面向未经后训练（SFT/RLHF）的纯基座模型 (Base Model)**：
>    - MMLU 诞生于 2020 年（GPT-3 时代），基座模型本质上是纯粹的“无结构文本续写器”，既不懂什么是“对话问答”，也没有“只输出一个选项字母”的指令遵循意识。若直接零样本（0-shot）提问，基座模型可能会把试卷继续往下编造，导致评测完全失效。
> 2. **作用是“格式对齐与模式激活”，而非“补充知识”**：
>    - 通过在开头提供 5 个标准的问答样例（5-shot），利用大模型强大的**上下文学习（In-Context Learning）**机制，强行约束模型捕捉到模式：“在看到 `Answer:` 时下一个 Token 必须输出选项字母”。
> 3. **严谨的“控制变量”原则**：
>    - 目的是**剥离模型对齐程度（是否听话、是否会按格式答题）的干扰，纯粹评估预训练沉淀的世界百科知识储备**。
> 4. **沿用至今的原因（历史一致性对比）**：
>    - 尽管今天的 Instruct/Chat 模型早已具备极强的 0-shot 答题能力，但为了与历史上各大经典技术报告（GPT-3/4、Chinchilla、PaLM、LLaMA 等）保持严谨的**苹果对苹果（Apple-to-Apple）横向基准对比**，前沿大模型报告（如 LLaMA 3、DeepSeek-V3）依然将 5-shot MMLU 作为最权威的官方对齐基准指标。

<img src="images/mmlu.png" width="700" />

[https://llm-stats.com/benchmarks/mmlu](https://llm-stats.com/benchmarks/mmlu)

[HELM MMLU for visualizing predictions](https://crfm.stanford.edu/helm/mmlu/latest/)

### MMLU-Pro (高难度进阶推理基准)[https://arxiv.org/abs/2406.01574](https://arxiv.org/abs/2406.01574)

- 全面剔除原版 MMLU 中的含噪、有歧义与过于基础的题目。

- 选项从 4 选 1 扩展至 10 选 1（大幅削弱蒙猜几率）。

- 全面引入思维链 (Chain of Thought, CoT) 引导长逻辑推理。

- 结果：各大顶尖模型的得分断崖式下跌 16%~33%，有效解决了榜单饱和问题。

<img src="images/mmlu-pro.png" width="700" />

[https://llm-stats.com/benchmarks/mmlu-pro](https://llm-stats.com/benchmarks/mmlu-pro)

[HELM MMLU-Pro for visualizing predictions](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro)

### GPQA (研究生级别防 Google 检索问答基准)[https://arxiv.org/abs/2311.12022](https://arxiv.org/abs/2311.12022)

- 由 61 位跨学科博士独立命题，专门针对前沿理科学术概念。

<img src="images/gpqa.png" width="700" />

- 对应领域的博士专家盲测基准准确率为 65%。

- 非本专业的普通人即便允许联网 Google 搜索 30 分钟，准确率也仅有 34%。

- 早期 GPT-4 在该基准上仅取得 39% 准确率。

[https://llm-stats.com/benchmarks/gpqa](https://llm-stats.com/benchmarks/gpqa)

[HELM GPQA for visualizing predictions](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)

### Humanity's Last Exam (HLE / 人类最后的考试)[https://arxiv.org/abs/2501.14249](https://arxiv.org/abs/2501.14249)

- 包含 2500 道极高难度的跨学科多模态难题（多选与开放简答题）。

<img src="images/hle-examples.png" width="700" />

- 设立 50 万美元巨额奖金池征集顶级难题，题目均经过前沿 LLM 严苛过滤与查重。

- 经过多轮前沿 LLM 与专家交叉评审，确保当前模型无法靠简单记忆检索答对。

<img src="images/hle-pipeline.png" width="700" />

<img src="images/hle-results.png" width="600" />

[https://llm-stats.com/benchmarks/hle](https://llm-stats.com/benchmarks/hle)



### 过滤阶段小结

- 随着模型能力持续飞跃，基准测试的难度上限在不断被推向极致。

- 选择题形式虽然可以无限提升难度，但无法完全等同于真实场景的复杂综合输出。

- 局限：无法涵盖没有唯一标准答案的开放式对话与协作任务。



## 2. 开放式对话与偏好评测 (Chat & Preference Benchmarks)

在现实生活中，绝大多数用户使用的是开放式提示词，而非结构化选择题：

示例数据源与候选混合配比：

**用户提示词**：*我想做一道甜菜羊乳酪沙拉。搭配什么香草比较合适，什么香草不合适？*

**模型回复**：*这里为您详细分析适合与不适合甜菜羊奶酪沙拉的香草搭配，结合了甜菜的泥土甜香与奶酪的浓郁微酸……*

> **核心挑战**：对于这种没有绝对标准答案的开放式生成，如何进行客观公正的打分？

### Chatbot Arena (大模型竞技场 / 盲测众包评估)[https://arxiv.org/abs/2403.04132](https://arxiv.org/abs/2403.04132)

数据收集机制：

1. 真实互联网用户输入任意提示词

2. 平台分派两台完全匿名的模型同时生成回答

3. 用户根据回答质量盲测投票选出胜者（或平局）

<img src="images/arena-beets.png" width="700" />

基于成对比较计算全局 ELO 积分：

- 建立 Bradley-Terry 概率模型：$P(A \text{ 胜过 } B) = \frac{1}{1 + 10^{(ELO_B - ELO_A)/400}}$

- 采用最大似然估计拟合数百万场对战记录，解算各大模型的全局 ELO 竞技天梯积分

[Arena AI (formerly Chatbot Arena)](https://arena.ai/leaderboard)

<img src="images/lmarena-leaderboard.png" width="400" />

Chatbot Arena 的优势与局限：

- **高生态真实性**：完全来自真实大众的使用需求（因为对用户免费且体验好）。

- **人群与偏见风险**：大众用户专业度差异大，容易受到主观偏见和恶意刷票影响。

- **风格与事实混淆**：人类倾向于给排版更漂亮、回答更冗长自信的回复投票，即便其包含事实错误（Sycophancy 谄媚现象）。

- 人类评审员如何判断复杂事实的准确性？模型是否容易通过迎合用户偏见来骗取选票？

- **动态自适应**：无需所有模型回答完全相同的题库，随时间灵活纳入新模型与新提示词。

- 保持持续更新，天然具备抵抗数据污染的能力。

**AlpacaEval (以 LLM 作为裁判的自动化评估, 2023)**[leaderboard](https://tatsu-lab.github.io/alpaca_eval/)

- 包含 805 条精选的多样化指令集

> 💡 **核心概念澄清：大模型语境下的“指令集 (Instruction Set)”到底是什么？**：
> 1. **与 CPU 体系结构指令集（x86/ARM/RISC-V）毫无关系**：
>    - 这里的“指令 (Instruction)”并非底层硬件的机器码指令，而是**自然语言形式的“用户提示词 / 任务请求 (User Prompts)”**。所谓的“指令集”，本质上就是一个**包含 805 道真实开放式人机交互任务的标准测试题库**。
> 2. **这 805 条“指令”长什么样？（典型测试示例）**：
>    - **创意写作与构想**：*“帮我想 5 个以深海探索为背景的悬疑科幻小说故事大纲。”*
>    - **日常开放咨询**：*“我想做一道甜菜羊乳酪沙拉，搭配什么香草最合适，什么香草千万不要放？”*
>    - **实用改写与提炼**：*“将以下这封措辞生硬的催款邮件改写得既礼貌客气、又态度明确严肃……”*
>    - **基础代码与推理**：*“写一个 Python 函数，检查输入字符串是否是回文串，要求忽略大小写与标点符号。”*
> 3. **评测运作机制 (LLM-as-a-Judge)**：
>    - 将这 805 条指令分别喂给“待测模型”和“基准模型 (如 GPT-4)”各自生成长文本回复；
>    - 再将双方的回复匿名、打乱顺序后提交给裁判大模型（如 GPT-4-Turbo）进行双盲裁决（选出哪边回复更优质），最终汇总统计出待测模型的**胜率 (Win Rate %)**。

- 指标：以强大的大模型作为裁判，计算各被测模型相对于基线模型的胜率 (Win Rate)

- **长度偏差 (Length Bias)**：LLM 裁判严重偏好更长的冗长回复，导致榜单被刻意刷分。

- Alpaca Eval 2.0 used regression to debias the metric[https://arxiv.org/pdf/2404.04475](https://arxiv.org/pdf/2404.04475)

- 我们如何评估这一评估指标本身的科学性？

- 检验标准：与 Chatbot Arena 真实人类盲测排名的相关系数极高：

<img src="https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png" width="500" />

<img src="images/alpacaeval-leaderboard.png" width="400" />

### WildBench (真实多轮对话综合基准)[https://arxiv.org/pdf/2406.04770](https://arxiv.org/pdf/2406.04770)

- 从 100 万真实人机对话中提炼出的 1024 个高难度测试样本

- 引入结构化核对清单 (Checklist) 指导裁判大模型逐步打分，大幅提升裁决可靠性

- 与 Chatbot Arena 人类盲测榜单呈现出极强的正相关性

<img src="images/wildbench.png" width="700" />

[HELM WildBench for visualizing predictions](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench)



### 过滤阶段小结

- **核心挑战**：如何对开放式生成进行客观、低方差的评估？

- **成对对比**：成对对比相比单点绝对打分能够提供高得多的判别信号

- **警惕偏见**：时刻防范来自人类或 LLM 裁判的长度偏见与自我偏好

- **核对准则**：制定详尽的评分细则与清单 (Rubric/Checklist) 是提升评估一致性的关键



## 3. 智能体基准 (Agentic Benchmarks)

从评估模型**说了什么**转向评估模型**在真实环境中做了什么**：

> **智能体 (Agent)** = 底座语言模型 (LLM) + 智能体脚手架系统 (Agent Scaffold / 编排控制逻辑)

考核需在真实环境中调用工具（终端命令、文件读写、代码调试）并长时间多轮迭代的复杂任务：

### SWE-bench (真实软件工程修复基准)[https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)

- 来自 12 个大型流行开源 Python 仓库的 2294 个真实 GitHub Issue 与代码补丁

- 任务：智能体自主阅读代码、复现 Bug、修改代码并提交有效 PR

- 验证标准：通过仓库原本自带的完整单元回归测试

<img src="images/swebench.png" width="800" />

[https://llm-stats.com/benchmarks/swe-bench-verified](https://llm-stats.com/benchmarks/swe-bench-verified)

### Terminal-Bench (通用计算机终端任务基准)[https://arxiv.org/abs/2601.11868](https://arxiv.org/abs/2601.11868)[website](https://www.tbench.ai/)

<img src="images/terminal-bench.png" width="700" />

- 基于纯粹的 Linux 终端环境：最通用的智能体数字交互界面

- 涵盖环境配置、故障诊断、管线搭建等真实系统工程运维任务

> 💡 **概念解析：系统运维中的“管线搭建 (Pipeline Construction)”指什么？**：
> - **核心含义**：指通过脚本或配置文件，将多个相互依赖、按工序顺序流转的任务串联起来的**端到端自动化流水线**（前一工序的输出作为后一工序的输入）。
> - **典型应用场景**：
>   1. **DevOps / CI-CD 流水线**：编写 GitHub Actions/GitLab CI 配置或自动化 Bash 脚本，自动串联“拉取依赖 $\to$ 静态代码检查 $\to$ 容器打包 $\to$ 运行回归测试 $\to$ 部署服务”。
>   2. **数据与 ML 处理流**：搭建自动化脚本，定时执行“原始脏数据抽取 $\to$ 清洗与去重 $\to$ 分词打包 $\to$ 送入模型训练”。
>   3. **Linux 命令行管道**：熟练利用 Shell 管道符（`|`）与工具链（grep, awk, sed, sort）组合实现多阶段流式文本过滤与系统指标监控。

<img src="images/terminal-bench-human-time.png" width="600" />

<img src="images/terminal-bench-results.png" width="600" />

[https://llm-stats.com/benchmarks/terminal-bench](https://llm-stats.com/benchmarks/terminal-bench)

### CyBench (网络安全夺旗攻防基准)[https://arxiv.org/abs/2408.08926](https://arxiv.org/abs/2408.08926)

<img src="images/cybench.png" width="700" />

- 包含 40 项专业的网络安全夺旗赛 (CTF) 攻防实战挑战

- 以初次攻破用时作为能力衡量指标

<img src="images/cybench-agent.png" width="700" />

<img src="images/cybench-results.png" width="600" />

[https://llm-stats.com/benchmarks/cybench](https://llm-stats.com/benchmarks/cybench)

### MLE-bench (机器学习工程实战基准)[https://arxiv.org/abs/2410.07095](https://arxiv.org/abs/2410.07095)

- 涵盖 75 项真实的 Kaggle 竞赛（从数据预处理到特征工程与模型微调）

<img src="images/mlebench.png" width="800" />

<img src="images/mlebench-results.png" width="700" />

Agent scaffolds [相关帖子](https://www.philschmid.de/agents-2.0-deep-agents)

<img src="https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png" width="400" />

- **显式规划 (Explicit Planning)**：维护动态任务清单，步步为营推进并勾选确认

- **分层委托 (Hierarchical Delegation)**：主智能体按职责调度子智能体协作（保持上下文整洁）

- **持久化记忆 (Persistent Memory)**：利用工作区文件系统沉淀中间状态与长期记忆

- **上下文工程 (Context Engineering)**：针对复杂执行流注入严密的规范与防幻觉指令

> 💡 **评测视角深度解析：为什么“模型评估”课要强调这四条工程技巧？**：
> 1. **避免变量混淆（测“模型本身”还是测“工程包装”？）**：
>    - 智能体公式表明 $\text{Agent} = \text{LLM} + \text{Scaffold}$。实验表明，保持底层模型不变，仅靠引入上述四条脚手架工程优化，在 SWE-bench 等长程任务上的得分就能产生 20%~50% 的断崖式暴涨。因此，智能体评测必须严格界定**测试对象是纯底座模型能力，还是整套工程系统的协同表现**。
> 2. **榜单规范与反刷榜机制**：
>    - 为防止团队用极其冗余的外部工程规则、重试机制暴力注水刷榜，正规 Agent 榜单（如 SWE-bench Verified）强制要求开源脚手架代码，并严格限制单任务的 API 调用次数与 Token 预算。
> 3. **评测驱动迭代（Eval-driven Development）的产物**：
>    - 这四条工程原则并非凭空发明，而是研究者在深入审计上百步复杂长任务的评测失败日志（Error Traces）后沉淀出的**核心对策**：显式规划解决“长程任务目标漂移”，分层委托解决“全日志塞入导致上下文爆炸与遗忘”，持久化记忆解决“长窗口记忆失真”，上下文工程解决“执行死循环”。



### 过滤阶段小结

- **能力拓展**：智能体架构极大地拓展了语言模型在物理与数字世界中的行动边界

- **脚手架工程至关重要**：脚手架设计的优劣直接决定了智能体在复杂长程任务中的成败

- **综合评估**：评估智能体 = 同时评估底层模型与上层脚手架的系统协同表现



## 4. 纯逻辑推理基准 (Pure Reasoning Benchmarks)

我们能否将纯粹的**逻辑推理 (Reasoning)** 能力与庞大的百科记忆知识剥离开来？

纯逻辑推理更能反映智能的本质（证明模型不仅仅是在死记硬背事实）。

### ARC-AGI (抽象推理与 AGI 挑战基准)[website](https://arcprize.org/arc-agi)

- 对普通人类而言 100% 简单可解，但对传统 AI 系统极具挑战。

- 每一个网格几何变换任务都是全新生成的视觉逻辑小游戏，单纯背诵语料毫无用处。

> 💡 **机制拆解：ARC-AGI 的评测输入、模型输出与判定方式**：
> 1. **底层数据格式（纯文本/符号矩阵，非图像渲染）**：
>    - 表面上是彩色拼图，底层本质是 **$1\times 1$ 至 $30\times 30$ 的二维整数矩阵（2D Integer Array）**。0~9 的十个整数分别映射一种颜色（如 `0: 黑色背景, 1: 蓝色, 2: 红色, 3: 绿色...`）。
> 2. **输入 Prompt 结构**：
>    - 评测框架以 JSON 或带空格换行的 ASCII 矩阵文本形式构造 Prompt。先提供 2~5 组已变换好的演示样本（`Demonstration: Input -> Output`），最后给出未见过的新样本 `Test Input`，引导模型归纳出其背后的抽象规律。
> 3. **模型输出形式**：
>    - **直接生成矩阵 (CoT + JSON)**：模型通过思维链逐步推理规律，最终输出一个目标二维整数数组。
>    - **程序合成 (Program Synthesis，高分打法)**：让模型生成一段 Python 变换代码（如 `def transform(grid): ...`），若代码在演示样例上全部通过，则运行该代码生成测试输出。
> 4. **结果判定（零容错精确匹配 Exact Match）**：
>    - 判定极其严苛：尺寸必须匹配且**所有网格单元的数值必须 100% 完全相同**（错 1 个格子即判 0 分，无过程分）。通常允许模型提交最多 2 个候选答案（Pass@2），任意一个命中即算通过。
> 5. **完整端到端测试示例（抽象规律：水平镜像翻转 + 红色点 2 变为绿色点 3）**：
>    - **真实输入 Prompt（发送给大模型的完整文本）**：
>      ```text
>      You are an expert puzzle solver in the Abstraction and Reasoning Corpus (ARC).
>      Each puzzle consists of 2D grids of integers (0-9) representing different colors:
>      0: black, 1: blue, 2: red, 3: green, 4: yellow, 5: grey, 6: magenta, 7: orange, 8: teal, 9: brown.
>      Infer the transformation rule from demonstrations, then predict the output for the test input.
> 
>      ==================== DEMONSTRATION 1 ====================
>      Input:
>      0 2 0
>      0 0 2
>      0 0 0
> 
>      Output:
>      0 3 0
>      3 0 0
>      0 0 0
> 
>      ==================== DEMONSTRATION 2 ====================
>      Input:
>      2 0 0
>      0 2 0
>      0 0 2
> 
>      Output:
>      0 0 3
>      0 3 0
>      3 0 0
> 
>      ==================== TEST PROBLEM ====================
>      Input:
>      2 2 0
>      0 2 0
>      0 0 0
> 
>      Please think step by step to determine the rule, and provide your final output grid enclosed in ```json and ``` code blocks.
>      Output:
>      ```
>    - **模型实际回复（思维链 CoT + 最终预测结果）**：
>      ````text
>      ### Reasoning:
>      1. In Demo 1, red (2) turns into green (3), and the point at row 1, col 2 moves to col 0 (horizontal flip).
>      2. Demo 2 confirms both rules: all 2s become 3s, and each row is horizontally mirrored/reversed.
>      3. For Test Input:
>         - Row 0: [2, 2, 0] -> change to 3 -> [3, 3, 0] -> horizontally reverse -> [0, 3, 3]
>         - Row 1: [0, 2, 0] -> change to 3 -> [0, 3, 0] -> horizontally reverse -> [0, 3, 0]
>         - Row 2: [0, 0, 0] -> horizontally reverse -> [0, 0, 0]
> 
>      ```json
>      [
>        [0, 3, 3],
>        [0, 3, 0],
>        [0, 0, 0]
>      ]
>      ```
>      ````
>    - **评测判分（Exact Match）**：评测程序从输出中提取 `json` 块中的二维数组，与真实标签 `[[0, 3, 3], [0, 3, 0], [0, 0, 0]]` 逐格比对，完全吻合则计 1 分，有任何数字不同则计 0 分。

- **ARC-AGI-1 (2019)**：初代几何网格推理挑战

<img src="https://arcprize.org/media/images/arc-task-grids.jpg" width="800" />

- **ARC-AGI-2 (2025)**：强化多步组合推理与抽象空间变换

<img src="https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png" width="800" />

<img src="images/arc-agi-results.png" width="700" />

- 传统预训练语言模型（即便参数量巨大）在此类任务上几乎毫无建树

- **突破**：具备深度长思维链强化学习推理的模型（如 o1, o3, DeepSeek-R1）使 ARC 得分产生质的飞跃

- ARC-AGI-3 (March 2026): interactive environments [相关帖子](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)

<img src="images/arc-agi-3.png" width="300" />

<img src="images/arc-agi-3-results.png" width="500" />



### 过滤阶段小结

- **目标**：将推理与知识剥离（极具学术挑战！）

- **范围限定**：限制在人类日常认知推理范畴内（非超人类超维数学问题）

- **暴露断层**：清晰暴露出当前大模型在系统 2 思考模式下的短板



<img src="images/crash-test-rating.jpeg" width="400" />



## 5. 安全性评测 (Safety Benchmarks)

### HarmBench (有害行为自动化安全基准)[https://arxiv.org/abs/2402.04249](https://arxiv.org/abs/2402.04249)

- 涵盖 510 项违反法律伦理与公序良俗的恶意行为指令

[HarmBench on HELM](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench)

[Example of safety failure](https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4)

### AIR-Bench (基于法规与治理框架的安全评测)[https://arxiv.org/abs/2407.17436](https://arxiv.org/abs/2407.17436)

- 基于全球监管政策与企业风控红线构建，细分为 314 个风险类别与 5694 条测试提示

- 细致分类为 314 个风险子类别，包含 5694 条专业攻击测试提示词

<img src="https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png" width="800" />

[HELM AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard)



### 越狱攻击与防护 (Jailbreaking)

- 经过对齐训练的模型学会了主动拒绝有害指令。

- **GCG 越狱攻击**：通过贪心坐标梯度优化自动生成对抗性后缀，诱导模型绕过安全护栏[https://arxiv.org/pdf/2307.15043](https://arxiv.org/pdf/2307.15043)

- **攻击可迁移性**：在开源模型（如 Llama）上生成的对抗提示能够成功迁移攻破闭源商业模型（如 GPT-4）。

<img src="images/gcg-examples.png" width="800" />

什么是真正的 AI 安全？

- 安全具有高度的**情境相关性**（政治、法律、文化习惯在不同地区差异巨大）。

- 风险具有多元性（幻觉、谄媚、诱导犯罪、偏见以及削弱人类批判性思维等）。

**双重用途 (Dual-Use) 困境**：顶尖的网络安全智能体既可以用于合法的安全防御渗透测试，也可以被武器化用于恶意黑客入侵。

> 💡 **深度解析：安全评测测试哲学、双重用途平衡与“四层安全护栏”架构**：
> 1. **评测测试哲学（主动提醒 vs. 恶意诱导）**：
>    - 安全基准（HarmBench / AIR-Bench）**绝不主动提醒模型“这是恶意命令请拒绝”**。现实中的攻击者不会自报家门，因此评测采用纯粹的“红蓝对抗”视角：直接向模型发出未经伪装的直球恶意请求（如编写病毒代码）或对抗性越狱提示（如角色扮演、GCG 梯度乱码后缀），考核模型能否**仅凭内置权重自发坚决拒答**，以**攻击成功率（Attack Success Rate, ASR，越低越好）**作为核心指标。
> 2. **双重用途困境与过度拒答（Over-Refusal）的平衡**：
>    - 若简单采用“敏感词一刀切”策略，模型会沦为把“kill 进程”都拒绝的“安全傻子”，极大阻碍渗透测试、漏洞防御与医学毒理等合法科研工作。
>    - **现代平衡策略**：
>      - ① **意图与武器化阈值判定**：区分“概念原理与防御修复”（放行）与“具备直接杀伤力的武器化可执行 Payload”（拦截）；
>      - ② **双向指标监控**：评测时同时考核“恶意攻击成功率（ASR）”与“良性请求误杀率（Over-Refusal Rate）”，追求两者皆趋于 0。
> 3. **工业界所谓的“安全护栏 (Safety Guardrails)”到底是什么？（纵深防御体系）**：
>    - 安全护栏并非仅指重新微调的大模型，而是一套**由外向内的四层防御工程架构**：
>      ```mermaid
>      flowchart TD
>          User["用户输入 Prompt"] --> Layer1["第 1 层：输入审查小模型 (Input Guardrail / 快速分类器)"]
>          Layer1 -->|"严重违规 (极端违法/暴力)"| Block1["直接拦截报错 (不消耗大模型算力)"]
>          Layer1 -->|"常规合规 / 需深入判断"| Layer2["第 2 层：系统级提示词规则 (System Prompt / 角色边界)"]
>          Layer2 --> Layer3["第 3 层：模型内生对齐权重 (Inherent Alignment / SFT & RLHF)"]
>          Layer3 --> Resp["模型生成初步候选 Token 流"]
>          Resp --> Layer4["第 4 层：流式输出审查小模型 (Output Guardrail / 实时监测)"]
>          Layer4 -->|"检测出高危可执行代码/敏感泄露"| Block2["掐断输出流并替换为合规说明"]
>          Layer4 -->|"安全合规"| FinalUser["最终呈现给用户"]
>      ```
>      - **第 1 层：输入审查小模型（Input Guardrail，如 Llama Guard / Moderation API）**：毫秒级过滤极端违规 Prompt，未触达主模型即在网关拦截；
>      - **第 2 层：系统级提示词规则（System Prompt）**：服务端注入隐藏防御指令，针对企业认证白名单客户可动态切换为“授权渗透测试助手”上下文；
>      - **第 3 层：内生权重安全对齐（Inherent Alignment via SFT/RLHF）**：将道德底线直接烧录进大模型神经网络内部；
>      - **第 4 层：流式输出审查小模型（Output Guardrail）**：实时监测生成 Token 流，发现吐出敏感数据或高危代码立即掐断。
>    - **权限解耦优势**：面对普通大众与白名单专业安全研究员，厂商无需重复部署天价成本的千亿基座大模型，**仅需在网关层动态放宽外挂审核小模型的阈值并切换 System Prompt 即可实现安全策略分级**。



## 6. 生态有效性与真实性 (Ecological Validity & Realism)

- 标准化考试单选题（如 GPQA）与真实工作流存在显著脱节。

- 竞技场虽然来自真人，但提示词质量与领域分布完全不可控。

### GDPVal (OpenAI GDP 核心行业生产力评测)[https://arxiv.org/pdf/2510.04374](https://arxiv.org/pdf/2510.04374)

- 覆盖占美国 GDP 前 9 大行业的 44 个核心高价值职业

- 任务均由平均拥有 14 年行业实战经验的资深从业专家精心设计

<img src="images/gdpval.png" width="700" />

### MedHELM (真实临床医疗工作流评测)[https://arxiv.org/abs/2505.23802](https://arxiv.org/abs/2505.23802)

- 传统医疗基准主要考查医师资格考试的选择题

- 联合 29 位一线临床医生构建的 121 项真实医疗任务（病历推断、多源诊断综合）

<img src="https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png" width="700" />

[MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard)

### Clio (Anthropic 用户真实交互意图分析)[https://arxiv.org/abs/2412.13678](https://arxiv.org/abs/2412.13678)

- 利用高隐私保护的 LLM 分析千万级真实脱敏用户交互

- 揭示人类在真实生产生活中对大模型的核心诉求分布

<img src="images/clio-table4.png" width="700" />

> ⚠️ **现实困境**：评估的“真实性 (Realism)”与用户的“数据隐私 (Privacy)”往往存在天然的冲突。



## 7. 评估有效性与数据污染 (Validity & Contamination)



### 训练集与测试集重叠（数据污染问题）

- **机器学习第一铁律**：严禁在测试集上进行训练！

- 传统时代：ImageNet、SQuAD 具有清晰严格的 Train/Test 划分。

- 大模型时代：模型在海量全网数据上预训练，且数据清单往往高度保密，极易发生基准题目泄露污染。

- **应对方案 1（模型统计推断）**：利用可交换性检验推断模型是否死记硬背了测试样本

- 利用测试样本在分布中的可交换性统计检验模型是否产生了记忆泄露[https://arxiv.org/pdf/2310.17623](https://arxiv.org/pdf/2310.17623)

<img src="images/contamination-exchangeability.png" width="500" />

> 💡 **深度解析：可交换性检验（Exchangeability Test）如何抓出“死记硬背”？**：
> 1. **前向传播输入的 Prompt 形式是什么？**
>    - **输入整篇长文本（非生成，单次 Forward Pass）**：将测试集的一批样本连续拼接成一个长序列（如 `[Q1 + A1] \n [Q2 + A2] \n [Q3 + A3]...`，甚至在隐藏答案的场景下直接拼接纯问题序列 `Q1 \n Q2 \n Q3...`）。模型执行单次前向传播，计算整个序列的自回归对数似然 $\log P(\text{Doc}) = \sum \log P(x_t \mid x_{<t})$，完全不依赖随机采样。
> 2. **核心判作弊原理（固有顺序 vs. 随机洗牌）**：
>    - **良性模型（具备可交换性）**：未见过题库的模型靠通用知识答题，样本之间在统计上相互独立。改变题目的呈现顺序（`Canonical Order` 固有爬虫顺序 vs. `Shuffled Order` 随机打乱顺序），模型预测的困惑度仅有轻微的双向随机扰动，均值无显著差异。
>    - **被污染模型（跨样本序列记忆泄漏）**：模型在预训练时直接把网页/开源仓库的测试文件当成长文本背诵了下来。当且仅当输入完全吻合网络原始固有顺序时，上文会成为极强的记忆触发器，导致模型对后续题目的预测对数概率发生**单向系统性暴涨（困惑度断崖式暴跌）**；一旦乱序，记忆链条断裂，概率立即跌落。通过**置换检验（Permutation Test）**，该偏差落在钟形分布 5 个标准差之外（$p < 10^{-5}$），铁证如山。
> 3. **该方案的工程适用边界（白盒 vs. 闭源纯黑盒）**：
>    - **失效场景**：该方法严格依赖模型的对数概率（Log-probability）。对于**不开放 `logprobs` 接口的闭源纯黑盒 API（如 Claude）**或算力无法本地部署巨型权重的评测方，此统计方法直接失效。
>    - **黑盒补充手段**：面对纯黑盒，业界通常改用**“金丝雀字符探针 (Canary GUID)”**或**“首句提示背诵整卷 (Prefix Continuation)”**；而终极的零污染解法则是**方案 3 的动态题库（如 LiveCodeBench 抓取最新周赛题）**与**方案 4 私有题库**。

- **应对方案 2（行业披露规范）**：推动厂商在技术报告中主动披露去污染指标与置信区间

- 呼吁各大模型研发机构在技术报告中主动披露详尽的去污染分析报告[https://arxiv.org/abs/2410.08385](https://arxiv.org/abs/2410.08385)

- **应对方案 3（动态题库）**：构建随时间持续抓取新题的动态基准

- LiveCodeBench、UncheatableEval：持续抓取最新编程竞赛与网页新闻作为动态测试集

- 注意：时间戳也并非绝对安全（因网络存在大量陈旧内容的搬运转发）

- **应对方案 4（私有化评测）**：使用完全不公开于公网的企业私有代码库或个人未公开笔记测试困惑度

- 企业使用完全保密的内部私有代码库进行回归评测

- 使用个人未公开的写作与笔记

- 对于测量困惑度而言最简单有效



### 评测集本身的质量缺陷与审计

- Fixed up SWE-Bench to produce SWE-Bench Verified [相关帖子](https://openai.com/index/introducing-swe-bench-verified/)

- 为 GSM8K 等经典基准剔除标注错误，制作“白金版 (Platinum)”高质量子集[https://arxiv.org/abs/2502.03461](https://arxiv.org/abs/2502.03461)

<img src="https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096" width="700" />

<img src="https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096" width="800" />

- 智能体基准漏洞：测试用例覆盖不足，导致极简的无效 Agent 偶然通过[https://arxiv.org/abs/2507.02825](https://arxiv.org/abs/2507.02825)

- Docent: use LLM to inspect agent traces to detect problems [相关帖子](https://transluce.org/introducing-docent)



## 8. 如何看待评估？(核心方法论)

不存在放之四海而皆准的评测，取决于你所服务的具体决策目标：

1. **企业选型采购**：在具体场景（如客服系统）下判断模型 A 与模型 B 谁的综合性价比更高。

2. **学术前沿研究**：衡量模型最本质的原始认知能力与智能边界（如纯逻辑推理）。

3. **政策与合规治理**：系统掌握模型潜在的社会效益与安全隐患。

4. **模型算法迭代**：开发者需要高信噪比的梯度反馈，以指导下一步架构与数据优化。



### 我们究竟在评估什么？

- **传统时代**：评估的是**算法方法 (Methods)**（在完全相同的训练集上公平对比算法创新）。

- **大模型时代**：主要评估的是**最终交付的模型/系统 (Models/Systems)**（厂商可以使用任何算力和私有数据）。

回归方法评估的经典范例：

- **nanoGPT Speedrun**：在固定数据集和计算预算下，比拼谁能以最短时间达到指定的验证损失，极大地激励了基础优化算法的创新！

> 💡 **深度解析：为什么评测目标不同？如何“锁死计算预算”？**：
> 1. **核心本质：评测目标的范式差异**：
>    - **评估“模型制品/系统” (Evaluating Models)**：如 MMLU、Chatbot Arena，关注的是黑盒终态能力，厂商可以用上万张卡以力破巧，第三方评测不管其研发成本。
>    - **评估“底层方法/算法” (Evaluating Methods)**：如 nanoGPT Speedrun 与 MLPerf，**评测对象是代码本身（优化器、CUDA 算子、网络架构、训练策略）**，要求戴着严格的算力镣铐跳舞，比拼单位算力下的极限收敛效率。
> 2. **在评测中如何精确“锁死计算预算 (Fixed Compute Budget)”？**：
>    - **方式 A（标准化硬件与挂钟时间，工业实战）**：严格限制硬件规格（如唯一指定单台 8x A100 机器）和训练数据切片，比拼达到目标验证损失（如 Validation Loss < 3.28）所需的**纯物理挂钟时间 (Wall-clock Time)**。极客们借此将原本 45 分钟的基线压缩到了 3.3 分钟（提速 13 倍）。
>    - **方式 B（纯理论浮点运算量 FLOPs，学术研究）**：根据公式 $C \approx 6ND$ 硬性封顶总运算次数（如锁死上限 $10^{19}$ FLOPs），在此预算内比拼最终能达到的最低 Loss。
> 3. **这种评测只限于大厂内部吗？**：
>    - **完全不是！它反而是开源社区与学术界极其活跃的公开第三方竞技场**：由于硬件基准和数据集完全公开统一，全球极客均可提交代码在标准化云虚拟机上复现并跑分；在此诞生的重大算法突破（如 Keller Jordan 的 Muon 优化器、新式学习率调度与算子融合）随后被各大公司迅速吸收进千亿大模型产线。

<img src="images/karpathy-nanogpt-speedrun.png" width="600" /> [相关帖子](https://x.com/karpathy/status/1846790537262571739)

- 评估**方法**激励学术界追求极致的算法与算子创新；

- 评估**模型系统**为下游产业落地提供了直观的选型参照。

> **无论如何，我们必须在评测之初清晰定义好这场游戏的核心规则！**



## 本讲核心总结 (Takeaways)

- **不存在万能的单一评估基准**：必须根据你试图衡量的具体能力（知识、推理、对话、安全性等）量身定制评估方案。

- **明确评估的游戏规则**：严格区分是评估基础算法方法 (Methods)、独立模型系统 (Models/Systems) 还是复合智能体 (Agents)。

- **三大核心考量维度**：任务难度 (Difficulty)、生态真实性 (Realism) 与评估有效性 (Validity)。



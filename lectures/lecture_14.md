```python
from dataclasses import dataclass
import numpy as np
import itertools
import mmh3

```


# 第 14 讲：数据处理与提纯管线 (Data II: Processing Pipeline)

> 核心议题：从万亿级原始互联网网页到干净的模型输入，如何设计并实现转换 (Transformation)、质量过滤 (Filtering)、模糊去重 (Deduplication) 与数据混合 (Data Mixing)？



### 上讲内容回顾

- **数据溯源**：在线服务 $\rightarrow$ 原始抓取归档 $\rightarrow$ 加工清洗后的规整语料库

- **核心考量**：服务条款限制、知识产权保护与合理使用抗辩



### 本讲核心内容

- **数据核心处理管线**：格式转换 (Transformation)、质量过滤 (Filtering)、模糊去重 (Deduplication)、数据混合 (Data Mixing)

- **中期训练与后训练**：合成数据 (Synthetic Data) 的构建与利用



## 1. 数据转换与提取 (Transformation)

原始数据并非现成的纯文本：它们以复杂的 **HTML 网页**、**PDF 论文/图书** 或 **Git 代码仓库目录树** 的形式存在。

原始文件通常是 HTML 网页结构、arXiv PDF 论文排版或包含多层级子目录的代码仓库。



### HTML 转纯文本 (核心转换任务)

- **主体抽取**：剥离导航栏、页脚版权、广告弹窗与侧边栏无用噪声，精准提取网页正文

- **非纯文本元素**：如何线性化处理复杂多维表格、图片说明与数学公式？

- **信息损失**：将树状 DOM 结构压缩为一维线性文本必然伴随着排版信息的丢失

- **经典解析工具**：`trafilatura`、`resiliparse`、`jusText`、`lynx` 等开源解析库

- **正文提取质量至关重要**：[dclm_2024](https://arxiv.org/abs/2406.11794)

<img src="images/dclm-wet.png" width="300" />

FinePDFs [相关帖子](https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog)

<img src="https://huggingfacefw-finepdfsblog.hf.space/_astro/pdf-description.Cb49jXc6_Z17eX4E.webp" width="600" />

- 数据源：从 Common Crawl 全量网页归档中提取的海量 PDF 文件

- 针对因体积过大被默认截断的 PDF 启动二次完整重抓

- 利用轻量化视觉语言模型 (VLM) 或 Docling 算子进行高精度文档 OCR 与公式表格重构

- 结合后处理过滤掉乱码页、扫描水印与缺失排版的残卷

- 尽最大可能保留双栏排版、分节标题与多级引用结构



## 2. 质量与内容过滤 (Filtering)

### 核心算法问题定义

> **数学抽象**：给定小规模的高质量**目标数据** $T$（如维基百科、高质量教科书）和海量的**原始数据** $R$（如 Common Crawl），设计算法从 $R$ 中筛选出在分布上最接近 $T$ 的优质子集 $T'$。

<img src="images/raw-target-schema.png" width="600" />

三大核心应用场景：

1. **语言识别 (Language ID)**：筛选特定目标语言（如英语或中文），剔除乱码与小语种噪声

2. **质量过滤 (Quality Filtering)**：区分行文流畅的高质量正文与低劣机器生成垃圾内容

3. **毒性过滤 (Toxicity Filtering)**：识别并剔除色情、仇恨言论、极端暴力等有害信息

过滤算法的核心设计诉求：

- **泛化能力**：既要学到 $T$ 的高质量特征，又要避免仅仅死记硬背 $T$ 的字面内容，允许筛选出新颖多样的数据 $T'$

- **极致吞吐速度**：算法必须具备极高的单核吞吐量，以处理几十甚至上百 TB 的海量原始数据 $R$

数据选择算法综述文献：[https://arxiv.org/abs/2402.16827](https://arxiv.org/abs/2402.16827)



### 通用过滤框架与打分函数设计

1. 基于目标集 $T$ 与原始集 $R$ 训练统计或轻量机器学习模型，导出打分函数 $\text{score}(x)$

2. 根据得分 $\text{score}(x)$ 设定阈值或采用概率采样保留优质文档

常用的分类器类型：

- **生成式 N-gram 语言模型 (如 KenLM)**：以文本在高质量集 $T$ 上的困惑度作为打分：$\text{score}(x) = p_T(x)$

- **判别式线性分类器 (如 fastText)**：计算文本属于高质量类的后验概率：$\text{score}(x) = p(T \mid x)$

- 使用方式：根据得分硬性截断 $\text{score}(x) \ge \tau$ 或根据打分进行帕累托随机采样。



### 基于模型的过滤在各大模型中的演进

- **坚持纯规则过滤**（担心模型偏见导致多样性骤降）：C4、Gopher、RefinedWeb、FineWeb、Dolma

- **引入分类模型过滤**（大幅提升预训练效率）：GPT-3、LLaMA、DCLM（*已成为行业主流趋势*）



### 语言识别实战 (Language Identification)

- 目标：精准识别并提取出特定语言的文本

- fastText language identification [相关文章](https://fasttext.cc/docs/en/language-identification.html)

- 开箱即用的预训练线性轻量模型，单 CPU 核心每秒可处理数十万字

- 原生支持 176 种全球语言的精准判别

- 训练集：基于维基百科、Tatoeba 翻译库以及东南欧新闻网的多语言语料训练

- Dolma 数据集仅保留判别为英语概率 $p(\text{English}) \ge 0.5$ 的网页 [dolma_2024](https://arxiv.org/abs/2402.00159)

### OpenMathText 数学语料库 [https://arxiv.org/pdf/2310.06786](https://arxiv.org/pdf/2310.06786)

- 目标：从 Common Crawl 海量网页中挖掘大规模高质量数学专业语料

- 第一阶段（规则筛选）：初筛包含 LaTeX 语法指令与数学符号的网页

- 第二阶段（KenLM 打分）：在数学证明库 ProofPile 上训练 KenLM，剔除困惑度 $> 15000$ 的异常离群值

- 第三阶段（fastText 精准判别）：训练轻量二分类器，设置自适应阈值保留高数学价值文本

- **显著成果**：产出了 147 亿高质量数学 Token，训练出的 1.4B 模型推理能力超越了在 20 倍普通数据上训练的模型！

### GPT-3 质量过滤机制 [https://arxiv.org/pdf/2005.14165](https://arxiv.org/pdf/2005.14165)

- **正例集 (Positives)**：采样自高质量语料库（维基百科、WebText2、精选图书）

- 负例集：普通 Common Crawl 随机抽样

Train linear classifier based on word features [相关文章](https://spark.apache.org/docs/latest/ml-features#tokenizer)

- 采用帕累托分布 (Pareto Distribution) 依据得分进行**软性随机保留**（避免硬阈值导致长尾知识被一刀切）：



```python
def keep_document(score: float) -> bool:
    return np.random.pareto(9) > 1 - score

```


### LLaMA / RedPajama 过滤策略 [https://arxiv.org/pdf/2302.13971](https://arxiv.org/pdf/2302.13971)

- 巧妙构造正例：抓取**被维基百科词条引用作为外部参考来源 (References) 的网页**作为高质量正例集

- 负例集：普通 Common Crawl 随机抽样

- 保留被分类器判定为正例的优质网页

### phi-1 教材级数据过滤与合成 [https://arxiv.org/pdf/2306.11644](https://arxiv.org/pdf/2306.11644)

- **核心哲学**：*“教材即一切”*——用极度高质量、结构清晰的代码与教材语料训练精简小模型 (1.5B)

- 语料构成：GPT-3.5/4 生成的合成编程教学用例 + 严格质量过滤的开源代码



```python
R = "The Stack 中的 Python 代码子集"   # 原始待清洗数据
prompt = "评估该代码文件对于学习基础编程概念的学生的教学价值"
T = "使用 GPT-4 配合此提示词对 R 的 10 万个样本进行分类标注以获得高质量正例"

```


- 利用预训练代码模型的嵌入特征，在 GPT-4 标注的高质量子集上训练随机森林分类器

- 从 ### The Stack (代码预训练语料库) 海量代码中精准筛选出具有高教学价值的代码文件

- **在 HumanEval 代码评测上的惊人效果**：

- 在未过滤的 ### The Stack (代码预训练语料库) 原始 Python 代码上训练：准确率仅为 12.19%

- 在精心过滤的高质量子集上训练：仅用 36K 步准确率即大幅跃升至 **17.68%**！

### Dolma 毒性与有害内容过滤 [dolma_2024](https://arxiv.org/abs/2402.00159)

- 训练数据集：Jigsaw 恶意评论数据集 (2018) [Kaggle 竞赛数据](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

- Project goal: help people have better discussions online [相关文章](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064)

- 标注标签：维基百科讨论页上的 {toxic, severe_toxic, obscene, threat, insult, identity_hate}



### 过滤强度的规模依赖效应 (Scale-Dependent Filtering)

- **不存在全局唯一的最优过滤阈值**：最优阈值高度取决于总计算预算 (FLOPs) 与训练步长

- **大计算量长周期训练**：需要更海量的数据储备，因此需适度放宽阈值，容忍稍低质量的数据以避免过拟合；

- **小计算量短周期训练**：对数据纯度要求极高，应采用严苛阈值，确保模型在有限步内学到最密集的知识。

<img src="images/data-filtering-scale.png" width="800" />



### 过滤阶段小结

- 数据过滤是决定预训练模型质量的分水岭；

- 标准范式：严谨定义高质量目标集 $T$（界定好数据的特征），利用统计与轻量模型外推并清洗原始海量语料 $R$。



## 3. 数据去重 (Deduplication)

互联网上存在两类极为普遍的重复内容：

1. **完全精确重复**（镜像站点、GitHub Fork 仓库）[古腾堡镜像列表](https://www.gutenberg.org/MIRRORS.ALL)

- 模糊近似重复：绝大部分内容相同、仅有少数词汇或格式差异

近似重复的经典现实案例：

- 各大网站大同小异的《服务条款》与开源协议声明（如 [MIT 许可证](https://opensource.org/license/mit)）

- 模板化套话文本（复制粘贴或由机器模板生成）<img src="https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png" width="600" />

- 拷贝粘贴时产生的微小空格与排版差异

例如在 C4 数据集中被完全重复了 **61,036 次** 的商品模板描述：

> *“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding...”*

[example page](https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG)

### 为什么对训练数据去重能显著提升语言模型表现？ [https://arxiv.org/pdf/2107.06499](https://arxiv.org/pdf/2107.06499)

1. **大幅提升训练效率**：剔除无意义的重复 Token，使相同计算预算下模型能学到更多全新知识。

2. **显著降低机械记忆风险**：高频重复是导致模型“死记硬背”并泄露训练集隐私/版权原文的元凶，去重可极大缓解该问题。

去重算法的三大设计要素：

1. **切分粒度 (Item)**：按句子 (Sentence)、按段落 (Paragraph) 还是按整篇文档 (Document) 进行对比？

2. **匹配标准 (Matching)**：精确字符匹配、包含相同子串、还是基于 Jaccard 相似度的模糊重合度？

3. **处理动作 (Action)**：仅保留一份唯一副本，还是将所有含重复的污染文档全部剔除？

> **去重算法的核心工程挑战**：
>
> 两两比较 $N$ 个文档的朴素算法复杂度为 $O(N^2)$。当 $N$ 达到百亿量级时，$O(N^2)$ 在算力上是完全不可行的。**我们必须依赖近线性时间复杂度 $O(N)$ 的高效哈希算法！**

- 去重的本质是文档/片段与全量语料之间的两两相似度比对；

- 必须设计具备近线性时间复杂度 $O(N)$ 的高效算法以支撑海量扩展；



### 哈希函数基础 (Hash Functions)
- 哈希函数 $h$ 将变长数据映射为固定长度的哈希值（整数或定长字符串）

- 哈希值体积远小于原数据，便于极速比对与哈希表检索

- **哈希碰撞 (Collision)**：当 $x \neq y$ 时出现 $h(x) = h(y)$

Tradeoff between efficiency and collision resistance [相关文章](https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed)

- **密码学哈希 (SHA-256)**：极强抗碰撞，计算相对耗时（广泛用于区块链与安全签名）

- **非密码学极速哈希 (DJB2, MurmurHash3, CityHash)**：不强调密码学抗碰撞，但计算速度极致飞快（广泛用于哈希表与去重）

本实验中我们将使用高性能的 `mmh3` (MurmurHash3)：



```python
h = mmh3.hash("hello")

```


### 精确去重 (Exact Deduplication)

1. 处理对象：字符串列表

2. 匹配规则：哈希值完全相同的精确匹配

3. 执行动作：去除重复项，仅保留单个唯一副本



```python
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]
hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)
deduped_items = [next(group) for h, group in hash_items]

```


- **优势**：逻辑简单直观，语义明确，精确度极高；

- **局限**：完全无法识别哪怕只有一个词或空格差异的近似重复；

- 此处代码采用 MapReduce 思想编写（先 Map 哈希再 Reduce 分组），极易在大数据集群上水平扩展并行计算。

**C4 数据集中的去重实验** [https://arxiv.org/pdf/1910.10683v4](https://arxiv.org/pdf/1910.10683v4)

进阶实战：按连续 3 个句子的滑动窗口粒度进行去重

匹配标准：子片段哈希值完全匹配

3. 执行动作：去除重复项，仅保留单个唯一副本

> ⚠️ **截断隐患**：若直接从文档中间硬性剔除连续 3 句的重复片段，会导致剩余上下文语义断裂，因此现代预训练更推荐文档级丢弃或段落级去重。



### 模糊近似集合匹配

### 相似度度量标准



### Jaccard 相似度 (Jaccard Similarity)

**定义**：集合 $A$ 与 $B$ 的 Jaccard 相似度等于交集大小除以并集大小：

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$



```python
A = {"1", "2", "3", "4"}
B = {"1", "2", "3", "5"}
def compute_jaccard(A, B):
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union
jaccard = compute_jaccard(A, B)

```


> **近似重复定义**：当且仅当两篇文档提取的特征集合 Jaccard 相似度 $J(A, B) \ge \tau$（阈值）时，判定两篇文档为**近似重复 (Near Duplicates)**。

**核心算法挑战**：如何在海量百亿级文档中，以近线性时间复杂度 $O(N)$ 找出所有满足阈值的近似重复对？



### MinHash 最小哈希算法

**MinHash 核心性质**：设计随机哈希函数 $h$，使得两个集合的最小哈希值碰撞的概率严格等于其 Jaccard 相似度：

$$P(\min h(A) = \min h(B)) = J(A, B)$$

通常在传统哈希表中，我们期望不同元素尽量映射到不同的哈希值以避免碰撞；

……但在相似度哈希中，我们**恰恰希望碰撞概率严格正比于它们的集合相似度**！



```python
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)

```


### 特征矩阵表示 (Characteristic Matrix)

元素 | 集合 A | 集合 B

1    | 1      | 1

2    | 1      | 1

3    | 1      | 1

4    | 1      | 0

5    | 0      | 1

随机哈希函数 $h$ 本质上对所有元素的全集施加了一次随机置换 (Permutation)。

观察置换后集合 $A$ 中出现的首个元素与集合 $B$ 中出现的首个元素：

并集 $A \cup B$ 中的每一个元素作为首个最小元素的概率完全均等：

- 如果首个出现的元素属于交集 {1, 2, 3}，则 $A$ 的首个元素与 $B$ 的首个元素完全一致（发生碰撞）；

- 如果首个出现的元素属于差集 {4, 5}，则 $A$ 与 $B$ 的首个元素不一致。



```python
n = 100  # Generate this many random hash functions
matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
estimated_jaccard = len([m for m in matches if m]) / len(matches)
assert abs(estimated_jaccard - jaccard) < 0.01

```


有了 MinHash 签名后，单个哈希碰撞只是一个二元随机事件，单次无法直接判断 $J(A, B) > \tau$。



### 局部敏感哈希 (Locality Sensitive Hashing, LSH)[book chapter](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf)

如果仅使用单个 MinHash 函数对文档打哈希：

碰撞概率 $P[A \text{ 与 } B \text{ 碰撞}] = J(A, B)$

虽然平均而言相似度越高的文档越容易碰撞，但随机方差极大（单次掷骰子）；

我们的目标：当 $J(A, B) > \tau$ 时极大概率碰撞，而当 $J(A, B) < \tau$ 时极大概率不碰撞！

我们需要将平缓的线性概率曲线“锐化”成一条陡峭的阶跃 S 型曲线！

**解决方案**：采用 $n$ 个独立的 MinHash 函数构建签名向量；

将签名向量划分为 **$b$ 个带 (Bands)**，每个分带包含 **$r$ 行 (Rows)**，满足 $n = b \times r$。



```python
n = 12      # Number of hash functions
b = 3       # Number of bands
r = 4       # Number of hash functions per band

```


哈希函数分带示意：

第 1 带 (h1~h4)  |  第 2 带 (h5~h8)  |  第 3 带 (h9~h12)

**判定准则 (AND-OR 逻辑)**：只要两篇文档在**某一个分带内全部 $r$ 个哈希值完全相等 (AND)**，就判定它们在全局发生碰撞并捕获为候选重复对 (OR)！

正是这种“带内全与 (AND)、带间取或 (OR)”的组合结构，构筑了陡峭的相似度筛选阈值！

设两文档的真实 Jaccard 相似度为 $s = J(A, B)$，则全局碰撞概率推导如下：



```python
def get_prob_collision(sim, b, r):
    prob_match = sim ** r                        # Probability that a fixed band matches
    prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches
    return prob_collision

```


**Example**



```python
prob_collision = get_prob_collision(sim=0.8, b=5, r=10)

```


<img src="https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png" width="600" />



```python
sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}

```


- **增大 $r$（每个带行数）**：使带内匹配条件更苛刻，S 型曲线向右移动（提高相似度门槛，减少误报）；



```python
probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}

```


- **增大 $b$（分带数量）**：增加匹配机会，S 型曲线向左移动（降低门槛，提高召回率）；



```python
probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}

```


<img src="https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png" width="600" />

工业界典型参数配置：[https://arxiv.org/pdf/2107.06499](https://arxiv.org/pdf/2107.06499): n = 9000, b = 20, r = 450



```python
b = 20
r = 450

```


发生相变的临界相似度阈值（S 曲线拐点）：



```python
threshold = (1 / b) ** (1 / r)

```


临界阈值下单个分带匹配概率：



```python
prob_match = (1 / b)

```


当相似度处于临界阈值 $s = (1/b)^{1/r}$ 时，碰撞概率趋近于常数 $1 - 1/e \approx 0.632$：



```python
prob_collision = 1 - (1 - 1 / b) ** b

```


```python
def billion(x):
    return x * 10**9
def trillion(x):
    return x * 10**12

```


语言模型通常需要在多个异构数据源构成的混合语料上进行预训练：

Datasets in Marin:[token viewer](https://huggingface.co/spaces/marin-community/token-count-viewer)

<img src="images/marin-token-viewer.png" width="800" />

The Pile[the_pile_2020 (EleutherAI)](https://arxiv.org/pdf/2101.00027.pdf)

<img src="https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png" width="600" />

**核心决策问题**：在多个数据源之间，我们应该如何科学设定各自的采样分布概率？

示例数据源与候选混合配比：



```python
sources = {"Wikipedia", "CC", "GitHub"}
p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}  # One possible data mixture

```


### 常见的基础配比策略 (Baselines)

1. **直觉调配 (Vibes)**：工程师凭经验和主观感觉手动微调权重（在行业早期极为普遍）；

2. **均匀采样 (Uniform)**：所有数据源享有完全平等的采样概率；

3. **等比例混合 (Proportional)**：按各数据源的实际 Token 总量成比例混合；

直觉规律：应当给予高质量数据源（维基百科、教材、高质量代码）更高的采样权重。

然而，加权必须注意以下两个关键制约因素：

1. **领域多样性**：必须确保模型在文学、代码、学术论文等互不替代的领域保持知识平衡；

2. **数据量有限与重复轮数**：高质数据源总量有限，加权过大会迫使模型对该数据源遍历多个 Epochs。

第二个考量至关重要且极富技巧性：

示例数据源与候选混合配比：



```python
source_token_counts = {
    "low": trillion(10),  # 10T tokens (abundant)
    "high": billion(10),  # 10B tokens (scarce)
}
p = {"low": 0.5, "high": 0.5}  # Naive data mixture
train_tokens = trillion(1)  # Train for 1T tokens
low_num_epochs = (p["low"] * train_tokens) / source_token_counts["low"]
high_num_epochs = (p["high"] * train_tokens) / source_token_counts["high"]

```


> ⚠️ **过拟合警示**：在小规模高质量数据上重复训练超过 50 个 Epochs 会导致模型死记硬背并严重损害泛化能力！

### UniMax 数据均衡算法 (Google, 2023)[https://arxiv.org/abs/2304.09151](https://arxiv.org/abs/2304.09151)

- **应用场景**：在多语言大模型训练中平衡英语与低资源长尾语种的数据配比；

- **以往方法**：在均匀采样与等比例采样之间取温度系数插值（$p(s) \propto N(s)^\alpha$）；

- **UniMax 核心思想**：尽量均匀采样各语种，但对任意数据源设定严格的**最大重复轮数上限 (Epoch Cap $C$)**；

- **数学约束**：对于所有数据源 $s$，要求 $p(s) \times N_{\text{train}} \le C \times N(s)$。

### 基于回归模型的数据配比优化 (RegMix, 2024-2026)[https://arxiv.org/abs/2407.01492](https://arxiv.org/abs/2407.01492)[https://arxiv.org/pdf/2602.12237](https://arxiv.org/pdf/2602.12237)

<img src="images/regmix.png" width="700" />

1. 在配比概率向量 $p$ 上定义先验分布（如狄利克雷分布 Dirichlet）；

2. 选用回归预测算法（如线性回归、梯度提升决策树 GBDT）；

3. 以数十组小算力模型在验证集上的下游损失作为回归拟合目标；

4. **核心权衡**：小规模算力实验的低成本 vs 跨尺度外推至万卡集群的预测准确度。

<img src="images/data-mixing-methods.png" width="700" />

- **假设 1**：回归拟合模型在最优极值点附近具有足够的预测精度；

- **假设 2**：在小模型上搜索出的最优数据配比能够平滑迁移至全尺寸超大模型。

然而，这里存在一个必须警惕的**规模依赖陷阱**：



```python
source_token_counts = {
    "low": trillion(10),  # 10T tokens (abundant)
    "high": billion(10),  # 10B tokens (scarce)
}

```


- 如果在小算力上仅训练极短步长（如 10B Token），高质量小语料只会被遍历极少轮次；



```python
p = {"low": 0.1, "high": 0.9}  # More mass on high quality data

```


- 但如果直接将此配比迁移到 10T Token 的超大模型上，高质量数据会被重复数十上百轮而导致严重过拟合！

### 模拟轮次缩放算法 (Simulated Epoching, 2025)[https://arxiv.org/pdf/2501.11747](https://arxiv.org/pdf/2501.11747)

- **核心思想**：让小规模算力实验在 Epoch 轮次分布上精确拟合大规模预训练时的状态；

- **具体实现**：将各数据源按相同比例进行降采样，使小模型在小 Token 预算下也能经历与全尺寸大模型相同的重复轮数；



```python
small_run_tokens = billion(10)
large_run_tokens = trillion(1)
ratio = small_run_tokens / large_run_tokens
downsampled_source_token_counts = {s: count * ratio for s, count in source_token_counts.items()}

```


- 在降采样后的混合语料中，重复轮次过高的数据源会提前暴露过拟合缺陷，从而使回归模型能够精准抑制过度重复；

- 最终求解出的配比权重更加均衡健壮！



```python
p = {"low": 0.7, "high": 0.3}  # More mass on high quality data

```


### 过滤阶段小结

- **核心问题**：如何科学设定维基百科、通用网页、代码等异构数据源的权重？

- ### 基于回归模型的数据配比优化 (RegMix, 2024-2026): estimate mixture → loss at small scale, optimize (analogous to scaling laws)

- **关键考量**：防范高质数据的过度重复与过拟合（解决方案：UniMax 轮数硬截断或 Simulated Epoching 降采样模拟）。



### 强化学习与推理合成数据标准构建流程

1. **构建多样化交互环境**（代码执行终端、数学推导沙盒、网页环境）

2. **定义高质量任务与提示词集合**（涵盖广泛的难度与领域分布）

3. **利用强能力教师模型生成长链条解答轨迹**（并借助执行器进行真值校验）

### OpenThoughts (前沿开源思维链合成语料, 2025)[https://arxiv.org/abs/2506.04178](https://arxiv.org/abs/2506.04178)

- 采用 QwQ-32B 作为教师模型，提炼生成 120 万条高质量复杂推理思维链 (CoT) 轨迹；

- 题库来自 27 个高质量人类与合成数据源（包括 StackExchange、NuminaMath 数学题库、化学与物理竞赛题）；

<img src="images/openthoughts-sources.png" width="500" />

- **多样性采样**：对每个提示词采样 16 条候选解答能大幅提升探索广度与优质解答覆盖率；

- **重要发现**：更庞大的模型不一定是更好的蒸馏教师（QwQ-32B 的思维链表述更适合中小模型学习）；

- 实验观察：直接对最终答案进行过滤并未带来明显性能增益。

- 精炼的高质量数学语料（如 OpenMath-2-Math）在提升模型推理上的效果显著优于庞大但含噪的多样化语料；

<img src="images/openthoughts-pipeline.png" width="600" />

### SWE-smith (自动化软件工程任务合成, 2025)[https://arxiv.org/abs/2504.21798](https://arxiv.org/abs/2504.21798)

<img src="images/swe-smith.png" width="500" />

- 核心机制：给定开源代码库，让大模型自动化在代码中注入精妙 Bug，并合成对应的 Issue 描述与单元测试；

- 在 128 个 GitHub 仓库上自动化产出了 5 万个高质量且可判分的真实软件修复任务；

### SWE-Zero (无沙盒免执行长轨迹蒸馏, 2026)[https://arxiv.org/abs/2604.01496](https://arxiv.org/abs/2604.01496)

- 现实痛点：软件工程任务依赖极其复杂的编译与依赖环境（与纯文本数学题截然不同）；

- 为成千上万个历史仓库搭建独立的 Docker 运行沙盒是巨大的基础设施与算力开销；

- **关键洞察**：顶尖大模型在预训练中已经内化了深刻的代码语义“世界模型”，许多修复无需反复运行即可一次性精准定位；

<img src="images/swezero-noexec.png" width="600" />

核心：顶尖大模型具备对代码执行语义的内在精确建模能力；

- ### SWE-Zero (无沙盒免执行长轨迹蒸馏, 2026): 300K agent trajectories that don't require repository-specific execution

- 覆盖 15 万个真实的 GitHub Pull Request；

- 基于 OpenHands 脚手架，严格剔除未来的 Git 提交以彻底防止 Agent 窥探答案作弊；

<img src="images/swezero-prompt.png" width="600" />

- 从 Qwen3-Coder-480B 强模型中蒸馏并进行严格一致性过滤；

- **SWE-Hero**：精选 1.3 万条高度依赖终端交互与执行反馈的高价值复杂长轨迹；

<img src="images/swezero-results.png" width="700" />

### SWE-rebench (交互式长程软件工程基准, 2025)[https://arxiv.org/pdf/2505.20411](https://arxiv.org/pdf/2505.20411)

- 汇集来自 3400 个 GitHub 仓库的 2.1 万个交互式 Python 真实工程任务；

- 审计了 GitHub Archive 归档中的 45 万个历史 PR；

- 利用前沿代码大模型自动化配置依赖并严密评估 PR 代码补丁的修复质量；

<img src="images/swe-rebench.png" width="600" />

### SWE-ZERO-12M (千万级超大规模智能体轨迹数据集)[data](https://huggingface.co/datasets/AlienKevin/### SWE-ZERO-12M (千万级超大规模智能体轨迹数据集))

- Scale ### SWE-Zero (无沙盒免执行长轨迹蒸馏, 2026) up to 12M agent trajectories

- Used the ### SWE-rebench (交互式长程软件工程基准, 2025)-v2 tasks (32K executable tasks + 120K nonexecutable tasks)

- 实验证明：仅使用 1.7B 超小模型配合 mini-swe-agent 脚手架，在此轨迹上训练后即可取得惊人的 50.4 pass@100 得分！

- [Example](https://huggingface.co/datasets/AlienKevin/### SWE-ZERO-12M (千万级超大规模智能体轨迹数据集)/viewer/default/train?row=5&conversation-viewer=0)



### 过滤阶段小结

- - **提示词构建的三种流派**：纯合成 (Fully-Synthetic)、半合成 (Semi-Synthetic，真实代码库 + 合成任务) 与纯真实 (Real GitHub PRs)；

- - **高质量回复来源**：来自具备强推理能力且适合作为教学蒸馏范例的顶尖教师模型；

- - **工程实操痛点**：真实代码执行环境的依赖配置极其繁琐沉重；

- - **成败关键**：需要辅以极度严密的结果真值校验、轨迹去噪与格式清洗。



### 过滤阶段小结

- **过滤策略**：训练轻量级分类器（语言判别、质量评分、毒性检测）来界定“什么是好数据”

- **高效去重**：利用哈希算法（MinHash + LSH）在大规模语料上实现近线性复杂度的模糊近似去重

- **数据配比**：在小算力规模上验证数据混合配方，外推预测并指导全尺寸大模型预训练

- **核心实战**：语言识别、领域分类、毒性过滤与代码清洗

- **后训练对齐**：构建高密度的合成指令与偏好数据

- **工程本质**：数据工程高度依赖对具体领域样本的深入观察、人工抽检与细致迭代。



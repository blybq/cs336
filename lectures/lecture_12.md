# 第 12 课：评估 (Lecture 12: evaluation)

- 到目前为止：我们已经涵盖了训练语言模型的所有内容（架构、训练、系统、缩放）。
- 缺失的部分：你在什么**数据**上进行训练？
- 数据塑造模型行为（代码？多语言？DNA？）。
- 在讨论数据之前，我们需要讨论我们希望模型展现出什么行为。

**评估**：给定一个模型，它有多“**好**”？

```python
from edtrace import text, link, image
from lecture_util import post_link
from references import mmlu_2021
```

```python
def what_is_good():
    text("Evaluation might appear to be a mechanical process:")
    text("1. Define some prompts")
    text("2. Send prompts to a model and get back responses")
    text("3. Compute accuracy")

    text("But actually, evaluation is a deep and important topic...")
    text("...which shapes the development of AI.")

    text("**Core challenge**: <font color=\"red\">abstract construct</font> → <font color=\"blue\">concrete metric</font>")

    text("Maybe a model is good if it does well on benchmarks...")
    link(title="Artificial Analysis", url="https://artificialanalysis.ai/")
    image("images/artificial-analysis.png", width=800)

    text("Maybe a model is good if it does well on benchmarks and is cheap to run...")
    image("images/artificial-analysis-cost.png", width=800)

    text("Maybe a model is good if people prefer its responses...")
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)

    text("Maybe a model is good if people simply choose to use (and pay for) it...")
    link(title="OpenRouter", url="https://openrouter.ai/rankings")
    image("images/openrouter.png", width=600)


def perplexity():
    text("- Recall: that a language model is a probability distribution **p(x)** over sequences of tokens.")
    text("- Perplexity (1/p(D))^(1/|D|) measures whether p assigns high probability to some dataset D.")

    text("- In pre-training, you minimize perplexity on the training set.")
    text("- The obvious thing is to measure perplexity on the test set.")
    text("- This is what people did traditionally in language modeling research.")

    text("Standard datasets:")
    text("- Penn Treebank (WSJ)")
    text("- WikiText-103 (Wikipedia)")
    text("- One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news)")
    text("Classic paradigm: in-distribution evaluation: train on train split and evaluate on test split of some dataset.")
    text("Pure CNNs+LSTMs on the One Billion Word Benchmark (perplexity 51.3 → 30.0) "), link("https://arxiv.org/abs/1602.02410")

    text("GPT-2:")
    text("- Trained on WebText (40GB text, websites linked from Reddit)")
    text("- Zero-shot on standard datasets (**out-of-distribution** evaluation)")
    image("images/gpt2-perplexity.png", width=800)
    text("- Works better on small datasets (PTB) where transfer is helpful, but not larger datasets (1BW)")

    text("Perplexity is all you need (more faith than science):")
    text("- True distribution is t, model is p.")
    text("- Best possible perplexity is H(t) obtained iff p = t.")
    text("- If p = t, then solve all the tasks: p(solution | problem)")
    text("- So by pushing down on perplexity, we will eventually \"reach AGI\".")

    text("Perplexity is maybe more than you need:")
    text("- Example: *Stanford was founded in 1885*")
    text("- Perplexity penalizes prediction on all tokens, some (e.g., *founded*) of which might not be relevant")
    text("- Solution: measure conditional perplexity p(response | prompt)^(1/|response|)")

    text("Some benchmarks are perplexity in disguise:")
    text("- Cloze tasks (fill in the blank): LAMBADA "), link("https://arxiv.org/abs/1606.06031")
    image("images/lambada.png", width=700)
    text("- Multiple choice sentence completion: HellaSwag "), link("https://arxiv.org/pdf/1905.07830")
    image("images/hellaswag.png", width=500)

    # 警告（如果你正在运行一个困惑度排行榜）：
    text("**Warning** (if you're running a perplexity leaderboard):")
    text("- People submit `LM` and you compute `log_prob = LM(test_data)`")
    text("- You need to trust that the probabilities are valid (sum to 1)")
    text("- For downstream tasks, `response = LM(prompt)` and compute accuracy on `response`")

    text("Summary:")
    text("- Perplexity is still used heavily in language model development (smooth scaling laws)")
    text("- Still need benchmarks that capture real-world situations (for the non-believers)...")


def exam_benchmarks():
    text("Exams are a useful way to test language models (as with humans):")
    text("- Have control over the subject and difficulty")
    text("- Design to have unambiguous correct answer, easy to grade")

    text("**Massive Multitask Language Understanding (MMLU)** "), link(mmlu_2021)
    text("- 57 subjects (e.g., math, US history, law, morality), multiple-choice")
    text("- \"collected by graduate and undergraduate students from freely available sources online\"")
    text("- Despite the name, MMLU is really about testing knowledge, not language understanding")
    text("- Evaluated on GPT-3 using few-shot prompting")
    image("images/mmlu.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu")
    link(title="HELM MMLU for visualizing predictions", url="https://crfm.stanford.edu/helm/mmlu/latest/")

    text("**MMLU-Pro** "), link("https://arxiv.org/abs/2406.01574")
    text("- Removed noisy/trivial questions from MMLU")
    text("- Expanded 4 choices to 10 choices")
    text("- Evaluated using chain of thought (gives model more of a chance)")
    text("- Accuracy of models drop by 16% to 33% (not as saturated)")
    image("images/mmlu-pro.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu-pro")
    link(title="HELM MMLU-Pro for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro")

    text("**Graduate-Level Google-Proof Q&A (GPQA)** "), link("https://arxiv.org/abs/2311.12022")
    text("- Questions written by 61 PhD contractors from Upwork")
    image("images/gpqa.png", width=700)
    text("- PhD experts achieve 65% accuracy")
    text("- Non-experts achieve 34% over 30 minutes with access to Google")
    text("- GPT-4 achieves 39%")
    link("https://llm-stats.com/benchmarks/gpqa")
    link(title="HELM GPQA for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa")

    text("**Humanity's Last Exam (HLE)** "), link("https://arxiv.org/abs/2501.14249")
    text("- 2500 questions: multimodal, many subjects, multiple-choice + short-answer")
    image("images/hle-examples.png", width=700)
    text("- Awarded $500K prize pool + co-authorship to question creators")
    text("- Filtered by frontier LLMs, multiple stages of review")
    image("images/hle-pipeline.png", width=700)
    image("images/hle-results.png", width=600)
    link("https://llm-stats.com/benchmarks/hle")

    text("Summary:")
    text("- Trend towards harder questions as models improve and saturate existing benchmarks")
    text("- Multiple-choice format can be as difficult as one wants")
    text("- Does not capture real usage (open-ended, doesn't necessarily exist correct answer)")


def chat_benchmarks():
    text("- So far, we've been evaluating on well-defined multiple-choice tasks.")
    text("- Most people don't ask multiple-choice exam questions to their AI assistant.")
    
    text("Example:")
    text("Prompt: *I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not work well?*")
    text("Response: *Here’s a breakdown of herbs that work well (and some that don’t) in a beet + goat cheese salad, based on how their flavors interact with the sweet-earthiness of beets and the tangy creaminess of goat cheese...")

    text("**Challenge**: how to evaluate an open-ended response?")

    text("**Chatbot Arena** "), link("https://arxiv.org/abs/2403.04132")
    text("Data collection:")
    text("- Random person from the Internet types in prompt")
    text("- They get response from two random (anonymized) models")
    text("- They rate which one is better")
    image("images/arena-beets.png", width=700)
    text("Compute ELO rankings based on pairwise comparisons:")
    text("- Define model: p(A wins against B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))")
    text("- Fit this model to maximize probability of pairwise comparisons")
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)
    text("Properties:")
    text("- Real-world prompts (free for users, incentives to actually use it)")
    text("- But who are these people? biases? spammers?")
    text("- Binary preference but conflates style and correctness")
    text("- How does the human even assess correctness?  Prone to sycophancy?")
    text("- Feature: don't need to feed same prompts to all models (important because human is rating)")
    text("- Dynamic: incorporates new prompts and models over time")

    text("**AlpacaEval** (2023)"), link(title="leaderboard", url="https://tatsu-lab.github.io/alpaca_eval/")
    text("- 805 instructions from various sources")
    text("- Metric: win rate against baseline model (GPT-4 preview) as judged by GPT-4 preview (potential bias?)")
    text("- Problem: LLM judges favor longer responses, resulted in leaderboard gaming")
    text("- Alpaca Eval 2.0 used regression to debias the metric "), link("https://arxiv.org/pdf/2404.04475")
    text("- How do we evaluate the metric?")
    text("- Correlation with Chatbot Arena (humans) is high:")
    image("https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png", width=500)
    image("images/alpacaeval-leaderboard.png", width=400)

    text("**WildBench** "), link("https://arxiv.org/pdf/2406.04770")
    text("- Sourced 1024 examples from 1M human-chatbot conversations")
    text("- Uses GPT-4 turbo as a judge with a checklist (like CoT for judging) + GPT-4 as a judge")
    text("- Well-correlated with Chatbot Arena (seems to be the de facto sanity check)")
    image("images/wildbench.png", width=700)
    link(title="HELM WildBench for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench")

    text("Summary:")
    text("- Challenge: how to evaluate open-ended responses?")
    text("- Pairwise comparisons between similar responses provide higher signal")
    text("- Beware of biases (both from humans and LLM judges)")
    text("- Checklist/rubric improves reliability (regardless of human or LLM judge)")


def agentic_benchmarks():
    text("Previously: evaluate what LMs say (chat)")
    text("Now: evaluate what LMs do (agents)")

    text("Agent = language model + agent scaffold (logic for deciding how to use the LM)")
    
    text("Consider tasks that require tool use (e.g., running code) and iterating over a period of time")

    text("**SWEBench** "), link("https://arxiv.org/abs/2310.06770")
    text("- 2294 tasks across 12 Python repositories")
    text("- Given codebase + issue description, submit a PR")
    text("- Evaluation metric: unit tests")
    image("images/swebench.png", width=800)
    link("https://llm-stats.com/benchmarks/swe-bench-verified")

    text("**TerminalBench** "), link("https://arxiv.org/abs/2601.11868"), link(title="website", url="https://www.tbench.ai/")
    image("images/terminal-bench.png", width=700)
    text("- Computer terminal environments: simple and universal")
    text("- 229 tasks crowdsourced from 93 contributors, 89 tasks constitute Terminal-Bench 2.0")
    image("images/terminal-bench-human-time.png", width=600)
    image("images/terminal-bench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/terminal-bench")

    text("**CyBench** "), link("https://arxiv.org/abs/2408.08926")
    image("images/cybench.png", width=700)
    text("- 40 Capture the Flag (CTF) tasks")
    text("- Use first-solve time as a measure of difficulty")
    image("images/cybench-agent.png", width=700)
    image("images/cybench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/cybench")

    text("**MLEBench** "), link("https://arxiv.org/abs/2410.07095")
    text("- 75 Kaggle competitions (require training models, processing data, etc.)")
    image("images/mlebench.png", width=800)
    image("images/mlebench-results.png", width=700)

    text("Agent scaffolds "), post_link("https://www.philschmid.de/agents-2.0-deep-agents")
    image("https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png", width=400)
    text("- Explicit planning: keep a todo list that gets checked off")
    text("- Hierarchical delegation: agents calling other sub-agents (clean context)")
    text("- Persistent memory: read/write files")
    text("- Extreme context engineering: explicit more instructions on process")

    text("Summary:")
    text("- Agents dramatically enhance the capability surface of language models")
    text("- Agent scaffolds are very important")
    text("- Evaluating agents = evaluating agent scaffold + language model")


def pure_reasoning_benchmarks():
    text("- All of the tasks so far require linguistic and world knowledge.")
    text("- Can we isolate **reasoning** from knowledge?")
    text("- Arguably, reasoning captures a more pure form of intelligence (isn't just about memorizing facts).")

    text("**ARC-AGI** "), link(title="website", url="https://arcprize.org/arc-agi")
    text("- 100\% solvable by humans, but challenging for AI")
    text("- Each task is unique, so memorization doesn't help.")

    text("- ARC-AGI-1 (2019): first iteration")
    image("https://arcprize.org/media/images/arc-task-grids.jpg", width=800)

    text("- ARC-AGI-2 (March 2025): more multi-step reasoning")
    image("https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png", width=800)

    image("images/arc-agi-results.png", width=700)
    text("- Pretrained language models didn't move the needle")
    text("- Reasoning models (o1, o3) started making things take off")

    text("- ARC-AGI-3 (March 2026): interactive environments "), post_link("https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf")
    image("images/arc-agi-3.png", width=300)
    image("images/arc-agi-3-results.png", width=500)

    text("Summary:")
    text("- Goal is to disentangle reasoning from knowledge (difficult to do!)")
    text("- Constrained to human reasoning (not superhuman reasoning)")
    text("- Clearly exposes gaps in current models")


def safety_benchmarks():
    image("https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg", width=400)
    text("What does safety mean for AI?")

    text("**HarmBench** "), link("https://arxiv.org/abs/2402.04249")
    text("- Based on 510 harmful behaviors that violate laws or norms")
    link(title="HarmBench on HELM", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench")
    link(title="Example of safety failure", url="https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4")

    text("**AIR-Bench** "), link("https://arxiv.org/abs/2407.17436")
    text("- Based on regulatory frameworks and company policies")
    text("- Taxonomized into 314 risk categories, 5694 prompts")
    image("https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png", width=800)
    link(title="HELM AIR-Bench", url="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard")

    text("Jailbreaking:")
    text("- Language models are trained to refuse harmful instructions")
    text("- Greedy Coordinate Gradient (GCG) automatically optimizes prompts to bypass safety "), link("https://arxiv.org/pdf/2307.15043")
    text("- Transfers from open-weight models (Llama) to closed models (GPT-4)")
    image("images/gcg-examples.png", width=800)

    text("What is safety?")
    text("- Many aspects of safety are strongly contextual (politics, law, social norms - which vary across countries)")
    text("- Many risks are quite varied (hallucinations, sycophancy, abetting crimes, inequality, losing critical thinking)")

    text("**Dual-use**: capable cybersecurity agents (Mythos) can be used to hack into a system or to do penetration testing")


def realism():
    text("**Ecological validity**: how well does an evaluation capture real-world use?")
    text("- Exam benchmarks (e.g., GPQA) are far away from real-world use.")
    text("- Chatbot Arena prompts are from real people, but distribution is uncontrolled.")

    text("**GDPVal** (OpenAI) "), link("https://arxiv.org/pdf/2510.04374")
    text("- 44 occupations from top 9 sectors according to US GDP")
    text("- Tasks come from professionals with ~14 years of experience")
    image("images/gdpval.png", width=700)

    text("**MedHELM** "), link("https://arxiv.org/abs/2505.23802")
    text("- Previous medical benchmarks were based on standardized exams")
    text("- 121 clinical tasks sourced from 29 clinicians, mixture of private and public datasets")
    image("https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png", width=700)
    link(title="MedHELM", url="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard")

    text("**Clio** (Anthropic) "), link("https://arxiv.org/abs/2412.13678")
    text("- Use language models to analyze real user data")
    text("- Share general patterns of what people are asking")
    image("images/clio-table4.png", width=700)

    text("Unfortunately, realism and privacy are sometimes at odds with each other.")


def validity():
    text("How do we know our evaluations are valid?")

    text("### Train-test overlap")
    text("- Machine learning 101: don't train on your test set")
    text("- Pre-foundation models (ImageNet, SQuAD): well-defined train-test splits")
    text("- Today: train on the Internet and don't tell people about your data")

    text("Route 1: try to infer train-test overlap from model")
    text("- Exploit exchangeability of data points "), link("https://arxiv.org/pdf/2310.17623")
    image("images/contamination-exchangeability.png", width=500)

    text("Route 2: encourage reporting norms (e.g., people report confidence intervals)")
    text("- Model providers should report train-test overlap "), link("https://arxiv.org/abs/2410.08385")

    text("Route 3: use fresh evals")
    text("- LiveCodeBench, UncheatableEval: scrape new webpages")
    text("- Timestamps aren't always safe due to copying either")

    text("Route 4: use private evals")
    text("- Companies use internal code bases that aren't on the Internet")
    text("- Use your personal writings")
    text("- Easiest for perplexity")

    text("### Dataset quality")
    text("- Fixed up SWE-Bench to produce SWE-Bench Verified "), post_link("https://openai.com/index/introducing-swe-bench-verified/")
    text("- Create Platinum versions of benchmarks "), link("https://arxiv.org/abs/2502.03461")
    image("https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096", width=700)
    image("https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096", width=800)
    text("- Problems with agentic benchmarks: insufficient test cases, trivial agent can solve task "), link("https://arxiv.org/abs/2507.02825")
    text("- Docent: use LLM to inspect agent traces to detect problems "), post_link("https://transluce.org/introducing-docent")


def how_to_think_about_evaluation():
    text("### What's the point of evaluation?")
    text("There is no one true evaluation; it depends on what question you're trying to answer.")
    text("1. User or company wants to make a purchase decision (model A or model B) for their use case (e.g., customer service chatbots).")
    text("2. Researchers want to measure the raw capabilities of a model (e.g., intelligence).")
    text("3. We want to understand the benefits + harms of a model (for business and policy reasons).")
    text("4. Model developers want to get feedback to improve the model.")

    text("### What are we evaluating?")
    text("- Pre-foundation models, we evaluated **methods** (standardized train-test splits).")
    text("- Today, we're (mostly) evaluating **models/systems** (anything goes).")

    text("There are some exceptions...")
    text("- nanogpt speedrun: fixed data, compute time to get to a particular validation loss")
    image("images/karpathy-nanogpt-speedrun.png", width=600), post_link("https://x.com/karpathy/status/1846790537262571739")

    text("Evaluating methods encourage algorithmic innovation from researchers.")
    text("Evaluating models/systems is useful for downstream users.")

    text("Either way, we need to define the rules of the game!")
```

## 什么是“好”？ (What is good?)

```python
what_is_good()
```

评估似乎是一个机械的过程：
1. 定义一些提示词
2. 将提示词发送给模型并获取回复
3. 计算准确率

但实际上，评估是一个深奥且重要的课题……
……它塑造了 AI 的发展。

**核心挑战**：<font color="red">抽象概念 (abstract construct)</font> → <font color="blue">具体指标 (concrete metric)</font>

也许一个模型如果能在基准测试上表现良好，它就是好的……
[Artificial Analysis](https://artificialanalysis.ai/)
<img src="images/artificial-analysis.png" width="800" />

也许一个模型如果能在基准测试上表现良好且运行便宜，它就是好的……
<img src="images/artificial-analysis-cost.png" width="800" />

也许一个模型如果人们更喜欢它的回复，它就是好的……
[Arena AI (formerly Chatbot Arena)](https://arena.ai/leaderboard)
<img src="images/lmarena-leaderboard.png" width="400" />

也许一个模型如果人们仅仅选择使用（并付费）它，它就是好的……
[OpenRouter](https://openrouter.ai/rankings)
<img src="images/openrouter.png" width="600" />

## 困惑度 (Perplexity)

```python
perplexity()
```

- 回顾：语言模型是 token 序列上的概率分布 **p(x)**。
- 困惑度 (Perplexity, (1/p(D))^(1/|D|)) 衡量 p 是否给某个数据集 D 分配了高概率。

- 在预训练中，你会最小化训练集上的困惑度。
- 显而易见的方法是衡量测试集上的困惑度。
- 这正是传统语言模型研究中的做法。

标准数据集：
- Penn Treebank (WSJ)
- WikiText-103 (Wikipedia)
- One Billion Word Benchmark (源自机器翻译 WMT11 - EuroParl、联合国、新闻)
经典范式：同分布评估 (in-distribution evaluation)：在某个数据集的训练集上训练，并在其测试集上评估。
One Billion Word Benchmark 上的纯 CNN+LSTM 模型（困惑度 51.3 → 30.0） [https://arxiv.org/abs/1602.02410](https://arxiv.org/abs/1602.02410)

GPT-2:
- 在 WebText 上进行训练（40GB 文本，源自 Reddit 上 high karma 链接的网页）
- 在标准数据集上进行零样本（Zero-shot）评估（**分布外**评估，out-of-distribution evaluation）
<img src="images/gpt2-perplexity.png" width="800" />
- 在迁移学习有帮助的小型数据集（PTB）上表现更好，但在大型数据集（1BW）上表现一般。

困惑度即一切（更多的是信仰而非科学）：
- 真实分布是 t，模型是 p。
- 只有当 p = t 时，才能获得最佳的困惑度 H(t)。
- 如果 p = t，则可以解决所有任务：p(解决方案 | 问题)
- 因此，通过降低困惑度，我们最终将“到达 AGI”。

困惑度也许超出了你的需求：
- 示例：*Stanford was founded in 1885*（斯坦福大学建于 1885 年）
- 困惑度惩罚了对所有 token 的预测，其中一些（例如 *founded*）可能并不重要。
- 解决方案：衡量条件困惑度 p(回复 | 提示词)^(1/|回复|)

有些基准测试实际上是伪装的困惑度测试：
- 完形填空任务（填空）：LAMBADA [https://arxiv.org/abs/1606.06031](https://arxiv.org/abs/1606.06031)
<img src="images/lambada.png" width="700" />
- 多选题句子补全：HellaSwag [https://arxiv.org/pdf/1905.07830](https://arxiv.org/pdf/1905.07830)
<img src="images/hellaswag.png" width="500" />

**警告**（如果你正在运营一个困惑度排行榜）：
- 用户提交 `LM`，你计算 `log_prob = LM(test_data)`
- 你需要信任概率是有效的（总和为 1）
- 对于下游任务，`response = LM(prompt)` 并计算 `response` 的准确率

总结：
- 困惑度在语言模型开发中仍然被重度使用（平滑的缩放定律，scaling laws）
- 仍然需要捕捉真实世界情况的基准测试（给那些不相信困惑度信仰的人）……

## 考试基准测试 (Exam benchmarks)

```python
exam_benchmarks()
```

考试是测试语言模型的一种有用方式（就像对人类一样）：
- 对科目和难度有控制权
- 设计为具有无歧义的正确答案，易于评分

**Massive Multitask Language Understanding (MMLU)** [[MMLU 论文]](https://arxiv.org/pdf/2009.03300.pdf)
- 57 个学科（例如数学、美国历史、法律、道德），多选题
- “由研究生和本科生从网上免费获取的资源中收集”
- 尽管名为语言理解，但 MMLU 实际上是关于测试知识，而不是语言理解
- 在 GPT-3 上使用 Few-shot 提示词进行评估
<img src="images/mmlu.png" width="700" />
[MMLU 排行榜](https://llm-stats.com/benchmarks/mmlu)
[用于可视化预测的 HELM MMLU](https://crfm.stanford.edu/helm/mmlu/latest/)

**MMLU-Pro** [https://arxiv.org/abs/2406.01574](https://arxiv.org/abs/2406.01574)
- 移除了 MMLU 中的噪点/琐碎问题
- 将 4 个选项扩展到了 10 个选项
- 使用思维链 (Chain of Thought, CoT) 进行评估（给模型更多的思考机会）
- 模型的准确率下降了 16% 到 33%（没有那么快饱和）
<img src="images/mmlu-pro.png" width="700" />
[MMLU-Pro 排行榜](https://llm-stats.com/benchmarks/mmlu-pro)
[用于可视化预测的 HELM MMLU-Pro](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro)

**Graduate-Level Google-Proof Q&A (GPQA)** [https://arxiv.org/abs/2311.12022](https://arxiv.org/abs/2311.12022)
- 问题由来自 Upwork 的 61 名博士外包撰写
<img src="images/gpqa.png" width="700" />
- 博士专家达到 65% 的准确率
- 非专家在使用谷歌搜索 30 分钟的情况下达到 34% 的准确率
- GPT-4 达到 39% 的准确率
[GPQA 排行榜](https://llm-stats.com/benchmarks/gpqa)
[用于可视化预测的 HELM GPQA](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)

**Humanity's Last Exam (HLE)** [https://arxiv.org/abs/2501.14249](https://arxiv.org/abs/2501.14249)
- 2500 个问题：多模态、多学科、多选题 + 简答题
<img src="images/hle-examples.png" width="700" />
- 给问题创建者提供了 50 万美元的奖金池 + 共同署名权
- 通过前沿 LLM 过滤，进行多个阶段的审查
<img src="images/hle-pipeline.png" width="700" />
<img src="images/hle-results.png" width="600" />
[HLE 排行榜](https://llm-stats.com/benchmarks/hle)

总结：
- 随着模型的改进和现有基准测试的饱和，趋势是提出更难的问题
- 多选题格式可以设计得任意难
- 无法捕捉真实的使用场景（开放式问题，不一定存在唯一正确答案）

## 聊天基准测试 (Chat benchmarks)

```python
chat_benchmarks()
```

- 到目前为止，我们一直在评估定义明确的多选题任务。
- 大多数人不会向他们的 AI 助手提问多选题。

示例：
提示词：*我想做一份甜菜山羊奶酪沙拉。哪些草药搭配合适，哪些不太合适？*
回复：*这里是一份甜菜 + 山羊奶酪沙拉中合适（和一些不合适）的草药清单，基于它们如何与甜菜的甜美泥土气味以及山羊奶酪的酸甜奶油味互动……*

**挑战**：如何评估一个开放式回复？

**Chatbot Arena** [https://arxiv.org/abs/2403.04132](https://arxiv.org/abs/2403.04132)
数据收集：
- 来自互联网的随机用户输入提示词
- 他们会得到两个随机（匿名）模型的回复
- 他们评分哪一个更好
<img src="images/arena-beets.png" width="700" />
基于成对比较计算 ELO 排名：
- 定义模型：p(A 赢了 B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))
- 拟合该模型以最大化成对比较的概率
[Arena AI (formerly Chatbot Arena)](https://arena.ai/leaderboard)
<img src="images/lmarena-leaderboard.png" width="400" />
特性：
- 真实世界的提示词（对用户免费，有动力去真正使用它）
- 但这些人是谁？偏见？垃圾邮件发送者？
- 二元偏好，但混淆了风格和正确性
- 人类甚至如何评估正确性？容易产生谄媚（sycophancy）？
- 特点：不需要向所有模型提供相同的提示词（这很重要，因为是人类在评分）
- 动态：随着时间的推移加入新的提示词和模型

**AlpacaEval** (2023) [排行榜](https://tatsu-lab.github.io/alpaca_eval/)
- 来自各种来源的 805 条指令
- 指标：由 GPT-4 preview 评判的针对基线模型（GPT-4 preview）的胜率（潜在的偏见？）
- 问题：LLM 裁判更喜欢长回复，导致排行榜被套路（gaming）
- Alpaca Eval 2.0 使用回归来消除指标的偏见 [https://arxiv.org/pdf/2404.04475](https://arxiv.org/pdf/2404.04475)
- 我们如何评估指标本身？
- 与 Chatbot Arena (人类) 的相关性很高：
<img src="https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png" width="500" />
<img src="images/alpacaeval-leaderboard.png" width="400" />

**WildBench** [https://arxiv.org/pdf/2406.04770](https://arxiv.org/pdf/2406.04770)
- 从 100 万次人机对话中筛选出 1024 个示例
- 使用 GPT-4 turbo 作为带有清单（checklist，类似于用于裁判的 CoT）的裁判 + GPT-4 作为裁判
- 与 Chatbot Arena 强相关（似乎是事实上的合理性检查）
<img src="images/wildbench.png" width="700" />
[HELM WildBench 预测可视化](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench)

总结：
- 挑战：如何评估开放式回复？
- 相似回复之间的成对比较能提供更强的信号
- 警惕偏见（来自人类和 LLM 裁判的偏见）
- 清单/标准评估可以提高可靠性（无论人类还是 LLM 裁判）

## 智能体基准测试 (Agentic benchmarks)

```python
agentic_benchmarks()
```

以前：评估语言模型说了什么（聊天）
现在：评估语言模型做了什么（智能体）

智能体 (Agent) = 语言模型 + 智能体脚手架 (Agent Scaffold, 用于决定如何使用 LM 的逻辑)

考虑需要使用工具（例如运行代码）并在一段时间内进行迭代的任务

**SWEBench** [https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
- 跨 12 个 Python 仓库的 2294 个任务
- 给定代码库 + 问题描述，提交一个 PR（拉取请求）
- 评估指标：单元测试
<img src="images/swebench.png" width="800" />
[SWE-Bench Verified 排行榜](https://llm-stats.com/benchmarks/swe-bench-verified)

**TerminalBench** [https://arxiv.org/abs/2601.11868](https://arxiv.org/abs/2601.11868) | [网站](https://www.tbench.ai/)
<img src="images/terminal-bench.png" width="700" />
- 计算机终端环境：简单且通用
- 229 个任务由 93 名贡献者众包完成，其中 89 个任务构成了 Terminal-Bench 2.0
<img src="images/terminal-bench-human-time.png" width="600" />
<img src="images/terminal-bench-results.png" width="600" />
[Terminal-Bench 排行榜](https://llm-stats.com/benchmarks/terminal-bench)

**CyBench** [https://arxiv.org/abs/2408.08926](https://arxiv.org/abs/2408.08926)
<img src="images/cybench.png" width="700" />
- 40 个夺旗赛（CTF）任务
- 使用首次解决时间来衡量难度
<img src="images/cybench-agent.png" width="700" />
<img src="images/cybench-results.png" width="600" />
[CyBench 排行榜](https://llm-stats.com/benchmarks/cybench)

**MLEBench** [https://arxiv.org/abs/2410.07095](https://arxiv.org/abs/2410.07095)
- 75 个 Kaggle 竞赛（需要训练模型、处理数据等）
<img src="images/mlebench.png" width="800" />
<img src="images/mlebench-results.png" width="700" />

智能体脚手架 [post](https://www.philschmid.de/agents-2.0-deep-agents)
<img src="https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png" width="400" />
- 显式规划：保留一个不断被核对的待办事项列表 (todo list)
- 分层授权 (Hierarchical delegation)：智能体调用其他子智能体（保持上下文干净）
- 持久内存：读/写文件
- 极端上下文工程：关于流程的显式指导

总结：
- 智能体极大增强了语言模型的能力范围
- 智能体脚手架（scaffold）非常重要
- 评估智能体 = 评估智能体脚手架 + 语言模型

## 纯推理基准测试 (Pure reasoning benchmarks)

```python
pure_reasoning_benchmarks()
```

- 到目前为止，所有的任务都需要语言和世界知识。
- 我们可以将**推理**与知识分离开来吗？
- 可以说，推理捕捉了更纯粹的智能形式（而不仅仅是记忆事实）。

**ARC-AGI** [website](https://arcprize.org/arc-agi)
- 人类 100% 可解，但对 AI 极具挑战
- 每个任务都是独特的，因此记忆没有用处。

- ARC-AGI-1 (2019)：第一次迭代
<img src="https://arcprize.org/media/images/arc-task-grids.jpg" width="800" />

- ARC-AGI-2 (2025 年 3 月)：更多多步推理
<img src="https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png" width="800" />

<img src="images/arc-agi-results.png" width="700" />
- 预训练语言模型未能取得突破
- 推理模型（o1、o3）开始使曲线起飞

- ARC-AGI-3 (2026 年 3 月)：交互式环境 [post](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
<img src="images/arc-agi-3.png" width="300" />
<img src="images/arc-agi-3-results.png" width="500" />

总结：
- 目标是将推理与知识解耦（这很难做到！）
- 受限于人类推理能力（而非超人推理）
- 清楚地暴露了当前模型的差距

## 安全基准测试 (Safety benchmarks)

```python
safety_benchmarks()
```

<img src="https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg" width="400" />
对 AI 来说安全意味着什么？

**HarmBench** [https://arxiv.org/abs/2402.04249](https://arxiv.org/abs/2402.04249)
- 基于 510 种违反法律或规范的有害行为
[HELM 上的 HarmBench](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench)
[安全失败示例](https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4)

**AIR-Bench** [https://arxiv.org/abs/2407.17436](https://arxiv.org/abs/2407.17436)
- 基于监管框架和公司政策
- 分类为 314 个风险类别，共 5694 个提示词
<img src="https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png" width="800" />
[HELM AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard)

越狱 (Jailbreaking)：
- 语言模型经过训练，会拒绝有害的指令
- 贪婪坐标梯度 (Greedy Coordinate Gradient, GCG) 自动优化提示词以绕过安全限制 [https://arxiv.org/pdf/2307.15043](https://arxiv.org/pdf/2307.15043)
- 可以从开源权重模型（Llama）迁移到闭源模型（GPT-4）
<img src="images/gcg-examples.png" width="800" />

什么是安全？
- 安全的许多方面具有强烈的上下文相关性（政治、法律、社会规范——因国家而异）
- 许多风险截然不同（幻觉、谄媚、协助犯罪、不平等、丧失批判性思维）

**双重用途 (Dual-use)**：功能强大的网络安全智能体（如 Mythos）既可以用于黑客入侵系统，也可以用于做渗透测试。

## 现实性 (Realism)

```python
realism()
```

**生态有效性 (Ecological validity)**：评估能在多大程度上捕获真实世界的使用情况？
- 考试基准测试（如 GPQA）距离真实世界的使用非常遥远。
- Chatbot Arena 的提示词来自真实用户，但其分布不受控制。

**GDPVal** (OpenAI) [https://arxiv.org/pdf/2510.04374](https://arxiv.org/pdf/2510.04374)
- 涵盖根据美国 GDP 排名前 9 大行业的 44 种职业
- 任务来自拥有大约 14 年经验的专业人士
<img src="images/gdpval.png" width="700" />

**MedHELM** [https://arxiv.org/abs/2505.23802](https://arxiv.org/abs/2505.23802)
- 以前的医学基准测试主要基于标准化考试
- 121 个临床任务，来源于 29 位临床医生，混合了私有和公开数据集
<img src="https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png" width="700" />
[MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard)

**Clio** (Anthropic) [https://arxiv.org/abs/2412.13678](https://arxiv.org/abs/2412.13678)
- 使用语言模型来分析真实用户的数据
- 分享人们正在问什么的通用模式
<img src="images/clio-table4.png" width="700" />

不幸的是，真实性（realism）和隐私（privacy）有时是相互矛盾的。

## 有效性 (Validity)

```python
validity()
```

我们如何知道我们的评估是有效的？

### 训练集-测试集重叠 (Train-test overlap)
- 机器学习基础常识：不要在测试集上进行训练
- 前基底模型时代（ImageNet, SQuAD）：定义明确的训练-测试集拆分
- 今天：在整个互联网上训练，并且不透露你的数据

路径 1：尝试从模型中推断训练-测试集重叠
- 利用数据点的可交换性 [https://arxiv.org/pdf/2310.17623](https://arxiv.org/pdf/2310.17623)
<img src="images/contamination-exchangeability.png" width="500" />

路径 2：鼓励报告规范（例如，报告置信区间）
- 模型提供商应该报告训练-测试集重叠情况 [https://arxiv.org/abs/2410.08385](https://arxiv.org/abs/2410.08385)

路径 3：使用新鲜的评估（fresh evals）
- LiveCodeBench, UncheatableEval：抓取新的网页
- 由于存在复制，时间戳并不总是绝对安全的

路径 4：使用私有评估
- 公司使用不在互联网上的内部代码库
- 使用你个人的写作内容
- 对困惑度而言最容易实现

### 数据集质量 (Dataset quality)
- 修改了 SWE-Bench 以产出 SWE-Bench Verified [post](https://openai.com/index/introducing-swe-bench-verified/)
- 创建基准测试的白金（Platinum）版本 [https://arxiv.org/abs/2502.03461](https://arxiv.org/abs/2502.03461)
<img src="https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096" width="700" />
<img src="https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096" width="800" />
- 智能体基准测试的问题：测试用例不足，平凡的智能体也能解决任务 [https://arxiv.org/abs/2507.02825](https://arxiv.org/abs/2507.02825)
- Docent：使用 LLM 检查智能体轨迹以检测问题 [post](https://transluce.org/introducing-docent)

## 如何看待评估 (How to think about evaluation)

```python
how_to_think_about_evaluation()
```

### 评估的目的是什么？
不存在唯一的评估标准；这取决于你想回答什么问题。
1. 用户或公司想要为其使用场景（例如客服聊天机器人）做出购买决策（模型 A 还是模型 B）。
2. 研究人员想要衡量模型的原始能力（例如智能）。
3. 我们想要理解模型的益处与危害（基于商业和政策原因）。
4. 模型开发者想要获得反馈以改进模型。

### 我们在评估什么？
- 在前基底模型时代，我们评估的是**方法**（标准化的训练-测试拆分）。
- 今天，我们（大多）评估的是**模型/系统**（一切皆可）。

有一些例外……
- nanogpt speedrun：固定的数据，计算达到特定验证损失所需的时间
<img src="images/karpathy-nanogpt-speedrun.png" width="600" /> | [post](https://x.com/karpathy/status/1846790537262571739)

评估方法鼓励研究人员进行算法创新。
评估模型/系统对下游用户非常有用。

无论如何，我们需要定义游戏规则！

## 结语 (Takeaways)

- 不存在唯一的评估标准；请根据你想衡量的内容选择评估方式。
- 明确定义游戏规则（方法 vs 模型 vs 智能体）。
- 考量因素：难度、现实性、有效性。

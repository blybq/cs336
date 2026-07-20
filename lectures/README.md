# Stanford CS336 官方讲义与课件（中文转译版）索引

本目录包含了从 Stanford CS336 官方 GitHub 仓库克隆的原始讲义转换而来的中文交互式课件。为了便于自学和逐段运行代码，我们对所有官方文件进行了如下处理：
1. **Python 脚本讲义**（可执行讲义） ── 改写为 **Jupyter Notebook (`.ipynb`)** 文件，将官方的英文解释翻译为中文，作为 Markdown 单元格；将代码块进行合理的逐段切分，并翻译了代码注释，您可以逐个运行单元格来调试和理解代码。
2. **PDF 幻灯片课件** ── 改写为 **Markdown (`.md`)** 文件，提取 PDF 文本并翻译为高质量中文，按 `## 第 X 页 (Page X)` 结构排版，合理保留了公式与核心代码段。

---

## 🗂️ 课件清单与快捷跳转链接

请点击以下链接直接查看或在 Jupyter 环境中打开对应的课件：

### 1. 交互式 Jupyter Notebook 讲义（Python 脚本改写）

| 课件编号 | 讲义链接与标题（支持点击跳转） | 改写内容概要 |
| :--- | :--- | :--- |
| **Lecture 01** | [lecture_01.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_01.ipynb) | 大模型概述与分词技术（BPE 算法手撕与编码解码实战） |
| **Lecture 02** | [lecture_02.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_02.ipynb) | 大模型资源预算计算（张量存储、FLOPs、算力/带宽 Roofline 分析） |
| **Lecture 06** | [lecture_06.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_06.ipynb) | 内核优化与 Triton 编程（CUDA 痛点、Triton 瓦片级矩阵乘法与 Softmax 实战） |
| **Lecture 07** | [lecture_07.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_07.ipynb) | 分布式训练并行策略（DDP、张量并行 Megatron-LM 切分、流水线并行 1F1B 实现） |
| **Lecture 10** | [lecture_10.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_10.ipynb) | 自回归推理加速与内存管理（KV Cache 计算、PagedAttention 模拟、投机采样概率验证） |
| **Lecture 12** | [lecture_12.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_12.ipynb) | 大模型评测方法学（PPL 评估、Benchmark 偏见、污染检测与位置偏见 shuffle 模拟） |
| **Lecture 13** | [lecture_13.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_13.ipynb) | 预训练数据集进化史与 Common Crawl 清洗（HTML 解析与高质量数据判定） |
| **Lecture 14** | [lecture_14.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_14.ipynb) | 实战网页文本质量过滤与去重（fastText 训练、布隆过滤器精确去重、MinHash+LSH 近似去重） |
| **Lecture 17** | [lecture_17.ipynb](file:///home/blybq/code-project/cs336/lectures/lecture_17.ipynb) | 多模态大模型（CLIP 视觉对齐、LLaVA 单视解剖、Qwen-VL 及 Chameleon 架构剖析） |

---

### 2. 中文 Markdown 讲义（PDF 幻灯片改写）

| 课件编号 | 课件链接与标题（支持点击跳转） | 主要技术内容 |
| :--- | :--- | :--- |
| **Lecture 03** | [lecture_03.md](file:///home/blybq/code-project/cs336/lectures/lecture_03.md) | Transformer 架构细节、Pre-LN/Post-LN 比较、RMSNorm 与 SwiGLU 物理意义 |
| **Lecture 04** | [lecture_04.md](file:///home/blybq/code-project/cs336/lectures/lecture_04.md) | 稀疏混合专家模型（MoE）路由设计、门控权重、负载均衡与细粒度专家拓扑 |
| **Lecture 05** | [lecture_05.md](file:///home/blybq/code-project/cs336/lectures/lecture_05.md) | GPU 硬件底层架构、SM与SP核心、SRAM与HBM层级、AllReduce等通信拓扑原理 |
| **Lecture 08** | [lecture_08.md](file:///home/blybq/code-project/cs336/lectures/lecture_08.md) | 激活重计算显存开销分析、分布式并行硬件拓扑、Megatron-LM 并行实战 |
| **Lecture 09** | [lecture_09.md](file:///home/blybq/code-project/cs336/lectures/lecture_09.md) | 计算标度律（Scaling Laws）幂律公式推导、Kaplan与Chinchilla法则演进过程 |
| **Lecture 11** | [lecture_11.md](file:///home/blybq/code-project/cs336/lectures/lecture_11.md) | μP 最大更新参数化公式推导、模型超宽度 LR/初始化权重缩放传递定理 |
| **Lecture 15** | [lecture_15.md](file:///home/blybq/code-project/cs336/lectures/lecture_15.md) | 监督微调（SFT）指令数据集演进、Schulman 幻觉假说、RLHF 强化学习与偏好模型 |
| **Lecture 16** | [lecture_16.md](file:///home/blybq/code-project/cs336/lectures/lecture_16.md) | 后训练 RL 算法演进、PPO 双网络弊端、DPO 无 Reward 数学推导、GRPO 去除 Critic 机制 |

---

## 💡 如何在本地运行讲义代码？

本目录中的 `.ipynb` 文件可以在支持 Jupyter 的 IDE（如 VSCode 或 PyCharm）中直接打开，也可以通过启动本地 Jupyter Server 来运行：

```bash
# 激活您的 Python 虚拟环境，并安装 Jupyter 运行环境
pip install jupyter notebook

# 在 /lectures 目录下启动 Jupyter
jupyter notebook
```

在运行前，请确保您已经完成了第一课和后续课程中的依赖库安装（如 `regex`, `tiktoken`, `edtrace` 等）。祝您学习顺利！

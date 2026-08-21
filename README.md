# 斯坦福 CS336：从头开始构建大模型 (Spring 2025) 课件总览

欢迎使用 **Stanford CS336: Language Modeling from Scratch (2025春季最新课程)** 的精炼课件库！

本课件库基于视频课程的 17 个分 P 字幕进行深度整理，完整剔除了口语废话、重复和闲聊，**100% 保留了核心技术细节、数学公式、硬件参数、算法实现伪代码，以及授课团队（如 Percy Liang 等）对前沿技术（如 DeepSeek-R1、GRPO 等）的启发性点拨与设计权衡（Trade-offs）**。

---

## 📚 课件核心知识模块分布

整个课程的 17 个章节可被归纳为大模型全栈研发的五大支柱：

```
                              ┌────────────────────────────────────────┐
                              │     Stanford CS336: Spring 2025        │
                              └───────────────────┬────────────────────┘
                                                  │
         ┌───────────────────┬────────────────────┼────────────────────┬────────────────────┐
         ▼                   ▼                    ▼                    ▼                    ▼
   【数据工程】         【架构与搭建】       【系统与并行】      【Scaling & 推理】     【后训练与 RL】
  ・ HTML提取清洗      ・ PyTorch手撕大模型 ・ GPU硬件架构与Triton  ・ 深入Chinchilla定律 ・ SFT length-bias
  ・ 困惑度/fastText  ・ 超参数/QK-Norm    ・ TP / PP / DDP 并行  ・ GQA/MLA/PageAttention ・ Bradley-Terry 模型
  ・ MinHash & LSH    ・ 详解 MoE 架构     ・ 3D 并行混合训练    ・ 评估集泄漏与偏差   ・ DPO 偏好对齐/GRPO
```

---

## 🗂️ 课件目录与各章要点索引

以下为 17 篇 Markdown 课件的详细列表，点击文件名即可直接查看对应课件内容：

| 章节编号 | 课件链接与标题 | 核心覆盖知识点摘要 |
| :--- | :--- | :--- |
| **P1** | [01_overview_and_tokenization.md](file:///home/blybq/code-project/cs336/notes/01_overview_and_tokenization.md) | 从零构建理念；BPE 与 WordPiece 算法详解；Token 遗漏与泄露风险 |
| **P2** | [02_pytorch_lm.md](file:///home/blybq/code-project/cs336/notes/02_pytorch_lm.md) | 手撕 PyTorch Transformer；Causal Masking 机制；训练吞吐量（TFLOPs/GPU/s）计算 |
| **P3** | [03_architecture_and_hyperparameters.md](file:///home/blybq/code-project/cs336/notes/03_architecture_and_hyperparameters.md) | QK-Norm 缓解注意力溢出；Z-loss 稳定训练；学习率（Cosine/WSD）调度与最大批大小权衡 |
| **P4** | [04_moe.md](file:///home/blybq/code-project/cs336/notes/04_moe.md) | 稀疏混合专家架构；Top-k 门控机制；辅助损失（Auxiliary Loss）与无辅助损失平衡策略 |
| **P5** | [05_gpu_and_distributed_basics.md](file:///home/blybq/code-project/cs336/notes/05_gpu_and_distributed_basics.md) | GPU 硬件架构（SM/SP、内存层次）；PyTorch DDP 伪代码；AllReduce 通信算子数学恒等式 |
| **P6** | [06_kernel_optimization_and_triton.md](file:///home/blybq/code-project/cs336/notes/06_kernel_optimization_and_triton.md) | 内存受限 vs 计算受限算子；Online Softmax 稳定算法；Triton Tile 级并行机制与内核优化 |
| **P7** | [07_parallelism_strategies.md](file:///home/blybq/code-project/cs336/notes/07_parallelism_strategies.md) | 张量并行（Tensor Parallelism）；流水线并行（Pipeline Parallelism）与 1F1B 调度；三维（3D）并行整合 |
| **P8** | [08_parallel_training_hands_on.md](file:///home/blybq/code-project/cs336/notes/08_parallel_training_hands_on.md) | 重计算（Recomputation）显存权衡；激活检查点（Activation Checkpointing）；Megatron-LM 源码分析 |
| **P9** | [09_scaling_laws.md](file:///home/blybq/code-project/cs336/notes/09_scaling_laws.md) | 计算量/参数量/数据量标度律；Chinchilla 拟合参数（最优配比 20 Tokens/参数）；幂律指数推导 |
| **P10** | [10_model_inference.md](file:///home/blybq/code-project/cs336/notes/10_model_inference.md) | KV Cache 机制与显存计算；PagedAttention 原理；投机采样（Speculative Decoding）数学原理 |
| **P11** | [11_using_scaling_laws.md](file:///home/blybq/code-project/cs336/notes/11_using_scaling_laws.md) | 利用极小模型预测极限效果；μP（最大参数化）超参传递定理；前沿大模型缩放实验方案设计 |
| **P12** | [12_model_evaluation.md](file:///home/blybq/code-project/cs336/notes/12_model_evaluation.md) | 模型评估污染（Data Contamination）；似然评估偏差；多选题位置偏差及鲁棒评估 |
| **P13** | [13_training_data_strategies.md](file:///home/blybq/code-project/cs336/notes/13_training_data_strategies.md) | HTML 标签清理与正文提取；基于规则与基于模型（fastText）的高质量数据过滤器设计 |
| **P14** | [14_data_filtering_and_deduplication.md](file:///home/blybq/code-project/cs336/notes/14_data_filtering_and_deduplication.md) | 精确去重（Bloom Filter 空间计算）；模糊去重（Jaccard 相似度、MinHash、LSH AND-OR 门设计） |
| **P15** | [15_sft_and_rlhf.md](file:///home/blybq/code-project/cs336/notes/15_sft_and_rlhf.md) | SFT 长度偏差与 Schulman 幻觉假说；Bradley-Terry 偏好模型；DPO（直接偏好优化）数学推导 |
| **P16** | [16_rl_algorithms.md](file:///home/blybq/code-project/cs336/notes/16_rl_algorithms.md) | 奖励黑客（Reward Hacking）现象；PPO 算法局限性；Kimi k1.5 与 DeepSeek-R1 前沿 RL 实践细节 |
| **P17** | [17_grpo.md](file:///home/blybq/code-project/cs336/notes/17_grpo.md) | GRPO 算法原理解析；组内相对优势（Group Relative Advantage）公式；强化学习基础设施显存瓶颈 |

---

## 🎯 亮点总结与重要启发点拨 (Heuristics)

*   **构建意识（Building Mindset）**：为了避免研究人员被过度抽象（Leaky Abstractions）所遮蔽，该课程一再强调“理解它的唯一方式是手撕它”。在学习这套课件时，您可以通过底层的算子分析（如 Online Softmax）和并行公式来理解为何在大模型中一个简单的操作（如 Layernorm）也会成为系统的最大瓶颈（P6/P8）。
*   **训练稳定性常数（Stability Tricks）**：现代万亿参数大模型的成功很大程度上依赖于小小的稳定性细节。课件中包含了对 **QK-Norm**、**Transformer Z-loss** 以及 **$\mu$P（Maximal Update Parameterization）** 等公式和原理的还原（P3/P11），这些是在数十上百 GPU 节点训练中防范 Loss Spike 的关键。
*   **强化学习革新（GRPO & DeepSeek-R1）**：RLHF 不再只是旧式的“训练 Reward 模型 + PPO 迭代”。最新的 GRPO（P16/P17）大幅降低了显存开销，并且在长文本生成（Kimi k1.5）、纯规则语言奖励（DeepSeek-R1-Zero）中得到了大规模验证，本课件对此有深入剖析。

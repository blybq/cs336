```python
import torch
import time
import math
import sys
import os
from inspect import isfunction
from typing import Callable
from torch import nn, tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from edtrace import text, image, link
from gpu_util import cuda_if_available
from lecture_util import article_link
```

# CS336: 从头开始构建语言模型 (2026春季)

# 第七讲：分布式并行训练 (Parallelism)

# Lecture 7: parallelism

上一讲主题：单块 GPU 内部的计算与内存并行优化。

本讲主题：跨多块 GPU 和多计算节点的分布式并行训练。

![](images/gpu-node-overview.png)

在这两种情况下，核心的**计算单元** (ALU / Tensor Core) 距离**数据源** (显存/内存) 都显得相当遥远。

核心思想：精心编排计算与传输的重叠，最大限度避免数据传输成为计算瓶颈。

分布式架构中的数据层级关系：

- 单节点、单 GPU 内部：L1 缓存 / 共享内存 (极快)

- 单节点、单 GPU 显存：HBM 显存

- 单节点、多 GPU 之间：NVLink/NVSwitch 通信总线

- 多节点、多 GPU 之间：Infiniband / 以太网网络连接 (最慢)

单 GPU 层面：利用算子融合与 Tiling 机制减少多余的显存读写。

多 GPU 层面：利用参数复制、分片等策略减少节点间的通信开销。

为什么我们需要采用多 GPU 并行？

1. **显存装不下**：随着模型增大，参数、优化器状态、梯度和中间激活值超出了单张 GPU 的显存容量。

2. **算力不够快**：希望联合更多 GPU (获取更大算力 FLOPs)，从而缩短训练时间。

[stdout for this lecture](var/traces/lecture_07_stdout.txt)

### 第一部分：分布式通信与计算的基本构建块

## 准备工作与分布式辅助函数 (Setup & Helpers)

在正式开始分布式并行编程之前，我们需要建立关于分布式计算的核心认知体系：

---
### 💡 核心机制与分布式前置知识体系速查

1. **SPMD 多进程并发执行模型 (Single Program, Multiple Data)**：
   - 分布式训练不是单进程遍历多卡，而是**由操作系统同时启动 $N$ 个独立的 Python 进程，各自绑定并独立控制一张 GPU**。
   - **`world_size`（总卡数）**：参与通信组的 GPU 总数量（如 4 或 8）；
   - **`rank`（卡号 / 进程编号）**：当前进程的唯一工号（取值范围 $0 \le \text{rank} < \text{world\_size}$），其中 `rank=0` 为主节点（Master）。
2. **多进程拉起机制 (`torch.multiprocessing` / `mp.spawn`)**：
   - 通过 `mp.spawn(fn=worker, args=(world_size, ...), nprocs=world_size, join=True)` 一键派生 `nprocs` 个子进程；
   - **传参约定**：`mp.spawn` 会**强制自动将当前子进程的 `rank`（0, 1, 2...）作为第 0 个参数**传给 `worker` 函数。
3. **分布式握手与通信后端 (`dist.init_process_group`)**：
   - 依赖 `MASTER_ADDR` 与 `MASTER_PORT`（通常指向 Rank 0）建立各节点间的握手协商；
   - **后端选择**：GPU 多卡通信必须指定 **`"nccl"`（NVIDIA 专为 NVLink/PCIe 优化的高性能集合通信库）**；纯 CPU 或单机调试可用 `"gloo"`。
4. **集合通信就地修改 (In-place) 与同步栅栏 (`dist.barrier`)**：
   - 大多数分布式通信（如 `dist.all_reduce(tensor)`）直接在原有张量显存上执行**就地修改 (In-place)**，以避免重复分配内存；
   - `dist.barrier()` 为全局同步栅栏，阻塞等待所有卡到达同一代码位置，消除各卡执行时差。

---

```python
class DisableDistributed:
    """
    Context manager that temporarily disables distributed functions (replaces with no-ops).
    This is for when we're tracing the lecture, since we can't trace through
    multiprocessing, so we just want to run the function directly without
    distributed communication.
    """
    # __enter__ 是 Python 自带的魔术方法：在进入 with 代码块时自动调用，用于控制资源的获取与环境设置
    # 此处将 dist 模块下的所有通信函数临时备份，并替换为啥都不干的空函数 (no-ops)，避免单进程追踪报错
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    # __exit__ 是 Python 自带的魔术方法：在离开 with 代码块时自动调用，用于控制资源的回收与环境还原
    # 此处将备份的原生 dist 通信函数 100% 无损还原回去
    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])
```

> 💡 **核心语法与反射机制注释**
>
> 1. **Python 反射“四剑客”速查**：
> - **`getattr(obj, "name", default)`**（等价于 `value = obj.name`）：动态**获取**属性值，若找不到则返回指定的默认值。
> - **`setattr(obj, "name", value)`**（等价于 `obj.name = value`）：动态**修改 / 注入**属性值。
> - **`hasattr(obj, "name")`**（等价于 `"name" in obj.__dict__`）：动态检查对象是否**拥有**某个属性。
> - **`delattr(obj, "name")`**（等价于 `del obj.name`）：动态**删除**某个属性。
>
> 2. **`setattr(dist, name, lambda *args, **kwargs: None)` 的参数解析**：
> - **位置参数划分**：`setattr` 严格接收 **3 个位置参数**：`(目标对象 dist, 属性名字符串 name, 新值对象)`。
> - **匿名函数对象**：第 3 个参数是一个完整的匿名函数对象 `(lambda *args, **kwargs: None)`，符合 `lambda 形参: 返回值表达式` 语法。
> - **万能 Mock 原理**：其中 `*args, **kwargs` 负责无差别打包“吞下”外部调用 `dist.xxx(...)` 时传入的任意位置参数与关键字参数，并统一静默返回 `None`（实现通用万能的空操作 Mock）。

```python
def setup(rank: int, world_size: int):
    """Initializes the distributed environment (called at start of process)."""
    # 设置主节点 Rank 0 的地址与端口以供分布式协商通信
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

```python
def cleanup():
    """Cleans up the distributed environment (called at end of process)."""
    torch.distributed.destroy_process_group()
```

```python
def spawn(func: Callable, world_size: int, *args, **kwargs):
    """
    Launches `world_size` processes that each calls `func` on world_size, args, kwargs.
    Note: if we are being traced (inside edtrace), we just run the function directly without multiprocessing and disable distributed functions.
    """
    if not sys.gettrace():
        # 多进程环境下多卡并发计算的通用流程
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
    else:
        # 当遇到 edtrace 调试追踪时，退化为单卡单进程直行测试
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
```

> 💡 **核心语法与多进程传参机制注释**
>
> 1. **`args = (world_size,) + args + tuple(kwargs.values())` 的组装原理**：
> - **各部分数据结构**：
>   - `(world_size,)`：末尾带有逗号的**单元素元组（Tuple）**（若无逗号则仅为带括号的普通整数）；
>   - `args`：由函数外层 `*args` 自动将位置参数列表打包而成的**元组（Tuple）**；
>   - `tuple(kwargs.values())`：通过 `tuple(...)` 构造函数将字典值视图强制转换而来的**元组（Tuple）**。
> - **加法拼接机制**：三个操作数均为 `tuple` 类型，利用 Python 元组重载的加法运算（`tuple.__add__`）顺次拼接为一个全新组装的扁平元组，以满足 `mp.spawn` 底层只接受元组参数包的硬性规范。
>
> 2. **多卡并发 vs. 单卡直行的形参契约对齐**：
> - **目标函数签名**：课件中所有的分布式入口函数签名均为 `func(rank: int, world_size: int, ...)`。
> - **多进程分支（`if`）**：传参时仅需从 `world_size` 开始包装，无需传递 `rank`。这是因为 **`mp.spawn` 具有固定的自动注入约定**，底层拉起子进程时会自动将当前子进程的卡号（`0, 1, 2...`）强制作为第 0 个参数注入并调用 `func`。
> - **单进程直行分支（`else`）**：在单卡单进程直行或调试（`inside edtrace`）时，脱离了 `mp.spawn` 的调度管理，必须**显式手动指定 `rank=0` 与逻辑总卡数 `world_size`（即 `(0, world_size,)`）**，从而严格防止函数形参发生错位，确保内部按 `world_size` 切分的数据分块数学逻辑依然正常运转。

```python
def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data
```

```python
def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    """Create parameters and put them on the `rank`-th GPU."""
    torch.random.manual_seed(0)  # 设置随机种子以保证实验可复现
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=cuda_if_available(rank)) / math.sqrt(num_outputs))
```

> 💡 **核心概念讨论与注释**
>
> 1. **`get_init_params` 的初始化粒度与工程考量**：
> - **单参数粒度**：该函数不负责整个模型的全量初始化，每次调用仅创建并初始化一个 `(num_inputs, num_outputs)` 的二维权重矩阵。模型包含多少层，就需要显式调用多少次（如列表推导式遍历 `num_layers`）。
> - **随机种子机制**：内部重置 `torch.random.manual_seed(0)` 既保证单卡调试的可复现性，又使得数据并行（DDP）中各 Rank 在本地独立初始化时自动获得完全一致的权重镜像，避免了初始权重的跨卡广播（Broadcast）通信开销。
>
> 2. **Tensor vs. 模型结构与 `nn.Parameter` 本质**：
> - **Tensor 本质**：纯粹的多维数值数组（ndarray），仅存储纯数值数据，内部**不包含**任何前向计算逻辑（无层结构与激活函数）。
> - **为什么仅返回单个 Tensor 即可**：在理解了 `get_init_params` 仅初始化单层参数后，由于该极简 MLP 省略了 Bias，单层仅含一个权重矩阵 $W$，因此该函数自然只需返回一个二维 Tensor 包装成的 `nn.Parameter` 即可。
> - **课件设计**：课件为凸显底层通信机制，未使用 `nn.Module` 封装，而是直接用 Python 列表装载权重 Tensor（`params = [W0, W1, ...]`），并通过外部 `for` 循环与 `@` 矩阵乘法驱动前向传播。
> - **`nn.Parameter` 返回类型**：返回 `torch.nn.parameter.Parameter` 实例，本质是 `torch.Tensor` 的直接子类（默认 `requires_grad=True` 且可被优化器捕获），代表单个权重张量实体（模型的“砖块”），而非整个模型结构（“大楼”）。

```python
def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b
```

```python
def summarize_tensor(tensor: tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"
```

> 💡 **核心语法与工程设计注释**
>
> 1. **`map(str, tensor.shape)` 语法机制**：
> - `str` 在此作为内置函数（可调用对象），将 `tensor.shape` 元组中的各整数维度逐项转换为字符串，配合 `"x".join(...)` 拼接为规则的形状字符串（如 `"128x1024"`）。
>
> 2. **`tensor.view(-1)` 的特殊语义**：
> - **自动推导通配符**：`-1` 在 `view()` 中并非表示倒数第一维，而是让 PyTorch 根据元素总数自动推导维度大小。
> - **彻底展平（Flatten）**：只传入一个 `-1` 相当于将任意高维张量（2D、3D、4D 等）在内存零拷贝的前提下重塑为一维向量，随后安全统一地通过 `[0]` 访问全局第一个元素。
>
> 3. **工程价值：“探针 / 金丝雀”检查（为什么只打印形状与首个浮点数）**：
> - **避免多卡 I/O 阻塞**：多卡/多节点并发打印完整张量会导致严重的控制台刷屏与磁盘 I/O 延迟。
> - **快速核验切分（Sharding）**：第一时间验证各并行策略（数据并行、张量并行、流水线并行）中的维度拆分与拼接是否数学正确。
> - **轻量状态把脉**：通过首个数值的动态变化确认模型是否在正常迭代（权重是否更新）、捕获 `NaN`/`Inf` 数值异常，并核验数据并行（DDP）中各卡权重是否严格同步一致。

```python
def render_duration(duration: float) -> str:
    # 自适应时间格式化：将秒级耗时转换为人类易读的单位 (us / ms / s)
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"
```

**集体通信操作 (Collective Operations)** 是分布式并行编程中最底层的概念基石。 [相关文章](https://en.wikipedia.org/wiki/Collective_operation)

- 这些操作早在 1980 年代的并行机集群设计文献中就已经成为经典。

- **集体 (Collective)** 意味着你需要在一个通信组内的多台设备间指定一种统一的通信拓扑。

- 相比由用户自己维护繁琐的卡对卡点对点通信，集体通信库往往能提供更为极致的网络拓扑性能优化。

**分布式设置**：

![](images/ranks.png)

- **Rank**：标识特定的 GPU 设备编号（例如 0, 1, 2, 3 等）

- **World size**：当前通信组内的 GPU 总卡数（例如 4）

主要的通信操作包括：

- Broadcast (广播)、Scatter (分发)、Gather (收集)、Reduce (规约) 等基础操作

- All-Gather、Reduce-Scatter、All-Reduce 等分布式训练的核心顶梁柱原语

- All-to-All (常用于混合专家模型 MoE 中路由数据)

**Broadcast (广播)**：将 Rank 0 卡上的数据完整复制到所有 Rank 卡上。

```python
rank0 = tensor([0., 1, 2, 3])
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

常见用例：Rank 0 负责从磁盘读取初始化检查点，然后 Broadcast 给其余 worker 同步参数状态。

**Scatter (分发)**：将 Rank 0 卡上的一个大张量按维度均匀切分，并分发到各个 Rank 卡上。

```python
rank0 = tensor([0., 1, 2, 3])
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
```

注：这对于理解 Reduce-Scatter 很有帮助。

**Gather (收集)**：将各个 Rank 卡上的小张量拼接，收集到 Rank 0 上形成一个大张量（Scatter 的反向操作）。

```python
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
rank0 = tensor([0., 1, 2, 3])
```

注：这对于理解 All-Gather 很有帮助。

**Reduce (规约)**：对所有 Rank 卡上的数据对应位置应用某种数学规约操作（如求和、求极值），最后只把结果保存在 Rank 0 上。

```python
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
rank0 = tensor([6.])  # Sum of all ranks (0 + 1 + 2 + 3)
```

注：这对于理解 All-Reduce 很有帮助。

**All-Gather**：在各个 Rank 卡上独立执行 Gather，使得最后所有 Rank 卡都拥有完全拼接后的完整大张量。

```python
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

典型用例：在 ZeRO/FSDP 中，各卡在平时只保存一份参数切片，前向计算时通过 All-Gather 收集并恢复成完整参数。

**Reduce-Scatter**：对数据对应位置执行数学规约，随后将规约结果按 Rank 维度切分分发到各个 Rank 上。

```python
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])
rank0 = tensor([6.])  # Sum along dim 0 (0 + 1 + 2 + 3)
rank1 = tensor([10.]) # Sum along dim 1 (1 + 2 + 3 + 4)
rank2 = tensor([14.]) # Sum along dim 2 (2 + 3 + 4 + 5)
rank3 = tensor([18.]) # Sum along dim 3 (3 + 4 + 5 + 6)
```

典型用例：在反向传播计算出梯度后，通过 Reduce-Scatter 进行梯度均值规约，并让各卡只存储梯度的一部分切片，从而节省显存。

**All-Reduce**：对所有卡上的数据应用数学规约，并将完整规约结果输出到所有卡上（等价于先 Reduce-Scatter 再 All-Gather）。

```python
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])
rank0 = tensor([6., 10, 14, 18])
rank1 = tensor([6., 10, 14, 18])
rank2 = tensor([6., 10, 14, 18])
rank3 = tensor([6., 10, 14, 18])
```

典型用例：在传统数据并行 (DDP) 中，反向传播后通过 All-Reduce 同步各卡的梯度，随后在所有卡上重复更新相同的完整参数。

将 All-Reduce 拆解为 Reduce-Scatter 与 All-Gather，极大促进了 ZeRO/FSDP 等显存友好型数据并行的发展。

**All-to-all**：最通用的多对多通信，每个 Rank 向所有其他 Rank 各自发送特定的张量分片。

```python
rank0 = tensor([0., 1, 2, 3])      # send  0 to rank 0,  1 to rank 1,  2 to rank 2,  3 to rank 3
rank1 = tensor([4., 5, 6, 7])      # send  4 to rank 0,  5 to rank 1,  6 to rank 2,  7 to rank 3
rank2 = tensor([8., 9, 10, 11])    # send  8 to rank 0,  9 to rank 1, 10 to rank 2, 11 to rank 3
rank3 = tensor([12., 13, 14, 15])  # send 12 to rank 0, 13 to rank 1, 14 to rank 2, 15 to rank 3
rank0 = tensor([0, 4, 8, 12])
rank1 = tensor([1, 5, 9, 13])
rank2 = tensor([2, 6, 10, 14])
rank3 = tensor([3, 7, 11, 15])
```

要点说明：

- 它是混合专家模型 (MoE) 的核心通信管道：每张卡持有不同的样本批次，通过 All-to-All 将不同的 Token 路由发送到特定的专家 (Expert) 卡上进行处理。

- 在数据均衡分布时，All-to-All 通信在逻辑上非常类似矩阵的转置。

- 它也能够处理不均衡的数据分片（但通常由于硬件负载考虑，应尽量做到样本均衡分配）。

> 💡 **核心概念澄清：All-to-All 在 MoE 专家路由中的物理本质**
>
> 1. **示例数字代表的是“Token 实体”，而非“向量维度”**：
> - 示例代码中 `rank0 = tensor([0., 1, 2, 3])` 的每个数字（如 `0.` 或 `1.`），抽象代表的是**一个完整的 Token（包含其完整的特征向量 `[hidden_dim]`）**，而绝不是把单个特征向量的各个维度拆开。
> - 在工业级代码中，输入张量形状通常为 `(num_ranks, tokens_per_rank, hidden_dim)`，All-to-All 仅在第 0 维（目标卡号/专家号）进行跨卡多对多交换，每个 Token 的 `[hidden_dim]` 维度作为不可分割的物理整体在卡间完整传输。
>
> 2. **MoE 路由的物理原子单位严格是【单个 Token（Token-level）】，而非整条 Sequence**：
> - **数学本质（Position-wise 逐位置独立）**：在 Transformer 中，Self-Attention 层需要关注序列上下文（无法拆散 Sequence），但 FFN / MoE 层在数学上是逐位置完全独立的（每个 Token 过 MLP 互不干扰）。因此进入 MoE 前会执行 `x.view(-1, hidden_dim)` 彻底打破序列边界，**同一句话中的不同单词会被打散、各自送往最匹配的专家卡**。
> - **工程价值（消除专家负载不均衡 Load Imbalance）**：若按整条 Sequence 路由，遇到长文本与短对话混杂时会导致部分专家显存过载、其余专家严重闲置空转；按单个 Token 细粒度打散，成千上万个 Token 能以近乎完美的均匀度平摊给各专家卡，最大化硬件计算与显存带宽利用率。

如何快速记忆这些名词原语：

- **Reduce**：表示对数据应用结合律/交换律的规约运算（求和、最小值、最大值）。

- **Scatter (分发)** 与 **Gather (收集)** 互为逆操作。

- **All-** 前缀：意味着最终数据接收端是所有参与的计算设备，而网络拓扑效率更高。

经典拓扑（家用/个人工作站环境）：

![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp)

- 同节点内的多张 GPU 通过 PCI(e) 总线完成通信（PCIe 7.0 x16 单向带宽可达 242 GB/s）。 [相关文章](https://en.wikipedia.org/wiki/PCI_Express)

- 跨节点多卡之间使用千兆/万兆以太网进行连接，这往往会带来毁灭性的网络延迟和带宽限制 (~200 MB/s)。

现代拓扑（数据中心高性能集群环境）：

![](images/gpu-node-overview.png)

典型多卡网络架构设计：

- **单节点 8 卡**：通过板载 NVLink 高速总线直接互连到 NVSwitch 交换芯片（B200 对应的 NVLink 5.0 可提供高达 1.8 TB/s 的卡间双向带宽，作为对比，HBM 显存带宽约为 8 TB/s）。

- **单 Pod 内 256 个节点**：由 PCIe 扩展出专用网卡 (Infiniband NIC/HCA)，通过 Infiniband 交换机互连，节点间跨网带宽约可达 ~0.05 TB/s。

- **集群/数据中心多 Pod**：采用常规光纤以太网完成超大规模的连接。

绕过 CPU 参与的数据传输：

- 传统的以太网传输需要操作系统 CPU 的频繁干预（需要多次拷贝数据至内核 Socket 缓冲区，建立 TCP 协议栈并打包，最后发送至网卡发送环缓冲区）。

- **远程直接内存访问 (RDMA)** 机制允许一张 GPU 绕过 CPU 控制直接读取或写入另一台机器上 GPU 的显存空间。

- Infiniband 网络天生完美支持 RDMA；而标准商业以太网往往不支持。

最新技术演进：

- **GB200/GB300 NVL72 柜机**：每盘包含 8 颗 GPU，单个机架放入 9 盘，形成由 72 颗 GPU 直接构成的巨大统一 NVLink 域。

- **RoCE 技术**：在常规以太网上承载 RDMA 流量，相比 Infiniband 成本更低，在 Meta 的超大规模集群中得到了极其广泛的应用。

### NVIDIA 集合通信库 (NCCL)

NCCL 负责将顶层的 Collective 集合通信原语（如 All-Reduce）转化为底层硬件网络的数据包进行传输。[talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/)

- 自动感知系统的底层拓扑（有多少张卡、多少个交换机、走 NVLink 还是走 PCIe）。

- 自动匹配并优化跨卡数据流的最佳路径。

- 直接调度定制化的 GPU CUDA Kernel 负责极速收发数据，免去 CPU 开销。

PyTorch 分布式框架 (`torch.distributed`)[documentation](https://pytorch.org/docs/stable/distributed.html)

- 提供了极为整洁的集合通信 API 接口（例如 `all_gather_into_tensor`）。

- 支持对接多种底层硬件后端：gloo (支持 CPU 分布式通信) 和 nccl (支持 GPU 高速通信)。

- 也封装了例如 FSDP 等的高阶接口（本课程暂不涉及，我们从底层写起）。

让我们来看几个实际运行的例子。

```python
def collective_operations_main(rank: int, world_size: int):
    """
    【函数作用】：集合通信三大核心原语教学演示与数学恒等式验证。
    通过在多卡上分别执行 All-Reduce 以及 (Reduce-Scatter + All-Gather)，直观证明并验证黄金等式：
        All-Reduce == Reduce-Scatter + All-Gather
    
    【参数说明】：
        - rank: 当前进程的卡号 (0 <= rank < world_size)
        - world_size: 通信组内的 GPU 总卡数
    """
    # 步骤 1：初始化分布式进程组与 NCCL 通信拓扑
    setup(rank, world_size)

    # ==========================================
    # 步骤 2：执行 All-Reduce (全规约) 示例
    # ==========================================
    # 【dist.barrier() 核心机制说明】：
    # 1. 对齐而非序列化：确保所有卡在栅栏处“全员到齐”才统一放行；但放行后由 OS 并发调度，无法控制 print 的物理打印顺序。
    # 2. 跨阶段“防火墙 (Phase Divider)”：确保当前阶段所有操作（含 print）彻底完成后才允许进入下一阶段，防止快卡抢跑导致输出串味与通信冲突。
    # 3. 隐式同步特性：集合通信（如 all_reduce）本身需全员握手交换数据，自带强同步属性，因此执行后无需多此一举调用 barrier()。
    dist.barrier()

    # 每张卡持有带 rank 偏移的张量：Rank 0 为 [0,1,2,3], Rank 1 为 [1,2,3,4]...
    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank

    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    # in-place 就地修改：所有卡对应位置元素求和，并将完整结果直接写回各卡的 data 中
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)

    # ==========================================
    # 步骤 3：执行 Reduce-Scatter (规约分发) 示例
    # ==========================================
    dist.barrier()

    # 输入：各卡持有长度为 world_size 的向量
    input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank
    # 输出：每张卡仅预先分配 1 个元素的局部标量显存
    output = torch.empty(1, device=cuda_if_available(rank))

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    # 对多卡输入对应位置求和，随后将第 i 个分块结果只分发给 Rank i
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    # ==========================================
    # 步骤 4：执行 All-Gather (全收集) 示例
    # ==========================================
    dist.barrier()

    # 将刚刚 Reduce-Scatter 得到的局部规约切片，作为 All-Gather 的输入
    input = output
    # 输出：预分配长度为 world_size 的完整向量显存
    output = torch.empty(world_size, device=cuda_if_available(rank))

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    # 所有卡互相收集彼此的局部切片，拼接还原为完整的全局规约向量
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    # 对比步骤 2 与步骤 4 的输出结果，在数值上完全一致！
    text("Indeed, all-reduce = reduce-scatter + all-gather!")

    # 步骤 5：优雅销毁分布式通信组并释放底层资源
    cleanup()
```

```python
# 启动 4 卡进程并发执行 All-Reduce、Reduce-Scatter 与 All-Gather 集合通信
spawn(collective_operations_main, world_size=4)
```

集群中的卡间通信到底能有多快？

```python
def all_reduce(rank: int, world_size: int, num_elements: int):
    """
    【函数作用】：全规约 (All-Reduce) 网络有效吞吐带宽基准测试 (Benchmarking)。
    通过跨卡传输大规模张量（如 100M float32 元素 = 400MB），精确测量通信物理耗时，
    并基于 NCCL 环形/树形通信等效数据传输模型，计算硬件互联总线（NVLink/PCIe）的实测有效带宽（GB/s）。
    
    【参数说明】：
        - rank: 当前进程的卡号
        - world_size: 参与压测的 GPU 总卡数
        - num_elements: 单卡测试张量的元素总数
    """
    setup(rank, world_size)

    # 创建用于压测的大规模显存张量
    data = torch.randn(num_elements, device=cuda_if_available(rank))

    # 步骤 1：通信预热 (Warmup)，消除 CUDA 内核首次启动和 NCCL 内存注册带来的初始化开销
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # 等待 GPU 异步流完成，排空执行队列
    dist.barrier()            # 跨进程同步栅栏，确保所有卡在同一起跑线上准备计时

    # 步骤 2：正式计时并执行 All-Reduce
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    # 【torch.cuda.synchronize() 与 dist.barrier() 的协同机制】：
    # 1. torch.cuda.synchronize() [单卡内部同步]: GPU 是异步执行引擎，CPU 发射完通信指令即返回；
    #    必须调用此函数阻塞本地 CPU 线程，等待本地 GPU 彻底完成底层所有通信与规约 CUDA 内核。
    # 2. dist.barrier() [跨卡集群同步]: 消除网络抖动与慢卡 (Straggler) 导致的停表时差（木桶效应），
    #    确保全集群所有卡在全部通信完毕后同时停表，从而测得准确的端到端集群网络物理耗时。
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # 步骤 3：计算网络有效带宽 (Effective Bandwidth)
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    # 【Ring All-Reduce 核心机理与公式推导】：
    # 1. 环形拓扑广播 (W-1 步)：理想的多卡通信将所有节点连成闭环；每张卡将局部数据广播给其余所有卡，
    #    在环中必须顺次向邻居接力转发 W - 1 次（两阶段合计经历 2 * (W - 1) 步转发）。
    # 2. 对称切分消除单点瓶颈 (切分为 W 份)：为避免主从模式（Parameter Server）下的 Master 网卡打爆瘫痪，
    #    各卡地位完全平等，每个节点只负责局部 1/W 分块的规约求和与保管；因此张量数据必须严格切分为 W 份。
    # 3. 硬件总线全并发利用率：每个传输时钟步，所有卡的 Send/Recv 端口均 100% 满负荷流水线运转，
    #    单卡实际物理吞吐量折合为 2 * size_bytes * (W - 1) / W。
    # 4. 代码计算形式还原：
    #    bandwidth = [2 * size_bytes * (world_size - 1) / world_size] / duration
    #              = sent_bytes / total_duration (将除以 world_size 挪入分母保持整数精度)
    sent_bytes = size_bytes * 2 * (world_size - 1)
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()
```

```python
def reduce_scatter(rank: int, world_size: int, num_elements: int):
    """
    【函数作用】：规约分发 (Reduce-Scatter) 网络有效吞吐带宽基准测试。
    测量大规模张量在只进行规约切分、不执行后续全收集时的传输耗时与有效带宽。
    与 All-Reduce 形成对比：它只传输 1 倍数据，耗费约一半时间，但实测物理带宽（GB/s）高度一致。
    
    【参数说明】：
        - rank: 当前进程的卡号
        - world_size: 参与压测的 GPU 总卡数
        - num_elements: 每个分块包含的元素个数
    """
    setup(rank, world_size)

    # 输入为一个 (world_size, num_elements) 的大矩阵，输出仅为一个切片缓冲区 (num_elements,)
    input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))
    output = torch.empty(num_elements, device=cuda_if_available(rank))

    # 步骤 1：通信预热操作与精准计时对齐
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()
    dist.barrier()

    # 步骤 2：执行 Reduce-Scatter 并统计物理耗时
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # 单卡内部同步：排空本地 GPU 通信流
    dist.barrier()            # 跨卡集群同步：消除慢卡时差，全员统一定格停表
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # 步骤 3：计算有效带宽（无后续 All-Gather 阶段，因此此处无 2x 系数）
    dist.barrier()
    data_bytes = input.element_size() * input.numel()
    sent_bytes = data_bytes * (world_size - 1)
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()
```

```python
# 压测 All-Reduce 与 Reduce-Scatter 的网络有效吞吐带宽
spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)
spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)
```

网络性能测试参考资料：

[How to reason about collective operations](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce)

[Sample benchmarking code](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py)

> 💡 **核心机理深度解析：为什么 Ring All-Reduce 必须切成 W 份并传递 (W-1) 次？**
>
> - **前置辨析：逻辑通信算法 vs. 真实物理拓扑**：此处的“环形”严格属于软件层面的**逻辑通信拓扑（Logical Algorithm）**，而非真实的物理硬件布线。关于数据中心中真实的物理网络拓扑——如 Google TPU 的多维环形网格 (Torus) 与 GPU 的树形/全互连结构 (Fat-Tree/A2A)——在 [Lecture 08: 并行化基础 (Parallelism Basics)](lecture_08.md) 中有详尽的对比与展开。
> - **1. 环形接力广播机制（转发 $W-1$ 次）**：
>   - 理想的多卡互联通信将各节点逻辑上连接成一个环（$0 \to 1 \to \dots \to W-1 \to 0$）。
>   - 任意卡的数据要广播给其他所有参与卡，只需顺时针接力传递，不需要发给自己，因此单阶段刚好需要传输 **$W - 1$** 次。
> - **2. 对称切分避免单点瓶颈（将数据均分为 $W$ 份）**：
>   - 传统主从模式（Parameter Server）下所有卡发给 Master，导致 Master 网卡严重拥塞瘫痪（Incast 问题）。
>   - 为了提高多卡利用率并实现节点完全对称，让每张卡仅负责张量局部 **$1/W$** 分块的累加与存储，因此张量在数学与工程上必须等分成 **$W$** 份。
> - **3. 硬件流水线并发的极致收益**：
>   - 在每个时间步，所有卡的发送通道（Send）和接收通道（Recv）同时 100% 满负荷打满，消除空泡；
>   - 经 Scatter-Reduce 与 All-Gather 两轮传递，单卡实际收发数据量恒定为 $2 \cdot S \cdot \frac{W-1}{W}$，使得单卡网络开销在卡数扩展至百卡、千卡时几乎恒定（$\approx 2S$），具备近乎无限的横向扩展性。

### 第二部分：分布式训练并行策略

我们将通过多层感知机 (MLP) 的最小化代码实现，逐个解剖不同的并行策略。

这极具代表性，因为 MLP 是 Transformer 模型中最主要的计算开销之一。

---

### 1. 数据并行 (Data Parallelism / DDP)

![](images/data-parallelism.png)

切分策略：数据切片分布在各卡上，各卡模型与参数完全一致。

```python
def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    """
    【函数作用】：数据并行 (Distributed Data Parallel / DDP) 最小化端到端训练闭环。
    【核心架构特征】：
        1. 切分策略：只切数据批次 (Batch)，不切模型参数；每张卡持有完全镜像的完整模型权重。
        2. 计算流程：局部前向 -> 局部 Loss -> 本地反向求导 (param.grad)；
        3. 通信关键：在优化器更新前，通过 dist.all_reduce(AVG) 全局同步梯度，确保更新后多卡权重始终镜像一致。
    
    【参数说明】：
        - rank: 当前进程的卡号
        - world_size: GPU 总卡数
        - data: 完整的全局输入批次张量 (batch_size, num_dim)
        - num_layers: MLP 隐藏层深度
        - num_steps: 训练迭代步数
    """
    setup(rank, world_size)

    # 步骤 1：切分全局 Batch 数据，提取当前卡负责的局部数据分片 (Local Batch)
    # [全局批次被纵向均匀切分]: Card 0 -> B0, Card 1 -> B1, Card 2 -> B2, Card 3 -> B3
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    data = data[start_index:end_index].to(cuda_if_available(rank))

    # 步骤 2：初始化模型参数（由于内部固定了随机种子，各卡本地独立生成一模一样的完整参数镜像）
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # 各卡本地独立维护 AdamW 的一阶与二阶动量状态

    for step in range(num_steps):
        # 步骤 3：前向传播（只处理本卡局部的一小批样本）
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        # 教学用的 Toy Loss（均方误差）：将多维张量压缩为可求导标量以触发反向传播。真实大模型训练会使用交叉熵损失（需补充 Label 构造、词表投影 Head 等大量与分布式通信无关的代码）
        loss = x.square().mean()  # 局部 Loss：由于样本不同，各卡算出的局部 Loss 数值各不相同

        # 步骤 4：反向传播（计算本卡样本对模型各参数的局部梯度）
        loss.backward()

        # 步骤 5：跨卡梯度全规约同步（这是数据并行 DDP 与单卡单机训练在代码层面的唯一核心差别！）
        # 将所有卡对应位置的梯度累加后除以 world_size 求均值，同步回每一张卡
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # 步骤 6：权重更新（因为各卡初始权重相同、收到的梯度均值相同、学习率相同，故步进后权重依然严格一致）
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[layer]) for layer in range(num_layers)]}", flush=True)

    cleanup()
```

```python
# 生成模拟批次数据并启动 4 卡数据并行 (DDP) 训练
data = generate_sample_data()
spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)
```

要点说明：

- **Loss 计算独立**：由于各卡处理的样本（Data shard）不同，前向传播得到的局部 Loss 也完全不同。

- **梯度 All-Reduce 同步**：反向传播后，需要对各卡的梯度进行 All-Reduce 并广播，以保证梯度完全一致。

- **参数完全镜像**：梯度一致，加上优化器以相同步长推进，保证了各卡在每次更新后权重完全相同。

下节预告：FSDP/ZeRO 并行，使用 All-Gather 和 Reduce-Scatter 消除重复持有完整模型参数的显存开销。

---

### 2. 张量并行 (Tensor Parallelism / TP)

![](images/tensor-parallelism.png)

切分策略：每一层的参数矩阵被按维度切分到不同的卡上，每次计算时需要卡间通信同步激活值。

```python
def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    """
    【函数作用】：张量并行 (Tensor Parallelism / TP / Megatron-LM 风格) 最小化前向传播闭环。
    【核心架构特征】：
        1. 切分策略：不切数据批次 (全量 Batch 广播给所有卡)；将每一层的权重矩阵按列 (特征宽度) 纵向切分；
        2. 显存节约：每张卡仅持有参数矩阵的 1 / world_size，彻底突破单卡放不下大模型的显存瓶颈；
        3. 通信关键：由于每张卡只算了部分特征维度，在进入下一层前必须调用 dist.all_gather 收集完整激活值。
    
    【参数说明】：
        - rank: 当前进程的卡号
        - world_size: GPU 总卡数
        - data: 完整的全局输入批次张量 (batch_size, num_dim)
        - num_layers: MLP 隐藏层深度
    """
    setup(rank, world_size)

    # 步骤 1：所有 worker 卡均加载完整的全量 Batch 数据
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    # 按特征通道数进行切分：每张卡只负责 local_num_dim 个隐藏层维度
    local_num_dim = int_divide(num_dim, world_size)

    # 步骤 2：创建分片权重矩阵 W_local，其形状为 (num_dim, local_num_dim)
    # [列并行切分示意]: [ W0 | W1 | W2 | W3 ]，每卡仅持有 1 个切片
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # 步骤 3：逐层进行切分前向计算与激活值通信拼接
    x = data
    for layer in range(num_layers):
        # 3.1 本卡局部矩阵乘法：(batch_size, num_dim) @ (num_dim, local_num_dim)
        # 得到局部特征激活值，形状为 (batch_size, local_num_dim)
        x = x @ params[layer]
        x = F.gelu(x)

        # 3.2 为各卡即将汇聚过来的激活切片预先分配显存容器
        activations = [torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank)) for _ in range(world_size)]

        # 3.3 跨卡全收集通信：tensor=x 为本卡数据源，tensor_list=activations 为接收各 Rank 切片并就地写入的容器（长度严格等于 world_size 且单张量形状需严格对齐）
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # 3.4 沿特征维度 (dim=1) 顺次拼接，无损恢复出完整的 (batch_size, num_dim) 全局激活矩阵，输入下一层
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

    cleanup()
```

```python
# 生成数据并启动 4 卡张量并行 (TP) 前向传播
data = generate_sample_data()
spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)
```

> 💡 **核心概念澄清：张量并行中的显存分布本质与通信机理**
>
> 1. **分块矩阵乘法与 `All-Gather` 的数学必然性**：
>    - **列并行切分**：将权重矩阵按列纵向切分 $W = [W_0 \mid W_1 \mid \dots \mid W_{k-1}]$，Rank $i$ 仅持有局部切片 $W_i$。
>    - **局部乘法与全量汇聚**：各卡计算 $Y_i = X \cdot W_i$，所得结果仅为最终输出特征在通道维度的局部切片。为了给下一层提供完整的输入特征 $Y = [Y_0, Y_1, \dots, Y_{k-1}]$，底层必须通过 **`dist.all_gather`** 跨卡收集所有切片并沿特征维度拼接还原。
>
> 2. **显存组成结构：哪些是占用显存的“绝对大头”？**
>    - **静态常驻显存（占 80%~90%+，单卡放不下的根本元凶）**：
>      - **模型权重 $W$**：全模型所有层自始至终永久驻留在显存中；
>      - **反向梯度 $\nabla W$**：与权重同等大小；
>      - **优化器状态（如 AdamW）**：需持久维护一阶动量 $M$、二阶动量 $V$ 及 FP32 主权重，显存占用通常高达模型权重的 **4 ~ 6 倍**！
>    - **动态瞬时显存（单层仅占数 MB，显存轻量级）**：
>      - 前向计算产生的中间激活值（Activation）。
>
> 3. **为什么单卡能轻松装下完整的前向激活矩阵？（$l \ll n$ 的数学解析）**：
>    - **矩阵尺度对比**：设单层权重矩阵为 $W \in \mathbb{R}^{m \times n}$（隐藏层中通常 $m = n$，元素总量为 $n^2$），输入与激活矩阵为 $X \in \mathbb{R}^{n \times l}$（元素总量为 $n \times l$），其中 $l$ 为有效批次样本数（$\text{Batch Size} \times \text{Sequence Length}$）。
>    - **尺度远小于权重（$l \ll n$）**：在典型微批次或单步前向中，$l \ll n$（如课件中 $n=1024$ 而 $l=128$，单层激活值仅为单层权重的 $\frac{l}{n} = \frac{1}{8}$，仅约 $0.5 \text{ MB}$）。
>    - **单层瞬时 vs. 全局累积**：全模型的数百 GB 权重和优化器是全局每一层全部累加永久驻留的，而当前层前向激活只是单层瞬时驻留。因此，**单卡装下拼接后的完整前向激活值没有任何显存压力**。张量并行（TP）的核心价值就是将不可承受的“静态常驻显存大头”成功切分到了 $\frac{1}{W}$。
>
> 4. **课件简化实现 vs. 工业级 Megatron-LM 架构**：
>    - 课件为演示单层纯列并行，故每层必须 `All-Gather` 恢复完整激活；
>    - 工业界真实架构（如 Megatron-LM）采用**“列并行（升维） + 行并行（降维）”成对设计**：第一层列并行后不通信，直接输入第二层行并行矩阵相乘，仅在末尾执行一次 **`All-Reduce(SUM)`**，从而使中间激活值全程保持分片状态，实现显存与通信的极致双赢。

---

### 3. 流水线并行 (Pipeline Parallelism / PP)

![](images/pipeline-parallelism.png)

切分策略：将网络深度层均匀分发到不同的 GPU 上，顺次执行前向与反向传输。

```python
def pipeline_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_micro_batches: int):
    """
    【函数作用】：流水线并行 (Pipeline Parallelism / PP / GPipe-1F1B 原型) 最小化前向传播闭环。
    【核心架构特征】：
        1. 切分策略：按网络深度 (Layer 维度) 将连续层切分给不同的 GPU（例如 4 层切给 2 卡，每卡 2 层）；
        2. 微批次机制 (Micro-batching)：将大 batch 切分成小块，使各卡顺次流水流动，极大压减流水线空闲气泡 (Bubble)；
        3. 通信关键：采用点对点通信 (P2P)，Rank i 通过 dist.recv 接收上游激活，算完后通过 dist.send 传给下游 Rank i+1。
    
    【参数说明】：
        - rank: 当前进程的卡号 (0 为首卡，world_size-1 为末卡)
        - world_size: GPU 总卡数（即流水线总阶段数 Pipeline Stages）
        - data: 完整的全局输入批次张量 (batch_size, num_dim)
        - num_layers: 整个模型的总层数
        - num_micro_batches: 微批次数量（通常划分越多，硬件并发利用率越高，但激活值显存占用增加）
    """
    setup(rank, world_size)

    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)

    # 步骤 1：顺次切分网络深度，各卡仅分配局部连续的 local_num_layers 层
    local_num_layers = int_divide(num_layers, world_size)
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]

    # 步骤 2：将数据 Batch 拆分为多个细粒度的 Micro-batch 以填充流水线
    micro_batch_size = int_divide(batch_size, num_micro_batches)
    if rank == 0:
        # 首卡持有真实输入，通过 chunk 切分成多个 micro-batch（类似 CPU 指令流水线，避免下游卡空转气泡）
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 非首卡：为来自上游的中间激活值预先分配接收缓冲区
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank)) for _ in range(num_micro_batches)]

    # 步骤 3：流水线迭代推演
    for x in micro_batches:
        # （注：在循环内直接使用并修改循环变量 x 的语法安全性与底层显存考量详见后文文字注释）
        # 3.1 若非第一级流水卡 (rank > 0)，通过 dist.recv (receive) 阻塞接收上游卡 (rank - 1) 传来的层间前向激活值
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # 3.2 运行当前卡所负责的网络层进行前向计算
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # 3.3 若非最后一级流水卡 (rank < world_size - 1)，将算好的激活值点对点发送给下游卡 (rank + 1)
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    text("Not handled: overlapping communication/computation to eliminate pipeline bubbles")

    cleanup()
```

```python
# 生成数据并启动 2 卡流水线并行 (PP) 前向传播（按 micro-batch 切分流水）
data = generate_sample_data()
spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)
```

> 💡 **核心语法与工程设计注释：循环中复用与修改变量 `x` 的底层本质**
>
> 1. **`for i in iter` 的控制流与指针本质（为什么安全）**：
>    - **迭代容器 vs. 循环变量**：在 `for i in iter` 中，`iter` 才是真正的迭代控制容器。真正危险的是在循环体内部直接修改 `iter` 本身（如增删列表元素从而破坏迭代器的游标指针）；
>    - **指针重绑定机制**：循环变量 `i`（代码中的 `x`）本质上只是一个指向特定显存地址的引用指针。每次循环开始时，底层解释器都会强制将其重新赋值指向容器中的下一个元素。因此在循环体内对其就地修改或重新赋值，丝毫不会对下一轮循环的正常推进产生任何副作用，完全安全。
>
> 2. **为什么在 `dist.recv` 中直接复用 `x` 接收而不引入新变量？（底层显存的宝贵性）**：
>    - **避免动态显存分配（Zero Dynamic Allocation）**：在 GPU 底层高性能编程中，显存极其宝贵，且动态申请显存（`cudaMalloc`）耗时昂贵。如果在循环内部每次都开辟新变量去接收网络数据，会引发严重的显存额外占用与内存碎片；
>    - **就地写入预分配缓冲区**：非首卡在循环前已将 `micro_batches` 作为空缓冲区一次性分配就绪，循环中直接借用 `x` 作为显存指针，通过 `dist.recv(tensor=x)` 让底层网络数据就地（In-place）写入覆盖，既极其节约宝贵的显存，又保证了流水线的高效零拷贝流转。

---

### 本讲暂未涵盖的议题

- 通信与计算的深度重叠优化

- 更为复杂的注意力机制并行等

- 其他高级并行形式（如序列并行、专家并行以及混合并行等）

- Jax/TPU 并行：只需在模型中定义张量的分片方式，底层的编译器将自动生成通信拓扑。[levanter](https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html)

- 但在 PyTorch 中，我们需要手动调用分布式原语，这非常有助于深刻理解底层机制。

---

### 第七讲总结

- 分布式并行有多种拆分维度：数据并行 (拆分 Batch)、张量/专家并行 (拆分宽度/通道数)、流水线并行 (拆分层深)、序列并行 (拆分序列长度)

- **数据并行**：DDP（借助 All-Reduce 同步梯度）以及 FSDP/ZeRO（结合 All-Gather 与 Reduce-Scatter 消除多余的显存持有）

- **张量并行**：将单层拆分到不同 GPU，由于每层均需同步激活值，因而极度依赖极高速的卡间带宽 (如 NVLink)

- **流水线并行**：将不同层部署到不同 GPU 顺次计算，对网络通信带宽要求低，但必须合理排布流水线以减小空闲泡泡 (Bubble)

- 系统设计的永恒权衡：是用**重算 (Recompute)**、**显存存储 (Memory)** 还是**跨卡通信 (Communicate)** 来解决局部硬件存储限制

- 尽管硬件网络在不断加速，但模型规模的膨胀使得这些多层次 of 分布式并行架构始终是前沿训练的必修课。

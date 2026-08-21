# 手撕大模型并行训练

## 1. 分布式硬件拓扑与软件生态栈

### 1.1 分布式硬件拓扑与直接通信
多卡/多节点训练中，避免数据传输瓶颈是最大化算力的前提。
* **低效方案 (传统以太网/PCIe)**: GPU 间的数据交换必须通过 PCIe 总线复制到 CPU 主机缓存中，再通过以太网发送。这会产生巨大的 CPU 额外开销和极高的延迟。
* **高效方案 (Direct GPU-to-GPU)**: 
  * **机内**: 现代集群（如 H100 节点）直接使用机内 **NVLink**（H100 拥有 18 个 NVLink 通道，总双向带宽高达 **900 GB/s**）直连 8 张 GPU。
  * **机间**: 跨机器通过 **Mellanox 网卡/交换机** 直接互联（支持 RDMA），绕过 CPU 及操作系统的内核缓冲区，实现 GPU 与远端 GPU 内存的直接读写。
  * **物理限制**: 显存带宽（HBM 访问）虽比跨卡带宽快数倍，但跨卡 NVLink 带宽增长依然受限于物理连接密度、功耗、发热量等硬件限制。

### 1.2 软件驱动生态：NCCL 与 PyTorch Distributed
* **NCCL (Nickel, NVIDIA Collective Communications Library)**: NVIDIA 提供的底层通信优化库。它在初始化时会自动探测硬件拓扑结构（如 GPU 之间的 NVLink 连通性、网络卡分布），并为各类集体操作选择最优的数据传输流路径和网络拓扑（环形或树形拓扑）。
* **PyTorch Distributed**: 封装了底层通信库的 Python API：
  * 支持多种通信后端（Backend）：针对 GPU 优先选择 **NCCL**，针对 CPU 调试或无 GPU 环境可选择 **Gloo**。
  * 提供了简洁的分布式控制逻辑，例如多进程调度器与进程组管理机制。

---

## 2. 核心通信原语的 PyTorch 实现与基准测试

### 2.1 分布式基本术语
* **World Size (世界规模)**: 参与分布式训练的 GPU/进程总数。
* **Rank (秩/等级)**: 每个进程的唯一整型标识符，范围从 $0$ 到 $\text{World Size} - 1$。
* **Barrier (屏障同步)**: `dist.barrier()` 强迫所有 Rank 在该代码行同步，直到所有节点到达后方可继续执行。通常用于控制打印日志对齐或保存 Checkpoint。

### 2.2 常见通信原语代码示例
利用 PyTorch Distributed，我们可以用几行 Python 代码执行复杂的张量分发。

#### 2.2.1 AllReduce
将所有 Rank 的张量相加，并把结果覆盖写回每个 Rank 的局部张量：
```python
# 假设每个 Rank 都有一个一维张量 tensor
import torch.distributed as dist

# 执行原地求和 AllReduce
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

#### 2.2.2 ReduceScatter
对所有 Rank 上的张量进行求和，但最终只将求和结果的对应切片保留在对应的 Rank 上（Rank $i$ 只保留第 $i$ 个分片）：
```python
# input_tensor_list 包含世界规模数量的张量切片
# output_tensor 用于存储归约后的局部切片
dist.reduce_scatter(output_tensor, input_tensor_list, op=dist.ReduceOp.SUM)
```

#### 2.2.3 AllGather
收集所有 Rank 的局部张量，并在所有设备上拼成完整的张量列表：
```python
# output_tensor_list 为用于接收完整结果的空张量列表
# input_tensor 为本地要发送的张量切片
dist.all_gather(output_tensor_list, input_tensor)
```

### 2.3 通信带宽的基准测试 (Benchmarking Bandwidth)
* **带宽计算公式 (AllReduce)**:
  对包含 $N$ 个元素、类型为 FP32（4 字节）的张量进行 AllReduce，单张显卡在网络中传输（发送和接收）的数据量为：
  $$\text{Bytes} = 2 \times \frac{\text{World Size} - 1}{\text{World Size}} \times N \times 4$$
  基于实测时间 $t$（必须在计时前后调用 `torch.cuda.synchronize()`），可以测出真实的网络吞吐带宽。
* **带宽计算公式 (ReduceScatter / AllGather)**:
  这两者在通信过程中只执行了 AllReduce 的一半步骤（无需两倍因子，传输量少一半）。因此，在优化器更新过程中，先 `ReduceScatter` 梯度、再 `AllGather` 更新后参数，总数据吞吐量正好与单次 `AllReduce` 等价。

---

## 3. 手撕三类模型并行策略的 PyTorch 代码实现

为了清晰说明底层原理，本课以四层 MLP 网络（矩阵乘法）为例。

### 3.1 朴素分布式数据并行 (DDP) 实现
* **核心思路**: 每个进程拥有完整的模型。数据集被切分给不同的 Rank，各自计算梯度。计算完毕后，在更新参数前，通过 `all_reduce` 梯度求平均。
```python
import torch
import torch.distributed as dist

def run_ddp(rank, world_size):
    # 1. 划分数据集（切分 Batch 维度）
    global_batch_size = 128
    local_batch_size = global_batch_size // world_size
    
    # 模拟获取局部批次数据
    local_x = get_local_data(rank, local_batch_size)
    
    # 2. 初始化网络参数（必须确保所有 Rank 使用相同的随机种子进行初始化，保证初始权重一致）
    mlp_layers = [torch.nn.Linear(1024, 1024).cuda() for _ in range(4)]
    optimizer = torch.optim.SGD([p for layer in mlp_layers for p in layer.parameters()], lr=0.01)
    
    # 3. 前向传播
    x = local_x
    for layer in mlp_layers:
        x = torch.relu(layer(x))
    loss = x.sum()
    
    # 4. 反向传播
    loss.backward()
    
    # 5. 【核心】：同步梯度
    # 对每层参数的梯度发起 AllReduce 平均
    for layer in mlp_layers:
        for param in layer.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size  # 求平均梯度
                
    # 6. 本地独立执行 Optimizer Step（由于梯度已同步，各 Rank 的权重更新完全一致）
    optimizer.step()
```

### 3.2 张量并行 (Tensor Parallel, TP) 模拟实现
* **核心思路**: 参数沿隐藏层维度切分。每个 GPU 仅计算输出通道的一部分，通过 `AllGather` 在前向传播中重新聚合完整的激活值。
```python
def run_tensor_parallel(rank, world_size):
    # 宽度方向切分：本地隐藏维度
    hidden_dim = 1024
    local_dim = hidden_dim // world_size
    
    # 每个 Rank 仅保存局部矩阵参数 W: [1024, 256]
    local_weight = torch.randn(hidden_dim, local_dim).cuda()
    
    # 前向计算第一步：计算局部激活值 [Batch, local_dim]
    local_activation = torch.matmul(x, local_weight)
    
    # 【核心】：全收集同步激活值
    # 为拼接完整激活值分配空间 [world_size, Batch, local_dim]
    gather_list = [torch.empty_like(local_activation) for _ in range(world_size)]
    dist.all_gather(gather_list, local_activation)
    
    # 沿特征维度拼接，重构完整激活值 [Batch, 1024]，作为下一层的输入
    full_activation = torch.cat(gather_list, dim=-1)
    
    # 反向传播的实现极其繁琐（涉及相反方向的 AllReduce 累加梯度），在 hands-on 中省略。
```

### 3.3 流水线并行 (Pipeline Parallel, PP) 模拟实现
* **核心思路**: 层沿深度方向拆分。GPU $i$ 计算完自己的层前向后，使用**点对点（P2P）通信**将激活值传递给 GPU $i+1$。
* **微批次设计**: 为了减小流水线气泡，输入批次被拆分为多个微批次（Micro-batches），依次送入流水线。
```python
def run_pipeline_parallel(rank, world_size):
    # 假设网络共有 4 层，有 2 个 Rank
    # Rank 0 负责第 0, 1 层；Rank 1 负责第 2, 3 层
    my_layers = [Layer(), Layer()]
    
    # 划分微批次 (4个微批次)
    micro_batches = torch.chunk(x, chunks=4, dim=0)
    
    for mb_idx in range(4):
        if rank == 0:
            # 阶段 0: 起始点，直接计算并向后发送
            x_in = micro_batches[mb_idx]
            x_out = run_layers(my_layers, x_in)
            
            # 使用同步点对点发送给 Rank 1
            dist.send(x_out, dst=1)
            
        elif rank == 1:
            # 阶段 1: 接收上游激活值，计算最终输出
            x_recv = torch.empty_like(micro_batches[mb_idx])
            
            # 同步等待接收来自 Rank 0 的数据
            dist.recv(x_recv, src=0)
            
            x_out = run_layers(my_layers, x_recv)
            # 计算局部 Loss 等...
```
* **优化方向**: 实际工程中绝不会使用同步 `dist.send`/`dist.recv`（这会导致严重的等待瓶颈），必须使用异步的 `dist.isend()` 与 `dist.irecv()` 句柄，从而实现网络传输与当前微批次计算的并行掩盖。

---

## 4. 激活值重计算 (Recomputation / Activation Checkpointing)

激活值重计算是缓解训练大模型时显存危机最有效的单卡优化技术之一。

### 4.1 为什么需要重计算？
在标准的神经网络反向传播（Backpropagation）中，为了计算参数梯度的链式法则，我们必须使用前向传播（Forward Pass）时计算出的中间激活值（Activations）。
例如，对于操作 $y = W \cdot x$，其梯度的计算公式为：
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T$$
这就意味着中间的输入 $x$ 必须一直保留在 GPU 的全局内存（HBM）中，直到反向传播阶段计算关于 $W$ 的梯度时才能被释放。
在大模型中，随着层数 $L$、隐藏层维度 $H$、序列长度 $S$ 以及 Batch 大小 $B$ 的增加，这些存活在前向与反向传播之间的激活值显存占用呈线性甚至二次方增长，往往会成为导致显存溢出（OOM）的首要原因。

---

### 4.2 算力换显存的 Trade-off 与 33% 额外 FLOPs 计算
**激活值重计算 (Activation Checkpointing)** 提出了一种通过“多花计算时间”来“节省显存空间”的置换策略：

![激活值保存与重计算模式显存流动对比](images/activation_recomputation_flow.drawio.png)

* **工作原理**:
  1. **前向阶段 (Checkpoint)**: 我们不保存每一层的全部中间激活值，而是仅在特定的边界处（例如每个 Transformer Block 的起始输入）设立 Checkpoint（检查点）并将其保留在 HBM 中。Block 内部计算出的所有临时激活值在用完后立即从显存中清除。
  2. **反向阶段 (Recomputation)**: 当反向传播到达某个 Block 时，我们从 Checkpoint 中取出该 Block 的输入，**重新在片上运行一次前向计算（Recompute）**，即时恢复出刚才被清除的中间激活值。然后立刻使用这些临时激活值计算出梯度，并在计算结束后立即将其再次清除。

#### FLOPs 额外开销的理论估算
我们可以精确计算启用重计算所引入的额外计算量占比：
1. 设网络某一部分的前向传播计算量为 $F$（FLOPs）。
2. 在反向传播中，我们需要计算关于输入激活值的梯度以及关于权重参数的梯度。由于数学公式的对称性，反向传播的计算量大约是前向传播的 **2 倍**，即 $2F$。
3. **朴素训练总计算量**:
   $$\text{FLOPs}_{\text{naive}} = F (\text{前向}) + 2F (\text{反向}) = 3F$$
4. **重计算模式总计算量**:
   由于反向传播时需要重新运行一次前向计算：
   $$\text{FLOPs}_{\text{recompute}} = F (\text{前向}) + F (\text{重跑前向}) + 2F (\text{反向}) = 4F$$
5. **计算开销占比**:
   $$\text{Overhead} = \frac{4F - 3F}{3F} = \frac{1}{3} \approx 33.3\%$$

* **结论**: 激活值重计算用 **33.3% 的额外计算开销**，换取了显存占用从随层数线性增加 $O(L)$ 下降到仅随检查点数量呈 $O(\sqrt{L})$ 或 $O(1)$ 的巨大显存缩减。在显存成为绝对瓶颈的场景下，这能够让我们把 Batch Size 扩大数倍，通过提升 GPU 计算的饱和度，端到端的吞吐量反而可能不降反升。

---

### 4.3 PyTorch 代码实现
在 PyTorch 中，我们可以非常简单地使用 `torch.utils.checkpoint.checkpoint` 函数来实现这一技术：

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.mlp = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # 内部前向逻辑
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class LargeModel(nn.Module):
    def __init__(self, num_layers=24, d_model=4096):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model) for _ in range(num_layers)])
        
    def forward(self, x, use_checkpoint=True):
        for layer in self.layers:
            if use_checkpoint:
                # 【核心】: 使用 PyTorch Checkpoint 封装该层的前向传播
                # 只有 layer 的输入 x 会被保存在显存中
                # layer 内部的临时激活值在 forward 结束时被丢弃，并在 backward 时重计算
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
```

---

## 5. 声明式并行 (JAX/Levanter) 与命令式并行 (PyTorch) 对比

* **PyTorch (命令式/P2P 集体通信)**:
  * 程序员必须显式定义每一处通信原语的位置和张量形状（如手动调用 `all_reduce`，手动分配 `all_gather` 列表等）。
  * 灵活性极高，有利于深度定制与硬件潜能压榨（例如 DeepSeek 在底层高度定制 NCCL 数据传输方案以极限榨干显存带宽）。
  * 缺点是系统簿记和状态控制非常繁琐（如 FSDP 的参数管理实现极其臃肿复杂）。
* **JAX (声明式/编译期分片)**:
  * JAX 允许程序员仅声明并行网格维度（Mesh）与逻辑维度的分片映射规则（如“把注意力头部维度分片到 GPU 网格的 X 轴”）。
  * **编译器自动调度**: 声明分片规则后，JAX 编译器（XLA）在编译计算图时，会自动生成最高效的底层重排原语与硬件通信序列，无需手写任何 Nickel 调用。代码非常精简易维护（如基于 JAX 的大模型套件 Levanter，开启全分片仅需十行声明）。

---

## 6. 工程运维现实：集群容错与硬件局限

* **GPU 故障率是大规模训练的物理现实**:
  * 随着集群规模扩大，GPU 硬件（如 H100）出现故障的概率大幅增加。
  * **Llama 3 报告案例**: 在其超大规模集群训练中，因 GPU 损坏、未预料的主机维护等硬件原因，共经历了 **148 次非计划中断**（占全部中断次数 of 30%）。
  * **解决方案**: 基础设施必须支持极其鲁棒的快速 Checkpoint 容错架构。
  * **静默数据损坏 (Silent Data Corruption)**: 最隐蔽的问题。GPU 不报错，但乘法器发生微小物理故障导致计算结果产生垃圾数据，这会慢慢污染整个计算图，最终使训练彻底发散。

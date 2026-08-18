# 第 6 讲：基准测试、性能分析与 Triton 内核编写

**CS336 - 计算机系统与深度学习**  
**讲师：Tatsu H**  

---

### 课程概述与今日目标

- **上节课回顾**：GPU 硬件架构的宏观概述与基本性能特征。
- **今日核心**：
  1. 如何对显卡上的操作进行**基准测试 (Benchmarking)** 与**性能分析 (Profiling)** 以定位瓶颈。
  2. 为什么**算子融合 (Operator Fusion)** 对缓解内存带宽受限极度关键。
  3. 学习编写 **OpenAI Triton 内核**（逐元素 GeLU 激活函数、行归约 Softmax、跨块归约行求和、以及分块矩阵乘法 Matmul）。

```python
import os
import time
from typing import Callable
import torch
from torch.profiler import ProfilerActivity
import triton
import triton.language as tl
from gpu_util import cuda_if_available

print(f"CUDA 可用状态: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前设备: {torch.cuda.get_device_name(0)}")
```

# 1. GPU 硬件与编程模型回顾 (Review of GPUs)

## 1.1 硬件规格规格对比 (Hardware Specs)

| 加速器 (Accelerator) | A100 | H100 | B200 |
| :--- | :---: | :---: | :---: |
| 流式多处理器数量 (# SMs) | 108 | 132 | 148 |
| 寄存器大小 (每个 SM) | 256 KB | 256 KB | 256 KB |
| L1 缓存 + 共享内存 (每个 SM) | 192 KB | 256 KB | 256 KB |
| L2 缓存大小 | 40 MB | 50 MB | 96-126 MB |
| HBM 容量 | 80 GB | 80 GB | 192 GB |
| 寄存器带宽 | ~116 TB/s | ~401 TB/s | ~447 TB/s |
| L1 缓存 + 共享内存带宽 | ~19 TB/s | ~33 TB/s | ~19 TB/s |
| L2 缓存带宽 | ~5-8 TB/s | ~12 TB/s | ~9 TB/s |
| HBM 带宽 | 2 TB/s | 3.35 TB/s | 8 TB/s |

*注：B200s 还具有张量内存 (TMEM) 用于张量核心（介于寄存器和共享内存之间），但它们对程序员是不可见的。*

## 1.2 GPU 编程模型 (Programming Model)

- **线程 (Thread)**：执行具体指令以处理数据的最小单位。
- **线程块 (Thread Block / CTA)**：协同处理同一块数据的线程集合，共享相同的 Shared Memory，并且必定被调度到同一个流式多处理器 (SM) 上运行。
- **网格 (Grid)**：包含多个独立线程块的整体集合。

### 为什么需要分线程块 (Thread Blocks)？
对于普通的逐元素 (elementwise) 算子，每个线程独自读取并处理一个元素就足够了；但是在非逐元素算子（如 Softmax 或矩阵乘法）中，线程间需要共享或累加数据。由于 HBM 访存非常慢，将数据载入流式多处理器本地的**共享内存 (Shared Memory)** 并进行块内通信是加速的关键。

## 1.3 编程模型与硬件交互对性能的影响

- **线程束 (Warps)**：线程块内部线程被划分为以 32 个为一组的 Warp。同一个 Warp 内的所有线程执行指令是完全同步 (lockstep) 的。如果出现分支选择（`if-else`），则需要进行串行排队执行，即**控制流分流分化 (Control Divergence)**，带来性能衰退。
- **线程束占用率 (Warp Occupancy)**：由于寄存器（每个 SM 仅有 256KB）与共享内存资源有限，每个线程如果占用的寄存器过多，会导致 SM 上同时并行的线程束数量变少（即低占用率）。但如果每个线程通过粗化计算了更多元素，低占用率不一定意味着低效率。  
  
下面我们在代码中来实际计算一下占用率：

```python
# 我们想要运行的配置
num_threads_per_block = 128
num_registers_per_thread = 160

# 硬件极限规格 (以 A100 为例)
max_registers = 65536  # 每个 SM 允许的最大寄存器数量
max_warps = 64         # 每个 SM 允许的并发线程束最大数量

# 保证每个线程最多使用 255 个寄存器
assert num_registers_per_thread <= 255
num_registers_per_block = num_threads_per_block * num_registers_per_thread  
num_blocks = max_registers // num_registers_per_block  # 寄存器限制了能同时运行的块数量
num_warps = num_blocks * num_threads_per_block / 32  
occupancy = num_warps / max_warps  

print(f"每个线程块消耗寄存器: {num_registers_per_block} 个")
print(f"每个 SM 并发块数: {num_blocks} 块")
print(f"每个 SM 并发线程束: {num_warps} 个")
print(f"Warp 占用率: {occupancy * 100:.2f}%")
```

- **银行冲突 (Bank Conflicts)**：片上共享内存在物理上被划分为 32 个银行 (Banks)，每周期一个 Bank 只能被一个线程访问。若多个线程同时访问同一个 Bank 且并非同一个地址，会导致硬件执行被串行化，引发银行冲突。
- **内存合并访问 (Memory Coalescing)**：Warp 内 32 个线程的访存请求会被硬件尽可能地合并为 128 字节的单次全局存取。如果每个线程访问跨度极大的非对齐地址，将大幅浪费 HBM 带宽。
- **块占用率 (Block Occupancy / Wave Quantization)**：网格内的线程块是以波次 (Waves) 被调度到各 SM 上的。如果块数不能整除 SM 数量，最后一波只会有极少的块在运行，造成硬件大量 SM 闲置。

# 2. 基准测试与性能分析 (Benchmarking and Profiling)

为了实现高效率优化，我们必须建立一个严谨的“分析-修改-再分析”的迭代闭环。
- **基准测试 (Benchmarking)**：用来测试一段算子操作的端到端耗时。
- **性能分析 (Profiling)**：用来探究底层的内核执行耗时细节，发现性能热点与瓶颈所在。

下面我们在代码中定义我们自己的基准测试与分析实用函数：

```python
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

def benchmark(run: Callable, num_warmups: int = 2, num_trials: int = 5) -> float:
    """通过运行多个 trial 来获取 run 算子的平均 GPU 耗时"""
    # 热身运行，用于排除编译/初始化带来的冷启动误差
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # 确保热身计算全部落盘完成

    times: list[float] = []
    for trial in range(num_trials):
        # 使用 CUDA Events 进行精确测时，避开 CPU 调度的随机抖动
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        run()
        end_event.record()

        torch.cuda.synchronize()  # 同步等待 GPU 完成
        times.append(start_event.elapsed_time(end_event))

    return mean(times)

def profile(run: Callable, num_warmups: int = 2):
    """利用 PyTorch 的性能分析器来展示底层内核调用状态"""
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        run()
        torch.cuda.synchronize()

    # 格式化表格显示输出
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=80,
        row_limit=5
    )
    return table

def run_operation2(dim: int, operation: Callable) -> Callable:
    """创建用于测试的闭包，输入两个随机方阵进行操作"""
    a = torch.randn(dim, dim, device=cuda_if_available())
    b = torch.randn(dim, dim, device=cuda_if_available())
    return lambda: operation(a, b)
```

### 2.1 运行矩阵乘法基准测试

```python
# 测试并记录 1024x1024 乘法耗时
matmul_1024 = run_operation2(1024, lambda a, b: a @ b)
print(f"1024 矩阵乘法时间: {benchmark(matmul_1024):.2f} ms")

# 考察维度从 256 到 8192 的立方级复杂度扩展特性
for dim in [256, 512, 1024, 2048, 4096]:
    fn = run_operation2(dim, lambda a, b: a @ b)
    print(f"维度 {dim}x{dim} 耗时: {benchmark(fn):.2f} ms")
```

### 2.2 运行性能分析器 (Profiling)

我们可以查看加法操作与乘法操作调用的具体 CUDA 内核名称：

```python
add_op = run_operation2(2048, lambda a, b: a + b)
print("=== ADD (2048) PROFILER ===")
print(profile(add_op))

matmul_op = run_operation2(2048, lambda a, b: a @ b)
print("\n=== MATMUL (2048) PROFILER ===")
print(profile(matmul_op))
```

# 3. 算子融合与 GeLU 激活函数性能对比 (Naive vs Built-in vs Compiled GeLU)

GeLU 的数学定义为：
$$\operatorname{GeLU}(x) = 0.5x \times (1 + \tanh(0.79788456(x + 0.044715x^3)))$$

如果我们使用 Naive 的 PyTorch 代码书写，会导致每次幂运算、加法、乘法都需要独立调用一个 GPU 内核，产生极其高昂的中间结果写回 HBM 再读出的带宽开销。通过**内核融合 (Kernel Fusion)**，我们可以让整个公式在一次 HBM 读取和写回内完成。

```python
def naive_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

def builtin_gelu(x: torch.Tensor):
    return torch.nn.functional.gelu(x, approximate="tanh")

def check_equal_1d(f1, f2):
    x = torch.randn(2048, device=cuda_if_available())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)

def run_operation1(dim: int, operation: Callable) -> Callable:
    x = torch.randn(dim, dim, device=cuda_if_available())
    return lambda: operation(x)

# 确保编译后的行为正确
compiled_gelu = torch.compile(naive_gelu)

check_equal_1d(naive_gelu, builtin_gelu)
check_equal_1d(naive_gelu, compiled_gelu)

# 开展对比基准测试
dim_size = 8192
naive_fn = run_operation1(dim_size, naive_gelu)
builtin_fn = run_operation1(dim_size, builtin_gelu)
compiled_fn = run_operation1(dim_size, compiled_gelu)

print(f"Naive GeLU 运行时间: {benchmark(naive_fn):.4f} ms")
print(f"Builtin GeLU 运行时间: {benchmark(builtin_fn):.4f} ms")
print(f"Compiled GeLU 运行时间: {benchmark(compiled_fn):.4f} ms")
```

我们来看看它们的 Profile 信息，编译版和内置版是高度融合的单算子内核：

```python
print("=== Naive GeLU ===")
print(profile(naive_fn))

print("\n=== Compiled GeLU ===")
print(profile(compiled_fn))
```

# 4. OpenAI Triton 简介与内核编写

- 在 CUDA 中：我们需要考虑极其精细的每个**线程**应该读写哪个寄存器，以及如何利用共享内存避开银行冲突，开发难度高。
- 在 Triton 中：我们以**线程块 (Thread Blocks)** 为维度组织逻辑。Triton 编译器会自动帮我们把高级的分块描述翻译为高性能的 PTX 指令，并自动完成硬件资源的调度与对齐优化。

## 4.1 Triton GeLU 激活函数实现

Triton 内核需要使用 `@triton.jit` 修饰。我们在这里实现近似 GeLU 的整个数据流动环节：

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # 计算当前处理块的起始绝对索引
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE

    # 计算当前块内的偏置索引
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # 防界限溢出掩码
    mask = offsets < num_elements

    # 1. 数据加载 (HBM -> 寄存器)
    x = tl.load(x_ptr + offsets, mask=mask)

    # 2. 片上融合数学计算
    # 使用公式: tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2.0 * a)
    tanh_val = (exp - 1.0) / (exp + 1.0)
    y = 0.5 * x * (1.0 + tanh_val)

    # 3. 数据写回 (寄存器 -> HBM)
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    num_elements = x.numel()
    BLOCK_SIZE = 1024  # 块尺寸，典型为 1024
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # 向上整除

    # 使用一维网格调用启动内核
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

# 正确性检验
x_test = torch.randn(65536, device=cuda_if_available())
y_naive_gelu = naive_gelu(x_test)
y_triton_gelu = triton_gelu(x_test)
print(f"Triton 激活函数正确性: {torch.allclose(y_naive_gelu, y_triton_gelu, atol=1e-6)}")
```

## 4.2 Triton Softmax 行归约实现

行归约算子需要对整行的元素执行归一化。在下面的算子实现中，我们假定**每一行的元素能够完整塞进单个线程块中 (Row fits in a block)**。

```python
def naive_softmax(x: torch.Tensor):
    # 行的最大值读取
    x_max = x.max(dim=1)[0]
    # 减去最大值后指数化
    numerator = torch.exp(x - x_max[:, None])
    # 计算加和倒数并归一化
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]

@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    # 每个网格的 program 代表一整行
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 读取行首地址，获取该行所有列的数据
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets

    # 如果超出当前行的物理列数，填补 -inf 以保证计算指数时不影响最大值提取与加总
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # 在共享内存中执行规约
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 写回显存
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)

def triton_softmax(x: torch.Tensor):
    M, N = x.shape
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(N)  # 寻找最接近的 2 的幂
    
    triton_softmax_kernel[(M,)]( 
        x_ptr=x, y_ptr=y, 
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )
    return y

# 正确性检验
x_s = torch.randn(128, 512, device=cuda_if_available())
y_s_torch = torch.nn.functional.softmax(x_s, dim=-1)
y_s_triton = triton_softmax(x_s)
print(f"Triton Softmax 正确性: {torch.allclose(y_s_torch, y_s_triton, atol=1e-6)}")
```

## 4.3 Triton 跨块行归约 (Row doesn't fit in a block)

如果每一行的长度极大，无法一次性塞进同一个线程块（Shared Memory 有物理容量界限）。我们需要采取分块迭代计算并逐步累加的策略：

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 当前负责处理哪一行
    row = tl.program_id(0)

    # 片上线程块局部求和累加器
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 对整行元素进行分段滑动加载并累加
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # 在线程块内部将各线程的结果累加出最终标量
    result = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, result)

def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024):
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y

# 正确性检验
x_large = torch.randn(32, 4096, device=cuda_if_available())
y_sum_torch = x_large.sum(dim=1)
y_sum_triton = triton_row_sum(x_large, BLOCK_SIZE=1024)
print(f"Triton 跨块行加总正确性: {torch.allclose(y_sum_torch, y_sum_triton, atol=1e-5)}")
```

## 4.4 Triton 分块矩阵乘法与内核融合 (Matmul ReLU Example)

这里我们通过将结果方阵拆分为多个瓦片/分块大小 ($64 \times 64$)，并在内层循环中迭代加载 $A$ 的行分块与 $B$ 的列分块（分块大小为 $32$），使用 GPU 内置的张量核心进行乘加，最后在写回 HBM 之前直接在片上完成 ReLU 运算（内核融合），实现顶级的算术强度。

```python
@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 定位当前块处理矩阵 C 的坐标瓦片 (m, n)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 确定物理索引偏移
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    indices_k = tl.arange(0, BLOCK_K)

    # 构成片上数据存取指针阵列
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn

    # 局部累加器
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 K 维度进行迭代分块加载
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)
        
        # 矩阵累乘 (使用 Tensor Cores)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 融合 ReLU 激活函数
    acc = tl.maximum(acc, 0.0)

    # 数据写回 HBM
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))

def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 分块瓦片设置
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return c

# 正确性检验
a_mat = torch.randn(256, 256, device=cuda_if_available())
b_mat = torch.randn(256, 256, device=cuda_if_available())
y_mat_torch = torch.nn.functional.relu(a_mat @ b_mat)
y_mat_triton = triton_matmul_relu(a_mat, b_mat)
print(f"Triton 融合矩阵乘积正确性: {torch.allclose(y_mat_torch, y_mat_triton, atol=1e-4)}")
```

---

# 5. 课程总结与重要结论 (Summary)

1. **算术强度与内存限制**：许多逐元素操作受限于 HBM 到寄存器的吞吐速度（Memory-bound）。
2. **算子融合 (Operator Fusion)** 是提升 GPU 算术强度的绝对核心。在 PyTorch 中使用 `torch.compile` 可自动在底层编译出高度融合的内核。
3. **Triton 让我们能手写融合内核**，其采用线程块级抽象，将寄存器与共享内存的管理隐藏，是编写定制高性能算子的最佳搭档。
4. **性能分析三部曲**：合理地执行基准测试定位速率，结合 Profiler 查看底层内核细节，针对性消除对齐不当、分化不均、波次量化等硬件瓶颈。

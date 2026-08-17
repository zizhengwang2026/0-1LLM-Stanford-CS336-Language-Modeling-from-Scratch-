# 第6讲 · 算子与 Triton 内核（CS336 L6 × DIY-LLM 第7章）

> **本讲位置**：CS336 **Systems** 块的深入。上一讲搞懂了 GPU 硬件"长什么样"，这一讲动手写跑在 GPU 上的"小程序"——**内核（kernel）**，以及怎么把它写得又快又省事。
>
> **一句话定位**：PyTorch 自带的算子够用，但当你要压榨最后一点性能（或实现课程作业里的高效算子）时，就得懂"内核怎么写、怎么融合、怎么用 Triton 不用写 CUDA"。

---

## 一、为什么需要自己写"内核"

PyTorch 里 `y = torch.nn.functional.gelu(x)` 一行就完事，背后是 NVIDIA 写好的 CUDA 算子。但现实里：

- 有些操作 PyTorch **没有现成融合算子**，拆成多步会反复读写显存，变慢。
- 课程作业（A2 Systems）要求你**手写高效算子**，理解性能从哪来。
- 想榨干新硬件（如 H100 的 FP8），通用算子不一定用得上底层特性。

所以这一讲的目标：能看懂、能写、能优化一个 GPU 内核。

---

## 二、CUDA 编程模型速记

回顾上一讲的层级，写内核时你要亲自安排"谁算哪个格子"：

| 概念 | 写代码时怎么用 |
|------|---------------|
| **Grid** | 一次 kernel 启动的全部 block 集合 |
| **Block** | 程序员指定大小（如 1024 线程），同 block 共享共享内存 |
| **Thread** | 用 `blockIdx.x * blockDim.x + threadIdx.x` 算出自己的全局序号 `i` |
| **Warp** | 32 线程一组，硬件自动管，你写代码时尽量别让它发散 |

**手写一个 GeLU 内核（C++/CUDA，`gelu.cu`）的核心逻辑**：

```cpp
__global__ void gelu_kernel(float* x, float* y, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 算出"我是第几个线程"
    if (i < num_elements) {                          // 越界保护
        float xi = x[i];
        y[i] = 0.5f * xi * (1.0f + tanhf(0.79788456f * (xi + 0.044715f * xi * xi * xi)));
    }
}
// 启动：block_size=1024，num_blocks = ceil(num_elements/1024)
gelu_kernel<<<num_blocks, block_size>>>(x, y, n);
```

> GeLU 公式：`0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x³)))`。Python 用 `load_inline` 把 `.cu` 当场编译就能调用。调试时设 `CUDA_LAUNCH_BLOCKING=1` 让报错定位到具体行。

---

## 三、先量再优化：基准测试与性能分析

CS336 反复强调的态度：**别凭感觉优化，先量哪里慢**。

1. **基准测试三件套**：预热（`num_warmups`，因为第一次跑要编译/初始化）→ `torch.cuda.synchronize()`（防止 CPU/GPU 异步导致计时提前结束）→ 多次 trial 取均值。
2. **PyTorch Profiler**：`torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)`，能看到时间花在哪个 kernel（如 `aten::mm` 占 60%）。
3. **Nsight Systems**（NVIDIA 专业工具）：能看 CPU/GPU 怎么协作，配合 NVTX 标注（`with nvtx.range("define_model")`）定位初始化、每个 step 的开销。

> 关键认知：一个 `a + b` 看起来简单，profiler 里能看到它其实是"启动一个 elementwise kernel + 同步"；而一次矩阵乘在 A100 上会分派到 Cutlass 的 `cutlass_80_simt_sgemm_64x64x16` 内核。优化要瞄准**占比最高的那个**（通常是 GEMM/矩阵乘）。

---

## 四、内核融合（Kernel Fusion）：少跑几趟

对比两种 GeLU 写法：

- **手动分步**：`y = 0.5 * x`、`y = y * (1 + ...)`、`y = y * tanh(...)` → 启动了 **3 个 CUDA 内核**，每次都要把 `y` 写回 HBM 再读出来。
- **PyTorch 融合版 / 手写单 kernel**：一次算完 → 只 **1 个内核**。

实测（16384 维）：手动 8.1ms vs PyTorch 融合 1.1ms。差距就在"**少了几趟 HBM 往返**"。这也解释了上一讲说的"算子融合提速 2–5 倍"。

---

## 五、Triton：以"块"为中心的编程

手写 CUDA 要管线程、共享内存、同步，很累。**Triton（OpenAI 出品）** 让你用 Python 写，按"块（block/tile）"思考，底层的内存合并、共享内存、调度它自动搞定。

**三个典型例子**：

1. **Softmax（整行入块）**：一个 program 处理一整行，用 `tl.load / tl.max / tl.exp / tl.sum` 一次读写 HBM，不用跨块通信。
2. **行求和（超块分块循环）**：沿序列维循环切 tile，局部累加，最后 `tl.sum` 跨线程归约。
3. **矩阵乘（二维 Tiling）**：二维 grid，沿 K 维循环把 A/B 的小块搬进共享内存，用 `tl.dot` 累加；算术强度提升到 O(tile_size)，还能顺手融合 ReLU。朴素 O(MNK) 是内存受限，分块后每个元素只读一次。

实测（16384 维）：manual 8.1ms / PyTorch 1.1ms / 手写 CUDA 1.84ms / **Triton 1.85ms** —— 接近手写性能，但代码量小一个量级。

> 一句话：**Triton = 用 Python 的写法，拿到接近手写 CUDA 的性能**。所以现在大量新算子（含 FlashAttention 的各种变体）都用 Triton 实现。

---

## 六、torch.compile：编译器替你融合

PyTorch 2.0 的 `torch.compile` 能在你不写任何 CUDA/Triton 的情况下，**自动把简单算子融合**。实测 `torch.compile(manual_gelu)` 只要 1.47ms，比手写 CUDA 还快。

但要注意边界：

- 对"形状已知 + 简单融合 + GEMM"很香，白送 ~10% 提速。
- 对**极其依赖底层硬件特性**的算子（如 FlashAttention-3 利用 H100 的异步 WGMMA），编译器**自动发现不了**，还是得手写 Triton/CUDA。

> 工程建议：**默认先 `torch.compile`**，遇到新架构、复杂算子且 GPU 利用率上不去时，再上 Triton/CUDA 手动优化。不要每个模块都手写内核。

---

## 七、算子优化的通用套路（记住这四条）

1. **提高算术强度**：用分块（Tiling）把数据搬进共享内存复用，减少 HBM 读取次数（从 N² 级降到 N 级）。
2. **融合相邻操作**：把"逐元素 + 激活 + 归一化"合成一个 kernel，消灭中间结果的 HBM 往返。
3. **避免 warp 发散**：循环/分支里别让 32 个线程走不同路。
4. **用对精度**：BF16/FP16 让 Tensor Core 火力全开；FP8 在 H100 上再翻倍。

> 数据参考（B200）：每 SM 寄存器 65536 个 / 256KB，HBM 带宽达 8 TB/s；每线程最多 255 个寄存器，寄存器压力过大会压低"占用率（occupancy）"，反而跑不饱。

---

## 八、CS336 英文原版 vs DIY-LLM 中文版 对照

| 维度 | CS336 L6（英文原版） | DIY-LLM 第7章（中文） |
|------|---------------------|---------------------|
| 侧重 | Triton 实操 + 算子性能建模 + 把 fused kernel 用到训练 | 从 CUDA 模型讲起，手写 GeLU→Triton→torch.compile 渐进对比 |
| 强项 | 给你"为什么这个 kernel 快"的量化模型 | 给你"从零写出一个快算子"的可运行代码路径 |
| 互补点 | 偏"原理与性能分析" | 偏"动手与代码" |

**学习建议**：跟着 DIY-LLM 把 `gelu.cu`、Triton softmax/matmul 各跑一遍，再用 CS336 的 roofline/算术强度去解释"为什么快"。

---

## 九、常见坑 & 面试可能问

1. **"内核融合为什么快？"** → 减少中间张量的 HBM 读写和 kernel 启动开销，本质是提高了算术强度。
2. **"Triton 和 CUDA 怎么选？"** → 先用 Triton（开发快、性能接近），极致场景再 CUDA。
3. **"torch.compile 是银弹吗？"** → 不是，复杂/硬件特化算子它优化不了。
4. **"profiler 显示时间花在 cudaDeviceSynchronize？"** → 说明你代码里其实没怎么算 GPU（比如只 sleep），或者没做同步导致计时错位。

---

## 参考与延伸

- Stanford CS336 Lecture 6: Kernels and Triton
- Datawhale《DIY-LLM》第7章：GPU 高性能编程
- Triton 官方教程（triton-lang.org）
- 论文：FlashAttention-3（利用 H100 异步特性的代表）

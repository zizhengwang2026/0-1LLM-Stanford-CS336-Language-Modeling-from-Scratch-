# 第6讲 · 算子与 Triton 内核（CS336 L6 × DIY-LLM 第7章）

> **本讲位置**：CS336 **Systems** 块的深入。上一讲搞懂了 GPU 硬件"长什么样"，这一讲动手写跑在 GPU 上的"小程序"——**内核（kernel）**，以及怎么把它写得又快又省事。
>
> **一句话定位**：PyTorch 自带的算子够用，但当你要压榨最后一点性能（或实现课程作业里的高效算子）时，就得懂"内核怎么写、怎么融合、怎么用 Triton 不用写 CUDA"。
>
> **难度评级：🔴 最难（三块硬骨头之一）**。原因：要懂 GPU 的执行模型（SIMT / block / warp）、手写 CUDA、还要建立"算术强度 / roofline"的性能直觉。研究里甚至提到"连 GPT-5.2、Claude 写 CUDA kernel 也吃力"——这是离应用层最远、最"底层"的一块。

---

## 🔤 先扫盲：重点术语大白话

| 术语 | 大白话 |
|------|--------|
| **内核 Kernel** | 一个跑在 GPU 上的小程序，比如"算一次矩阵乘"。 |
| **线程 Thread** | GPU 上干活的最小单元，成千上万个一起并行。 |
| **线程块 Block** | 一组能共享"手边工具箱(共享内存)"的线程。 |
| **Warp** | 32 个线程组成的小队，硬件一次性一起指挥。 |
| **Grid** | 一次内核启动的全部线程块集合。 |
| **HBM 显存** | GPU 的大仓库，容量大但慢。 |
| **共享内存 Shared Memory** | 线程块手边的小工具箱，快但小。 |
| **寄存器 Register** | 线程兜里随身工具，最快但极少。 |
| **内核融合 Fusion** | 把多个操作合成一个内核，少跑几趟显存。 |
| **算术强度** | 每搬 1 字节数据能做几次计算；越高越不卡带宽。 |
| **Roofline** | 一张"算力屋顶 vs 带宽斜坡"的图，看程序卡在哪。 |
| **Triton** | OpenAI 出的工具，用 Python 写 GPU 算子，接近手写 CUDA 性能。 |
| **torch.compile** | PyTorch 的编译器，自动帮你融合简单算子。 |
| **memory-bound / compute-bound** | 卡在"搬数据慢" / 卡在"算得慢"。 |

**记住这条主线**：GPU 优化 = 少跑显存(HBM) → 多在手边(寄存器/共享内存)干活 → 融合操作、提高算术强度。

---

## 零、先用"工厂流水线"打比方

| GPU 概念 | 工厂类比 | 一句话 |
|---------|---------|--------|
| **Kernel（内核）** | 一道独立工序（如"拧螺丝"） | 一个在 GPU 上跑的小程序 |
| **Thread（线程）** | 一个工人 | 干同一件事的最小单元 |
| **Block（线程块）** | 一个班组（共享一个工具箱） | 一组能互相协作的线程 |
| **Warp** | 班组的"小分队"（32 人一排） | 硬件一次性指挥的 32 个线程 |
| **Grid** | 全厂所有班组 | 一次 kernel 启动的全部线程 |
| **HBM（显存）** | 厂外大仓库 | 大但慢，跑一趟贵 |
| **Shared Memory** | 班组手边的工具箱 | 小但快，块内共享 |
| **Register** | 工人兜里的随身工具 | 最快，但数量有限 |

**核心矛盾**：仓库（HBM）很远，工人（线程）每次去仓库拿/放东西都要花时间。性能优化的本质就是**少跑仓库、多在手边（寄存器/共享内存）把事干完**。

---

## 一、为什么需要自己写"内核"

PyTorch 里 `y = torch.nn.functional.gelu(x)` 一行就完事，背后是 NVIDIA 写好的 CUDA 算子。但现实里：

- 有些操作 PyTorch **没有现成融合算子**，拆成多步会反复读写显存，变慢。
- 课程作业（A2 Systems）要求你**手写高效算子**，理解性能从哪来。
- 想榨干新硬件（如 H100 的 FP8），通用算子不一定用得上底层特性。

所以这一讲的目标：能看懂、能写、能优化一个 GPU 内核。

---

## 二、CUDA 编程模型速记（这块必须啃下）

GPU 启动一个 kernel 时，硬件把线程组织成"Grid → Block → Thread"三层。写 kernel 时你要亲自安排"谁算哪个格子"：

| 概念 | 写代码时怎么用 |
|------|--------------|
| **Grid** | 一次 kernel 启动的全部 block 集合 |
| **Block** | 程序员指定大小（如 1024 线程），同 block 共享共享内存 |
| **Thread** | 用 `blockIdx.x * blockDim.x + threadIdx.x` 算出自己的全局序号 `i` |
| **Warp** | 32 线程一组，硬件自动管，你写代码时尽量别让它发散 |

**手写一个 GeLU 内核（C++/CUDA，`gelu.cu`）的核心逻辑**：

```cpp
__global__ void gelu_kernel(float* x, float* y, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 算出"我是第几个线程"
    if (i < num_elements) {                          // 越界保护（线程数常比数据多）
        float xi = x[i];
        y[i] = 0.5f * xi * (1.0f + tanhf(0.79788456f * (xi + 0.044715f * xi * xi * xi)));
    }
}
// 启动：block_size=1024，num_blocks = ceil(num_elements/1024)
gelu_kernel<<<num_blocks, block_size>>>(x, y, n);
```

> GeLU 公式：`0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))`。Python 用 `load_inline` 把 `.cu` 当场编译就能调用。调试时设 `CUDA_LAUNCH_BLOCKING=1` 让报错定位到具体行（否则 GPU 异步，报错会"串台"）。

**一个极易踩的坑：线程数 ≠ 数据数**。你通常启动"向上取整"的 block 数，所以必须 `if (i < num_elements)` 越界保护，否则多余线程会读写非法内存。

---

## 三、先量再优化：基准测试与性能分析（态度最重要）

CS336 反复强调：**别凭感觉优化，先量哪里慢**。

1. **基准测试三件套**：预热（`num_warmups`，第一次跑要编译/初始化）→ `torch.cuda.synchronize()`（防止 CPU/GPU 异步导致计时提前结束）→ 多次 trial 取均值。
2. **PyTorch Profiler**：`torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)`，能看到时间花在哪个 kernel（如 `aten::mm` 占 60%）。
3. **Nsight Systems**（NVIDIA 专业工具）：看 CPU/GPU 怎么协作，配 NVTX 标注（`with nvtx.range("define_model")`）。

> 关键认知：一个 `a + b` 看起来简单，profiler 里能看到它其实是"启动一个 elementwise kernel + 同步"；而一次矩阵乘在 A100 上会分派到 Cutlass 的 `cutlass_80_simt_sgemm_64x64x16` 内核。**优化要瞄准占比最高的那个**（通常是 GEMM/矩阵乘），别在 1% 的代码上较劲。

---

## 四、内核融合（Kernel Fusion）：少跑几趟仓库

对比两种 GeLU 写法：

- **手动分步**：`y = 0.5 * x`、`y = y * (1 + ...)`、`y = y * tanh(...)` → 启动了 **3 个 CUDA 内核**，每次都要把 `y` 写回 HBM 再读出来。
- **PyTorch 融合版 / 手写单 kernel**：一次算完 → 只 **1 个内核**。

实测（16384 维）：手动 8.1ms vs PyTorch 融合 1.1ms。差距就在"**少了几趟 HBM 往返**"。

### 4.1 用"算术强度"解释为什么快（重点）

**算术强度（Arithmetic Intensity, AI）= 有用计算量 / 搬运的数据量（FLOPs / Bytes）**。

- 每次从 HBM 读一个元素再写回，搬运 2 字节（fp16）或 4 字节（fp32），却只做了几次运算 → AI 极低 → **被内存带宽卡住（memory-bound）**。
- 融合后，中间结果留在寄存器/共享内存，不写回 HBM → 同样的运算，搬运量骤减 → AI 飙升 → **更靠近算力上限（compute-bound）**。

> **Roofline 模型直觉**：画一张"算力上限（屋顶）vs 内存带宽（斜坡）"的图。AI 低的程序贴在斜坡上（卡带宽），AI 高的程序顶到屋顶（卡算力）。融合就是把程序从斜坡"推"向屋顶。这也是为什么"融合常带来 2–5 倍提速"——它把瓶颈从慢的带宽挪到了快的算力。

### 4.2 为什么不全融合？融合的代价

- **寄存器压力**：每个中间值占寄存器，太多会压低"占用率（occupancy）"，反而跑不饱。
- **调试噩梦**：你调试的不是"加 + 乘"，而是 `FusedAddGeluMultiplyLayerNormKernel_v7`——出错时很难定位哪步算错。
- **硬件差异**：给 H100 调好的融合，到 A100/MI300 可能要换策略。

---

## 五、Triton：以"块"为中心的编程

手写 CUDA 要管线程、共享内存、同步，很累。**Triton（OpenAI 出品）** 让你用 Python 写，按"块（block/tile）"思考，底层的内存合并、共享内存、调度它自动搞定。

**三个典型例子**：

1. **Softmax（整行入块）**：一个 program 处理一整行，用 `tl.load / tl.max / tl.exp / tl.sum` 一次读写 HBM，不用跨块通信。
2. **行求和（超块分块循环）**：沿序列维循环切 tile，局部累加，最后 `tl.sum` 跨线程归约。
3. **矩阵乘（二维 Tiling）**：二维 grid，沿 K 维循环把 A/B 的小块搬进共享内存，用 `tl.dot` 累加；算术强度提升到 O(tile_size)，还能顺手融合 ReLU。朴素 O(MNK) 是内存受限，分块后每个元素只读一次。

实测（16384 维）：manual 8.1ms / PyTorch 1.1ms / 手写 CUDA 1.84ms / **Triton 1.85ms** —— 接近手写性能，但代码量小一个量级。

> **一句话**：Triton = 用 Python 的写法，拿到接近手写 CUDA 的性能。所以现在大量新算子（含 FlashAttention 的各种变体）都用 Triton 实现。
>
> **CUDA vs Triton 怎么选**：研究里的实测——Triton 有时比"手调 CUDA"还快 15%（因为自动 tiling 更贴合硬件）；但 CUDA 在"要绝对控制"的场景（实时渲染、特殊内存布局）仍不可替代。工程上：**默认 Triton，极致场景上 CUDA**。

---

## 六、torch.compile：编译器替你融合

PyTorch 2.0 的 `torch.compile` 能在你不写任何 CUDA/Triton 的情况下，**自动把简单算子融合**。实测 `torch.compile(manual_gelu)` 只要 1.47ms，比手写 CUDA 还快。

但要注意边界：

- 对"形状已知 + 简单融合 + GEMM"很香，白送 ~10% 提速。
- 对**极其依赖底层硬件特性**的算子（如 FlashAttention-3 利用 H100 的异步 WGMMA、FP8 张量核），编译器**自动发现不了**，还是得手写 Triton/CUDA。

> 工程建议：**默认先 `torch.compile`**，遇到新架构、复杂算子且 GPU 利用率上不去时，再上 Triton/CUDA 手动优化。不要每个模块都手写内核——那是过度工程。

---

## 七、算子优化的通用套路（记住这四条）

1. **提高算术强度**：用分块（Tiling）重复——把数据搬进共享内存复用，减少 HBM 读取次数。
2. **融合相邻操作**：把"逐元素 + 激活 + 归一化"合成一个 kernel，消灭中间结果的 HBM 往返。
3. **避免 warp 发散**：循环/分支里别让 32 个线程走不同路（否则一半在等另一半）。
4. **用对精度**：BF16/FP16 让 Tensor Core 火力全开；FP8 在 H100 上再翻倍。

> 数据参考（B200）：每 SM 寄存器 65536 个 / 256KB，HBM 带宽达 8 TB/s；每线程最多 255 个寄存器，寄存器压力过大会压低 occupancy，反而跑不饱。

---

## 八、CS336 英文原版 vs DIY-LLM 中文版 对照

| 维度 | CS336 L6（英文原版） | DIY-LLM 第7章（中文） |
|------|---------------------|---------------------|
| 侧重 | Triton 实操 + 算子性能建模 + 把 fused kernel 用到训练 | 从 CUDA 模型讲起，手写 GeLU→Triton→torch.compile 渐进对比 |
| 强项 | 给你"为什么这个 kernel 快"的量化模型 | 给你"从零写出一个快算子"的可运行代码路径 |
| 互补点 | 偏"原理与性能分析" | 偏"动手与代码" |

**学习建议**：跟着 DIY-LLM 把 `gelu.cu`、Triton softmax/matmul 各跑一遍，再用 CS336 的 roofline/算术强度去解释"为什么快"。

---

## 九、常见坑 & 面试深挖

1. **"内核融合为什么快？"** → 减少中间张量的 HBM 读写和 kernel 启动开销，本质是提高算术强度，把瓶颈从带宽挪到算力。
2. **"为什么线程数常比数据多、还要越界保护？"** → block 大小固定（如 1024），数据不一定整除，多余线程会读写非法内存，必须 `if (i < n)`。
3. **"Triton 和 CUDA 怎么选？"** → 先用 Triton（开发快、性能接近），极致场景再 CUDA。
4. **"torch.compile 是银弹吗？"** → 不是，复杂/硬件特化算子它优化不了。
5. **"profiler 显示时间花在 cudaDeviceSynchronize？"** → 说明代码里其实没怎么算 GPU（比如只 sleep），或没做同步导致计时错位。
6. **"什么是 memory-bound vs compute-bound？"** → 前者卡在显存带宽（逐元素操作），后者卡在算力（大矩阵乘）；融合把前者推向后者。

---

## 十、动手验证建议

- 用 `load_inline` 把 `gelu.cu` 编译跑通，故意去掉越界保护看会不会崩。
- 写个"分步 3 kernel" vs "融合 1 kernel"的基准，亲眼看到 HBM 往返的代价。
- 读 Triton 官方 tutorial 的 fused softmax，画一张"算术强度"草图理解 tiling 的收益。

---

## 参考与延伸

- Stanford CS336 Lecture 6: Kernels and Triton
- Datawhale《DIY-LLM》第7章：GPU 高性能编程
- Triton 官方教程（triton-lang.org）
- 论文：FlashAttention-3（利用 H100 异步特性的代表）
- 工具：Nsight Systems、PyTorch Profiler、torch.compile、Triton

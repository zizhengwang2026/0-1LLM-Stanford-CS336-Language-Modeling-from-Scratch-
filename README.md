# 🧠 0→1 LLM · Stanford CS336 从零构建大语言模型 · 学习笔记仓库

<p align="center">
  <img src="https://img.shields.io/badge/Course-CS336%20Language%20Modeling%20from%20Scratch-8A2BE2?style=for-the-badge" alt="Course" />
  <img src="https://img.shields.io/badge/Topic-LLM%20from%20Scratch-1f6feb?style=for-the-badge" alt="Topic" />
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge" alt="Python" />
  <img src="https://img.shields.io/badge/Built%20with-PyTorch-EE4C2C?style=for-the-badge" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Level-Graduate%20%2F%20Hardcore-FF6F00?style=for-the-badge" alt="Level" />
  <img src="https://img.shields.io/badge/Status-Learning%20%F0%9F%93%9A-2da44e?style=for-the-badge" alt="Status" />
</p>

<p align="center">
  <b>不只「会用」大模型，还要「能造」大模型。</b><br/>
  跟着斯坦福 CS 336，从 PyTorch 张量原语把 LLM 的每一行代码啃懂，并把「为什么」写清楚。
</p>

> **Course philosophy (original):** The idea draws from the course philosophy of CS 336 — just as an operating‑systems course has you write an OS from scratch, you implement all LLM components (tokenizer, Transformer, training, data, system optimization, alignment) yourself starting from PyTorch tensor primitives, and document the pitfalls encountered and the underlying trade‑offs.

---

## 📑 目录

- [🎯 这个仓库是什么](#-这个仓库是什么)
- [🗺️ 学习主线：Stanford CS 336](#-学习主线stanford-cs-336)
- [📚 配套中文教程：Datawhale《DIY-LLM》](#-配套中文教程datawhalediy-llm)
- [🔍 教程 vs CS 336：一致吗？](#-教程-vs-cs-336一致吗)
- [🌟 这课能做出什么？（教学版 vs 产品级）](#-这课能做出什么教学版-vs-产品级)
- [💼 学完有什么用？能投哪些 AI 岗位？](#-学完有什么用能投哪些-ai-岗位)
- [📂 仓库结构（学习路径）](#-仓库结构学习路径)
- [✅ 学习进度](#-学习进度)
- [🔗 资源链接](#-资源链接)
- [📝 说明](#-说明)

---

## 🎯 这个仓库是什么

这不是官方仓库，是个人学习笔记的公开集合。目标只有一句话：
**不只「会用」大模型，还要「能造」大模型。**

思路来自 CS 336 的课程哲学——像操作系统课让你从零写一个 OS 一样，
从 PyTorch 张量原语出发，自己实现 LLM 的全部组件（分词器、Transformer、训练、数据、系统优化、对齐），
并把踩过的坑和背后的权衡记下来。

---

## 🗺️ 学习主线：Stanford CS 336

课程官网：<https://cs336.stanford.edu/>
主讲：Percy Liang、Tatsunori Hashimoto（2026 起新增 Marcel Rød）

课程核心：学生只用 PyTorch 原语（**不能用** `nn.Transformer` / `nn.Linear` 等高层 API），
完整实现一个大语言模型。课程由 5 个大作业（Assignment）串起来：

| 作业 | 主题 | 内容 |
| --- | --- | --- |
| **A1 Basics** | 分词器 + Transformer | BPE 分词器 + 完整 Transformer（RoPE / 缩放点积注意力 / RMSNorm / FFN）+ AdamW + 训练循环，训一个最小语言模型 |
| **A2 Systems** | 系统与性能 | 性能分析与 benchmark；用 Triton 自己写 FlashAttention2；内存高效、分布式的训练代码（DDP） |
| **A3 Scaling** | 扩展定律 | 理解各组件在不同规模下的作用；拟合扩展定律（IsoFLOP / Chinchilla），预测最优模型大小 |
| **A4 Data** | 数据工程 | 把 Common Crawl 原始 HTML 转成可用预训练数据；质量过滤、去重、PII（个人信息）移除 |
| **A5 Alignment** | 对齐与推理 RL | 监督微调（SFT）+ 强化学习（推理）+ 可选 DPO 安全对齐 |

### 🎤 讲座主题地图（约 19 讲）

| 主题 | 内容 |
| --- | --- |
| 分词（1 讲） | BPE、SentencePiece |
| PyTorch 与资源核算（1 讲） | Roofline 模型、内存分析、性能分析 |
| 架构与超参（1 讲） | Transformer 变体、训练配置 |
| MoE 混合专家（1 讲） | 稀疏激活、路由策略 |
| GPU 与 TPU（1 讲） | 硬件架构、计算原理 |
| Kernel 与 Triton（1 讲） | GPU 编程、自定义算子 |
| 并行策略（2 讲） | 数据并行、张量并行、流水线并行 |
| 扩展定律（2 讲） | IsoFLOP、Chinchilla 法则 |
| 推理（1 讲） | KV cache、推测解码 |
| 评估（1 讲） | HELM、基准测试设计 |
| 数据（2 讲） | 数据源、清洗、过滤、混合 |
| 对齐（3 讲） | RLHF、DPO、RL 算法与系统 |
| 客座讲座（2 讲） | 业界专家分享 |

---

## 📚 配套中文教程：Datawhale《DIY-LLM》

- 在线教程：<https://datawhalechina.github.io/diy-llm/>
- 代码仓库：<https://github.com/datawhalechina/diy-llm>

中文友好、循序渐进，章节与 CS 336 主题基本对应，适合作为英文原版的前置 / 互补材料。
（本仓库参考的第 2 章「分词器」就系统讲了 BPE / WordPiece / Unigram / SentencePiece 的原理与训练流程。）

---

## 🔍 教程 vs CS 336：一致吗？

**主干一致、定位不同、可互补——但不是同一份材料。**

| 维度 | Datawhale《DIY-LLM》 | Stanford CS 336 |
| --- | --- | --- |
| 定位 | 中文友好、循序渐进的动手教程 | 斯坦福研究生硬核课，从零实现 LLM |
| 语言 | 中文 | 英文（视频 / 作业全英文） |
| 实现要求 | 借助 HF `transformers`、`sentencepiece` 等库 | 只用 PyTorch 张量原语，禁用高层 API |
| 覆盖主题 | 分词器、模型、训练、数据、对齐（偏基础） | 同主题但更深：加系统优化、GPU kernel、分布式、扩展定律 |
| 作业量 | 每章代码 + 练习 | 5 个大作业，代码量约 10 倍+ |
| 适合谁 | 自学打基础、中文母语者 | 有扎实 PyTorch / 系统底子、能投入大量时间 |

> **建议路径：DIY-LLM 中文打基础 → CS 336 英文原版啃实现。**

---

## 🌟 这课能做出什么？（教学版 vs 产品级）

> **一句话回答：** 按这课从 0 到 1 做出来的，是和豆包 / Kimi / DeepSeek **同物种的一个真实小模型**——
> 配方一模一样，只是小很多、能力弱很多。它不会变成「另一个豆包」，但你会彻底搞懂豆包到底是怎么被造出来的。

<div align="center">

| 维度 | 🧩 教学版（CS336 作业） | 🏭 产品级（豆包 / Kimi / DeepSeek） |
| --- | --- | --- |
| **架构** | Transformer 自回归 | Transformer 自回归（同） |
| **参数规模** | 几十 M ~ 几亿 | 千亿 ~ 万亿 |
| **训练数据** | 几 GB 小语料 | 万亿 token 全网多语料 |
| **算力投入** | CPU / 几张卡 | 数千~上万张 H100，烧数百万~千万美元 |
| **能力表现** | 通顺短句（玩具级） | 对话 / 推理 / 编码 / 多模态（产品级） |
| **是真模型吗** | ✅ 是，能生成文本 | ✅ 是 |

</div>

**一样的地方（最关键）：** 你做出来的和豆包 / DeepSeek 一样，都是 Transformer 自回归大语言模型——
同样的架构、同样的「预测下一个词」训练方式、同样的注意力机制。课程 A1 就要你从零训一个**真能生成文本的 LLM**，
它不是模拟、不是演示，是货真价实的模型。

**不一样的地方（体型差）：** 参数、数据、算力、能力差了好几个数量级。豆包 / Kimi / DeepSeek 这些名字，
其实指的是「模型 + 对齐 + 产品」一整包——CS336 覆盖的是**模型本身 + 训练系统 + 对齐（SFT/RLHF）**这一大块，
而它们作为「产品」还多了 App、对话界面、安全护栏、Agent 能力这些应用层东西。

> **所以：** 这课教的是「造它们的完整配方」。学会之后，简历上写「能从零实现 LLM」、投大模型训练 / AI Infra 岗，
> 就是实打实的硬证据。

---

## 💼 学完有什么用？能投哪些 AI 岗位？

CS336 学的是 AI 技术栈里**偏底层、偏「造模型 + 训推系统」**那一块（区别于最上层的应用 / Agent）。

**AI Infra 是什么？** = AI 基础设施（Infrastructure）。比喻：大模型像工厂生产的产品，
AI Infra 是**厂房 + 电网 + 物流系统**——它不生产产品本身，但负责让产品能大规模、低成本、稳稳地造出来和跑起来。
（具体即：GPU 集群调度、几千张卡一起训练、CUDA / Triton Kernel、推理优化 vLLM。）
CS336 的 **A2（系统 / FlashAttention / 分布式）+ A3（扩展定律）+ A4（数据）** 正是 AI Infra 的核心。

**对应招聘方向（2026 大厂热门）：**

| 岗位 | 对口内容 |
| --- | --- |
| 大模型训练 / 预训练工程师 | 千卡集群调度、Megatron / DeepSpeed / FSDP |
| 训练 Infra 工程师 | 分布式训练、算子优化、通信优化 |
| AI 系统 / 推理引擎优化 | CUDA、TensorRT、vLLM |
| 对齐工程师 | SFT / RLHF / DPO |

**跟 Agent 有关系吗？** 基本没关系，但底层功力是加分。Agent 在应用层（把现成模型当大脑接工具干活），
CS336 是「造大脑」——两层不同、不是一回事，这课确实不教 Agent。但学了 CS336，你更懂模型边界，
做 Agent 时不会把它当黑箱。

> 对「软体机器人 / 具身智能」方向：底层训练 / 系统能力是**稀缺加分项**。把 CS336 啃下来，
> 比只写「会用 LangChain 搭 Agent」更能体现「能造」的实力。

---

## 📂 仓库结构（学习路径）

按 CS 336 的 5 个 Assignment 组织笔记：

```
.
├── 01-tokenizer/        # 分词器：BPE / WordPiece / Unigram / SentencePiece
├── 02-transformer/      # 模型架构：注意力 / RoPE / RMSNorm / FFN
├── 03-training/         # 优化器(AdamW) / 训练循环 / 损失函数
├── 04-systems/          # 性能分析 / FlashAttention / Triton / 分布式
├── 05-scaling/          # 扩展定律：IsoFLOP / Chinchilla
├── 06-data/             # 数据清洗 / 过滤 / 去重 / PII 移除
├── 07-alignment/        # SFT / RLHF / DPO / GRPO
└── notes/               # 讲座重点解释 & 随想
```

---

## ✅ 学习进度

- [ ] **A1** 分词器与 Transformer 基础
- [ ] **A2** 系统优化（FlashAttention / 分布式）
- [ ] **A3** 扩展定律
- [ ] **A4** 数据工程
- [ ] **A5** 对齐与推理 RL

---

## 🔗 资源链接

- 课程官网：<https://cs336.stanford.edu/>
- 讲座视频（YouTube）：CS336 playlist
- 官方作业代码：<https://github.com/stanford-cs336>
- 中文教程《DIY-LLM》：<https://datawhalechina.github.io/diy-llm/>

---

## 📝 说明

个人学习笔记，非官方资料。内容基于公开课程材料与个人理解整理，欢迎讨论与指正。

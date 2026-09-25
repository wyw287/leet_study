# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这是什么

一个「LeetCode 式」的刷题框架:学习者只写 kernel(或 PyTorch 的 `forward`),
框架负责数据分配/搬运、编译、计时、正确性对拍、越界与竞态检查。

**用户通常在做题,不是在改框架。** 如果请求含糊,先分清是哪一种 —— 改框架代码和
做题目是完全不同的工作(后者只需要编辑 `solutions/<id>/solution.cu|py`)。

## 命令

`leet` 装在项目内 venv 里,**不在 PATH 上**:

```bash
.venv/bin/leet doctor            # 环境自检(nvcc / GPU / sanitizer / claude / torch / venv)
.venv/bin/leet list              # 题库与完成状态
.venv/bin/leet start 1           # 从模板生成解答文件并打印题面
.venv/bin/leet test 1            # 主命令:编译 + 判分
.venv/bin/leet test 1 --no-sanitize   # 跳过内存检查(快一些)
.venv/bin/leet validate --all     # ★ 这就是本项目的「测试套件」
.venv/bin/leet bench 1 --repeat 200
.venv/bin/leet stats
```

### 改完代码必须跑这个

```bash
.venv/bin/leet validate --all
```

它会为每道题做:文件齐备 → 题面有实质内容 → 准备基线 → **基线与参考解对拍** →
性能地板 → **模板必须失败** → **四个劣化解必须失败**。退出码 0 才算通过。

这是唯一能证明「改动没有打破已有行为」的手段 —— 本项目没有 pytest 之类的测试。
全量跑实测约 **4.7 分钟**(6 道题:5 道 CUDA × ~40s + 1 道 PyTorch × ~85s),
改单个模块时可以只跑 `.venv/bin/leet validate 1` 快速确认。

> 注意 PyTorch 题比 CUDA 题**更慢** —— 它没有编译,但要跑 6 个变体,每个都是独立
> Python 进程(import torch 1.6s)且在 CPU 上重算一遍 float64 参考解。
> 详见 `docs/design.md` 第六节末尾。

## 架构:两层的分界在哪

```
spec.yaml ─→ codegen ─→ harness.cu ─┐
                                     ├─→ nvcc ─→ 可执行文件 ─→ 同构 JSON ─┐
reference.cpp(CPU 参考解/oracle)───┘                                    │
baseline.cu ─→ 同一个 harness 再编一次 ─────────────────────────────────┤
compute-sanitizer ──────────────────────────────────────────────────────┤
                                                                        ↓
                                          judge.py(编排)→ report.py(渲染)
```

**第一层(科目无关)**:`spec.py` / `judge.py` / `report.py` / `bank.py` / `cli.py` /
`authoring/`。它们只依赖 `subjects/base.py` 里的抽象,**不含任何硬编码的文件名或语言特性**。

**第二层(科目相关)**:`subjects/cuda.py` 与 `subjects/pytorch.py`。每个科目声明自己的
文件名、入口键、劣化解生成方式、报告措辞。

**关键抽象是 `Artifact`**(`subjects/base.py`):一份实现"准备完成"后的不透明句柄。
CUDA 下是编译出的可执行文件,PyTorch 下就是 `.py` 源码本身 —— 所以 PyTorch 科目
**没有 codegen**(Python 可以自省,C++ 不行)。加新科目的完整契约见
`docs/design.md` 第七节。

**所有科目必须输出同一套 JSON 结构**(`outputs[].bad/max_abs_err/first_bad_idx`、
`perf.median_ms/gb_per_s`、`guards`),因为报告层与诊断层直接消费它。

## 不能削弱的设计(每一条都是踩坑换来的)

1. **L2 清空必须用「读」,不能用「写」。** 写会在 L2 里留下脏行,被计时的 kernel
   一开始就得先写回它们才能腾地方 —— 实测把 01 题的带宽从 1024 GB/s 压到 512 GB/s,
   而且**失去单调性**(元素更少的用例反而更慢)。`codegen.py` 的 `leet_flush_kernel` 是读的。

2. **区分度检查是出题可信的唯一保障。** `subjects/*.mutants()` 生成「空实现 / 全填 0 /
   全填 1 / 只写首元素」四种必然错误的实现,`authoring/validate.py` 要求**每一个都被判失败**。
   自动出的题最危险的失败模式不是参考解写错(那会让基线对拍失败,容易发现),
   而是**测试太弱、随便写写就过**,人眼看不出来。不要为了「让检查通过」而放宽它。

3. **性能只评级,不卡关。** 正确性决定通过与否,性能给 S/A/B/C。这是用户明确的选择
   (不打击初学者)。别把性能改成硬门槛。

4. **门槛必须物理可达。** `perf.grades` 要从实测倒推。反例:转置题基线只跑到
   267 GB/s 而峰值 1008 GB/s,理论最大加速比 3.8x —— 设 S=8x 是永远拿不到的废门槛。
   `leet validate` 会检查这个。**定门槛前先测量,不要凭感觉写。**

5. **访存瓶颈题用 `metric: bandwidth`,不是 `speedup`。** 这类题的朴素 CUDA 实现
   **本身就是最优算法**(都跑满带宽),加速比恒等于 1.0x,毫无区分度。01/02 就是这样。

6. **参考解固定叫 `reference(ctx)` / `void reference(RefCtx&)`** —— 它是 oracle,
   不是学习者的接口。学习者的入口名由 `spec.entry` 决定(CUDA 是 `kernel`+`launcher`,
   PyTorch 是 `function`)。别把两者混起来。

7. **不要新增「每个用例起一个进程」的路径。** 见下。

## 本机特性:起进程极贵(会决定你的设计)

| 操作 | 实测耗时 |
|---|---|
| **CUDA 上下文初始化**(空程序只调 `cudaFree(0)`) | **4.4 s** |
| **`nvidia-smi`**(任何参数,且不缓存) | **4.3 s** |
| 裸 python 启动 | 0.02 s |

后果:harness **一个进程跑完所有用例**(`--case all`)且稳定性重复在进程内
(`--verify-repeat`);`config.py` 对 GPU 探测做两级缓存(进程内 memo +
`~/.cache/leetstudy/gpu.json`),`Config.arch` 是**惰性属性**(只有真要编译时才探测)。
曾经因为每用例一个进程 + 每次探测 nvidia-smi,`leet test` 要 49 秒、`leet list` 要 4.6 秒;
现在是 ~11 秒(源码未变时)和 0.16 秒。**别把这些优化改回去。**

纯 Python 侧不受此影响(起进程 0.02 秒),别把 CUDA 的经验套到解释型科目上。

## 环境约束

- **Python 3.9** —— 不能用 `X | None` 写法;新文件头部加 `from __future__ import annotations`
- **默认 `python3` 是 `/home/user49/anaconda3/bin/python3`(他人目录)**,不要往里装包;
  一律用项目内 `.venv`
- **torch 必须锁 cu121**(驱动 535.183.01 只支持到 CUDA 12.2)。PyPI 上 `torch<=2.4.1`
  的默认 wheel 是 cu121,`>=2.5` 是 cu124+ 会不可用
- **下载分流**:国内镜像(清华)直连、**不要挂代理**;国际站点(pypi.org /
  download.pytorch.org)挂 `http://127.0.0.1:11451`。挂错方向会让 pip 直接卡死
- `load_inline`(自定义 CUDA 算子题)需要 `.venv/bin` 在 **PATH** 上 ——
  torch 是按「PATH 上有没有 ninja 可执行文件」判断的,光 import 不算

## `leet new` 会拉起一个嵌套的 Claude 实例

它会**读写本仓库**(`problems/`、`solutions/`、`build/`)、自己跑 `leet validate`,
而且**与你共用同一个 project memory 目录**(所以它可能写出你没写过的 memory)。

它看不到你的操作。曾发生过:嵌套实例观测到外层会话清理 `solutions/` 的动作后,
把自己在同一时段写入的文件也归因给了「另一个并发会话」。**排查文件变动时先排除
自己这一侧,再看那个时间窗里有没有 `leet new` 在跑。**

它留下的 `solutions/<id>/solution.cu` 是**剧透**,要删掉;它记录的 `progress.json`
成绩不算用户的,要重置。

## 出题与讲评走的也是本地 claude

`authoring/agent.py` 用 `claude -p` headless 调用,**prompt 走 stdin**
(不能走 argv:`--allowedTools` 是变参选项会把它吃掉)。**不要硬编码 `--model`** ——
本机是自定义模型接入,`--model haiku` 会报 `unrecognized_model`。

出题提示词按科目分(CUDA / PyTorch 各一套),范例题直接从 `problems/` 里读 ——
这样提示词与真实 spec 格式永远同步,不会漂移。

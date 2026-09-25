# leet_study —— LeetCode 式的编程刷题框架

把一个知识点拆成一道道有判分的题,让你只写**核心的那几行**,剩下的自动完成。

目前支持两个科目:

| 科目 | 你写什么 | 题号前缀 |
|---|---|---|
| **CUDA** | kernel + 启动配置(数据分配/搬运/计时/对拍/越界检查全自动) | `01-` `02-` … |
| **PyTorch** | 一个 `forward(ctx)`(用 torch 算子,或写自定义 CUDA 算子) | `py01-` … |

```
$ leet test 01
题目 01-vector-add  向量加法
难度 ●○○○○   标签 elementwise memory-bound
──────────────────────────────────────────────
编译    ✓  nvcc -O3 -lineinfo  2.4s
正确性
          exact          n=16777216  ✓  max_err 0
          ragged         n=16777259  ✓  max_err 0
          exact 稳定性   重复 3 次   ✓
          ragged 稳定性  重复 3 次   ✓
内存    ✓  memcheck 0 errors  5.0s
性能
  exact   0.180 ms  1117 GB/s  (111% 峰值)  ██████████ S
  ragged  0.180 ms  1117 GB/s  (111% 峰值)  ██████████ S

判定   ✅ 通过
```

## 为什么做这个

初学 CUDA 的人,在真正练到 kernel 之前,大量时间被消耗在**脚手架**上:配环境、
写 `cudaMalloc`/`cudaMemcpy` 样板、调 nvcc 参数、搭计时框架、以及对着一个越界访问
查两小时。

这些工作对"学 CUDA"本身是噪音。这个框架把它们全部接了过去:

| 通常要自己做的 | 这里 |
|---|---|
| `cudaMalloc` / `cudaMemcpy` / `cudaFree` | 框架负责,你直接拿到 device 指针 |
| nvcc 参数(`-O3 -arch=sm_89 -lineinfo`) | 框架按本机 GPU 自动决定 |
| 写计时循环、多次预热、取中位数 | 框架做,而且**每次迭代前清空 L2** |
| 自己写参考实现对拍、定容差 | 框架用 CPU 参考解对拍 |
| 排查越界访问 / 漏 `__syncthreads` | 哨兵区 + memcheck + racecheck 自动跑 |
| 判断"我这个算快还是慢" | 相对基线的加速比 / 有效带宽占峰值百分比 |

## 快速上手

```bash
# 一次性环境(项目内 venv,不污染系统的 anaconda)
python3 -m venv .venv
.venv/bin/pip install -e .

# 自检:nvcc / GPU / compute-sanitizer / claude / Python 环境
.venv/bin/leet doctor

# 浏览题库 → 读题 → 生成工作区 → 编辑 → 判分
.venv/bin/leet list
.venv/bin/leet show 1
.venv/bin/leet start 1
$EDITOR solutions/01-vector-add/solution.cu
.venv/bin/leet test 1
```

代码是能直接改的教科书:`solutions/` 下是你的工作区,`problems/` 下是题目定义。

## 文档地图

| 你想… | 看这个 |
|---|---|
| **做题** | 本文件 + 题面(`leet show <题号>`) |
| **出一道具(或改题)** | [`docs/authoring.md`](docs/authoring.md) |
| **理解为什么这样设计** | [`docs/design.md`](docs/design.md) |
| **遇到报错 / 装不上** | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| **让 AI 助手接着改这个项目** | [`CLAUDE.md`](CLAUDE.md) |
| **改配置** | [`config.yaml.example`](config.yaml.example) |

`docs/design.md` 里全是实测数字(`nvidia-smi` 4.3 秒、L2 脏行把带宽从 1024 压到
512 GB/s 这类),想知道「为什么这么做」的时候翻它。

## 命令

| 命令 | 作用 |
|---|---|
| `leet doctor` | 环境自检(nvcc / GPU / sanitizer / claude / torch / ninja / venv) |
| `leet list [--tag T] [--diff N] [--status todo\|done]` | 题库浏览 + 完成状态 |
| `leet show [题号]` | 读题面 |
| `leet start <题号> [--force]` | 从模板生成解答文件 |
| `leet test [题号] [--case C] [--no-sanitize] [--race] [--gpu N] [-v]` | **主命令**:编译 + 判分 |
| `leet bench [题号] [--repeat N]` | 只测性能,重复更多次 |
| `leet review [题号] [--fresh]` | 让本地 claude 讲评你的 kernel |
| `leet new "<需求>" [--subject pytorch] [--count N]` | 让本地 claude 自动出题(含自验证与修复回路) |
| `leet validate [--all\|<题号>]` | 题库健康检查 |
| `leet stats` | 学习进度看板 |
| `leet clean` | 清理编译产物 |

### 题号可以省略

`test` / `bench` / `review` / `show` 的题号都是可选的 —— 省略时**自动选最近编辑过的
那份解答**,并且会明确告诉你是谁:

```
$ leet test
未指定题号 → 最近编辑的解答: 03-matrix-transpose (2 分钟前)
```

于是常见的工作流变成:

```bash
leet start 5      # 开始一道新题
$EDITOR solutions/05-conv2d/solution.cu
leet test         # 不用敲题号
leet test         # 改了再跑,还是不用敲
```

判据是解答文件的修改时间,所以 `leet start` 之后紧接着 `leet test` 也符合直觉。
想指定别的题就正常传题号。

题号本身也支持 `1` / `01` / `vector` 这类宽松写法。

> `leet` 装在项目内 venv 里,**不在 PATH 上**。嫌麻烦就加个别名:
> `echo "alias leet='<仓库路径>/.venv/bin/leet'" >> ~/.bashrc`

### `review` 会复用上次 `test` 的结果

`leet test` 会把判题结果存进 `build/<题号>/last_verdict.json`。如果之后跑
`leet review`,而**解答、题目的 spec、参考解、基线都没变**,就直接复用那份数据,
省掉一次判题(实测 11.3 s → 4.5 s,省下的时间在 claude 那几分钟面前不算大,
但至少不用白等一次 CUDA 上下文初始化):

```
$ leet review
复用上次 test 的结果(2 分钟前;解答自那之后没改过)—— 想重新判题就加 --fresh
```

想强制重测加 `--fresh`。判据是**内容哈希**而不是文件修改时间 —— `touch`、
`git checkout`、编辑器无改动保存都会动 mtime,但那些情况下缓存依然有效。

**`test` 绝不复用缓存。** `review` 把判题数据当作"给模型的上下文",略有陈旧无伤大雅;
而 `test` 的数据是用来**评级**的 —— 复用就等于把成绩变成"上次测的时候机器忙不忙"
的产物。同一个理由也让本框架拒绝了缓存基线耗时(`docs/design.md` 第六节)。

选项也必须一致:如果缓存那次是 `leet test --no-sanitize`(没做内存检查),
而 `leet review` 想要消毒报告,就会重测一遍。

## 耗时参考

| 命令 | 典型耗时 |
|---|---|
| `leet list` / `show` / `stats` | ~0.2 s |
| `leet test`(首次或改了源码) | ~14–21 s(2 次 nvcc 编译 + 3 个 CUDA 进程) |
| `leet test`(源码未变,编译缓存命中) | ~11 s |
| `leet validate <题号>` | 实测:CUDA 题 ~40 s;PyTorch 题 ~85 s(原因见下) |
| `leet validate --all` | 实测 ~4.7 分钟(6 道题)—— 改框架后应当跑它 |
| `leet new` | 20–40 分钟 |

> **为什么 PyTorch 题的 validate 反而比 CUDA 题慢**(85s vs 40s),尽管它不需要编译?
> 因为区分度检查要跑 **6 个变体**(基线 + 模板 + 4 个劣化解),每个都是一次独立的
> Python 进程 —— 要 import torch(1.6 s),并在 CPU 上以 float64 重算一遍参考解
> (8192×8192 的 softmax,五遍 67M 元素的运算)。
> CUDA 侧只有 2 个变体跑完整用例,且一个 harness 进程能跑完所有用例。
> 教训:**「不需要编译」不等于「快」** —— 成本会转移到进程启动、库导入和重算 oracle 上。
> 详见 [`docs/design.md`](docs/design.md) 第六节末尾。

## 工作原理

一份 `spec.yaml` 描述一道题的全部机器可读信息,**由它生成 harness**:

```
spec.yaml ──┐
            ├─→ codegen ─→ ctx.h + harness.cu ─→ nvcc ─→ 可执行文件
template.cu ┘     (LaunchCtx: device 指针 / RefCtx: host 指针)
reference.cpp ──────────────┘  (CPU 参考解,ground truth)
baseline.cu ──→ 同一个 harness 再编译一次 ──────→ 加速比的分母
```

关键机制:

- **同一份 harness 编译两次**(用户解 / 基线),计时与校验逻辑逐字相同 → 加速比可比
- **哨兵区**:每块缓冲前后各留 4096 个元素填特殊值。越界写会踩到哨兵,立刻报出
  「往前越界了 N 个」还是「往后越界了 N 个」
- **毒值**:out 缓冲在校验前填 NaN(浮点)/ `0x5EED5EED`(整数)。「kernel 根本没写
  输出」会表现为清一色的毒值,而不是碰巧等于某个合法值
- **L2 清空**:每次计时迭代前读一遍大于 L2 的缓冲。4090 有 72MB L2,而一道题往往
  只有几 MB —— 不清缓存测的是**缓存带宽**,数字能虚高好几倍
- **launch 后置检查**:`cudaGetLastError()` + `cudaDeviceSynchronize()`,把
  `invalid configuration argument`、`illegal memory access` 这类错误在源头抓出来

### 评分:两种指标,按瓶颈选

- **访存瓶颈题**(向量加法、转置):朴素 CUDA 实现往往**就是最优算法**,加速比
  恒等于 1.0x,毫无区分度。这类题按**有效带宽占峰值百分比**评分。
- **计算瓶颈/并行度不足的题**(归约、矩阵乘):按**相对基线的加速比**评分。

门槛写在 spec 里,而且**必须物理可达** —— 比如转置题基线只跑到 267 GB/s、峰值是
1008 GB/s,那么理论最大加速比就是 3.8x,设 S=8x 是永远拿不到的废门槛。
`leet validate` 会检查这一点。

性能**只评级、不卡关**:正确性决定通过与否,性能给 S/A/B/C。初学阶段被性能门槛
卡住很打击人,优化动力靠评级就够了。

### 题库自验证(区分度检查)

一道题最容易出的问题不是"参考解写错了"(那会让基线对拍失败,容易发现),而是
**测试太弱**:随便写点什么都能过。这种题做起来毫无意义,又很难靠人眼发现。

`leet validate` 的做法是注入一组**必然错误**的实现,要求每一个都被判失败:

| 注入的劣化解 | 若它竟然通过了,说明 |
|---|---|
| 空实现 | 漏写输出也能过 —— 测试形同虚设 |
| 输出全填 0 | 期望值恰好是 0 —— 用例数据退化 |
| 输出全填 1 | 期望值是常数 —— 用例数据退化 |
| 只写每块首元素 | 只检查了首元素 |

这组劣化解都是从 spec **泛化生成**的(只需要知道有哪些输出缓冲),所以对任何题目
都适用 —— 这正是它能用于全自动出题的原因。

### 全自动出题

```bash
leet new "出一道关于 bank conflict 的题"
leet new --count 3 "关于 bank conflict 的题,难度递进"
leet new --subject pytorch "出一道 LayerNorm 的题"
```

本地 claude 会读完 spec 格式说明 + 一道已通过的范例,自己写文件、编译、跑
`leet validate` 迭代。完成之后**框架再独立验证一遍**(不采信它的自我声明),
不过关就把失败报告回喂给它修,最多若干轮。

**`--count N` 是逐道开独立会话**,不是让一次会话出 N 道 —— 一道要 25~40 分钟,
塞进一次会话会撞超时。逐道开还有两个好处:每道题的上下文是干净的,而且
前一道的产出会自动进入后一道的「不要重复出这些」列表。跑完会分类汇总哪几道可用:

```
✓ 可用:2 道  06-bank-conflict, 07-shared-reduce
✗ 需处理:1 道  08-bad-example
```

出题者的工具权限走白名单(`Read/Write/Edit/Glob/Grep` + `leet`/`nvcc`/
`compute-sanitizer` 三个命令),不使用 `--dangerously-skip-permissions`。

## 科目:同一套框架,多种语言

一道题属于哪个科目由 `spec.yaml` 的 `subject` 字段决定(默认 `cuda`)。科目决定
**谁来准备代码、怎么运行、怎么检查**:

| 科目 | 你写什么 | 准备阶段 | 内存检查 | 题号前缀 |
|---|---|---|---|---|
| `cuda` | `.cu` 里的 kernel + 启动配置 | nvcc 编译 | memcheck / racecheck | `01-` |
| `pytorch` | `.py` 里的 `forward(ctx)` | 语法预检(无需编译) | 可选(仅自定义算子题需要) | `py01-` |

**用户接口各科目不同,但判题数据格式完全一致** —— 报告层与诊断层不区分科目。

### PyTorch 科目

```yaml
subject: pytorch
entry:
  function: forward            # 用户要实现的函数名
```

`forward(ctx)` 的两种输出写法都支持:

```python
def forward(ctx):
    return torch.softmax(ctx.x.t(), dim=1).t()      # 惯用写法,推荐
    # 或者:
    ctx.out.copy_(...)                               # 预分配缓冲,但多一次全量拷贝
```

为什么两种都留:`ctx.out` 是框架预分配的、带**哨兵区**的缓冲,能拿到
「往前/往后越界了多少个元素」这种精度的诊断 —— 这对**自定义 CUDA 算子题**很有用。
但它会比 `return` 多一次全量访存,在访存瓶颈题上足以让正确解法拿不到应有评级
(py01 实测:3.30x → 1.87x)。所以让题目作者和解答者自己选。

判题用的是 **CPU 上的纯 torch 参考解**(与 CUDA 侧同理:oracle 不能在 GPU 上
自己引入竞态/越界),计时用 `torch.cuda.Event`,每次迭代前同样清 L2。

### 加一个新科目

实现 `subjects/base.py` 里的 `Subject` 基类,在 `subjects/__init__.py` 的
`_REGISTRY` 里登记即可。需要提供:

- `prepare_variant(problem, impl_src, out_dir, label) -> BuildResult(artifact=…)`
- `run_case(artifact, case, perf, …) -> CaseResult`(输出的 JSON 结构要和 CUDA 侧一致)
- 可选:`mutants()`(劣化解生成)、`sanitize()`、文件名、`build_label`

`Artifact` 是不透明句柄 —— CUDA 下是编译出的可执行文件,PyTorch 下就是 `.py` 源码
本身。Triton 只需复用 PyTorch 科目的 runner(Triton 的 JIT 对框架透明)。

完整的接口契约、`CaseResult.raw` 的 JSON 结构、以及可选方法的语义,
见 [`docs/design.md`](docs/design.md) 第七节。

## 目录结构

```
README.md          本文件:是什么、怎么用
CLAUDE.md          给 AI 助手的项目约定(用什么命令、哪些设计不能改)
docs/
  design.md        设计决策与理由 —— 为什么这样实现(带实测数字)
  authoring.md     出题指南:文件契约、spec 字段参考、定门槛的方法
  troubleshooting.md  排错手册:症状 → 原因 → 处理
config.yaml.example  配置样例(全部字段可选,改哪项取消注释即可)
src/leetstudy/     框架代码(各模块头部有说明)
problems/<id>/
  spec.yaml        机器可读定义:科目、接口契约、用例、容差、评分门槛
  problem.md       中文题面:讲解 + 提示 + 陷阱 + 思考题
  template.cu     ┐
  reference.cpp   ├ CUDA 科目的四个文件(由 spec.subject 决定实际用哪套)
  baseline.cu     │
  ────────────────┘
  template.py     ┐
  reference.py    ├ PyTorch 科目的三个文件
  baseline.py     ┘
solutions/<id>/
  solution.cu      你的解答(扩展名随科目)
build/             编译产物(可 leet clean 清掉)
```

### 想读代码

按这个顺序读最省力:

1. `problems/01-vector-add/spec.yaml` —— 先看一道题长什么样
2. `src/leetstudy/spec.py` —— 数据模型(字段的权威定义)
3. `src/leetstudy/subjects/base.py` —— 科目接口与 `Artifact` 抽象
4. `src/leetstudy/codegen.py` —— 生成的 harness 长什么样(头部注释说明了设计)
5. `src/leetstudy/judge.py` —— 编排:准备 → 跑用例 → 计时 → 消毒 → 诊断 → 评级

## 配置

复制 [`config.yaml.example`](config.yaml.example) 成 `config.yaml` 即可 ——
**所有字段都是可选的**,默认值就是「自动」,改哪项取消哪项的注释。
临时改动用环境变量(全部带 `LEETSTUDY_` 前缀)更方便。

优先级:配置文件 < 环境变量 < 命令行参数。

| 配置项 | 环境变量 | 说明 |
|---|---|---|
| `arch` | `LEETSTUDY_ARCH` | 目标架构。**默认惰性探测** —— 只有真要编译时才跑 nvidia-smi |
| `gpu` | `LEETSTUDY_GPU` | 指定物理 GPU。留空则自动挑最闲的 |
| `nvcc` / `sanitizer` / `claude_bin` | `LEETSTUDY_NVCC` 等 | 工具路径 |
| `claude_model` | `LEETSTUDY_CLAUDE_MODEL` | 出题/讲评用的模型。**默认不指定**,继承你当前的 claude 配置 |
| `peak_bandwidth_gbps` | `LEETSTUDY_PEAK_BANDWIDTH_GBPS` | 覆盖峰值带宽(默认按 GPU 名称查内置表) |
| `claude_allowed_tools` | — | 出题者的工具白名单(默认只放开读写 + leet/nvcc/sanitizer) |
| `compile_timeout` / `run_timeout` / `sanitizer_timeout` / `author_timeout` | 同名大写 | 各类超时(秒) |

多卡机器上,框架默认自动挑**最空闲**的一张;若已设置 `CUDA_VISIBLE_DEVICES` 则不干预。
**同一次判题内 GPU 选择是固定的** —— 所有用例必须落在同一张卡上,加速比才可比。

## 已知限制

- **报出的带宽可能略超标称峰值**(比如 4090 上测到 1117 GB/s)。这是事件计时的
  固有现象:停止计时的瞬间,最后一批写还在 L2 里没落盘,少算了这部分显存流量。
  不影响相对比较,`leet test` 会在出现时注明。
- **单用例耗时的可评级下限是 20µs**。更小的用例会被跳过评级(仍参与正确性判定),
  因为那个量级上测的是启动开销和计时器抖动。
- **这台机器上起 CUDA 进程很贵**:CUDA 上下文初始化实测 4.4 秒(空程序亦然),
  `nvidia-smi` 单次 4.3 秒。框架已经据此做了缓存与进程合并(`leet test` 从 49 秒
  降到 11–21 秒),但**任何新增的「每个用例起一个进程」路径都会立刻退化**。
  详见 [`docs/design.md`](docs/design.md) 第六节。
- **只支持 `cuda` 与 `pytorch` 两个科目**。判题层已经抽象好了
  (`subjects/base.py` 的 `Subject` 基类 + `Artifact` 句柄),加 Triton 只需复用
  PyTorch 科目的 runner。见 [`docs/design.md`](docs/design.md) 第七节。
- **`--race`(racecheck)很慢**,只在小用例上跑。默认只在检测到「同一输入重复跑
  结果不一致」这个竞态特征时才自动触发。
- **PyTorch 科目没有哨兵区**(除非解答选择写进 `ctx.out`)。纯 PyTorch 算子不可能
  越界写用户张量,所以这个损失可接受;需要越界诊断的场景用 `ctx.out` 写法,
  框架会照常检查。

## 关于题库来源

框架不包含任何第三方题库。题目全部为本项目自建(手写或由 `leet new` 生成)。

调研过现有平台,记录如下以免将来误用:

| 项目 | 许可证 | 可否复用 |
|---|---|---|
| srush/GPU-Puzzles | MIT | ✅ |
| sinatrasC/pmpp-eval(53 题) | MIT | ✅ 最完整的可复用题库 |
| stanford-cs149/asst5-kernels | MIT | ✅ |
| gpu-mode/popcorn-cli、lectures | MIT / Apache-2.0 | ✅ |
| gpu-mode/reference-kernels | 受限(Researcher Reciprocity) | ⚠️ 条件性 |
| **AlphaGPU/leetgpu-challenges** | **CC BY-NC-ND 4.0** | ❌ 禁商用、禁衍生 |

## 学习路径

按题目难度递进刷即可。规划中的完整路径(对应 PMPP 教材 / UIUC ECE408 的经典顺序):

| # | 题目 | 科目 | 考点 | 难度 |
|---|---|---|---|---|
| 01 | 向量加法 | cuda | 线程索引、边界、coalescing | 1 |
| 02 | SAXPY | cuda | 标量参数、内存带宽 | 1 |
| 03 | 矩阵转置 | cuda | 二维索引、共享内存分块、bank conflict | 2 |
| 04 | 归约求和 | cuda | 树形归约、同步与竞态、warp shuffle | 3 |
| 05 | 二维卷积 | cuda | halo 边界、共享内存分块、寄存器分块 | 3 |
| py01 | 列方向 Softmax | pytorch | 归约维的访存方向、布局敏感性 | 2 |
| 06 | 矩阵乘(分块) | cuda | shared memory tiling、算术强度 | 3 |
| 07 | 直方图 | cuda | 原子操作、私有化 | 3 |
| 08 | 前缀和 | cuda | Kogge-Stone / Brent-Kung、double buffering | 4 |
| 09 | Softmax(手写 kernel) | cuda | 数值稳定、块内归约、融合 | 3 |
| 10 | LayerNorm | cuda | 两遍 vs Welford、算子融合 | 4 |
| 11 | 寄存器分块矩阵乘 | cuda | register blocking、float4 向量化 | 5 |
| 12 | Flash Attention | cuda | online softmax、tiling 融合 | 5 |

未完成的题目可以用 `leet new` 生成,再人工过一遍。

两条轨道的配合也值得注意:`py01` 让你先摸到"归约维放错地方"的代价(纯 PyTorch
就能修),而 `09` 要你亲手写 kernel 把同一件事做到 2 遍访存的下限 —— 先看到差距,
再去填平它。

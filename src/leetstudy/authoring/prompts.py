"""出题与讲评的提示词模板。

这里最关键的是把「框架的物理约束」讲清楚 —— 出题最容易犯的错不是代码写错,
而是设计出一批**没法评级**的题(用例太小、门槛物理上不可达),或者**测试太弱**的题。
下面每一条约束都对应一次真实的踩坑。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

SPEC_SCHEMA_PYTORCH = r"""
## spec.yaml 完整字段说明(PyTorch 科目)

```yaml
id: <目录名,必须与目录名完全一致,建议 py 前缀,如 py02-layernorm>
title: <中文标题>
subject: pytorch                # 必须写,否则会被当成 cuda 科目
difficulty: 1..5
tags: [<英文标签>]
concepts: [<中文考点,写给学习者看>]
statement: problem.md

# ---- 接口契约:决定 forward(ctx) 里 ctx 有哪些字段 ----
buffers:
  - name: <Python 标识符>
    dtype: f32 | f64 | i32 | i64 | u32 | u8      # 直接映射到 torch dtype
    shape: [<参数名或整数字面量>]                 # 空列表 [] 表示标量
    role: in | out | scratch
    fill: uniform | positive | randint | zero    # 仅 role=in 有效
    #   uniform  → [-1, 1)
    #   positive → (0, 1]   给 log / 开方 / 除法等定义域为正的题用
    #   randint  → 0..99 的整数
    #   zero     → 全 0
params:
  - {name: <标识符>, dtype: i32 | f32 | ...}
    # 标量参数,会作为字段出现在 ctx 上。形状表达式里可以引用它们。
    # 浮点类型的参数可以给小数取值(如 alpha: 0.5、eps: 1e-5)。

entry:
  function: forward
    # 学习者要实现的函数名。注意:参考解用的名字是**固定的** reference(ctx),
    # 与 entry.function 无关 —— 它是 oracle,不是学习者的接口。

cases:
  - {name: <用例名>, params: {<每个参数都给一个取值>}}
    # 取值必须是正数;整数类型的参数必须是整数(会被用来算张量形状)

verify:
  atol: 1e-5      # 过宽的容差会让错误实现轻松通过,框架会拒绝 atol>1.0 / rtol>0.1
  rtol: 1e-4
  repeat: 3       # 每个用例重复跑几次。>1 用于捕捉不确定性

perf:
  enabled: true
  bound: memory | compute
  metric: bandwidth | speedup
    #   bandwidth —— 按「有效带宽占峰值百分比」评级。**访存瓶颈题的基线往往
    #                已经是"用对了 API 但撞上坏布局",用加速比没有区分度。**
    #   speedup   —— 按「相对基线的加速比」评级。纯 PyTorch 题大多用这个:
    #                基线是一段自然的、但慢的写法,优化空间来自布局/融合。
    #   metric 不写时按 bound 推断:memory→bandwidth。
  repeat: 30
  warmup: 5
  grades: {...}   # 含义随 metric 变:
                  #   bandwidth → 占峰值百分比,如 {B: 50, A: 70, S: 85}
                  #   speedup   → 倍数,如 {B: 1.5, A: 2.2, S: 3.0}
  flush_l2: true  # 默认开启,不要关。4090 有 72MB L2,不清缓存测的是缓存带宽。

sanitize:
  memcheck: false
    # 纯 PyTorch 题(只用 torch 算子)**必须设 false** —— 没有自定义 kernel,
    # compute-sanitizer 只会把整个 torch 库拖慢几十倍而毫无所得。
    # 只有当题目要求学习者写**自定义 CUDA 算子**(如 load_inline)时才设 true。
```

## 运行环境(必须按这个来设计,否则题目会不可用)

- GPU: NVIDIA RTX 4090 ×5,**理论峰值带宽 1008 GB/s**;torch 2.4.1+cu121
- 框架自动在每次计时迭代前**清空 L2**(读一个 256MB 的缓冲),所以测到的是真实显存带宽
- 计时用 `torch.cuda.Event`,分辨率约 0.5µs。因此:
  - **总耗时不到 20µs 的用例会被自动跳过评级**。注意纯 PyTorch 题的算子很容易
    跑得很快 —— 用例规模要给足。经验值:访存型题目数据量要 ≥ 100MB
    (如 32M 个 f32 = 128MB,约 130µs)。
  - 但也不能太大:基线耗时超过 5 秒会让做题体验崩坏。
- 参考解在 **CPU** 上用 torch 跑(与 CUDA 侧同理:oracle 必须自身不可能有 GPU
  侧的错误)。用 float64 计算以保证接近真值。
- 学习者的 `forward(ctx)` 可以 `return` 张量,也可以写进 `ctx.<out>` 再 return None。
  两种都支持。注意:`ctx.out` 是预分配缓冲,写它要多一次全量拷贝 —— 对访存瓶颈题
  这个代价很显著,出题时要把这件事讲清楚。
- 框架在 `out` / `scratch` 缓冲前后各留 4096 个元素的**哨兵区**填特殊值;但只要学习者
  选择 `return` 张量,哨兵区就用不上了(PyTorch 自己分配)。纯 PyTorch 算子也不可能
  越界写用户张量,所以这个损失可接受 —— **但题面里不要承诺"越界会被抓到"**。
"""

REQUIREMENTS_PYTORCH = r"""
## 硬性要求

### reference.py —— 标准答案(oracle)
- 只定义 `def reference(ctx):`,用 `ctx.` 访问所有张量
- **必须完全正确**。它是判分的基准,写错了整道题就废了
- 在 **CPU** 上跑。浮点计算请先 `.to(torch.float64)`,最后再 `to(原 dtype)` 写回
- 不要写 `forward` 这个名字 —— 那是学习者的接口

### baseline.py —— 性能基线(加速比的分母)
- 必须定义 `def forward(ctx):`(与 entry.function 同名)
- **必须正确**(框架会检查:基线与参考解在**所有**用例上必须一致)
- 要是**自然、直白**的写法 —— 它是学习者要超越的对象。
  纯 PyTorch 题的基线通常是"用对了 API、但布局或写法不够好",
  而不是"故意写烂"。这样学习者学到的是**真实的经验判断**,不是应付题目。
- 基线必须真的**有优化空间**。定 grades 之前先实测一遍:
  如果基线已经是理论最优(比如就是 `torch.softmax` 的正确用法),
  那这道题不该出,或者要改成 `bandwidth` 指标并诚实说明"没有优化空间"。
- **不要**用 `ctx.out.copy_(...)` 写基线 —— 那次拷贝是额外开销,
  会让基线显得比实际更慢。直接 `return` 张量。

### template.py —— 给学习者的骨架
- 只写 `import torch` + `def forward(ctx):` 的**空壳**,里面是 TODO 注释
- **必须不完整到无法通过测试** —— 框架会检查「空模板必须被判失败」
- 注释要有教学价值:讲清这道题的思路、常见陷阱、以及为什么

### problem.md —— 题面
- **中文**,面向会写 PyTorch 但没做过性能优化的人
- 结构:题目描述 → 为什么这道题重要 → 慢在哪(要有**实测数字**)→ 解法思路 →
  陷阱 → 评分说明 → 3~4 个思考题
- 解释要落到**物理原因**上(访存模式、算子融合、中间张量物化、同步点),
  不要只说「建议这样写」
- **诚实**:如果某个"优化"实测下来没有用甚至更慢,就直说 —— 那往往是最有价值的
  一段内容。不要写你没验证过的断言。

### cases
- 至少一个用例大到可评级(见上文 20µs 规则,纯 PyTorch 题尤其要注意)
- **必须包含边界用例**:非 2 的幂、质数规模、或者会让某一维退化的形状
- 形状的选择本身要能说明问题(比如"瘦高" vs "方阵"会得出不同结论 —— 那就都放上)
"""

WORKFLOW_PYTORCH = r"""
## 你该怎么做

**按这个顺序做,不要跳步。** 写文件很快,而测量与迭代很慢 —— 先把该写的写完。

> **不要通读框架源码**(`src/leetstudy/` 下的文件)。出题需要的信息都在上面。

### 第一步:想清楚(不要动笔)
这道题的**性能瓶颈**是什么?纯 PyTorch 题的优化空间通常来自这几处:
   - 归约发生在访存不连续的维度上
   - 中间结果被反复物化(该融合的没融合)
   - 多余的全量拷贝(布局转换、dtype 转换、`.contiguous()` 用错地方)
   - 同步点(`.item()` / `.cpu()` / `.numpy()`)打断了 GPU 流水
   - 算子本身是多个小内核,而可以合并成一次

先想清楚是哪一类,再决定基线怎么写、指标怎么选。

### 第二步:一口气写完 4 个文件
1. `spec.yaml`
2. `reference.py`(CPU 参考解,float64)
3. `baseline.py`(自然但慢的写法)
4. `template.py`(空壳)
5. **`problem.md`(题面)** —— 一定要在这一步写完,不要留占位符。
   `leet validate` 会检查题面至少有 600 字符、有章节结构、含「思考题」。

### 第三步:测量与调参(这一步最容易做错)
6. **先实测,再定 gates**。注意 PyTorch 题的 `leet validate` **并不快**
   (实测约 85 秒):它要跑 6 个变体(基线 + 模板 + 4 个劣化解),每个都是一次
   独立的 Python 进程,要 import torch 并在 CPU 上重算一遍参考解。
   所以**别在循环里反复跑它** —— 先把该改的一次性改完。
   另外它只会告诉你**基线**的耗时 ——
   想知道优化解能拿多少分,你要自己写一份正解放进 `solutions/<id>/solution.py`,
   跑 `leet test <id>` 看实际加速比。
   **门槛必须物理可达**:先量出正解的加速比上限,再把 S 设在它的 ~90%,
   A/B 依次下移。设一个永远拿不到的门槛等于没有评级。
7. 迭代到 `leet validate <id>` 全部通过(尤其是「模板必须失败」和四个劣化解)。
8. **把那份优化解留下来当参考解**:存成 `problems/<id>/optimal.cu`,
   并把 `solutions/<id>/` 整个删掉(答案不该出现在学习者的工作区里)。
   参考解是给卡住的学习者看的(`leet solution <题号>`),同时**它也是
   `leet validate` 证明门槛可达的依据** —— 有它在,才谈得上"这个 S 级真的拿得到"。
   写完在题面「评分」一节里说明它拿到什么评级。

### 最后
用一句话汇报:题目 id、考点、指标与门槛、**实测数字**(基线耗时、正解耗时、加速比)。
"""

SPEC_SCHEMA = r"""
## spec.yaml 完整字段说明

```yaml
id: <目录名,必须与目录名完全一致,如 05-conv2d>
title: <中文标题>
difficulty: 1..5
tags: [<英文标签>]
concepts: [<中文考点,写给学习者看>]
statement: problem.md

# ---- 接口契约:决定 LaunchCtx / RefCtx 的字段 ----
buffers:
  - name: <C 标识符>
    dtype: f32 | f64 | i32 | i64 | u32 | u8
    shape: [<参数名或整数字面量,可多项表示多维>]   # 空列表 [] 表示标量
    role: in | out | scratch
    fill: uniform | positive | randint | zero      # 仅 role=in 有效
    #   uniform  → [-1, 1)
    #   positive → (0, 1]   给 log / 开方 / 除法等定义域为正的题用
    #   randint  → 0..99 的整数(浮点类型则转成浮点)
    #   zero     → 全 0
params:
  - {name: <C 标识符>, dtype: i32 | ...}
    # 标量参数。会同时作为字段出现在 LaunchCtx 和 RefCtx 里。
    # 形状表达式里可以引用它们。

entry:
  kernel: <__global__ 函数名>
  launcher: <host 端启动函数名,形如 void launcher(LaunchCtx& ctx)>

cases:
  - {name: <用例名>, params: {<每个参数都给一个取值>}}
    # 参数取值必须是正整数(框架用它算缓冲大小)

verify:
  atol: 1e-5      # 过宽的容差会让错误实现轻松通过,框架会拒绝 atol>1.0 / rtol>0.1
  rtol: 1e-4
  repeat: 3       # 每个用例重复跑几次。>1 用于捕捉竞态/未初始化内存导致的「有时对有时错」

perf:
  enabled: true
  bound: memory | compute
  metric: bandwidth | speedup
    #   bandwidth —— 按「有效带宽占峰值百分比」评级。**访存瓶颈题的朴素 CUDA 实现
    #                往往就是最优算法**(比如向量加法、转置的分块版),这类题用加速比
    #                毫无区分度,必须用带宽。
    #   speedup   —— 按「相对基线的加速比」评级。适合朴素实现远非最优的题
    #                (归约、矩阵乘等)。metric 不写时按 bound 推断:memory→bandwidth。
  repeat: 50
  warmup: 10
  grades: {...}   # 含义随 metric 变:
                  #   bandwidth → 占峰值百分比,如 {B: 50, A: 70, S: 85}
                  #   speedup   → 倍数,如 {B: 3.0, A: 10.0, S: 20.0}
  flush_l2: true  # 默认开启,不要关。4090 有 72MB L2,不清缓存测的是缓存带宽。

sanitize:
  memcheck: true
  racecheck_case: <某个小用例的名字>
    # racecheck 极慢,只能跑最小的用例。这个用例要小到几毫秒内能跑完。
```

## 运行环境(必须按这个来设计,否则题目会不可用)

- GPU: NVIDIA RTX 4090 ×5,sm_89,**理论峰值带宽 1008 GB/s**
- 框架自动在每次计时迭代前**清空 L2**(读一遍大于 L2 的缓冲),所以测到的是真实显存带宽
- 计时用 cudaEvent,**分辨率约 0.5µs**。因此:
  - **总耗时不到 20µs 的用例会被自动跳过评级**(那个量级测的是启动开销和抖动)。
    每道题至少要有一个用例大到可评级 —— 经验值:访存型题目数据量要 ≥ 100MB
    (如 32M 个 float = 128MB,约 130µs);计算型题目要让 kernel 跑到几百 µs。
  - 但用例也不能太大:基线耗时超过 5 秒会让做题体验崩坏。
- 用户**不需要写 cudaMalloc/cudaMemcpy** —— 框架负责分配与搬运,用户只拿到 device 指针。
- 框架在每块缓冲前后各留 4096 个元素的**哨兵区**填特殊值,越界写会被立刻检测到。
- 框架在每次校验执行前把 out 缓冲填成**毒值**(浮点 NaN、整数 0x5EED5EED),
  所以「kernel 没写输出」会被精确识别。
- 每个用例的实际输出会与 `reference.cpp` 的结果逐一比对。
"""

REQUIREMENTS = r"""
## 硬性要求

### reference.cpp —— 标准答案(oracle)
- **CPU 实现**,只写 `void reference(RefCtx& ctx)`,用 `ctx.` 访问所有缓冲与参数
- **必须完全正确**。它是判分的基准,写错了整道题就废了
- 浮点累加请用 `double` 中间变量,保证接近真值
- 不要用任何 CUDA API,也不要用 Scratch 缓冲(RefCtx 里没有 scratch 字段)

### baseline.cu —— 性能基线(加速比的分母)
- **必须正确**(框架会检查:基线与参考解在**所有**用例上必须一致)
- 必须定义与 spec 中同名的 `kernel` 与 `launcher`
- 要是**朴素、直白**的写法 —— 它是学习者要超越的对象
- 但如果这道题是访存瓶颈、而朴素写法已经接近最优(比如向量加法),
  那就要把 spec 的 metric 设成 `bandwidth`,并**在题面里诚实说明「这题没有优化空间,
  它的价值是让你看到带宽天花板」** —— 不要假装有挑战

### template.cu —— 给学习者的骨架
- 只写 `#include "ctx.h"` + kernel 与 launcher 的**空壳**,里面是 TODO 注释
- **必须不完整到无法通过测试** —— 框架会检查「空模板必须被判失败」
- 注释要有教学价值:写清楚这道题的思路、常见陷阱、以及为什么

### problem.md —— 题面
- **中文**,面向刚学 CUDA 的人
- 结构:题目描述 → 为什么这道题重要 → 解法思路(不要直接给完整代码)→
  陷阱提醒 → 评分说明 → 3 个思考题
- 解释要落到**物理原因**上(为什么慢、为什么快),不要只说「这样做性能更好」
- 诚实:如果这题没有优化空间,就直说

### cases
- 至少一个用例大到可评级(见上文 20µs 规则)
- **必须包含边界用例**:非 2 的幂、质数规模的参数,用来考边界处理
- 至少一个小用例供 racecheck 使用(如果 spec 里配了 racecheck_case)
"""

WORKFLOW = r"""
## 你该怎么做

**按这个顺序做,不要跳步。** 写文件很快,而 `leet validate` 要反复编译,
很慢 —— 先把该写的写完,再进入验证循环。这样即使中途被打断,留下的也是一道
完整的题,而不是半成品。

> **不要通读框架源码**(`src/leetstudy/` 下的文件)。出题需要的 spec 格式、
> 检查规则、约束都已经在上面写全了,源码里没有额外信息。花时间读它只会拖慢进度。
> 想找题面写法参考,读一道已有题目的 `problem.md` 就够了。

### 第一步:想清楚(不要动笔)
这道题的**物理瓶颈**是什么(访存 / 计算 / 并行度不足 / 同步开销)?
这决定 metric 与 grades,写错了后面全是白费。

### 第二步:一口气写完 5 个文件
1. `spec.yaml`
2. `reference.cpp`(CPU 参考解,必须完全正确)
3. `baseline.cu`(朴素 CUDA 实现,必须正确且慢)
4. `template.cu`(骨架,必须**不完整到无法通过测试**)
5. **`problem.md`(题面)** —— 一定要在这一步写完,不要留成占位符。
   `leet validate` 会检查题面是否有实质内容(至少 600 字符、有章节结构、
   有「思考题」小节),占位符是过不了的。

### 第三步:验证与调参
6. 跑 `leet validate <id>`。它会做这些检查,任何一项不过都必须修:
   - 文件齐备 / **题面有实质内容** / 编译基线 / 基线通过对拍(基线与参考解必须完全一致)
   - 性能地板(至少一个用例大到可评级)
   - **模板必须失败** —— 空模板要是能过,说明测试形同虚设
   - **劣化解必须失败** —— 框架会注入「全填 0」「全填 1」「只写首元素」等必然错误的
     实现,每一个都必须被判失败。任意一个竟然通过了,说明这道题的测试太弱。
7. **测量真实数字,再据此定 grades**。不要凭感觉写门槛 ——
   `leet validate` 会打印基线的实际耗时。想看优化解能拿多少分,就写一份正确的
   优化解放进 `solutions/<id>/solution.cu`,跑 `leet test <id>`;
   但要**先临时把耗时最久的用例改小**,测出趋势后再改回真实规模,
   否则光是一次 test 就要等很久。
   **门槛必须物理可达**:例如转置题基线只跑到 267 GB/s、峰值是 1008 GB/s,
   那么理论最大加速比就是 3.8x,设 S=8x 是永远拿不到的废门槛。
8. **把那份优化解留下来当参考解**:存成 `problems/<id>/optimal.py`,
   并把 `solutions/<id>/` 整个删掉(答案不该出现在学习者的工作区里)。
   参考解是给卡住的学习者看的(`leet solution <题号>`),同时**它也是
   `leet validate` 证明门槛可达的依据** —— 有它在,才谈得上"这个 S 级真的拿得到"。
   写完在题面「评分」一节里说明它拿到什么评级。

### 最后
用一句话汇报:题目 id、考点、指标与门槛、实测数字。
"""

SPEC_SCHEMA_CPP = r"""
## spec.yaml 完整字段说明(C++ 优化题)

```yaml
id: cpp02-<名字>          # 必须与目录名一致。**前缀必须是 cpp**,不要用纯数字 ——
                          # 各科目有各自的编号空间,数字部分重复会让 `leet test <n>`
                          # 有歧义并直接失败(bank.resolve 对纯数字走前缀匹配)
title: <中文标题>
subject: cpp              # 必须写,否则会被当成 cuda 科目
difficulty: 1..5
tags: [<英文标签,如 cache loop-order>]
concepts: [<中文考点,写给学习者看>]
statement: problem.md

# ---- 接口契约:决定 Ctx 的字段 ----
buffers:
  - name: <C 标识符>
    dtype: f32 | f64 | i32 | i64 | u32 | u8
    shape: [<参数名或整数字面量>]      # 空列表 [] 表示标量
    role: in | out | scratch
    fill: uniform | positive | randint | zero    # 仅 role=in 有效
params:
  - {name: <C 标识符>, dtype: i32 | ...}
    # 标量参数。形状表达式里可以引用它们。浮点参数可以给小数(如 alpha: 0.5)

entry:
  function: <学习者要改快的那个函数名>
    # 签名固定为 void f(Ctx& ctx) —— 框架按这个签名调用,不能改

cases:
  - {name: <用例名>, params: {<每个参数都给一个取值>}}
    # 取值必须是正数;整数类型的参数必须是整数(会被用来算缓冲大小)

verify:
  atol: 1e-5      # 过宽会让错误实现轻松通过,框架会拒绝 atol>1.0 / rtol>0.1
  rtol: 1e-4
  repeat: 2       # 单线程 C++ 是确定性的,不必像 CUDA 那样重复很多次

perf:
  enabled: true           # required_grade 要求它为 true
  bound: memory | compute
  metric: bandwidth | speedup
    #   优化题绝大多数用 speedup(相对基线快了多少倍)
  repeat: 12              # ★ 不要用默认的 50!CPU 题动辄几十毫秒,
                          #   50 次 + 10 次预热 = 几分钟白等。按基线实测耗时定:
                          #   基线 80ms → repeat 12、warmup 3 就够取中位数了
  warmup: 3
  grades: {B: 3.0, A: 9.0, S: 13.0}   # 必须从实测倒推,见「工作流」
  required_grade: B       # ★★ 优化题**必须**设,否则这道题是无效的
    #   基线本身就是正确代码,不设它的话学习者把原代码原样交回来就算通过。
    #   设了之后「正确性 + 评级达标」才算过,「模板必须失败」那条检查也才重新有效
  flush_l2: true          # 默认开启,不要关

sanitize:
  memcheck: false         # cpp 科目暂未接入 ASan;写了 true 也会被跳过
```

## 运行环境(必须按这个来设计)

- **编译选项由框架钉死**:`g++ -O3 -march=native -std=c++17`,**不带 `-ffast-math`**。
  这个事实决定了很多题的可行性 —— 见下面「哪些缺陷真的有优化空间」
- 计时用 `steady_clock`,调用是同步的,每次计时迭代前**清缓存**(读一遍 128MB)
- **噪声地板 6–7%**(跨进程漂移;同进程内只差 1%,但机器忙时会明显变差)。
  所以门槛之间至少要留 10% 的间隔
- 单核 DRAM 带宽只有 ~30 GB/s。访存瓶颈的题在这个上限附近就会封顶
- 学习者的函数被**反复调用**(稳定性重复 + 计时循环),所以不能依赖
  "第一次调用时初始化"这类状态
"""

REQUIREMENTS_CPP = r"""
## 硬性要求

### reference.cpp —— 标准答案(oracle)
- 只定义 `void reference(Ctx& ctx)`,用 `ctx.` 访问所有缓冲与参数
- **必须完全正确**。它是判分的基准,写错了整道题就废了
- 浮点累加请用 `double` 中间变量,保证接近真值
- 不要写 `entry.function` 那个名字 —— 那是学习者的接口

### baseline.cpp —— 性能基线(加速比的分母)
- 必须定义 `void <entry.function>(Ctx& ctx)`
- **必须正确**(框架会检查:基线与参考解在**所有**用例上必须一致)
- **它就是那段"正确但有性能缺陷"的代码** —— 不是"故意写烂",而是
  "照着定义直译、完全没考虑访存"的自然写法。学习者学到的是真实的经验判断

### template.cpp —— 学习者拿到的起点
- **必须与 baseline.cpp 是同一段代码**(可以有更详细的教学注释和 TODO)。
  这一条是硬的:学习者的起点加速比必须正好是 1.00x。
  如果你把模板写得和基线不一样,起点就不是 1.0x,整道题的语义就乱了
- 注释要有教学价值:指出该往哪个方向想,但**不要写出答案**
- 框架会检查「模板必须失败」—— 对优化题来说就是「模板达不到 required_grade」

### optimal.cpp —— 参考解(★ 出题的关键)
- 一份**能达到目标评级**的实现。它是"门槛物理可达"的唯一依据
- 带注释讲清为什么这样写快、试过哪些死路
- 卡住的学习者用 `leet solution <题号>` 看它

### problem.md —— 题面
- **中文**,面向会写 C++ 但没做过性能优化的人
- 结构:题目描述 → 为什么这道题重要 → **慢在哪(要有实测数字)** →
  解法思路(不要直接给完整代码)→ 陷阱 → 评分说明 → 3~4 个思考题
- 解释要落到**物理原因**上(cache line 多大、步长是多少、为什么预取器失效),
  不要只说「这样写更快」
- **诚实**:如果某个"优化"实测下来没用甚至更慢,就直说 —— 那往往是最有价值
  的一段。不要写你没验证过的断言
- 「评分」一节要写明 `required_grade`:**这道题必须达到 X 级才算通过**,
  以及为什么(基线本身正确,只看正确性的话交回原代码就算过)

### cases
- 至少一个用例大到可评级(基线耗时 ≥ 1ms 比较稳妥)
- **必须包含边界用例**:非 2 的幂、质数规模
- 但也要注意**别把用例开太大** —— 基线几十毫秒 × (repeat+warmup) 次
  × 7 个变体,会让 `leet validate` 变得很慢
"""

WORKFLOW_CPP = r"""
## 你该怎么做

**这个科目与另外两个最大的不同:能不能出成题,必须先测量才知道。**

> **不要通读框架源码**(`src/leetstudy/` 下的文件)。出题需要的 spec 格式、
> 检查规则、约束都已经在上面写全了,源码里没有额外信息。花时间读它只会拖慢进度。
> 想找题面和 spec 的写法参考,读 `problems/cpp01-matmul-loop-order/` 那一份就够了
> —— 它是完整的范例,提示词末尾也整份附上了。

### 第一步:先想清楚"这个缺陷到底有多少优化空间"(不要动笔)

这是本科目唯一的硬性要求。实测下来,**「看起来像陷阱」和「真是陷阱」大约各占
一半** —— 编译器在 `-O3 -march=native` 下已经自己处理掉了不少经典陷阱。
下面这张表是实测结果,直接决定你的选题空间:

| 缺陷类型 | 实测优化空间 | 说明 |
|---|---|---|
| 矩阵乘循环次序 `ijk`→`ikj` | **12–26x** | ✅ 首选题材 |
| 矩阵乘分块(cache blocking) | **33x** | ✅ 编译器不会自动做 |
| 归约单累加器 → 4 路独立累加 | **3.4x** | ✅ 无 fast-math 时编译器不许重排浮点 |
| AoS → SoA(只取结构体部分字段) | **2.2x** | ✅ 编译器改不了你的数据布局 |
| 虚函数 → 具体类型 / 去虚化 | **1.72x** | ⚠️ 幅度小,四档门槛会挤在一起 |
| `std::function` 递归 → 泛型 lambda | **1.47x** | ⚠️ 同上 |
| 补 `__restrict` | 1.1x | ❌ **编译器已经自己修了**(插运行时别名检查) |
| `range-for` 的 `auto` 值拷贝 | **1.00x** | ❌ **编译器已经自己修了**(拷贝是死代码) |

**因此:**
- 优化空间 **> 2x** 才值得做成评级题(B/A/S 三档要拉开)
- 1.2–1.7x 的那类不要做成评级题,除非你把它设计成**判断题**
  (让学习者先猜瓶颈在哪,再用测量验证)
- ≈1.0x 的直接放弃,换一个题材

> 这张表是实测来的,不是推测。**不要想当然地认为某个东西慢** ——
> 上一轮就有两个"经典陷阱"实测下来是 1.0x,编译器早就修好了。

### 第二步:一口气写完 6 个文件
1. `spec.yaml`(`required_grade` 先随便填一个偏高的目标,比如 `{B: 2.0, A: 5.0, S: 10.0}`,
   等测出来再改)
2. `reference.cpp`(CPU 参考解,double 累加)
3. `baseline.cpp`(有缺陷但正确的实现)
4. `template.cpp`(**与 baseline 同一段代码** + 教学注释)
5. `optimal.cpp`(改好的版本)
6. **`problem.md`(题面)** —— 一定要在这一步写完,不要留占位符。
   `leet validate` 会检查题面至少 600 字符、有章节结构、含「思考题」

### 第三步:测量,然后定门槛

```bash
leet validate cpp02        # 首次约 20~30 秒(要编译 7 个变体),之后缓存命中约 10 秒
```

它会打印基线耗时和**参考解实际拿到的评级与倍速**,例如:

```
✓  参考解能证明门槛可达    参考解达标:最好用例 square 16.19x,评级 S
```

拿到这个数字之后:
- **门槛从实测倒推**:S ≈ 实测值的 75~80%,A ≈ 55~60%,B ≈ 20%
  (不要卡到 90% —— CPU 侧噪声比 GPU 大,见下)
- ★ **一定要用"最差值",不是典型值。** 同一份代码多测几次,并且**在机器忙的时候
  也测一次**,拿最差的那个数来定。踩过这个坑:某题安静时稳定 16.2~16.9x,
  按 90% 设了 S=14.5,结果一次带负载的运行里参考解只拿到 14.05x —— 于是 S
  变成了「你挑了个机器空闲的时候测」。门槛要设在**即便机器忙也达得到**的位置
- 把实测数字写进 spec.yaml 的注释里(照 `problems/cpp01-matmul-loop-order/spec.yaml`
  的样子写)

### 第四步:如果空间不够,换题材 —— 不要降低门槛

如果 `leet validate` 报:

```
✗  参考解能证明门槛可达   参考解只拿到 C 级(1.12x)——
                          说明门槛设高了,存在不了这样的实现
```

**这说明这个缺陷没有足够的优化空间,不是你门槛设错了。** 正确做法是**换一个
题材**(回到第一步那张表挑一个 >2x 的),而不是把 grades 改成 `{B: 1.05, ...}`
去迁就它 —— 那样出出来的是一道没有区分度的废题。

### 第五步:确认区分度检查全绿

`leet validate` 会注入 4 个必然错误的实现(空实现 / 全填 0 / 全填 1 / 只写首元素),
每一个都必须被判失败。加上:
- **模板必须失败** —— 对优化题来说就是「模板达不到 required_grade」
  (模板 == 基线,加速比 1.00x,自然达不到 B)
- **参考解能证明门槛可达** —— 参考解必须拿到 A 或 S

### 最后
用一句话汇报:题目 id、考点、指标与门槛、**实测数字**(基线耗时、参考解耗时、加速比)。
"""


REVIEW_PROMPT = r"""你是一位 CUDA 教学助手。下面是一个学习者对某道刷题题的解答,以及框架给出的
判题数据。请给出**针对这份代码**的讲评。

要求:
1. 先判断他卡在哪一层的认知上(索引?访存模式?同步?并行度?数值?)
2. 指出**具体行**的问题,不要泛泛而谈
3. 解释物理原因,不要只说「建议这样做」
4. 给出下一步**具体可执行**的优化方向,并说明预期能提升多少、为什么
5. 如果他已经做得很好,就说清楚好在哪、以及还有什么边界可以推
6. 用中文,直接给结论,不要客套

不要重写他的代码给他抄 —— 指出方向和原因,让他自己改。
"""


#: 各科目的默认范例题目(提示词里会把它整份读进来当 few-shot)
DEFAULT_EXAMPLE = {
    "cuda": "01-vector-add",
    "pytorch": "py01-softmax-dim0",
    "cpp": "cpp01-matmul-loop-order",
}

#: 各科目的范例文件。范例里必须带上参考解(`optimal.*`)—— 否则出题者只能从
#: 文字描述猜它该长什么样,而"照着范例做"比"照着说明做"可靠得多。
EXAMPLE_FILES = {
    "cuda": ("spec.yaml", "reference.cpp", "baseline.cu", "template.cu", "optimal.cu"),
    "pytorch": ("spec.yaml", "reference.py", "baseline.py", "template.py", "optimal.py"),
    "cpp": ("spec.yaml", "reference.cpp", "baseline.cpp", "template.cpp", "optimal.cpp"),
}

#: 已经写好出题提示词的科目。**注册科目 ≠ 能出题** ——
#: `subjects.available()` 里有什么,和这里有什么,是两件事。
#: 加了科目却忘了加提示词的话,出题会拿到错误的那套说明并产出格式错乱的题目,
#: 所以 `_blocks()` 对未知科目直接抛错,而不是回落到 CUDA。
AUTHORABLE = ("cuda", "pytorch", "cpp")


def supports(subject: str) -> bool:
    """这个科目能自动出题吗?"""
    return subject in AUTHORABLE


#: 从需求原文里认出科目的关键词。**故意只收明确无歧义的字面量** ——
#: 这个函数的用途是「拦下明显搞错的情况」,不是「智能判断科目」。
#: 误报的代价是用户白跑一次(加个 flag),所以宁可漏,不可错。
_SUBJECT_SIGNALS = (
    # 先看 pytorch:它点名了框架本身,是最强的信号。
    # 「pytorch 自定义 CUDA 算子」这种同时含 cuda 字样的需求,科目仍然是 pytorch,
    # 所以它必须排在 cuda 前面。
    ("pytorch", ("pytorch", "torch")),
    # cuda 的典型说法。「用 C++ 写 CUDA kernel」同时含 c++ 与 kernel,该判 cuda。
    ("cuda", ("cuda", "kernel", "核函数", "shared memory", "共享内存",
              "__syncthreads", "线程块", "网格", "blockdim", "griddim")),
    # cpp 优化题的说法
    ("cpp", ("c++", "cpp", "编译器优化", "cache line", "循环次序", "访存模式")),
)


def suggest_subject(requirement: str) -> Optional[str]:
    """从需求原文猜科目。认不出来返回 None。

    只按 `_SUBJECT_SIGNALS` 的顺序取**第一个**命中的科目 —— 这是有意的优先级,
    不是打分:命中了就返回,不比较命中次数。
    """
    text = (requirement or "").lower()
    for subject, words in _SUBJECT_SIGNALS:
        if any(w in text for w in words):
            return subject
    return None


def _blocks(subject: str):
    """按科目选对应的三块提示词。

    **不做静默回落。** 早先这里是「是 pytorch 就返回 pytorch 那套,否则返回
    CUDA 那套」—— 于是 `leet new --subject cpp` 会拿到 CUDA 的说明:
    模型被告知「学习者写 kernel 与启动配置」,然后产出一堆 .cu 文件。
    这比直接报错更糟 —— 它看起来是支持的。
    """
    if subject == "pytorch":
        return SPEC_SCHEMA_PYTORCH, REQUIREMENTS_PYTORCH, WORKFLOW_PYTORCH
    if subject == "cpp":
        return SPEC_SCHEMA_CPP, REQUIREMENTS_CPP, WORKFLOW_CPP
    if subject == "cuda":
        return SPEC_SCHEMA, REQUIREMENTS, WORKFLOW
    raise ValueError(
        f"{subject!r} 科目还没有出题提示词(可用:{' / '.join(AUTHORABLE)})。\n"
        f"科目本身能用(可以手写题目、leet test / validate 都正常),"
        f"只是还不能让本地 claude 自动出题。\n"
        f"要支持的话,照 prompts.py 里 PyTorch 那套加三块:"
        f"SPEC_SCHEMA_{subject.upper()} / REQUIREMENTS_{subject.upper()} / "
        f"WORKFLOW_{subject.upper()},并登记进 AUTHORABLE。"
    )


def build_author_prompt(requirement: str, root: Path, subject: str = "cuda",
                        example_id: str = "", existing_ids: List[str] = None,
                        only_one: bool = False) -> str:
    """组装出题用的完整提示词。

    直接把题库里一道已通过验证的题完整读进来当范例 —— 这样提示词与真实的
    spec 格式永远同步,不会因为文档漂移而生成出格式过时的题目。

    范例文件按科目取:PyTorch 题没有 template.cu 之类。

    only_one=True 时明确要求「本次只出一道具」。用于 `leet new --count N` ——
    那里是逐道开独立会话,不能让每个会话都去尝试出完全部 N 道。
    """
    schema, requirements, workflow = _blocks(subject)
    example_id = example_id or DEFAULT_EXAMPLE.get(subject, "01-vector-add")
    example_dir = root / "problems" / example_id
    example_files = EXAMPLE_FILES.get(subject, EXAMPLE_FILES["cuda"])
    example_text = ""
    for fname in example_files:
        f = example_dir / fname
        if f.is_file():
            example_text += f"\n### {fname}\n```\n{f.read_text(encoding='utf-8')}\n```\n"

    existing = ""
    if existing_ids:
        existing = (
            "\n## 已有题目(不要重复出这些;新题要与之互补)\n"
            + "\n".join(f"  - {i}" for i in existing_ids)
            + "\n"
        )

    subject_line = {
        "pytorch": (
            "**本批题目属于 `pytorch` 科目** —— 学习者用 Python 写 `forward(ctx)`,"
            "不需要写 CUDA。"
        ),
        "cpp": (
            "**本批题目属于 `cpp` 科目,而且是「优化题」** —— 基线和模板本身就是"
            "**完全正确**的代码,学习者要做的是把它**改快**。所以 `perf.required_grade` "
            "必须设,而且门槛必须从实测倒推。这是本科目与另外两个最根本的差别。"
        ),
        "cuda": "**本批题目属于 `cuda` 科目** —— 学习者写 kernel 与启动配置。",
    }.get(subject, "**本批题目属于 `cuda` 科目** —— 学习者写 kernel 与启动配置。")

    # `leet new --count N` 是逐道开独立会话的。学习者的需求原文里往往写着
    # 「出三道…」,如果不明确约束,每个会话都会去尝试出完全部 N 道 ——
    # 既浪费,又会撞超时。
    scope_line = (
        "\n**本次只出一道具。** 上面的需求里可能提到多道题,那是 `leet new --count`\n"
        "按道拆开、一道一个会话执行的 —— 你这次**只负责其中一道**,不要试图把\n"
        "全部题目一次做完。\n"
        if only_one else ""
    )

    return f"""你是一位 CUDA / PyTorch / C++ 性能教学专家,正在为一个「LeetCode 式的刷题框架」出题。

# 学习者的需求

{requirement}
{existing}
# 你要产出什么

{subject_line}

在 `problems/<新题id>/` 目录下产出 5 个文件:spec.yaml / problem.md / template /
reference / baseline,外加**参考解 `optimal`**(详见下面的工作流)。
扩展名随科目:PyTorch 是 `.py`,CUDA 是 `.cu` / `.cpp`,C++ 优化题统一是 `.cpp`。
id 用小写短横线风格;**前缀随科目**(CUDA 用纯数字如 `08-foo`,PyTorch 用
`py05-foo`,C++ 用 `cpp02-foo`)—— 各科目有各自的编号空间,不要共用数字序列。
{scope_line}

{schema}
{requirements}
{workflow}

# 完整范例(已通过全部校验,请严格对照它的格式与注释风格)

```{example_text}```

# 开始

{_closing(subject)}

现在开始。完成后运行 `leet validate <id>` 确认全部通过。
"""


#: 收尾处反复强调的三个坑。**按科目分开写** —— 三个科目的失败模式不一样:
#: CUDA 怕模板放水,优化题怕"缺陷其实没有优化空间"却硬把门槛调低,
#: PyTorch 怕标注错误。
_CLOSING = {
    "cuda": """记住三个最容易出错的地方:
1. **用例太小** → 性能评分完全落空(必须有一个用例大到可评级)
2. **门槛物理不可达** → 评级形同虚设(先测量再定门槛)
3. **模板能通过测试** → 这道题没有意义(空模板必须失败)""",
    "pytorch": """记住三个最容易出错的地方:
1. **用例太小** → 性能评分完全落空(必须有一个用例大到可评级)
2. **门槛物理不可达** → 评级形同虚设(先测量再定门槛)
3. **模板能通过测试** → 这道题没有意义(空模板必须失败)""",
    "cpp": """记住三个最容易出错的地方:
1. **缺陷其实没有优化空间** → 实测空间不到 2x 就**换题材**,不要靠调低门槛迁就它。
   实测过一半的"经典陷阱"在 `-O3` 下是假的(编译器早就修好了)
2. **模板和基线不是同一段代码** → 学习者的起点就不是 1.00x,语义乱掉
3. **没设 `perf.required_grade`** → 基线本身就是正确代码,不设的话把原代码
   原样交回来就算通过,这道题是废的""",
}


def _closing(subject: str) -> str:
    return _CLOSING.get(subject, _CLOSING["cuda"])


def build_review_prompt(problem_id: str, title: str, statement: str, solution_src: str,
                        spec_yaml: str, baseline_src: str, verdict_summary: str) -> str:
    """组装讲评用的提示词。"""
    return f"""{REVIEW_PROMPT}

# 题目

{problem_id} {title}

## 题面

{statement}

## 题目规格(spec.yaml)

```yaml
{spec_yaml}
```

## 性能基线(学习者要超越的实现)

```cuda
{baseline_src}
```

# 学习者的解答

```cuda
{solution_src}
```

# 框架给出的判题数据

{verdict_summary}

# 请开始讲评
"""

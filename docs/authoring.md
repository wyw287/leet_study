# 出题指南

给一道题需要 5 个文件。本文说明它们的契约、字段含义、以及**必须避开的坑**。

两条路径:

- **`leet new "<需求>" [--subject pytorch]`** —— 本地 claude 自动产出 + 自验证 + 修复回路。
  见本文末尾「自动出题」。
- **手写** —— 本文主体。改现有题目也适用。

---

## 一、文件契约

```
problems/<id>/
  spec.yaml        机器可读定义(唯一驱动代码生成的文件)
  problem.md       中文题面:讲解 + 提示 + 陷阱 + 思考题
  template.cu      给学习者的骨架 —— 必须**不完整到无法通过测试**
  reference.cpp    CPU 参考解(oracle)—— 必须**完全正确**
  baseline.cu      朴素实现(加速比的分母)—— 必须**正确且慢**
```

PyTorch 科目把后三个换成 `.py`(`template.py` / `reference.py` / `baseline.py`),
并且三个文件里的函数名不同:

| 文件 | 函数名 | 说明 |
|---|---|---|
| `reference.py` | **固定** `def reference(ctx):` | oracle,与 CUDA 侧 `void reference(RefCtx&)` 对齐 |
| `baseline.py` / `template.py` | `spec.entry.function` 指定的名字 | 学习者的接口 |

**这三条是硬规则,`leet validate` 会检查:**

1. **参考解必须完全正确** —— 它是判分基准,错了整道题就废了
2. **模板必须失败** —— 空模板要是能过,说明测试形同虚设
3. **基线必须与参考解在全部用例上一致** —— 这是本题唯一的正确性验证手段

---

## 二、spec.yaml 字段参考

```yaml
id: 06-example                # 必须与目录名一致;小写短横线;数字前缀排前面
title: 中文标题
subject: cuda                 # cuda(默认) | pytorch
difficulty: 1..5
tags: [英文标签]               # 用于 leet list --tag
concepts: [中文考点]           # 写给学习者看
statement: problem.md

# ---- 接口契约(驱动 ctx.h / forward(ctx) 的字段) ----
buffers:
  - name: input               # C / Python 标识符
    dtype: f32                # f32 f64 i32 i64 u32 u8
    shape: [m, n]             # 每项是参数名或整数字面量;[] 表示标量
    role: in | out | scratch
    fill: uniform             # 仅 role=in:uniform[-1,1) positive(0,1] randint[0,100) zero
params:
  - {name: m, dtype: i32}     # 标量参数;形状表达式里可引用
    # 浮点类型的参数可以给小数取值(alpha: 0.5、eps: 1e-5)

entry:
  kernel: example             # CUDA:__global__ 函数名
  launcher: example_launch    # CUDA:host 端启动函数名,签名固定 void f(LaunchCtx&)
  # function: forward         # PyTorch:学习者要实现的函数名

cases:
  - {name: exact, params: {m: 4096, n: 4096}}
  # 每个参数都要给值;必须是正数;整数类型的参数必须是整数

verify:
  atol: 1e-5                  # 框架会拒绝 atol>1.0 / rtol>0.1(过宽等于没检查)
  rtol: 1e-4
  repeat: 3                   # 稳定性重复次数;抓竞态/未初始化内存

perf:
  enabled: true
  bound: memory | compute
  metric: bandwidth | speedup # 不写时按 bound 推断(memory→bandwidth)
  repeat: 50
  warmup: 10
  grades: {B: 50, A: 70, S: 85}   # 含义随 metric:bandwidth 是占峰值%,speedup 是倍数
  flush_l2: true              # 别关

sanitize:
  memcheck: true
  racecheck_case: tiny        # racecheck 极慢,只能跑最小用例
```

### role 的语义

| role | 填充 | 参与比对 | 额外 |
|---|---|---|---|
| `in` | 按 `fill` 随机填充(种子由用例名派生,可复现) | 否 | — |
| `out` | 校验前填毒值 | **是** | 有哨兵区 |
| `scratch` | 校验前填毒值 | 否 | 给需要工作区的题用(如归约) |

### 关于 dtype

- 浮点类型用 `atol/rtol` 比对;整数类型要求**精确相等**
- 两边都是 NaN 视为相等;只有一边是 NaN 算错
- PyTorch 科目的 dtype 直接映射到 torch dtype

---

## 三、用例设计:三个必须

### 1. 至少一个用例大到可评级

**低于 20µs 的用例会被自动跳过评级**(仍参与正确性判定)。经验值:
访存型题目数据量 ≥ 100MB(如 32M 个 f32 = 128MB,4090 上约 130µs)。

太小的话性能评分会完全落空 —— `leet validate` 会拦下这种题。

但也不能太大:基线耗时超过 5 秒会让做题体验崩坏。

### 2. 必须有边界用例

非 2 的幂、质数规模、不能被 tile 整除的形状。这是**最常抓出 bug** 的一类用例。

一个真实例子:向量加法题里,`exact`(n=2²⁴,能被 256 整除)通过,而 `ragged`
(n=16777259,质数)抓出了漏掉的 `if (i < n)` —— 哨兵区报出「往后越界 213 个元素」。

### 3. 最小用例留给 racecheck

`compute-sanitizer --tool racecheck` 极慢,只跑最小的那个用例。
它要小到几毫秒内跑完(如 25 万个元素)。

---

## 四、定门槛:必须从实测倒推

**这是出题最容易做错的一步。** 反例:转置题基线只跑到 267 GB/s、峰值 1008 GB/s,
理论最大加速比就是 **3.8x** —— 设 `S: 8.0` 是永远拿不到的废门槛。

### 步骤

```bash
# 1. 写完全部文件后,先验证结构与基线
leet validate 6

# 2. 看基线实际耗时(validate 会打印各用例的基线 ms)

# 3. 写一份**正确的优化解**放到 solutions/06-example/solution.cu
#    —— 这一步是关键,不能跳过

# 4. 量出正解能拿到多少
leet test 6

# 5. 按正解的数字倒推门槛:
#       S ≈ 正解的 90%
#       A ≈ 正解的 65%
#       B ≈ 正解的一半(但明显高于 1.0)
#    把实测数字写进 spec.yaml 的注释里(见现有题目的做法)

# 6. ★ 删掉 solutions/06-example/ —— 别把答案留在学习者工作区
```

**为什么一定要实测**:同一个直觉在访存题和计算题上会得出完全不同的结论。
比如「先转置再归约」在 PyTorch 里既能拿到 3.3x(带拷贝),也能拿到 13x(不拷贝),
差别只在于接口选择 —— 不跑一遍根本看不出来。

---

## 五、自检工作流

```bash
leet validate 6          # 单题。实测:CUDA 题约 40 秒,PyTorch 题约 85 秒
leet validate --all      # 全库;改框架后必须跑这个
```

> **为什么 PyTorch 题反而更慢**(尽管不需要编译):判题要跑 6 个变体
> (基线 + 模板 + 4 个劣化解),每个都是一次独立的 Python 进程 —— 要 import torch
> (1.6 秒)、并在 CPU 上以 float64 重算一遍参考解。典型规模下这一遍参考解就要
> 几百毫秒到几秒。所以**别在循环里反复跑 validate**,先把该改的一次性改完。

`leet validate` 的检查项:

| 检查 | 在干什么 |
|---|---|
| 文件齐备 | 5 个文件都在 |
| 题面有实质内容 | ≥600 字符、有章节结构、含「思考题」小节 |
| 准备基线 | 能编译(或 Python 语法预检通过) |
| **基线通过对拍** | 基线与参考解在**全部**用例上一致 |
| 性能地板 | 至少一个用例大到可评级;没有用例慢到崩坏 |
| **模板必须失败** | 空模板被判失败 |
| **劣化解必须失败:×4** | 空实现 / 全填 0 / 全填 1 / 只写首元素,每一个都被判失败 |

最后两组是**区分度检查** —— 它是让自动出题可信的关键闸门。
任意一个劣化解竟然通过了,说明这道题的测试太弱,必须打回。

---

## 六、常见错误清单

| 症状 | 原因 | 修法 |
|---|---|---|
| `性能地板` 失败:没有用例可评级 | 用例太小 | 放大数据量(访存型 ≥100MB) |
| `模板必须失败` 失败 | 模板里留了能跑通的最小实现 | 只留空壳 + TODO |
| `劣化解必须失败:zeros` 失败 | 期望值恰好全为 0 | 换 `fill` 模式(如 `positive`)或换输入分布 |
| `基线通过对拍` 失败 | 参考解或基线有 bug;或容差过严 | 先用 CPU 手算几个元素核对 |
| `题面有实质内容` 失败 | 留了占位符 | 补写完整讲解(这是最容易偷懒的一项) |
| racecheck 超时 | 用例太大 | 把 `racecheck_case` 指向最小用例 |
| 编译报错在框架文件里 | 改了 kernel/launcher 的函数签名 | 签名必须与 `spec.entry` 完全一致 |

### 关于容差

- **过严**:归约、扫描这类题的求和顺序与 CPU 参考解天然不同,`1e-7` 会误杀正确实现
- **过宽**:`atol=1.0` 会让错误实现轻松通过,框架会直接拒绝
- 数值敏感的操作(`log`/开方/除法)给输入用 `fill: positive`,避免定义域外输入

---

## 七、自动出题

```bash
leet new "出两道关于 bank conflict 的题,难度递进"
leet new --subject pytorch "出一道 LayerNorm 的题"
```

流程:

1. 组装提示词:spec 格式说明 + 硬性要求 + 工作流 + **一道已通过的完整范例**
   (范例题直接从 `problems/` 读,所以提示词与真实格式永远同步)
2. 本地 claude 以 headless 方式产出文件,并自己跑 `leet validate` 迭代
3. **框架再独立验证一遍**(不采信它的自我声明)
4. 不过关就把失败报告回喂给它修,最多 `--repair-rounds` 轮

出题者的工具权限走白名单(`Read/Write/Edit/Glob/Grep` + `leet`/`nvcc`/
`compute-sanitizer`),不使用 `--dangerously-skip-permissions`。

### 提示词里反复强调的三件事

因为这是自动出题最容易犯的三个错:

1. **用例太小** → 性能评分完全落空
2. **门槛物理不可达** → 评级形同虚设(必须先测量)
3. **模板能通过测试** → 这道题没有意义

### 出完之后

- `leet new` 留下的 `solutions/<id>/` 是剧透,应当删掉(提示词里要求了,但确认一下)
- 通读一遍 `problem.md` —— 自动生成的讲解可能有一本正经的错误
  (真实案例:生成的题面里断言 `.contiguous()` 是必需的,实测后发现并非如此)
- 跑一次 `leet validate --all` 确认没影响其它题目

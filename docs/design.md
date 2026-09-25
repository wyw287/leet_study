# 设计决策与理由

这份文档记录**为什么这样设计**。多数结论来自实测,而不是推演 —— 凡是带数字的,
都是在本机(5× RTX 4090,驱动 535,CUDA 12.2)上真跑出来的。

---

## 一、核心机制:spec 驱动的代码生成

### 问题

一道 CUDA 练习题涉及四份代码:学习者的 kernel、参考解、性能基线、以及驱动它们的
harness。如果每道题都手写一份 harness,会有两个后果:题目创作成本极高,而且
**学习者解与基线的计时/校验逻辑会产生细微差异,加速比就不可比了**。

### 做法

`spec.yaml` 声明接口契约(`buffers` / `params`),由它生成 `ctx.h`:

```cpp
struct LaunchCtx { const float* a; const float* b; float* c; int n; };
struct RefCtx    { const float* a; const float* b; float* c; int n; };
```

字段名来自 spec。于是 harness、参考解、学习者模板**共享同一份接口定义**,不会漂移。

然后把**同一个 harness 编译两次** —— 一次链学习者的实现,一次链基线。
计时与校验逻辑逐字相同,只有 `impl.cu` 不同。加速比因此可信。

`LaunchCtx` 给学习者(device 指针 + stream),`RefCtx` 给参考解(host 指针)。
参考解跑在 **CPU** 上,这是刻意的:oracle 必须自身不可能有 GPU 侧的错误
(竞态、越界、同步遗漏)。CPU 版慢一点无所谓。

### 为什么后来加了 `Artifact`

最初的科目接口是 `run_case(self, exe: Path, ...)`。这隐含一个假设 ——
**每个科目都产出原生可执行文件**。CUDA 是这样,PyTorch 不是(它的"产物"是一个
可以被 import 的 Python 模块,没有可执行文件)。

`Artifact` 把这件事抽象掉:

```python
@dataclass
class Artifact:
    kind: str                    # "exe" | "module"
    path: Path
    extra: Dict[str, Any]        # 科目私有信息(如题目目录)
```

改完之后 `judge` / `report` / `cli` / `bank` / `validate` **一行都不用动** ——
它们只消费 `Artifact` 和 `CaseResult`,不关心后面是可执行文件还是 Python 模块。

这也是为什么 PyTorch 科目**没有 codegen**:那 587 行存在的唯一理由是 C++ 无法自省
(要按 spec 拼出结构体定义和 main 函数)。Python 可以自省,所以它的 runner 就是一个
普通的、可 import 的模块。

---

## 二、诊断:三个机制,对应三类初学者困境

### 1. 哨兵区 —— 抓越界写

每块缓冲前后各留 4096 个元素填一个特征值,交给 kernel 的是**内区**指针:

```
[哨兵 4096][        内区 n        ][哨兵 4096]
              ↑ 交给 kernel 的指针
```

跑完检查两端哨兵有没有被改。好处不只是"能发现越界",而是能**区分方向**:

> ⚠️ 越界写:缓冲 `c` **后面**的哨兵区被改写了 213 个元素。
> 最常见的原因就是漏了 `if (i < n)` 这类边界判断。

`compute-sanitizer` 的 memcheck 也能抓越界(而且能抓越界**读**),但它慢、要显式开启。
哨兵区是常驻的第一道防线,成本几乎为零。

### 2. 毒值 —— 抓"没写输出"

`out` / `scratch` 缓冲在每次校验执行前填毒值:浮点填 `NaN`,整数填 `0x5EED5EED`。

这样"kernel 根本没写入"会表现为**清一色的毒值**,而不是碰巧等于某个合法值。
诊断层据此能说出:

> 输出 `c` 全部是 NaN,正好是框架填入的毒值 —— 说明你的 kernel 没有写入这块缓冲。

这是初学者最常见的失败模式之一,而它用"结果不对"四个字是说不清的。

### 3. launch 后置检查 —— 抓启动配置错误

`launch()` 返回后立刻 `cudaGetLastError()`,再 `cudaDeviceSynchronize()`:

```cpp
launch(lctx);
cudaError_t e_launch = cudaGetLastError();      // 启动请求被拒?
if (e_launch != cudaSuccess) LEET_FAIL("launch", e_launch);
cudaError_t e_sync = cudaDeviceSynchronize();   // 异步错误(kernel 跑挂了)?
if (e_sync != cudaSuccess) LEET_FAIL("执行中(异步错误)", e_sync);
```

覆盖 `invalid configuration argument`(网格/块维度算成 0)、
`too many resources requested`、`illegal memory access` 等,并翻译成中文提示。

---

## 三、计时:两个必须做对的事

### 1. 每次迭代前清空 L2 —— 而且要用「读」

**这是本项目最重要的一次实测修正。**

4090 的 L2 有 72MB,而一道题的数据常常只有几 MB。不清缓存的话,题目整个驻留在
L2 里,**测出来的是缓存带宽而不是显存带宽**,数字能虚高好几倍。

但"清缓存"本身也有讲究。第一版用 `cudaMemset` 写一个大于 L2 的缓冲,结果:

| 用例 | 元素数 | 实测 |
|---|---|---|
| `exact` | 16,777,216 | 512 GB/s |
| `ragged` | 16,777,259 | **689 GB/s** |

**元素更多的反而更快** —— 不单调,说明测量有系统性错误。

原因:写清缓存会在 L2 里留下 **72MB 脏行**。被计时的 kernel 一开始就得先把这些
脏行写回 DRAM 才能腾地方,这份额外访存污染了测量,而且污染程度随 kernel 而异。

改成**读**一个大于 L2 的缓冲后(读进来的都是干净行,驱逐无代价):

| 用例 | 实测 |
|---|---|
| `exact` | 1117 GB/s |
| `ragged` | 1117 GB/s |

单调、可复现,且正好在 4090 的峰值带宽(1008 GB/s)量级。

> 占比略超 100% 是事件计时的固有现象:停止计时的瞬间,最后一批写还在 L2 里没落盘,
> 少算了这部分显存流量。报告里会注明,不影响相对比较。

### 2. 用例要大到让计时误差可忽略

`cudaEvent` 的分辨率约 0.5µs。一个 12µs 的 kernel,误差就是 4%。

所以:

- **用例规模**:访存型题目数据量要 ≥ 100MB(如 32M 个 float = 128MB,约 130µs)
- **低于 20µs 的用例不参与评级**(仍参与正确性判定)。那个量级上测的是启动开销和
  计时器抖动,不是 kernel 性能。
- **上限**:基线耗时超过 5 秒会让做题体验崩坏。`leet validate` 会检查这两端。

---

## 四、评分:两套指标,按瓶颈选

### 为什么访存瓶颈题不能用加速比

向量加法的朴素实现 —— 一线程一元素、`if (i < n)`、合并访问 —— **已经是理论最优**。
它跑满带宽,任何"优化"都不会让它更快。所以:

- 朴素版 vs 优化版 = 1.0x,**毫无区分度**
- 而"距带宽天花板还有多远"才是这类题真正值得看的东西

于是访存瓶颈题按**有效带宽占峰值百分比**评分:

```yaml
perf:
  bound: memory
  metric: bandwidth
  grades: {B: 50, A: 70, S: 85}   # 占峰值百分比
```

### 为什么计算瓶颈题用加速比

归约题的基线只有一个线程块(128 个 SM 里 127 个闲着),转置题的基线写侧不合并
(只跑到 267 GB/s)。这类题的朴素实现**远非最优**,加速比有真实区分度:

```yaml
perf:
  bound: compute
  metric: speedup
  grades: {B: 3.0, A: 10.0, S: 20.0}
```

### 门槛必须物理可达

转置题第一版我设了 `S: 8.0`。但基线只跑到 267 GB/s、峰值 1008 GB/s,
**理论最大加速比就是 3.8x** —— 8x 是永远拿不到的废门槛。

`leet validate` 会检查这一点,但更根本的做法是:**先测量,再定门槛**。
出题提示词里也反复强调这一点(这是自动出题最容易犯的错之一)。

### 性能只评级,不卡关

正确性(含越界检测)决定通过与否,性能给 S/A/B/C。
初学阶段被性能门槛卡住很打击人,而优化动力靠评级就够了。

---

## 五、题库自验证:区分度检查

### 问题

自动生成或手写的题目,最危险的失败模式不是"参考解写错了"—— 那会让基线对拍失败,
很容易发现。真正的危险是**测试太弱**:随便写点什么都能过。这种题做起来毫无意义,
而且**人眼看不出来**。

### 做法

`leet validate` 注入一组**必然错误**的实现,要求每一个都被判失败:

| 劣化解 | 若它竟然通过了,说明 |
|---|---|
| 空实现 | 漏写输出也能过 —— 测试形同虚设 |
| 输出全填 0 | 期望值恰好是 0 —— 用例数据退化 |
| 输出全填 1 | 期望值是常数 —— 用例数据退化 |
| 只写每块首元素 | 只检查了首元素 |

关键性质:这组劣化解**从 spec 泛化生成**(只需要知道有哪些 out 缓冲),
与具体算法无关。所以它们对任何题目都适用 —— 这正是它能用于**全自动出题**的原因:
出题者不需要为每道题手写反例,框架能自己证明题目的测试有区分度。

### 差分测试:基线 vs 参考解

参考解是"标准答案",没法自己验自己。但基线是一份独立写出来的朴素实现 ——
如果它在**所有**用例上都和参考解一致,两边同时写错的可能性极低。

这是本项目唯一的"正确性验证"手段,替代了通常的单元测试。

---

## 六、性能工程:一个被机器特性逼出来的架构

### 发现

本机(共享服务器)上,**起进程极贵**:

| 操作 | 实测 |
|---|---|
| CUDA 上下文初始化(空程序只调 `cudaFree(0)`) | **4.4 s** |
| `nvidia-smi`(任何参数,不缓存) | **4.3 s** |
| 裸 python 启动 | 0.02 s |

两者同源:都是驱动初始化。这让第一版设计完全崩了:

- harness 每个用例起一个进程、每次稳定性重复再起一个 → 一次 `leet test` 起 **8 个进程**
  = 35 秒纯启动开销,而 kernel 本身只跑 0.18ms
- `load_config()` 每条命令都调 `nvidia-smi` 探测 GPU 架构 → `leet list` 要 4.6 秒,
  而它根本不需要 GPU 信息

### 对策

1. **harness 一个进程跑完所有用例**(`--case all`),稳定性重复也在进程内
   (`--verify-repeat`);L2 清空缓冲整个进程只分配一次
2. **架构探测惰性化**:`Config.arch` 是 property,首次访问才算
3. **GPU 探测两级缓存**:进程内 memo + `~/.cache/leetstudy/gpu.json`
   (利用率 TTL 60s、架构 TTL 30 天)
4. **一次判题内固定 GPU 选择** —— 这既是省探测,也是正确性要求:
   同一次运行的所有用例必须落在同一张卡上,加速比才可比
5. **编译缓存**:按内容哈希(源码 + spec + 参考解 + 架构 + 编译选项)判断,
   基线永远命中,学习者的源码改了才重编

效果:

| 命令 | 修复前 | 修复后 |
|---|---|---|
| `leet list` | 4.56 s | 0.16 s |
| `leet test`(冷编译) | 49 s | 21 s |
| `leet test`(源码未变) | 49 s | 11 s |

### 一处刻意不做的优化

还可以**缓存基线耗时**,让 `leet test` 只起一个 CUDA 进程(再省约 6 秒)。

放弃了。因为基线的测量时刻与学习者解答的测量时刻如果机器负载不同,加速比就失真,
而评级正是基于这个比值。**宁可慢 6 秒,不可错一个评级。**

### 同样的理由决定了另一个缓存该放在哪一层

`leet test` 会把判题结果存进 `build/<题号>/last_verdict.json`。`leet review` 在
「解答、spec、参考解、基线都没变且选项一致」时直接复用它,省掉一次判题
(实测 11.3 s → 4.5 s)。

**但 `test` 自己绝不复用。** 同一个缓存,两个消费者,处理方式不同,判据是
「这份数据的用途是什么」:

| 消费者 | 数据的用途 | 能不能用旧数据 |
|---|---|---|
| `review` | 给模型的**上下文** | 能 —— 略有陈旧不影响讲评 |
| `test` | 决定**评级** | 不能 —— 会把成绩变成"上次测的时候机器忙不忙" |

这和上面拒绝缓存基线耗时是同一条原则。设计缓存时先问「这份数据是用来
**判断**的,还是用来**参考**的」—— 前者必须新鲜。

实现上有三处细节值得留意:

- **判据用内容哈希,不用 mtime。** `touch` / `git checkout` / 编辑器无改动保存都会
  动 mtime,但内容没变就该继续命中;反过来内容变了必须失效。
- **选项也进判据。** `leet test --no-sanitize` 那次没有内存检查结果,
  不能拿去满足一个想要消毒报告的 `review`。
- **不存派生值。** 缓存里只有原始数据(`raw`),评级、加速比、带宽占比在读回来时
  由 `_score()` 重算 —— 评分逻辑只留一处,避免两处算法漂移。改动 Verdict 结构时
  递增 `_CACHE_VERSION` 即可让旧缓存自然失效。

### 一个反直觉的发现:PyTorch 题的 validate 比 CUDA 题还慢

直觉上「解释型语言不用编译,应该更快」—— 实测不是:

| | `leet validate` 耗时 |
|---|---|
| CUDA 题(`01-vector-add`) | **40 s** |
| PyTorch 题(`py01-softmax-dim0`) | **85 s** |

原因是区分度检查要跑 **6 个变体**(基线 + 模板 + 4 个劣化解),而每个变体:

1. 是一次**独立的 Python 进程**(要 import torch,1.6 秒)
2. 要在 **CPU 上以 float64 重算一遍参考解** —— 8192×8192 的 softmax 参考解,
   max/减/exp/求和/除 五遍 67M 元素的 float64 运算,单次就是秒级

而 CUDA 侧只有 2 个变体需要跑完整用例(基线 + 用户解),劣化解与模板只跑最小用例,
且 harness 一个进程能跑完全部 —— 编译虽慢(2.5 秒/次),但没有 6 次进程启动
和 6 次 CPU 参考解重算。

> 教训:**「不需要编译」不等于「快」。** 成本会转移到别处 —— 进程启动、库导入、
> 以及每次都要重算的 oracle。这也是为什么 `docs/authoring.md` 里提醒
> 「别在循环里反复跑 validate」。

一个尚未做的优化:把参考解的输出按 (题目, 用例, 种子) 缓存到磁盘,
6 个变体就能共用一份 oracle 结果。CUDA 侧同样受益(那里是 2 份)。

---

## 七、加一个新科目

需要三步:

**1. 实现 `Subject` 基类**(`subjects/base.py`)。必须提供:

```python
class MySubject(Subject):
    name = "mylang"
    template_filename = "template.xx"
    reference_filename = "reference.xx"
    baseline_filename = "baseline.xx"
    solution_filename = "solution.xx"
    required_entry_keys = ("function",)   # spec.entry 里必须有的键
    build_label = "准备"                   # 报告里的措辞
    build_note = "语法检查"

    def prepare_variant(self, problem, impl_src, out_dir, label="variant") -> BuildResult:
        ...   # 编译 / 语法预检;产物包成 Artifact

    def run_case(self, artifact, case, perf=False, timeout=None, extra_args=None) -> CaseResult:
        ...
```

可选覆盖:`run_all_cases`(默认逐个跑;若起进程很贵就覆盖它)、`sanitize`、
`mutants`、`scaffold`。

**2. 在 `subjects/__init__.py` 的 `_REGISTRY` 里登记**(模块路径 + 类名,惰性导入)。

**3. 保证 `CaseResult.raw` 的 JSON 结构与其它科目一致** —— 报告层与诊断层直接消费它:

```json
{
  "case": "...", "params": {...}, "ok": true,
  "outputs": [{"name","count","bad","max_abs_err","max_rel_err",
               "first_bad_idx","first_got","first_exp","nan_count","inf_count"}],
  "guards": {"<buf>": {"front": 0, "back": 0}},
  "verify_pass": 3, "verify_total": 3,
  "perf": {"median_ms","min_ms","runs","warmup","bytes_moved","gb_per_s"}
}
```

**Triton 会很轻松**:torch 已经带了 triton 3.0.0,而 Triton 的 JIT 对框架透明 ——
直接复用 PyTorch 科目的 runner 即可。

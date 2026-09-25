# 排错手册

按「症状 → 原因 → 处理」组织。环境相关的几条来自本机实测,别的机器上未必成立。

---

## 一、环境

### `leet: command not found`

`leet` 装在项目内 venv 里,不在 PATH 上。

```bash
# 直接用全路径
/home/25yww/Projects/leet_study/.venv/bin/leet list
# 或者加个别名
echo "alias leet='/home/25yww/Projects/leet_study/.venv/bin/leet'" >> ~/.bashrc
```

### `leet doctor` 报「Python 环境」不通过

默认 `python3` 是 `/home/user49/anaconda3/bin/python3` —— **别人 home 目录下的共享
anaconda**。项目一律用自己的 `.venv`,不要往 anaconda 里装包。

```bash
.venv/bin/leet ...        # 用 venv 的
# 或 source .venv/bin/activate
```

### pip 装包卡住不动

**下载要分国际/国内两条路,别一刀切挂代理。**

| 目标 | 怎么做 | 实测 |
|---|---|---|
| 国内镜像(清华等) | **直连,不要挂代理** | 直连 0.22s,挂代理 0.66s;挂代理时 pip 拉 torch 会直接卡死 |
| 国际站点(pypi.org / download.pytorch.org) | 挂 `http://127.0.0.1:11451` | 直连 12s 且只拿到 125KB,挂代理 2.25s 拿全 |

```bash
# 国内镜像(推荐,快)
.venv/bin/pip install <包> -i https://pypi.tuna.tsinghua.edu.cn/simple

# 国际站点
export https_proxy=http://127.0.0.1:11451 http_proxy=http://127.0.0.1:11451
```

### torch 装了但用不了 CUDA / 报驱动版本不兼容

驱动是 **535.183.01,最高支持 CUDA 12.2**。所以只能装 **cu121 及更早**的构建:

- PyPI 上 `torch<=2.4.1` 的默认 wheel 是 **cu121** ✅
- `torch>=2.5` 的默认 wheel 是 **cu124+**,驱动过老,不可用 ❌

```bash
.venv/bin/pip install torch==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/leet doctor      # 会打印 torch 版本与 CUDA 是否可用
```

### 自定义 CUDA 算子题报 `Ninja is required to load C++ extensions`

ninja 装了,但 **torch 是按「PATH 上有没有 ninja 可执行文件」判断的**,
光在 Python 里 `import ninja` 不算。

框架已经处理了(给子进程注入 `.venv/bin`)。若在框架外手动跑,自己加:

```bash
export PATH=/home/25yww/Projects/leet_study/.venv/bin:$PATH
```

### 命令莫名其妙的慢

已优化过,现在 `leet list` ≈ 0.16s、`leet test` ≈ 11–21s。如果明显更慢:

- 检查 `~/.cache/leetstudy/gpu.json` 是否存在 —— 删了会让下一条命令重探 GPU
- 这台机器 **`nvidia-smi` 单次要 4.3 秒、CUDA 上下文初始化要 4.4 秒**。
  框架对两者都做了缓存/惰性化,但如果有人新增了「每个用例起一个进程」的路径,
  立刻就会退化。详见 `docs/design.md` 第六节。

---

## 二、判题

### 编译失败

诊断层会挑出属于**你的文件**的报错并给出行号。常见原因:

- kernel 或 launcher 的**函数签名**与 `spec.entry` 不一致
  (签名必须完全匹配 —— 框架按名字调用)
- 模板里改了 `#include "ctx.h"`

### `CUDA 错误 cudaErrorInvalidConfiguration`

启动配置非法。最常见的是**网格算成了 0**:

```cuda
int grid = (ctx.n + block - 1) / block;   // n=0 时 grid=0 → 报错
```

也可能线程块超过 1024 线程,或 grid 维度超过 2^31-1。

### `CUDA 错误 cudaErrorIllegalAddress`

非法地址访问 —— 读了或写了不属于你的显存。检查下标上界。

`leet test <题号> --race` 能拿到出错的具体行号。

### 「输出 `c` 全部是 NaN,正好是框架填入的毒值」

**kernel 没有写入这块缓冲**。检查:

- kernel 真的被启动了吗?(launcher 里有没有 `<<< >>>`)
- 下标是否落在这个缓冲的范围内?
- 是不是把结果写到了别的缓冲?

### 「越界写:缓冲 `x` 后面的哨兵区被改写了 N 个元素」

最常见的就是**漏了边界判断**。写法上 `n` 不整除线程总数时,最后一批线程会越界。

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];     // ← 这个 if 不能省
```

哨兵区的好处是能告诉你是**往前**还是**往后**越界,以及越界了多少个元素。

### 「结果不稳定,重复 N 次只对了 M 次」

几乎可以断定是**竞态**或读了未初始化的内存。检查共享内存的读写是否需要
`__syncthreads()`。

`leet test <题号> --race` 会跑 `compute-sanitizer --tool racecheck` 精确定位
(很慢,只在小用例上跑)。

### 结果不匹配,但误差很小(1e-7 量级)

**这是正常的**,不是 bug。归约、扫描这类操作的求和顺序与 CPU 参考解不同,
浮点结果必然有微小差异。看容差是否合理即可。

若误差大得多,检查:边界处理、初始化、以及是否有未定义行为。

### 性能那一栏显示「样本太小(6µs),测的是开销而非 kernel 性能」

用例规模不够。`cudaEvent` 分辨率约 0.5µs,20µs 以下的测量里计时器抖动和启动开销
会主导结果。这是**题目设计**的问题(用例该放大),不是解答的问题。

### 带宽占比超过 100%

正常现象。事件计时的固有特性:停止计时的瞬间,最后一批写还在 L2 里没落盘,
少算了这部分显存流量。不影响相对比较。

---

## 三、`leet new`(自动出题)

### 跑很久没动静

出题是「写 → 编译 → 跑 → 改」的长循环,一次 20–40 分钟。瓶颈是模型响应延迟
加上 `leet validate` 反复编译。默认超时 3600 秒,可用
`LEETSTUDY_AUTHOR_TIMEOUT` 调整。

看进度:`leet new` 会实时汇报每一步(读了哪个文件、执行了什么命令)。

### 报 `unrecognized_model`

本机 claude 是**自定义模型接入**,不能硬编码 `--model`。
框架默认继承你的 claude 配置,别去指定具体模型。

### 出来的题验证不过

框架会自动把失败报告回喂给出题者修,最多 `--repair-rounds` 轮(默认 3)。
多轮仍不过就会退出并保留现场,可以自己看看 `problems/<id>/` 下的文件。

### 发现工作区里有我没写过的文件变动

**先排除自己这一侧。** `leet new` 会拉起一个嵌套的 Claude 实例,它读写同一份仓库,
而且与你共用同一个 project memory 目录。它看不到你的操作 —— 曾经发生过:
嵌套实例观测到外层会话清理 `solutions/` 的动作后,把自己同时段写入的文件也归因成
「另一个并发会话」。

排查方法:按时间戳取证,再看那个时间窗里有没有 `leet new` 在跑。

它留下的 `solutions/<id>/solution.cu` 是剧透,要删掉;它记录的 `progress.json`
成绩不算你的,要重置。

---

## 四、题库

### 某道题在 `leet list` 里消失了

`spec.yaml` 加载失败。`leet list` 会在表格下方打印加载失败的题目与具体原因。
常见原因:YAML 缩进错误、`id` 与目录名不一致、`shape` 引用了未声明的参数名。

### `leet list` 里看不到 `progress.json` 的变化

`progress.json` 在 `.gitignore` 里,是**本地进度**,不进版本库。
换台机器或删掉它,进度就清零了。

### 磁盘占用

`build/` 是编译产物,可以随时清:

```bash
leet clean          # 会显示占用大小并确认
```

`~/.cache/pip` 可能很大(本项目实测 17GB)。要清用 `pip cache purge` ——
但会影响其它项目的安装速度,自行权衡。

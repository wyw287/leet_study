# leet web —— 本地 web 界面

把 `leet` 的刷题循环搬到浏览器里。**独立工具,不 import 框架代码。**

```bash
.venv/bin/python webui/server.py            # http://127.0.0.1:8765
.venv/bin/python webui/server.py --open     # 顺便把浏览器打开
.venv/bin/python webui/server.py --port 9000
```

零依赖 —— 只用 Python 标准库(外加本来就在 venv 里的 PyYAML)。前端是单个
HTML 文件,原生 JS,没有构建步骤。

## 它能做什么

- 浏览题库(按科目分组、显示完成状态与最佳评级)
- 读题面(含表格、代码块、引用)
- 在线编辑解答、保存
- 一键 `test` / `bench`,**实时看输出**
- 看判题结果:逐用例的误差、guards、耗时、带宽、加速比、评级、框架生成的中文提示
- 看参考解

## 语法高亮

用 **Pygments**,服务端渲染。选它而不是前端库,理由有四条:

- **已经在 venv 里** —— 它是 `rich` 的依赖,所以零新增依赖(不是我们直接声明的,
  所以 `_highlight()` 拿不到时会返回 `None`,前端退回纯文本,不会整个挂掉)
- **能区分语言**,而这在这里不是可有可无的:`cuda` / `cpp` / `python` 三个 lexer
  正好对上三种解答(`.cu` / `.cpp` / `.py`)。实测 CUDA lexer 会把 `__global__`
  认成保留关键字(`kr`),纯 C++ lexer 认不出来
- **不依赖 CDN** —— 这台机器访问国际站点要挂代理,网页里引 CDN 会让界面在
  离线或没代理时直接废掉
- **够快**:47 行的解答 1.8ms、103 行的参考解 4.1ms,所以连「边打字边高亮」
  都撑得住

### 两处用法

**参考解页**:服务端在 `/api/problems/<id>` 里直接返回高亮好的 `optimal_html`,
前端塞进 `<pre class="hl">`。

**解答页**:这是个可编辑的 `<textarea>`,而 textarea 内部没法上色。用的是
**透明 textarea 叠在高亮 `<pre>` 上**的做法 —— 输入防抖 180ms 后把源码
POST 到 `/api/highlight`,拿回高亮 HTML 更新底层。

### ⚠️ 打字期间必须让 textarea 显示自己的文字

这是踩过的坑,而且很反直觉:**可见的文字是 `<pre>` 画的,而 textarea 是透明的。**
`<pre>` 要等防抖 + 服务端往返才更新 —— 于是每敲一个字符要过约 200ms 才看得见,
手感就是「慢半拍」。**服务端高亮不可能做到零延迟**,所以这个方案必须让出即时反馈:

| 状态 | 高亮层 `<pre>` | textarea |
|---|---|---|
| 静止 / 高亮就绪 | `visible` | `color: transparent` |
| 刚敲键 | `hidden` | `color: var(--fg)`(原生渲染,零延迟) |

敲键时把 `.ed` 上的 `hl-on` 类摘掉,高亮算完再加回去。两态字形度量完全一致,
切换只是颜色变化,不跳。离开编辑区(`blur`)会立刻高亮,不等防抖。

**顺便澄清一个常见误解**:保存是**只按按钮**才写盘的,防抖只影响高亮,
不会把编辑内容提交到任何地方。

实测每次编辑的开销(纯客户端):替换 `<pre>` innerHTML 0.36ms +
`fitEditor` 0.04ms;服务端 Pygments 2~4ms;本地 HTTP 往返 <3ms —— 合计 6~7ms,
不是瓶颈。瓶颈是上面那个"文字慢半拍"。

**两层都不滚动**:高度由 `fitEditor()` 按内容撑开,纵向交给**页面**滚(和参考解页
一致),横向用 `pre-wrap` 折行。这样可以完全不做"把 textarea 的滚动位置同步给
`<pre>`"那件事 —— 而那正是最初 bug 的来源。

> ⚠️ 这个方案有四个坑,都踩过:
>
> 1. **两层的 `font` / `line-height` / `padding` / `tab-size` / `white-space` /
>    各种 wrap 必须逐项一致**,差一点光标就和文字错位。改动 `.ed` 时别只改一层。
> 2. **高度和滚动要设在 `<pre>` 上,不是它里面的 `<code>`。**
>    踩了两次:可见的是 `<pre>`,把 `height` / `scrollTop` 设到 `<code>` 上毫无
>    效果 —— 表现是「透明的 textarea 撑开了,可见的高亮层还卡在 320px,
>    下半截一片空白,而滚动条照样能滑」。认准 `box.querySelector("pre")`。
> 3. **编辑器里的 `<pre>` 必须解掉 `pre.hl` 的 `max-height:560px`。**
>    那是给参考解预览用的上限;留在这里会卡住高度,而它有 `pointer-events:none`,
>    鼠标滚不动它 —— 超出部分就永远够不着了。
> 4. **量高度别去压 textarea。**「设成 20px 再读 scrollHeight」会让 textarea
>    为了保持光标可见而内部滚动,回来就对不齐。改成让 `<pre>` 的 `height:auto`
>    按内容撑开再读 `offsetHeight` —— 它不影响编辑区。
>
> 验证办法:让页面把两层的 `getComputedStyle` / `getBoundingClientRect().height`
> 渲染到页面上,再用 `firefox --headless --screenshot` 截一张 —— 数值不对就是没对齐。
> 一次截图就能看到,不需要浏览器驱动。

配色:浅色用 Pygments 的 `default`,深色用 `monokai`,由 `/api/highlight.css`
按 `prefers-color-scheme` 输出两套。只出 token 颜色,**背景留给页面自己的
CSS**,这样代码块底色和其余界面一致。

## 设计:为什么是「以用户身份调 CLI」而不是 import 框架

框架的内部 API(`judge.Verdict`、`subjects.Subject` …)可以随便改,只要
`leet` 命令还能用,这个界面就不用跟着动。

通常这种做法的脆弱点是**解析给人看的终端输出**。这里不存在那个问题,因为
web 需要的数据全都有文件级的结构化通道:

| 数据 | 来源 |
|---|---|
| 题目定义 | `problems/*/spec.yaml` |
| 题面 | `problems/*/problem.md` |
| 完成状态 | `progress.json` |
| 判题结果 | `build/*/last_verdict.json` |
| 解答源码 | `solutions/*/solution.*` |

所以规则是:**数据一律读文件,只有「动作」才走 CLI**。

`spec.yaml` 是唯一直接依赖的框架数据格式 —— 但它是出题文档里公开的契约
(「唯一驱动代码生成的文件」),比 import 内部类稳定得多。

> 唯一一处硬编码:参考解的文件名(`optimal.cu` / `optimal.cpp` / `optimal.py`)
> 按科目做了个三行的映射表。代价是加新科目要在这里补一行;收益是完全不依赖
> 框架的 Python API。

## 为什么是串行任务队列

`leet test` 要 11~21 秒(CUDA 上下文初始化就占 4.4 秒),`bench` 更久。所以
HTTP 请求不能同步跑任务 —— 提交进队列,前端轮询进度。

**单工作线程**是刻意的:框架的 `build/<题号>/`、`progress.json` 都是单份的,
同一道题并发跑两次会互相覆盖编译产物。串行执行把这个问题从根上消掉。

## 安全

这个工具的本质是**执行任意学习者代码**(nvcc/g++ 编译并运行解答)—— 那就是
RCE by design。所以:

- **默认只绑 `127.0.0.1`**。要换地址用 `--host`,但先想清楚。
- 能触发的动作是**白名单**(`test` / `bench` / `start` / `solution`),
  界面传什么都不能越出这几个。
- 题号会做路径穿越校验,只接受题库里真实存在的目录名。

## 已知限制

- **单用户**。并发请求会排队,但 `progress.json` 之类的共享状态没有跨进程锁。
- **判题缓存的版本**:`last_verdict.json` 带版本号,框架改动其结构时会失效。
  旧缓存在界面上会显示不出评级 —— 重新 `test` 一次就有。
- 只覆盖**做题**循环。出题(`leet new`,8~40 分钟)和 `validate` 没有接进来。

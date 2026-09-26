"""诊断 —— 把原始判题数据翻译成初学者能直接行动的提示。

这里的每一条都对应一个真实的初学者高频卡点。原则是:**说清现象 + 指向原因 +
给出下一步**,而不是丢一个错误码。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .spec import Problem
from .subjects.base import BuildResult, CaseResult, SanitizeResult

# CUDA 运行时错误 → 中文解释与常见成因。
# 这些是初学者最常撞的,值得逐个写清楚。
CUDA_ERROR_HINTS: Dict[str, str] = {
    "cudaErrorInvalidConfiguration": (
        "启动配置非法。常见原因:grid 或 block 维度算成了 0(比如 n=0 时 "
        "grid=(n+block-1)/block 会得 0),或线程块超过 1024 线程、"
        "grid 维度超过 2^31-1。"
    ),
    "cudaErrorInvalidValue": (
        "传了非法参数。检查 <<<grid, block>>> 里的数字是否为 0 或为负。"
    ),
    "cudaErrorLaunchOutOfResources": (
        "一个线程块要的资源超过硬件上限。典型是共享内存开太大,"
        "或每线程寄存器用量过高。试着调大线程块数、调小每块共享内存。"
    ),
    "cudaErrorLaunchFailure": "kernel 执行中挂掉了,通常是越界访存或非法指令。建议开 memcheck 看具体位置。",
    "cudaErrorIllegalAddress": (
        "非法地址访问 —— 读了或写了不属于你的显存。检查下标上界,"
        "以及 blockIdx/threadIdx 的组合是否可能超出数组长度。"
    ),
    "cudaErrorMisalignedAddress": (
        "地址未对齐。如果你用了 float4/int4 这类向量类型,"
        "访问地址必须是 16 字节对齐的。"
    ),
    "cudaErrorMemoryAllocation": "显存不足。可以调小用例规模。",
    "cudaErrorTimeout": "kernel 执行超时,通常是死循环或工作量爆炸。检查循环变量有没有推进。",
    "cudaErrorUnspecifiedLaunchFailure": "kernel 异常终止,通常是越界或竞态。建议开 memcheck / racecheck。",
}

# 出错阶段 → 说明
_STAGE_HINTS = {
    "launch": "kernel 还没开始跑,启动请求就被拒绝了",
    "执行中(异步错误)": "kernel 已经在跑了,运行期间出错(异步报出)",
    "预热": "性能测试的预热阶段出错",
    "计时循环": "性能测试的计时阶段出错",
}


def describe_runtime_error(stage: Optional[str], name: Optional[str],
                           text: Optional[str]) -> List[str]:
    """把一次运行期错误翻译成人话。

    错误名有两种来源:CUDA 运行时(如 cudaErrorIllegalAddress),
    或解释型语言的异常类名(如 RuntimeError / ValueError)。
    """
    if not name:
        return []
    out: List[str] = []
    where = _STAGE_HINTS.get(stage or "", "")

    if name.startswith("cudaError"):
        out.append(f"CUDA 错误 {name}:{text}('{where or '执行期间'}')")
        hint = CUDA_ERROR_HINTS.get(name)
        if hint:
            out.append(hint)
        if name in ("cudaErrorIllegalAddress", "cudaErrorMisalignedAddress",
                    "cudaErrorUnspecifiedLaunchFailure"):
            out.append("跑一次 `leet test <题号> --race` 可以拿到出错的具体行号。")
        return out

    # 非 CUDA 错误:解释型语言的异常
    out.append(f"{where or '执行'}时抛出了 {name}:{text}")
    if name in ("SyntaxError", "IndentationError"):
        out.append("这是语法错误,检查缩进与括号匹配。")
    elif name == "AttributeError":
        out.append("属性不存在。检查你是否拼错了 ctx 上的字段名 —— "
                   "可用字段见题面或 template 顶部的注释。")
    elif name in ("TypeError", "ValueError"):
        out.append("参数类型/取值不对。检查张量的 dtype 与形状是否与题面一致。")
    elif name == "IndexError":
        out.append("下标越界。检查张量的维度与索引范围。")
    elif name == "NotImplementedError":
        out.append("还有没实现的部分 —— 是不是某个 TODO 忘了填?")
    return out


# 兼容旧名
describe_cuda_error = describe_runtime_error


def describe_outputs(problem: Problem, result: CaseResult, case_name: str = "") -> List[str]:
    """从逐输出统计里读出失败模式。

    case_name 非空时给每条提示加上用例前缀 —— 多用例的题里,不说是哪个用例的
    报错基本没法用。
    """
    hints: List[str] = []
    prefix = f"[{case_name}] " if case_name else ""
    for out in result.outputs:
        name = out.get("name", "?")
        count = int(out.get("count") or 0)
        bad = int(out.get("bad") or 0)
        nan_count = int(out.get("nan_count") or 0)
        inf_count = int(out.get("inf_count") or 0)
        if bad == 0:
            continue

        buf = next((b for b in problem.buffers if b.name == name), None)
        dtype = buf.dtype if buf else "f32"
        is_float = dtype in ("f32", "f64")

        # 清一色毒值 → kernel 根本没写这块输出
        if is_float and count and nan_count == count:
            hints.append(
                f"{prefix}输出 {name} 全部是 NaN,正好是框架填入的毒值 —— "
                f"说明你的 kernel 没有写入这块缓冲。"
                f"检查 kernel 是否真的被启动了、下标是否落在这个缓冲的范围内。"
            )
            continue

        if is_float and inf_count and inf_count == count:
            hints.append(f"{prefix}输出 {name} 全部是 Inf,检查是否有除零或指数溢出。")
            continue

        if count and bad == count:
            hints.append(
                f"{prefix}输出 {name} 的 {count} 个元素**全部**不对。"
                f"更像是整体写错了位置或写错了值,而不是边界问题。"
            )
        elif count and bad == count // 2:
            hints.append(
                f"{prefix}输出 {name} 恰好一半元素不对 —— 很像循环上界只覆盖了一半"
                f"(比如用了 n/2 而不是 n),或者只处理了一个线程块。"
            )
        else:
            idx = out.get("first_bad_idx", -1)
            got = out.get("first_got")
            exp = out.get("first_exp")
            hints.append(
                f"{prefix}输出 {name}:{count} 个元素里错了 {bad} 个,"
                f"首个失配在下标 {idx}(得到 {got},期望 {exp})。"
            )
            if is_float and nan_count:
                hints.append(f"  其中 {nan_count} 个是 NaN —— 检查是否有 0/0、越界读。")

        if is_float and out.get("max_abs_err") is not None:
            hints.append(
                f"  最大绝对误差 {out['max_abs_err']:.3g},"
                f"最大相对误差 {out.get('max_rel_err', 0):.3g}。"
                f"若误差很小但不为 0,通常是浮点累加顺序不同所致,属正常。"
            )
    return hints


def describe_guards(problem: Problem, result: CaseResult) -> List[str]:
    """哨兵区被踩 → 越界写。这是最值得说清楚的一类错误。"""
    hints: List[str] = []
    guards = result.guards
    for buf in problem.buffers:
        g = guards.get(buf.name)
        if not g:
            continue
        front = int(g.get("front") or 0)
        back = int(g.get("back") or 0)
        if front:
            hints.append(
                f"⚠️  越界写:缓冲 `{buf.name}` **前面**的哨兵区被改写了 {front} 个元素。"
                f"说明你的 kernel 写到了下标 0 之前(负下标或下溢)。"
            )
        if back:
            hints.append(
                f"⚠️  越界写:缓冲 `{buf.name}` **后面**的哨兵区被改写了 {back} 个元素。"
                f"说明你的 kernel 写到了数组末尾之后 —— 最常见的原因就是漏了 "
                f"`if (i < n)` 这类边界判断。"
            )
    if hints:
        hints.append("越界写有时「看起来结果是对的」,但它在悄悄破坏别的数据,必须修掉。")
    return hints


def describe_build_failure(problem: Problem, build: BuildResult, user_src: Optional[Path]) -> List[str]:
    """编译失败 → 挑出属于用户文件的报错。"""
    log = build.log
    if not log:
        return ["编译失败,但没有输出。请手动跑一次编译命令看看。"]
    hints: List[str] = []
    user_path = str(user_src.resolve()) if user_src else ""

    # nvcc 的报错格式:/path/to/file.cu(123): error: xxx
    errs = re.findall(r"^(/[^(\n]+)\((\d+)\):\s*(?:fatal\s+)?error:\s*(.+)$",
                      log, re.MULTILINE)
    user_errs = [e for e in errs if user_path and e[0] == user_path]
    if user_errs:
        hints.append("你的代码里有编译错误:")
        for path, line, msg in user_errs[:5]:
            hints.append(f"  第 {line} 行:{msg}")
        hints.append(f"(文件:{user_path})")
    elif errs:
        hints.append("编译报错发生在框架生成的文件里,这通常意味着模板接口被改坏了:")
        for path, line, msg in errs[:3]:
            hints.append(f"  {Path(path).name}:{line}: {msg}")
        hints.append("检查你是否改动了 kernel 或 launcher 的函数签名。")
    elif "No such file" in log:
        hints.append("找不到源文件。是不是还没执行 `leet start`?")
    else:
        hints.append("编译失败,原始输出如下(截断):")
        hints.append("\n".join(log.splitlines()[:12]))
    return hints


# --------------------------------------------------------------------------- #
# 源码静态提示(不影响判分,只作教学提醒)
# --------------------------------------------------------------------------- #

_RE_CUDA_MALLOC = re.compile(r"\bcudaMalloc\b|\bcudaFree\b")
_RE_CUDA_MEMCPY = re.compile(r"\bcudaMemcpy\b")
_RE_SYNC = re.compile(r"\bcudaDeviceSynchronize\b")
_RE_SHARED = re.compile(r"__shared__")
_RE_SYNCTHREADS = re.compile(r"__syncthreads\s*\(")
_RE_LAUNCH = re.compile(r"<<<")


def source_hints(src: str, subject: str = "cuda") -> List[str]:
    """对用户源码做轻量静态扫描,给出提示。

    提示是**科目相关**的:CUDA 关心有没有启动 kernel、有没有漏同步;
    PyTorch 关心有没有同步点、有没有把张量当 Python 对象逐个处理;
    C++(优化题)关心有没有把外层框架的活重复做一遍。

    ⚠️ 科目必须在这里分派干净。曾经 cpp 科目落进了 CUDA 分支,于是
    `leet test cpp01` 对着一个纯 C++ 文件报「源码里没有找到 <<< >>> 启动语法」
    —— 提示本身没错,但科目搞错了,读起来像框架坏了。
    """
    if subject == "pytorch":
        return _pytorch_source_hints(src)
    if subject == "cpp":
        return _cpp_source_hints(src)

    def strip_comments(text: str) -> str:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//[^\n]*", "", text)
        return text

    return _cuda_source_hints(strip_comments(src))


def _cpp_source_hints(src: str) -> List[str]:
    """C++ 优化题的提示。

    这里刻意**不给"你应该这样优化"的建议** —— 那等于把答案写在提示里。
    只提醒三类「把框架已经做好的事又做了一遍」的情况,那些都是纯粹的浪费,
    指出来不泄露思路。
    """
    hints: List[str] = []
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    code = re.sub(r"//[^\n]*", "", code)

    if re.search(r"\b(malloc|calloc|realloc|new\s+\w|std::vector\s*<)", code):
        hints.append(
            "检测到函数里自己分配了内存。框架已经把输入输出都准备好了;"
            "如果每调用一次就分配一次,分配器本身的开销会算进性能里 —— "
            "自带工作区的话,考虑放进 spec 声明的 scratch 缓冲,或只分配一次。"
        )
    if re.search(r"\b(printf|std::cout|fprintf|cerr)\b", code):
        hints.append(
            "检测到函数里有打印。I/O 极慢且会被计入耗时,调试完记得删掉。"
        )
    if re.search(r"\b(ifstream|ofstream|fopen|fwrite)\b", code):
        hints.append("检测到文件读写,它会被计入耗时。")
    return hints


def _cuda_source_hints(code: str) -> List[str]:
    hints: List[str] = []

    if _RE_CUDA_MALLOC.search(code):
        hints.append(
            "检测到 launcher 里调用了 cudaMalloc/cudaFree。本题的数据已由框架分配好,"
            "不需要自己开显存;而且这些调用在计时区内,会拖慢测得的性能。"
        )
    if _RE_CUDA_MEMCPY.search(code):
        hints.append(
            "检测到 launcher 里调用了 cudaMemcpy。搬运已由框架完成,"
            "自己再拷贝既多余又会把耗时算进性能里。"
        )
    if _RE_SYNC.search(code):
        hints.append(
            "检测到 cudaDeviceSynchronize。框架在计时区外自己会同步,"
            "放在 launcher 里会强制打断流水并计入耗时。"
        )
    if _RE_SHARED.search(code) and not _RE_SYNCTHREADS.search(code):
        hints.append(
            "用了 __shared__ 但全文找不到 __syncthreads()。"
            "多个线程读写同一块共享内存却没有同步,结果会随机出错 —— "
            "这类 bug 往往「有时对有时错」,建议开 racecheck 确认。"
        )
    if not _RE_LAUNCH.search(code):
        hints.append("源码里没有找到 <<< >>> 启动语法,launcher 似乎没有真正启动 kernel。")
    return hints


# PyTorch 侧的高频性能陷阱:同步点、逐元素 Python 循环
_RE_TORCH_SYNC = re.compile(r"torch\.cuda\.synchronize\s*\(|\.item\s*\(\s*\)|\.cpu\s*\(\s*\)|\.numpy\s*\(\s*\)")
# 注意要允许链式属性:本题的张量挂在 ctx 上,真实写法是 ctx.x.shape[0]。
# 早先只写了 \w+\.shape,结果恰好漏掉最该命中的那一种。
_RE_TORCH_LOOP = re.compile(
    r"for\s+\w+\s+in\s+range\s*\(\s*[\w.]+\.(?:shape|size)"
)


def _python_code_only(src: str) -> str:
    """把注释与字符串字面量(含 docstring)抹成等长的空白,只留可执行代码。

    两个都必须做对:

    * **为什么用 tokenize 而不是正则** —— 只有它能可靠区分「字符串里提到某写法」
      和「真的写了那种写法」。这不是理论问题:模板的 docstring 里专门有句提醒
      「不要写 `ctx.out = ...`」,正则会把**那句提示本身**当成违规代码,
      于是每道 PyTorch 题一跑就报一次假警报。

    * **为什么是「抹成等长空白」而不是「取出 token 再拼回去」** —— 后者会改变
      相邻符号之间的距离,`.item()` 会变成 `. item ( )`,于是所有依赖相邻性的
      正则(如 `.item\\s*\\(`)全部失效。原地打码则完整保留原有间距与换行。
    """
    import io
    import tokenize

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src                # 语法不完整时退回原文;编译阶段另有更准确的报错

    # 行号/列号 → 绝对偏移
    line_starts = []
    offset = 0
    for line in src.splitlines(keepends=True):
        line_starts.append(offset)
        offset += len(line)

    def abs_pos(row: int, col: int) -> int:
        if row - 1 >= len(line_starts):
            return len(src)
        return min(line_starts[row - 1] + col, len(src))

    chars = list(src)
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        for i in range(abs_pos(*tok.start), abs_pos(*tok.end)):
            if chars[i] != "\n":       # 保留换行,行结构不乱
                chars[i] = " "
    return "".join(chars)


def _ctx_rebindings(src: str) -> List[str]:
    """用 AST 找出**真的**对 `ctx.<字段>` 的赋值,返回字段名。

    比正则精确:它只看赋值语句的目标,不会被字符串或注释里的同名文本骗到。
    而且能拿到具体字段名 —— 「你把 ctx.out 重新赋值了」比
    「你对 ctx 的某个字段赋值了」有用得多。
    """
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "ctx"):
                names.append(target.attr)
    return names


def _pytorch_source_hints(src: str) -> List[str]:
    hints: List[str] = []
    code = _python_code_only(src)

    if _RE_TORCH_SYNC.search(code):
        hints.append(
            "检测到 .item() / .cpu() / .numpy() 或 torch.cuda.synchronize()。"
            "它们会强制同步 —— GPU 的异步流水会被打断,而且这段等待会被计入计时。"
            "如果要读取某个值来决定下一步,想清楚能不能用张量算子表达。"
        )
    if _RE_TORCH_LOOP.search(code):
        hints.append(
            "检测到对张量形状做 Python 层 for 循环。每个假设的「元素」在这里其实"
            "都是一次独立的内核启动(几十微秒起),而一次算子调用只启动一个内核。"
            "尽量把操作表达成整张量的算子。"
        )

    rebound = list(dict.fromkeys(_ctx_rebindings(src)))
    if rebound:
        fields = "、".join(f"ctx.{n}" for n in rebound)
        affected = [n for n in rebound if n in ("out",)]
        if affected:
            hints.append(
                f"检测到对 {fields} 的赋值。你把它指向了新张量,"
                f"框架就看不到你的结果了 —— 请写进 ctx.out 本身(用 .copy_),"
                f"或者直接 return 张量。"
            )
        else:
            hints.append(
                f"检测到对 {fields} 的赋值。ctx 上的字段由框架准备好,"
                f"重新赋值不会改变实际的数据;要写结果请写进 ctx.out,"
                f"或者直接 return 张量。"
            )
    return hints


def describe_sanitizer(res: SanitizeResult) -> List[str]:
    if res.skipped:
        return [f"{res.tool} 检查已跳过:{res.skip_reason}"]
    if res.ok:
        return []
    out: List[str] = []
    if res.tool == "memcheck":
        out.append(f"内存检查发现 {res.errors} 个错误(越界读写 / 未初始化访问 / 对齐问题)。")
        out.append("下面是与你的 kernel 相关的报错行:")
    else:
        out.append(
            f"竞态检查发现 {res.errors} 个 hazard(共享内存读写缺少同步)。"
        )
        out.append("竞态的特点是**结果不稳定**:同样的输入有时对有时错。必须修掉。")
    # 挑出带行号的片段
    lines = [ln for ln in (res.output or "").splitlines()
             if re.search(r"\.cu\(\d+\)", ln) or "Saved host backtrace" in ln]
    for ln in lines[:8]:
        out.append("  " + ln.strip())
    return out

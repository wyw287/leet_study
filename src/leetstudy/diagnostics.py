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


def describe_cuda_error(stage: Optional[str], name: Optional[str], text: Optional[str]) -> List[str]:
    """把一个 CUDA 运行时错误翻译成人话。"""
    if not name:
        return []
    out: List[str] = []
    where = _STAGE_HINTS.get(stage or "", "")
    out.append(f"CUDA 错误 {name}:{text}('{where or '执行期间'}')")
    hint = CUDA_ERROR_HINTS.get(name)
    if hint:
        out.append(hint)
    if name in ("cudaErrorIllegalAddress", "cudaErrorMisalignedAddress",
                "cudaErrorUnspecifiedLaunchFailure"):
        out.append("跑一次 `leet test <题号> --sanitize` 可以拿到出错的具体行号。")
    return out


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


def source_hints(src: str) -> List[str]:
    """对用户源码做轻量静态扫描,给出提示。"""

    def strip_comments(text: str) -> str:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//[^\n]*", "", text)
        return text

    code = strip_comments(src)
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

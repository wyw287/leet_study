"""题库自验证 —— 核心是「区分度检查」。

一道题最容易出的问题不是「参考解写错了」(那会让基线对拍失败,容易发现),
而是**测试太弱**:随便写点什么都能过。这种题做起来毫无意义,而且很难靠人眼发现。

做法:注入一组**必然错误**的实现 —— 空实现、全填 0、全填 1、只写每块的首元素 ——
要求每一个都被判为失败。任意一个竟然通过了,就说明这道题的测试形同虚设,必须打回。

这组劣化解都是**从 spec 泛化生成**的(只需要知道有哪些 out 缓冲),所以对任何题目
都适用,不需要针对算法写特例。这正是这个检查能用于全自动出题的原因。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..config import Config
from ..spec import DTYPES, Problem, missing_files
from ..subjects.cuda import CudaSubject

# 一道题的基线耗时合理区间(毫秒)。
# 太快 → 用例太小,cudaEvent 的分辨率(~0.5µs)会主导测量,数字不可信;
# 太慢 → 做题体验崩坏(每次 test 都要等好几秒)。
MIN_BASELINE_MS = 0.02
MAX_BASELINE_MS = 5000.0

Check = Tuple[str, bool, str]


# --------------------------------------------------------------------------- #
# 劣化解生成
# --------------------------------------------------------------------------- #

_MUTANT_KINDS = ("noop", "zeros", "ones", "first_only")

_MUTANT_DOC = {
    "noop": "空实现(什么都不做)—— 必须被抓到,否则漏写输出也能过",
    "zeros": "把输出全填 0 —— 若这也能过,说明期望值恰好是 0,测试退化",
    "ones": "把输出全填 1 —— 若这也能过,说明期望值是常数,测试退化",
    "first_only": "只写每块输出的第 0 个元素 —— 若这也能过,说明只检查了首元素",
}


def render_mutant(problem: Problem, kind: str) -> str:
    """按 spec 泛化生成一份必然错误的实现(只需定义 launcher)。"""
    lines: List[str] = [
        "// 自动生成的劣化解 —— 用于区分度检查,不该出现在题目目录里。\n",
        f"// 类型:{kind} —— {_MUTANT_DOC[kind]}\n",
        '#include "ctx.h"\n\n',
    ]

    if kind == "noop":
        lines.append(
            f"void {problem.launcher}(LaunchCtx& ctx) {{\n"
            "    (void)ctx;   // 什么都不做\n"
            "}\n"
        )
        return "".join(lines)

    # 后三种都用「填充 kernel」实现
    fill_ops: List[str] = []
    for b in problem.outputs:
        ct = b.ctype
        kname = f"leet_mut_fill_{b.name}"
        lines.append(
            f"__global__ void {kname}({ct}* buf, long long n, {ct} v) {{\n"
            f"    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
            f"    long long stride = (long long)gridDim.x * blockDim.x;\n"
            f"    for (; i < n; i += stride) buf[i] = v;\n"
            f"}}\n\n"
        )
        count = f"leet_mut_n_{b.name}"
        if kind == "first_only":
            # 只写首元素:仍用同一个 kernel,但只覆盖 n=1
            fill_ops.append(
                f"    long long {count} = 1;   // 故意只覆盖首元素\n"
                f"    {kname}<<<1, 1>>>(ctx.{b.name}, {count}, "
                f"({ct}){'1' if b.dtype in ('f32', 'f64') else '1'});\n"
            )
        else:
            value = "0" if kind == "zeros" else "1"
            cast = f"({ct}){value}"
            fill_ops.append(
                f"    long long {count} = {b.count_expr(prefix='ctx.')};\n"
                f"    {kname}<<<(unsigned)(({count} + 255) / 256), 256>>>"
                f"(ctx.{b.name}, {count}, {cast});\n"
            )

    lines.append(f"void {problem.launcher}(LaunchCtx& ctx) {{\n")
    lines.extend(fill_ops)
    lines.append("}\n")
    return "".join(lines)


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #

MIN_STATEMENT_CHARS = 600
MIN_STATEMENT_HEADINGS = 2


def validate_problem(cfg: Config, problem: Problem) -> List[Check]:
    """对一道题做全套检查,返回 [(检查项, 是否通过, 说明)]。"""
    subject = CudaSubject(cfg)
    build_root = cfg.build_dir_for(problem.id)
    checks: List[Check] = []

    # ---- 1. 文件齐备 ----
    missing = missing_files(problem)
    if missing:
        checks.append(("文件齐备", False, "缺少:" + ", ".join(missing)))
        return checks  # 缺文件后面的都做不了
    checks.append(("文件齐备", True, "spec / 题面 / 模板 / 参考解 / 基线 齐全"))

    # ---- 1b. 题面得有实质内容 ----
    #   只看文件存在是不够的:自动出题很容易留下一个占位符桩文件,
    #   而题面恰恰是学习者唯一会读的东西。
    statement = problem.statement_text()
    headings = statement.count("\n#") + statement.count("\n##")
    if len(statement) < MIN_STATEMENT_CHARS:
        checks.append(("题面有实质内容", False,
                       f"题面只有 {len(statement)} 字符(至少要 {MIN_STATEMENT_CHARS})—— "
                       f"看起来还是占位符,请补上完整讲解"))
    elif headings < MIN_STATEMENT_HEADINGS:
        checks.append(("题面有实质内容", False,
                       f"题面缺少章节结构(只找到 {headings} 个标题)—— "
                       f"应有「题目 / 思路 / 陷阱 / 评分 / 思考题」这样的分节"))
    elif "思考" not in statement:
        checks.append(("题面有实质内容", False,
                       "题面没有「思考题」小节 —— 那是学习者做完之后继续深入的入口"))
    else:
        checks.append(("题面有实质内容", True,
                       f"{len(statement)} 字符,{headings} 个章节"))

    # ---- 2. 编译基线 ----
    baseline_build = subject.compile_variant(
        problem, problem.root / "baseline.cu", build_root, "baseline"
    )
    if not baseline_build.ok:
        first = baseline_build.log.splitlines()[:3]
        checks.append(("编译基线", False, " / ".join(first)))
        return checks
    checks.append(("编译基线", True, f"{baseline_build.seconds:.1f}s"))

    # ---- 3. 基线通过对拍 ----
    #   参考解本身是「标准答案」,没法自己验自己。但基线是一份独立写出来的朴素
    #   CUDA 实现 —— 如果它和参考解在每个用例上都一致,两边同时写错的可能性极低。
    #   这是差分测试:用两个独立实现互相印证。
    bad_cases: List[str] = []
    baseline_ms: dict = {}
    for case in problem.cases:
        r = subject.run_case(baseline_build.exe, case.name, perf=problem.perf.enabled)
        if not r.ok:
            why = r.error_name or "结果不匹配"
            bad_cases.append(f"{case.name}({why})")
        elif r.perf_data.get("median_ms"):
            baseline_ms[case.name] = r.perf_data["median_ms"]
    if bad_cases:
        checks.append(("基线通过对拍", False,
                       f"基线与参考解不一致:{', '.join(bad_cases)} —— "
                       f"参考解或基线有 bug,或容差过严"))
        return checks
    checks.append(("基线通过对拍", True, f"{len(problem.cases)} 个用例全部一致"))

    # ---- 4. 性能地板 ----
    #   注意:题目里**故意**放小用例是合理的(考边界、跑 racecheck),它们会被
    #   判分逻辑自动跳过评级。所以这里的要求是「至少要有一个用例大到可评级」,
    #   而不是「每个用例都得够大」。
    if problem.perf.enabled and baseline_ms:
        gradable = {k: v for k, v in baseline_ms.items() if v >= MIN_BASELINE_MS}
        too_slow = {k: v for k, v in baseline_ms.items() if v > MAX_BASELINE_MS}
        if too_slow:
            detail = ";".join(f"{k} {v:.0f}ms" for k, v in too_slow.items())
            checks.append(("性能地板", False,
                           f"这些用例的基线太慢(>上限 {MAX_BASELINE_MS:.0f}ms),"
                           f"做题体验会很差:{detail}"))
        elif not gradable:
            fastest = min(baseline_ms.values())
            checks.append(("性能地板", False,
                           f"没有任何用例大到可评级(最快的基线仅 {fastest * 1000:.0f}µs,"
                           f"低于下限 {MIN_BASELINE_MS * 1000:.0f}µs)—— "
                           f"这道题的性能评分会完全落空,请放大某个用例的规模"))
        else:
            skipped = sorted(set(baseline_ms) - set(gradable))
            note = f"{len(gradable)} 个用例可评级"
            if skipped:
                note += f",{len(skipped)} 个太小(不参与评级:{', '.join(skipped)})"
            checks.append(("性能地板", True, note))
    else:
        checks.append(("性能地板", True, "本题未开启性能评分,跳过"))

    # ---- 5. 劣化解必须被判失败(区分度检查)----
    positive_control = _template_check(subject, problem, build_root)
    checks.append(positive_control)

    for kind in _MUTANT_KINDS:
        checks.append(_mutant_check(subject, problem, build_root, kind))

    return checks


def _run_cases(subject: CudaSubject, exe: Path, problem: Problem,
               cases=None) -> Tuple[bool, str]:
    """跑若干用例,返回 (是否全部通过, 说明)。

    cases 为 None 时跑全部;区分度检查只传最小用例 ——
    那些检查只关心「劣化解会不会失败」,能在一个小用例上失败就足以证明
    测试有区分度,没必要为它跑上千万个元素的大用例。
    """
    targets = problem.cases if cases is None else list(cases)
    passed: List[str] = []
    for case in targets:
        r = subject.run_case(exe, case.name, perf=False)
        if r.ok:
            passed.append(case.name)
    if not passed:
        return False, f"在 {len(targets)} 个用例上都被判失败"
    return True, f"竟然通过了这些用例:{', '.join(passed)}"


def _template_check(subject: CudaSubject, problem: Problem, build_root: Path) -> Check:
    name = "模板必须失败"
    build = subject.compile_variant(
        problem, problem.root / "template.cu", build_root, "chk_template"
    )
    if not build.ok:
        # 模板编译不过也算「失败」,但更可能是模板本身有语法问题,标出来
        return (name, True, "模板编译不通过(当作失败处理,但建议检查模板语法)")
    ok, detail = _run_cases(subject, build.exe, problem, [problem.smallest_case])
    if ok:
        return (name, False, f"⚠️ 空模板竟然能通过 —— {detail}。这道题在放水!")
    return (name, True, "模板被判失败,符合预期")


def _mutant_check(subject: CudaSubject, problem: Problem, build_root: Path,
                  kind: str) -> Check:
    name = f"劣化解必须失败:{kind}"
    src = build_root / "mutants" / f"{kind}.cu"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(render_mutant(problem, kind), encoding="utf-8")

    build = subject.compile_variant(problem, src, build_root, f"chk_{kind}")
    if not build.ok:
        return (name, False,
                "劣化解编译失败,无法完成检查:" + " / ".join(build.log.splitlines()[:2]))
    ok, detail = _run_cases(subject, build.exe, problem, [problem.smallest_case])
    if ok:
        return (name, False, f"⚠️ {_MUTANT_DOC[kind]} —— 但它通过了。{detail}")
    return (name, True, "被判失败,符合预期")

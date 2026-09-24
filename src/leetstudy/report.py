"""终端报告渲染(rich)。

输出的信息顺序刻意设计成「先能跑,再跑对,再跑快」:
编译 → 正确性 → 稳定性 → 内存/竞态 → 性能 → 提示 → 判定。
初学者最该关心的是前面几项,性能放后面,提示放最后压轴。
"""
from __future__ import annotations

from typing import List, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .bank import Entry, Progress
from .config import Check, Config
from .judge import Verdict
from .spec import Problem
from .subjects.base import SanitizeResult

OK = "[green]✓[/green]"
BAD = "[red]✗[/red]"
SKIP = "[yellow]⏭[/yellow]"

_GRADE_COLOR = {"S": "bright_magenta", "A": "bright_green", "B": "green", "C": "yellow"}


def difficulty_bar(level: int, width: int = 5) -> str:
    return "●" * level + "○" * (width - level)


def grade_bar(grade: Optional[str], width: int = 10) -> str:
    if not grade:
        return "░" * width
    filled = {"S": width, "A": int(width * 0.75), "B": int(width * 0.5),
              "C": int(width * 0.25)}.get(grade, 0)
    color = _GRADE_COLOR.get(grade, "white")
    return f"[{color}]{'█' * filled}[/{color}]{'░' * (width - filled)}"


# --------------------------------------------------------------------------- #
# 判题结果
# --------------------------------------------------------------------------- #

def render_verdict(
    console: Console,
    cfg: Config,
    verdict: Verdict,
    entry: Optional[Entry] = None,
    show_hints: bool = True,
) -> None:
    prob = verdict.problem

    console.print()
    head = Text()
    head.append(f"题目 {prob.id}", style="bold")
    head.append(f"  {prob.title}")
    console.print(head)
    meta = f"难度 {difficulty_bar(prob.difficulty)}"
    if prob.tags:
        meta += "   标签 " + " ".join(prob.tags)
    console.print(f"[dim]{meta}[/dim]")
    console.print("[dim]" + "─" * 62 + "[/dim]")

    # ---- 编译 ----
    _render_build(console, verdict)

    if verdict.fatal:
        console.print()
        _render_hints(console, verdict.hints)
        console.print(f"\n[bold red]判定   ✗ {verdict.fatal}[/bold red]\n")
        return

    # ---- 正确性 ----
    _render_correctness(console, verdict)

    # ---- 消毒检查 ----
    if verdict.checks:
        _render_sanitizers(console, verdict.checks)

    # ---- 性能 ----
    _render_perf(console, cfg, verdict)

    # ---- 提示 ----
    if show_hints and verdict.hints:
        console.print()
        _render_hints(console, verdict.hints)

    # ---- 判定 ----
    console.print()
    if verdict.passed:
        line = Text("判定   ✅ 通过", style="bold green")
        if entry and entry.solved and entry.best_grade == verdict.grade:
            line.append("   (与历史最佳持平)")
        console.print(line)
    else:
        console.print(Text("判定   ✗ 未通过", style="bold red"))
    console.print()


def _render_build(console: Console, verdict: Verdict) -> None:
    build = verdict.build
    if build is None:
        console.print("编译    [dim]未执行[/dim]")
        return
    if build.ok:
        console.print(f"编译    {OK}  nvcc -O3 -lineinfo  [dim]{build.seconds:.1f}s[/dim]")
    else:
        console.print(f"编译    {BAD}  [red]失败[/red] [dim]{build.seconds:.1f}s[/dim]")


def _render_correctness(console: Console, verdict: Verdict) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", no_wrap=True)
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_column(justify="left")

    table.add_row("[bold]正确性[/bold]", "", "", "")
    for cv in verdict.cases:
        r = cv.result
        if r.ok:
            mark = OK
            detail = "[dim]max_err 0[/dim]"
            outs = r.outputs
            if outs:
                err = max((o.get("max_abs_err") or 0) for o in outs)
                rel = max((o.get("max_rel_err") or 0) for o in outs)
                if err:
                    detail = f"[dim]max_err {err:.3g} / rel {rel:.3g}[/dim]"
        elif r.error_name:
            mark = BAD
            detail = f"[red]{r.error_name}[/red]"
        elif r.timed_out:
            mark = BAD
            detail = "[red]超时[/red]"
        else:
            mark = BAD
            bad = sum(int(o.get("bad") or 0) for o in r.outputs)
            total = sum(int(o.get("count") or 0) for o in r.outputs)
            tripped = any((v.get("front") or v.get("back"))
                          for v in r.guards.values())
            # 数值可能全对而败在越界写上 —— 这时说「N 个元素不对」会自相矛盾
            if bad == 0 and tripped:
                detail = "[red]数值正确,但发生了越界写[/red]"
            elif bad == 0:
                detail = "[red]未通过(原因见下方提示)[/red]"
            else:
                detail = f"[red]{bad}/{total} 个元素不对[/red]"
        table.add_row("", f"  {cv.case}", f"[dim]{cv.params_text()}[/dim]", f"{mark}  {detail}")

    # 稳定性
    for cv in verdict.cases:
        if not cv.ok or cv.repeats <= 1:
            continue
        if cv.stable:
            table.add_row("", f"  {cv.case} 稳定性",
                          f"[dim]重复 {cv.repeats} 次[/dim]", f"{OK}")
        else:
            table.add_row("", f"  {cv.case} 稳定性",
                          f"[dim]重复 {cv.repeats} 次[/dim]",
                          f"{BAD}  [red]只对了 {cv.repeats_ok} 次[/red]")
    console.print(table)


def _render_sanitizers(console: Console, checks: List[SanitizeResult]) -> None:
    for chk in checks:
        label = "内存" if chk.tool == "memcheck" else "竞态"
        if chk.skipped:
            console.print(f"{label}    {SKIP}  [dim]{chk.skip_reason}[/dim]")
        elif chk.ok:
            console.print(f"{label}    {OK}  [dim]{chk.tool} {chk.summary}"
                          f"  {chk.seconds:.1f}s[/dim]")
        else:
            console.print(f"{label}    {BAD}  [red]{chk.summary}[/red]"
                          f" [dim]{chk.seconds:.1f}s[/dim]")


def _render_perf(console: Console, cfg: Config, verdict: Verdict) -> None:
    prob = verdict.problem
    if not prob.perf.enabled:
        return
    # 只对跑对的用例评性能 —— 正确性没过时,性能数字没有意义
    with_perf = [c for c in verdict.cases if c.perf and c.ok]
    if not with_perf:
        if verdict.failed_cases or verdict.unstable_cases:
            console.print("性能    [dim]正确性未通过,不评性能[/dim]")
        return

    peak = cfg.resolve_peak_bandwidth()
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(justify="right")
    table.add_column(justify="right")
    table.add_column(justify="left")

    for cv in with_perf:
        ms = cv.perf.get("median_ms")
        ms_txt = f"{ms:.3f} ms" if ms else "—"

        # 太小的用例:报数字,但不评级 —— 这个量级测的是开销和抖动
        if cv.too_small:
            note = f"样本太小({ms * 1000:.0f}µs),测的是开销而非 kernel 性能"
            table.add_row(f"  {cv.case}", ms_txt, f"[dim]{note}[/dim]",
                          "[dim]不评级[/dim]")
            continue

        if prob.perf.is_bandwidth_metric:
            gb = cv.perf.get("gb_per_s") or 0
            pct = cv.bandwidth_pct
            metric_txt = f"{gb:.0f} GB/s"
            if pct is not None:
                metric_txt += f"  [dim]({pct:.0f}% 峰值)[/dim]"
            table.add_row(f"  {cv.case}", ms_txt, metric_txt,
                          f"{grade_bar(cv.grade)} [bold {_GRADE_COLOR.get(cv.grade, 'white')}]"
                          f"{cv.grade or '—'}[/]")
        else:
            sp = f"{cv.speedup:.2f}x" if cv.speedup else "—"
            table.add_row(f"  {cv.case}", ms_txt, f"[dim]相对基线[/dim] {sp}",
                          f"{grade_bar(cv.grade)} [bold {_GRADE_COLOR.get(cv.grade, 'white')}]"
                          f"{cv.grade or '—'}[/]")

    console.print("[bold]性能[/bold]")
    console.print(table)

    scored = [c for c in with_perf if c.metric_value is not None]
    # 次要参照:另一种指标也报一下
    if prob.perf.is_bandwidth_metric:
        base_speeds = [(c.case, c.speedup) for c in scored if c.speedup]
        if base_speeds:
            txt = "  ".join(f"{c} {s:.2f}x" for c, s in base_speeds)
            console.print(f"        [dim]相对基线:{txt}"
                          f"(访存瓶颈题上基线已近最优,≈1.0x 属正常)[/dim]")
    # 超过标称峰值时说明一下,免得以为是 bug
    if peak and any((c.bandwidth_pct or 0) > 102 for c in scored):
        console.print("        [dim]注:占比略超 100% 是事件计时的固有现象 —— "
                      "停止计时的瞬间最后一批写还在 L2 里未落盘,少算了这部分流量。[/dim]")


def _render_hints(console: Console, hints: List[str]) -> None:
    if not hints:
        return
    body = Text()
    for i, h in enumerate(hints):
        if i:
            body.append("\n")
        style = "yellow" if h.startswith("⚠️") else "white"
        body.append(h, style=style)
    console.print(Panel(body, title="[bold]提示[/bold]", border_style="cyan",
                        padding=(0, 1)))


# --------------------------------------------------------------------------- #
# 题面 / 列表 / 统计 / 自检
# --------------------------------------------------------------------------- #

def render_statement(console: Console, problem: Problem, path_text: str = "") -> None:
    """渲染题面。

    必须走 Markdown 渲染器而不是直接 print:题面里有 `c[i] = a[i] + b[i]`
    这样的下标,rich 的标记解析会把 `[i]` 当成样式标签吃掉。Markdown 渲染器
    把代码块和行内代码当代码处理,不会被误解析。
    """
    console.print()
    text = problem.statement_text()
    try:
        from rich.markdown import Markdown
        console.print(Markdown(text))
    except Exception:
        # 兜底:markup=False 至少能保证下标不被吃掉
        console.print(text, markup=False, highlight=False)
    if path_text:
        console.print(f"[dim]{path_text}[/dim]")
    console.print()


def render_problem_list(
    console: Console, problems: List[Problem], progress: Progress,
    tag: Optional[str] = None, difficulty: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    rows = []
    for p in problems:
        if tag and tag not in p.tags:
            continue
        if difficulty is not None and p.difficulty != difficulty:
            continue
        e = progress.get(p.id)
        if status == "todo" and e.solved:
            continue
        if status == "done" and not e.solved:
            continue
        rows.append((p, e))

    if not rows:
        console.print("[dim]没有匹配的题目。[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1),
                  expand=False)
    table.add_column("题号", no_wrap=True)
    table.add_column("题目", no_wrap=True, max_width=20, overflow="ellipsis")
    table.add_column("难度", no_wrap=True)
    table.add_column("状态", no_wrap=True)
    table.add_column("最佳", justify="right", no_wrap=True)
    # 标签只显示前两个:全列出来会把标题挤到换行,反而更难读。
    # 要按标签过滤直接用 --tag。
    table.add_column("标签", style="dim", no_wrap=True, max_width=26, overflow="ellipsis")

    for p, e in rows:
        if e.solved:
            state = "[green]✓ 已通过[/green]"
        elif e.attempts:
            state = f"[yellow]◐ 试过 {e.attempts} 次[/yellow]"
        else:
            state = "[dim]○ 未开始[/dim]"

        best = ""
        if e.best_metric is not None:
            if e.metric_name == "bandwidth":
                best = f"{e.best_metric:.0f} GB/s"
            else:
                best = f"{e.best_metric:.2f}x"
            if e.best_grade:
                color = _GRADE_COLOR.get(e.best_grade, "white")
                best += f" [{color}]{e.best_grade}[/]"

        tags = " ".join(p.tags[:2])
        if len(p.tags) > 2:
            tags += f" +{len(p.tags) - 2}"

        table.add_row(p.id, p.title, f"[dim]{difficulty_bar(p.difficulty)}[/dim]",
                      state, best, tags)
    console.print()
    console.print(table)
    solved = sum(1 for p, e in rows if e.solved)
    console.print(f"\n[dim]{solved}/{len(rows)} 已通过[/dim]\n")


def render_stats(console: Console, problems: List[Problem], progress: Progress) -> None:
    total = len(problems)
    solved = [p for p in problems if progress.get(p.id).solved]
    attempted = [p for p in problems if progress.get(p.id).attempts]

    console.print()
    console.print("[bold]学习进度[/bold]")
    console.print("[dim]" + "─" * 62 + "[/dim]")
    frac = len(solved) / total if total else 0.0
    width = 40
    filled = int(width * frac)
    console.print(f"  已通过  [green]{'█' * filled}[/green]{'░' * (width - filled)}"
                  f"  {len(solved)}/{total}")
    console.print(f"  尝试过  {len(attempted)} 题    总提交 {sum(progress.get(p.id).attempts for p in problems)} 次")

    by_diff: dict = {}
    for p in problems:
        e = progress.get(p.id)
        d = by_diff.setdefault(p.difficulty, [0, 0])
        d[1] += 1
        if e.solved:
            d[0] += 1
    if by_diff:
        console.print()
        for level in sorted(by_diff):
            done, tot = by_diff[level]
            console.print(f"  难度 {difficulty_bar(level)}  {done}/{tot}")

    graded = [(p, progress.get(p.id)) for p in problems
              if progress.get(p.id).best_grade]
    if graded:
        console.print()
        console.print("  评级分布")
        order = {"S": 4, "A": 3, "B": 2, "C": 1}
        for letter in ("S", "A", "B", "C"):
            items = [(p, e) for p, e in graded if e.best_grade == letter]
            if not items:
                continue
            color = _GRADE_COLOR[letter]
            console.print(f"    [{color}]{letter}[/{color}]  "
                          f"{len(items):>2} 题   [dim]"
                          + " ".join(p.id.split("-")[0] for p, _ in items) + "[/dim]")
    console.print()


def verdict_to_text(verdict: Verdict, cfg: Config) -> str:
    """把判题结果压成一段纯文本,喂给出题/讲评的提示词。

    不要用 rich 标记 —— 这段文本会直接进模型上下文。
    """
    prob = verdict.problem
    lines: List[str] = []
    lines.append(f"题目:{prob.id} {prob.title}(难度 {prob.difficulty}/5)")
    lines.append(f"评分指标:{prob.perf.metric};门槛 {prob.perf.grades}")
    if verdict.build:
        lines.append(f"编译:{'成功' if verdict.build.ok else '失败'} "
                     f"({verdict.build.seconds:.1f}s)")
        if not verdict.build.ok:
            lines.append(verdict.build.log[:2000])

    lines.append("")
    lines.append("用例结果:")
    for cv in verdict.cases:
        r = cv.result
        if r.ok:
            extra = ""
            if cv.stable is False:
                extra = f"但重复 {cv.repeats} 次里只对了 {cv.repeats_ok} 次(不稳定)"
            lines.append(f"  {cv.case} ({cv.params_text()}): 正确 {extra}")
        elif r.error_name:
            lines.append(f"  {cv.case}: CUDA 错误 {r.error_name} @ {r.stage} —— {r.error_str}")
        elif r.timed_out:
            lines.append(f"  {cv.case}: 超时")
        else:
            outs = ", ".join(
                f"{o.get('name')}: {o.get('bad')}/{o.get('count')} 个元素错,"
                f"max_abs_err={o.get('max_abs_err')}, first_bad_idx={o.get('first_bad_idx')}"
                f"(得到 {o.get('first_got')}, 期望 {o.get('first_exp')})"
                for o in r.outputs
            )
            lines.append(f"  {cv.case}: 结果不匹配 —— {outs}")
        guards = r.guards
        for name, g in (guards or {}).items():
            if g.get("front") or g.get("back"):
                lines.append(
                    f"    ⚠ 越界写:缓冲 {name} 前面被踩 {g.get('front')} 个、"
                    f"后面被踩 {g.get('back')} 个元素"
                )

    if prob.perf.enabled:
        lines.append("")
        lines.append("性能:")
        for cv in verdict.cases:
            if not cv.perf:
                continue
            ms = cv.perf.get("median_ms")
            if cv.too_small:
                lines.append(f"  {cv.case}: {ms:.4f} ms(样本太小,不评级)")
                continue
            bits = [f"{cv.case}: {ms:.4f} ms"]
            if cv.perf.get("gb_per_s"):
                bits.append(f"{cv.perf['gb_per_s']:.0f} GB/s")
                if cv.bandwidth_pct:
                    bits.append(f"占峰值 {cv.bandwidth_pct:.0f}%")
            if cv.speedup:
                bits.append(f"相对基线 {cv.speedup:.2f}x")
            if cv.grade:
                bits.append(f"评级 {cv.grade}")
            lines.append("  " + ", ".join(bits))

    if verdict.checks:
        lines.append("")
        for chk in verdict.checks:
            if chk.skipped:
                lines.append(f"{chk.tool}: 已跳过({chk.skip_reason})")
            else:
                lines.append(f"{chk.tool}: {chk.summary}")
                if not chk.ok:
                    lines.append(chk.output[:1500])

    if verdict.hints:
        lines.append("")
        lines.append("框架自动诊断出的问题:")
        for h in verdict.hints:
            lines.append("  - " + h)

    lines.append("")
    lines.append(f"判定:{'通过' if verdict.passed else '未通过'}")
    return "\n".join(lines)


def render_doctor(console: Console, checks: List[Check], cfg: Config) -> None:
    console.print()
    console.print("[bold]环境自检[/bold]")
    console.print("[dim]" + "─" * 62 + "[/dim]")
    failed = 0
    for c in checks:
        mark = OK if c.ok else BAD
        if not c.ok:
            failed += 1
        console.print(f"  {mark}  {c.name:<18} [dim]{c.detail}[/dim]")
        if not c.ok and c.hint:
            console.print(f"       [yellow]→ {c.hint}[/yellow]")
    console.print()
    if failed:
        console.print(f"[yellow]{failed} 项需要处理。[/yellow]")
    else:
        console.print("[green]全部就绪。[/green]")
    console.print()


def render_broken(console: Console, broken) -> None:
    if not broken:
        return
    console.print("\n[red]以下题目加载失败:[/red]")
    for path, err in broken:
        console.print(f"  [red]{path.name}[/red]  [dim]{err}[/dim]")

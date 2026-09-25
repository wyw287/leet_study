"""leet 命令行入口。

命令设计成 LeetCode 式的肌肉记忆:
    leet list / leet show 1 / leet start 1 / leet test 1
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console

from . import bank, report, subjects
from . import config as cfgmod
# 注意:这里不能写成 `from . import judge` —— 下面 from .judge import judge
# 会把名字 judge 覆盖成函数,于是 judge.save_verdict 这类模块级调用就断了。
# 直接按名导入需要的符号,避免这个歧义。
from .judge import judge, load_verdict, save_verdict
from .spec import SpecError

console = Console()


def _load_bank(cfg):
    problems, broken = bank.discover_with_errors(cfg.problems_dir)
    return problems, broken


def _find(cfg, query: Optional[str], *, allow_recent: bool = False):
    """把题号解析成 Problem。

    query 为空且 allow_recent 时,自动选**最近编辑过的解答**所在的题目 ——
    做题时反复 `leet test` 不必每次敲题号。
    选了什么会明确打印出来(带编辑时间),避免"它怎么选了这个"的困惑。
    """
    problems, broken = _load_bank(cfg)
    report.render_broken(console, broken)

    if not query:
        if not allow_recent:
            console.print("[red]需要指定题号。[/red]")
            sys.exit(1)
        found = bank.find_recent_solution(cfg.solutions_dir, problems)
        if found is None:
            console.print("[red]没有找到任何解答文件。[/red]")
            console.print("[dim]先用 `leet start <题号>` 创建一份。[/dim]")
            if problems:
                console.print("[dim]可用题目:[/dim]")
                for p in problems:
                    console.print(f"  [bold]{p.id}[/bold]  {p.title}")
            sys.exit(1)
        prob, _path, age = found
        console.print(
            f"[dim]未指定题号 → 最近编辑的解答:[/dim] "
            f"[bold]{prob.id}[/bold] [dim]({report.humanize_age(age)})[/dim]"
        )
        return prob

    prob = bank.resolve(problems, query)
    if prob is None:
        hits = bank.resolve_many(problems, query)
        if hits:
            console.print(f"[yellow]'{query}' 匹配到多道题,请指明:[/yellow]")
            for h in hits:
                console.print(f"  [bold]{h.id}[/bold]  {h.title}")
        else:
            console.print(f"[red]找不到题目 '{query}'[/red]")
            if problems:
                console.print("[dim]可用题目:[/dim]")
                for p in problems:
                    console.print(f"  [bold]{p.id}[/bold]  {p.title}")
        sys.exit(1)
    return prob


# --------------------------------------------------------------------------- #

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--root", type=click.Path(exists=True, file_okay=False), default=None,
              help="仓库根目录(默认自动向上查找)")
@click.version_option(package_name="leetstudy", prog_name="leet")
@click.pass_context
def main(ctx: click.Context, root: Optional[str]) -> None:
    """leet —— LeetCode 式的 CUDA 刷题工具。

    只写 kernel 和启动配置,分配/搬运/计时/校验/越界检测全部自动完成。
    """
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfgmod.load_config(Path(root) if root else None)


# --------------------------------------------------------------------------- #

@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """检查环境(nvcc / GPU / compute-sanitizer / claude / Python 环境)。"""
    cfg = ctx.obj["cfg"]
    report.render_doctor(console, cfgmod.doctor(cfg), cfg)


@main.command(name="list")
@click.option("--tag", default=None, help="按标签过滤")
@click.option("--diff", "difficulty", type=int, default=None, help="按难度过滤 1-5")
@click.option("--status", type=click.Choice(["all", "todo", "done"]), default="all")
@click.pass_context
def list_cmd(ctx: click.Context, tag: Optional[str], difficulty: Optional[int],
             status: str) -> None:
    """浏览题库与完成状态。"""
    cfg = ctx.obj["cfg"]
    problems, broken = _load_bank(cfg)
    progress = bank.Progress.load(cfg.root / bank.PROGRESS_FILENAME)
    report.render_problem_list(
        console, problems, progress, tag, difficulty,
        None if status == "all" else status,
    )
    report.render_broken(console, broken)


@main.command()
@click.argument("problem_id", required=False)
@click.pass_context
def show(ctx: click.Context, problem_id: Optional[str]) -> None:
    """读题面。省略题号则取最近编辑过的解答。"""
    cfg = ctx.obj["cfg"]
    prob = _find(cfg, problem_id, allow_recent=True)
    sol = bank.solution_path(cfg.solutions_dir, prob)
    tip = f"解答文件:{sol}" if sol.is_file() else f"还没开始。运行 `leet start {prob.id}` 创建工作区。"
    report.render_statement(console, prob, tip)


@main.command()
@click.argument("problem_id")
@click.option("--force", is_flag=True, help="覆盖已有的解答文件")
@click.pass_context
def start(ctx: click.Context, problem_id: str, force: bool) -> None:
    """创建工作区(从模板生成解答文件)并打印题面。"""
    cfg = ctx.obj["cfg"]
    prob = _find(cfg, problem_id)
    sol = bank.solution_path(cfg.solutions_dir, prob)

    if sol.is_file() and not force:
        console.print(f"[yellow]{sol} 已存在,未覆盖。[/yellow]")
        console.print("[dim]要重新生成请加 --force(会丢失你的代码)[/dim]")
    else:
        template = prob.root / subjects.template_filename(prob.subject)
        if not template.is_file():
            console.print(f"[red]题目缺少模板文件 {template}[/red]")
            sys.exit(1)
        sol.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template, sol)
        console.print(f"[green]已创建[/green] {sol}")
        # 首次开始记一次,让 list 里能区分「没开始」和「试过」
        progress = bank.Progress.load(cfg.root / bank.PROGRESS_FILENAME)
        progress.record(prob.id, solved=False, grade=None, metric_value=None,
                        metric_name="", median_ms=None)
        progress.save()

    report.render_statement(console, prob, f"编辑 {sol} 后运行:leet test {prob.id}")


@main.command()
@click.argument("problem_id", required=False)
@click.option("--case", "cases", multiple=True, help="只跑指定用例(可重复)")
@click.option("--no-sanitize", is_flag=True, help="跳过内存检查(更快)")
@click.option("--race", is_flag=True, help="强制跑竞态检查(很慢,仅在小用例上)")
@click.option("--gpu", type=int, default=None, help="指定物理 GPU 序号")
@click.option("-v", "--verbose", is_flag=True, help="显示原始输出")
@click.pass_context
def test(ctx: click.Context, problem_id: Optional[str], cases: List[str],
         no_sanitize: bool, race: bool, gpu: Optional[int], verbose: bool) -> None:
    """编译 + 判分。这是主命令。省略题号则取最近编辑过的解答。"""
    cfg = ctx.obj["cfg"]
    if gpu is not None:
        cfg.gpu = gpu
    prob = _find(cfg, problem_id, allow_recent=True)
    sol = bank.solution_path(cfg.solutions_dir, prob)

    if not sol.is_file():
        console.print(f"[red]还没有解答文件:{sol}[/red]")
        console.print(f"[dim]先运行 `leet start {prob.id}`[/dim]")
        sys.exit(1)

    with console.status("[dim]编译并运行...[/dim]", spinner="dots"):
        verdict = judge(
            cfg, prob, sol,
            do_sanitize=not no_sanitize,
            force_racecheck=race,
            case_filter=list(cases) or None,
            verbose=verbose,
        )

    progress = bank.Progress.load(cfg.root / bank.PROGRESS_FILENAME)
    best = verdict.best_case
    entry = progress.record(
        prob.id,
        solved=verdict.passed,
        grade=verdict.grade,
        metric_value=best.metric_value if best else None,
        metric_name=prob.perf.metric,
        median_ms=(best.perf.get("median_ms") if best else None),
    )
    progress.save()

    # 存下来供 `leet review` 复用(见 judge.py 顶部关于这个缓存边界的说明)
    save_verdict(cfg, verdict, _test_cache_options(no_sanitize, race, cases))

    report.render_verdict(console, cfg, verdict, entry=entry)

    if verbose:
        _dump_raw(console, verdict)


def _test_cache_options(no_sanitize: bool, race: bool, cases) -> dict:
    """`leet test` 这次跑的选项。必须与 review 侧完全一致才能复用缓存。

    选项不同,判出来的数据就不可互换 —— 比如 --no-sanitize 那次没有内存检查
    结果,拿去讲评就等于漏了一类诊断。
    """
    return {
        "do_sanitize": not no_sanitize,
        "force_racecheck": bool(race),
        "case_filter": sorted(cases or []),
        "perf_repeat": None,
    }


@main.command()
@click.argument("problem_id", required=False)
@click.option("--repeat", type=int, default=200, help="计时重复次数(默认 200,比 test 更稳)")
@click.option("--case", "cases", multiple=True)
@click.option("--gpu", type=int, default=None)
@click.pass_context
def bench(ctx: click.Context, problem_id: Optional[str], repeat: int,
          cases: List[str], gpu: Optional[int]) -> None:
    """只测性能,不跑内存/竞态检查,重复更多次以获得更稳的数字。省略题号则取最近编辑过的解答。"""
    cfg = ctx.obj["cfg"]
    if gpu is not None:
        cfg.gpu = gpu
    prob = _find(cfg, problem_id, allow_recent=True)
    sol = bank.solution_path(cfg.solutions_dir, prob)
    if not sol.is_file():
        console.print(f"[red]还没有解答文件:{sol}[/red]")
        sys.exit(1)

    with console.status(f"[dim]计时中(重复 {repeat} 次)...[/dim]", spinner="dots"):
        verdict = judge(cfg, prob, sol, do_sanitize=False,
                        case_filter=list(cases) or None,
                        perf_repeat=repeat)
    report.render_verdict(console, cfg, verdict, show_hints=False)


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """学习进度看板。"""
    cfg = ctx.obj["cfg"]
    problems, broken = _load_bank(cfg)
    progress = bank.Progress.load(cfg.root / bank.PROGRESS_FILENAME)
    if not problems:
        console.print("[yellow]题库是空的。[/yellow]")
        return
    report.render_stats(console, problems, progress)
    report.render_broken(console, broken)


@main.command()
@click.option("--all", "check_all", is_flag=True, help="检查所有题目")
@click.argument("problem_id", required=False)
@click.pass_context
def validate(ctx: click.Context, problem_id: Optional[str], check_all: bool) -> None:
    """题库健康检查:参考解要对、模板要错、基线要能跑。

    最后一条最关键:如果连空模板都能「通过」,说明这道题的测试形同虚设。
    """
    from .authoring.validate import validate_problem

    cfg = ctx.obj["cfg"]
    problems, broken = _load_bank(cfg)
    report.render_broken(console, broken)
    if problem_id:
        problems = [_find(cfg, problem_id)]
    elif not check_all:
        console.print("[dim]用法:leet validate --all 或 leet validate <题号>[/dim]")
        return

    total_fail = 0
    for prob in problems:
        console.print(f"\n[bold]{prob.id}[/bold]  {prob.title}")
        results = validate_problem(cfg, prob)
        for name, ok, detail in results:
            mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {mark}  {name:<24} [dim]{detail}[/dim]")
            if not ok:
                total_fail += 1
    console.print()
    if total_fail:
        console.print(f"[red]{total_fail} 项检查未通过。[/red]\n")
        sys.exit(1)
    console.print("[green]全部检查通过。[/green]\n")


@main.command()
@click.option("--yes", is_flag=True, help="不确认直接删")
@click.pass_context
def clean(ctx: click.Context, yes: bool) -> None:
    """清掉 build/ 下的编译产物(磁盘紧张时用)。"""
    cfg = ctx.obj["cfg"]
    if not cfg.build_dir.is_dir():
        console.print("[dim]没有编译产物。[/dim]")
        return
    size = sum(f.stat().st_size for f in cfg.build_dir.rglob("*") if f.is_file())
    if not yes:
        click.confirm(f"删除 {cfg.build_dir} ({size / 1e6:.1f} MB)?", abort=True)
    shutil.rmtree(cfg.build_dir, ignore_errors=True)
    console.print(f"[green]已清理[/green] {size / 1e6:.1f} MB")


@main.command()
@click.argument("requirement", nargs=-1, required=True)
@click.option("--subject", type=click.Choice(subjects.available()), default=subjects.DEFAULT_SUBJECT,
              help="出哪个科目的题(默认 cuda)")
@click.option("--count", type=int, default=1,
              help="出几道。>1 时**逐道开独立会话**(每道一个新会话/新超时)")
@click.option("--repair-rounds", type=int, default=3,
              help="独立验证失败时,回喂给出题者修复的最大轮数")
@click.option("--no-validate", is_flag=True, help="跳过独立验证(不建议)")
@click.pass_context
def new(ctx: click.Context, requirement: tuple, subject: str, count: int,
        repair_rounds: int, no_validate: bool) -> None:
    """让本地 claude 自动出题 + 自验证 + 自动修复。

    例:``leet new 出一道关于共享内存 bank conflict 的题``

    例:``leet new --count 3 --subject pytorch 出三道 LayerNorm 相关的题``

    出题者会自己跑 `leet validate` 迭代;完成之后框架**再独立验证一遍**
    (不采信它的自我声明),不过关就把报告回喂给它修。

    `--count N` 是**逐道开独立会话**,不是让一次会话出 N 道 —— 后者会撞超时:
    实测一道 CUDA 题要 25~40 分钟(写文件 → 反复编译 → 跑 validate → 测门槛),
    三道塞进一次会话就是两小时。逐道开还有两个好处:每道题的上下文是干净的;
    而且前一道的产出会自动出现在后一道的「不要重复出这些」列表里。
    """
    from .authoring import agent, prompts
    cfg = ctx.obj["cfg"]
    req = " ".join(requirement)

    if agent.claude_path(cfg) is None:
        console.print(f"[red]找不到 claude CLI({cfg.claude_bin})[/red]")
        console.print("[dim]可用 LEETSTUDY_CLAUDE_BIN 指定路径[/dim]")
        sys.exit(1)

    count = max(1, count)
    if count > 1:
        console.print(f"\n[bold]准备出 {count} 道题[/bold]  [dim]科目 {subject};"
                      f"逐道独立会话,每道有各自的超时[/dim]")

    succeeded: List[str] = []
    failed: List[str] = []
    for i in range(1, count + 1):
        if count > 1:
            console.print(f"\n[bold cyan]═══ 第 {i}/{count} 道 ═══[/bold cyan]")
        created, passed = _author_one(
            cfg, agent, prompts, req, subject, repair_rounds, no_validate,
            only_one=count > 1,
        )
        if not created:
            failed.append(f"第 {i} 道(未产出任何题目)")
            continue
        for pid in created:
            (succeeded if pid in passed else failed).append(pid)

    # ---- 总结 ----
    if count > 1:
        console.print("\n" + "═" * 62)
        if succeeded:
            console.print(f"[green]✓ 可用:{len(succeeded)} 道[/green]  "
                          + ", ".join(succeeded))
        if failed:
            console.print(f"[red]✗ 需处理:{len(failed)} 道[/red]  "
                          + ", ".join(failed))
            console.print("[dim]这些要么没产出,要么验证没过。"
                          "可以自己看看 problems/ 下的文件,或重跑 leet new。[/dim]")
        console.print()
        if not succeeded:
            sys.exit(1)


def _author_one(cfg, agent, prompts, req: str, subject: str, repair_rounds: int,
                no_validate: bool, only_one: bool) -> tuple:
    """出一道题的完整流程:出题 → 独立验证 → 回喂修复。

    返回 (本次产出的题目 id 列表, 其中通过验证的 id 列表)。
    """
    from .authoring.validate import validate_problem

    before = {p.id for p in bank.discover(cfg.problems_dir)}
    prompt = prompts.build_author_prompt(
        req, cfg.root, subject=subject,
        existing_ids=sorted(before) or None, only_one=only_one,
    )

    console.print()
    console.print(f"[bold]出题中[/bold]  [dim]科目 {subject};"
                  f"本地 claude 会写文件、自验证,过程实时汇报[/dim]")
    console.print("[dim]" + "─" * 62 + "[/dim]")
    result = agent.run(cfg, prompt, on_event=lambda m: console.print(f"[dim]  {m}[/dim]"))

    if not result.ok:
        console.print(f"\n[red]出题失败:[/red]{result.error}")
        return [], []
    console.print(f"[dim]  (耗时 {result.seconds:.0f}s)[/dim]")

    problems, broken = bank.discover_with_errors(cfg.problems_dir)
    created = [p for p in problems if p.id not in before]
    report.render_broken(console, broken)
    if not created:
        console.print("\n[yellow]没有检测到新题目。[/yellow]")
        if result.text:
            console.print("\n" + result.text[:1500])
        return [], []

    console.print(f"\n[green]产出了 {len(created)} 道题:[/green]"
                  + ", ".join(p.id for p in created))

    if no_validate:
        return [p.id for p in created], [p.id for p in created]

    # ---- 独立验证 + 修复回路 ----
    for round_no in range(repair_rounds + 1):
        console.print(f"\n[bold]独立验证[/bold](第 {round_no + 1} 轮)"
                      f"  [dim]不采信出题者的自我声明[/dim]")
        console.print("[dim]" + "─" * 62 + "[/dim]")
        per_problem = {}
        for prob in created:
            results = validate_problem(cfg, prob)
            per_problem[prob.id] = results
            failed_checks = [r for r in results if not r[1]]
            mark = "[green]✓[/green]" if not failed_checks else "[red]✗[/red]"
            console.print(f"  {mark} {prob.id}  "
                          f"[dim]{len(results) - len(failed_checks)}/{len(results)} 项通过[/dim]")
            for name, ok, detail in failed_checks:
                console.print(f"      [red]✗ {name}[/red]  [dim]{detail[:150]}[/dim]")

        failures = {pid: [r for r in res if not r[1]]
                    for pid, res in per_problem.items()}
        if not any(failures.values()):
            console.print("\n[green]全部通过 —— 新题已可用。[/green]")
            console.print("[dim]试一下:leet show <题号> / leet start <题号>[/dim]\n")
            return [p.id for p in created], [p.id for p in created]

        if round_no == repair_rounds:
            break

        # 回喂修复
        report_text = []
        for pid, failed_checks in failures.items():
            if not failed_checks:
                continue
            report_text.append(f"## {pid}")
            for name, _, detail in failed_checks:
                report_text.append(f"- 检查项「{name}」未通过:{detail}")
        repair_prompt = (
            "你刚才出的题没有通过框架的独立验证。下面是失败报告:\n\n"
            + "\n".join(report_text)
            + "\n\n请修复这些问题。注意:不要为了让检查通过而放宽容差或删掉检查 —— "
              "要真正修好实现或用例。修完后重新运行 `leet validate <id>` 确认全部通过。"
        )
        console.print(f"\n[yellow]回喂给出题者修复(还有 {repair_rounds - round_no} 轮)[/yellow]")
        fix = agent.run(cfg, repair_prompt,
                        on_event=lambda m: console.print(f"[dim]  {m}[/dim]"))
        if not fix.ok:
            console.print(f"[red]修复调用失败:[/red]{fix.error}")
            break
        # 题目文件可能被改过了,重新加载
        rebuilt = {p.id: p for p in bank.discover(cfg.problems_dir)}
        created = [rebuilt.get(p.id, p) for p in created]

    console.print("\n[red]仍有检查未通过 —— 这些题先不要用。[/red]")
    console.print("[dim]可以自己看看 problems/ 下的文件,或重试 `leet new`。[/dim]\n")
    # 注意:这里必须 return 而不是 sys.exit —— --count>1 时还要跑下一道
    return [p.id for p in created], []


@main.command()
@click.argument("problem_id", required=False)
@click.option("--no-sanitize", is_flag=True, help="讲评前跳过内存检查")
@click.option("--fresh", is_flag=True,
              help="忽略上次 test 的结果,重新判题(默认解答没改就复用)")
@click.pass_context
def review(ctx: click.Context, problem_id: Optional[str], no_sanitize: bool,
           fresh: bool) -> None:
    """让本地 claude 讲评你的 kernel(结合性能数据与消毒报告)。省略题号则取最近编辑过的解答。"""
    from .authoring import agent, prompts

    cfg = ctx.obj["cfg"]
    prob = _find(cfg, problem_id, allow_recent=True)
    sol = bank.solution_path(cfg.solutions_dir, prob)
    if not sol.is_file():
        console.print(f"[red]还没有解答文件:{sol}[/red]")
        console.print(f"[dim]先运行 `leet start {prob.id}`[/dim]")
        sys.exit(1)
    if agent.claude_path(cfg) is None:
        console.print(f"[red]找不到 claude CLI({cfg.claude_bin})[/red]")
        sys.exit(1)

    options = _test_cache_options(no_sanitize, False, [])
    verdict = None
    if not fresh:
        cached = load_verdict(cfg, prob, sol, options)
        if cached is not None:
            verdict, age = cached
            console.print()
            console.print(
                f"[dim]复用上次 test 的结果({report.humanize_age(age)};"
                f"解答自那之后没改过)—— 想重新判题就加 --fresh[/dim]"
            )
    if verdict is None:
        with console.status("[dim]跑一遍判题以收集数据...[/dim]", spinner="dots"):
            verdict = judge(cfg, prob, sol, do_sanitize=not no_sanitize)
        save_verdict(cfg, verdict, options)

    console.print()
    console.print("[bold]判题数据[/bold]")
    console.print(f"[dim]{report.verdict_to_text(verdict, cfg)}[/dim]")

    prompt = prompts.build_review_prompt(
        problem_id=prob.id,
        title=prob.title,
        statement=prob.statement_text(),
        solution_src=sol.read_text(encoding="utf-8"),
        spec_yaml=(prob.root / "spec.yaml").read_text(encoding="utf-8"),
        baseline_src=(prob.root / subjects.baseline_filename(prob.subject))
            .read_text(encoding="utf-8"),
        verdict_summary=report.verdict_to_text(verdict, cfg),
    )

    console.print("\n[bold]讲评中[/bold]  [dim](本地 claude 分析中)[/dim]")
    console.print("[dim]" + "─" * 62 + "[/dim]")
    result = agent.run(cfg, prompt, on_event=lambda m: console.print(f"[dim]  {m}[/dim]"))

    console.print()
    if not result.ok:
        console.print(f"[red]讲评失败:[/red]{result.error}")
        sys.exit(1)
    try:
        from rich.markdown import Markdown
        console.print(Markdown(result.text))
    except Exception:
        console.print(result.text, markup=False, highlight=False)
    console.print()


@main.command()
@click.argument("problem_id", required=False)
@click.pass_context
def solution(ctx: click.Context, problem_id: Optional[str]) -> None:
    """看这道题的参考解 —— 一份能达到目标评级的实现。省略题号则取最近编辑过的解答。

    卡住的时候用。没通过这道题时会给一句提示,但不拦着你。
    """
    from rich.syntax import Syntax

    cfg = ctx.obj["cfg"]
    prob = _find(cfg, problem_id, allow_recent=True)

    fname = subjects.optimal_filename(prob.subject)
    path = (prob.root / fname) if fname else None
    if path is None or not path.is_file():
        console.print(f"[yellow]{prob.id} 还没有参考解。[/yellow]")
        console.print(f"[dim]新建的题目会自带一份(problems/<id>/{fname or 'optimal.*'});"
                      f"存量题目在陆续补。[/dim]")
        console.print(f"[dim]不过题面里的「解法思路」一节本身就是很好的指引:"
                      f"leet show {prob.id}[/dim]")
        sys.exit(1)

    # 软提示:没通过就给一句话,不阻断 —— 学习工具该提醒,但不该替人做决定
    progress = bank.Progress.load(cfg.root / bank.PROGRESS_FILENAME)
    if not progress.get(prob.id).solved:
        console.print()
        console.print("[yellow]你还没通过这道题 —— 先自己试过再看,收获会大得多。[/yellow]")

    console.print()
    console.print(f"[bold]{prob.id} 参考解[/bold]  {prob.title}")
    console.print(f"[dim]{path.name} · 评分指标 {prob.perf.metric} · "
                  f"门槛 {prob.perf.grades or '(未设)'}[/dim]")
    console.print("[dim]" + "─" * 62 + "[/dim]")

    lang = "cuda" if prob.subject == "cuda" else "python"
    console.print(Syntax(path.read_text(encoding="utf-8"), lang,
                         line_numbers=True, background_color="default"))

    console.print()
    console.print(f"[dim]它为什么快、有哪些坑,题面里讲了:leet show {prob.id}[/dim]")
    console.print(f"[dim]想自己验证一遍:先把上面的代码存进 "
                  f"{bank.solution_path(cfg.solutions_dir, prob)},再 leet test[/dim]\n")


def _dump_raw(console: Console, verdict) -> None:
    import json
    console.print("\n[dim]── 原始数据 ──[/dim]")
    if verdict.build and not verdict.build.ok:
        console.print(verdict.build.log)
    for cv in verdict.cases:
        console.print(f"[dim]{cv.case}:[/dim] {json.dumps(cv.result.raw, ensure_ascii=False)}")
    for chk in verdict.checks:
        if chk.output and (not chk.ok or chk.errors):
            console.print(f"[dim]── {chk.tool} ──[/dim]")
            console.print(chk.output[:3000])


if __name__ == "__main__":
    main()

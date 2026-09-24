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

from . import bank, report
from . import config as cfgmod
from .judge import judge
from .spec import SpecError

console = Console()


def _load_bank(cfg):
    problems, broken = bank.discover_with_errors(cfg.problems_dir)
    return problems, broken


def _find(cfg, query: str):
    problems, broken = _load_bank(cfg)
    report.render_broken(console, broken)
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
@click.argument("problem_id")
@click.pass_context
def show(ctx: click.Context, problem_id: str) -> None:
    """读题面。"""
    cfg = ctx.obj["cfg"]
    prob = _find(cfg, problem_id)
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
        template = prob.root / "template.cu"
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
@click.argument("problem_id")
@click.option("--case", "cases", multiple=True, help="只跑指定用例(可重复)")
@click.option("--no-sanitize", is_flag=True, help="跳过内存检查(更快)")
@click.option("--race", is_flag=True, help="强制跑竞态检查(很慢,仅在小用例上)")
@click.option("--gpu", type=int, default=None, help="指定物理 GPU 序号")
@click.option("-v", "--verbose", is_flag=True, help="显示原始输出")
@click.pass_context
def test(ctx: click.Context, problem_id: str, cases: List[str], no_sanitize: bool,
         race: bool, gpu: Optional[int], verbose: bool) -> None:
    """编译 + 判分。这是主命令。"""
    cfg = ctx.obj["cfg"]
    if gpu is not None:
        cfg.gpu = gpu
    prob = _find(cfg, problem_id)
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

    report.render_verdict(console, cfg, verdict, entry=entry)

    if verbose:
        _dump_raw(console, verdict)


@main.command()
@click.argument("problem_id")
@click.option("--repeat", type=int, default=200, help="计时重复次数(默认 200,比 test 更稳)")
@click.option("--case", "cases", multiple=True)
@click.option("--gpu", type=int, default=None)
@click.pass_context
def bench(ctx: click.Context, problem_id: str, repeat: int, cases: List[str],
          gpu: Optional[int]) -> None:
    """只测性能,不跑内存/竞态检查,重复更多次以获得更稳的数字。"""
    cfg = ctx.obj["cfg"]
    if gpu is not None:
        cfg.gpu = gpu
    prob = _find(cfg, problem_id)
    sol = bank.solution_path(cfg.solutions_dir, prob)
    if not sol.is_file():
        console.print(f"[red]还没有解答文件:{sol}[/red]")
        sys.exit(1)

    with console.status(f"[dim]计时中(重复 {repeat} 次)...[/dim]", spinner="dots"):
        verdict = judge(cfg, prob, sol, do_sanitize=False,
                        case_filter=list(cases) or None)
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
@click.option("--repair-rounds", type=int, default=3,
              help="独立验证失败时,回喂给出题者修复的最大轮数")
@click.option("--no-validate", is_flag=True, help="跳过独立验证(不建议)")
@click.pass_context
def new(ctx: click.Context, requirement: tuple, repair_rounds: int,
        no_validate: bool) -> None:
    """让本地 claude 自动出题 + 自验证 + 自动修复。

    例:leet new 创建两道关于共享内存 bank conflict 的题,难度递进

    出题者会自己跑 `leet validate` 迭代;完成之后框架**再独立验证一遍**
    (不采信它的自我声明),不过关就把报告回喂给它修。
    """
    from .authoring import agent, prompts
    from .authoring.validate import validate_problem

    cfg = ctx.obj["cfg"]
    req = " ".join(requirement)

    if agent.claude_path(cfg) is None:
        console.print(f"[red]找不到 claude CLI({cfg.claude_bin})[/red]")
        console.print("[dim]可用 LEETSTUDY_CLAUDE_BIN 指定路径[/dim]")
        sys.exit(1)

    before = {p.id for p in bank.discover(cfg.problems_dir)}
    prompt = prompts.build_author_prompt(
        req, cfg.root, existing_ids=sorted(before) or None
    )

    console.print()
    console.print("[bold]出题中[/bold]  [dim]本地 claude 会写文件、编译、自验证,过程实时汇报[/dim]")
    console.print("[dim]" + "─" * 62 + "[/dim]")
    result = agent.run(cfg, prompt, on_event=lambda m: console.print(f"[dim]  {m}[/dim]"))

    if not result.ok:
        console.print(f"\n[red]出题失败:[/red]{result.error}")
        sys.exit(1)
    console.print(f"[dim]  (耗时 {result.seconds:.0f}s)[/dim]")

    problems, broken = bank.discover_with_errors(cfg.problems_dir)
    created = [p for p in problems if p.id not in before]
    report.render_broken(console, broken)
    if not created:
        console.print("\n[yellow]没有检测到新题目。[/yellow]")
        if result.text:
            console.print("\n" + result.text[:1500])
        sys.exit(1)

    console.print(f"\n[green]产出了 {len(created)} 道题:[/green]"
                  + ", ".join(p.id for p in created))

    if no_validate:
        return

    # ---- 独立验证 + 修复回路 ----
    for round_no in range(repair_rounds + 1):
        console.print(f"\n[bold]独立验证[/bold](第 {round_no + 1} 轮)"
                      f"  [dim]不采信出题者的自我声明[/dim]")
        console.print("[dim]" + "─" * 62 + "[/dim]")
        per_problem = {}
        for prob in created:
            results = validate_problem(cfg, prob)
            per_problem[prob.id] = results
            failed = [r for r in results if not r[1]]
            mark = "[green]✓[/green]" if not failed else "[red]✗[/red]"
            console.print(f"  {mark} {prob.id}  "
                          f"[dim]{len(results) - len(failed)}/{len(results)} 项通过[/dim]")
            for name, ok, detail in failed:
                console.print(f"      [red]✗ {name}[/red]  [dim]{detail[:150]}[/dim]")

        failures = {pid: [r for r in res if not r[1]]
                    for pid, res in per_problem.items()}
        if not any(failures.values()):
            console.print("\n[green]全部通过 —— 新题已可用。[/green]")
            console.print("[dim]试一下:leet show <题号> / leet start <题号>[/dim]\n")
            return

        if round_no == repair_rounds:
            break

        # 回喂修复
        report_text = []
        for pid, failed in failures.items():
            if not failed:
                continue
            report_text.append(f"## {pid}")
            for name, _, detail in failed:
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
    sys.exit(1)


@main.command()
@click.argument("problem_id")
@click.option("--no-sanitize", is_flag=True, help="讲评前跳过内存检查")
@click.pass_context
def review(ctx: click.Context, problem_id: str, no_sanitize: bool) -> None:
    """让本地 claude 讲评你的 kernel(结合性能数据与消毒报告)。"""
    from .authoring import agent, prompts

    cfg = ctx.obj["cfg"]
    prob = _find(cfg, problem_id)
    sol = bank.solution_path(cfg.solutions_dir, prob)
    if not sol.is_file():
        console.print(f"[red]还没有解答文件:{sol}[/red]")
        console.print(f"[dim]先运行 `leet start {prob.id}`[/dim]")
        sys.exit(1)
    if agent.claude_path(cfg) is None:
        console.print(f"[red]找不到 claude CLI({cfg.claude_bin})[/red]")
        sys.exit(1)

    with console.status("[dim]先跑一遍判题以收集数据...[/dim]", spinner="dots"):
        verdict = judge(cfg, prob, sol, do_sanitize=not no_sanitize)

    console.print()
    console.print("[bold]判题数据[/bold]")
    console.print(f"[dim]{report.verdict_to_text(verdict, cfg)}[/dim]")

    prompt = prompts.build_review_prompt(
        problem_id=prob.id,
        title=prob.title,
        statement=prob.statement_text(),
        solution_src=sol.read_text(encoding="utf-8"),
        spec_yaml=(prob.root / "spec.yaml").read_text(encoding="utf-8"),
        baseline_src=(prob.root / "baseline.cu").read_text(encoding="utf-8"),
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

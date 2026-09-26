"""题库自验证 —— 核心是「区分度检查」。

一道题最容易出的问题不是「参考解写错了」(那会让基线对拍失败,容易发现),
而是**测试太弱**:随便写点什么都能过。这种题做起来毫无意义,而且很难靠人眼发现。

做法:注入一组**必然错误**的实现 —— 空实现、全填 0、全填 1、只写首元素 ——
要求每一个都被判为失败。任意一个竟然通过了,就说明这道题的测试形同虚设,必须打回。

这组劣化解由**科目自己提供**(`Subject.mutants`),因为它们与语言强相关:
CUDA 版是写 `<kernel><<<...>>>`,PyTorch 版是 `ctx.out.fill_(0)`。
但两边都遵循同一条原则:**从 spec 泛化生成,只需要知道有哪些 out 缓冲**,
与具体算法无关 —— 这正是这个检查能用于全自动出题的原因。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import subjects
from ..config import Config
from ..spec import Problem, missing_files
from ..subjects.base import Subject

# 一道题的基线耗时合理区间(毫秒)。
# 太快 → 用例太小,cudaEvent 的分辨率(~0.5µs)会主导测量,数字不可信;
# 太慢 → 做题体验崩坏(每次 test 都要等好几秒)。
MIN_BASELINE_MS = 0.02
MAX_BASELINE_MS = 5000.0

MIN_STATEMENT_CHARS = 600
MIN_STATEMENT_HEADINGS = 2

Check = Tuple[str, bool, str]


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #

def validate_problem(cfg: Config, problem: Problem) -> List[Check]:
    """对一道题做全套检查,返回 [(检查项, 是否通过, 说明)]。"""
    checks: List[Check] = []

    # ---- 0. 科目可用 ----
    if not subjects.is_registered(problem.subject):
        return [("科目已注册", False,
                 f"subject={problem.subject!r} 未知;已注册:{', '.join(subjects.available())}")]
    subject = subjects.get(problem.subject, cfg)
    build_root = cfg.build_dir_for(problem.id)

    # ---- 0b. entry 键齐备(科目自己声明要哪些) ----
    need = subjects.required_entry_keys(problem.subject)
    absent = [k for k in need if k not in problem.entry]
    if absent:
        return [("entry 声明齐备", False,
                 f"{problem.subject} 科目要求 entry 里有 {list(need)};"
                 f"缺少 {absent}(当前:{sorted(problem.entry)})")]
    checks.append(("entry 声明齐备", True,
                   ", ".join(f"{k}={problem.entry[k]}" for k in need)))

    # ---- 1. 文件齐备 ----
    missing = missing_files(problem, subject.required_filenames())
    if missing:
        checks.append(("文件齐备", False, "缺少:" + ", ".join(missing)))
        return checks  # 缺文件后面的都做不了
    checks.append(("文件齐备", True,
                   "spec / 题面 / " + " / ".join(subject.required_filenames()) + " 齐全"))

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

    # ---- 2. 准备基线 ----
    baseline_build = subject.prepare_variant(
        problem, problem.root / subject.baseline_filename, build_root, "baseline"
    )
    if not baseline_build.ok:
        first = baseline_build.log.splitlines()[:3]
        checks.append(("准备基线", False, " / ".join(first)))
        return checks
    checks.append(("准备基线", True, f"{baseline_build.seconds:.1f}s"))

    # ---- 3. 基线通过对拍 ----
    #   参考解本身是「标准答案」,没法自己验自己。但基线是一份独立写出来的朴素
    #   实现 —— 如果它和参考解在每个用例上都一致,两边同时写错的可能性极低。
    #   这是差分测试:用两个独立实现互相印证。
    baseline_results = {r.case: r for r in subject.run_all_cases(
        baseline_build.artifact, [c.name for c in problem.cases],
        perf=problem.perf.enabled, verify_repeat=1,
    )}
    bad_cases = [f"{n}({r.error_name or '结果不匹配'})"
                 for n, r in baseline_results.items() if not r.ok]
    baseline_ms = {n: r.perf_data["median_ms"]
                   for n, r in baseline_results.items()
                   if r.perf_data.get("median_ms")}
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

    # ---- 5. 参考解:能证明门槛物理可达吗 ----
    checks.append(_optimal_check(subject, problem, build_root, baseline_results, cfg))

    # ---- 6. 模板与劣化解必须被判失败(区分度检查)----
    #   优化题(设了 required_grade)走的是「正确性 + 评级达标」判据 ——
    #   因为它的模板就是那段正确但慢的代码,只看正确性必然通过。
    checks.append(_template_check(subject, problem, build_root, cfg, baseline_results))
    for kind in subject.mutants(problem):
        checks.append(_mutant_check(subject, problem, build_root, kind, cfg,
                                    baseline_results))

    return checks


def _optimal_check(subject, problem: Problem, build_root: Path,
                   baseline_results: Dict[str, Any], cfg: Config) -> Check:
    """参考解存在吗?它能达到目标评级吗?

    这项检查解决的是一个此前只能靠"出题者自觉"的问题:**性能门槛是否物理可达**。
    在它之前,`leet validate` 只能查"基线别太慢",无法证明"真的存在一个拿 S 的实现" ——
    于是设一个永远拿不到的门槛也没人拦得住(转置题的 S=8x 就是这么来的)。

    有了一份达到目标评级的参考解,门槛就有了存在性证明。

    参考解缺失**不算失败**(存量题目需要时间补),但会明确标注出来。
    """
    name = "参考解能证明门槛可达"
    fname = getattr(subject, "optimal_filename", "")
    if not fname:
        return (name, True, "该科目未定义参考解文件,跳过")

    path = problem.root / fname
    if not path.is_file():
        # 有评级门槛的题**必须**有参考解 —— 否则「门槛物理可达」这件事没有任何
        # 依据,一个永远拿不到的门槛也没人拦得住。没有门槛的题则无所谓。
        if problem.perf.enabled and problem.perf.grades:
            return (name, False,
                    f"未提供 {fname} —— 本题设了评级门槛 {problem.perf.grades},"
                    f"没有参考解就无法证明它真的达得到。"
                    f"请写一份能达到 S 级的实现放到 {fname}")
        return (name, True, f"未提供 {fname}(本题未设评级门槛,不影响)")

    build = subject.prepare_variant(problem, path, build_root, "chk_optimal")
    if not build.ok:
        return (name, False, "参考解无法通过准备阶段:"
                + " / ".join(build.log.splitlines()[:2]))

    results = subject.run_all_cases(
        build.artifact, [c.name for c in problem.cases],
        perf=problem.perf.enabled, verify_repeat=1,
    )

    # 复用**判分时那套**评分逻辑,不另写一份 —— 否则两处算法会漂移
    verdict = _score_build(problem, results, cfg, baseline_results)

    failed = [c.case for c in verdict.cases if not c.ok]
    if failed:
        return (name, False, f"参考解在用例 {', '.join(failed)} 上没过正确性")

    if not problem.perf.grades:
        return (name, True, f"参考解通过;本题未设评级门槛")

    grade = verdict.grade
    best = verdict.best_case
    measured = "?"
    if best is not None and best.metric_value is not None:
        if problem.perf.is_bandwidth_metric:
            # 注意:带宽型题的 metric_value 是 GB/s,评级才用「占峰值百分比」
            pct = best.bandwidth_pct
            measured = (f"{pct:.0f}% 峰值({best.metric_value:.0f} GB/s)"
                        if pct is not None else f"{best.metric_value:.0f} GB/s")
        else:
            measured = f"{best.metric_value:.2f}x"
    if grade in ("S", "A") and problem.perf.meets(grade):
        extra = ("(本题要求 " + problem.perf.required_grade + " 级)"
                 if problem.perf.required_grade else "")
        return (name, True, f"参考解达标:最好用例 {best.case if best else '?'} "
                            f"{measured},评级 {grade}{extra}")
    if not problem.perf.meets(grade):
        return (name, False,
                f"参考解只拿到 {grade} 级({measured}),而本题要求 "
                f"{problem.perf.required_grade} 级才算通过 —— 没有参考解能达标,"
                f"门槛设高了;请按实测下调 perf.grades")
    return (name, False,
            f"参考解只拿到 {grade} 级({measured})—— 说明门槛设高了,"
            f"存在不了这样的实现;请按实测下调 perf.grades,或改进 {fname}")


def _solution_ext(subject: Subject) -> str:
    return Path(subject.solution_filename).suffix or ".txt"


def _run_cases(subject, problem, artifact, cfg, baseline_results,
               cases=None) -> Tuple[bool, str]:
    """这份实现算不算「通过」?返回 (是否通过, 说明)。

    判据必须与 `judge` 完全一致 —— 否则区分度检查会在优化题上悄悄失效:

    * **普通题**:只看正确性(性能只评级不卡关)。空模板必然写不对输出,
      所以「模板必须失败」这条检查是有效的。
    * **优化题**(设了 `perf.required_grade`):模板就是那段**正确但慢**的代码,
      只看正确性的话它必然通过 —— 检查形同虚设。所以这里要按同一套判据
      连性能一起算:正确性 + 评级达标。模板 == 基线,加速比约 1.0,
      自然达不到 B 级,检查重新有效。

    cases 为 None 时跑全部;区分度检查只传最小用例 ——
    那些检查只关心「劣化解会不会失败」,能在一个小用例上失败就足以证明
    测试有区分度,没必要为它跑上千万个元素的大用例。
    但优化题例外:小用例不参与评级,必须跑**能评级**的那个用例才判得出来。
    """
    needs_grade = bool(problem.perf.required_grade) and problem.perf.enabled
    targets = list(problem.cases) if cases is None else list(cases)

    # 先不计时地跑一遍:正确性不对就直接算失败,没必要付计时的代价。
    # 四个劣化解全都在这一步被拦下 —— 它们只需要跑最小用例就露馅,
    # 所以「劣化解必须失败」这组检查几乎是免费的。
    results = subject.run_all_cases(
        artifact, [c.name for c in targets], perf=False, verify_repeat=1,
    )
    bad = [r.case for r in results if not r.ok]
    if bad or not results:
        why = f"正确性没过:{', '.join(bad)}" if bad else "没有拿到结果"
        return False, f"在 {len(results)} 个用例上被判失败({why})"

    if not needs_grade:
        passed = [r.case for r in results if r.ok]
        return True, f"竟然通过了这些用例:{', '.join(passed)}"

    # 正确性过了才计时。计时必须跑**全部**用例 —— 评级取所有用例里最好的那次,
    # 只跑最小用例会把它误判成「不达标」。
    # 优化题的模板就是那段正确但慢的代码,会一路走到这里。
    graded = subject.run_all_cases(
        artifact, [c.name for c in problem.cases], perf=True, verify_repeat=1,
    )
    verdict = _score_build(problem, graded, cfg, baseline_results)
    if verdict.passed:
        return True, (f"竟然达标了({verdict.grade} 级)—— "
                      f"这道题要求 {problem.perf.required_grade} 级")
    return False, (f"正确但只拿到 {verdict.grade or '无法评级'} 级,"
                   f"没到要求的 {problem.perf.required_grade} 级")


def _score_build(problem: Problem, results: List[Any], cfg: Config,
                 baseline_results: Optional[Dict[str, Any]] = None):
    """把一组用例结果按**判题时那套**评分逻辑算成 Verdict。

    刻意复用 `judge.score_verdict` 而不是另写一份 —— 否则两处算法会漂移,
    表现是「validate 说这道题的门槛可达,但学习者怎么都刷不到那个评级」。
    """
    from ..judge import CaseVerdict, Verdict, score_verdict
    verdict = Verdict(problem=problem)
    for r in results:
        verdict.cases.append(CaseVerdict(
            case=r.case, ok=r.ok, result=r,
            baseline=(baseline_results or {}).get(r.case),
            repeats=r.verify_total, repeats_ok=r.verify_pass,
        ))
    score_verdict(cfg, problem, verdict)
    verdict.passed = bool(verdict.cases) and not verdict.failed_cases
    if verdict.passed and problem.perf.required_grade:
        if not problem.perf.meets(verdict.grade):
            verdict.passed = False
            verdict.grade_short = True
    return verdict


def _template_check(subject: Subject, problem: Problem, build_root: Path,
                    cfg: Config, baseline_results: Dict[str, Any]) -> Check:
    name = "模板必须失败"
    build = subject.prepare_variant(
        problem, problem.root / subject.template_filename, build_root, "chk_template"
    )
    if not build.ok:
        # 模板准备失败也算「失败」,但更可能是模板本身有语法问题,标出来
        return (name, True, "模板无法通过准备阶段(当作失败处理,但建议检查模板语法)")
    ok, detail = _run_cases(subject, problem, build.artifact, cfg, baseline_results,
                            [problem.smallest_case])
    if ok:
        return (name, False, f"⚠️ 空模板竟然能通过 —— {detail}。这道题在放水!")
    return (name, True, f"模板被判失败,符合预期({detail})")


def _mutant_check(subject: Subject, problem: Problem, build_root: Path,
                  kind: str, cfg: Config, baseline_results: Dict[str, Any]) -> Check:
    name = f"劣化解必须失败:{kind}"
    source = subject.mutants(problem).get(kind)
    if source is None:
        return (name, False, "科目没有提供这个劣化解")

    src = build_root / "mutants" / f"{kind}{_solution_ext(subject)}"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(source, encoding="utf-8")

    build = subject.prepare_variant(problem, src, build_root, f"chk_{kind}")
    if not build.ok:
        return (name, False,
                "劣化解无法通过准备阶段,检查无法完成:"
                + " / ".join(build.log.splitlines()[:2]))
    ok, detail = _run_cases(subject, problem, build.artifact, cfg, baseline_results,
                            [problem.smallest_case])
    if ok:
        doc = subject.mutant_docs.get(kind, kind)
        return (name, False, f"⚠️ {doc} —— 但它通过了。{detail}")
    return (name, True, "被判失败,符合预期")

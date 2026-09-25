"""判题编排:编译 → 逐用例执行 → 稳定性 → 计时 → 消毒检查 → 诊断 → 评级。

设计取舍
--------
* **性能只评级、不卡关**:正确性(含越界检测)决定通过与否,性能给 S/A/B/C 与
  百分比。初学阶段被性能门槛卡住会打击人,而优化动力靠评级就够了。
* **memcheck 默认常开**:它能抓到「结果碰巧对了但内存写坏了」这类问题,
  而那正是最难自查的一类 bug。用 --no-sanitize 可关掉换速度。
* **racecheck 按需触发**:它极慢,只在显式要求、或检测到「同一输入重复跑结果
  不一致」这个竞态特征时才跑。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import diagnostics, subjects
from .config import Config
from .spec import Case, Problem
from .subjects.base import BuildResult, CaseResult, SanitizeResult

WARNING_GUARD_HITS = "guard_hits"

# 低于这个耗时(毫秒)的用例不参与评级。
# cudaEvent 的分辨率约 0.5µs,而 20µs 以下的测量里,计时器抖动、启动开销、
# L2 残留都会主导结果 —— 这个量级上的"加速比"没有意义。
# 小用例仍然参与正确性判定(它们本来就是用来考边界和跑 racecheck 的)。
MIN_GRADABLE_MS = 0.02


@dataclass
class CaseVerdict:
    case: str
    ok: bool
    result: CaseResult
    baseline: Optional[CaseResult] = None
    repeats: int = 1
    repeats_ok: int = 0
    speedup: Optional[float] = None
    bandwidth_pct: Optional[float] = None
    metric_value: Optional[float] = None
    grade: Optional[str] = None
    too_small: bool = False

    @property
    def perf(self) -> Dict[str, float]:
        return self.result.perf_data

    @property
    def stable(self) -> bool:
        return self.repeats_ok == self.repeats

    @property
    def params(self) -> Dict[str, float]:
        return self.result.raw.get("params") or {}

    def params_text(self) -> str:
        # 整数参数显示成整数(2 而不是 2.0),浮点参数保留小数
        bits = []
        for k, v in self.params.items():
            if isinstance(v, float) and v.is_integer():
                bits.append(f"{k}={int(v)}")
            else:
                bits.append(f"{k}={v}")
        return ", ".join(bits)


@dataclass
class Verdict:
    problem: Problem
    passed: bool = False
    build: Optional[BuildResult] = None
    baseline_build: Optional[BuildResult] = None
    cases: List[CaseVerdict] = field(default_factory=list)
    checks: List[SanitizeResult] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    fatal: Optional[str] = None
    seconds: float = 0.0
    solution_src: Optional[Path] = None

    @property
    def failed_cases(self) -> List[CaseVerdict]:
        return [c for c in self.cases if not c.ok]

    @property
    def unstable_cases(self) -> List[CaseVerdict]:
        return [c for c in self.cases if c.ok and not c.stable]

    @property
    def grade(self) -> Optional[str]:
        """整体评级取所有用例里最好的一次(及格线按最差算由 passed 负责)。"""
        grades = [c.grade for c in self.cases if c.grade]
        if not grades:
            return None
        order = {"S": 4, "A": 3, "B": 2, "C": 1}
        return max(grades, key=lambda g: order.get(g, 0))

    @property
    def best_case(self) -> Optional[CaseVerdict]:
        scored = [c for c in self.cases if c.metric_value is not None]
        if not scored:
            return None
        return max(scored, key=lambda c: c.metric_value or 0.0)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def judge(
    cfg: Config,
    problem: Problem,
    solution_src: Path,
    do_sanitize: bool = True,
    force_racecheck: bool = False,
    case_filter: Optional[List[str]] = None,
    verbose: bool = False,
) -> Verdict:
    t0 = time.time()
    subject = subjects.get(problem.subject, cfg)
    build_root = cfg.build_dir_for(problem.id)
    verdict = Verdict(problem=problem, solution_src=Path(solution_src))
    verdict.hints = diagnostics.source_hints(
        Path(solution_src).read_text(encoding="utf-8")
        if Path(solution_src).is_file() else "",
        subject=problem.subject,
    )

    # ---- 1. 准备用户解(编译 / 语法预检)----
    build = subject.prepare_variant(problem, Path(solution_src), build_root, "user")
    verdict.build = build
    if not build.ok:
        verdict.fatal = "准备失败" if subject.name != "cuda" else "编译失败"
        verdict.hints = diagnostics.describe_build_failure(
            problem, build, Path(solution_src)
        ) + verdict.hints
        verdict.seconds = time.time() - t0
        return verdict

    # ---- 2. 准备基线(失败不致命:只是没有加速比可算)----
    verdict.baseline_build = subject.prepare_variant(
        problem, problem.root / subject.baseline_filename, build_root, "baseline"
    )

    # ---- 3. 逐用例:正确性 + 稳定性 ----
    selected: List[Case] = problem.cases
    if case_filter:
        wanted = set(case_filter)
        selected = [c for c in problem.cases if c.name in wanted]
        if not selected:
            verdict.fatal = (
                f"没有匹配的用例:{case_filter};可选:{problem.case_names}"
            )
            verdict.seconds = time.time() - t0
            return verdict

    reps = max(1, problem.verify.repeat)
    for case in selected:
        result = subject.run_case(build.artifact, case.name, perf=problem.perf.enabled)
        ok_count = 1 if result.ok else 0
        # 再跑几遍确认稳定 —— 抓竞态 / 未初始化内存这类「有时对有时错」的问题
        for _ in range(reps - 1):
            again = subject.run_case(build.artifact, case.name, perf=False)
            if again.ok:
                ok_count += 1

        cv = CaseVerdict(
            case=case.name, ok=result.ok, result=result,
            repeats=reps, repeats_ok=ok_count,
        )
        # 只在用例跑对时才测基线:正确性没过的话,性能数字没有意义,
        # 而基线计时(50 次 + 每次清 L2)不便宜,不该白花。
        if result.ok and verdict.baseline_build.ok and verdict.baseline_build.artifact:
            cv.baseline = subject.run_case(
                verdict.baseline_build.artifact, case.name, perf=problem.perf.enabled
            )
        verdict.cases.append(cv)

    _score(cfg, problem, verdict)

    # ---- 4. 通过判定:正确性 + 稳定性,性能不卡关 ----
    verdict.passed = (
        bool(verdict.cases)
        and not verdict.failed_cases
        and not verdict.unstable_cases
    )

    # ---- 5. 消毒检查 ----
    if do_sanitize and subject.supports_sanitize:
        smallest = problem.smallest_case.name
        if problem.sanitize.memcheck:
            verdict.checks.append(
                subject.sanitize(build.artifact, smallest, "memcheck")
            )
        # racecheck 极慢:只在显式要求、或看到竞态特征(重复结果不一致)时跑
        want_race = force_racecheck or bool(verdict.unstable_cases)
        if want_race and problem.sanitize.racecheck_case:
            verdict.checks.append(
                subject.sanitize(build.artifact, problem.sanitize.racecheck_case, "racecheck")
            )

    # ---- 6. 诊断 ----
    verdict.hints = _collect_hints(problem, verdict) + verdict.hints
    verdict.seconds = time.time() - t0
    return verdict


# --------------------------------------------------------------------------- #
# 评分
# --------------------------------------------------------------------------- #

def _score(cfg: Config, problem: Problem, verdict: Verdict) -> None:
    peak = cfg.resolve_peak_bandwidth()
    for cv in verdict.cases:
        if not cv.ok:
            continue
        perf = cv.perf
        if not perf:
            continue

        # 加速比总是算,作为次要信息展示(访存题上也想知道相对基线如何)
        if cv.baseline and cv.baseline.ok:
            bt = cv.baseline.perf_data.get("median_ms")
            ut = perf.get("median_ms")
            if bt and ut and ut > 0:
                cv.speedup = bt / ut

        # 太小的用例不评级:这个量级上测的是开销与抖动,不是 kernel 性能
        ms = perf.get("median_ms")
        if ms is not None and ms < MIN_GRADABLE_MS:
            cv.too_small = True
            continue

        if problem.perf.is_bandwidth_metric:
            gb = perf.get("gb_per_s")
            if gb and gb > 0:
                cv.metric_value = gb
                if peak:
                    cv.bandwidth_pct = gb / peak * 100.0
                    cv.grade = problem.perf.grade_of(cv.bandwidth_pct)
        else:
            if cv.speedup is not None:
                cv.metric_value = cv.speedup
                cv.grade = problem.perf.grade_of(cv.speedup)


# --------------------------------------------------------------------------- #
# 诊断汇总
# --------------------------------------------------------------------------- #

def _collect_hints(problem: Problem, verdict: Verdict) -> List[str]:
    hints: List[str] = []

    # 越界写(哨兵区被踩)—— 优先级最高,单独说
    for cv in verdict.cases:
        hints.extend(diagnostics.describe_guards(problem, cv.result))

    # CUDA 运行时错误
    seen_errors = set()
    for cv in verdict.cases:
        r = cv.result
        if r.error_name and r.error_name not in seen_errors:
            seen_errors.add(r.error_name)
            hints.extend(
                diagnostics.describe_cuda_error(r.stage, r.error_name, r.error_str)
            )
        if r.timed_out:
            hints.append(f"用例 {cv.case} 执行超时。{r.stderr.strip()}")

    # 数值层面的失败模式
    for cv in verdict.cases:
        if not cv.result.ok and not cv.result.error_name:
            hints.extend(
                diagnostics.describe_outputs(problem, cv.result, case_name=cv.case)
            )

    # 不稳定 —— 竞态特征
    for cv in verdict.unstable_cases:
        hints.append(
            f"用例 {cv.case} 重复 {cv.repeats} 次中只对了 {cv.repeats_ok} 次 —— "
            f"结果不稳定,几乎可以断定是竞态或读了未初始化的内存。"
            f"检查共享内存/全局内存的读写是否需要 __syncthreads()。"
        )

    # 消毒检查
    for chk in verdict.checks:
        hints.extend(diagnostics.describe_sanitizer(chk))

    # 结果对了但很慢
    for cv in verdict.cases:
        if cv.ok and cv.speedup is not None and cv.speedup < 0.7:
            hints.append(
                f"用例 {cv.case}:你的实现比基线还慢 {1/cv.speedup:.2f}x。"
                f"基线只是最朴素的写法,慢于它通常意味着访存模式有问题"
                f"(比如跨步访问破坏了合并,或者线程利用率太低)。"
            )

    # 去重保序
    seen = set()
    uniq: List[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq

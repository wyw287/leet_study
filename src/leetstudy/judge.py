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

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# 判题结果缓存
#
# 用途:让 `leet review` 在「刚 test 过、解答没再改」时直接复用上次的数据,
# 省掉一次判题(本机实测 11~21 秒,其中大半是 CUDA 上下文初始化)。
#
# 边界:**只有 review 用,test 绝不用。** review 把判题数据当"给模型的上下文",
# 略有陈旧无伤大雅;而 test 的数据是用来评级的 —— 复用就等于把成绩变成
# "上次测的时候机器忙不忙"的产物。同一个理由也让本框架拒绝了缓存基线耗时。
# --------------------------------------------------------------------------- #

CACHE_FILENAME = "last_verdict.json"
_CACHE_VERSION = 1        # 改动 Verdict 结构或判题语义时递增,旧缓存自然失效


def _cache_path(cfg: Config, problem: Problem) -> Path:
    return cfg.build_dir_for(problem.id) / CACHE_FILENAME


def verdict_signature(problem: Problem, solution_src: Path,
                      options: Dict[str, Any]) -> str:
    """缓存的判据:解答内容 + 所有能影响判题结果的输入 + 选项。

    用**内容哈希**而不是文件 mtime:`touch`、`git checkout`、编辑器无改动保存
    都会动 mtime,但那些情况下缓存依然有效;反过来内容变了就一定要重测。

    options 必须包含所有会改变判题结果的开关(是否跑消毒检查、用例过滤、
    计时重复次数等)—— 选项不同,数据就不可互换。
    """
    h = hashlib.sha256()
    h.update(f"leetstudy-verdict-v{_CACHE_VERSION}\n".encode())
    h.update(json.dumps(options, sort_keys=True, default=str).encode())
    h.update(b"\n")

    watched = [Path(solution_src), problem.root / "spec.yaml"]
    try:
        watched.append(problem.root / subjects.reference_filename(problem.subject))
        watched.append(problem.root / subjects.baseline_filename(problem.subject))
    except KeyError:
        pass
    for path in watched:
        h.update(f"--{path.name}--\n".encode())
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<missing>")
        h.update(b"\n")
    return h.hexdigest()


def _case_to_dict(cv: CaseVerdict) -> Dict[str, Any]:
    return {
        "case": cv.case,
        "ok": cv.ok,
        "raw": cv.result.raw,
        "perf": cv.result.perf,
        "repeats": cv.repeats,
        "repeats_ok": cv.repeats_ok,
        "baseline_raw": cv.baseline.raw if cv.baseline is not None else None,
        "baseline_ok": cv.baseline.ok if cv.baseline is not None else None,
    }


def _dict_to_case(d: Dict[str, Any]) -> CaseVerdict:
    result = CaseResult(case=str(d.get("case") or "?"), ok=bool(d.get("ok")),
                        raw=d.get("raw") or {}, perf=bool(d.get("perf")))
    baseline = None
    if d.get("baseline_raw") is not None:
        baseline = CaseResult(case=str(d.get("case") or "?"),
                              ok=bool(d.get("baseline_ok")),
                              raw=d["baseline_raw"], perf=bool(d.get("perf")))
    return CaseVerdict(
        case=result.case, ok=result.ok, result=result, baseline=baseline,
        repeats=int(d.get("repeats") or 1), repeats_ok=int(d.get("repeats_ok") or 0),
    )


def save_verdict(cfg: Config, verdict: Verdict, options: Dict[str, Any]) -> None:
    """把判题结果落盘。写失败不是错误 —— 只是下次 review 还得重测一遍。"""
    payload = {
        "version": _CACHE_VERSION,
        "created_at": time.time(),
        "signature": verdict_signature(verdict.problem, verdict.solution_src, options)
        if verdict.solution_src else None,
        "options": options,
        "passed": verdict.passed,
        "fatal": verdict.fatal,
        "seconds": verdict.seconds,
        "hints": verdict.hints,
        "build": ({"ok": verdict.build.ok, "seconds": verdict.build.seconds,
                   "log": verdict.build.log[:4000]}
                  if verdict.build else None),
        "baseline_ok": bool(verdict.baseline_build and verdict.baseline_build.ok),
        "cases": [_case_to_dict(cv) for cv in verdict.cases],
        "checks": [{
            "tool": c.tool, "ok": c.ok, "errors": c.errors, "summary": c.summary,
            "skipped": c.skipped, "skip_reason": c.skip_reason, "seconds": c.seconds,
            # 只有失败时才留输出 —— 成功时的 sanitizer 输出又长又没用
            "output": "" if c.ok else (c.output or "")[:8000],
        } for c in verdict.checks],
    }
    try:
        path = _cache_path(cfg, verdict.problem)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError):
        pass


def load_verdict(cfg: Config, problem: Problem, solution_src: Path,
                 options: Dict[str, Any]) -> Optional[Tuple[Verdict, float]]:
    """读回上次的判题结果。返回 (Verdict, 距今秒数);判据不符或没有则 None。

    读回来的 Verdict 会重新过一遍 `score_verdict` —— 评分逻辑只留一处,不在缓存里存
    派生出来的评级,避免两处算法漂移。
    """
    path = _cache_path(cfg, problem)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
        return None

    want = verdict_signature(problem, solution_src, options)
    if payload.get("signature") != want:
        return None

    verdict = Verdict(problem=problem, solution_src=Path(solution_src))
    verdict.passed = bool(payload.get("passed"))
    verdict.fatal = payload.get("fatal")
    verdict.seconds = float(payload.get("seconds") or 0.0)
    verdict.hints = [str(h) for h in (payload.get("hints") or [])]

    b = payload.get("build")
    if b:
        verdict.build = BuildResult(ok=bool(b.get("ok")),
                                    seconds=float(b.get("seconds") or 0.0),
                                    stderr=str(b.get("log") or ""))
    if payload.get("baseline_ok"):
        verdict.baseline_build = BuildResult(ok=True)

    verdict.cases = [_dict_to_case(d) for d in (payload.get("cases") or [])]
    verdict.checks = [SanitizeResult(
        tool=str(c.get("tool") or "?"), ok=bool(c.get("ok")),
        errors=int(c.get("errors") or 0), summary=str(c.get("summary") or ""),
        skipped=bool(c.get("skipped")), skip_reason=str(c.get("skip_reason") or ""),
        seconds=float(c.get("seconds") or 0.0), output=str(c.get("output") or ""),
    ) for c in (payload.get("checks") or [])]

    # 评级/加速比/带宽占比都从这里重算,不存进缓存
    score_verdict(cfg, problem, verdict)
    age = max(0.0, time.time() - float(payload.get("created_at") or 0))
    return verdict, age


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
    perf_repeat: Optional[int] = None,
) -> Verdict:
    """跑一次完整判题。

    perf_repeat 覆盖题目 spec 里的计时重复次数(leet bench 用它跑更多次以更稳)。
    用户解与基线会用**同一个**次数,否则加速比不可比。
    """
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
    #   一次进程跑完所有用例(以及进程内的稳定性重复)。科目可以覆盖这个行为;
    #   CUDA 科目必须覆盖 —— 本机 CUDA 上下文初始化要 4.4 秒,用例一多就吃不消。
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
    # 用户解与基线共用同一组计时参数,加速比才可比
    perf_args = ["--repeat", str(perf_repeat)] if perf_repeat else None
    names = [c.name for c in selected]

    results = subject.run_all_cases(
        build.artifact, names,
        perf=problem.perf.enabled, verify_repeat=reps, extra_args=perf_args,
    )
    by_case = {r.case: r for r in results}

    # 基线:只对**跑对**的用例计时(正确性没过时性能数字没有意义,
    # 而基线计时不便宜);同样一次进程跑完。
    baseline_by_case: Dict[str, CaseResult] = {}
    ok_names = [r.case for r in results if r.ok]
    if ok_names and verdict.baseline_build.ok and verdict.baseline_build.artifact:
        for br in subject.run_all_cases(
            verdict.baseline_build.artifact, ok_names,
            perf=problem.perf.enabled, verify_repeat=1, extra_args=perf_args,
        ):
            baseline_by_case[br.case] = br

    for case in selected:
        result = by_case.get(case.name)
        if result is None:
            result = CaseResult(case=case.name, ok=False,
                                stderr="没有拿到这个用例的结果")
        cv = CaseVerdict(
            case=case.name, ok=result.ok, result=result,
            repeats=result.verify_total, repeats_ok=result.verify_pass,
            baseline=baseline_by_case.get(case.name),
        )
        verdict.cases.append(cv)

    score_verdict(cfg, problem, verdict)

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

def score_verdict(cfg: Config, problem: Problem, verdict: Verdict) -> None:
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

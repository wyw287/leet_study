"""C++ 科目 —— CPU 上的「优化题」。

它和另外两个科目的关系
----------------------
* `cuda` / `pytorch` 的题问的是「**写出**一个正确的实现」,性能只评级、不卡关。
* 这个科目的题问的是「**把给定代码改快**」—— 基线本身就是正确代码。所以它多了一个
  `perf.required_grade`:正确性 + 评级达标才算过(见 `spec.Perf.required_grade`)。
  没有这个字段的话,学习者把原代码原样交回来就算通过。

编译选项是**框架钉死的**
------------------------
`-O3 -march=native` 不带 `-ffast-math`。这一条不是随手写的:

* 不带 fast-math,编译器**不允许重排浮点运算**,所以「归约用 4 个累加器」这类
  结构性优化它做不了 —— 那正是要考的东西。
* 一旦放开学到者改选项,他可以只加 `-ffast-math` 就白拿这类题(实测:同一份源码,
  点积从 0.975 ns/元素 变成 0.243,**没改一行代码**)。
* 同理也不能用 `-fno-tree-vectorize` 去人为制造优化空间 —— 那等于教「SIMD 比标量快」
  这种在真实世界里不成立的东西(实测:开了自动向量化之后,手写 AVX2 在 6 个算子里
  只有 2 个还赢,且那 2 个各有别的解释)。

所以选项锁死,并且题面里要说明白。
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import codegen_cpu
from ..config import Config
from ..spec import Problem
from .base import Artifact, BuildResult, CaseResult, SanitizeResult, Subject

#: 编译选项。**改动这里等于改动所有已出题目的难度** —— 见模块 docstring。
#: 刻意不含 -ffast-math:放开它,编译器就能重排浮点归约,一批结构性题会失去意义。
_CFLAGS = ("-O3", "-march=native", "-std=c++17")


_MUTANT_DOC = {
    "noop": "空实现(什么都不做)—— 必须被抓到,否则漏写输出也能过",
    "zeros": "把输出全填 0 —— 若这也能过,说明期望值恰好是 0,测试退化",
    "ones": "把输出全填 1 —— 若这也能过,说明期望值是常数,测试退化",
    "first_only": "只写输出的第 0 个元素 —— 若这也能过,说明只检查了首元素",
}



def render_mutant(problem: Problem, kind: str) -> str:
    """按 spec 泛化生成一份必然错误的 C++ 实现(只需定义入口函数)。"""
    fn = problem.entry.get("function", "solve")
    lines: List[str] = [
        "// 自动生成的劣化解 —— 用于区分度检查,不该出现在题目目录里。\n",
        f"// 类型:{kind} —— {_MUTANT_DOC[kind]}\n",
        '#include "ctx.h"\n\n',
    ]

    if kind == "noop":
        lines.append(
            f"void {fn}(Ctx& ctx) {{\n"
            "    (void)ctx;   // 什么都不做\n"
            "}\n"
        )
        return "".join(lines)

    body: List[str] = []
    for b in problem.outputs:
        count = b.count_expr(prefix="ctx.")
        if kind == "first_only":
            body.append(f"    ctx.{b.name}[0] = ({b.ctype})1;   // 故意只写首元素\n")
        else:
            value = "0" if kind == "zeros" else "1"
            body.append(
                f"    for (int64_t i = 0; i < {count}; ++i)"
                f" ctx.{b.name}[i] = ({b.ctype}){value};\n"
            )

    lines.append(f"void {fn}(Ctx& ctx) {{\n")
    lines.extend(body)
    lines.append("}\n")
    return "".join(lines)


class CppSubject(Subject):
    """Subject 协议的 C++ 实现(CPU,单线程计时)。"""

    name = "cpp"
    template_filename = "template.cpp"
    reference_filename = "reference.cpp"
    baseline_filename = "baseline.cpp"
    solution_filename = "solution.cpp"
    optimal_filename = "optimal.cpp"
    required_entry_keys = ("function",)
    mutant_docs = _MUTANT_DOC
    build_label = "编译"
    build_note = "g++ -O3 -march=native"
    #: ASan/UBSan 走「另编一个带消毒选项的二进制」,与 CUDA 的
    #: compute-sanitizer(直接跑现有二进制)不同,所以先留关闭。
    supports_sanitize = False

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # 准备(编译)
    # ------------------------------------------------------------------ #

    def prepare_variant(
        self,
        problem: Problem,
        impl_src: Path,
        out_dir: Path,
        label: str = "variant",
    ) -> BuildResult:
        """把一份实现编译成可执行文件。

        与 CUDA 侧同样带编译缓存,判据是内容哈希而不是 mtime。
        """
        build_root = Path(out_dir).resolve()
        impl_src = Path(impl_src).resolve()
        variant_dir = build_root / label
        exe = variant_dir / "harness"

        reference = (problem.root / self.reference_filename).resolve()
        if not reference.is_file():
            return BuildResult(ok=False, stderr=f"缺少参考解文件 {reference}")

        flags = tuple(_CFLAGS)
        stamp = _build_stamp(problem, impl_src, reference, flags)
        stamp_file = variant_dir / "build.stamp"
        if exe.is_file() and stamp_file.is_file():
            try:
                if stamp_file.read_text(encoding="utf-8").strip() == stamp:
                    return BuildResult(
                        ok=True, artifact=Artifact(kind="exe", path=exe),
                        seconds=0.0, cmd=[], stdout="(编译缓存命中)",
                    )
            except OSError:
                pass

        codegen_cpu.write_ctx_h(problem, build_root)
        files = codegen_cpu.write_variant(problem, variant_dir, impl_src)

        cmd: List[str] = [
            self.cfg.cxx, *flags,
            "-I", str(build_root),
            str(files["harness"].resolve()),
            str(reference),
            "-o", str(exe),
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=self.cfg.compile_timeout, cwd=str(variant_dir),
            )
        except subprocess.TimeoutExpired:
            return BuildResult(ok=False, cmd=cmd, seconds=time.time() - t0,
                               stderr=f"编译超时(>{self.cfg.compile_timeout}s)")
        except FileNotFoundError:
            return BuildResult(
                ok=False, cmd=cmd,
                stderr=f"找不到 C++ 编译器:{self.cfg.cxx}\n"
                       "安装 g++(或 clang++),或设置 LEETSTUDY_CXX=/path/to/g++",
            )
        except OSError as exc:
            return BuildResult(ok=False, cmd=cmd, stderr=f"编译失败:{exc}")

        ok = proc.returncode == 0 and exe.is_file()
        if ok:
            try:
                stamp_file.write_text(stamp, encoding="utf-8")
            except OSError:
                pass
        return BuildResult(
            ok=ok, artifact=Artifact(kind="exe", path=exe) if ok else None,
            cmd=cmd, stdout=proc.stdout, stderr=proc.stderr,
            seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def run_case(
        self,
        artifact: Artifact,
        case: str,
        perf: bool = False,
        timeout: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> CaseResult:
        exe = artifact.path
        cmd = [str(exe), case] + (["--perf"] if perf else []) + list(extra_args or [])
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=timeout or self.cfg.run_timeout, cwd=str(exe.parent),
            )
        except subprocess.TimeoutExpired:
            return CaseResult(
                case=case, ok=False, timed_out=True, perf=perf,
                seconds=time.time() - t0,
                stderr=f"执行超时(>{timeout or self.cfg.run_timeout}s)。"
                       "常见原因:死循环,或把工作量放大了几个数量级。",
            )
        except OSError as exc:
            return CaseResult(case=case, ok=False, perf=perf, stderr=f"无法执行:{exc}")

        raw = _parse_marker(proc.stdout)
        return CaseResult(
            case=case, ok=bool(raw.get("ok")), raw=raw, stdout=proc.stdout,
            stderr=proc.stderr, seconds=time.time() - t0,
            returncode=proc.returncode, perf=perf,
        )

    def run_all_cases(
        self,
        artifact: Artifact,
        cases: List[str],
        perf: bool = False,
        verify_repeat: int = 1,
        timeout: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> List[CaseResult]:
        """一次进程跑完所有用例 —— 与 CUDA 侧同一套约定,理由见基类 docstring。

        CPU 上进程启动本身便宜(0.02 秒),但一个进程跑完所有用例能让**计时环境
        可比**(同一份 harness 状态、同一块清缓存缓冲),而且 CPU 题动辄几十毫秒,
        逐用例起进程的累积开销也不小。
        """
        exe = artifact.path
        selector = cases[0] if len(cases) == 1 else "all"
        cmd = [str(exe), "--case", selector,
               "--verify-repeat", str(max(1, verify_repeat))]
        if perf:
            cmd.append("--perf")
        cmd += list(extra_args or [])

        limit = timeout or max(self.cfg.run_timeout, 60 * len(cases))
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=limit, cwd=str(exe.parent),
            )
        except subprocess.TimeoutExpired:
            why = (f"执行超时(>{limit}s)。常见原因:死循环,"
                   f"或把工作量放大了几个数量级。")
            return [CaseResult(case=c, ok=False, timed_out=True, perf=perf,
                               seconds=time.time() - t0, stderr=why) for c in cases]
        except OSError as exc:
            return [CaseResult(case=c, ok=False, perf=perf,
                               stderr=f"无法执行:{exc}") for c in cases]

        elapsed = time.time() - t0
        by_case = {r.get("case"): r for r in _parse_markers(proc.stdout) if r.get("case")}
        results: List[CaseResult] = []
        for name in cases:
            raw = by_case.get(name)
            if raw is None:
                # 没产出结果 —— 进程多半在半路挂了(段错误等)。
                # 把已完成的用例结果保留住,不要因为一个用例崩了就把全部丢掉。
                results.append(CaseResult(
                    case=name, ok=False, raw={}, perf=perf, stdout=proc.stdout,
                    stderr=proc.stderr, seconds=elapsed, returncode=proc.returncode,
                ))
            else:
                results.append(CaseResult(
                    case=name, ok=bool(raw.get("ok")), raw=raw, perf=perf,
                    stdout=proc.stdout, stderr=proc.stderr, seconds=elapsed,
                    returncode=proc.returncode,
                ))
        return results

    # ------------------------------------------------------------------ #
    # 内存检查
    #
    # 暂未接入。CUDA 侧是拿现成的二进制去跑 compute-sanitizer;而 ASan 需要
    # **重新编译**(而且带 -fsanitize 的产物慢好几倍,绝不能拿它计时)。
    # 要接的话就在这里另编一份到 <label>_asan/ 再跑 —— 见 prepare_variant 的说明。
    # 哨兵区不受影响:它一直在工作,越界写照样能被抓到。
    # ------------------------------------------------------------------ #

    def sanitize(
        self,
        artifact: Artifact,
        case: str,
        tool: str,
        timeout: Optional[int] = None,
    ) -> SanitizeResult:
        return SanitizeResult(
            tool=tool, ok=True, skipped=True,
            skip_reason="cpp 科目暂未接入 ASan/UBSan(哨兵区仍然有效)",
        )

    # ------------------------------------------------------------------ #
    # 劣化解
    # ------------------------------------------------------------------ #

    def mutants(self, problem: Problem) -> Dict[str, str]:
        return {kind: render_mutant(problem, kind) for kind in _MUTANT_DOC}


# --------------------------------------------------------------------------- #
# 解析辅助
# --------------------------------------------------------------------------- #

def _build_stamp(problem: Problem, impl_src: Path, reference: Path,
                 flags: tuple) -> str:
    """编译缓存的判据:源码 + 参考解 + spec + 编译选项的内容哈希。

    刻意用**内容**而不是 mtime:mtime 会被 touch / git checkout 之类与代码无关的
    操作改动,而那些情况下重新编译纯属浪费;反过来内容变了一定要重编。
    """
    h = hashlib.sha256()
    h.update(f"flags={' '.join(flags)}\n".encode())
    for path in (impl_src, reference, problem.root / "spec.yaml"):
        h.update(f"--{path.name}--\n".encode())
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<missing>")
        h.update(b"\n")
    return h.hexdigest()


def _parse_markers(stdout: str) -> List[Dict[str, Any]]:
    """解析输出里**所有**带标记的 JSON 行(一个进程会为每个用例打印一行)。"""
    from ..codegen import JSON_MARKER
    found: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        idx = line.find(JSON_MARKER)
        if idx < 0:
            continue
        try:
            parsed = json.loads(line[idx + len(JSON_MARKER):].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            found.append(parsed)
    return found


def _parse_marker(stdout: str) -> Dict[str, Any]:
    found = _parse_markers(stdout)
    return found[0] if found else {}

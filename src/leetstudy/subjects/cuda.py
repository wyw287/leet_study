"""CUDA 科目的判题实现:nvcc 编译、执行、compute-sanitizer 消毒检查。"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import codegen
from ..config import Config
from ..spec import Problem
from .base import Artifact, BuildResult, CaseResult, SanitizeResult, Subject

# -O3:性能题必须开优化,否则测的是未优化代码
# -lineinfo:让 compute-sanitizer / cuda-gdb 能报出行号(不影响优化)
_NVCC_FLAGS = ("-O3", "-lineinfo", "-std=c++17")

_MEMCHECK_SUMMARY = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+error", re.I)
_RACECHECK_SUMMARY = re.compile(r"RACECHECK SUMMARY:\s*(\d+)\s+hazard", re.I)

# 劣化解:必须每一个都被判失败,否则说明题目的测试太弱。
# 它们都是**从 spec 泛化生成**的 —— 只需要知道有哪些 out 缓冲,与具体算法无关,
# 所以对任何题目都适用,不需要手写特例。
_MUTANT_DOC = {
    "noop": "空实现(什么都不做)—— 必须被抓到,否则漏写输出也能过",
    "zeros": "把输出全填 0 —— 若这也能过,说明期望值恰好是 0,测试退化",
    "ones": "把输出全填 1 —— 若这也能过,说明期望值是常数,测试退化",
    "first_only": "只写每块输出的第 0 个元素 —— 若这也能过,说明只检查了首元素",
}


def render_mutant(problem: Problem, kind: str) -> str:
    """按 spec 泛化生成一份必然错误的 CUDA 实现(只需定义 launcher)。"""
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
                f"    {kname}<<<1, 1>>>(ctx.{b.name}, {count}, ({ct})1);\n"
            )
        else:
            value = "0" if kind == "zeros" else "1"
            fill_ops.append(
                f"    long long {count} = {b.count_expr(prefix='ctx.')};\n"
                f"    {kname}<<<((unsigned)(({count} + 255) / 256)), 256>>>"
                f"(ctx.{b.name}, {count}, ({ct}){value});\n"
            )

    lines.append(f"void {problem.launcher}(LaunchCtx& ctx) {{\n")
    lines.extend(fill_ops)
    lines.append("}\n")
    return "".join(lines)


class CudaSubject(Subject):
    """Subject 协议的 CUDA 实现。"""

    name = "cuda"
    template_filename = "template.cu"
    reference_filename = "reference.cpp"
    baseline_filename = "baseline.cu"
    solution_filename = "solution.cu"
    supports_sanitize = True
    required_entry_keys = ("kernel", "launcher")
    mutant_docs = _MUTANT_DOC
    build_label = "编译"
    build_note = "nvcc -O3 -lineinfo"

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

        out_dir 是 build/<problem>/,每个变体在 out_dir/<label>/ 下独立编译。
        """
        # 全部转绝对路径:nvcc 的工作目录是 variant_dir,相对路径会解析错
        build_root = Path(out_dir).resolve()
        impl_src = Path(impl_src).resolve()

        codegen.write_ctx_h(problem, build_root)
        variant_dir = build_root / label
        files = codegen.write_variant(problem, variant_dir, impl_src)
        exe = variant_dir / "harness"

        reference = (problem.root / self.reference_filename).resolve()
        if not reference.is_file():
            return BuildResult(
                ok=False,
                stderr=f"缺少参考解文件 {reference}",
            )

        cmd: List[str] = [
            self.cfg.nvcc,
            *_NVCC_FLAGS,
            f"-arch={self.cfg.arch}",
            "-I", str(build_root),
            str(files["harness"].resolve()),
            str(reference),
            "-o", str(exe),
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.cfg.compile_timeout,
                cwd=str(variant_dir),
            )
        except subprocess.TimeoutExpired:
            return BuildResult(
                ok=False, cmd=cmd, seconds=time.time() - t0,
                stderr=f"编译超时(>{self.cfg.compile_timeout}s)",
            )
        except FileNotFoundError:
            return BuildResult(
                ok=False, cmd=cmd,
                stderr=f"找不到 nvcc:{self.cfg.nvcc}\n"
                       "安装 CUDA Toolkit,或设置 LEETSTUDY_NVCC=/path/to/nvcc",
            )
        except OSError as exc:
            return BuildResult(ok=False, cmd=cmd, stderr=f"编译失败:{exc}")

        ok = proc.returncode == 0 and exe.is_file()
        return BuildResult(
            ok=ok,
            artifact=Artifact(kind="exe", path=exe) if ok else None,
            cmd=cmd,
            stdout=proc.stdout,
            stderr=proc.stderr,
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
        env = self.cfg.env_for_gpu(None)
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout or self.cfg.run_timeout,
                env=env,
                cwd=str(exe.parent),
            )
        except subprocess.TimeoutExpired:
            return CaseResult(
                case=case, ok=False, timed_out=True, perf=perf,
                seconds=time.time() - t0,
                stderr=f"执行超时(>{timeout or self.cfg.run_timeout}s)。"
                       "常见原因:kernel 死循环,或启动配置把工作量放大了几个数量级。",
            )
        except OSError as exc:
            return CaseResult(case=case, ok=False, perf=perf, stderr=f"无法执行:{exc}")

        raw = _parse_marker(proc.stdout)
        return CaseResult(
            case=case,
            ok=bool(raw.get("ok")),
            raw=raw,
            stdout=proc.stdout,
            stderr=proc.stderr,
            seconds=time.time() - t0,
            returncode=proc.returncode,
            perf=perf,
        )

    # ------------------------------------------------------------------ #
    # 内存 / 竞态检查
    # ------------------------------------------------------------------ #

    def sanitize(
        self,
        artifact: Artifact,
        case: str,
        tool: str,
        timeout: Optional[int] = None,
    ) -> SanitizeResult:
        san = shutil.which(self.cfg.sanitizer) or (
            self.cfg.sanitizer if Path(self.cfg.sanitizer).is_file() else None
        )
        if san is None:
            return SanitizeResult(
                tool=tool, ok=True, skipped=True,
                skip_reason=f"未找到 {self.cfg.sanitizer},跳过检查",
            )

        cmd = [san, "--tool", tool, "--error-exitcode", "1",
               str(artifact.path), case]
        env = self.cfg.env_for_gpu(None)
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout or self.cfg.sanitizer_timeout,
                env=env,
                cwd=str(artifact.path.parent),
            )
        except subprocess.TimeoutExpired:
            return SanitizeResult(
                tool=tool, ok=False, skipped=True,
                skip_reason=f"检查超时(>{timeout or self.cfg.sanitizer_timeout}s),"
                            "该用例可能太大;考虑给 racecheck 指定更小的 case",
                seconds=time.time() - t0,
            )
        except OSError as exc:
            return SanitizeResult(
                tool=tool, ok=True, skipped=True, skip_reason=f"无法运行:{exc}",
            )

        out = proc.stdout or ""
        errors, summary = _parse_sanitizer(tool, out)
        return SanitizeResult(
            tool=tool,
            ok=errors == 0,
            errors=errors,
            summary=summary,
            output=out,
            seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------ #
    # 劣化解
    # ------------------------------------------------------------------ #

    def mutants(self, problem: Problem) -> Dict[str, str]:
        return {kind: render_mutant(problem, kind) for kind in _MUTANT_DOC}


# --------------------------------------------------------------------------- #
# 解析辅助
# --------------------------------------------------------------------------- #

def _parse_marker(stdout: str) -> Dict[str, Any]:
    """从输出里找出 harness 打印的那行 JSON。

    用户 kernel 里的 printf 可能混在输出中,所以只认带标记的那一行。
    """
    for line in stdout.splitlines():
        idx = line.find(codegen.JSON_MARKER)
        if idx < 0:
            continue
        payload = line[idx + len(codegen.JSON_MARKER):].strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _parse_sanitizer(tool: str, output: str) -> tuple:
    """返回 (错误数, 摘要行)。"""
    pattern = _MEMCHECK_SUMMARY if tool == "memcheck" else _RACECHECK_SUMMARY
    matches = pattern.findall(output)
    if matches:
        errors = int(matches[-1])
        unit = "errors" if tool == "memcheck" else "hazards"
        return errors, f"{errors} {unit}"

    # 没有 summary 行:可能是程序本身崩了,或 sanitizer 用法有误
    if "========= " not in output:
        return 0, "无检查输出"
    return 0, "未识别到摘要"

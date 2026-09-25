"""PyTorch 科目的判题实现。

与 CUDA 侧最大的不同:**没有代码生成**。

CUDA 之所以需要 587 行的 codegen,是因为 C++ 没法自省 —— 要按 spec 拼出
结构体定义和 main 函数。Python 可以自省,所以这里的 runner 是一个普通的、
可 import 的模块(`pytorch_runner.py`),"准备"阶段几乎什么都不用做。

用户接口刻意做成 `forward(ctx)` 而不是惯用的 `forward(*tensors) -> tensor`:
框架预分配输出张量,并把它做成带哨兵区的缓冲区的一个视图,这样
「往前/往后越界了多少个元素」这种精度的诊断才能保住。代价是与
「返回新张量」的写法不同 —— 对学习工具来说,诊断精度更值钱。

这个科目同时覆盖两类题:
  ① 纯 PyTorch 算子题:只用 torch 算子拼,无自定义 kernel
  ③ PyTorch + 自定义 CUDA 算子:在解答里用 torch.utils.cpp_extension.load_inline
     写 CUDA 并绑进来。此时把 spec 的 `sanitize.memcheck` 打开,
     compute-sanitizer 就能跑到这些自定义 kernel 上(memcheck/racecheck 都支持)。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..spec import Problem
from .base import Artifact, BuildResult, CaseResult, SanitizeResult, Subject

_MEMCHECK_SUMMARY = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+error", re.I)
_RACECHECK_SUMMARY = re.compile(r"RACECHECK SUMMARY:\s*(\d+)\s+hazard", re.I)

# 劣化解:每个都必须被题目的测试判失败。与 CUDA 侧同构,只是换成 Python 写法。
_MUTANT_DOC = {
    "noop": "空实现(什么都不做)—— 必须被抓到,否则漏写输出也能过",
    "zeros": "把输出全填 0 —— 若这也能过,说明期望值恰好是 0,测试退化",
    "ones": "把输出全填 1 —— 若这也能过,说明期望值是常数,测试退化",
    "first_only": "只写每块输出的第 0 个元素 —— 若这也能过,说明只检查了首元素",
}

# PyTorch 题的首次运行可能要 JIT 编译扩展(load_inline),比纯 Python 慢得多
_MIN_RUN_TIMEOUT = 300


def render_mutant(problem: Problem, kind: str) -> str:
    """按 spec 泛化生成一份必然错误的 PyTorch 实现。"""
    fn = problem.function or "forward"
    out_names = [b.name for b in problem.outputs]

    lines = [
        '"""自动生成的劣化解 —— 用于区分度检查,不该出现在题目目录里。"""\n',
        f"# 类型:{kind} —— {_MUTANT_DOC[kind]}\n\n\n",
    ]

    if kind == "noop":
        lines.append(f"def {fn}(ctx):\n    return None   # 什么都不做\n")
        return "".join(lines)

    lines.append(f"def {fn}(ctx):\n")
    if kind == "first_only":
        for name in out_names:
            lines.append(
                f"    # 只写首元素\n"
                f"    flat = ctx.{name}.reshape(-1)\n"
                f"    if flat.numel() > 0:\n"
                f"        flat[0] = 1\n"
            )
    else:
        value = "0" if kind == "zeros" else "1"
        for name in out_names:
            lines.append(f"    ctx.{name}.fill_({value})\n")
    return "".join(lines)


class PyTorchSubject(Subject):
    """Subject 协议的 PyTorch 实现。"""

    name = "pytorch"
    template_filename = "template.py"
    reference_filename = "reference.py"
    baseline_filename = "baseline.py"
    solution_filename = "solution.py"
    supports_sanitize = True
    required_entry_keys = ("function",)
    mutant_docs = _MUTANT_DOC
    build_label = "准备"
    build_note = "Python 语法检查"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # 子进程环境
    # ------------------------------------------------------------------ #

    def _arch_list(self) -> str:
        """把 nvcc 风格的 sm_89 转成 torch 的 TORCH_CUDA_ARCH_LIST 风格 8.9。"""
        digits = self.cfg.arch.replace("sm_", "")
        return f"{digits[:-1]}.{digits[-1]}" if len(digits) >= 2 else digits

    def _env(self) -> Dict[str, str]:
        env = self.cfg.env_for_gpu(None)

        # 源码树兜底(可编辑安装下本就可见)
        src_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src_root + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")

        # 把项目 venv 的 bin 放进 PATH。这条很关键:torch.utils.cpp_extension
        # 是**通过 PATH 上有没有 ninja 可执行文件**来判断能否用 load_inline 的,
        # 光在 Python 里 import ninja 不算。少了这一步,自定义 CUDA 算子题会报
        # "Ninja is required to load C++ extensions"。
        venv_bin = self.cfg.root / ".venv" / "bin"
        if venv_bin.is_dir():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        # 直接告诉 torch 目标架构,省得它自己探测(探测会打印一长串警告,
        # 还可能编出多于一个架构而白白拉长首次编译时间)
        env.setdefault("TORCH_CUDA_ARCH_LIST", self._arch_list())
        return env

    # ------------------------------------------------------------------ #
    # 准备(解释型:只做语法预检)
    # ------------------------------------------------------------------ #

    def prepare_variant(
        self,
        problem: Problem,
        impl_src: Path,
        out_dir: Path,
        label: str = "variant",
    ) -> BuildResult:
        impl_src = Path(impl_src).resolve()
        t0 = time.time()
        if not impl_src.is_file():
            return BuildResult(ok=False, stderr=f"源文件不存在:{impl_src}")

        # 语法预检:把语法错误在"准备阶段"就报出来,而不是等运行时抛一个栈很深的
        # SyntaxError。这对应 CUDA 侧的 nvcc 编译。
        # 用 ast.parse 而不是 py_compile —— 后者会往用户目录写 .pyc。
        try:
            import ast
            ast.parse(impl_src.read_text(encoding="utf-8"), filename=str(impl_src))
        except SyntaxError as exc:
            # 刻意对齐 nvcc 的报错格式 `<文件>(<行>): error: <信息>`,
            # 这样 diagnostics.describe_build_failure 不用为科目写分支。
            where = f"({exc.lineno})" if exc.lineno else ""
            return BuildResult(
                ok=False, seconds=time.time() - t0,
                stderr=f"{impl_src}{where}: error: {exc.msg}",
            )
        except (OSError, UnicodeDecodeError) as exc:
            return BuildResult(ok=False, stderr=f"读取失败:{exc}",
                               seconds=time.time() - t0)

        return BuildResult(
            ok=True,
            seconds=time.time() - t0,
            # 解释型语言没有"编译产物",产物就是源码本身。
            # 把题目目录带在 extra 里,让 Artifact 自描述 —— run_case 就不再需要 problem。
            # 必须转绝对路径:子进程的工作目录是解答文件所在目录,相对路径会解析错。
            artifact=Artifact(kind="module", path=impl_src,
                              extra={"spec_dir": str(problem.root.resolve())}),
        )

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def _runner_cmd(self, artifact: Artifact, case: str, perf: bool) -> List[str]:
        return [
            sys.executable, "-m", "leetstudy.subjects.pytorch_runner",
            "--spec", str(artifact.extra.get("spec_dir", "")),
            "--impl", str(artifact.path),
            "--case", case,
        ] + (["--perf"] if perf else [])

    def run_case(
        self,
        artifact: Artifact,
        case: str,
        perf: bool = False,
        timeout: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> CaseResult:
        cmd = self._runner_cmd(artifact, case, perf) + list(extra_args or [])
        env = self._env()

        limit = timeout or max(self.cfg.run_timeout, _MIN_RUN_TIMEOUT)
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=limit,
                env=env,
                cwd=str(artifact.path.parent),
            )
        except subprocess.TimeoutExpired:
            return CaseResult(
                case=case, ok=False, timed_out=True, perf=perf,
                seconds=time.time() - t0,
                stderr=f"执行超时(>{limit}s)。首次运行若在 JIT 编译自定义扩展会较慢;"
                       f"也可能解答里有死循环或过大的计算量。",
            )
        except OSError as exc:
            return CaseResult(case=case, ok=False, perf=perf, stderr=f"无法执行:{exc}")

        raw = self._parse_marker(proc.stdout)
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
    # 消毒检查(只对自定义 CUDA 算子题有意义)
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
            return SanitizeResult(tool=tool, ok=True, skipped=True,
                                  skip_reason=f"未找到 {self.cfg.sanitizer},跳过检查")

        # compute-sanitizer 可以包住整个 python 进程,所以自定义 CUDA 算子里
        # 的越界/竞态一样能被抓到(带行号)。
        cmd = [san, "--tool", tool, "--error-exitcode", "1"] + \
              self._runner_cmd(artifact, case, perf=False)
        env = self._env()

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
                skip_reason=f"检查超时(>{timeout or self.cfg.sanitizer_timeout}s)",
                seconds=time.time() - t0,
            )
        except OSError as exc:
            return SanitizeResult(tool=tool, ok=True, skipped=True,
                                  skip_reason=f"无法运行:{exc}")

        out = proc.stdout or ""
        pattern = _MEMCHECK_SUMMARY if tool == "memcheck" else _RACECHECK_SUMMARY
        matches = pattern.findall(out)
        if matches:
            errors = int(matches[-1])
            unit = "errors" if tool == "memcheck" else "hazards"
            summary = f"{errors} {unit}"
        elif "========= " not in out:
            errors, summary = 0, "无检查输出"
        else:
            errors, summary = 0, "未识别到摘要"
        return SanitizeResult(tool=tool, ok=errors == 0, errors=errors,
                              summary=summary, output=out,
                              seconds=time.time() - t0)

    # ------------------------------------------------------------------ #
    # 劣化解
    # ------------------------------------------------------------------ #

    def mutants(self, problem: Problem) -> Dict[str, str]:
        return {kind: render_mutant(problem, kind) for kind in _MUTANT_DOC}

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_marker(stdout: str) -> Dict[str, Any]:
        from .base import JSON_MARKER
        for line in stdout.splitlines():
            idx = line.find(JSON_MARKER)
            if idx < 0:
                continue
            import json
            try:
                parsed = json.loads(line[idx + len(JSON_MARKER):].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

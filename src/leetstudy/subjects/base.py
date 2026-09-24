"""判题科目接口 —— 未来接 PyTorch / Triton 的插入点。

v1 只有 CUDA 一个实现。judge.py 只依赖这里的语义,不依赖 CUDA 细节;
将来加新科目只需新增一个满足 Subject 协议的模块,不动 CLI 与 judge。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class BuildResult:
    ok: bool
    exe: Optional[Path] = None
    cmd: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0

    @property
    def log(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


@dataclass
class CaseResult:
    """一次用例执行的结果。raw 是 harness 打印的 JSON 原文。"""

    case: str
    ok: bool
    raw: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    returncode: int = 0
    timed_out: bool = False
    perf: bool = False

    @property
    def stage(self) -> Optional[str]:
        """出错阶段(launch / 执行中(异步错误) / H2D(x) / ...),正常时为 None。"""
        return self.raw.get("stage")

    @property
    def error_name(self) -> Optional[str]:
        return self.raw.get("error_name")

    @property
    def error_str(self) -> Optional[str]:
        return self.raw.get("error_str")

    @property
    def outputs(self) -> List[Dict[str, Any]]:
        return self.raw.get("outputs") or []

    @property
    def guards(self) -> Dict[str, Dict[str, int]]:
        return self.raw.get("guards") or {}

    @property
    def perf_data(self) -> Dict[str, float]:
        return self.raw.get("perf") or {}


@dataclass
class SanitizeResult:
    tool: str                      # memcheck / racecheck
    ok: bool
    errors: int = 0
    summary: str = ""
    output: str = ""
    skipped: bool = False
    skip_reason: str = ""
    seconds: float = 0.0


@runtime_checkable
class Subject(Protocol):
    """一个编程科目的判题能力。"""

    name: str

    def compile_variant(
        self,
        problem: Any,
        impl_src: Path,
        out_dir: Path,
        label: str = "",
    ) -> BuildResult:
        """把一份实现(template / 用户解 / baseline / 参考解)编译成可执行文件。"""
        ...

    def run_case(
        self,
        exe: Path,
        case: str,
        perf: bool = False,
        timeout: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> CaseResult:
        """运行一个用例,返回 harness 的 JSON 结构化结果。"""
        ...

    def sanitize(self, exe: Path, case: str, tool: str, timeout: Optional[int] = None) -> SanitizeResult:
        """跑一次内存/竞态检查。"""
        ...

    def scaffold(self, problem: Any) -> str:
        """返回给用户填空的初始代码。"""
        ...

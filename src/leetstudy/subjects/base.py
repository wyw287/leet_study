"""判题科目接口 —— 加新科目(PyTorch / Triton / …)的插入点。

设计要点
--------
**为什么是 Artifact 而不是 exe**:最早的协议把产物写成 `exe: Path`,这隐含了一个
假设 —— 每个科目都产出原生可执行文件。CUDA 是这样,但 PyTorch 不是:它的"产物"
是一个可以被 import 的 Python 模块,没有 exe。`Artifact` 把这件事抽象掉,
judge / report / cli / bank / validate 都不需要知道后面是可执行文件还是模块。

**为什么用 ABC 而不是 Protocol**:科目之间大量逻辑是共通的(读模板、默认不支持
消毒检查、默认没有劣化解),用基类能把这些默认实现写一次。Protocol 只能声明。

**科目自己声明文件名**:`baseline.cu` / `baseline.py` 这类差异收在科目里,
上层不再出现任何硬编码的文件名。

约定:所有科目的 `CaseResult.raw` 必须产出**同一套 JSON 结构**
(`outputs[].bad / max_abs_err / first_bad_idx / …`、`perf.median_ms / gb_per_s`、
`guards`),因为报告层与诊断层直接消费它。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: runner 输出结果时使用的行前缀。所有科目共用同一套输出格式,
#: 这样报告层与诊断层不需要区分科目。前缀是为了让用户代码里的 print 不干扰解析。
JSON_MARKER = "@@LEET_JSON@@"


@dataclass
class Artifact:
    """一份实现在"准备完成"之后的不透明句柄。

    CUDA 下 kind="exe"(编译出的可执行文件),PyTorch 下 kind="module"
    (可以直接 import 的 .py)。extra 放科目私有的附加信息。
    """

    kind: str
    path: Path
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.kind}:{self.path.name}"


@dataclass
class BuildResult:
    ok: bool
    artifact: Optional[Artifact] = None
    cmd: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0

    @property
    def log(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()

    # 兼容旧字段名:不少地方只想拿到"产物路径"
    @property
    def exe(self) -> Optional[Path]:
        return self.artifact.path if self.artifact else None


@dataclass
class CaseResult:
    """一次用例执行的结果。raw 是 runner 打印的 JSON(结构见模块 docstring)。"""

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

    @property
    def verify_pass(self) -> int:
        """稳定性重复里通过了几次。"""
        return int(self.raw.get("verify_pass", 1 if self.ok else 0))

    @property
    def verify_total(self) -> int:
        return int(self.raw.get("verify_total", 1))

    @property
    def stable(self) -> bool:
        """同一份输入重复跑是否每次都通过 —— 不通过就是竞态/未初始化内存的特征。"""
        return self.verify_pass == self.verify_total


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


class Subject(abc.ABC):
    """一个编程科目的判题能力。

    子类至少要实现 `prepare_variant` 与 `run_case`,并按需覆盖文件名、
    劣化解生成、消毒检查。
    """

    #: 科目标识,与 spec.yaml 里的 `subject` 字段对应
    name: str = ""

    #: 一个科目要用的文件名。上层不硬编码任何名字。
    template_filename: str = ""
    reference_filename: str = ""
    baseline_filename: str = ""
    solution_filename: str = ""

    #: 是否支持 compute-sanitizer 那类消毒检查
    supports_sanitize: bool = False

    #: 报告里怎么描述"准备"这一步(各科目说法不同)
    build_label: str = "准备"
    build_note: str = ""

    #: 劣化解的说明文字,用于 `leet validate` 的检查项命名
    mutant_docs: Dict[str, str] = {}

    # ---- 必须实现 -------------------------------------------------------- #

    @abc.abstractmethod
    def prepare_variant(
        self,
        problem: Any,
        impl_src: Path,
        out_dir: Path,
        label: str = "variant",
    ) -> BuildResult:
        """把一份实现(template / 用户解 / baseline / 参考解)准备成可运行形态。

        CUDA 下是编译;解释型语言下可能什么都不做,只把源码路径包成 Artifact。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_case(
        self,
        artifact: Artifact,
        case: str,
        perf: bool = False,
        timeout: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> CaseResult:
        """运行一个用例,返回结构化结果。"""
        raise NotImplementedError

    def run_all_cases(
        self,
        artifact: Artifact,
        cases: List[str],
        perf: bool = False,
        verify_repeat: int = 1,
        timeout: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> List[CaseResult]:
        """批量跑用例,返回与 cases 等长的结果列表(顺序一致)。

        默认实现是逐个跑 —— 那对解释型科目够用(Python 启动便宜)。CUDA 科目
        会覆盖它:本机实测 CUDA 上下文初始化要 4.4 秒,每个用例起一个进程
        根本无法接受,所以它一次进程跑完所有用例。

        实现者要保证:返回结果的 `verify_pass/verify_total` 反映**进程内**的
        稳定性重复次数,而不是靠外层反复调用。
        """
        out: List[CaseResult] = []
        for case in cases:
            first: Optional[CaseResult] = None
            passed = 0
            for _ in range(max(1, verify_repeat)):
                r = self.run_case(artifact, case, perf=perf and first is None,
                                  timeout=timeout, extra_args=extra_args)
                if first is None:
                    first = r
                if r.ok:
                    passed += 1
            assert first is not None
            # 用总的重复次数改写稳定性计数:默认实现下由外层循环提供
            first.raw = {**first.raw,
                         "verify_pass": passed,
                         "verify_total": max(1, verify_repeat),
                         "ok": passed == max(1, verify_repeat)}
            first.ok = bool(first.raw["ok"])
            out.append(first)
        return out

    # ---- 可选实现(有默认值) ---------------------------------------------- #

    def sanitize(
        self,
        artifact: Artifact,
        case: str,
        tool: str,
        timeout: Optional[int] = None,
    ) -> SanitizeResult:
        return SanitizeResult(
            tool=tool, ok=True, skipped=True,
            skip_reason=f"{self.name} 科目不做 {tool} 检查",
        )

    def mutants(self, problem: Any) -> Dict[str, str]:
        """返回 {劣化解名: 源码},用于区分度检查。

        每个劣化解都必须是**必然错误**的;如果有任何一个竟然通过了题目的测试,
        说明这道题的测试太弱。返回空字典表示该科目不做区分度检查。
        """
        return {}

    def scaffold(self, problem: Any) -> str:
        """给用户填空的初始代码(默认读模板文件)。"""
        path = problem.root / self.template_filename
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return f"// 缺少模板文件 {self.template_filename}\n"

    def required_filenames(self) -> List[str]:
        return [self.template_filename, self.reference_filename, self.baseline_filename]

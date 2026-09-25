"""题目规格(R) —— spec.yaml 的数据模型与加载校验。

一份 spec.yaml 描述一道题的全部机器可读信息:
  * 接口契约(buffers / params)—— 驱动 codegen 生成 LaunchCtx / RefCtx 的字段
  * 测试用例(cases)
  * 容差(verify)、性能评级门槛(perf)、消毒检查配置(sanitize)

spec.yaml 既由人手写,也由 LLM 自动生成,因此校验必须严格且报错要具体 ——
生成器写错了要能一眼看出错在哪一行。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


class SpecError(Exception):
    """spec.yaml 格式或语义错误。消息面向作者,尽量指明字段。"""


# dtype → (C 类型, 字节数)
DTYPES: Dict[str, Tuple[str, int]] = {
    "f32": ("float", 4),
    "f64": ("double", 8),
    "i32": ("int", 4),
    "i64": ("long long", 8),
    "u32": ("unsigned int", 4),
    "u8": ("unsigned char", 1),
}

ROLES = ("in", "out", "scratch")
# in 缓冲的填充策略:uniform=[-1,1);positive=(0,1];randint=整数;zero=全 0
FILLS = ("uniform", "positive", "randint", "zero")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT = re.compile(r"^\d+$")
# 题目 id 同时用作目录名与命令参数,是 slug 而非 C 标识符,允许以数字开头
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Buffer:
    """一块由 harness 分配的内存。

    shape 是表达式列表,每项要么是 params 里的参数名,要么是整数字面量。
    shape 为空表示标量(1 个元素),用于归约这类只输出单个值的题。
    """

    name: str
    dtype: str
    shape: List[str]
    role: str
    fill: str = "uniform"

    @property
    def ctype(self) -> str:
        return DTYPES[self.dtype][0]

    @property
    def ptr_type(self) -> str:
        """Ctx 结构体里该字段的类型(in 缓冲加 const)。"""
        return f"const {self.ctype}*" if self.role == "in" else f"{self.ctype}*"

    @property
    def is_scalar(self) -> bool:
        return not self.shape

    def count_expr(self, prefix: str = "") -> str:
        """C 表达式:该缓冲的元素个数。

        prefix 用于限定参数的来源:harness 里参数是局部变量(prefix=""),
        而 launcher 里参数挂在 ctx 上(prefix="ctx.")。
        """
        if not self.shape:
            return "1"
        return " * ".join(f"((int64_t)({prefix}{d}))" for d in self.shape)

    def describe(self) -> str:
        return f"{self.name}: {self.dtype}[{', '.join(self.shape) or '标量'}] ({self.role})"

    def resolve_shape(self, params: Dict[str, float]) -> List[int]:
        """把形状表达式求出具体维度。形状项要么是参数名,要么是整数字面量。

        解释型科目(PyTorch)直接用这个建张量;C 侧走 count_expr 生成表达式。
        """
        dims: List[int] = []
        for dim in self.shape:
            if dim in params:
                dims.append(int(params[dim]))
            else:
                dims.append(int(dim))
        return dims

    def count(self, params: Dict[str, float]) -> int:
        """元素总数。标量(shape 为空)返回 1。"""
        total = 1
        for d in self.resolve_shape(params):
            total *= d
        return total


@dataclass(frozen=True)
class Param:
    """标量参数,如 n / m / k。会作为字段同时出现在 LaunchCtx 和 RefCtx 里。"""

    name: str
    dtype: str

    @property
    def ctype(self) -> str:
        return DTYPES[self.dtype][0]


@dataclass(frozen=True)
class Case:
    """一个测试用例:一组参数取值。

    取值统一存成 float —— 既能表示整数参数(缓冲大小),也能表示浮点参数
    (如 alpha、eps、temperature)。整数参数在校验阶段已确认取整数值。
    """

    name: str
    params: Dict[str, float]


@dataclass(frozen=True)
class Verify:
    atol: float = 1e-5
    rtol: float = 1e-4
    # 每个 case 重复执行的次数。>1 可捕捉竞态、未初始化内存、越界写导致的
    # 「这次对下次错」类问题。
    repeat: int = 1


@dataclass(frozen=True)
class Perf:
    enabled: bool = True
    repeat: int = 50
    warmup: int = 10
    # 评分指标:
    #   speedup   —— 相对基线的加速比。适合计算瓶颈题(如矩阵乘:朴素实现远非最优)。
    #   bandwidth —— 有效带宽占峰值的百分比。适合访存瓶颈题 ——
    #                这类题的朴素 CUDA 实现本身就是最优算法(都跑满带宽),
    #                用加速比评分毫无区分度,必须看带宽利用率。
    metric: str = "speedup"
    # 门槛,含义随 metric 变化:
    #   speedup   → 倍数,如 {"B": 1.5, "A": 3.0, "S": 6.0}
    #   bandwidth → 占峰值百分比,如 {"B": 45, "A": 65, "S": 82}
    grades: Dict[str, float] = field(default_factory=dict)
    # 每次计时迭代前清空 L2。必须默认开启:4090 有 72MB L2,而一道题的
    # 数据常常只有几 MB —— 不清 L2 的话整个题目驻留在缓存里,测出来的
    # 是缓存带宽而非真实显存带宽,数字会严重失真。
    flush_l2: bool = True
    # 只影响报告措辞:memory = 访存瓶颈,compute = 计算瓶颈。
    bound: str = "memory"

    def grade_of(self, value: float) -> str:
        """把指标值映射到评级。从高到低取第一个达标者。"""
        for letter, threshold in sorted(
            self.grades.items(), key=lambda kv: kv[1], reverse=True
        ):
            if value >= threshold:
                return letter
        return "C"

    @property
    def is_bandwidth_metric(self) -> bool:
        return self.metric == "bandwidth"


@dataclass(frozen=True)
class Sanitize:
    memcheck: bool = True
    # racecheck 极慢,只跑一个最小 case。None 表示不做竞态检查。
    racecheck_case: Optional[str] = None


@dataclass(frozen=True)
class Problem:
    id: str
    title: str
    difficulty: int
    tags: List[str]
    concepts: List[str]
    statement: str
    buffers: List[Buffer]
    params: List[Param]
    cases: List[Case]
    #: 判题科目(cuda / pytorch / …)。决定由哪个 Subject 来编译与运行。
    subject: str
    #: 入口声明。各科目自己决定要哪些键:
    #:   cuda    → {kernel: __global__ 函数名, launcher: host 端启动函数名}
    #:   pytorch → {function: 被调用的 Python 函数名}
    entry: Dict[str, str]
    verify: Verify
    perf: Perf
    sanitize: Sanitize
    root: Path

    # -- 入口(向后兼容的便捷属性) ------------------------------------------ #

    @property
    def kernel(self) -> str:
        """CUDA 科目用:__global__ 函数名。"""
        return self.entry.get("kernel", "")

    @property
    def launcher(self) -> str:
        """CUDA 科目用:host 端启动函数名。"""
        return self.entry.get("launcher", "")

    @property
    def function(self) -> str:
        """解释型科目用:被调用的 Python 函数名。"""
        return self.entry.get("function", "")

    # -- 便捷视图 ---------------------------------------------------------- #

    @property
    def inputs(self) -> List[Buffer]:
        return [b for b in self.buffers if b.role == "in"]

    @property
    def outputs(self) -> List[Buffer]:
        return [b for b in self.buffers if b.role == "out"]

    @property
    def scratches(self) -> List[Buffer]:
        return [b for b in self.buffers if b.role == "scratch"]

    @property
    def case_names(self) -> List[str]:
        return [c.name for c in self.cases]

    def case(self, name: str) -> Case:
        for c in self.cases:
            if c.name == name:
                return c
        raise SpecError(f"题目 {self.id} 没有名为 {name!r} 的用例(可选:{self.case_names})")

    @property
    def smallest_case(self) -> Case:
        """参数规模最小的用例 —— racecheck 用。"""
        def size(c: Case) -> int:
            return max(c.params.values()) if c.params else 0
        return min(self.cases, key=size)

    def statement_text(self) -> str:
        path = self.root / self.statement
        if not path.is_file():
            return f"(题面文件缺失: {path})"
        return path.read_text(encoding="utf-8")

    @property
    def difficulty_dots(self) -> str:
        return "●" * self.difficulty + "○" * (5 - self.difficulty)


# --------------------------------------------------------------------------- #
# 加载与校验
# --------------------------------------------------------------------------- #

def _need(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise SpecError(f"{where} 缺少必填字段 {key!r}")
    return d[key]


def _as_bool(v: Any, where: str) -> bool:
    if isinstance(v, bool):
        return v
    raise SpecError(f"{where} 应为 true/false,得到 {v!r}")


def _parse_buffer(raw: Dict[str, Any], where: str, param_names: List[str]) -> Buffer:
    name = str(_need(raw, "name", where))
    if not _IDENT.match(name):
        raise SpecError(f"{where}.name={name!r} 不是合法的 C 标识符")

    dtype = str(_need(raw, "dtype", where))
    if dtype not in DTYPES:
        raise SpecError(
            f"{where}.dtype={dtype!r} 不支持,可用:{', '.join(sorted(DTYPES))}"
        )

    role = str(_need(raw, "role", where))
    if role not in ROLES:
        raise SpecError(f"{where}.role={role!r} 不支持,可用:{', '.join(ROLES)}")

    # shape 可省略 → 标量
    raw_shape = raw.get("shape") or []
    if not isinstance(raw_shape, list):
        raise SpecError(f"{where}.shape 应为列表,如 [n] 或 [m, k];标量用 []")
    shape: List[str] = []
    for i, dim in enumerate(raw_shape):
        dim_s = str(dim)
        if _INT.match(dim_s):
            shape.append(dim_s)
        elif dim_s in param_names:
            shape.append(dim_s)
        else:
            raise SpecError(
                f"{where}.shape[{i}]={dim_s!r} 既不是整数也不是已声明的参数名"
                f"(已声明参数:{param_names or '无'})"
            )

    fill = str(raw.get("fill") or "uniform")
    if fill not in FILLS:
        raise SpecError(f"{where}.fill={fill!r} 不支持,可用:{', '.join(FILLS)}")
    if fill != "uniform" and role != "in":
        raise SpecError(f"{where}.fill 只对 role=in 的缓冲有意义")

    return Buffer(name=name, dtype=dtype, shape=shape, role=role, fill=fill)


def _parse_param(raw: Dict[str, Any], where: str) -> Param:
    name = str(_need(raw, "name", where))
    if not _IDENT.match(name):
        raise SpecError(f"{where}.name={name!r} 不是合法的 C 标识符")
    dtype = str(_need(raw, "dtype", where))
    if dtype not in DTYPES:
        raise SpecError(f"{where}.dtype={dtype!r} 不支持,可用:{', '.join(sorted(DTYPES))}")
    return Param(name=name, dtype=dtype)


def parse_problem(raw: Dict[str, Any], root: Path) -> Problem:
    """由已解析的 yaml 字典构造 Problem,并做完整性校验。"""
    if not isinstance(raw, dict):
        raise SpecError(f"{root}/spec.yaml 顶层应为映射(dict)")

    pid = str(_need(raw, "id", "spec"))
    if not _SLUG.match(pid):
        raise SpecError(f"id={pid!r} 不是合法标识;只允许字母数字下划线连字符")
    if root.name != pid:
        raise SpecError(f"目录名 {root.name!r} 与 spec 中的 id={pid!r} 不一致")

    difficulty = int(raw.get("difficulty", 1))
    if not 1 <= difficulty <= 5:
        raise SpecError(f"difficulty={difficulty} 应在 1..5")

    params = [
        _parse_param(p, f"params[{i}]")
        for i, p in enumerate(raw.get("params") or [])
    ]
    param_names = [p.name for p in params]
    if len(set(param_names)) != len(param_names):
        raise SpecError(f"参数名重复:{param_names}")

    buffers = [
        _parse_buffer(b, f"buffers[{i}]", param_names)
        for i, b in enumerate(raw.get("buffers") or [])
    ]
    if not buffers:
        raise SpecError("spec 至少要声明一个 buffer")
    buf_names = [b.name for b in buffers]
    if len(set(buf_names)) != len(buf_names):
        raise SpecError(f"buffer 名重复:{buf_names}")
    overlap = set(buf_names) & set(param_names)
    if overlap:
        raise SpecError(f"buffer 名与参数名冲突:{sorted(overlap)}")
    if not any(b.role == "out" for b in buffers):
        raise SpecError("spec 至少要有一个 role=out 的 buffer,否则无从判分")

    # 科目:决定由谁来编译与运行。默认 cuda,让既有题目无需改动。
    subject = str(raw.get("subject") or "cuda").strip().lower()
    if not _SLUG.match(subject):
        raise SpecError(f"subject={subject!r} 不是合法标识")

    # 入口声明按科目自解释:这里只做通用校验(必须是 标识符 → 标识符 的映射),
    # 具体要哪些键由对应 Subject 检查(见 subjects/*.py 的 required_entry_keys)。
    raw_entry = _need(raw, "entry", "spec")
    if not isinstance(raw_entry, dict) or not raw_entry:
        raise SpecError("entry 应为非空映射,如 {kernel: foo, launcher: foo_launch}")
    entry: Dict[str, str] = {}
    for key, val in raw_entry.items():
        if not _IDENT.match(str(key)):
            raise SpecError(f"entry 的键 {key!r} 不是合法标识符")
        if not _IDENT.match(str(val)):
            raise SpecError(f"entry.{key}={val!r} 不是合法标识符")
        entry[str(key)] = str(val)

    raw_cases = raw.get("cases") or []
    if not raw_cases:
        raise SpecError("spec 至少要有一个 case")
    cases: List[Case] = []
    for i, rc in enumerate(raw_cases):
        where = f"cases[{i}]"
        cname = str(_need(rc, "name", where))
        cparams = dict(rc.get("params") or {})
        missing = [p for p in param_names if p not in cparams]
        if missing:
            raise SpecError(f"{where}({cname}) 缺少参数取值:{missing}")
        extra = [k for k in cparams if k not in param_names]
        if extra:
            raise SpecError(f"{where}({cname}) 含未声明的参数:{extra}")
        bad = [k for k, v in cparams.items() if not isinstance(v, (int, float)) or v <= 0]
        if bad:
            raise SpecError(f"{where}({cname}) 的参数取值应为正数,出错:{bad}")
        # 整数类型的参数必须是整数(它会被用来算缓冲大小)
        int_dtypes = {p.name for p in params if p.dtype not in ("f32", "f64")}
        frac = [k for k in int_dtypes
                if float(cparams[k]) != int(cparams[k])]
        if frac:
            raise SpecError(
                f"{where}({cname}) 的参数 {frac} 声明为整数类型,不能给小数取值"
            )
        cases.append(Case(name=cname, params={k: float(v) for k, v in cparams.items()}))
    if len({c.name for c in cases}) != len(cases):
        raise SpecError("case 名重复")

    vraw = raw.get("verify") or {}
    verify = Verify(
        atol=float(vraw.get("atol", 1e-5)),
        rtol=float(vraw.get("rtol", 1e-4)),
        repeat=int(vraw.get("repeat", 1)),
    )
    if verify.repeat < 1:
        raise SpecError("verify.repeat 至少为 1")
    # 容差宽到形同虚设是个常见错误,顺手拦一下(f32 相对误差到 1e-1 就没意义了)
    if verify.atol > 1.0 or verify.rtol > 1e-1:
        raise SpecError(
            f"verify 容差过宽(atol={verify.atol}, rtol={verify.rtol}),"
            "会让错误实现轻松通过;请收紧到 f32 合理范围(如 atol 1e-5 / rtol 1e-4)"
        )

    praw = raw.get("perf") or {}
    bound = str(praw.get("bound") or "memory")
    if bound not in ("memory", "compute"):
        raise SpecError(f"perf.bound={bound!r} 不支持,可用:memory / compute")
    # 未显式指定指标时,按瓶颈类型给默认值 —— 访存型看带宽才有区分度
    default_metric = "bandwidth" if bound == "memory" else "speedup"
    metric = str(praw.get("metric") or default_metric)
    if metric not in ("speedup", "bandwidth"):
        raise SpecError(f"perf.metric={metric!r} 不支持,可用:speedup / bandwidth")
    perf = Perf(
        enabled=bool(praw.get("enabled", True)),
        repeat=int(praw.get("repeat", 50)),
        warmup=int(praw.get("warmup", 10)),
        metric=metric,
        grades={str(k): float(v) for k, v in (praw.get("grades") or {}).items()},
        flush_l2=bool(praw.get("flush_l2", True)),
        bound=bound,
    )
    if perf.enabled and (perf.repeat < 1 or perf.warmup < 0):
        raise SpecError("perf.repeat 至少为 1,perf.warmup 不能为负")

    sraw = raw.get("sanitize") or {}
    race_case = sraw.get("racecheck_case")
    sanitize = Sanitize(
        memcheck=_as_bool(sraw.get("memcheck", True), "sanitize.memcheck"),
        racecheck_case=str(race_case) if race_case else None,
    )
    if sanitize.racecheck_case and sanitize.racecheck_case not in [c.name for c in cases]:
        raise SpecError(
            f"sanitize.racecheck_case={sanitize.racecheck_case!r} 不是已定义的 case"
            f"(可选:{[c.name for c in cases]})"
        )

    statement = str(raw.get("statement") or "problem.md")

    return Problem(
        id=pid,
        title=str(raw.get("title") or pid),
        difficulty=difficulty,
        tags=[str(t) for t in (raw.get("tags") or [])],
        concepts=[str(c) for c in (raw.get("concepts") or [])],
        statement=statement,
        buffers=buffers,
        params=params,
        cases=cases,
        subject=subject,
        entry=entry,
        verify=verify,
        perf=perf,
        sanitize=sanitize,
        root=root,
    )


def load_problem(problem_dir) -> Problem:
    """从题目目录加载 spec.yaml。接受 Path 或字符串。"""
    problem_dir = Path(problem_dir)
    spec_path = problem_dir / "spec.yaml"
    if not spec_path.is_file():
        raise SpecError(f"找不到 {spec_path}")
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"{spec_path} YAML 解析失败:\n{exc}") from exc
    return parse_problem(raw, problem_dir)


def required_files(problem: Problem, impl_filenames: Sequence[str] = ()) -> List[str]:
    """一道题必须齐备的文件。

    impl_filenames 由科目提供(template / reference / baseline 的文件名),
    spec.py 自己不硬编码任何科目相关的名字。
    """
    return [problem.statement, "spec.yaml", *impl_filenames]


def missing_files(problem: Problem, impl_filenames: Sequence[str] = ()) -> List[str]:
    return [f for f in required_files(problem, impl_filenames)
            if not (problem.root / f).is_file()]

"""配置与环境探测(nvcc / GPU / claude)。

本机是共享服务器(5×4090,256 核),因此:
  * 默认环境是他人目录下的 anaconda,不得写入 —— 本框架一律用项目内 .venv
  * GPU 需要自动挑选空闲的那张,避免和别人打架
  * claude 是本机自定义模型接入,不能硬编码 --model
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# 配置文件位置(相对仓库根)与环境变量前缀
CONFIG_FILENAME = "config.yaml"
ENV_PREFIX = "LEETSTUDY_"


def find_root(start: Optional[Path] = None) -> Path:
    """向上查找仓库根(含 pyproject.toml 且 name = leetstudy)。"""
    env_root = os.environ.get(ENV_PREFIX + "ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(encoding="utf-8")
            except OSError:
                continue
            if "leetstudy" in text:
                return candidate
    # 兜底:调用方所在位置(源码树)
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], timeout: int = 20) -> Tuple[int, str]:
    """跑一个探测命令,返回 (returncode, stdout+stderr)。绝不抛异常。"""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        return proc.returncode, proc.stdout or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    compute_cap: str
    mem_used_mib: int
    mem_total_mib: int
    util_pct: int

    @property
    def free_mib(self) -> int:
        return max(0, self.mem_total_mib - self.mem_used_mib)

    def __str__(self) -> str:
        return (
            f"GPU {self.index}  {self.name}  sm_{self.compute_cap.replace('.', '')}  "
            f"显存 {self.mem_used_mib}/{self.mem_total_mib} MiB  利用率 {self.util_pct}%"
        )


# nvidia-smi 在某些机器上**极慢**(本机实测每次 4.3 秒,与参数无关,也不缓存)。
# 而它出现的地方大多只是想知道"有哪些卡、算力多少" —— 这些信息一天之内不会变。
# 所以做两级缓存:
#   * 进程内 memo        —— 同一条命令里多次探测只付一次代价
#   * 磁盘缓存(带 TTL)  —— 跨命令复用;利用率取短 TTL,架构取长 TTL
_UTIL_TTL_SECONDS = 60.0         # 显存/利用率:一分钟的陈旧无伤大雅
#   为什么不是几秒:一次判题要好几十秒,如果 TTL 比判题还短,那每个用例都会
#   重新探测一次(本机 4.3s/次 × 每个用例)。而"哪张卡最闲"这个判断按分钟级
#   陈旧完全够用 —— GPU 占用是以训练任务为尺度变化的,不是以秒。
_ARCH_TTL_SECONDS = 30 * 86400   # 架构/卡名:机器不变就不会变
_nvidia_smi_memo: Dict[str, Any] = {"at": 0.0, "gpus": None}
#: 本次进程内已经定下来的 GPU 选择。一次判题里绝不该改主意 ——
#: 既省掉重复探测,也保证同一次运行的所有用例落在同一张卡上,计时才可比。
_pick_memo: Dict[Any, Optional[int]] = {}


def _cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "leetstudy" / "gpu.json"


def _read_disk_cache() -> Dict[str, Any]:
    try:
        return json.loads(_cache_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_disk_cache(data: Dict[str, Any]) -> None:
    path = _cache_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # 缓存写不进去不是错误,只是下次还得再探一次


def _probe_gpus() -> List[GpuInfo]:
    """真正去问 nvidia-smi。这是唯一会付那 4 秒代价的地方。"""
    code, out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return []
    gpus: List[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                compute_cap=parts[2],
                mem_used_mib=int(parts[3]),
                mem_total_mib=int(parts[4]),
                util_pct=int(parts[5]),
            ))
        except ValueError:
            continue
    return gpus


def list_gpus() -> List[GpuInfo]:
    """枚举物理 GPU,带缓存。

    缓存顺序:进程内 memo → 磁盘缓存(TTL 内)→ 重新探测。
    绝大多数命令落在这三层的前两层,不必付 nvidia-smi 的启动开销。
    """
    now = time.time()

    # 1) 进程内
    if _nvidia_smi_memo["gpus"] is not None and \
            now - _nvidia_smi_memo["at"] < _UTIL_TTL_SECONDS:
        return _nvidia_smi_memo["gpus"]

    # 2) 磁盘
    disk = _read_disk_cache()
    probed_at = float(disk.get("probed_at") or 0)
    if disk.get("gpus") and now - probed_at < _UTIL_TTL_SECONDS:
        gpus = [GpuInfo(**g) for g in disk["gpus"]]
        _nvidia_smi_memo.update({"at": now, "gpus": gpus})
        return gpus

    # 3) 真去探测
    gpus = _probe_gpus()
    _nvidia_smi_memo.update({"at": now, "gpus": gpus})
    if gpus:
        disk.update({
            "probed_at": now,
            "gpus": [asdict(g) for g in gpus],
            # 架构信息一次探测长期有效
            "arch": f"sm_{gpus[0].compute_cap.replace('.', '')}",
            "arch_at": now,
            "gpu_name": gpus[0].name,
        })
        _write_disk_cache(disk)
    return gpus


def pick_gpu(explicit: Optional[int] = None) -> Optional[int]:
    """挑一张空闲 GPU。

    优先级:显式指定 > CUDA_VISIBLE_DEVICES > 按 (利用率, 已用显存) 升序挑最闲的。
    返回物理 GPU 序号;None 表示交给驱动默认(不设 CUDA_VISIBLE_DEVICES)。

    结果在进程内缓存:同一次判题的所有用例必须落在同一张卡上,否则计时不可比;
    顺带也省掉反复探测 nvidia-smi 的开销。
    """
    if explicit is not None:
        return explicit
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return None  # 用户已经安排好可见性,别覆盖

    if explicit in _pick_memo:
        return _pick_memo[explicit]

    gpus = list_gpus()
    if not gpus:
        chosen = None
    else:
        best = min(gpus, key=lambda g: (g.util_pct, g.mem_used_mib))
        # 唯一一张卡时也别费事
        chosen = best.index if len(gpus) > 1 else None
    _pick_memo[explicit] = chosen
    return chosen


# --------------------------------------------------------------------------- #
# 编译
# --------------------------------------------------------------------------- #

def detect_arch(use_cache: bool = True) -> str:
    """探测目标架构,如 sm_89。用于 nvcc -arch / TORCH_CUDA_ARCH_LIST。

    优先走磁盘缓存 —— 架构是一台机器上最不会变的信息,而探测它要 4 秒。
    """
    if use_cache:
        disk = _read_disk_cache()
        arch = disk.get("arch")
        arch_at = float(disk.get("arch_at") or 0)
        if arch and time.time() - arch_at < _ARCH_TTL_SECONDS:
            return str(arch)

    gpus = list_gpus()
    if gpus:
        cap = gpus[0].compute_cap.replace(".", "")
        if re.match(r"^\d+$", cap):
            return f"sm_{cap}"

    # 退路:取 nvcc 支持的最高架构
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    code, out = _run([nvcc, "--list-gpu-arch"])
    if code == 0:
        archs = re.findall(r"compute_(\d+)", out)
        if archs:
            return f"sm_{max(archs, key=int)}"
    return "sm_75"


# 常见 GPU 的标称峰值带宽(GB/s),用于把实测 GB/s 换算成「占峰值百分比」。
# 仅作评分参照。实测值可能略高于标称 —— 因为 cudaEvent 停止计时的瞬间,
# 最后一批写还在 L2 里没落盘,少算了这部分显存流量。
PEAK_BANDWIDTH_GBPS: dict = {
    "RTX 5090": 1792.0,
    "RTX 4090": 1008.0,
    "RTX 4080": 717.0,
    "RTX 3090": 936.0,
    "RTX 3080": 760.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-SXM4-80GB": 2039.0,
    "A100-PCIE": 1555.0,
    "H100-SXM5": 3350.0,
    "H100-PCIE": 2039.0,
    "L40S": 864.0,
    "V100-SXM2": 900.0,
    "V100-PCIE": 898.0,
    "T4": 320.0,
    "A10": 600.0,
}


def lookup_peak_bandwidth(gpu_name: str) -> Optional[float]:
    """按 GPU 名称查标称峰值带宽。名称取最长匹配,避免 V100-SXM2 命中 V100-PCIE。"""
    best: Optional[tuple] = None
    for key, value in PEAK_BANDWIDTH_GBPS.items():
        if key.lower() in gpu_name.lower():
            if best is None or len(key) > best[0]:
                best = (len(key), value)
    return best[1] if best else None


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    root: Path
    nvcc: str = "nvcc"
    sanitizer: str = "compute-sanitizer"
    claude_bin: str = "claude"
    #: 目标架构。**惰性求值** —— 探测它要跑 nvidia-smi,而本机实测每次要 4 秒;
    #: 可是 `leet list` / `show` / `start` 这些命令压根不需要它。
    #: 只有真要编译/跑题时才付这个代价。外部仍可用 `cfg.arch = "sm_80"` 覆盖。
    _arch: Optional[str] = None
    gpu: Optional[int] = None            # 显式指定的物理 GPU
    claude_model: Optional[str] = None   # None = 继承用户默认(本机是自定义模型)
    claude_extra_args: List[str] = field(default_factory=list)
    compile_timeout: int = 180
    run_timeout: int = 120
    sanitizer_timeout: int = 600
    # 单次出题的模型调用超时。出题是一个「写 → 编译 → 跑 → 改」的长循环,
    # 往往要十几到几十分钟,默认给足。
    author_timeout: int = 3600
    # 显式覆盖峰值带宽(GB/s);None 表示按 GPU 名称自动查表
    peak_bw_override: Optional[float] = None
    # 出题/讲评时放给 claude 的工具白名单(None = 用 agent 模块的默认值)
    claude_allowed_tools: Optional[List[str]] = None

    def resolve_peak_bandwidth(self) -> Optional[float]:
        """本题所在的 GPU 的标称峰值带宽。查不到返回 None(此时不显示占比)。"""
        if self.peak_bw_override:
            return self.peak_bw_override
        gpus = list_gpus()
        if not gpus:
            return None
        return lookup_peak_bandwidth(gpus[0].name)

    @property
    def arch(self) -> str:
        """目标架构,如 sm_89。**首次访问时才探测**。

        探测要跑 nvidia-smi(本机约 4 秒),而只有真的要编译或跑题时才需要它 ——
        `leet list` / `show` / `start` 都不碰这个属性,所以不会付这个代价。
        """
        if self._arch is None:
            self._arch = detect_arch()
        return self._arch

    @arch.setter
    def arch(self, value: Optional[str]) -> None:
        self._arch = value

    @property
    def problems_dir(self) -> Path:
        return self.root / "problems"

    @property
    def solutions_dir(self) -> Path:
        return self.root / "solutions"

    @property
    def build_dir(self) -> Path:
        return self.root / "build"

    def build_dir_for(self, problem_id: str) -> Path:
        return self.build_dir / problem_id

    def env_for_gpu(self, gpu: Optional[int]) -> dict:
        """返回设置了 CUDA_VISIBLE_DEVICES 的环境变量副本。"""
        env = dict(os.environ)
        chosen = self.gpu if gpu is None else gpu
        if chosen is None:
            chosen = pick_gpu(None)
        if chosen is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(chosen)
        return env

    def resolve_gpu(self, override: Optional[int]) -> Optional[int]:
        return pick_gpu(self.gpu if override is None else override)


def load_config(root: Optional[Path] = None) -> Config:
    """加载配置:config.yaml < 环境变量 < 调用方覆盖。

    root 可以是 Path 或字符串,内部统一转绝对路径 ——
    编译器/子进程的工作目录未必是当前目录。
    """
    root = (Path(root).expanduser() if root else find_root()).resolve()
    cfg = Config(root=root)

    cfg_path = root / CONFIG_FILENAME
    raw: dict = {}
    if cfg_path.is_file():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except yaml.YAMLError:
            raw = {}

    def env(key: str) -> Optional[str]:
        return os.environ.get(ENV_PREFIX + key)

    cfg.nvcc = env("NVCC") or str(raw.get("nvcc") or cfg.nvcc)
    cfg.sanitizer = env("SANITIZER") or str(raw.get("sanitizer") or cfg.sanitizer)
    cfg.claude_bin = env("CLAUDE_BIN") or str(raw.get("claude_bin") or cfg.claude_bin)
    # 架构留空 → 惰性探测;只有显式配置时才在这里定下来
    explicit_arch = env("ARCH") or raw.get("arch")
    cfg.arch = str(explicit_arch) if explicit_arch else None
    cfg.claude_model = env("CLAUDE_MODEL") or raw.get("claude_model") or None

    # 路径类设置:相对路径按仓库根解析
    for attr in ("nvcc", "sanitizer", "claude_bin"):
        value = getattr(cfg, attr)
        if "/" in value:
            p = Path(value).expanduser()
            if not p.is_absolute():
                p = root / p
            setattr(cfg, attr, str(p))

    raw_extra = raw.get("claude_extra_args") or []
    if isinstance(raw_extra, list):
        cfg.claude_extra_args = [str(a) for a in raw_extra]

    raw_tools = raw.get("claude_allowed_tools")
    if isinstance(raw_tools, list):
        cfg.claude_allowed_tools = [str(t) for t in raw_tools]

    if raw.get("gpu") is not None:
        try:
            cfg.gpu = int(raw["gpu"])
        except (TypeError, ValueError):
            cfg.gpu = None
    if env("GPU"):
        try:
            cfg.gpu = int(env("GPU"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass

    for attr, default in (
        ("compile_timeout", cfg.compile_timeout),
        ("run_timeout", cfg.run_timeout),
        ("sanitizer_timeout", cfg.sanitizer_timeout),
        ("author_timeout", cfg.author_timeout),
    ):
        value = env(attr.upper()) or raw.get(attr)
        try:
            setattr(cfg, attr, int(value))
        except (TypeError, ValueError):
            setattr(cfg, attr, default)

    peak = env("PEAK_BANDWIDTH_GBPS") or raw.get("peak_bandwidth_gbps")
    try:
        cfg.peak_bw_override = float(peak) if peak else None
    except (TypeError, ValueError):
        cfg.peak_bw_override = None

    return cfg


# --------------------------------------------------------------------------- #
# 环境自检(leet doctor 用)
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def doctor(cfg: Config) -> List[Check]:
    checks: List[Check] = []

    # nvcc
    nvcc_path = shutil.which(cfg.nvcc) or (cfg.nvcc if Path(cfg.nvcc).is_file() else None)
    if nvcc_path:
        code, out = _run([nvcc_path, "--version"])
        ver = ""
        if code == 0:
            m = re.search(r"release (\d+\.\d+)", out)
            ver = f"release {m.group(1)}" if m else ""
        checks.append(Check("nvcc", True, f"{nvcc_path} {ver}".strip()))
    else:
        checks.append(Check(
            "nvcc", False, f"未找到 {cfg.nvcc!r}",
            "安装 CUDA Toolkit 或设置 LEETSTUDY_NVCC=/path/to/nvcc",
        ))

    # GPU
    gpus = list_gpus()
    if gpus:
        free = [g for g in gpus if g.util_pct < 10 and g.mem_used_mib < 1024]
        detail = f"{len(gpus)} 张卡,{len(free)} 张空闲;目标架构 {cfg.arch}"
        checks.append(Check("GPU", True, detail))
    else:
        checks.append(Check(
            "GPU", False, "nvidia-smi 不可用或未检测到 GPU",
            "确认驱动安装、容器有 --gpus 权限",
        ))

    # compute-sanitizer
    san = shutil.which(cfg.sanitizer) or (
        cfg.sanitizer if Path(cfg.sanitizer).is_file() else None
    )
    if san:
        checks.append(Check("compute-sanitizer", True, san))
    else:
        checks.append(Check(
            "compute-sanitizer", False, f"未找到 {cfg.sanitizer!r}",
            "通常随 CUDA Toolkit 提供;缺失只影响内存/竞态检查,不影响判分",
        ))

    # claude CLI(出题与讲评依赖)
    claude = shutil.which(cfg.claude_bin) or (
        cfg.claude_bin if Path(cfg.claude_bin).is_file() else None
    )
    if claude:
        code, out = _run([claude, "--version"], timeout=30)
        ver = out.strip().splitlines()[0] if code == 0 and out.strip() else ""
        model_note = cfg.claude_model or "继承用户默认"
        checks.append(Check("claude CLI", True, f"{claude}  {ver}  (模型: {model_note})"))
    else:
        checks.append(Check(
            "claude CLI", False, f"未找到 {cfg.claude_bin!r}",
            "出题(leet new)与讲评(leet review)需要它;设置 LEETSTUDY_CLAUDE_BIN",
        ))

    # PyTorch 科目依赖(纯 CUDA 题不需要)
    try:
        import importlib.util
        spec_torch = importlib.util.find_spec("torch")
        if spec_torch is None:
            checks.append(Check(
                "torch", False, "未安装",
                "PyTorch 科目的题目需要它:pip install torch==2.4.1"
                "(驱动 535 支持到 CUDA 12.2,只能装 cu121 及更早的构建)",
            ))
        else:
            import torch as _torch
            cuda_ok = _torch.cuda.is_available()
            detail = f"{_torch.__version__}  CUDA {'可用' if cuda_ok else '不可用'}"
            checks.append(Check(
                "torch", cuda_ok, detail,
                "" if cuda_ok else
                "torch 装上了但用不了 CUDA —— 多半是 wheel 的 CUDA 版本高于驱动支持的上限",
            ))
    except Exception as exc:  # noqa: BLE001  doctor 不该因为任何意外而崩
        checks.append(Check("torch", False, f"检查失败:{type(exc).__name__}: {exc}"))

    # ninja:只在「PyTorch + 自定义 CUDA 算子」的题上才需要(load_inline 用它编译)
    if shutil.which("ninja") or (cfg_root_bin := (cfg.root / ".venv" / "bin" / "ninja")).is_file():
        checks.append(Check("ninja", True, "可用(自定义 CUDA 算子题需要)"))

    # venv:确认没有误用他人的 anaconda
    import sys
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    expected = cfg.root / ".venv"
    if in_venv and Path(sys.prefix) == expected:
        checks.append(Check("Python 环境", True, f"项目 venv:{sys.prefix}"))
    elif in_venv:
        checks.append(Check("Python 环境", True, f"虚拟环境:{sys.prefix}"))
    else:
        checks.append(Check(
            "Python 环境", False, f"未在虚拟环境中运行({sys.prefix})",
            f"用 {expected}/bin/leet 或先 source {expected}/bin/activate",
        ))

    return checks

"""判题科目注册表。

科目按名字解析(spec.yaml 里的 `subject` 字段)。用惰性导入 —— 科目模块可能依赖
重型库(torch),`leet list` 这类命令不该为此付出启动代价。
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Tuple, Type
from ..config import Config
from .base import Artifact, BuildResult, CaseResult, JSON_MARKER, SanitizeResult, Subject

#: 科目名 → (模块路径, 类名)
_REGISTRY: Dict[str, Tuple[str, str]] = {
    "cuda": ("leetstudy.subjects.cuda", "CudaSubject"),
    "pytorch": ("leetstudy.subjects.pytorch", "PyTorchSubject"),
}

#: spec.yaml 未声明 subject 时用哪个(保持既有 CUDA 题目无需改动)
DEFAULT_SUBJECT = "cuda"


def available() -> List[str]:
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    return (name or "").strip().lower() in _REGISTRY


def _subject_class(name: str) -> Type[Subject]:
    key = (name or DEFAULT_SUBJECT).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"未知科目 {name!r};已注册:{', '.join(available())}")
    module_name, class_name = _REGISTRY[key]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get(name: str, cfg: Config) -> Subject:
    """按名字取科目实例。

    科目的构造函数统一接收 Config,这样它可以拿 nvcc 路径、GPU 选择、超时等设置。
    """
    return _subject_class(name)(cfg)


# --------------------------------------------------------------------------- #
# 只读元信息:不实例化科目就能拿到(给 CLI / bank / validate 用)
#
# 这些走类属性而不是实例,是为了让 `leet list` 这类命令不必构造科目、
# 也就不必导入它依赖的重型库。
# --------------------------------------------------------------------------- #

def solution_filename(name: str) -> str:
    return _subject_class(name).solution_filename


def template_filename(name: str) -> str:
    return _subject_class(name).template_filename


def baseline_filename(name: str) -> str:
    return _subject_class(name).baseline_filename


def reference_filename(name: str) -> str:
    return _subject_class(name).reference_filename


def optimal_filename(name: str) -> str:
    """参考解的文件名(可能为空 —— 该科目没定义,或题目还没补)。"""
    return str(getattr(_subject_class(name), "optimal_filename", ""))


def impl_filenames(name: str) -> List[str]:
    """科目需要齐备的实现文件(template / reference / baseline)。"""
    return list(_subject_class(name).required_filenames())


def required_entry_keys(name: str) -> Tuple[str, ...]:
    return tuple(getattr(_subject_class(name), "required_entry_keys", ()))


def mutant_docs(name: str) -> Dict[str, str]:
    return dict(getattr(_subject_class(name), "mutant_docs", {}))


def supports_sanitize(name: str) -> bool:
    return bool(getattr(_subject_class(name), "supports_sanitize", False))


def build_label(name: str) -> str:
    return str(getattr(_subject_class(name), "build_label", "准备"))


def build_note(name: str) -> str:
    return str(getattr(_subject_class(name), "build_note", ""))


__all__ = [
    "Artifact", "BuildResult", "CaseResult", "JSON_MARKER", "SanitizeResult",
    "Subject", "available", "get", "is_registered", "DEFAULT_SUBJECT",
    "solution_filename", "template_filename", "baseline_filename",
    "reference_filename", "optimal_filename", "impl_filenames", "required_entry_keys",
    "mutant_docs", "supports_sanitize", "build_label", "build_note",
]

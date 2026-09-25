"""题库:发现题目、按题号查找、记录做题进度。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .spec import Problem, SpecError, load_problem, missing_files

PROGRESS_FILENAME = "progress.json"


def discover(problems_dir: Path) -> List[Problem]:
    """扫描题库目录,返回可加载的题目(按 id 排序)。坏掉的题目会被跳过。"""
    good, _ = discover_with_errors(problems_dir)
    return good


def discover_with_errors(problems_dir: Path) -> Tuple[List[Problem], List[Tuple[Path, str]]]:
    """同上,但把加载失败的题目也返回,便于 `leet validate` 报出来。"""
    problems: List[Problem] = []
    broken: List[Tuple[Path, str]] = []
    if not problems_dir.is_dir():
        return problems, broken
    for entry in sorted(problems_dir.iterdir()):
        if not entry.is_dir() or not (entry / "spec.yaml").is_file():
            continue
        try:
            problems.append(load_problem(entry))
        except SpecError as exc:
            broken.append((entry, str(exc)))
    problems.sort(key=lambda p: p.id)
    return problems, broken


def resolve(problems: List[Problem], query: str) -> Optional[Problem]:
    """按题号查找,支持几种自然的写法。

    匹配优先级:完全一致 > 数字前缀(1 / 01 → 01-vector-add) > 唯一子串。
    """
    q = query.strip().lower()
    if not q:
        return None

    for p in problems:
        if p.id.lower() == q:
            return p

    # 纯数字 → 匹配 id 开头的数字部分
    if re.match(r"^\d+$", q):
        want = int(q)
        hits = []
        for p in problems:
            m = re.match(r"^(\d+)", p.id)
            if m and int(m.group(1)) == want:
                hits.append(p)
        if len(hits) == 1:
            return hits[0]

    hits = [p for p in problems if q in p.id.lower() or q in p.title.lower()]
    if len(hits) == 1:
        return hits[0]
    return None


def resolve_many(problems: List[Problem], query: str) -> List[Problem]:
    """宽松匹配,返回所有候选(用于给出「你是想找这些吗」的提示)。"""
    q = query.strip().lower()
    return [p for p in problems
            if q in p.id.lower() or q in p.title.lower()]


# --------------------------------------------------------------------------- #
# 进度
# --------------------------------------------------------------------------- #

@dataclass
class Entry:
    solved: bool = False
    attempts: int = 0
    best_grade: Optional[str] = None
    best_metric: Optional[float] = None
    metric_name: str = ""
    best_ms: Optional[float] = None
    last: str = ""


@dataclass
class Progress:
    path: Path
    entries: Dict[str, Entry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Progress":
        entries: Dict[str, Entry] = {}
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            for pid, data in (raw.get("problems") or {}).items():
                if isinstance(data, dict):
                    entries[pid] = Entry(
                        solved=bool(data.get("solved")),
                        attempts=int(data.get("attempts") or 0),
                        best_grade=data.get("best_grade"),
                        best_metric=data.get("best_metric"),
                        metric_name=str(data.get("metric_name") or ""),
                        best_ms=data.get("best_ms"),
                        last=str(data.get("last") or ""),
                    )
        return cls(path=path, entries=entries)

    def get(self, problem_id: str) -> Entry:
        return self.entries.get(problem_id) or Entry()

    def save(self) -> None:
        payload = {
            "problems": {
                pid: {
                    "solved": e.solved,
                    "attempts": e.attempts,
                    "best_grade": e.best_grade,
                    "best_metric": e.best_metric,
                    "metric_name": e.metric_name,
                    "best_ms": e.best_ms,
                    "last": e.last,
                }
                for pid, e in sorted(self.entries.items())
            }
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def record(
        self,
        problem_id: str,
        solved: bool,
        grade: Optional[str],
        metric_value: Optional[float],
        metric_name: str,
        median_ms: Optional[float],
    ) -> Entry:
        """记录一次尝试。只在更好的成绩上覆盖最佳值。"""
        order = {"S": 4, "A": 3, "B": 2, "C": 1}
        e = self.entries.get(problem_id) or Entry()
        e.attempts += 1
        e.last = date.today().isoformat()
        if solved:
            e.solved = True
        # 指标方向:加速比越大越好;带宽占比也是越大越好。两者都是"越大越好"。
        if metric_value is not None:
            if e.best_metric is None or metric_value > e.best_metric:
                e.best_metric = metric_value
                e.metric_name = metric_name
        if grade and (not e.best_grade
                      or order.get(grade, 0) > order.get(e.best_grade, 0)):
            e.best_grade = grade
        if median_ms is not None:
            if e.best_ms is None or median_ms < e.best_ms:
                e.best_ms = median_ms
        self.entries[problem_id] = e
        return e


def solution_path(solutions_dir: Path, problem: Problem) -> Path:
    """解答文件路径。扩展名由题目所属科目决定(solution.cu / solution.py)。"""
    from . import subjects
    return solutions_dir / problem.id / subjects.solution_filename(problem.subject)


def all_missing_files(problems: List[Problem]) -> List[Tuple[str, List[str]]]:
    """返回文件不齐的题目,供 `leet validate` 使用。"""
    from . import subjects
    out = []
    for p in problems:
        try:
            names = subjects.impl_filenames(p.subject)
        except KeyError:
            names = []
        missing = missing_files(p, names)
        if missing:
            out.append((p.id, missing))
    return out

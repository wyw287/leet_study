"""PyTorch 科目的 runner —— CUDA 侧 harness.cu 的对等物。

由 PyTorchSubject 以子进程方式调用::

    python -m leetstudy.subjects.pytorch_runner \\
        --spec <题目目录> --impl <解答.py> --case <用例名> [--perf]

职责与 CUDA harness 逐条对应:

    构造输入(种子由用例名决定) → 跑 CPU 参考解得到期望值 → 把输入搬上 GPU
    → 毒化输出 → 调用用户的 forward(ctx) → 检查哨兵区 → 搬回来逐步比对
    → 计时(每次迭代前清 L2) → 打印一行 JSON

为了让报告层与诊断层完全复用,输出的 JSON 字段与 CUDA 侧**逐字一致**。

为什么参考解跑在 CPU 上:与 CUDA 侧同理 —— 参考解是 ground truth,
必须自身不可能有 GPU 侧的错误(竞态、越界、同步遗漏)。CPU torch 慢一点没关系。

用户接口是 `forward(ctx)` 而不是 `forward(*tensors)`,这是刻意的:
框架预分配输出张量并把它做成带哨兵区的缓冲区的一个视图,才能拿到
「往前/往后越界了多少个元素」这种精度的诊断。代价是与「返回新张量」
的惯用写法不同 —— 但这个取舍对学习工具是值得的。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..spec import Buffer, Problem, load_problem
from .base import JSON_MARKER

# 哨兵区元素个数,与 CUDA 侧保持一致
GUARD_ELEMS = 4096

# 哨兵值(与 CUDA 侧同源):浮点用一个特征值而不是 NaN —— NaN 不等于自身,没法比较
_GUARD_VALUE = {
    "f32": -1.2345678e33,
    "f64": -1.234567890123e300,
    "i32": 0x600D600D,
    "i64": 0x600D600D600D600D,
    "u32": 0x600D600D,
    "u8": 0x6D,
}
# 毒值:输出缓冲在校验前填这个,用来识别「kernel 根本没写输出」
_POISON_VALUE = {
    "f32": float("nan"),
    "f64": float("nan"),
    "i32": 0x5EED5EED,
    "i64": 0x5EED5EED5EED5EED,
    "u32": 0x5EED5EED,
    "u8": 0x5E,
}

FLOAT_DTYPES = ("f32", "f64")

# 清 L2 用的缓冲大小。
#
# 为什么是写死的保守值而不是查出来的真实值:torch 2.4 的 device properties 里
# 没有 L2 字段(2.4.1 实测只有 name / major / minor / total_memory /
# multi_processor_count / regs_per_multiprocessor …),nvidia-smi 也不支持
# `l2_cache_size` 查询。所以这里取一个**比现存所有 GPU 的 L2 都大**的值:
#   4090 = 72MB,A100 = 40MB,H100 = 50MB,B200 = 126MB —— 256MB 全部覆盖。
#
# 方向是刻意选的:**清过头只是慢**(这部分在计时区外,不影响测得的数字),
# 清不够则会让题目数据驻留在缓存里,测出来的是缓存带宽而不是显存带宽 ——
# 那是会得出错误结论的。宁可慢,不可错。
_DEFAULT_FLUSH_BYTES = 256 * 1024 * 1024


def _flush_bytes() -> int:
    import os
    raw = os.environ.get("LEETSTUDY_L2_BYTES")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return _DEFAULT_FLUSH_BYTES


def _torch_dtype(name: str):
    import torch
    table = {
        "f32": torch.float32,
        "f64": torch.float64,
        "i32": torch.int32,
        "i64": torch.int64,
        "u8": torch.uint8,
    }
    if name == "u32":
        # torch 的 uint32 支持分版本,没有就退化成 int64(数值语义仍正确)
        return getattr(torch, "uint32", torch.int64)
    if name not in table:
        raise ValueError(f"不支持的 dtype: {name}")
    return table[name]


class Ctx:
    """传给用户/参考解的上下文。字段名来自 spec 的 buffers 与 params。

    做成允许随意读写的普通对象(而不是 frozen dataclass),是为了让使用手感
    与 CUDA 侧的 LaunchCtx 一致 —— 但框架会检测「你把 ctx.out 重新赋值了」
    这种写法,因为那样框架就看不到你的结果了。
    """

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={type(v).__name__}" for k, v in self.__dict__.items())
        return f"Ctx({items})"


# --------------------------------------------------------------------------- #
def _seed_of(name: str) -> int:
    """由用例名派生种子 —— 同一个用例每次跑看到同一组数据。"""
    h = 1469598103934665603
    for ch in name:
        h ^= ord(ch)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h or 1


def _load_module(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def _fill(tensor, mode: str, seed: int) -> None:
    import torch
    gen = torch.Generator(device=tensor.device).manual_seed(seed)
    if tensor.dtype.is_floating_point:
        if mode == "zero":
            tensor.zero_()
        elif mode == "positive":
            tensor.uniform_(1e-3, 1.0, generator=gen)
        elif mode == "randint":
            tensor.copy_(torch.randint(0, 100, tuple(tensor.shape),
                                       generator=gen, device=tensor.device).to(tensor.dtype))
        else:
            tensor.uniform_(-1.0, 1.0, generator=gen)
    else:
        if mode == "zero":
            tensor.zero_()
        elif mode == "positive":
            tensor.random_(1, 100, generator=gen)
        else:
            tensor.random_(0, 100, generator=gen)


class Runner:
    def __init__(self, problem: Problem, impl_path: Path) -> None:
        import torch  # noqa: F401  在这里导入,让缺 torch 的环境报错更清楚
        self.problem = problem
        self.impl_path = impl_path
        self.torch = torch

    # ---- 张量构造 -------------------------------------------------------- #

    def _make(self, buf: Buffer, params: Dict[str, float], device: str,
              guarded: bool) -> Tuple[Any, Any]:
        """返回 (给用户看的张量, 用于检查哨兵的完整缓冲区)。

        guarded=False 时不加哨兵区,两者返回同一个张量。
        """
        torch = self.torch
        dtype = _torch_dtype(buf.dtype)
        shape = buf.resolve_shape(params)
        n = buf.count(params)

        if not guarded:
            t = torch.empty(shape or (1,), dtype=dtype, device=device)
            return t, t

        flat = torch.full((n + 2 * GUARD_ELEMS,), _GUARD_VALUE[buf.dtype],
                          dtype=dtype, device=device)
        view = flat[GUARD_ELEMS:GUARD_ELEMS + n]
        if shape:
            view = view.reshape(shape)
        return view, flat

    @staticmethod
    def _guard_hits(flat, dtype: str) -> Dict[str, int]:
        import torch
        guard = _GUARD_VALUE[dtype]
        front = flat[:GUARD_ELEMS]
        back = flat[GUARD_ELEMS + (flat.numel() - 2 * GUARD_ELEMS):]
        return {
            "front": int((front != guard).sum().item()),
            "back": int((back != guard).sum().item()),
        }

    # ---- 主流程 ---------------------------------------------------------- #

    def run(self, case_name: str, do_perf: bool, warmup: int, repeat: int) -> int:
        torch = self.torch
        problem = self.problem
        try:
            case = problem.case(case_name)
        except Exception as exc:
            print(f"未知用例 {case_name}: {exc}", file=sys.stderr)
            return 2
        params = dict(case.params)

        # ---------- 1. CPU 参考解(ground truth) ---------- #
        # 参考解用**固定名字** reference(ctx),与 CUDA 侧 reference.cpp 的定义
        # `void reference(RefCtx&)` 对齐 —— 它是 oracle,不是用户的接口。
        # 用户要实现的入口名由 spec 的 entry.function 决定。
        try:
            ref_mod = _load_module(problem.root / "reference.py", "_leet_reference")
        except Exception as exc:
            return self._bail(case_name, "加载参考解", exc)
        ref_fn = getattr(ref_mod, "reference", None)
        if ref_fn is None:
            return self._bail(
                case_name, "加载参考解",
                NameError("reference.py 里必须定义 reference(ctx)"),
            )

        ref_fields: Dict[str, Any] = {}
        for buf in problem.buffers:
            if buf.role == "scratch":
                continue
            t, _ = self._make(buf, params, "cpu", guarded=False)
            if buf.role == "in":
                _fill(t, buf.fill, _seed_of(case_name))
            ref_fields[buf.name] = t
        for prm in problem.params:
            ref_fields[prm.name] = (int(params[prm.name])
                                    if prm.dtype not in FLOAT_DTYPES else float(params[prm.name]))
        try:
            ref_fn(Ctx(**ref_fields))
        except Exception as exc:
            return self._bail(case_name, "执行参考解", exc)
        expected = {b.name: ref_fields[b.name].clone() for b in problem.outputs}

        # ---------- 2. GPU 侧用户解 ---------- #
        try:
            user_mod = _load_module(self.impl_path, "_leet_user")
        except Exception as exc:
            return self._bail(case_name, "加载解答", exc)
        user_fn = getattr(user_mod, problem.function, None)
        if user_fn is None:
            return self._bail(
                case_name, "加载解答",
                NameError(
                    f"解答文件里没有定义 {problem.function}() —— "
                    f"题目的 entry.function 要求这个函数名"
                ),
            )

        device = "cuda"
        tensors: Dict[str, Any] = {}
        flats: Dict[str, Any] = {}
        poisoned: List[str] = []
        for buf in problem.buffers:
            guarded = True
            t, flat = self._make(buf, params, device, guarded=guarded)
            tensors[buf.name] = t
            flats[buf.name] = flat
            if buf.role == "in":
                # 输入用与 CPU 侧相同的种子生成,再搬到 GPU —— 两边看到同一组数据
                host_t, _ = self._make(buf, params, "cpu", guarded=False)
                _fill(host_t, buf.fill, _seed_of(case_name))
                t.copy_(host_t.reshape(t.shape))
            elif buf.role in ("out", "scratch"):
                t.fill_(_POISON_VALUE[buf.dtype])
                if buf.role == "out":
                    poisoned.append(buf.name)
        torch.cuda.synchronize()

        ctx = Ctx(**tensors)
        for prm in problem.params:
            ctx.__dict__[prm.name] = (int(params[prm.name])
                                      if prm.dtype not in FLOAT_DTYPES
                                      else float(params[prm.name]))
        bound = {name: id(tensors[name]) for name in poisoned}

        torch.cuda.synchronize()
        try:
            returned = user_fn(ctx)
            torch.cuda.synchronize()
        except Exception as exc:
            return self._announce_failure(case_name, exc)

        # 用户把 ctx.out 重新赋值过?那框架看不到结果
        rebound = [n for n in poisoned if id(getattr(ctx, n, None)) != bound[n]]

        # 用户直接 return 了张量?那也是完全合法的 PyTorch 写法,优先采用。
        # 为什么要支持两种写法:往预分配缓冲里 copy_ 会多一次全量访存,
        # 对访存瓶颈题来说这个代价足以让「正确解法」拿不到应有的评级 ——
        # 接口本身不该扭曲测量结果。
        provided = None
        try:
            provided = self._collect_result(returned, ctx, problem)
        except Exception as exc:
            return self._bail(case_name, "读取返回值", exc)
        for name, tensor in provided.items():
            if name not in tensors:
                return self._bail(
                    case_name, "读取返回值",
                    ValueError(f"返回值里有未知的输出缓冲 {name!r};"
                               f"本题的输出是 {[b.name for b in problem.outputs]}"),
                )
            tensors[name] = tensor

        # ---------- 3. 哨兵区 ---------- #
        # 只有「写进框架预分配缓冲」的那些才需要检查哨兵 —— 用户 return 出来的
        # 张量是 PyTorch 自己分配的,没有哨兵区可言。
        guards = {}
        for b in problem.buffers:
            if b.name in provided:
                guards[b.name] = {"front": 0, "back": 0}
            else:
                guards[b.name] = self._guard_hits(flats[b.name], b.dtype)

        # ---------- 4. 逐步比对 ---------- #
        outputs = []
        for buf in problem.outputs:
            got = tensors[buf.name].detach().reshape(-1).cpu()
            # 期望值跟随实际输出的 dtype(用户可能返回了更高精度)
            exp = expected[buf.name].detach().reshape(-1).to(got.dtype)
            if got.numel() != exp.numel():
                return self._bail(
                    case_name, "读取返回值",
                    ValueError(
                        f"输出 {buf.name} 的元素个数不对:"
                        f"得到 {got.numel()},期望 {exp.numel()}"
                    ),
                )
            outputs.append(self._compare(buf, got, exp))

        # ---------- 5. 计时 ---------- #
        perf = None
        if do_perf and problem.perf.enabled:
            perf = self._time(user_fn, ctx, warmup, repeat, params)

        # ---------- 6. 输出 ---------- #
        ok = all(o["bad"] == 0 for o in outputs) and all(
            g["front"] == 0 and g["back"] == 0 for g in guards.values())
        payload: Dict[str, Any] = {
            "case": case_name,
            "params": {k: (int(v) if float(v).is_integer() else v) for k, v in params.items()},
            "outputs": outputs,
            "guards": guards,
            "ok": ok,
        }
        if rebound:
            payload["rebound"] = rebound
        if perf:
            payload["perf"] = perf
        self._emit(payload)
        return 0

    # ---- 返回值归一化 ---------------------------------------------------- #

    def _collect_result(self, returned: Any, ctx: Any, problem: Problem) -> Dict[str, Any]:
        """把 forward 的返回值归一化成 {输出缓冲名: 张量}。

        支持几种写法(按 PyTorch 的惯用程度排序):

            return tensor             单输出题的惯用写法
            return (t1, t2, …)        按 spec 里 outputs 的顺序
            return {"name": tensor}   显式指定(多输出时最清楚)
            return None               表示已经写进 ctx.<name> 了

        为什么允许 return:往预分配缓冲里 copy_ 会多一次全量访存,对访存瓶颈题
        来说,这个代价足以让正确解法拿不到应有的评级 —— 接口不该扭曲测量。
        代价是 return 出来的张量由 PyTorch 分配,没有哨兵区可查;
        但纯 PyTorch 算子不可能越界写用户张量,这个损失可以接受。
        """
        import torch
        outs = [b.name for b in problem.outputs]
        if returned is None:
            return {}
        if isinstance(returned, torch.Tensor):
            if len(outs) != 1:
                raise ValueError(
                    f"本题有 {len(outs)} 个输出 {outs},不能只 return 一个张量;"
                    f"请 return 元组或字典,或者写进 ctx.<名字>"
                )
            return {outs[0]: returned}
        if isinstance(returned, dict):
            return {str(k): v for k, v in returned.items()}
        if isinstance(returned, (tuple, list)):
            if len(returned) != len(outs):
                raise ValueError(
                    f"return 了 {len(returned)} 个值,但本题有 {len(outs)} 个输出 {outs}"
                )
            return dict(zip(outs, returned))
        raise ValueError(
            f"不认识的返回值类型 {type(returned).__name__};"
            f"可以 return 张量 / 元组 / 字典,或写进 ctx.<名字> 后 return None"
        )

    # ---- 比对 ------------------------------------------------------------ #

    def _compare(self, buf: Buffer, got, exp) -> Dict[str, Any]:
        import torch
        is_float = buf.dtype in FLOAT_DTYPES
        n = int(got.numel())
        atol = self.problem.verify.atol
        rtol = self.problem.verify.rtol

        if is_float:
            both_nan = torch.isnan(got) & torch.isnan(exp)
            exact = got == exp                      # 覆盖 inf == inf
            diff = (got - exp).abs()
            tol = atol + rtol * exp.abs()
            bad_mask = ~(both_nan | exact | (diff <= tol))
            nan_count = int(torch.isnan(got).sum().item())
            inf_count = int(torch.isinf(got).sum().item())
            finite = ~both_nan
            max_abs = float(diff[finite].max().item()) if bool(finite.any()) else 0.0
            denom = exp.abs().clamp_min(1e-30)
            max_rel = float((diff / denom)[finite].max().item()) if bool(finite.any()) else 0.0
        else:
            diff = (got.to(torch.int64) - exp.to(torch.int64)).abs()
            bad_mask = diff != 0
            nan_count = inf_count = 0
            max_abs = float(diff.max().item()) if n else 0.0
            max_rel = 0.0

        bad = int(bad_mask.sum().item())
        first_bad = -1
        first_got = first_exp = None
        if bad:
            idx = int(bad_mask.nonzero()[0].item())
            first_bad = idx
            first_got = float(got[idx].item())
            first_exp = float(exp[idx].item())

        def jnum(v: Optional[float]) -> Optional[float]:
            if v is None:
                return None
            return v if v == v and abs(v) != float("inf") else None

        return {
            "name": buf.name,
            "count": n,
            "bad": bad,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "first_bad_idx": first_bad,
            "first_got": jnum(first_got),
            "first_exp": jnum(first_exp),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

    # ---- 计时 ------------------------------------------------------------ #

    def _time(self, fn, ctx, warmup: int, repeat: int,
              params: Dict[str, float]) -> Dict[str, Any]:
        torch = self.torch
        problem = self.problem

        # 有效带宽按「至少搬动的字节数」算:in 读 + out 写
        sizes = {"f32": 4, "f64": 8, "i32": 4, "i64": 8, "u32": 4, "u8": 1}
        bytes_moved = sum(
            sizes[b.dtype] * b.count(params)
            for b in problem.inputs + problem.outputs
        )

        flush = None
        if problem.perf.flush_l2:
            flush = torch.empty(_flush_bytes(), dtype=torch.int32, device="cuda")

        for _ in range(max(0, warmup)):
            if flush is not None:
                flush.sum()
            fn(ctx)
        torch.cuda.synchronize()

        times: List[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(max(1, repeat)):
            if flush is not None:
                flush.sum()                       # 计时区外
            start.record()
            fn(ctx)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median = times[len(times) // 2]
        out = {
            "median_ms": median,
            "min_ms": times[0],
            "runs": len(times),
            "warmup": max(0, warmup),
            "bytes_moved": float(bytes_moved),
            "gb_per_s": (bytes_moved / median * 1e-6) if median > 0 else -1.0,
        }
        return out

    # ---- 输出 ------------------------------------------------------------ #

    def _emit(self, payload: Dict[str, Any]) -> None:
        print(f"{JSON_MARKER}{json.dumps(payload, ensure_ascii=False)}")
        sys.stdout.flush()

    def _bail(self, case: str, stage: str, exc: BaseException) -> int:
        self._emit({
            "case": case, "stage": stage,
            "error_name": type(exc).__name__,
            "error_str": str(exc)[:500],
            "ok": False,
        })
        return 1

    def _announce_failure(self, case: str, exc: BaseException) -> int:
        """用户代码抛异常 —— 尽量把 CUDA 错误翻译出来。"""
        text = str(exc)
        name = type(exc).__name__
        # torch 把 CUDA 错误包在 RuntimeError 里,信息形如
        # "CUDA error: an illegal memory access was encountered"
        if "illegal memory access" in text:
            name = "cudaErrorIllegalAddress"
        elif "out of memory" in text:
            name = "cudaErrorMemoryAllocation"
        elif "misaligned" in text:
            name = "cudaErrorMisalignedAddress"
        elif "invalid configuration" in text or "invalid argument" in text:
            name = "cudaErrorInvalidConfiguration"
        payload = {
            "case": case,
            "stage": "执行",
            "error_name": name,
            "error_str": text[:500],
            "ok": False,
        }
        self._emit(payload)
        if "--traceback" in sys.argv:
            traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="leetstudy PyTorch runner")
    ap.add_argument("--spec", required=True, help="题目目录(含 spec.yaml)")
    ap.add_argument("--impl", required=True, help="解答文件路径")
    ap.add_argument("--case", required=True, help="用例名")
    ap.add_argument("--perf", action="store_true", help="额外做性能计时")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        problem = load_problem(Path(args.spec))
    except Exception as exc:
        print(f"加载题目失败:{exc}", file=sys.stderr)
        return 2

    runner = Runner(problem, Path(args.impl))
    return runner.run(
        args.case,
        do_perf=args.perf,
        warmup=problem.perf.warmup if args.warmup is None else args.warmup,
        repeat=problem.perf.repeat if args.repeat is None else args.repeat,
    )


if __name__ == "__main__":
    sys.exit(main())

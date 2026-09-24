"""代码生成 —— 把 spec.yaml 变成可编译的 harness。

产物(在 build/<problem>/ 下):
  ctx.h              LaunchCtx / RefCtx 结构体,harness 与 reference.cpp 共享
  <variant>/impl.cu      #include 具体实现(template / 用户解 / baseline)
  <variant>/harness.cu   生成的 main:分配 → 填充 → 执行 → 计时 → 校验 → 输出 JSON

设计要点
--------
* 同一份 spec 生成同一个 harness,用户解与 baseline 用完全相同的计时与校验逻辑,
  保证加速比可比。
* **哨兵区**:每个缓冲前后各留 kGuardElems 个元素填哨兵值,交给 kernel 的是内区
  指针。越界写会踩哨兵,跑完立刻发现,且能分辨是往前还是往后越界。这是常驻的
  第一道防线(memcheck 也能抓,但它慢且要显式开启)。
* **毒值**:out / scratch 在每次校验执行前填毒值(浮点 NaN、整数 0x5EED5EED),
  这样「kernel 根本没写输出」会表现为清一色的毒值,而不是碰巧等于某个合法值。
* **L2 清空**:每次计时迭代前写一遍大于 L2 的缓冲,把题目数据挤出缓存。
  4090 有 72MB L2,而题目往往只有几 MB —— 不清 L2 测的是缓存带宽,加速比失真。
* launch 之后立刻检查 cudaGetLastError 与 cudaDeviceSynchronize,
  把 invalid configuration argument / illegal memory access 这类错误在源头抓出来。
* 输出用 @@LEET_JSON@@ 前缀单行打印,避免用户 kernel 里的 printf 干扰解析。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .spec import Buffer, Problem

JSON_MARKER = "@@LEET_JSON@@"

# 毒值:浮点用 NaN,整数用 0x5EED5EED 系列(避开 0 / -1 这类可能合法的值)
_POISON = {
    "f32": 'nanf("")',
    "f64": 'nan("")',
    "i32": "(int)0x5EED5EED",
    "i64": "(long long)0x5EED5EED5EED5EEDLL",
    "u32": "(unsigned int)0x5EED5EEDu",
    "u8": "(unsigned char)0x5E",
}
_FLOAT_DTYPES = ("f32", "f64")

# 各 dtype 的 C 类型 → 哨兵值构造函数(与 prelude 里的 leet_guard_value 特化对应)
_GUARD_CALL = {
    "f32": "leet_guard_value<float>()",
    "f64": "leet_guard_value<double>()",
    "i32": "leet_guard_value<int>()",
    "i64": "leet_guard_value<long long>()",
    "u32": "leet_guard_value<unsigned int>()",
    "u8": "leet_guard_value<unsigned char>()",
}

_GENERATED_HEADER = (
    "// ============================================================\n"
    "// 本文件由 leetstudy 自动生成,请勿手改。\n"
    "// 要改题目行为请改 spec.yaml / template.cu / reference.cpp / baseline.cu。\n"
    "// ============================================================\n"
)


# --------------------------------------------------------------------------- #
# ctx.h
# --------------------------------------------------------------------------- #

def render_ctx_h(problem: Problem) -> str:
    """生成 Ctx 结构体头文件。

    LaunchCtx 面向用户(device 指针 + 流),RefCtx 面向参考解(host 指针)。
    两者字段名都来自 spec 的 buffers / params,因此三方接口不会漂移。

    注意:LaunchCtx 里的指针指向「内区」,前后各有哨兵区,用户不应依赖越界行为。
    """
    out: List[str] = [_GENERATED_HEADER]
    out.append("#pragma once\n")
    out.append("#include <cstdint>\n")
    out.append("#include <cuda_runtime.h>\n\n")

    out.append("// 传给用户的 launcher。所有指针都已在 device 上,数据已就绪。\n")
    out.append("// 每个缓冲前后都有哨兵区,越界写会被检测到。\n")
    out.append("struct LaunchCtx {\n")
    for b in problem.buffers:
        out.append(f"    {b.ptr_type} {b.name};")
        out.append(f"  // {'x'.join(b.shape) or '标量'} ({b.dtype}, {b.role})\n")
    for p in problem.params:
        out.append(f"    {p.ctype} {p.name};\n")
    out.append("    cudaStream_t stream = nullptr;  // nullptr = 默认流\n")
    out.append("};\n\n")

    out.append("// 传给参考解(reference.cpp)的 host 视图。\n")
    out.append("struct RefCtx {\n")
    for b in problem.buffers:
        if b.role == "scratch":
            continue
        out.append(f"    {b.ptr_type} {b.name};")
        out.append(f"  // {'x'.join(b.shape) or '标量'} ({b.dtype}, {b.role})\n")
    for p in problem.params:
        out.append(f"    {p.ctype} {p.name};\n")
    out.append("};\n")

    return "".join(out)


# --------------------------------------------------------------------------- #
# 各种小片段
# --------------------------------------------------------------------------- #

def _fill_expr(buf: Buffer) -> str:
    """生成把 rng 的一个采样写入 buffer 元素的 C 表达式。"""
    ct = buf.ctype
    if buf.fill == "zero":
        return f"({ct})0"
    if buf.fill == "randint":
        return f"({ct})(rng.next_u64() % 100u)"
    if buf.fill == "positive":
        # (0, 1] —— 给 log / 除法等定义域为正的题用
        return f"({ct})(rng.uni() + 1e-3)"
    # uniform:[-1, 1)
    return f"({ct})(rng.uni() * 2.0 - 1.0)"


def _compare_body(buf: Buffer, atol: float, rtol: float) -> str:
    """单个 out buffer 的逐元素比较体,直接累加到外层变量(max_abs_<name> 等)。"""
    n = buf.name
    if buf.dtype in _FLOAT_DTYPES:
        # 两边都是 NaN 视为相等(有些题合法地产生 NaN);只有一边是 NaN 则算错。
        return f"""        bool both_nan = (std::isnan(got) && std::isnan(exp));
        bool one_nan  = (std::isnan(got) != std::isnan(exp));
        if (std::isnan(got)) ++nan_{n};
        else if (std::isinf(got)) ++inf_{n};
        double d = std::fabs(got - exp);
        double tol = {atol!r} + {rtol!r} * std::fabs(exp);
        double rel = d / (std::fabs(exp) > 1e-30 ? std::fabs(exp) : 1e-30);
        if (!both_nan) {{
            if (d > max_abs_{n}) max_abs_{n} = d;
            if (rel > max_rel_{n}) max_rel_{n} = rel;
        }}
        if (!(both_nan || (!one_nan && d <= tol))) {{
            ++bad_{n};
            if (first_bad_{n} < 0) {{ first_bad_{n} = i; first_got_{n} = got; first_exp_{n} = exp; }}
        }}
"""
    # 整数类型要求精确相等(atol/rtol 对整数无意义)
    return f"""        long long di = (long long)got - (long long)exp;
        double ad = (double)(di < 0 ? -di : di);
        if (ad > max_abs_{n}) max_abs_{n} = ad;
        if (di != 0) {{
            ++bad_{n};
            if (first_bad_{n} < 0) {{ first_bad_{n} = i; first_got_{n} = got; first_exp_{n} = exp; }}
        }}
"""


def _ok_expr(problem: Problem) -> str:
    conds = [f"bad_{b.name} == 0" for b in problem.outputs]
    conds += [f"gf_{b.name} == 0 && gb_{b.name} == 0" for b in problem.buffers]
    return "(" + " && ".join(conds) + ")" if conds else "true"


def _bytes_moved_expr(problem: Problem) -> str:
    """本题一次执行至少要搬动的字节数(in 读 + out 写)。

    这是有效带宽的「下限」:好的实现可能读得更少(如用 shared memory 复用),
    所以按它算出的带宽占比是保守估计,不会虚高。
    """
    terms = [f"sizeof({b.ctype}) * (size_t)leet_n_{b.name}"
             for b in problem.inputs + problem.outputs]
    return " + ".join(terms) if terms else "0"


# --------------------------------------------------------------------------- #
# harness.cu
# --------------------------------------------------------------------------- #

_HARNESS_PRELUDE = """#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "ctx.h"

// 具体实现:用户解答 / 模板 / baseline
#include "@IMPL@"

// 由 reference.cpp 提供(单独编译单元)
void reference(RefCtx& ctx);

#define LEET_JSON "@MARKER@"

// ------------------------------------------------------------------
// 可复现随机数:同一个 case 名 → 同一组输入,保证用户与 baseline 看到相同数据
// ------------------------------------------------------------------
struct LeetRng {
    uint64_t s;
    explicit LeetRng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next_u64() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
    double uni() { return (double)(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
};

static uint64_t leet_seed_of(const char* name) {
    uint64_t h = 1469598103934665603ULL;               // FNV-1a
    for (const char* p = name; *p; ++p) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
    return h;
}

// JSON 数字:非有限值输出 null(JSON 不允许 NaN/Inf 字面量)
static void leet_jnum(char* buf, size_t cap, double v) {
    if (std::isnan(v) || std::isinf(v)) std::snprintf(buf, cap, "null");
    else std::snprintf(buf, cap, "%.9g", v);
}

// ------------------------------------------------------------------
// 哨兵区
// 每个缓冲前后各留 kGuardElems 个元素填哨兵值,交给 kernel 的是内区指针。
// 越界写会踩到哨兵,跑完立刻能发现,并区分是「往前越界」还是「往后越界」——
// 这比只报一句「结果不对」有用得多。memcheck 也能抓越界,但它慢且要显式开启,
// 哨兵区是常驻的第一道防线。
// ------------------------------------------------------------------
static const int64_t kGuardElems = 4096;

template <typename T> static T leet_guard_value();
template <> float leet_guard_value<float>() { return -1.2345678e33f; }
template <> double leet_guard_value<double>() { return -1.234567890123e300; }
template <> int leet_guard_value<int>() { return (int)0x600D600D; }
template <> long long leet_guard_value<long long>() { return (long long)0x600D600D600D600DLL; }
template <> unsigned int leet_guard_value<unsigned int>() { return (unsigned int)0x600D600Du; }
template <> unsigned char leet_guard_value<unsigned char>() { return (unsigned char)0x6D; }

template <typename T>
static void leet_guard_check(const T* gdev, int64_t interior,
                             long long* front_hits, long long* back_hits) {
    *front_hits = 0; *back_hits = 0;
    const T sentinel = leet_guard_value<T>();
    std::vector<T> hf((size_t)kGuardElems), hb((size_t)kGuardElems);
    if (cudaMemcpy(hf.data(), gdev, sizeof(T) * (size_t)kGuardElems,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return;
    if (cudaMemcpy(hb.data(), gdev + kGuardElems + interior, sizeof(T) * (size_t)kGuardElems,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return;
    for (int64_t i = 0; i < kGuardElems; ++i) if (hf[i] != sentinel) ++(*front_hits);
    for (int64_t i = 0; i < kGuardElems; ++i) if (hb[i] != sentinel) ++(*back_hits);
}

// 出错的统一出口:仍然打印 JSON,让上层能给出可读诊断
static int leet_bail(const char* case_name, const char* stage, cudaError_t err) {
    std::printf("%s{\\"case\\":\\"%s\\",\\"stage\\":\\"%s\\",\\"error_name\\":\\"%s\\","
                "\\"error_str\\":\\"%s\\",\\"ok\\":false}\\n",
                LEET_JSON, case_name, stage, cudaGetErrorName(err), cudaGetErrorString(err));
    std::fflush(stdout);
    return 1;
}

// ------------------------------------------------------------------
// L2 清空
// 用「读」而不是「写」来驱逐缓存:写会在 L2 里留下脏行,被计时 kernel 一开始
// 就得先写回这些脏行才能腾地方 —— 那份额外访存会污染测量,让结果偏慢,
// 而且偏慢的程度随 kernel 而异,数字不可比。读进来的都是干净行,驱逐无代价。
// ------------------------------------------------------------------
__global__ void leet_flush_kernel(const int* __restrict__ buf, long long n, int* __restrict__ sink) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    int acc = 0;
    for (; i < n; i += stride) acc += buf[i];
    if (acc == 0x7FFFFFFF) *sink = acc;    // 防止整个循环被优化掉
}
"""


def render_harness_cu(problem: Problem, impl_relpath: str = "impl.cu") -> str:
    """生成 harness 的 main。impl_relpath 是相对本文件的实现文件路径。"""
    atol = problem.verify.atol
    rtol = problem.verify.rtol
    out_bufs = problem.outputs
    p: List[str] = [_GENERATED_HEADER]
    p.append(
        _HARNESS_PRELUDE.replace("@IMPL@", impl_relpath).replace("@MARKER@", JSON_MARKER)
    )

    # ---- 参数表 ----
    # 一律用 double 存:既能容纳整数参数(缓冲大小),也能容纳浮点参数
    # (alpha / eps / temperature 之类)。生成时会按声明的类型转换。
    p.append("\n// ---------------- 参数表(来自 spec.cases) ----------------\n")
    p.append("struct LeetParams {\n")
    for prm in problem.params:
        p.append(f"    double {prm.name};\n")
    p.append("};\n\n")
    p.append("static const char* kCaseNames[] = {"
             + ", ".join(f'"{c.name}"' for c in problem.cases) + "};\n")
    p.append("static const LeetParams kCases[] = {\n    " + ",\n    ".join(
        "{" + ", ".join(repr(float(c.params[prm.name])) for prm in problem.params) + "}"
        for c in problem.cases
    ) + "\n};\n")
    p.append(f"static const int kNumCases = {len(problem.cases)};\n")

    # ---- main 开头与参数解析 ----
    p.append(f"""
int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::fprintf(stderr, "用法: %s <case> [--perf] [--warmup N] [--repeat N]\\n", argv[0]);
        std::fprintf(stderr, "可用用例: ");
        for (int i = 0; i < kNumCases; ++i) std::fprintf(stderr, "%s ", kCaseNames[i]);
        std::fprintf(stderr, "\\n");
        return 2;
    }}
    const char* case_name = argv[1];
    bool do_perf = false;
    int warmup = {problem.perf.warmup};
    int perf_repeat = {problem.perf.repeat};
    for (int i = 2; i < argc; ++i) {{
        if (!std::strcmp(argv[i], "--perf")) do_perf = true;
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) perf_repeat = std::atoi(argv[++i]);
    }}

    const LeetParams* cp = nullptr;
    for (int i = 0; i < kNumCases; ++i)
        if (!std::strcmp(kCaseNames[i], case_name)) {{ cp = &kCases[i]; break; }}
    if (!cp) {{ std::fprintf(stderr, "未知用例: %s\\n", case_name); return 2; }}
    const LeetParams& C = *cp;
""")
    for prm in problem.params:
        p.append(f"    {prm.ctype} {prm.name} = ({prm.ctype})C.{prm.name};\n")

    # ---- host 缓冲 ----
    p.append("\n    // ---------------- host 侧准备 ----------------\n")
    for b in problem.buffers:
        p.append(f"    int64_t leet_n_{b.name} = {b.count_expr()};\n")
    for b in problem.inputs + problem.scratches:
        p.append(f"    {b.ctype}* h_{b.name} = ({b.ctype}*)std::malloc("
                 f"sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")
    for b in out_bufs:
        # h_* 接收 device 回传的实际结果,e_* 放参考解的期望值
        p.append(f"    {b.ctype}* h_{b.name} = ({b.ctype}*)std::malloc("
                 f"sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")
        p.append(f"    {b.ctype}* e_{b.name} = ({b.ctype}*)std::malloc("
                 f"sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")

    # 输入填充
    for i, b in enumerate(problem.inputs):
        p.append(f"    {{ LeetRng rng(leet_seed_of(case_name) ^ {i + 1}ULL);\n"
                 f"      for (int64_t j = 0; j < leet_n_{b.name}; ++j)"
                 f" h_{b.name}[j] = {_fill_expr(b)}; }}\n")
    for b in problem.scratches:
        p.append(f"    std::memset(h_{b.name}, 0, sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")
    for b in out_bufs:
        p.append(f"    std::memset(e_{b.name}, 0, sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")

    # ---- 参考解 ----
    p.append("\n    // ---------------- 参考解(ground truth) ----------------\n")
    p.append("    RefCtx rctx{};\n")
    for b in problem.inputs:
        p.append(f"    rctx.{b.name} = h_{b.name};\n")
    for b in out_bufs:
        p.append(f"    rctx.{b.name} = e_{b.name};\n")
    for prm in problem.params:
        p.append(f"    rctx.{prm.name} = {prm.name};\n")
    p.append("    reference(rctx);\n")

    # ---- device 缓冲(带哨兵区) ----
    p.append("\n    // ---------------- device 侧分配与上传(含哨兵区) ----------------\n")
    for b in problem.buffers:
        ct = b.ctype
        p.append(
            f"    {ct}* g_{b.name} = nullptr;   // 含前后哨兵区的完整分配\n"
            f"    if (cudaMalloc((void**)&g_{b.name}, sizeof({ct}) *"
            f" (size_t)(leet_n_{b.name} + 2 * kGuardElems)) != cudaSuccess)"
            f" return leet_bail(case_name, \"cudaMalloc({b.name})\", cudaGetLastError());\n"
            f"    {ct}* d_{b.name} = g_{b.name} + kGuardElems;   // 交给 kernel 的内区\n"
        )

    for b in problem.inputs:
        ct = b.ctype
        p.append(
            f"    {{ std::vector<{ct}> tmp((size_t)(leet_n_{b.name} + 2 * kGuardElems),"
            f" {_GUARD_CALL[b.dtype]});\n"
            f"      std::memcpy(tmp.data() + kGuardElems, h_{b.name}, sizeof({ct}) * (size_t)leet_n_{b.name});\n"
            f"      if (cudaMemcpy(g_{b.name}, tmp.data(), sizeof({ct}) *"
            f" (size_t)(leet_n_{b.name} + 2 * kGuardElems), cudaMemcpyHostToDevice) != cudaSuccess)"
            f" return leet_bail(case_name, \"H2D({b.name})\", cudaGetLastError()); }}\n"
        )

    # ---- LaunchCtx ----
    p.append("\n    LaunchCtx lctx{};\n")
    for b in problem.buffers:
        p.append(f"    lctx.{b.name} = d_{b.name};\n")
    for prm in problem.params:
        p.append(f"    lctx.{prm.name} = {prm.name};\n")
    p.append("    lctx.stream = nullptr;\n")

    # ---- 毒化 + 一次干净执行 ----
    p.append("\n    // ---------------- 正确性:一次干净的执行 ----------------\n")
    p.append("    // out / scratch 填毒值,让「没写输出」表现为清一色的毒值\n")
    for b in out_bufs + problem.scratches:
        ct = b.ctype
        p.append(
            f"    {{ std::vector<{ct}> tmp((size_t)(leet_n_{b.name} + 2 * kGuardElems),"
            f" {_GUARD_CALL[b.dtype]});\n"
            f"      for (int64_t j = 0; j < leet_n_{b.name}; ++j) tmp[kGuardElems + j] = {_POISON[b.dtype]};\n"
            f"      cudaMemcpy(g_{b.name}, tmp.data(), sizeof({ct}) *"
            f" (size_t)(leet_n_{b.name} + 2 * kGuardElems), cudaMemcpyHostToDevice); }}\n"
        )

    p.append("""
    (void)cudaGetLastError();          // 清空历史错误,避免误判成本次的
    cudaEvent_t ev_a, ev_b;
    cudaEventCreate(&ev_a); cudaEventCreate(&ev_b);

    @LAUNCHER@(lctx);
    cudaError_t e_launch = cudaGetLastError();
    if (e_launch != cudaSuccess) return leet_bail(case_name, "launch", e_launch);
    cudaError_t e_sync = cudaDeviceSynchronize();
    if (e_sync != cudaSuccess) return leet_bail(case_name, "执行中(异步错误)", e_sync);
""".replace("@LAUNCHER@", problem.launcher))
    for b in out_bufs:
        p.append(f"    if (cudaMemcpy(h_{b.name}, d_{b.name}, sizeof({b.ctype}) * (size_t)leet_n_{b.name},"
                 f" cudaMemcpyDeviceToHost) != cudaSuccess)"
                 f" return leet_bail(case_name, \"D2H({b.name})\", cudaGetLastError());\n")

    # ---- 哨兵检查 ----
    p.append("\n    // ---------------- 哨兵检查(越界写) ----------------\n")
    for b in problem.buffers:
        p.append(f"    long long gf_{b.name} = 0, gb_{b.name} = 0;\n")
        p.append(f"    leet_guard_check<{b.ctype}>(g_{b.name}, leet_n_{b.name},"
                 f" &gf_{b.name}, &gb_{b.name});\n")

    # ---- 逐输出校验 ----
    p.append("\n    // ---------------- 逐输出校验 ----------------\n")
    for b in out_bufs:
        p.append(f"""
    double max_abs_{b.name} = 0.0, max_rel_{b.name} = 0.0;
    long long bad_{b.name} = 0, first_bad_{b.name} = -1, nan_{b.name} = 0, inf_{b.name} = 0;
    double first_got_{b.name} = std::nan(""), first_exp_{b.name} = std::nan("");
    for (int64_t i = 0; i < leet_n_{b.name}; ++i) {{
        double got = (double)h_{b.name}[i];
        double exp = (double)e_{b.name}[i];
{_compare_body(b, atol, rtol)}    }}
""")

    # ---- 性能 ----
    p.append(f"""
    // ---------------- 性能 ----------------
    double median_ms = -1.0, min_ms = -1.0, gb_per_s = -1.0;
    double bytes_moved = (double)({_bytes_moved_expr(problem)});
    if (do_perf) {{
        // 清 L2:读一遍大于 L2 的缓冲,把题目数据挤出缓存。
        // 不这样做的话,几 MB 的题目会整个驻留 L2(4090 有 72MB),
        // 测出来的是缓存带宽而不是显存带宽,数字严重失真。
        int* flush_buf = nullptr;
        int* flush_sink = nullptr;
        long long flush_ints = 0;
        if ({'true' if problem.perf.flush_l2 else 'false'}) {{
            int l2_bytes = 0;
            cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
            if (l2_bytes <= 0) l2_bytes = 32 * 1024 * 1024;
            // 2×L2 确保把题目数据彻底挤出去
            flush_ints = (long long)((size_t)l2_bytes * 2 / sizeof(int));
            if (cudaMalloc((void**)&flush_buf, (size_t)flush_ints * sizeof(int)) != cudaSuccess) {{
                flush_buf = nullptr;
            }} else if (cudaMemset(flush_buf, 1, (size_t)flush_ints * sizeof(int)) != cudaSuccess) {{
                cudaFree(flush_buf); flush_buf = nullptr;
            }} else if (cudaMalloc((void**)&flush_sink, sizeof(int)) != cudaSuccess) {{
                flush_sink = nullptr;
            }}
        }}
        // 读满整个 flush 缓冲,用干净行把 L2 里的题目数据替换掉。
        // 与用户 kernel 同流,天然串行;计时事件记录在其后,不计入开销。
        auto leet_flush_l2 = [&]() {{
            if (!flush_buf) return;
            int blocks = 1024;
            leet_flush_kernel<<<blocks, 256>>>(flush_buf, flush_ints, flush_sink);
        }};

        for (int i = 0; i < warmup; ++i) {{
            leet_flush_l2();
            {problem.launcher}(lctx);
        }}
        cudaError_t e_warm = cudaDeviceSynchronize();
        if (e_warm != cudaSuccess) return leet_bail(case_name, "预热", e_warm);

        std::vector<double> ts;
        ts.reserve((size_t)perf_repeat);
        for (int i = 0; i < perf_repeat; ++i) {{
            leet_flush_l2();                                        // 计时区外
            cudaEventRecord(ev_a, nullptr);
            {problem.launcher}(lctx);
            cudaEventRecord(ev_b, nullptr);
            cudaEventSynchronize(ev_b);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev_a, ev_b);
            cudaError_t e_t = cudaGetLastError();
            if (e_t != cudaSuccess) return leet_bail(case_name, "计时循环", e_t);
            ts.push_back((double)ms);
        }}
        std::sort(ts.begin(), ts.end());
        min_ms = ts.front();
        median_ms = ts[ts.size() / 2];
        if (median_ms > 0.0) gb_per_s = bytes_moved / median_ms * 1e-6;
        if (flush_buf) cudaFree(flush_buf);
        if (flush_sink) cudaFree(flush_sink);
    }}
    cudaEventDestroy(ev_a); cudaEventDestroy(ev_b);
""")

    # ---- 输出 JSON ----
    p.append("\n    // ---------------- 结果(单行 JSON,前缀标记便于解析) ----------------\n")
    p.append('    std::printf("%s{", LEET_JSON);\n')
    p.append('    std::printf("\\"case\\":\\"%s\\"", case_name);\n')
    p.append('    std::printf(",\\"params\\":{");\n')
    for i, prm in enumerate(problem.params):
        comma = "" if i == 0 else ", "
        p.append(f'    std::printf("{comma}\\"{prm.name}\\":");'
                 f' {{ char nb[64]; leet_jnum(nb, sizeof(nb), (double){prm.name});'
                 f' std::printf("%s", nb); }}\n')
    p.append('    std::printf("}");\n')
    p.append('    std::printf(",\\"outputs\\":[");\n')
    for i, b in enumerate(out_bufs):
        comma = "" if i == 0 else ", "
        p.append(f"""
    {{
        char nb[64];
        std::printf("{comma}{{\\"name\\":\\"{b.name}\\",\\"count\\":%lld,\\"bad\\":%lld,",
                    (long long)leet_n_{b.name}, bad_{b.name});
        std::printf("\\"max_abs_err\\":%.9g,\\"max_rel_err\\":%.9g,", max_abs_{b.name}, max_rel_{b.name});
        std::printf("\\"first_bad_idx\\":%lld,", first_bad_{b.name});
        leet_jnum(nb, sizeof(nb), first_got_{b.name}); std::printf("\\"first_got\\":%s,", nb);
        leet_jnum(nb, sizeof(nb), first_exp_{b.name}); std::printf("\\"first_exp\\":%s,", nb);
        std::printf("\\"nan_count\\":%lld,\\"inf_count\\":%lld}}", nan_{b.name}, inf_{b.name});
    }}
""")
    p.append('    std::printf("]");\n')
    p.append('    std::printf(",\\"guards\\":{");\n')
    for i, b in enumerate(problem.buffers):
        comma = "" if i == 0 else ", "
        p.append(f'    std::printf("{comma}\\"{b.name}\\":{{\\"front\\":%lld,\\"back\\":%lld}}",'
                 f' gf_{b.name}, gb_{b.name});\n')
    p.append('    std::printf("}");\n')
    p.append(f'    std::printf(",\\"ok\\":%s", ({_ok_expr(problem)}) ? "true" : "false");\n')
    p.append('    if (do_perf) std::printf(",\\"perf\\":{\\"median_ms\\":%.6f,\\"min_ms\\":%.6f,'
             '\\"runs\\":%d,\\"warmup\\":%d,\\"bytes_moved\\":%.0f,\\"gb_per_s\\":%.3f}",'
             ' median_ms, min_ms, perf_repeat, warmup, bytes_moved, gb_per_s);\n')
    p.append('    std::printf("}\\n");\n')
    p.append("    std::fflush(stdout);\n")

    # ---- 收尾 ----
    p.append("\n    // ---------------- 释放 ----------------\n")
    for b in problem.buffers:
        p.append(f"    cudaFree(g_{b.name});\n")
    for b in problem.inputs + problem.scratches + out_bufs:
        p.append(f"    std::free(h_{b.name});\n")
    for b in out_bufs:
        p.append(f"    std::free(e_{b.name});\n")
    p.append("    return 0;\n}\n")

    return "".join(p)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #

def render_impl_cu(impl_path: Path) -> str:
    """生成 impl.cu:把具体实现拉进来。"""
    return (
        "// 自动生成:把具体实现包含进来(用户解答 / 模板 / baseline)\n"
        f'#include "{impl_path}"\n'
    )


def write_ctx_h(problem: Problem, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ctx.h"
    path.write_text(render_ctx_h(problem), encoding="utf-8")
    return path


def write_variant(problem: Problem, out_dir: Path, impl_path: Path) -> Dict[str, Path]:
    """为一个实现变体生成 harness 与 impl,返回 {impl, harness} 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    impl_cu = out_dir / "impl.cu"
    harness_cu = out_dir / "harness.cu"
    impl_cu.write_text(render_impl_cu(impl_path), encoding="utf-8")
    harness_cu.write_text(render_harness_cu(problem), encoding="utf-8")
    return {"impl": impl_cu, "harness": harness_cu}

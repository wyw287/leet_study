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
from .subjects.base import JSON_MARKER

# 兼容旧引用:早期 JSON_MARKER 定义在本模块里
__all__ = ["JSON_MARKER"]

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


def _compare_body(buf: Buffer, atol: float, rtol: float,
                  indent: str = "        ") -> str:
    """单个 out buffer 的逐元素比较体,直接累加到外层变量(max_abs_<name> 等)。"""
    n = buf.name
    if buf.dtype in _FLOAT_DTYPES:
        # 两边都是 NaN 视为相等(有些题合法地产生 NaN);只有一边是 NaN 则算错。
        body = f"""bool both_nan = (std::isnan(got) && std::isnan(exp));
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
    else:
        # 整数类型要求精确相等(atol/rtol 对整数无意义)
        body = f"""long long di = (long long)got - (long long)exp;
double ad = (double)(di < 0 ? -di : di);
if (ad > max_abs_{n}) max_abs_{n} = ad;
if (di != 0) {{
    ++bad_{n};
    if (first_bad_{n} < 0) {{ first_bad_{n} = i; first_got_{n} = got; first_exp_{n} = exp; }}
}}
"""
    return "".join(indent + line if line.strip() else line
                   for line in body.splitlines(keepends=True))


def _ok_expr(problem: Problem) -> str:
    """判「通过」的条件:每个稳定性重复都过 + 每个输出的数值都对 + 没有越界写。"""
    conds = ["verify_pass == verify_repeat"]
    conds += [f"bad_{b.name} == 0" for b in problem.outputs]
    conds += [f"gf_{b.name} == 0 && gb_{b.name} == 0" for b in problem.buffers]
    return "(" + " && ".join(conds) + ")"


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

// 出错的统一出口:打印一行错误 JSON。**不返回** —— 调用方在 lambda 里,
// 用 LEET_FAIL 宏做 return。
static void leet_bail(const char* case_name, const char* stage, cudaError_t err) {
    std::printf("%s{\\"case\\":\\"%s\\",\\"stage\\":\\"%s\\",\\"error_name\\":\\"%s\\","
                "\\"error_str\\":\\"%s\\",\\"ok\\":false}\\n",
                LEET_JSON, case_name, stage, cudaGetErrorName(err), cudaGetErrorString(err));
    std::fflush(stdout);
}

// 出错即打印并跳出当前用例(用例体是 lambda,所以 return 是合法的)
#define LEET_FAIL(stage, err) do { leet_bail(case_name, stage, err); return; } while (0)

// device 缓冲的 RAII 包装:任何提前 return 都会释放,不会把显存漏给下一个用例
struct LeetDev {
    void* p = nullptr;
    ~LeetDev() { if (p) cudaFree(p); }
    LeetDev(const LeetDev&) = delete;
    LeetDev& operator=(const LeetDev&) = delete;
    LeetDev() = default;
};

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


def _cpp(template: str, **kw) -> str:
    """填充 C++ 模板。

    刻意用 @@X@@ 占位而不是 f-string:C++ 代码里全是花括号,在 f-string 里
    要逐个写成 {{ }} —— 既难读又极易漏。占位符没有这个问题。
    """
    for key, val in kw.items():
        template = template.replace(f"@@{key}@@", str(val))
    return template


def _indent_block(text: str, indent: str) -> str:
    """给整段 C++ 加上缩进。

    注意要保留末尾换行 —— splitlines() 会把它吃掉,导致后面拼上来的代码
    被粘在同一行上(这个坑真实踩过:生成的代码里注释和语句连成了一行)。
    """
    out = "\n".join(indent + ln if ln.strip() else ln for ln in text.splitlines())
    if text.endswith("\n"):
        out += "\n"
    return out


def render_harness_cu(problem: Problem, impl_relpath: str = "impl.cu") -> str:
    """生成 harness 的 main。

    **一个进程跑完所有用例**,而不是每个用例起一个进程。这不是微优化:
    本机实测 CUDA 上下文初始化要 4.4 秒(空程序 `cudaFree(0)` 亦然),
    而 kernel 本身往往只跑一百多微秒 —— 若每用例一个进程,进程启动开销会占掉
    判题总时间的 90% 以上。所以:
      * 默认跑全部用例(可用 --case <名字> 只跑一个,sanitizer 需要)
      * 稳定性重复(--verify-repeat)也在进程内循环,而不是重复起进程
      * L2 清空缓冲整个进程只分配一次

    错误处理:每个用例体是一个 lambda,出错时打印错误 JSON 并 return,然后继续
    下一个用例 —— 一个用例挂掉不该让其余用例的结果全部丢失。
    """
    atol = problem.verify.atol
    rtol = problem.verify.rtol
    out_bufs = problem.outputs
    poison_targets = out_bufs + problem.scratches
    ind = "            "      # lambda 体内的缩进(12 空格)

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

    # ---- main:参数解析 + 用例循环 + flush 缓冲 ----
    p.append(_cpp("""
int main(int argc, char** argv) {
    const char* case_sel = "all";      // "all" = 跑全部用例
    bool do_perf = false;
    int warmup = @@WARMUP@@;
    int perf_repeat = @@PERF_REPEAT@@;
    int verify_repeat = @@VERIFY_REPEAT@@;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--perf")) do_perf = true;
        else if (!std::strcmp(argv[i], "--case") && i + 1 < argc) case_sel = argv[++i];
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) perf_repeat = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verify-repeat") && i + 1 < argc) verify_repeat = std::atoi(argv[++i]);
        else if (argv[i][0] != '-') case_sel = argv[i];
    }

    std::vector<int> todo;
    for (int ci = 0; ci < kNumCases; ++ci)
        if (!std::strcmp(case_sel, "all") || !std::strcmp(kCaseNames[ci], case_sel))
            todo.push_back(ci);
    if (todo.empty()) {
        std::fprintf(stderr, "未知用例: %s(可用: all", case_sel);
        for (int i = 0; i < kNumCases; ++i) std::fprintf(stderr, " %s", kCaseNames[i]);
        std::fprintf(stderr, ")\\n");
        return 2;
    }

    // 清 L2 的缓冲:整个进程只分配一次,所有用例复用
    int* flush_buf = nullptr;
    int* flush_sink = nullptr;
    long long flush_ints = 0;
    if (@@FLUSH@@) {
        int l2_bytes = 0;
        cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
        if (l2_bytes <= 0) l2_bytes = 32 * 1024 * 1024;
        flush_ints = (long long)((size_t)l2_bytes * 2 / sizeof(int));
        if (cudaMalloc((void**)&flush_buf, (size_t)flush_ints * sizeof(int)) != cudaSuccess) {
            flush_buf = nullptr;
        } else if (cudaMemset(flush_buf, 1, (size_t)flush_ints * sizeof(int)) != cudaSuccess) {
            cudaFree(flush_buf); flush_buf = nullptr;
        } else if (cudaMalloc((void**)&flush_sink, sizeof(int)) != cudaSuccess) {
            flush_sink = nullptr;
        }
    }
    // 读满整个 flush 缓冲,用干净行把 L2 里的题目数据替换掉。
    // 与用户 kernel 同流,天然串行;计时事件记录在其后,不计入开销。
    auto leet_flush_l2 = [&]() {
        if (flush_buf) leet_flush_kernel<<<1024, 256>>>(flush_buf, flush_ints, flush_sink);
    };

    for (int ci : todo) {
        const char* case_name = kCaseNames[ci];
        const LeetParams& C = kCases[ci];

        // 单个用例的全部流程。包成 lambda 是为了能用 return 早退,
        // 同时让 RAII(LeetDev / std::vector)负责释放 —— 不把显存漏给下一个用例。
        auto leet_one_case = [&]() {
""", WARMUP=problem.perf.warmup, PERF_REPEAT=problem.perf.repeat,
        VERIFY_REPEAT=max(1, problem.verify.repeat),
        FLUSH="true" if problem.perf.flush_l2 else "false"))

    # 参数局部变量
    for prm in problem.params:
        p.append(f"{ind}{prm.ctype} {prm.name} = ({prm.ctype})C.{prm.name};\n")

    # ---- host 缓冲(std::vector:免手动释放,早退也不漏) ----
    p.append(f"\n{ind}// ---------------- host 侧准备 ----------------\n")
    for b in problem.buffers:
        p.append(f"{ind}int64_t leet_n_{b.name} = {b.count_expr()};\n")
    for b in problem.inputs + problem.scratches:
        p.append(f"{ind}std::vector<{b.ctype}> h_{b.name}((size_t)leet_n_{b.name});\n")
    for b in out_bufs:
        p.append(f"{ind}std::vector<{b.ctype}> h_{b.name}((size_t)leet_n_{b.name});\n")
        p.append(f"{ind}std::vector<{b.ctype}> e_{b.name}((size_t)leet_n_{b.name});\n")

    for i, b in enumerate(problem.inputs):
        p.append(f"{ind}{{ LeetRng rng(leet_seed_of(case_name) ^ {i + 1}ULL);\n"
                 f"{ind}  for (int64_t j = 0; j < leet_n_{b.name}; ++j)"
                 f" h_{b.name}[j] = {_fill_expr(b)}; }}\n")
    for b in problem.scratches:
        p.append(f"{ind}std::memset(h_{b.name}.data(), 0,"
                 f" sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")
    for b in out_bufs:
        p.append(f"{ind}std::memset(e_{b.name}.data(), 0,"
                 f" sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")

    # ---- 参考解 ----
    p.append(f"\n{ind}// ---------------- 参考解(ground truth) ----------------\n")
    p.append(f"{ind}RefCtx rctx{{}};\n")
    for b in problem.inputs:
        p.append(f"{ind}rctx.{b.name} = h_{b.name}.data();\n")
    for b in out_bufs:
        p.append(f"{ind}rctx.{b.name} = e_{b.name}.data();\n")
    for prm in problem.params:
        p.append(f"{ind}rctx.{prm.name} = {prm.name};\n")
    p.append(f"{ind}reference(rctx);\n")

    # ---- device 缓冲(RAII + 哨兵区) ----
    p.append(f"\n{ind}// ---------------- device 侧分配与上传(含哨兵区) ----------------\n")
    for b in problem.buffers:
        p.append(
            f"{ind}LeetDev g_{b.name};   // 含前后哨兵区的完整分配\n"
            f"{ind}if (cudaMalloc(&g_{b.name}.p, sizeof({b.ctype}) *"
            f" (size_t)(leet_n_{b.name} + 2 * kGuardElems)) != cudaSuccess)"
            f" LEET_FAIL(\"cudaMalloc({b.name})\", cudaGetLastError());\n"
            f"{ind}{b.ctype}* d_{b.name} = ({b.ctype}*)g_{b.name}.p + kGuardElems;"
            f"   // 交给 kernel 的内区\n"
        )

    for b in problem.inputs:
        ct = b.ctype
        p.append(
            f"{ind}{{ std::vector<{ct}> tmp((size_t)(leet_n_{b.name} + 2 * kGuardElems),"
            f" {_GUARD_CALL[b.dtype]});\n"
            f"{ind}  std::memcpy(tmp.data() + kGuardElems, h_{b.name}.data(),"
            f" sizeof({ct}) * (size_t)leet_n_{b.name});\n"
            f"{ind}  if (cudaMemcpy(g_{b.name}.p, tmp.data(), sizeof({ct}) *"
            f" (size_t)(leet_n_{b.name} + 2 * kGuardElems), cudaMemcpyHostToDevice)"
            f" != cudaSuccess) LEET_FAIL(\"H2D({b.name})\", cudaGetLastError()); }}\n"
        )

    # ---- LaunchCtx ----
    p.append(f"\n{ind}LaunchCtx lctx{{}};\n")
    for b in problem.buffers:
        p.append(f"{ind}lctx.{b.name} = d_{b.name};\n")
    for prm in problem.params:
        p.append(f"{ind}lctx.{prm.name} = {prm.name};\n")
    p.append(f"{ind}lctx.stream = nullptr;\n")

    # ---- 统计量 ----
    p.append(f"\n{ind}// ---------------- 统计量(跨稳定性重复累计/覆盖) ----------------\n")
    for b in problem.buffers:
        p.append(f"{ind}long long gf_{b.name} = 0, gb_{b.name} = 0;"
                 f"   // 哨兵:任一次踩到都累加\n")
    for b in out_bufs:
        p.append(f"{ind}double max_abs_{b.name} = 0.0, max_rel_{b.name} = 0.0;\n")
        p.append(f"{ind}long long bad_{b.name} = 0, first_bad_{b.name} = -1,"
                 f" nan_{b.name} = 0, inf_{b.name} = 0;\n")
        p.append(f"{ind}double first_got_{b.name} = std::nan(\"\"),"
                 f" first_exp_{b.name} = std::nan(\"\");\n")
    p.append(f"{ind}long long verify_pass = 0;\n")

    # ---- 稳定性循环 ----
    p.append(_indent_block("""
// ---------------- 正确性:重复 @@N@@ 次 ----------------
// 同一份输入反复跑,任何一次结果不对都算失败 —— 这是抓竞态 / 未初始化内存
// 这类「有时对有时错」问题的基本手段。
cudaEvent_t ev_a, ev_b;
cudaEventCreate(&ev_a); cudaEventCreate(&ev_b);

for (int vr = 0; vr < verify_repeat; ++vr) {
    // out / scratch 填毒值:让「没写输出」表现为清一色的毒值
""".replace("@@N@@", str(max(1, problem.verify.repeat))), ind))
    for b in poison_targets:
        ct = b.ctype
        p.append(
            f"{ind}    {{ std::vector<{ct}> tmp((size_t)(leet_n_{b.name} + 2 * kGuardElems),"
            f" {_GUARD_CALL[b.dtype]});\n"
            f"{ind}      for (int64_t j = 0; j < leet_n_{b.name}; ++j)"
            f" tmp[kGuardElems + j] = {_POISON[b.dtype]};\n"
            f"{ind}      cudaMemcpy(g_{b.name}.p, tmp.data(), sizeof({ct}) *"
            f" (size_t)(leet_n_{b.name} + 2 * kGuardElems), cudaMemcpyHostToDevice); }}\n"
        )
    p.append(_indent_block("""
(void)cudaGetLastError();      // 清空历史错误,避免误判成本次的
@@LAUNCHER@@(lctx);
cudaError_t e_launch = cudaGetLastError();
if (e_launch != cudaSuccess) LEET_FAIL("launch", e_launch);
cudaError_t e_sync = cudaDeviceSynchronize();
if (e_sync != cudaSuccess) LEET_FAIL("执行中(异步错误)", e_sync);
""".replace("@@LAUNCHER@@", problem.launcher), ind + "    "))
    for b in out_bufs:
        p.append(f"{ind}    if (cudaMemcpy(h_{b.name}.data(), d_{b.name},"
                 f" sizeof({b.ctype}) * (size_t)leet_n_{b.name}, cudaMemcpyDeviceToHost)"
                 f" != cudaSuccess) LEET_FAIL(\"D2H({b.name})\", cudaGetLastError());\n")

    # 哨兵检查(跨重复累计)
    p.append(f"\n{ind}    // 哨兵检查(越界写):跨重复累计\n")
    for b in problem.buffers:
        p.append(f"{ind}    {{ long long f = 0, b2 = 0;"
                 f" leet_guard_check<{b.ctype}>((const {b.ctype}*)g_{b.name}.p,"
                 f" leet_n_{b.name}, &f, &b2);"
                 f" gf_{b.name} += f; gb_{b.name} += b2; }}\n")

    # 逐输出比对(每轮重置)
    p.append(f"\n{ind}    // 逐输出校验(最后一轮的结果会被报告出去)\n")
    for b in out_bufs:
        p.append(f"\n{ind}    max_abs_{b.name} = 0.0; max_rel_{b.name} = 0.0;\n"
                 f"{ind}    bad_{b.name} = 0; first_bad_{b.name} = -1;"
                 f" nan_{b.name} = 0; inf_{b.name} = 0;\n"
                 f"{ind}    first_got_{b.name} = std::nan(\"\");"
                 f" first_exp_{b.name} = std::nan(\"\");\n"
                 f"{ind}    for (int64_t i = 0; i < leet_n_{b.name}; ++i) {{\n"
                 f"{ind}        double got = (double)h_{b.name}[i];\n"
                 f"{ind}        double exp = (double)e_{b.name}[i];\n"
                 + _compare_body(b, atol, rtol, indent=ind + "        ")
                 + f"{ind}    }}\n")

    pass_cond = " && ".join(f"bad_{b.name} == 0" for b in out_bufs) or "true"
    guard_cond = " && ".join(f"gf_{b.name} == 0 && gb_{b.name} == 0"
                             for b in problem.buffers) or "true"
    p.append(f"{ind}    if (({pass_cond}) && ({guard_cond})) ++verify_pass;\n"
             f"{ind}}}\n")

    # ---- 性能 ----
    p.append(_indent_block("""
// ---------------- 性能 ----------------
double median_ms = -1.0, min_ms = -1.0, gb_per_s = -1.0;
double bytes_moved = (double)(@@BYTES@@);
if (do_perf) {
    for (int i = 0; i < warmup; ++i) {
        leet_flush_l2();
        @@LAUNCHER@@(lctx);
    }
    cudaError_t e_warm = cudaDeviceSynchronize();
    if (e_warm != cudaSuccess) LEET_FAIL("预热", e_warm);

    std::vector<double> ts;
    ts.reserve((size_t)perf_repeat);
    for (int i = 0; i < perf_repeat; ++i) {
        leet_flush_l2();                    // 计时区外
        cudaEventRecord(ev_a, nullptr);
        @@LAUNCHER@@(lctx);
        cudaEventRecord(ev_b, nullptr);
        cudaEventSynchronize(ev_b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_a, ev_b);
        cudaError_t e_t = cudaGetLastError();
        if (e_t != cudaSuccess) LEET_FAIL("计时循环", e_t);
        ts.push_back((double)ms);
    }
    std::sort(ts.begin(), ts.end());
    min_ms = ts.front();
    median_ms = ts[ts.size() / 2];
    if (median_ms > 0.0) gb_per_s = bytes_moved / median_ms * 1e-6;
}
""".replace("@@BYTES@@", _bytes_moved_expr(problem))
   .replace("@@LAUNCHER@@", problem.launcher), ind))

    # ---- 输出 JSON ----
    p.append(f"\n{ind}// ---------------- 结果(单行 JSON,前缀标记便于解析) ----------------\n")
    p.append(f'{ind}std::printf("%s{{", LEET_JSON);\n')
    p.append(f'{ind}std::printf("\\"case\\":\\"%s\\"", case_name);\n')
    p.append(f'{ind}std::printf(",\\"params\\":{{");\n')
    for i, prm in enumerate(problem.params):
        comma = "" if i == 0 else ", "
        p.append(f'{ind}std::printf("{comma}\\"{prm.name}\\":");'
                 f' {{ char nb[64]; leet_jnum(nb, sizeof(nb), (double){prm.name});'
                 f' std::printf("%s", nb); }}\n')
    p.append(f'{ind}std::printf("}}");\n')
    p.append(f'{ind}std::printf(",\\"outputs\\":[");\n')
    for i, b in enumerate(out_bufs):
        comma = "" if i == 0 else ", "
        p.append(_indent_block("""
{
    char nb[64];
    std::printf("@@COMMA@@{\\"name\\":\\"@@NAME@@\\",\\"count\\":%lld,\\"bad\\":%lld,",
                (long long)leet_n_@@NAME@@, bad_@@NAME@@);
    std::printf("\\"max_abs_err\\":%.9g,\\"max_rel_err\\":%.9g,",
                max_abs_@@NAME@@, max_rel_@@NAME@@);
    std::printf("\\"first_bad_idx\\":%lld,", first_bad_@@NAME@@);
    leet_jnum(nb, sizeof(nb), first_got_@@NAME@@);
    std::printf("\\"first_got\\":%s,", nb);
    leet_jnum(nb, sizeof(nb), first_exp_@@NAME@@);
    std::printf("\\"first_exp\\":%s,", nb);
    std::printf("\\"nan_count\\":%lld,\\"inf_count\\":%lld}", nan_@@NAME@@, inf_@@NAME@@);
}
""".replace("@@NAME@@", b.name).replace("@@COMMA@@", comma), ind))
    p.append(f'{ind}std::printf("]");\n')
    p.append(f'{ind}std::printf(",\\"guards\\":{{");\n')
    for i, b in enumerate(problem.buffers):
        comma = "" if i == 0 else ", "
        p.append(f'{ind}std::printf("{comma}\\"{b.name}\\":{{\\"front\\":%lld,\\"back\\":%lld}}",'
                 f' gf_{b.name}, gb_{b.name});\n')
    p.append(f'{ind}std::printf("}}");\n')
    p.append(f'{ind}std::printf(",\\"verify_pass\\":%lld", verify_pass);\n')
    p.append(f'{ind}std::printf(",\\"verify_total\\":%d", verify_repeat);\n')
    p.append(f'{ind}std::printf(",\\"ok\\":%s", ({_ok_expr(problem)}) ? "true" : "false");\n')
    p.append(_indent_block("""
if (do_perf) std::printf(",\\"perf\\":{\\"median_ms\\":%.6f,\\"min_ms\\":%.6f,"
                         "\\"runs\\":%d,\\"warmup\\":%d,\\"bytes_moved\\":%.0f,"
                         "\\"gb_per_s\\":%.3f}",
                         median_ms, min_ms, perf_repeat, warmup, bytes_moved, gb_per_s);
std::printf("}\\n");
std::fflush(stdout);
""", ind))

    p.append(_indent_block("""
cudaEventDestroy(ev_a); cudaEventDestroy(ev_b);
};   // leet_one_case
leet_one_case();
    }

    if (flush_buf) cudaFree(flush_buf);
    if (flush_sink) cudaFree(flush_sink);
    return 0;
}
""", ind[:4]))
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

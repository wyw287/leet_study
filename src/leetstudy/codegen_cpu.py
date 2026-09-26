"""代码生成(CPU / C++ 科目)—— 把 spec.yaml 变成可编译的 C++ harness。

产物(在 build/<problem>/ 下):
  ctx.h              Ctx 结构体,harness 与 reference.cpp 共享
  <variant>/impl.cpp     #include 具体实现(template / 用户解 / baseline / 参考解)
  <variant>/harness.cpp  生成的 main:分配 → 填充 → 执行 → 计时 → 校验 → 输出 JSON

与 CUDA 科目的区别
------------------
* **只有一个 Ctx**。CUDA 要分 LaunchCtx(device 指针)与 RefCtx(host 指针),
  两者字段不同;CPU 上实现与参考解看到的都是同一块 host 内存,没必要分。
* **没有 device 内存管理、没有 stream、没有 event**。调用是同步的,计时直接用
  `steady_clock` —— 不需要 CUDA 那套 event 记录(那是为了绕开异步 launch)。
* **哨兵区不需要 memcpy**。CUDA 要把哨兵搬回主机才能比对;CPU 上它就是内存,
  直接读。

刻意复用的部分
--------------
比对语义(`_compare_body`)、填充分布(`_fill_expr`)、毒值(`_POISON`)、
哨兵值(`_GUARD_CALL`)、通过条件(`_ok_expr`)全部从 `codegen` 导入,而不是
复制一份 —— 这些是**报告层与诊断层直接消费的契约**,两个科目各写一份迟早会漂移,
而漂移的表现是「同一个错误在 CUDA 题上被判失败、在 C++ 题上被判通过」。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .codegen import (
    _GENERATED_HEADER, _GUARD_CALL, _POISON, _bytes_moved_expr, _compare_body,
    _cpp, _fill_expr, _indent_block, _ok_expr,
)
from .spec import Problem
from .subjects.base import JSON_MARKER

__all__ = ["render_ctx_h", "render_harness_cpp", "write_ctx_h", "write_variant"]


# --------------------------------------------------------------------------- #
# ctx.h
# --------------------------------------------------------------------------- #

def render_ctx_h(problem: Problem) -> str:
    """生成 Ctx 结构体头文件。

    实现与参考解**共用同一个结构体** —— 两者看到的都是 host 指针,没有区分的必要。
    字段名来自 spec 的 buffers / params,所以三方接口不会漂移。

    指针指向「内区」,前后各有哨兵区:越界写会踩到哨兵,跑完立刻能发现。
    """
    out: List[str] = [_GENERATED_HEADER]
    out.append("#pragma once\n")
    out.append("#include <cstdint>\n\n")

    out.append("// 传给实现与参考解的上下文。所有指针都已在 host 上,数据已就绪。\n")
    out.append("// 每块缓冲前后都有哨兵区,越界写会被检测到。\n")
    out.append("struct Ctx {\n")
    for b in problem.buffers:
        out.append(f"    {b.ptr_type} {b.name};")
        out.append(f"  // {'x'.join(b.shape) or '标量'} ({b.dtype}, {b.role})\n")
    for p in problem.params:
        out.append(f"    {p.ctype} {p.name};\n")
    out.append("};\n")

    return "".join(out)


# --------------------------------------------------------------------------- #
# harness.cpp
# --------------------------------------------------------------------------- #

_HARNESS_PRELUDE = """#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#include "ctx.h"

// 具体实现:用户解答 / 模板 / baseline / 参考解
#include "@IMPL@"

// 由 reference.cpp 提供(单独编译单元)
void reference(Ctx& ctx);

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
// 每个缓冲前后各留 kGuardElems 个元素填哨兵值,交给实现的是内区指针。
// 越界写会踩到哨兵,跑完立刻能发现,并区分是「往前越界」还是「往后越界」。
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
static void leet_guard_check(const T* base, int64_t interior,
                             long long* front_hits, long long* back_hits) {
    *front_hits = 0; *back_hits = 0;
    const T sentinel = leet_guard_value<T>();
    for (int64_t i = 0; i < kGuardElems; ++i) if (base[i] != sentinel) ++(*front_hits);
    const T* tail = base + kGuardElems + interior;
    for (int64_t i = 0; i < kGuardElems; ++i) if (tail[i] != sentinel) ++(*back_hits);
}

// 出错的统一出口:打印一行错误 JSON 并跳出当前用例(用例体是 lambda)
static void leet_bail(const char* case_name, const char* stage, const char* detail) {
    std::printf("%s{\\"case\\":\\"%s\\",\\"stage\\":\\"%s\\",\\"error_name\\":\\"%s\\","
                "\\"ok\\":false}\\n",
                LEET_JSON, case_name, stage, detail);
    std::fflush(stdout);
}

#define LEET_FAIL(stage, detail) do { leet_bail(case_name, stage, detail); return; } while (0)

// ------------------------------------------------------------------
// 缓存清空
//
// 与 CUDA 侧同理、理由不同。CUDA 那边是 4090 有 72MB L2、题目只有几 MB,不清
// 就会测成缓存带宽。CPU 这边题目数据同样可能整个住在 L3 里。
//
// 为什么只需要 ~128MB 而不是 L3 的全部 512MB:本机 L3 是分片的(每个 CCX
// 32MB),**单线程只会填充自己所属那一片**。所以清 128MB 已经有 4 倍余量,
// 而读 512MB 要多花 4 倍时间(实测读 1GB 约 100ms,会直接主导整个计时循环)。
//
// 用「读」而不是「写」:写会留下脏行,被计时的函数一开始就得先写回它们才能
// 腾地方 —— 那份额外访存会污染测量。
// ------------------------------------------------------------------
static void leet_flush(const char* buf, size_t n) {
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < n; i += 64) sink += (unsigned char)buf[i];
    (void)sink;
}

static size_t leet_flush_bytes() {
    size_t l3 = 0;
    if (FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r")) {
        char line[64] = {0};
        if (std::fgets(line, sizeof(line), f)) {
            long long v = std::atoll(line);
            size_t len = std::strlen(line);
            char unit = len ? line[len - 1] : 0;
            if (unit == 'K' || unit == 'k') v *= 1024;
            else if (unit == 'M' || unit == 'm') v *= 1024 * 1024;
            if (v > 0) l3 = (size_t)v;
        }
        std::fclose(f);
    }
    // 单核只填充自己那一片,取 L3 的 1/4 就够;夹在 [32MB, 128MB]:
    // 下限保证小机器上也真能清干净,上限保证不会让清缓存主导计时循环。
    size_t want = l3 ? l3 / 4 : (64u << 20);
    if (want < (32u << 20)) want = 32u << 20;
    if (want > (128u << 20)) want = 128u << 20;
    return want;
}
"""


def render_harness_cpp(problem: Problem, impl_relpath: str = "impl.cpp") -> str:
    """生成 harness 的 main。

    与 CUDA 侧同样**一个进程跑完所有用例**。CPU 上进程启动本身很便宜(实测
    0.02 秒),但这里仍然这么做,理由变成了「一致性」:同一份 harness 逻辑
    跑所有用例,计时环境才可比。而且 CPU 题动辄几十毫秒,重复起进程也会累积。

    错误处理同 CUDA:每个用例体是一个 lambda,出错时打印错误 JSON 并 return,
    然后继续下一个用例 —— 一个用例挂掉不该让其余用例的结果全部丢失。
    """
    atol = problem.verify.atol
    rtol = problem.verify.rtol
    out_bufs = problem.outputs
    poison_targets = out_bufs + problem.scratches
    ind = "            "      # lambda 体内的缩进(12 空格)
    fn = problem.entry.get("function", "")

    p: List[str] = [_GENERATED_HEADER]
    p.append(
        _HARNESS_PRELUDE.replace("@IMPL@", impl_relpath).replace("@MARKER@", JSON_MARKER)
    )

    # ---- 参数表 ----
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

    // 清缓存的缓冲:整个进程只分配一次,所有用例复用
    std::vector<char> flush_buf;
    size_t flush_n = 0;
    if (@@FLUSH@@) {
        flush_n = leet_flush_bytes();
        flush_buf.assign(flush_n, 1);
    }
    auto leet_flush_cache = [&]() {
        if (flush_n) leet_flush(flush_buf.data(), flush_n);
    };

    for (int ci : todo) {
        const char* case_name = kCaseNames[ci];
        const LeetParams& C = kCases[ci];

        // 单个用例的全部流程。包成 lambda 是为了能用 return 早退,
        // 同时让 RAII(std::vector)负责释放 —— 不把内存漏给下一个用例。
        auto leet_one_case = [&]() {
""", WARMUP=problem.perf.warmup, PERF_REPEAT=problem.perf.repeat,
        VERIFY_REPEAT=max(1, problem.verify.repeat),
        FLUSH="true" if problem.perf.flush_l2 else "false"))

    # 参数局部变量
    for prm in problem.params:
        p.append(f"{ind}{prm.ctype} {prm.name} = ({prm.ctype})C.{prm.name};\n")

    # ---- 缓冲(带哨兵区) ----
    p.append(f"\n{ind}// ---------------- 缓冲(每块前后各留 kGuardElems 个哨兵) ----------------\n")
    for b in problem.buffers:
        p.append(f"{ind}int64_t leet_n_{b.name} = {b.count_expr()};\n")
    for b in problem.buffers:
        p.append(
            f"{ind}std::vector<{b.ctype}> g_{b.name}"
            f"((size_t)(leet_n_{b.name} + 2 * kGuardElems), {_GUARD_CALL[b.dtype]});\n"
            f"{ind}{b.ctype}* {b.name} = g_{b.name}.data() + kGuardElems;"
            f"   // 交给实现的内区\n"
        )

    # ---- 填充输入 ----
    p.append(f"\n{ind}// ---------------- 输入填充(种子由用例名派生,可复现) ----------------\n")
    for i, b in enumerate(problem.inputs):
        p.append(f"{ind}{{ LeetRng rng(leet_seed_of(case_name) ^ {i + 1}ULL);\n"
                 f"{ind}  for (int64_t j = 0; j < leet_n_{b.name}; ++j)"
                 f" {b.name}[j] = {_fill_expr(b)}; }}\n")
    for b in problem.scratches:
        p.append(f"{ind}std::memset({b.name}, 0,"
                 f" sizeof({b.ctype}) * (size_t)leet_n_{b.name});\n")

    # ---- 期望值(参考解) ----
    p.append(f"\n{ind}// ---------------- 参考解(ground truth) ----------------\n")
    for b in out_bufs:
        p.append(f"{ind}std::vector<{b.ctype}> e_{b.name}((size_t)leet_n_{b.name});\n")
    p.append(f"{ind}{{\n{ind}  Ctx rctx{{}};\n")
    for b in problem.inputs:
        p.append(f"{ind}  rctx.{b.name} = {b.name};\n")
    for b in out_bufs:
        p.append(f"{ind}  rctx.{b.name} = e_{b.name}.data();\n")
    for b in problem.scratches:
        p.append(f"{ind}  rctx.{b.name} = {b.name};\n")
    for prm in problem.params:
        p.append(f"{ind}  rctx.{prm.name} = {prm.name};\n")
    p.append(f"{ind}  reference(rctx);\n{ind}}}\n")

    # ---- Ctx ----
    p.append(f"\n{ind}Ctx ctx{{}};\n")
    for b in problem.buffers:
        p.append(f"{ind}ctx.{b.name} = {b.name};\n")
    for prm in problem.params:
        p.append(f"{ind}ctx.{prm.name} = {prm.name};\n")

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
// 同一份输入反复跑,任何一次结果不对都算失败。
for (int vr = 0; vr < verify_repeat; ++vr) {
    // out / scratch 填毒值:让「没写输出」表现为清一色的毒值
""".replace("@@N@@", str(max(1, problem.verify.repeat))), ind))
    for b in poison_targets:
        p.append(
            f"{ind}    for (int64_t j = 0; j < leet_n_{b.name}; ++j)"
            f" {b.name}[j] = {_POISON[b.dtype]};\n"
        )
    p.append(f"{ind}    @@FN@@(ctx);\n".replace("@@FN@@", fn))
    for b in problem.buffers:
        p.append(f"{ind}    {{ long long f = 0, b2 = 0;"
                 f" leet_guard_check<{b.ctype}>(g_{b.name}.data(),"
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
                 f"{ind}        double got = (double){b.name}[i];\n"
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
// 调用是同步的,所以直接用 steady_clock —— 不需要 CUDA 那套 event 记录
// (那是为了绕开异步 launch,把计时框在 device 侧)。
double median_ms = -1.0, min_ms = -1.0, gb_per_s = -1.0;
double bytes_moved = (double)(@@BYTES@@);
if (do_perf) {
    for (int i = 0; i < warmup; ++i) {
        leet_flush_cache();
        @@FN@@(ctx);
    }

    std::vector<double> ts;
    ts.reserve((size_t)perf_repeat);
    for (int i = 0; i < perf_repeat; ++i) {
        leet_flush_cache();                    // 计时区外
        auto t0 = std::chrono::steady_clock::now();
        @@FN@@(ctx);
        auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    min_ms = ts.front();
    median_ms = ts[ts.size() / 2];
    if (median_ms > 0.0) gb_per_s = bytes_moved / median_ms * 1e-6;
}
""".replace("@@BYTES@@", _bytes_moved_expr(problem)).replace("@@FN@@", fn), ind))

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
};   // leet_one_case
leet_one_case();
    }

    return 0;
}
""", ind[:4]))
    return "".join(p)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #

def render_impl_cpp(impl_path: Path) -> str:
    """生成 impl.cpp:把具体实现拉进来。"""
    return (
        "// 自动生成:把具体实现包含进来(用户解答 / 模板 / baseline / 参考解)\n"
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
    impl_cpp = out_dir / "impl.cpp"
    harness_cpp = out_dir / "harness.cpp"
    impl_cpp.write_text(render_impl_cpp(impl_path), encoding="utf-8")
    harness_cpp.write_text(render_harness_cpp(problem), encoding="utf-8")
    return {"impl": impl_cpp, "harness": harness_cpp}

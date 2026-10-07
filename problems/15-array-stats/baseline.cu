// 题目 15 的性能基线:三趟遍历,每一趟只算一个统计量。
//
// 这是评分的分母。注意它**不是**一个偷懒的实现 —— 它每一步都写得很正常:
//
//   * 跨步循环(grid-stride)→ 全局读是**合并**的,没有一个字节浪费;
//   * 每个线程先在寄存器里累加,再走块内树形归约 → 块内没有原子冲突;
//   * 块间不做全局原子,而是每块写 partial、第二级 kernel(单块)收尾;
//   * 空块也会写下合法的 partial(min 的空值是 +inf,不是 0)。
//
// 它唯一的毛病是:**同一个 256MiB 的数组被从头到尾读了三遍。**
// 而这题的成本结构就是「读一遍 = 266µs」—— 三遍 = 800µs(实测 0.855 ms,
// 314 GB/s,31% 峰值,差的那几十 µs 是四次 kernel 启动和三次块内归约的尾巴)。
// 换句话说:基线的每一个字节访存都是完美的,它慢**只**因为读的次数多。
//
// ★ 三个统计量为什么要写成三个 **kernel**?
//   因为如果写成一个 kernel 里的三个连续循环,nvcc 会把它们**融合**成一个
//   (三个循环读的是同一个 __restrict__ 指针,没有任何东西挡住它)—— 编译完
//   就自动变成了本题的答案,基线也就不成立了。实测过:融合版跑出 88% 峰值,
//   和参考解的 92% 几乎没有区别。拆成三次 kernel 启动才是真实的"三趟"。
//   (每多一趟,除 266µs 的流量之外还要多花约 17µs:一次启动 + 一次块内归约的
//    尾巴。实测 1 趟 0.289ms / 2 趟 0.572ms / 3 趟 0.855ms,正好每档 +266µs ——
//    贵的从来不是启动,是流量。)

#include "ctx.h"

#define BLOCK 256          // 第一级:每块 256 线程
#define FINAL_BLOCK 256    // 第二级:单块 256 线程

enum StatOp { OP_SUM = 0, OP_MIN = 1, OP_MAX = 2 };

template <int OP> __device__ __forceinline__ float stat_combine(float a, float b) {
    if (OP == OP_SUM) return a + b;
    if (OP == OP_MIN) return fminf(a, b);
    return fmaxf(a, b);
}

template <int OP> __device__ __forceinline__ float stat_identity() {
    if (OP == OP_SUM) return 0.f;
    if (OP == OP_MIN) return __int_as_float(0x7f800000);    // +inf
    return -__int_as_float(0x7f800000);                     // -inf
}

// 一趟遍历,只算**一个**统计量。三个统计量 = 三次这样的遍历。
// partial 的布局是 [blockIdx.x * 3 + OP]。
template <int OP>
__global__ void stats_pass(const float* __restrict__ x,
                           float* __restrict__ partial,
                           int n) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;

    float acc = stat_identity<OP>();
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        acc = stat_combine<OP>(acc, x[i]);
    }

    s[tid] = acc;
    __syncthreads();
    for (int k = BLOCK / 2; k > 0; k >>= 1) {
        if (tid < k) s[tid] = stat_combine<OP>(s[tid], s[tid + k]);
        __syncthreads();
    }
    // 无条件写:领不到元素的块也要写下自己的单位元。
    if (tid == 0) partial[3 * blockIdx.x + OP] = s[0];
}

// 第二级:单块,把 grid 个三元组合成三个标量。
//
// 为什么不直接让第一级 atomicAdd / atomicMin 到全局?
//   * atomicAdd 有 float 版本,但 min/max **只有整数版**(硬件就没提供 float 的),
//     想在 float 上做就得自己写 CAS 循环,还得处理"负数区间位模式反过来"。
//   * "每块写 partial + 第二级 kernel"两条路都通,而且块数再多也不会退化成
//     对同一个地址串行排队。这是最省心的写法,参考解也用它。
__global__ void stats_final(const float* __restrict__ partial,
                            float* __restrict__ sum,
                            float* __restrict__ min_val,
                            float* __restrict__ max_val,
                            int nblocks) {
    __shared__ float ss[FINAL_BLOCK], sn[FINAL_BLOCK], sx[FINAL_BLOCK];
    const int tid = threadIdx.x;

    float a  = 0.f;
    float lo = __int_as_float(0x7f800000);
    float hi = -__int_as_float(0x7f800000);
    for (int b = tid; b < nblocks; b += FINAL_BLOCK) {
        a  += partial[3 * b + 0];
        lo  = fminf(lo, partial[3 * b + 1]);
        hi  = fmaxf(hi, partial[3 * b + 2]);
    }

    // 三个量共用同一轮同步:一次 __syncthreads 管三个归约。
    ss[tid] = a; sn[tid] = lo; sx[tid] = hi;
    __syncthreads();
    for (int k = FINAL_BLOCK / 2; k > 0; k >>= 1) {
        if (tid < k) {
            ss[tid] += ss[tid + k];
            sn[tid] = fminf(sn[tid], sn[tid + k]);
            sx[tid] = fmaxf(sx[tid], sx[tid + k]);
        }
        __syncthreads();
    }
    // 这三个写是**覆盖写**(不是累加),所以不需要 cudaMemset 去清毒值。
    if (tid == 0) {
        sum[0]     = ss[0];
        min_val[0] = sn[0];
        max_val[0] = sx[0];
    }
}

void array_stats_launch(LaunchCtx& ctx) {
    const int block = BLOCK;
    // 上限是 partial 能装下的块数(见 spec 的 max_blocks)。
    int grid = (ctx.n + block - 1) / block;
    if (grid > ctx.max_blocks) grid = ctx.max_blocks;
    if (grid < 1) grid = 1;

    // 三次遍历:每一次都把整个数组从显存里读一遍。
    stats_pass<OP_SUM><<<grid, block, 0, ctx.stream>>>(ctx.x, ctx.partial, ctx.n);
    stats_pass<OP_MIN><<<grid, block, 0, ctx.stream>>>(ctx.x, ctx.partial, ctx.n);
    stats_pass<OP_MAX><<<grid, block, 0, ctx.stream>>>(ctx.x, ctx.partial, ctx.n);

    stats_final<<<1, FINAL_BLOCK, 0, ctx.stream>>>(ctx.partial, ctx.sum,
                                                   ctx.min_val, ctx.max_val, grid);
}

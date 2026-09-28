// 题目 09 的性能基线:最直白的"三趟"实现。
//
//   ① row_max_kernel   求每行最大值      → rmax[rows]
//   ② row_sum_kernel   求每行 exp 之和   → rsum[rows]
//   ③ row_softmax      归一化并写出 y
//
// 这三趟每一趟本身都是规矩的写法:一行一个 block、跨步循环、访存合并、
// 共享内存树形归约、同步齐全。它**没有写坏** —— 唯一的浪费是:
// 同一份 134MB 的输入,被从 DRAM 里搬了三遍。
//
// 为什么这三趟要拆成三个 kernel(而不是一个 kernel 里读三遍全局内存)?
// 因为这样才真的贵:两个 kernel 之间有全局屏障,等第一个 kernel 跑完,
// 整个 134MB 已经被流过去了,而 L2 只有 72MB —— 上一趟读进来的行早就被挤干净,
// 第二三趟只能重新从 DRAM 读。这 536MB 的账就是这么来的,也决定了本题
// 2.0x 的加速比天花板(536/268)。
//
// 顺带:rmax / rsum 这两个 scratch 缓冲是"三趟"的直接后果 ——
// 需要把逐行的中间量落到 DRAM 里,说明输入你至少读了两遍。
// 融合成一趟的做法一个 scratch 都不需要。

#include "ctx.h"

#define BLOCK 256

// ---------------------------------------------------------------------------
// ① 每行一个 block:求这一行的最大值(已缩放)
// ---------------------------------------------------------------------------
__global__ void row_max_kernel(const float* __restrict__ x,
                               float* __restrict__ rmax,
                               int rows, int cols, float s) {
    __shared__ float sm[BLOCK];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = threadIdx.x;
    const float* xr = x + (size_t)r * (size_t)cols;

    float m = -1e30f;
    for (int c = t; c < cols; c += BLOCK) {
        m = fmaxf(m, s * xr[c]);
    }

    // 树形归约。cols 比 BLOCK 小时,多出来的线程贡献的是中性元 -1e30f,不影响结果。
    sm[t] = m;
    __syncthreads();
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (t < st) sm[t] = fmaxf(sm[t], sm[t + st]);
        __syncthreads();
    }
    if (t == 0) rmax[r] = sm[0];
}

// ---------------------------------------------------------------------------
// ② 每行一个 block:求这一行 exp(s*x - max) 的和
//    注意它**又把 x 读了一遍** —— 这就是被浪费掉的那 134MB 之一
// ---------------------------------------------------------------------------
__global__ void row_sum_kernel(const float* __restrict__ x,
                               const float* __restrict__ rmax,
                               float* __restrict__ rsum,
                               int rows, int cols, float s) {
    __shared__ float sm[BLOCK];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = threadIdx.x;
    const float* xr = x + (size_t)r * (size_t)cols;
    const float m = rmax[r];

    float acc = 0.f;
    for (int c = t; c < cols; c += BLOCK) {
        acc += expf(s * xr[c] - m);
    }

    sm[t] = acc;
    __syncthreads();
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (t < st) sm[t] += sm[t + st];
        __syncthreads();
    }
    if (t == 0) rsum[r] = sm[0];
}

// ---------------------------------------------------------------------------
// ③ 逐元素:y = exp(s*x - max) / sum
//    这里 exp 又算了一遍(第二遍),而且 x 读了第三遍
// ---------------------------------------------------------------------------
__global__ void row_softmax(const float* __restrict__ x,
                            const float* __restrict__ rmax,
                            const float* __restrict__ rsum,
                            float* __restrict__ y,
                            int rows, int cols, float s) {
    const int r = blockIdx.y;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * (size_t)cols;
    float* yr = y + (size_t)r * (size_t)cols;

    const float m = rmax[r];
    const float inv = 1.0f / rsum[r];   // 除法只在每个线程上做一次

    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += gridDim.x * blockDim.x) {
        yr[c] = expf(s * xr[c] - m) * inv;
    }
}

void row_softmax_launch(LaunchCtx& ctx) {
    const float s = (float)ctx.scale;

    row_max_kernel<<<ctx.rows, BLOCK>>>(ctx.x, ctx.rmax, ctx.rows, ctx.cols, s);
    row_sum_kernel<<<ctx.rows, BLOCK>>>(ctx.x, ctx.rmax, ctx.rsum, ctx.rows, ctx.cols, s);

    const dim3 block(BLOCK);
    const dim3 grid((ctx.cols + BLOCK - 1) / BLOCK, ctx.rows);
    row_softmax<<<grid, block>>>(ctx.x, ctx.rmax, ctx.rsum, ctx.y,
                                 ctx.rows, ctx.cols, s);
}

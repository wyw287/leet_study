// 题目 17 的性能基线:一个线程负责一个输出元素(也就是一列)。
//
// 这是**最自然**的写法 —— 01 题就是这么写的,而且那是对的写法。
// 它的访存也完全合并:warp 里相邻的 lane 读相邻的列,一条 load 指令正好
// 覆盖 128 个连续字节,一次不多一次不少。
//
// 它唯一的问题是:**只有 cols 个线程**。
//   big 用例 cols = 512 ⇒ **2 个 256 线程的块** ⇒ 16 个 warp。
//   128 个 SM 里 2 个在干活,每个线程要串行做 131072 次"load + 加"。
//   (这张卡要 ~4096 个 warp 才叫装满,你用掉了 0.4%。)
//
// 实测:3.011 ms / 89 GB/s = **9% 峰值**。
//
// 也就是说:这个 kernel 的瓶颈不是带宽、不是合并、不是指令数,而是
// **并行度**。显存墙只是"它本可以到的地方",不是"它撞上的东西"。
//
// 基线的作用是给出一个下限:至少不该比它慢。

#include "ctx.h"

__global__ void colsum(const float* x, float* out, int rows, int cols) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float acc = 0.0f;
    for (int i = 0; i < rows; ++i) {
        acc += x[(long)i * cols + j];
    }
    out[j] = acc;
}

void colsum_launch(LaunchCtx& ctx) {
    const int block = 256;
    const int grid = (ctx.cols + block - 1) / block;   // cols=512 ⇒ 2 个块
    colsum<<<grid, block>>>(ctx.x, ctx.out, ctx.rows, ctx.cols);
}

// 题目 03 的性能基线:每个线程负责一个元素,直接读写全局内存。
//
// 读是合并的(同一 warp 的相邻线程读 input 里相邻的列),
// 但写是跨步的(相邻线程写 output 里相隔 m 个元素的位置)。
// 跨步写会让每一次 warp 写都散成 32 个独立的显存事务 ——
// 这正是本题要你修掉的东西。

#include "ctx.h"

__global__ void transpose(const float* input, float* output, int m, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // 列
    int r = blockIdx.y * blockDim.y + threadIdx.y;   // 行
    if (r < m && c < n) {
        output[(long long)c * m + r] = input[(long long)r * n + c];
    }
}

void transpose_launch(LaunchCtx& ctx) {
    dim3 block(32, 8);
    dim3 grid((ctx.n + block.x - 1) / block.x, (ctx.m + block.y - 1) / block.y);
    transpose<<<grid, block>>>(ctx.input, ctx.output, ctx.m, ctx.n);
}

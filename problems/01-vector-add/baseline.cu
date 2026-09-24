// 题目 01 的性能基线:最朴素的 CUDA 实现,一个线程算一个元素。
//
// 这是「加速比的分母」。对向量加法这种访存瓶颈题,朴素实现已经接近
// 最优 —— 所以本题用「有效带宽占峰值百分比」评分,而不是加速比。
// 基线的作用是给出一个下限:至少不该比它慢。

#include "ctx.h"

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vecadd_launch(LaunchCtx& ctx) {
    int block = 256;
    int grid = (ctx.n + block - 1) / block;
    vecadd<<<grid, block>>>(ctx.a, ctx.b, ctx.c, ctx.n);
}

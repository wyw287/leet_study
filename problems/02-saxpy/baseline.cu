// 题目 02 的性能基线:最朴素的一线程一元素实现。
//
// 和向量加法一样,这是访存瓶颈题 —— 每个元素做 1 次乘加,却要搬 12 字节。
// 朴素实现已经接近最优,所以本题也按带宽利用率评分,而不是加速比。

#include "ctx.h"

__global__ void saxpy(const float* x, const float* y, float* out,
                      float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}

void saxpy_launch(LaunchCtx& ctx) {
    int block = 256;
    int grid = (ctx.n + block - 1) / block;
    saxpy<<<grid, block>>>(ctx.x, ctx.y, ctx.out, ctx.alpha, ctx.n);
}

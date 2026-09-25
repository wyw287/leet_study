// 题目 02 的参考解
//
// 和向量加法一样,**这道题没有优化空间**。区别只在于它多了一个标量参数,
// 而那个参数的处理方式恰好是个常见误区。
//
// 实测:0.180 ms / 1117 GB/s,已经贴在 4090 的标称峰值(1008 GB/s)上。

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

// ---------------------------------------------------------------------------
// 常见误区:把 alpha 也搬到显存里
//
//     float* d_alpha;
//     cudaMalloc(&d_alpha, sizeof(float));
//     cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice);
//     saxpy<<<...>>>(x, y, out, d_alpha, n);        // 传指针而不是值
//
// 这是错的,而且有两个代价:
//
//   1. 多一次 cudaMalloc + cudaMemcpy。后者会强制同步,把流水打断。
//   2. kernel 里读 *d_alpha 变成一次**全局内存访问** —— 每个线程都要去显存
//      取同一个值。虽然 L2 会挡住大部分,但比寄存器里直接拿到差远了。
//
// 正确认识:CUDA 的 kernel 参数走一块专门的**常量内存**,而且硬件知道
// "同一个 warp 里所有线程读的是同一个地址",会做**广播** —— 一次事务
// 就能喂饱整个 warp。所以标量参数传参几乎零成本。
//
// 这也是为什么本框架的接口把 alpha 放在 LaunchCtx 里当**值**传,而不是指针。
// ---------------------------------------------------------------------------

// 题目 06 的性能基线:一个线程算一个输出元素,每个输出把 a 的一整行和
// b 的一整列从头读一遍。
//
// 这份实现**完全正确**,访存也是合并的:一个 warp 的 32 个线程共享同一行
// (threadIdx.x 是列),所以 a[row][k] 是广播、b[k][col] 是 32 个连续地址。
// 换句话说,它没有犯任何「低级错误」,慢的原因是算术强度太低:
//
//   n=2048 时,每个输出要读 2n = 4096 个 float = 16KB,而 4.19M 个输出
//   总共要请求 68.7 GB —— 而这题真正需要搬的字节只有三块 16MiB 矩阵 = 48MiB。
//   每个字节平均被请求了 1000 次以上。
//
// 这些重复请求并不都落到 DRAM(48MiB 装得进 4090 的 72MB L2),但它们**每
// 一次都要占一条访存指令、占一个 L1 的 wavefront** —— L1 到寄存器这条路的
// 吞吐是每 SM 每 cycle 128B,这才是基线的瓶颈。
//
// 修的方向不是「让它合并」(已经合并了),而是**让它少读**:把一块数据搬进
// 共享内存,让它在被换出之前服务成百上千次 FMA。

#include "ctx.h"

__global__ void matmul(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out,
                       int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 网格是向上取整的,最后一行/列要判边界。
    if (row >= n || col >= n) return;

    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        // a[row][k]:一个 warp 里 32 个线程读同一个地址 → 广播
        // b[k][col]:32 个连续地址 → 合并
        acc += a[(long long)row * n + k] * b[(long long)k * n + col];
    }
    out[(long long)row * n + col] = acc;
}

void matmul_launch(LaunchCtx& ctx) {
    dim3 block(32, 8);
    dim3 grid((ctx.n + block.x - 1) / block.x, (ctx.n + block.y - 1) / block.y);
    matmul<<<grid, block>>>(ctx.a, ctx.b, ctx.out, ctx.n);
}

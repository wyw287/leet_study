// 题目 08 的性能基线:一个线程算一行,内层顺序扫过整行。
//
// 这份实现**完全正确**,也没有任何「低级错误」—— 它甚至是有缓存的:
// 同一线程连着读 a[r][c]、a[r][c+1] …… 全是连续的。慢的原因有两个,
// 而且两个都**不在「字节数」这一侧**:
//
// 1) 并行度不够。整个 kernel 只有 m 个线程。m=4096 时 = 16 个 256 线程块,
//    而 4090 有 128 个 SM —— 112 个 SM 从头到尾没事干。
//
// 2) warp 内的访存是 32 路散开的。同一个 warp 的 32 个线程(threadIdx.x 相邻)
//    读的是**相隔 n 个 float 的 32 个不同行**:一条 load 指令要拆成 32 个
//    32 字节的 sector,一共 1024 字节,其中这一次只有 128 字节有用。
//
//    这 1024 字节并没有浪费 —— 每个 sector 里剩下的 7 个 float 会被同一线程的
//    后 7 次迭代用掉(一个 warp 的活跃工作集只有 1KB,L1 装得下),所以
//    **DRAM 流量是准的**。多出来的是**请求数**:L1 每 cycle 只能处理 4 个 sector,
//    同一份数据用 32 个 sector 的方式取,就比用 4 个 sector 的方式慢 8 倍。
//
// 所以这题的瓶颈不在「搬了多少字节」(128MiB,从头到尾都是准的),而在
// 「有多少线程在搬」和「一条 load 指令覆盖多少连续地址」这两件事上。
// 实测这两者各自的份量:只把线程数翻 32 倍(访存照旧)只快 1.8 倍;
// 改对访存模式(线程数顺带翻 32 倍)快 6 倍。
// 修法的核心是后者:**让一条 load 指令覆盖一段连续的地址**。

#include "ctx.h"

__global__ void gemv(const float* __restrict__ a,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    // 网格是向上取整的,最后一批线程要判边界。
    if (row >= m) return;

    const float* rowp = a + (size_t)row * (size_t)n;
    float acc = 0.0f;
    for (int col = 0; col < n; ++col) {
        acc += rowp[col] * x[col];
    }
    y[row] = acc;
}

void gemv_launch(LaunchCtx& ctx) {
    const int block = 256;
    const int grid = (ctx.m + block - 1) / block;
    gemv<<<grid, block>>>(ctx.a, ctx.x, ctx.y, ctx.m, ctx.n);
}

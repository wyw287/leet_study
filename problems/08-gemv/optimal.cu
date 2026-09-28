// 题目 08 的参考解 —— 一个 warp 一行 + warp shuffle 归约 + grid-stride
//
// 实测(4090,框架每次计时前自动清 L2;测的都是 wide,即 m=4096 / n=8192,
// 128MiB 的矩阵):
//     基线(1 线程 1 行)            0.863 ms   156 GB/s    15% 峰值   1.00x
//     只加并行度(还是散着取数)     0.477 ms   281 GB/s    28% 峰值   1.81x
//     本实现 + 网格封顶 16 块       0.315 ms   426 GB/s    42% 峰值   2.74x
//     本实现 + 网格封顶 24 块       0.228 ms   588 GB/s    58% 峰值   3.78x
//     本实现 + 网格封顶 40 块       0.162 ms   830 GB/s    82% 峰值   5.34x
//     本实现 + 网格封顶 64 块       0.146 ms   917 GB/s    91% 峰值   5.90x
//     本实现(warp 一行 + shuffle)  0.144 ms   931 GB/s    92% 峰值   5.98x
//     (ragged 用例上本实现是 0.145 ms / 924 GB/s / 92% —— 两个用例一致)
//
// 三处改动,每一处都对着一个具体的瓶颈:
//
//   ① **让 lane 决定行内位置,而不是让线程决定行号。**
//      基线里一条 load 指令覆盖 32 个相隔 n 个 float 的地址 → 32 个 sector;
//      改成 `col = lane; col += 32` 之后,一条 load 覆盖连续的 128 字节 = 4 个 sector。
//      少了 8 倍的请求,L1 那条路立刻不堵了。**这是这题唯一真正重要的一步。**
//
//   ② **warp 内归约走 shuffle,不走共享内存。**
//      32 个 lane 的结果本来就在同一个 warp 里,硬件保证它们锁步执行,
//      所以既不需要 __syncthreads,也不占共享内存。五步折半,每步一条指令。
//      (对比 04-reduction-sum:那里要跨线程块归约,才必须落一次共享内存。)
//
//   ③ **grid-stride**:线程数不再被 m 绑死。m=257 时也能铺满足够的 warp。
//
// 关于这道题的"天花板":134MB ÷ 1008 GB/s ≈ 133µs,实测跑到了 144µs(92%)。
// 这份实现已经贴着墙了,剩下的那点差距来自 DRAM 本身的效率 —— read-only 流
// 也到不了 100% 的理论峰值(刷新、bank 冲突、ECC 都要分走一份)。

#include "ctx.h"

__global__ void gemv(const float* __restrict__ a,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m, int n) {
    const int lane           = threadIdx.x & 31;
    const int warps_per_blk  = blockDim.x >> 5;
    const int warp           = blockIdx.x * warps_per_blk + (threadIdx.x >> 5);
    const int total_warps    = (gridDim.x * blockDim.x) >> 5;

    // 一个 warp 领一行。row 只跟 warp 号有关 → warp 内 32 个 lane 拿到的
    // row 完全一致,这个循环不会让 warp 分叉(后面 shuffle 要求全 warp 汇合)。
    for (int row = warp; row < m; row += total_warps) {
        const float* __restrict__ rowp = a + (size_t)row * (size_t)n;

        float acc = 0.0f;
        // 合并访问:同一 warp 的 32 个 lane 读 rowp[lane .. lane+31] ——
        // 连续的 128 字节 = 4 个 sector。
        // 边界由循环条件兜住:col 一旦 >= n 就停,不会有任何一次越界读;
        // 少走几轮的 lane 其 acc 保持 0,对求和没有影响。
        for (int col = lane; col < n; col += 32) {
            acc += rowp[col] * x[col];
        }

        // 五步折半。此处 32 个 lane 必须全部汇合、全部活着 —— 循环已经结束,
        // 没有任何分支把 warp 拆开,所以 mask 用全 1 是安全的。
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }

        // 只有 lane 0 的结果是完整的和。别的 lane 写同一个地址 = 竞态。
        if (lane == 0) {
            y[row] = acc;
        }
    }
}

void gemv_launch(LaunchCtx& ctx) {
    const int block = 256;                       // = 8 个 warp
    const int warps_per_blk = block / 32;
    // 需要多少行就开多少 warp(向上取整),再由 grid-stride 兜住零头。
    const int grid = (ctx.m + warps_per_blk - 1) / warps_per_blk;
    gemv<<<grid, block>>>(ctx.a, ctx.x, ctx.y, ctx.m, ctx.n);
}

// ---------------------------------------------------------------------------
// 值得亲手验证的三件事(都可以用 `leet bench 08` 试)
//
// 1. **float4 会不会更快?实测:不会。** 让每个 lane 一次搬 16 字节,一条 load
//    指令覆盖 32×16 = 512 字节,请求数再降到 1/4:
//
//        const float4* rowp4 = reinterpret_cast<const float4*>(a + (size_t)row * n);
//        const float4* x4    = reinterpret_cast<const float4*>(x);
//        for (int c = lane; c < n/4; c += 32) { float4 av = rowp4[c], xv = x4[c]; ... }
//
//    实测 0.144 ms / 930 GB/s —— 和标量版的 931 GB/s 分不出差别。它能省的是
//    **指令数**(LSU 的发射压力),省不下 DRAM 的字节数;而瓶颈早就在 DRAM 上了。
//    这就是"瓶颈在带宽,不在指令"最干净的一个证据。
//
//    还有一个真陷阱:float4 要求 16 字节对齐。ragged 用例的 n=8191 不是 4 的倍数,
//    此时**每一行的行首都不对齐**(row*8191*4 字节),直接 reinterpret_cast 会
//    报 misaligned address;必须回退到标量路径(n 是 4 的倍数才走向量路径)。
//
// 2. **把网格开小试试(这比扫块大小有意思得多)。** 参考解的网格是
//    (m+7)/8 = 512 块、每块 8 个 warp。在 launcher 里加一句
//    `if (grid > N) grid = N;` 再跑,实测(wide 用例):
//
//        N = 16 块(128 个 warp) → 0.315 ms / 426 GB/s / 42% 峰值
//        N = 24 块(192 个 warp) → 0.228 ms / 588 GB/s / 58%
//        N = 40 块(320 个 warp) → 0.162 ms / 830 GB/s / 82%
//        N = 64 块(512 个 warp) → 0.146 ms / 917 GB/s / 91%
//        N = 512 块(4096 个 warp,原样) → 0.144 ms / 931 GB/s / 92%
//
//    两个结论:① 并行度确实有一道坎,但**很低** —— 512 个 warp 就贴到天花板了
//    (4090 有 128 个 SM,合每 SM 4 个块),这题不值得你去抠 occupancy;
//    ② 反过来,只开几十个块的话,访存改得再漂亮也上不去。
//    **先判断带宽损失来自哪一侧,再决定改什么** —— 这是本题最值钱的一句话。
//
// 3. **把内层循环换回「每个 lane 领行内连续的一段」试试。** 别的都不动 ——
//    还是一个 warp 一行、还是同样的线程数、同样的并行度,只把取数方式改成
//
//        int chunk = (n + 31) / 32, c0 = lane * chunk, c1 = min(n, c0 + chunk);
//        for (int c = c0; c < c1; ++c) acc += rowp[c] * x[c];
//
//    实测 0.477 ms / 281 GB/s(**28%**)—— 比基线只快 1.8 倍。这一条比任何解释
//    都更能说明问题:**这题的瓶颈是访存模式,不是并行度**。线程数从 4096 涨到
//    131072 换不来带宽,把一条 load 指令从 32 个 sector 压到 4 个才换得来。
// ---------------------------------------------------------------------------

// 题目 03 的参考解 —— 分块共享内存转置
//
// 实测(4096×4096):
//     基线(朴素,写侧跨步)  0.479 ms   ≈ 267 GB/s
//     本实现                0.129 ms   ≈ 992 GB/s   → 3.71x
//
// 4090 峰值 1008 GB/s —— 也就是说这个版本已经**贴在带宽天花板上**了。
// 本题理论最大加速比就是 1008/267 ≈ 3.8x,拿不到更高不是实现问题,是物理限制。
//
// 核心思路(题面「解法思路」一节讲得更细):
//   读进共享内存时是合并的,从共享内存读出时按转置后的次序,
//   于是写回全局内存时也是合并的 —— 跨步这件事被挪到了片上,
//   而共享内存没有"cache line"的概念,代价小得多。

#include "ctx.h"

#define TILE 32

__global__ void transpose(const float* input, float* output, int m, int n) {
    // +1 是这个 kernel 里最划算的一个字符:
    // 不加的话,读 tile[tx][ty] 时同一 warp 的线程会落在同一个 bank 上,
    // 32 次访问被串行化成 32 个周期(bank conflict)。多一列把地址错开,
    // 冲突就消失了。值得亲手把 +1 去掉测一次,亲眼看看差别。
    __shared__ float tile[TILE][TILE + 1];

    int col0 = blockIdx.x * TILE;
    int row0 = blockIdx.y * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 第一步:合并读入共享内存。
    // 相邻的 tx 读连续的地址 → 硬件能合并成少的显存事务。
    if (row0 + ty < m && col0 + tx < n) {
        tile[ty][tx] = input[(long long)(row0 + ty) * n + (col0 + tx)];
    }
    // 同步不能省:上面写的和下面读的**不是同一批线程**。
    // 漏掉会得到随机错误的结果(racecheck 能抓到)。
    __syncthreads();

    // 第二步:转置着写出。
    // 注意 r/c 与 tx/ty 的对应关系反了过来 —— 读 tile[tx][ty] 时,
    // 相邻的 tx 走的是 tile 的**第一维**,在共享内存里是连续的(配合 +1 无冲突);
    // 而写出去的地址 (r * m + c) 对相邻 tx 也是连续的 → 合并写。
    int r = col0 + ty;
    int c = row0 + tx;
    if (c < m && r < n) {
        output[(long long)r * m + c] = tile[tx][ty];
    }
}

void transpose_launch(LaunchCtx& ctx) {
    dim3 block(TILE, TILE);
    dim3 grid((ctx.n + TILE - 1) / TILE, (ctx.m + TILE - 1) / TILE);
    transpose<<<grid, block>>>(ctx.input, ctx.output, ctx.m, ctx.n);
}

// ---------------------------------------------------------------------------
// 还能再快吗?几乎没有空间了 —— 已经 992/1008。想继续压只有两条路:
//
//   1. TILE 改大(64×64):共享内存占用变成 16KB,会限制每个 SM 能同时驻留的
//      块数(occupancy)。多数情况下得不偿失,值得自己测一次。
//   2. 用 float4 向量化:一次搬 16 字节。这能减少指令数,但带宽已经是瓶颈,
//      收益有限 —— 而且要求矩阵维度与对齐满足条件,非方阵用例上反而要特殊处理。
//
// 一个更值得练的方向:把这道题和 01/02 对比着看 ——
// 那两道题怎么改都是合并访问,所以感受不到访存模式的威力;
// 这道题天然有一侧不合并,才让共享内存有了用武之地。
// ---------------------------------------------------------------------------

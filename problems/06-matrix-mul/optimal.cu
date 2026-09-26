// 题目 06 的参考解 —— 共享内存分块 + 4×4 寄存器分块
//
// 思路只有一句话:让每一次访存都尽量多喂几次 FMA。
//
//   朴素版(基线):1 次读 → 1 次 FMA        (算术强度 0.25 FLOP/float)
//   只做共享内存分块:1 次读 → 1 次 FMA     ← 实测只快 17%,见下
//   本实现(再叠 4×4 寄存器分块):8 次读 → 16 次 FMA (算术强度 2 FLOP/float)
//
// 实测(4090,n=2048,框架自动清 L2):
//     基线(1 线程 1 输出)                3230 µs   1.00x
//     只做共享内存分块(TILE=32,R=1)      2887 µs   1.17x
//     分块 + 2×2 寄存器分块               986 µs   3.41x
//     本实现                              562 µs   5.74x   ≈ 30.6 TFLOPS
//
// ---------------------------------------------------------------------------
// 为什么第一步(只分块)几乎没用 —— 这道题最容易踩空的地方
//
// 共享内存和 L1 在芯片上是**同一块 SRAM**,LDS 和 LDG 走同一条访存流水线,
// 吞吐都是每 SM 每 cycle 128B。所以「把全局读换成共享内存读」并不会让访存
// 指令变少 —— 一个线程算一个输出时,每个 FMA 仍要配 2 次共享内存读
// (从 A 取一个、从 B 取一个),上限被钉死在「每 cycle 128B ÷ 8B = 16 次 FMA」,
// 也就是峰值算力的 1/8。基线本来就有 L1 和广播帮着兜底,两者于是几乎打平:
// 实测 1.17x —— 分块这一步**单独做等于白做**。
//
// 真正的杠杆在**寄存器分块**:一个线程算 R×R 个输出,那么这一轮 k 只需要
// 读 R 个 A 值 + R 个 B 值,就能做 R² 次 FMA。R=4 时每次访存摊到 2 次 FMA,
// 是「一读一算」的 4 倍。实测 R=2 拿 3.41x、R=4 拿 5.74x —— 收益主要在这一步。
//
// 代价是 R² 个累加器必须待在寄存器里,所以**所有下标都必须是编译期常数**:
// 内层循环靠 #pragma unroll + 常数边界展开。一旦下标是运行时变量,编译器会把
// acc[][] 甩到 local memory(实际就是显存),性能比基线还差 —— 这是这类
// kernel 最典型的翻车方式,和 05 题里「kw 是运行时值」是同一个坑。
//
// ---------------------------------------------------------------------------
// 参数选择:TILE=64、块内 16×16 线程、每线程 4×4
//
//   * 共享内存 2×64×64×4B = 32KB → 一个 SM(228KB)能驻留 3 个块(用了
//     __launch_bounds__ 明确要求 3,免得编译器按自己的偏好把寄存器用满)
//   * 复用度:一个字节从全局搬进来后被 64 行/列复用 → 全局请求量降到基线的 1/64
//   * 每线程 16 个累加器 + 8 个操作数,寄存器压力约 40 个,离 255 的上限很远
//
// 【试过的死路,供参考 —— 都有实测数字】
//   * 只做共享内存分块(1 输出/线程):1.17x。共享内存带宽把上限钉在
//     16 FMA/cycle/SM(峰值的 1/8),而基线本来就有 L1 兜着 —— 白做
//   * 2×2 寄存器分块:3.41x。方向对了,但每次访存只摊到 1 次 FMA,不够
//   * TILE=128:共享内存 2×128×128×4 = 128KB,一个 SM 连一个块都放不下
//     (4090 每 SM 上限 100KB),occupancy 直接崩掉
//   * 块内 32×32 线程配 4×4 分块:TILE 被迫到 128,同样撞共享内存上限
//   * R=8(每线程 64 个输出):累加器就要 64 个寄存器,加上操作数和地址,
//     寄存器溢出到 local memory,反而变慢
// ---------------------------------------------------------------------------

#include "ctx.h"

#define TILE  64                          // 一个块负责 out 的 TILE×TILE
#define TSIDE 16                          // 块内 TSIDE×TSIDE = 256 线程
#define REG   4                           // 每线程 REG×REG 个输出
#define NTHREADS (TSIDE * TSIDE)

__global__ void __launch_bounds__(NTHREADS, 3)
matmul_tiled(const float* __restrict__ a,
             const float* __restrict__ b,
             float* __restrict__ out,
             int n) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TSIDE + tx;
    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;

    float acc[REG][REG];
#pragma unroll
    for (int i = 0; i < REG; ++i)
#pragma unroll
        for (int j = 0; j < REG; ++j) acc[i][j] = 0.0f;

    const int ntile = (n + TILE - 1) / TILE;

    // 本线程负责的那 R×R 个输出在 tile 内的起点
    const int sr = ty * REG;
    const int sc = tx * REG;

    for (int t = 0; t < ntile; ++t) {
        const int k0 = t * TILE;

        // ---- 协作搬运两块 TILE×TILE,越界的位置填 0 ----
        // 扁平索引 + 跨步:相邻线程读相邻的列,全局读是合并的。
        // 越界填 0 而不是跳过,是为了让下面的 k 循环一个分支都不用判
        // (0 乘任何数都是 0,不影响结果)。
#pragma unroll
        for (int i = tid; i < TILE * TILE; i += NTHREADS) {
            const int r = i / TILE;
            const int c = i % TILE;

            const int ar = row0 + r, ac = k0 + c;
            as[r][c] = (ar < n && ac < n) ? a[(long long)ar * n + ac] : 0.0f;

            const int br = k0 + r, bc = col0 + c;
            bs[r][c] = (br < n && bc < n) ? b[(long long)br * n + bc] : 0.0f;
        }
        __syncthreads();

        // ---- 计算:每轮 k 读 4+4 个共享内存值,做 16 次 FMA ----
        // bs 按 float4 读:bs 的行距是 64 个 float(256B),tx*REG 又是 4 的倍数,
        // 所以这一定是 16B 对齐的。一条 LDS.128 顶四条标量 LDS。
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float ar[REG];
#pragma unroll
            for (int i = 0; i < REG; ++i) ar[i] = as[sr + i][k];

            const float4 bv = *reinterpret_cast<const float4*>(&bs[k][sc]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};

#pragma unroll
            for (int i = 0; i < REG; ++i)
#pragma unroll
                for (int j = 0; j < REG; ++j)
                    acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }

        // 这一处最容易漏:不等所有线程读完就搬下一块,快的线程会把慢的线程
        // 还在读的数据覆盖掉 —— 而且往往「有时对有时错」。
        __syncthreads();
    }

    // ---- 写回,判边界 ----
    // 越界填 0 让计算免判边界,但**写回必须判**:n=1531 时最后一块 tile
    // 会伸出矩阵外,写出去就踩到别的缓冲(框架的哨兵区会当场抓到)。
#pragma unroll
    for (int i = 0; i < REG; ++i) {
        const int gr = row0 + sr + i;
        if (gr >= n) continue;
#pragma unroll
        for (int j = 0; j < REG; ++j) {
            const int gc = col0 + sc + j;
            if (gc < n) out[(long long)gr * n + gc] = acc[i][j];
        }
    }
}

void matmul_launch(LaunchCtx& ctx) {
    const dim3 block(TSIDE, TSIDE);
    const dim3 grid((ctx.n + TILE - 1) / TILE, (ctx.n + TILE - 1) / TILE);
    matmul_tiled<<<grid, block>>>(ctx.a, ctx.b, ctx.out, ctx.n);
}

// ---------------------------------------------------------------------------
// 值得亲手验证的两件事:
//
// 1. **只做共享内存分块会怎样?** 把 REG 改成 1(块内 64×64 线程、每线程
//    一个输出)跑一遍。共享内存流量翻 4 倍,你应该看到耗时明显回升 ——
//    这能让你亲眼确认「LDS 和 LDG 共用同一条流水线」不是一句空话。
//
// 2. **全局读要不要向量化?** 搬运 tile 时把 `a[...]` 换成 float4 读,
//    只有当 n % 4 == 0 时才是合法的(n=1531 时行首不是 16B 对齐的,
//    硬转会读到错位的数据或直接崩)。所以真实库要么对 n 做对齐分支,
//    要么整个走标量路径 —— 本题 n=2048 和 n=100 都满足 n%4==0,
//    但 n=1531 不满足,这就是为什么参考解老老实实用了标量搬运。
// ---------------------------------------------------------------------------

// 题目 16 的参考解:__ballot_sync —— 一条 warp 指令凑出一个 word。
//
// 这道题**只有一个洞察**,剩下的都是边界细节:
//
//   输出的粒度比线程大。一个 word 装 32 个元素的结论,而一个线程只知道自己
//   那一个。基线的做法是"抢"(32 个 lane 去 atomicOr 同一个地址,每次都要
//   一次 L2 往返),参考解的做法是"投票"(__ballot_sync:32 个 lane 各交一个
//   bool,一条指令返回一个 u32)。
//
//   实测(2026-09-29,4090,框架自动清 L2,big 用例 n = 2^26 = 264MiB 访存):
//       基线(一元素一线程 + atomicOr)   1.019 ms   272 GB/s   27%
//       本实现(__ballot_sync)           0.293 ms   945 GB/s   94%   → 3.48x
//   94% 已经贴着"读一遍"这条物理上限(264MiB / 1008 GB/s = 275µs;剩下那 6%
//   是 DRAM 跑不满标称值 + 少量固定开销)。
//
// 实现上就三件事:
//   ① 每个 lane 算自己的谓词 —— 越界的 lane 给 false,**但不能提前 return**;
//   ② __ballot_sync(0xffffffff, pred) 凑出这一组 32 个元素的 word;
//   ③ 每个 warp 的 lane 0 把结果写下去(lane 0 的下标正好是 word 的左端点)。
//
// 为什么 lane 0 的下标就是 word 的起点:blockDim.x 是 32 的倍数,所以
// warp w 的 32 个 lane 拿到的下标是 base..base+31,而 base = blockIdx.x*blockDim.x
// + w*32 一定是 32 的倍数 ⇒ 这 32 个元素正好落在 word (base>>5) 里。
// 这不是巧合,是"线程映射必须和输出粒度对齐"这条要求的直接后果 ——
// 如果让每个线程跳着取数(比如 threadIdx.x*gridDim.x+blockIdx.x),投票出来的
// 32 个位就不属于同一个 word,整个方案立刻垮掉。
//
// ⚠️ 但**别把这道题的答案记成"要用 __ballot_sync"**。实测把四条完全不同的路
//    都跑了一遍,它们全部落在 84%~94%:
//
//      __ballot_sync（本文件）                          945 GB/s   94%
//      块内私有化（共享内存原子 + 每 word 单写者）       945 GB/s   94%
//      共享内存中转（strided 读,32 路 bank conflict）   929 GB/s   92%
//      一线程一个 word + float4 读                       842 GB/s   84%
//      多元素一线程 + atomicOr（争用降到 8 路以下）      881 GB/s   87%
//
//    它们唯一的共同点是:**x 合并地读一遍、每个 word 只有一个写者**。
//    到了这一步,手段已经不重要了 —— 这就是"贴到墙了"的意思。
//    值钱的判断是"我还在不在抢同一个地址",不是"我用没用哪条指令"。

#include "ctx.h"

__global__ void pack_bits(const float* __restrict__ x,
                          unsigned int* __restrict__ bits,
                          int n, int nw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ① 越界的 lane 谓词为 false。
    //    注意这里**不能**写成 `if (i >= n) return;` ——
    //    __ballot_sync 要求 mask 里点名的每一个 lane 都执行到这条指令,
    //    提前退出就是"部分线程参与 barrier",未定义行为(实测会给出残缺的
    //    word,而且 racecheck / synccheck 会报出来)。
    //    写成一个谓词就没有这个问题:i >= n 的 lane 照样走到 ballot,只是交 false。
    const bool pos = (i < n) && (x[i] > 0.0f);

    // ② 一条指令,32 个 lane 的谓词变成一个 word。第 k 位 = 第 k 个 lane 的 pos。
    const unsigned int m = __ballot_sync(0xffffffffu, pos);

    // ③ 每个 warp 只让 lane 0 写。lane 0 的 i 就是这一组 32 个元素的下标左端点,
    //    所以它管的 word 就是 i>>5。
    //    最后一个块里可能有整段越界的 warp(它们的 lane 0 的 i 也 >= n),
    //    这时 i>>5 可能 >= nw —— 必须挡住,否则会写到哨兵区。
    if ((i & 31) == 0) {
        const int w = i >> 5;
        if (w < nw) {
            bits[w] = m;
        }
    }
}

void pack_bits_launch(LaunchCtx& ctx) {
    // 块大小必须是 32 的倍数 —— 否则一个 warp 会横跨两个块,第 ③ 步就错了。
    // 256 和 512 都在最优点附近;这题的瓶颈是流量,块大小几乎不敏感。
    const int block = 256;
    const int grid = (ctx.n + block - 1) / block;   // 向上取整
    pack_bits<<<grid, block, 0, ctx.stream>>>(ctx.x, ctx.bits, ctx.n, ctx.nw);

    // 没有 memset:每一个 word 都被恰好一个 lane 0 线程**完整地覆盖写**,
    // 所以毒值不需要擦。这是 ballot 方案相对原子方案的第二笔好处
    // (第一笔是不用抢)。验证一下"No word 被漏掉":
    //   对任意 w < nw,元素 32w 一定存在(因为 nw = ceil(n/32) ⇒ 32w < n),
    //   而 grid*block >= n > 32w ⇒ 一定有线程拿到下标 32w ⇒ 它一定是某个 warp
    //   的 lane 0 ⇒ 它一定写了 bits[w]。证毕。
}

// ---------------------------------------------------------------------------
// 值得亲手验证的三件事:
//
// 1. **换个"每线程管 32 个连续元素"的写法会怎样?**
//
//        int w = blockIdx.x * blockDim.x + threadIdx.x;
//        if (w >= nw) return;
//        unsigned int m = 0u;
//        for (int b = 0; b < 32; ++b) {
//            int i = (w << 5) + b;
//            if (i < n && x[i] > 0.0f) m |= (1u << b);
//        }
//        bits[w] = m;
//
//    它正确、不需要投票指令,而且每个 word 也是被完整覆盖写。
//    但它把"连续性"从 warp 手里拿走了:同一个 warp 的 32 个 lane 访问的地址
//    相隔 32 个 float = 128 字节,一条 load 指令要碰 32 个不同的 sector,
//    L1 的取数压力涨 8 倍。这是 13 题那个坑的另一种形态 ——
//    **"一个线程管一段连续数据"在 GPU 上几乎总是错的**。
//
// 2. **把 atomicOr 的地址打散会怎样?**
//    把 `i >> 5` 故意改成 `(i >> 5) & 7`,结果会错,但速度快很多。
//    这个实验说明:原子操作的代价 ∝ 争用程度,不 ∝ 操作数。
//    (07 题直方图讲的是同一件事的另一面:私有化。)
//
// 3. **tiny 用例(262147 个元素、8193 个 word)上,基线和参考解差多少?**
//    实测:big 上差 3.48 倍,tiny 上只差 2.0 倍(0.008 ms vs 0.004 ms)。
//    争用要有足够多的 warp 同时在场才显出来,数据量一小,"32 个 lane 抢一个
//    门"的排队根本排不起来。**"慢"从来不是某个指令的属性,而是"在这个规模、
//    这个访存模式下"的属性。**
// ---------------------------------------------------------------------------

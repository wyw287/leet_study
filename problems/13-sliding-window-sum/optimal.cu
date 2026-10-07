// 题目 13 的参考解 —— 滚动和,但让 warp 来承担"连续"这件事
//
// 先说一个**反直觉的实测结果**,这题的坑就在这里(big = n=16777216, w=512):
//
//     基线:一输出一线程,窗口加一遍                1.683 ms    80 GB/s    8%
//     滚动和 + 每线程管一段连续输出(CHUNK=128)      0.700 ms   192 GB/s   19%
//     两趟全局:前缀和落 tmp + 再做差(4 个 kernel)   0.415 ms   324 GB/s   32%
//     本实现                                       0.135 ms   993 GB/s   99%
//
// 中间那两行是这道题真正的"学费":
//   * 滚动和本身**一分钱都不值**。它要求一个线程管连续输出 ⇒ 同一 warp 的
//     32 个 lane 访问的地址相隔 CHUNK 个 int(512 字节),一条 load 指令覆盖
//     32 个 sector,每个 lane 还独吞 4 条 cache line —— 几十个 warp 一起把 L1
//     冲垮。**"少算"做对了,却把"会合"赔了进去**(08 题那个坑,原样搬了过来)。
//   * 两趟方案是**完整而正确的**,32% 不是手艺问题,是账:中间结果落一次显存,
//     流量 8 → 16 字节/元素,物理上限就是峰值的一半。
//
// 本实现:把"连续"这件事交给 **warp** —— 一个 warp 领一段连续输出,
// warp 内部的 32 个 lane 仍然**交错**着取数(合并访问保住),而"滚动"这件事
// 由 warp 内的一次前缀和(shuffle scan)完成:
//
//   第 i 个输出 = carry + Σ_{j=lo..i} (x[j] - x[j-w])
//
// 那个 Σ 就是 d[j] = x[j] - x[j-w] 的**前缀和**,而 warp 内做前缀和只要
// 5 步 __shfl_up_sync(所以 10 题学的 scan 在这里派上用场)。
// 每个输出摊到的指令数:
//     2 条 load(x[i] 和 x[i-w],都是合并的)+ 1 条 sub
//   + 5 步 scan 的 5/32 + 1 条 add(加 carry)+ 1 条 store ≈ 5.3 条
// 对比"每个输出 ~80 条 load"的配额 —— 于是又回到带宽墙前面。
//
// ragged 用例(n=16777259, w=509)上本实现是 0.140 ms / 957 GB/s / 95%,
// 和 big 一致 —— 说明边界处理与 w 是不是 2 的幂都不影响它。

#include "ctx.h"

// 一个 warp 领多少个输出(必须是 32 的倍数)。越大,初始化那笔 O(w)
// 的钱摊得越薄;越小,并行度越高(4090 有 128 个 SM,别把自己饿死)。
#define SPAN 2048

__global__ void window_sum(const int* __restrict__ x,
                           int* __restrict__ out,
                           int n, int w) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const int lo = warp * SPAN;              // 本 warp 负责 [lo, hi)
    if (lo >= n) return;
    int hi = lo + SPAN;
    if (hi > n) hi = n;

    // ---- carry = out[lo-1] = 窗口 [max(0,lo-w), lo-1] 的和 ----
    // 这笔钱是 O(w),但只花一次,warp 内 32 个 lane 分摊 + 一次归约,
    // 摊到 SPAN 个输出上可以忽略。
    int carry = 0;
    {
        int clo = lo - w;
        if (clo < 0) clo = 0;
        int c = 0;
        for (int j = clo + lane; j < lo; j += 32) {
            c += x[j];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            c += __shfl_down_sync(0xffffffffu, c, off);
        }
        carry = __shfl_sync(0xffffffffu, c, 0);
    }

    // ---- 逐轮推进:lane l 负责输出 i = i0 + l ----
    // lane 之间取的是**连续**地址(合并),轮与轮之间靠 carry 串起来。
    for (int i0 = lo; i0 < hi; i0 += 32) {
        const int i = i0 + lane;

        // d[i] = x[i] - x[i-w];前 w 个输出没有东西可减(窗口被裁到 0),
        // 此时 d = x[i] —— 前缀和自然给出 sum(x[0..i])。
        // 尾巴上超出 hi 的 lane 贡献 0,不参与求和也不写出。
        int d = 0;
        if (i < hi) {
            d = x[i];
            if (i >= w) d -= x[i - w];
        }

        // warp 内 inclusive 前缀和(Hillis-Steele,5 步)。第 k 步之后,
        // 每个 lane 拿到的是它往前 2^k 个 lane 的和。
        int s = d;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int t = __shfl_up_sync(0xffffffffu, s, off);
            if (lane >= off) s += t;
        }

        if (i < hi) out[i] = carry + s;

        // 这一轮的总和 = lane 31 的前缀和;下一轮的起点。
        carry += __shfl_sync(0xffffffffu, s, 31);
    }
}

void window_sum_launch(LaunchCtx& ctx) {
    const int block = 256;                       // = 8 个 warp
    const int warps_per_blk = block / 32;
    const int grid = (ctx.n + warps_per_blk * SPAN - 1) / (warps_per_blk * SPAN);
    window_sum<<<grid, block>>>(ctx.x, ctx.out, ctx.n, ctx.w);
}

// ---------------------------------------------------------------------------
// 实测把这几条路放在一起看(数字见 spec.yaml):
//
//   朴素:窗口加一遍                          w+1 条 load/输出      8%
//   滚动和 + 每线程一段连续输出                指令少一个数量级      11%~19%
//   两趟全局:前缀和 + 差分                     流量 16 字节/元素     32%
//   本实现:滚动和 + warp 交错 + shuffle scan   ~5 条指令/输出        99%
//
// 第三条和第四条之间**没有斜坡**:两趟方案的上限被它的流量钉死在 50%,
// 想过 80% 只能把中间那趟省掉 —— 而那要求"连续"这件事由 warp 或 block
// 来承担。这就是这道题的形状。
//
// ---------------------------------------------------------------------------
// 还能再往哪走?
//
// **[块内共享内存 + 块内前缀和 + 差分]** 把 x 的一段(含左边 w-1 个元素的
// 光环)合并地读进共享内存,在共享内存里做块内前缀和 PS,再让每个输出读
// 两个位置:out[i] = PS[i] - PS[i-w]。它比本实现多买到的,是**共享内存里的
// 前缀和可以整块复用**(本实现每一轮都要重算一次 warp scan);代价是多一次
// 共享内存往返和一个 __syncthreads,而且要当心 bank 冲突:lane 之间相隔
// CHUNK 个 int,CHUNK 是 32 的倍数时 32 个 lane 会全部落在同一个 bank 上。
// 注意:块内前缀和的"进位"不用担心 —— PS[i] 和 PS[i-w] 里的进位是同一份,
// 相减自动抵消,**跨块进位根本不存在**。
//
// **[chained scan 版的两趟]** 想留在两趟那一档又想把 32% 往上推:把"块内和
// → 单块扫描 → 加 carry"三个 kernel 合成一个(11 题的 decoupled lookback),
// 流量就从 20 字节/元素降到 16 字节/元素 —— 上限仍然是 50%,但能更贴近它。
// 这条路的价值在于它演示了一个取舍:**有些时候"多搬一趟"比"融合"更划算**,
// 前提是你把两边的账都算清楚了。
// ---------------------------------------------------------------------------

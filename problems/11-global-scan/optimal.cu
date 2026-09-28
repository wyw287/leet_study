// 题目 11 的参考解(三阶段跨块扫描)
//
// 核心思路:一条 3200 万长的依赖链,一个块装不下,块之间又没有同步原语 ——
// 所以先把问题**缩小**,再放大回去:
//
//   阶段 1  chunk_sum   每个块只回答一个小问题:"我这一段的和是多少?"
//                       → part[0..nblk-1],一共 1024 个数
//   阶段 2  scan_part   就 1024 个数,一个 block 扫完 → 变成"每个块的排他偏移"
//   阶段 3  global_scan 每个块回到自己那段,做**真正的扫描**并加上偏移
//
// 为什么阶段 2 必须是另一个 kernel:阶段 1 的所有块必须先写完 part,
// 阶段 2 才能读。而块与块之间没有同步原语,唯一现成的全局屏障就是 kernel 边界。
//
// 为什么阶段 3 要重新读一遍 x:阶段 1 只留下了"段和",段内每个元素的值早丢了
// (128MiB 装不进片上)。所以总流量 = n(阶段1读)+ n(阶段3读)+ n(阶段3写)= 3n。
// 这 3n 就是本题的天花板,不是实现不够好 —— 想做成 2n 只有一条路:
// 单遍扫描 + decoupled look-back(边扫边等前驱块的**包含**前缀),
// 那要原子操作 + 内存序,见题面思考题 3。
//
// 这道题里"块内"那部分怎么做几乎不影响成绩(DRAM 流量被题目定死),
// 所以参考解就用最直白的 tile + 块内 Hillis-Steele —— 好读、好验证。
// 真正值钱的是**认出总功可以从 O(n log n) 降到 O(n)**,以及跨块的那两刀切在哪。

#include "ctx.h"

#define BLOCK 256

// ---------------------------------------------------------------------------
// 阶段 1:块内归约 —— 第 blockIdx.x 段的元素之和 → part[blockIdx.x]
// ---------------------------------------------------------------------------
__global__ void chunk_sum(const float* __restrict__ x,
                          float* __restrict__ part,
                          int n, int chunk) {
    __shared__ float sm[BLOCK];
    const int t = threadIdx.x;
    const int beg = blockIdx.x * chunk;          // n ≤ 2^25,乘出来不会溢出 int
    const int end = min(n, beg + chunk);

    // 跨步循环:同一 warp 的 32 个线程地址连续,一次 128 字节的访问全用上
    float s = 0.0f;
    for (int i = beg + t; i < end; i += BLOCK) s += x[i];

    sm[t] = s;
    __syncthreads();
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (t < st) sm[t] += sm[t + st];
        __syncthreads();
    }
    // 空段(end <= beg)的 s 是 0,这里照样写一个 0 出去 —— 必须写。
    // part 在每次校验前会被填成毒值,漏写一个,NaN 就会顺着块间偏移
    // 污染它后面所有的块。
    if (t == 0) part[blockIdx.x] = sm[0];
}

// ---------------------------------------------------------------------------
// 阶段 2:把 part[0..nblk-1] 原地变成**排他**前缀和
//        (part[b] 从"段 b 的和"变成"段 0..b-1 的和")
//
// nblk ≤ 1024,一个 block 就够 —— 这点工作量怎么扫都快,不是本题的瓶颈。
// 用 blockDim = 1024 启动;t >= nblk 的线程补 0,不参与结果。
// ---------------------------------------------------------------------------
__global__ void scan_part(float* __restrict__ part, int nblk) {
    __shared__ float sm[1024];
    const int t = threadIdx.x;
    const float v = (t < nblk) ? part[t] : 0.0f;   // 自己那份,等会儿要减掉

    sm[t] = v;
    __syncthreads();

    // 块内 Hillis-Steele。原地扫不能直接写:线程 t 要读 sm[t-s],
    // 而线程 t-s 可能在同一轮里已经把它改掉了。所以"先读进寄存器 →
    // 屏障 → 再写回",两个屏障一轮。
    for (int s = 1; s < nblk; s <<= 1) {
        const float add = (t >= s) ? sm[t - s] : 0.0f;
        __syncthreads();
        sm[t] += add;
        __syncthreads();
    }

    // sm[t] 现在是**包含**前缀;题目要的块偏移是**排他**前缀,差自己那一份。
    if (t < nblk) part[t] = sm[t] - v;
}

// ---------------------------------------------------------------------------
// 阶段 3:每段真正扫描一遍,加上块间偏移,写出 y
//
// 段长 32768 远大于 BLOCK,所以一个 block 要一个 tile 一个 tile 地走,
// 并把上一个 tile 的总和作为**进位**带进下一个 tile。
// 注意前两个阶段做完之后 part[b] 已经是排他偏移,直接当初始进位用。
// ---------------------------------------------------------------------------
__global__ void global_scan(const float* __restrict__ x,
                            float* __restrict__ y,
                            float* __restrict__ part,
                            int n, int nblk) {
    __shared__ float sm[BLOCK];
    const int t = threadIdx.x;
    const int chunk = (n + nblk - 1) / nblk;
    const int beg = blockIdx.x * chunk;
    const int end = min(n, beg + chunk);

    float carry = part[blockIdx.x];

    for (int base = beg; base < end; base += BLOCK) {
        const int i = base + t;
        // 越界补 0:既避免越界读,又让 sm[BLOCK-1] 天然等于本 tile 的
        // **有效元素之和** —— 尾巴那一段不用额外判断,进位自然是对的。
        sm[t] = (i < end) ? x[i] : 0.0f;
        __syncthreads();

        for (int s = 1; s < BLOCK; s <<= 1) {
            const float add = (t >= s) ? sm[t - s] : 0.0f;
            __syncthreads();
            sm[t] += add;
            __syncthreads();
        }

        // 本轮 tile 的总和 = 扫描结果最后一个槽位。所有线程都要用它更新进位。
        const float last = sm[BLOCK - 1];
        // 这道屏障不能省:last 是**所有线程**都要读的同一个槽位,
        // 少了它,持有 sm[BLOCK-1] 的那个线程可能已经进入下一轮并把它
        // 覆盖成本 tile 的第一个元素 —— 进位就错了,而且只在特定调度下错。
        __syncthreads();

        if (i < end) y[i] = sm[t] + carry;
        carry += last;
    }
}

void global_scan_launch(LaunchCtx& ctx) {
    const int chunk = (ctx.n + ctx.nblk - 1) / ctx.nblk;

    chunk_sum<<<ctx.nblk, BLOCK>>>(ctx.x, ctx.part, ctx.n, chunk);
    scan_part<<<1, 1024>>>(ctx.part, ctx.nblk);
    global_scan<<<ctx.nblk, BLOCK>>>(ctx.x, ctx.y, ctx.part, ctx.n, ctx.nblk);
}

// ---------------------------------------------------------------------------
// 实测(4090,框架自动清 L2,big 用例 n=2^25):
//     基线(Hillis-Steele 25 趟)  7.51 ms   1.00x
//     这份参考解                  0.455 ms  16.51x   ← 3n 天花板(399µs)的 88%
// 加速比的上限是 50n/3n ≈ 16.7x,所以 16.51x 基本就是"贴着墙"了。
//
// ---------------------------------------------------------------------------
// 想自己验证的两件事:
//
// 1. **块内怎么扫,对成绩几乎没有影响。** 实测:把 BLOCK 从 256 改成 512
//    (tile 变长、每块的 __syncthreads 次数减半),耗时 0.455 → 0.451 ms,
//    差 0.9%,在噪声以内。DRAM 流量一个字节都没变,而 16 次屏障摊到 2KB 的
//    tile 上本来就是零头。**本题的钱不在块内。**
//
//    真正决定成绩的是并行度:把阶段 3 换成"线程 0 串行扫一整段",
//    结构完全一样、总功也还是 3n,耗时却是 1.730 ms(4.34x)——
//    因为全卡只剩 1024 个线程、每线程 1 个在途 load,有效带宽掉到 ~170GB/s。
//    这是延迟受限,不是带宽受限。
//
// 2. **阶段 2 能不能并进阶段 1 或阶段 3?** 不能省掉"所有段和都写完之后
//    才能开始扫"这个顺序约束。但有一类技巧可以:让阶段 3 里每个块
//    **自己**把 part[0..b-1] 累加一遍(而不是读阶段 2 的结果)——
//    那样省掉一次 kernel 启动,代价是 O(nblk²) 次读,而它在 L2 里。
//    两个大用例上这只是几微秒的差别,可以自己试。
// ---------------------------------------------------------------------------

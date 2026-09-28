// 题目 11 的性能基线:Hillis-Steele 直接铺在整个数组上。
//
// 这道题的基线**不是最笨的写法** —— 最笨的是"一个线程从头扫到尾",
// 实测 1327.9 ms,**比这条基线还慢 176 倍**。这里没有拿它当基线,因为那会让
// 加速比的刻度失去意义:0.006x 到 16.5x 跨了三个半数量级,S/A/B/C 四个档次
// 会全挤在一起;而且一条延迟受限的串行链把"并行度不够"和"多做了 log n 倍功"
// 两件事混在一起,而这道题要教的只有后面那一件。
// 这条基线是结构上真并行、只是多做了 log n 倍功 —— 正好把要教的东西隔离出来。
//
// 这条基线是"在 10 题里学会 Hillis-Steele 之后,原样套到一整个大数组上"的结果:
//
//   第 1 趟:每个元素加上它左边 1 个元素
//   第 2 趟:每个元素加上它左边 2 个元素
//   ...
//   第 p 趟:每个元素加上它左边 2^(p-1) 个元素
//   趟数 = ceil(log2(n))。n = 2^25 时是 25 趟。
//
// 它每一趟都是对的,每一趟也都跑满了带宽 —— 唯一的毛病是**总功**:
// 25 趟 × (读 134MB + 写 134MB) = 6.7GB,而两遍法只要 0.40GB。
// 换句话说:基线不是"写得烂",它是"做了约 17 倍的无用功"。
// 所以本题用**加速比**评级 —— 两者的有效带宽都是 ~890GB/s,
// 用"占峰值带宽百分比"根本分不出好坏。
//
// 趟与趟之间必须有一次全局屏障(否则第 p 趟会读到第 p 趟自己正在写的值 ——
// 这正是"原地 Hillis-Steele"错在哪)。CUDA 里唯一现成的全局屏障就是
// **kernel 边界**,所以这里老老实实启动 25 次(launcher 是 host 代码,
// 循环启动 25 个 kernel 是合法写法,而且它们天然串行)。
//
// 乒乓:一趟的输入是上一趟的输出,所以需要两块缓冲交替用。
// ctx.part 就是那个和输入等长的暂存区 —— 它在 spec 里存在的理由正是这条基线。
// 结果最终必须落在 ctx.y 上,所以用"总趟数的奇偶"决定第一趟往哪儿写。

#include "ctx.h"

#define BLOCK 256

__global__ void global_scan(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n, int step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // i < step 时左边没有东西可加 —— 补 0 就是单位元,不必分支到另一条路径
        out[i] = in[i] + (i >= step ? in[i - step] : 0.0f);
    }
}

void global_scan_launch(LaunchCtx& ctx) {
    const int n = ctx.n;

    // 趟数 = ceil(log2(n)):最后一个 step 必须满足 2*step >= n,
    // 否则数组开头那一段还没被累加进最后一个元素。
    int passes = 0;
    for (int s = 1; s < n; s <<= 1) ++passes;

    const int grid = (n + BLOCK - 1) / BLOCK;

    const float* src = ctx.x;
    int p = 0;
    for (int s = 1; s < n; s <<= 1, ++p) {
        // 总趟数是奇数 → 第 0 趟写 y(后面 2,4,... 趟也写 y,最后一趟正好在 y)
        // 总趟数是偶数 → 第 1 趟写 y
        const bool to_y = (passes & 1) ? ((p & 1) == 0) : ((p & 1) == 1);
        float* dst = to_y ? ctx.y : ctx.part;
        global_scan<<<grid, BLOCK>>>(src, dst, n, s);
        src = dst;
    }
}

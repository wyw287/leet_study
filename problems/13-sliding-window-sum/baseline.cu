// 题目 13 的性能基线:一个线程算一个输出,老老实实把窗口里的 w 个数加一遍。
//
// 这份实现**完全正确**,而且显存那一侧是满分的 —— 这一点必须先说清楚,
// 因为它决定了这道题到底在考什么:
//
//   * **合并访问满分。** 同一个 warp 的 32 个线程在每个 j 上读的是连续的
//     32 个 int(128 字节 = 4 个 sector),一条 load 指令覆盖的地址是连着的。
//   * **显存流量也是满分。** 相邻输出的窗口重叠 w-1 个元素,所以每个 x
//     元素会被读 w 次;但这 w 次里只有 1 次落空,其余全是 L1 命中
//     (一个 warp 的活跃工作集只有 (w+32) 个 int ≈ 2KB)。
//     **显存上每个元素只走一次,8 字节/元素,一分不多。**
//
// 显存那本账满分,慢的原因在**另一本账**:
//
//   **L1 的取数带宽。** 每个输出要发 w+1 = 513 条 load。全卡 L1 的取数能力
//   约 1e13 条 load/s(128 SM × 32 lane/cycle × 2.5GHz 的量级),而显存墙
//   只允许 126e9 个输出/s(1008 GB/s ÷ 8 字节)—— 两者相除:
//
//       每个输出只有大约 80 条 load 的配额。
//
//   513 条超了 6 倍,所以它的天花板就是带宽墙的六分之一上下。
//   实测:1.683 ms / 80 GB/s / **8% 峰值** —— 而它的字节数一点没浪费。
//
// 要背下来的一句话:**命中 L1 不等于免费。** L1 的带宽比显存高一个数量级,
// 但它是有限的;当每个字节被取 w 次,这笔账就会盖过显存。
//
// (w 小的时候配额用不完,这份实现直接贴着带宽墙跑。同一份代码,窗口一小
//  就"没问题",窗口一大就"慢十倍" —— 慢不慢取决于把两本账一起算的结果。)

#include "ctx.h"

__global__ void window_sum(const int* __restrict__ x,
                           int* __restrict__ out,
                           int n, int w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // 数组开头那 w-1 个输出的窗口不满,裁到下标 0。
    int lo = i - w + 1;
    if (lo < 0) lo = 0;

    int acc = 0;
    for (int j = lo; j <= i; ++j) {
        acc += x[j];
    }
    out[i] = acc;
}

void window_sum_launch(LaunchCtx& ctx) {
    const int block = 256;
    const int grid = (ctx.n + block - 1) / block;
    window_sum<<<grid, block>>>(ctx.x, ctx.out, ctx.n, ctx.w);
}

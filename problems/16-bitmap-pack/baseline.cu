// 题目 16 的性能基线:一元素一线程 + atomicOr,最直白的写法。
//
// 它是**正确**的(原子操作保证同一个 word 的 32 个位不会互相踩掉),但它的
// 输出粒度是错的:一个 warp 的 32 个 lane 处理的正好是同一段 32 个连续元素,
// 于是它们的 i>>5 完全相同 —— **32 个 lane 全部去 atomicOr 同一个地址**。
//
// 实测(2026-09-29,4090,big 用例):1.019 ms / 272 GB/s / **27% 峰值**。
//
// 注意别把结论说成"原子操作慢" —— 那是错的。下面这组实测数字把真正的自变量
// 暴露得很清楚(同一个 kernel,只是每个线程多管几个元素、争用路数跟着变):
//
//     32 路争用(本文件)  272 GB/s   27%
//     16 路争用           444 GB/s   44%
//      8 路争用           881 GB/s   87%
//      4 路争用           878 GB/s   87%
//
// **原子指令一条都没少发**(每元素还是一条),8 路之后却直接贴着墙。
// 贵的从来不是原子操作,是"同一个地址被 32 个 lane 轮着访问"—— 每一次都要
// 一次 L2 往返。想自己验证的话,把 `i >> 5` 故意改成 `(i >> 5) & 7`
// (结果会错),速度会说明问题。
//
// 另一件必须注意的事:out 缓冲每轮被框架填成毒值(0x5EED5EED),所以原子版本
// **必须先清零**。这一趟 memset 也是要花钱的(8MiB,约 8µs)——
// "用原子就得先清零"本身就是原子方案的一笔附加账:参考解不需要它。

#include "ctx.h"

__global__ void pack_bits(const float* __restrict__ x,
                          unsigned int* __restrict__ bits,
                          int n, int nw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] > 0.0f) {
        atomicOr(&bits[i >> 5], 1u << (i & 31));
    }
}

void pack_bits_launch(LaunchCtx& ctx) {
    // 先清零 —— 少了这一句,毒值会留在没有正元素的那些 word 里。
    cudaMemsetAsync(ctx.bits, 0, (size_t)ctx.nw * sizeof(unsigned int), ctx.stream);

    int block = 256;
    int grid = (ctx.n + block - 1) / block;
    pack_bits<<<grid, block, 0, ctx.stream>>>(ctx.x, ctx.bits, ctx.n, ctx.nw);
}

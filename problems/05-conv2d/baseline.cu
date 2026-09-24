// 题目 05 的性能基线:一个线程算一个输出像素,kh*kw 个采样点全部从全局内存读。
//
// 这份实现**完全正确**,而且访存是合并的(同一个 warp 里相邻线程读相邻的列)。
// 它慢的原因不在 DRAM 带宽,而在「重复读」:
//
//   每个输出像素要读 kh*kw 个输入像素,而相邻输出像素的采样窗口大量重叠 ——
//   7×7 时每个输入像素平均被读了 49 次。这 49 次里只有第一次可能来自 DRAM,
//   后面 48 次命中 L1/L2 —— 也就是说数据没少搬,但 L1 到寄存器这条路的
//   流量被放大了 49 倍,而 L1 的带宽(每 SM 每 cycle 128B)是有限的。
//
// 另外还有一个不显眼但很贵的细节:采样窗口在 x 方向平移 kx 格之后,
// 一个 warp 的 32 个 float 往往不再落在 128B 对齐的边界上,于是本来一次
// 就能取完的 warp 访存要拆成两次 —— 49 个 tap 里有 42 个是这种「错位」的。
//
// 这些都是 baseline 要暴露的东西,也是使用者要修掉的东西。

#include "ctx.h"

__global__ void conv2d(const float* __restrict__ image,
                       const float* __restrict__ kernel,
                       float* __restrict__ out,
                       int h, int w, int kh, int kw, int oh, int ow) {
    // h 只有分块版填 halo 时才用得上;朴素版所有采样点都必然在界内。
    (void)h;

    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (oy >= oh || ox >= ow) return;

    float acc = 0.0f;
    for (int ky = 0; ky < kh; ++ky) {
        const float* row = image + (long long)(oy + ky) * w + ox;
        for (int kx = 0; kx < kw; ++kx) {
            acc += row[kx] * kernel[ky * kw + kx];
        }
    }
    out[(long long)oy * ow + ox] = acc;
}

void conv2d_launch(LaunchCtx& ctx) {
    dim3 block(32, 8);
    dim3 grid((ctx.ow + block.x - 1) / block.x, (ctx.oh + block.y - 1) / block.y);
    conv2d<<<grid, block>>>(ctx.image, ctx.kernel, ctx.out,
                            ctx.h, ctx.w, ctx.kh, ctx.kw, ctx.oh, ctx.ow);
}

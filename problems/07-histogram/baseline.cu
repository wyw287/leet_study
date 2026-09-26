// 题目 07 的性能基线:最朴素的 CUDA 实现 —— 一个线程读一个元素,
// 直接 atomicAdd 到全局的 bins 上。
//
// 这是「加速比的分母」,但本题的评分看的是**有效带宽占峰值的百分比**
// (见 spec.yaml 的说明和题面):朴素实现慢得离谱,慢的原因和显存一点关系都没有。
//
// 为什么慢:同一 warp 里 32 个线程的数据几乎落在 32 个不同的桶上,而
// **同一个地址**的原子加在 L2 里是被串行化的 —— 一个地址一次,排队。
// 3300 万次加法摊到 100 个有数据的桶上,每个桶要处理 33 万次;即便这些桶
// 分属不同的 L2 slice、完全并行,这条串行链也长达毫秒级。
// 而把 128MB 从头读一遍只需要约 130µs。
//
// 实测(4090,big 用例):基线约 6.2 ms、22 GB/s,只有峰值的 2%。
// 换句话说:基线根本没碰带宽的天花板,它撞的是原子单元的天花板。
//
// 另外注意 cudaMemsetAsync 那一行 —— bins 是 out 缓冲,框架在每次执行前
// 会把它填成毒值 0x5EED5EED(用来精确识别「kernel 没写输出」)。
// 累加型输出必须先自己清零,否则结果是「毒值 + 计数」,而且会稳定地错。

#include "ctx.h"

__global__ void histogram(const int* __restrict__ data,
                          int* __restrict__ bins,
                          int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = data[i];
        if (v >= 0 && v < nbins) {
            atomicAdd(&bins[v], 1);
        }
    }
}

void histogram_launch(LaunchCtx& ctx) {
    // 累加型输出:先清零。ctx.stream 为 nullptr 时就是默认流,顺序天然成立。
    cudaMemsetAsync(ctx.bins, 0, sizeof(int) * (size_t)ctx.nbins, ctx.stream);

    int block = 256;
    int grid = (ctx.n + block - 1) / block;   // 一个线程一个元素,不需要 grid-stride
    histogram<<<grid, block, 0, ctx.stream>>>(ctx.data, ctx.bins, ctx.n, ctx.nbins);
}

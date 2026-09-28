// 题目 10 的性能基线:最直白的"一个线程扫一行"。
//
// 这是"把题目翻译成 CUDA"的第一反应 —— 行与行互不相干,那就一行一个线程,
// 每个线程从行首累加到行尾。它**没有写错**:依赖链天然被满足,不需要任何
// 同步,也不会有竞态。它只是慢,而且慢的原因有两个,都能从物理上讲清楚:
//
//   ① 访存不合并。同一 warp 的 32 个线程分别在扫 32 个**不同**的行:
//      线程 r 读 x[r*cols + c],相邻线程的地址差 cols×4 = 4KB。
//      硬件是按 128 字节的 cache line 取数的,而一个线程一次只用其中 4 个字节
//      —— 有效带宽掉到 1/32 以下。(01 题里"把下标写成跨步的"那个实验,
//      这里是它的极端版本:不是跨步 32,而是跨步 4096 字节。)
//
//   ② 并行度只有 rows 个线程。4090 有 128 个 SM、能同时容纳十几万个线程,
//      这里只有 32768 个线程在干活(还大多挤在一个 block 里),而每个线程
//      内部是一条 **cols 长的依赖链**(acc += x[c] 必须一步一步来)。
//      显存延迟 600 周期,没有足够的 warp 去遮盖它。
//
// 修掉这两个毛病就是这道题的全部内容:把并行度从"行间"挪到"行内",
// 把访问从"跨行"变成"连续"。参考解见 optimal.cu。

#include "ctx.h"

__global__ void row_scan(const float* __restrict__ x,
                         float* __restrict__ y,
                         int rows, int cols) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;                 // rows 不整除线程总数时的边界

    const float* xr = x + (size_t)r * (size_t)cols;
    float*       yr = y + (size_t)r * (size_t)cols;

    float acc = 0.f;
    for (int c = 0; c < cols; ++c) {
        acc += xr[c];
        yr[c] = acc;
    }
}

void row_scan_launch(LaunchCtx& ctx) {
    const int block = 256;
    const int grid = (ctx.rows + block - 1) / block;
    row_scan<<<grid, block>>>(ctx.x, ctx.y, ctx.rows, ctx.cols);
}

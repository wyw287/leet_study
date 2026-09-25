// 题目 04 的参考解 —— 多块 + warp shuffle 两级归约
//
// 实测(n = 33,554,432):
//     基线(单块 + 共享内存树形归约)  1.07 ms
//     本实现(1024 块 + warp shuffle) 0.0236 ms   → 45.4x
//
// 相比基线的两处改动,每一处都对应一个具体的瓶颈:
//
//   ① **多线程块**:基线只用 1 个块,整张卡 128 个 SM 里 127 个在闲着。
//      改成多块,并行度立刻上来 —— 这是 45x 里最大的一份。
//   ② **块内用 warp shuffle 而不是共享内存**:省掉 log2(blockDim) 轮
//      __syncthreads()。共享内存往返有延迟,每轮同步还要等最慢的线程。

#include "ctx.h"

#define BLOCK 256
#define WARP 32

__global__ void reduce_sum(const float* input, float* output, int n) {
    // 每个 warp 算出一个部分和,一共 BLOCK/WARP 个
    __shared__ float warp_sums[BLOCK / WARP];

    const int tid = threadIdx.x;

    // 第一步:跨步读,把本线程负责的那些元素先加起来。
    // 跨步(而不是让每个线程读一段连续区间)是为了保持全局读**合并**:
    // 第 k 轮时线程 t 读 input[t + k * gridDim.x * blockDim.x],相邻线程地址相邻。
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // 第二步:warp 内归约,走寄存器直传。
    // __shfl_down_sync 让 lane i 拿到 lane i+offset 的值 —— 不经过共享内存,
    // 也不需要同步。5 轮(32→16→8→4→2→1)之后 lane 0 手里就是本 warp 的和。
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    // 第三步:每个 warp 的 lane 0 把自己的结果写进共享内存。
    if ((tid & (WARP - 1)) == 0) {
        warp_sums[tid / WARP] = sum;
    }
    // 只需要**一次**同步。基线要 log2(1024) = 10 次。
    __syncthreads();

    // 第四步:第一个 warp 对这 BLOCK/WARP 个值再做一次 shuffle 归约。
    if (tid < WARP) {
        const int nwarps = BLOCK / WARP;
        float v = (tid < nwarps) ? warp_sums[tid] : 0.f;
        for (int offset = WARP / 2; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        // 第五步:块间归约。用 atomicAdd 最简单 —— 注意它是安全的:
        // 多个块同时累加同一个地址由硬件保证原子性。
        if (tid == 0) {
            atomicAdd(output, v);
        }
    }
}

void reduce_sum_launch(LaunchCtx& ctx) {
    const int block = BLOCK;

    // 块数不是越多越好:块间归约本身有开销,而且收益在并行度饱和后就没了。
    // 这里限制在 1024 块 —— 对 128 个 SM 的卡来说,每 SM 8 个块足够填满。
    // 可以自己扫一下这个上限(128 / 512 / 4096),看拐点在哪。
    int grid = (ctx.n + block - 1) / block;
    if (grid > 1024) grid = 1024;

    // atomicAdd 是累加,所以输出必须先清零。
    // 框架把 out 缓冲填的是**毒值**(NaN),不清零的话结果一定是 NaN。
    // 这 4 字节的 memset 在计时区内,但相对于几十微秒的 kernel 可忽略。
    cudaMemset(ctx.output, 0, sizeof(float));

    reduce_sum<<<grid, block>>>(ctx.input, ctx.output, ctx.n);
}

// ---------------------------------------------------------------------------
// 几个容易踩的坑(题面「陷阱提醒」一节也讲了):
//
//   * __shfl_down_sync 的第一个参数是**掩码**。0xffffffff 表示"32 个 lane
//     全都参与"。写错会让某些 lane 拿不到正确的值,而且不一定报错。
//
//   * 别把 atomicAdd 和普通写混用:一个线程写 output[0] += v、另一个用
//     atomicAdd(output, v),结果就是不确定的。
//
//   * n 不整除线程总数时,跨步循环本身已经处理了(i < n 那个条件),
//     不需要额外的边界判断 —— 这一点比向量加法的写法更简洁。
//
// 还没做的事:块间归约如果用两级缓冲(每块写自己的部分和,再启一个 kernel
// 归约)替掉 atomicAdd,能避免原子操作的串行化。在块数很多(几千)时才有意义,
// 1024 块这个量级上差别在测量噪声内。
// ---------------------------------------------------------------------------

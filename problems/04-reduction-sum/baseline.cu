// 题目 04 的性能基线:单个线程块内的共享内存树形归约。
//
// 实现是对的(同步也齐),但只用了一个线程块 —— 整张卡 128 个 SM 里有 127 个
// 在闲着。数据量一大,一个 SM 的带宽就成了瓶颈。
//
// 这是「并行度不足」的典型形态,也是本题要你修掉的第一件事。
// 基线本身就用了共享内存,所以想大幅超越它必须再往前一步:
// 多线程块 + 块内用 warp shuffle 减少共享内存往返 + 块间二级归约。

#include "ctx.h"

#define BLOCK 1024

__global__ void reduce_sum(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;

    // 每个线程先把自己那一份加完(跨步访问 → 全局内存是合并的)
    float sum = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // 树形归约:每一轮活下来的线程数减半
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

void reduce_sum_launch(LaunchCtx& ctx) {
    reduce_sum<<<1, BLOCK>>>(ctx.input, ctx.output, ctx.n);
}

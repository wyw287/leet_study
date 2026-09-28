// 题目 09 的参考解
//
// 一句话:**把一整行缓存到片上,输入就只需要从 DRAM 读一遍。**
//
// 一个 block 负责一行,分三段,段与段之间用 __syncthreads 隔开:
//
//   ① 合并地把这一行搬进共享内存,同时用寄存器求出本线程的局部最大值
//      → 块内归约得到整行的 m
//   ② 遍历共享内存:e = expf(v - m),把 e **写回共享内存**,同时累加局部的和
//      → 块内归约得到 sum
//   ③ 再遍历共享内存:y = e * (1/sum),合并地写回 DRAM
//
// 访存账(以 square 用例 8192×4096 = 134MB 为例):
//
//   DRAM       : 读 x 一遍 + 写 y 一遍 = 8 字节/元素 = 268MB
//   共享内存   : 写 4 + 读 4 + 写 4 + 读 4 = 16 字节/元素
//
// 共享内存看着比 DRAM 多一倍,但它是**片上**的:4090 每个 SM 每周期能给 128 字节,
// 128 个 SM × 2.5GHz ≈ 41TB/s,是 DRAM 峰值(1TB/s)的 40 倍。
// 所以那 16 字节/元素折合下来只有几微秒,而省掉的 268MB DRAM 流量是 260µs。
//
// 基线是 3 读 + 1 写 = 16 字节/元素 DRAM,本题的加速比天花板就是 16/8 = 2.0x。
//
// 两个小细节:
//   * 1.0f/sum 提到循环外。f32 除法在硬件上是"倒数近似 + 牛顿迭代",十几条指令,
//     每个元素做一次纯属浪费。
//   * 第②段把 e 写回共享内存,第③段就不用再算一次 expf(SFU 是有限的)。
//
// 实测(2026-09-27,4090,框架自动清 L2;square 用例 = 134MB 读 + 134MB 写):
//
//   基线(3 个 kernel,输入读三遍)      0.569 ms   1.00x   有效带宽  942 GB/s
//   2 个 kernel(第一趟出 max 与 sum)   0.426 ms   1.33x   有效带宽  944 GB/s
//   一个 kernel 但三趟都读全局          0.246 ms   2.31x   有效带宽 1090 GB/s
//   本解(一个 kernel + 行留在片上)     0.238 ms   2.39x   有效带宽 1126 GB/s
//
// 第三行是这道题最反直觉的实测结果,也是它真正想教的东西:
// **一个 kernel 内部的重读会命中 L2**(一行只有 16KB,而 L2 有 72MB),
// 所以"把三趟合成一个 kernel"这一步就拿到了几乎全部收益(1.33x → 2.31x),
// 把整行搬进共享内存只多拿 3%。基线之所以真的贵,是因为两个 kernel 之间
// 隔着全局屏障:等第一个 kernel 跑完,134MB 的输入早把 72MB 的 L2 冲干净了,
// 第二三趟只能回 DRAM 重新读。
//
// 那为什么还要写共享内存这一版?因为"装得进 L2"不是理所当然的:
// 行更长、或者要跨 block 协作(比如 flash attention 那种分块)时,
// 片上缓存才是唯一能保证"只读一遍"的手段,而且是可移植的写法。

#include "ctx.h"

#define BLOCK 256

// ---------------------------------------------------------------------------
// 块内归约两个小工具:结果放在 red[0],所有线程都能读到。
//
// 注意开头那个 __syncthreads():它不是多余的 —— 上一次归约的 red[0] 刚被所有
// 线程读过,而这一行马上就要往 red[] 里写,读和写之间必须有屏障。
// 实测:把它删掉,racecheck 报 3 个 hazard,而正确性对拍照样通过 ——
// 这种竞态属于"没翻车但确实错了"的那一类,只有竞态检查抓得住。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_max(float v, float* red) {
    const int t = threadIdx.x;
    __syncthreads();
    red[t] = v;
    __syncthreads();
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (t < st) red[t] = fmaxf(red[t], red[t + st]);
        __syncthreads();
    }
    return red[0];
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    const int t = threadIdx.x;
    __syncthreads();
    red[t] = v;
    __syncthreads();
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    return red[0];
}

__global__ void row_softmax(const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols, float s) {
    extern __shared__ float sh[];
    float* row = sh;          // 前 cols 个 float:这一行的数据
    float* red = sh + cols;   // 后面 BLOCK 个:归约暂存区

    const int r = blockIdx.x;
    if (r >= rows) return;    // 网格就是 rows 个 block,这一句只是保险
    const int t = threadIdx.x;
    const float* xr = x + (size_t)r * (size_t)cols;
    float* yr = y + (size_t)r * (size_t)cols;

    // ① 搬进共享内存 + 局部最大值。
    //    跨步循环天然处理 cols 不整除 BLOCK、以及 cols < BLOCK 的情况:
    //    领不到元素的线程不进循环,它的局部最大值保持中性元 -1e30,
    //    归约时不会影响结果。
    float m = -1e30f;
    for (int c = t; c < cols; c += BLOCK) {
        const float v = s * xr[c];
        row[c] = v;
        m = fmaxf(m, v);
    }
    m = block_max(m, red);

    // ② 减去最大值之后才 exp —— 这是数值稳定的关键。
    //    scale=100 的用例里 s*x 能到 100,不减去 m 的话 expf(100) = inf。
    float acc = 0.f;
    for (int c = t; c < cols; c += BLOCK) {
        const float e = expf(row[c] - m);
        row[c] = e;           // 写回去,第③段就不用再算一次 expf
        acc += e;
    }
    const float sum = block_sum(acc, red);

    // ③ 归一化。除法只做一次。
    const float inv = 1.0f / sum;
    for (int c = t; c < cols; c += BLOCK) {
        yr[c] = row[c] * inv;
    }
}

void row_softmax_launch(LaunchCtx& ctx) {
    // 一行一个 block。动态共享内存 = 行数据 + 归约暂存区。
    //
    // cols 很大时(一行超过约 11K 个 float)会顶到 48KB 的默认上限;
    // 4090 允许申请到 99KB(cudaFuncSetAttribute),但再大就必须换算法:
    // 一边读一边维护 (max, sum) 的在线算法(online softmax)可以只读一遍
    // 而不缓存整行 —— 那是 flash attention 的核心,留给下一题。
    const size_t shbytes = (size_t)(ctx.cols + BLOCK) * sizeof(float);
    row_softmax<<<ctx.rows, BLOCK, shbytes>>>(ctx.x, ctx.y, ctx.rows, ctx.cols,
                                              (float)ctx.scale);
}

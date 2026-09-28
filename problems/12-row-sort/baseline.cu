// 题目 12 的性能基线:奇数-偶数换位排序(odd-even transposition sort)。
//
// 这是「并行冒泡」的教科书写法,也是一个人第一次拿到「一行数据要排序」
// 时最自然的落笔:一个 block 负责一行,整行搬进共享内存,然后
//
//     第 0 趟:比较 (0,1) (2,3) (4,5) ...  挨着的成对比较
//     第 1 趟:比较 (1,2) (3,4) (5,6) ...  错开一位
//     第 2 趟:又是 (0,1) (2,3) ...
//     ...
//     一共 cols 趟,每趟后面一次 __syncthreads()
//
// 它每一趟都是对的(n 个元素的 0-1 序列在 n 趟之内必然有序,这是标准结论),
// 每一趟也都在全速跑 —— 唯一的毛病是**它需要 cols 趟**。
//
// 这道题的全部成本几乎就是「趟数 × 一次屏障的价钱」:
//   * cols = 1024 → 1024 次 __syncthreads
//   * bitonic 排序 → log²(n)/2 ≈ 55 次
//   * 计数排序     → 3 次
// DRAM 流量三者完全一样(读 4 字节 + 写 4 字节),所以差距全部来自屏障
// 与配套的指令发射。实测(4090,big 用例)基线 **15.643 ms** —— 是这道题
// DRAM 下限(266µs)的 59 倍。**它慢的原因和显存毫无关系。**
//
// 为什么不让它更笨(比如"一个线程串行插排一行"):
//   那会慢到另一个数量级上,加速比会跨三个半数量级,S/A/B/C 四个档次
//   全挤在一起。这条基线是**结构上真并行、只是多做了 256 倍趟数**
//   (cols=1024 趟 vs 计数排序的 4 趟),正好把这道题要教的那件事
//   (趟数就是成本)单独隔离出来。
//
// 说明:本题所有用例 cols ≤ 1024,所以共享内存按 cols 动态开;
// 每趟只有前 ceil(cols/2) 个线程在干活,其余线程只负责到达屏障。

#include "ctx.h"

__global__ void row_sort(const int* __restrict__ x,
                         int* __restrict__ y,
                         int rows, int cols) {
    extern __shared__ int s[];          // cols 个 int

    const int r = blockIdx.x;           // 一个 block 一行
    if (r >= rows) return;              // 块内所有线程一起返回,不会半路分叉
    const int t = threadIdx.x;

    const int* __restrict__ xr = x + (long)r * cols;
    int* __restrict__ yr = y + (long)r * cols;

    for (int i = t; i < cols; i += blockDim.x) s[i] = xr[i];
    __syncthreads();

    for (int phase = 0; phase < cols; ++phase) {
        // 偶数趟比 (0,1)(2,3)...,奇数趟比 (1,2)(3,4)...
        const int i = 2 * t + (phase & 1);
        if (i + 1 < cols && s[i] > s[i + 1]) {
            const int tmp = s[i];
            s[i] = s[i + 1];
            s[i + 1] = tmp;
        }
        // 少这一次屏障,相邻的两次比较就会读到写了一半的数据 ——
        // 这正是「每趟一次屏障」的代价所在,也是本题的主角。
        __syncthreads();
    }

    for (int i = t; i < cols; i += blockDim.x) yr[i] = s[i];
}

void row_sort_launch(LaunchCtx& ctx) {
    const int block = 1024;
    const size_t shbytes = (size_t)ctx.cols * sizeof(int);
    // 一个 block 一行:网格大小 = 行数,不需要向上取整
    row_sort<<<ctx.rows, block, shbytes, ctx.stream>>>(ctx.x, ctx.y,
                                                       ctx.rows, ctx.cols);
}

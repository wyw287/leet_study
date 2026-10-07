// 题目 15:一次遍历算三个统计量(和 / 最小 / 最大)
//
// 只在本文件里写代码。device 内存的分配、数据搬运、计时、校验、越界检测
// 都由框架完成 —— 你只需要写 kernel 和决定启动配置。
//
//   ctx.x        const float*  device 指针,长度 ctx.n,值域 [-1, 1)
//   ctx.sum      float*        长度 1,写 Σ x[i]
//   ctx.min_val  float*        长度 1,写 min x[i]
//   ctx.max_val  float*        长度 1,写 max x[i]
//   ctx.n        int           元素个数
//   ctx.partial  float*        长度 ctx.max_blocks * 3,scratch(可不用)
//   ctx.max_blocks int         你的 grid 的上限(partial 能装下的块数)
//   ctx.stream   cudaStream_t  启动 kernel 时传给它(nullptr 就是默认流)
//
// 题面在 problem.md 里,**先读完再动手** —— 这道题的考点不在"会不会写归约",
// 而在"你打算把这块数据读几遍"。

#include "ctx.h"

// ---------------------------------------------------------------------------
// 1) 写 kernel
// ---------------------------------------------------------------------------
// 建议的结构(三选一,前两个都行,只要想清楚你读了几遍):
//
//   (a) 一个 kernel 搞定三个量 + 一个第二级 kernel 收尾(推荐)
//   (b) 一个 kernel 搞定三个量 + 全局原子操作收尾(注意 min/max 没有 float 版)
//   (c) 三个 kernel 各管一个量 —— 能过,但你会拿到最低的评级,因为
//       你把 256MiB 读了三遍(见 spec/题面的带宽刻度)
//
// 不管选哪个,第一级 kernel 的形状都是这样的:
//
//     __global__ void array_stats(const float* x, float* partial,
//                                 float* sum, float* min_val, float* max_val, int n) {
//         // 每个线程维护三个累加器,只遍历一遍数组
//         float a  = 0.f;
//         float lo =  __int_as_float(0x7f800000);   // +inf:min 的单位元,不是 0!
//         float hi = -__int_as_float(0x7f800000);   // -inf:max 的单位元
//         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
//              i += blockDim.x * gridDim.x) {
//             float v = x[i];
//             // TODO: 同时更新 a / lo / hi
//         }
//         // TODO: 块内归约,写出本块的 (sum, min, max) —— 三个量
//     }
//
// 提示:
//   * min 的单位元写 0 是**错的**(+inf 才是)。数据里有负数,0 会把最小值
//     污染成 0,而 sum 看起来完全正常。`__int_as_float(0x7f800000)` 就是 +inf。
//   * 块内归约可以用共享内存(每个量一个数组,一轮同步管三个),
//     也可以先做 warp shuffle 再汇总。用共享内存更好懂。
//   * 别忘了 __syncthreads():块内归约里"一个线程写、另一个线程读"
//     之间必须有同步,漏掉就是竞态 —— 有时对有时错。
__global__ void array_stats(const float* x, float* partial,
                            float* sum, float* min_val, float* max_val, int n) {
    // TODO
}

// ---------------------------------------------------------------------------
// 2) 第二级:块间归约
// ---------------------------------------------------------------------------
// 第一级每个块只算出自己那份 (sum, min, max),还要合起来。
// 最简单的一条路:每块把三元组写进 partial[blockIdx.x * 3 + {0,1,2}],
// 再启动**一个**单块 kernel 把这 grid 个三元组归约成三个标量。
//
// 注意:grid 是你在 launcher 里选的,第二级 kernel 需要知道它 ——
// 当成参数传进去就行。
//
// 另一条路是全局原子操作:atomicAdd 有 float 版本,但 atomicMin/atomicMax
// **只有整数版本**。想在 float 上用,要么自己写 CAS 循环,要么换个思路
// (题面「陷阱提醒」里讲了为什么 float 的位模式不能直接当整数比)。
__global__ void array_stats_final(const float* partial, float* sum,
                                  float* min_val, float* max_val, int nblocks) {
    // TODO(如果走"三个独立 kernel"那条路,这个函数可以不要)
}

// ---------------------------------------------------------------------------
// 3) 决定启动配置(这本身就是考点)
// ---------------------------------------------------------------------------
void array_stats_launch(LaunchCtx& ctx) {
    // TODO:
    //   int grid = ...;                       // 别超过 ctx.max_blocks
    //   array_stats<<<grid, 256, 0, ctx.stream>>>(...);
    //   array_stats_final<<<1, 256, 0, ctx.stream>>>(...);
    //
    // 想清楚两件事:
    //   ① 每个元素被**读**了几次?(这是本题唯一的成本)
    //   ② 块数开多少?(128 个 SM,块数太少就有 SM 在闲着)
}

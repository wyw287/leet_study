// 题目 14 的性能基线:一个线程算一个输出,老老实实把 8 个翻转位置读一遍。
//
// 这份实现**完全正确**,而且访存那一侧是"教科书式正确"的 —— 先说清楚,
// 因为它决定了这道题到底在考什么:
//
//   * **合并访问满分。** 同一个 warp 的 32 个 lane 读的是同一行里连续的 32 个
//     float(128 字节 = 4 个 sector),8 个 tap 每一个都是这样。
//     08 题里那个"一条 load 指令拆成 32 个 sector"的坑,这里一个都没有。
//   * **sector 一个字节都没浪费。** 每个 tap 都是整行连续取数,读进来的
//     128 字节全部有用。13 题里那个"L1 取数爆表"的坑,这里也没有。
//   * **算术强度不是问题。** 每个输出 8 次加法 / 8 字节访存 = 1 FLOP/Byte,
//     比 4090 的平衡点(约 82 FLOP/Byte)低两个数量级 —— 算力侧完全空闲。
//
// 显存那一侧看起来也是对的:**输出的每个字节只写一次**。毛病出在**读**:
//
//   每个输出要读 8 个位置,而这 8 个位置分成四组、落在四行上:
//
//       (b, i)  (b, mi)  (mb, i)  (mb, mi)         mi = h-1-i, mb = d-1-b
//
//   每组读两个位置:第 j 个和它的左右镜像 mj。同一组里的这两个位置在同一行内,
//   最多隔 w = 4096 个元素 = 16KB —— **这两个总是同时命中,不用管它**。
//
//   要算的是**组与组之间**。一个元素被读到 8 次,这 8 次不是同时发生的,
//   而是分散在四个时刻上:
//
//       (b,i) ↔ (b, mi) :最多  h*w = 33.5M 个元素
//       (b,i) ↔ (mb, i) :恒等于 h*w = 33.5M 个元素
//       (mb,i)↔ (mb, mi):最多  h*w = 33.5M 个元素
//
//   kernel 每处理一个输出平均要搬十来个字节,所以"隔了多少个元素"基本就等于
//   "隔了多少流量":33.5M 个元素 ≈ 400MB **以上**。
//
//   而 L2 只有 **72MB**。两次访问之间隔了几十万个元素的时候,那一行早就被
//   后来者冲出去了 —— 只能再走一趟显存。实测:四个时刻里有三个真的落空,
//   等效流量 ≈ 3.5 读 + 1 写(而参考解是 1 读 + 1 写)。
//
//   (顺带说明形状:切片 h*w = 33.5M 个元素是**故意**选的。如果把它摊成
//    2048×256×128 —— 同样是 268MB —— 切片只有 128KB,(b,i) ↔ (b,mi) 这一组
//    就落回"几万个元素"那一档、被缓存接住了,8 次读里只有 1 次落空,
//    基线能跑到 68%,梯度就没了。**张量比 L2 大是必要条件,不是充分条件。**)
//
// 一句话:**这道题不考"怎么取数",考的是"什么时候取"。**
// 缓存能不能救你,取决于两次使用之间隔了多少流量(复用距离),
// 而不是取决于访问模式好不好看。

#include "ctx.h"

__global__ void flip_avg(const float* __restrict__ x,
                         float* __restrict__ out,
                         int d, int h, int w) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // 最内层:线程沿它走
    const int i = blockIdx.y;
    const int b = blockIdx.z;
    if (j >= w) return;

    const int mb = d - 1 - b;
    const int mi = h - 1 - i;
    const int mj = w - 1 - j;

    // 四个行首的线性下标。层与层之间差 h*w,行与行之间差 w。
    const int s0 = b * h * w;    // 第 b  层
    const int s1 = mb * h * w;   // 第 mb 层
    const int r0 = i * w;        // 第 i  行
    const int r1 = mi * w;       // 第 mi 行

    const float s = x[s0 + r0 + j] + x[s0 + r0 + mj]
                  + x[s0 + r1 + j] + x[s0 + r1 + mj]
                  + x[s1 + r0 + j] + x[s1 + r0 + mj]
                  + x[s1 + r1 + j] + x[s1 + r1 + mj];

    out[s0 + r0 + j] = 0.125f * s;
}

void flip_avg_launch(LaunchCtx& ctx) {
    // 三维网格:一个 block 管一行的 128 个元素,x 沿 w(连续的那一维),
    // y 沿 h,z 沿 d。w=4096 与 blockDim.x 对齐,行首都在 128 字节边界上。
    const dim3 block(128, 1, 1);
    const dim3 grid((ctx.w + block.x - 1) / block.x, ctx.h, ctx.d);
    flip_avg<<<grid, block>>>(ctx.x, ctx.out, ctx.d, ctx.h, ctx.w);
}

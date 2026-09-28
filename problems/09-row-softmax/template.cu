// 题目 09:行 Softmax(把三趟压成一趟)
//
// 只在本文件里写代码。device 内存的分配、数据搬运、计时、校验、越界检测
// 都由框架完成 —— 你只需要写 kernel 和决定启动配置。
//
// 语义:对 rows×cols 行优先矩阵的**每一行**独立地做
//
//     m       = max_c (scale * x[r][c])
//     y[r][c] = exp(scale*x[r][c] - m) / Σ_c exp(scale*x[r][c] - m)
//
//   ctx.x      const float*  device 指针,长度 rows*cols(行优先)
//   ctx.y      float*        device 指针,长度 rows*cols,需要你写入
//   ctx.rmax   float*        device 指针,长度 rows  —— scratch,想用就用
//   ctx.rsum   float*        device 指针,长度 rows  —— scratch,想用就用
//   ctx.rows / ctx.cols      矩阵尺寸(运行时才知道)
//   ctx.scale  int           缩放系数(用例里是 1 或 100)
//
// 提醒:scratch 缓冲在每次校验前会被框架填成毒值,用之前必须自己写。

#include "ctx.h"

// ---------------------------------------------------------------------------
// 1) 写 kernel
// ---------------------------------------------------------------------------
__global__ void row_softmax(const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols, float s) {
    // TODO: 一个 block 负责一行(blockIdx.x 就是行号),分三段做完:
    //
    //   ① 把这一行搬进共享内存,同时用寄存器求出本线程看到的局部最大值;
    //      然后做**块内归约**得到整行的 m。
    //   ② 遍历共享内存:e = expf(v - m),把 e **写回共享内存**,
    //      同时累加出本线程的局部和;再做一次块内归约得到 sum。
    //   ③ 再遍历共享内存:y[c] = e * (1/sum),合并地写回全局内存。
    //
    // 为什么这样能快一倍?因为基线把 x 从 DRAM 读了**三遍**(求 max 一遍、
    // 求 sum 一遍、归一化一遍),而上面这个结构只读一遍、写一遍。
    // 一行 4KB~16KB 完全装得下共享内存,片上读写几乎不要钱。
    //
    // 一句诚实的补充:决定性的那一步其实是"三趟合进一个 kernel"——
    // 去掉两个 kernel 之间的全局屏障之后,L2 自己就能把这一行留住,
    // 于是重读几乎不花 DRAM 流量。显式用共享内存是在这个基础上再省掉
    // L2 的往返;数据大到 L2 装不下时,它才是唯一可行的办法。
    //
    // 提示(共享内存):大小是运行时才知道的 cols 个 float,所以要用
    //
    //     extern __shared__ float sh[];     // 前 cols 个放行数据
    //     float* row = sh;
    //     float* red = sh + cols;           // 后面 BLOCK 个当归约暂存区
    //
    // 启动时把字节数当第三个参数传进去:
    //
    //     row_softmax<<<grid, block, (cols + BLOCK) * sizeof(float)>>>(...);
    //
    // 归约可以照抄 04 题的写法(共享内存树形归约);也可以先在本线程累加,
    // 再用 warp shuffle 在一轮里解决一个 warp,最后只归约每个 warp 的值。
    //
    // 踩坑提示:
    //   * 两个归约之间、归约与使用之间,`__syncthreads()` 一个都不能漏,
    //     而且不能放在分支里(会死锁或读到写了一半的数据)。
    //   * **必须先减去最大值**:scale=100 的用例里 exp 的参数能到 100,
    //     expf(100) = inf,不减去最大值就是 inf/inf = NaN。
    //   * 1.0f/sum 提到循环外面只算一次;f32 除法是十几条指令。
    //   * cols 不整除 blockDim(4093 = 15×256 + 253)、cols 比 blockDim 还小(129)
    //     都要照顾到 —— 跨步循环 `for (c = t; c < cols; c += BLOCK)` 天然能处理,
    //     但共享内存里没被写到的位置是垃圾值,别去读。
}

// ---------------------------------------------------------------------------
// 2) 决定启动配置(这本身就是考点)
// ---------------------------------------------------------------------------
void row_softmax_launch(LaunchCtx& ctx) {
    // TODO: 选块大小与网格大小,算出动态共享内存的字节数,启动 kernel
    //
    // 提示:
    //   int block = 256;
    //   dim3 grid(ctx.rows);                       // 一行一个 block
    //   size_t shbytes = (size_t)(ctx.cols + block) * sizeof(float);
    //   row_softmax<<<grid, block, shbytes>>>(ctx.x, ctx.y, ctx.rows, ctx.cols,
    //                                         (float)ctx.scale);
    //
    // 注意 shbytes 只在 cols 不太大时成立:一行 4096 个 float 是 16KB,
    // 加上归约暂存区约 17KB,在 48KB 的默认上限内。cols 上万就要另想办法。
}

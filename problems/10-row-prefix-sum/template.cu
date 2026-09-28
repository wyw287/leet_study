// 题目 10:行内前缀和(并行 scan 入门)
//
// 只在本文件里写代码。device 内存的分配、数据搬运、计时、校验、越界检测
// 都由框架完成 —— 你只需要写 kernel 和决定启动配置。
//
// 语义:对 rows×cols 行优先矩阵的**每一行**独立地做 inclusive 前缀和
//
//     y[r][0] = x[r][0]
//     y[r][c] = y[r][c-1] + x[r][c]          (c > 0)
//
//   ctx.x     const float*  device 指针,长度 rows*cols(行优先)
//   ctx.y     float*        device 指针,长度 rows*cols,需要你写入
//   ctx.rows / ctx.cols     矩阵尺寸(运行时才知道)
//
// 这道题和你做过的 01/05/09 有一个本质区别:
//   那些题的每个输出元素都能**独立**算出来,想并行就把它们摊给线程。
//   前缀和不行 —— y[r][c] 依赖 y[r][c-1],这是一条长度为 cols 的**依赖链**。
//   怎么把一条链拆成很多份并行做,就是这道题要学的东西。
//
// 基线是"一个线程扫一行":并行度只有 rows 个线程,而且同一 warp 的 32 个线程
// 在 32 个不同的行上(地址差 4KB),访存完全不合并。你要同时修掉这两条。

#include "ctx.h"

#define BLOCK 256     // 线程块大小。可以改,想想改成多少更合适。

__global__ void row_scan(const float* __restrict__ x,
                         float* __restrict__ y,
                         int rows, int cols) {
    extern __shared__ float sh[];
    float* row = sh;          // 前 cols 个 float:这一行的数据
    float* red = sh + cols;   // 后面 BLOCK 个 float:块内 scan 的暂存区

    const int r = blockIdx.x;              // 一个 block 负责一行
    if (r >= rows) return;                 // 网格就是 rows 个 block,这句只是保险
    const int t = threadIdx.x;
    const float* xr = x + (size_t)r * (size_t)cols;
    float*       yr = y + (size_t)r * (size_t)cols;

    // ① 合并地把这一行搬进共享内存。
    //    注意 `c = t; c += BLOCK` 这个**跨步**写法:同一个 warp 的 32 个线程
    //    地址连续,一次 128 字节的访问全用上。这一步是基线最缺的东西。
    for (int c = t; c < cols; c += BLOCK) row[c] = xr[c];
    __syncthreads();

    // ② TODO: 在共享内存里对这一行做前缀和。两条路任选(也可以都想一遍):
    //
    //   路 A —— Hillis-Steele:步长 d 从 1 翻倍到 cols,每一轮
    //           row[c] += row[c-d]  (c >= d 时)。log2(cols) 轮就结束,
    //           总工作量 O(cols·log cols) —— 比串行多做了 log 倍。
    //           小心:**不能就这么原地写**。同一轮里,线程 A 可能已经改掉
    //           row[c-d],而线程 B 还要读它。要么"先算到寄存器 → __syncthreads()
    //           → 再写回",要么开两块共享内存乒乓(读 A 写 B,下一轮反过来)。
    //
    //   路 B —— 两段式(总功 O(cols)):把这一行切成 BLOCK 段(第 t 段是
    //           row[t*K .. t*K+K),K = ceil(cols/BLOCK)),每个线程把自己那段
    //           **串行**扫掉(段内累加),把每段的总和记下来;然后对这 BLOCK 个
    //           总和做一次块内 scan 得到"段间偏移",最后每个线程把偏移加回
    //           自己那一段。串行部分只在片上进行,总功和串行算法一样是 O(cols)。
    //
    //   两条路都可以拿满分。想想为什么:瓶颈是 DRAM(每个元素 8 字节),
    //   而共享内存带宽是 DRAM 的 40 倍 —— log 倍的多余工作落在片上,淹没了。
    //
    // ③ TODO: 把扫完的结果**合并地**写回 yr[](还是 `c = t; c += BLOCK` 那个写法)。

    (void)red;   // 用不到就删掉这一行(路 B 会用到它存"每段的总和")
}

void row_scan_launch(LaunchCtx& ctx) {
    // 一行一个 block。动态共享内存 = 这一行的数据 + 块内 scan 的暂存区。
    //
    // 注意 cols 是运行时才知道的 —— 一行 1280 个 float 是 5KB,加上暂存区
    // 约 6KB,离 48KB 的默认上限还很远。但如果一行长到上万,就得改算法
    // (分块扫,或者干脆一个 block 处理多行)。
    const size_t shbytes = (size_t)(ctx.cols + BLOCK) * sizeof(float);
    row_scan<<<ctx.rows, BLOCK, shbytes>>>(ctx.x, ctx.y, ctx.rows, ctx.cols);
}

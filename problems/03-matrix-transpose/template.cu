// 题目 03:矩阵转置
//
// input 是 m 行 n 列的矩阵(row-major,即 input[r * n + c] 是第 r 行第 c 列)
// output 是它的转置,共 n 行 m 列(output[c * m + r] 是第 c 行第 r 列)
//
//   ctx.input   const float*  device 指针,长度 m*n
//   ctx.output  float*        device 指针,长度 n*m,需要你写入
//   ctx.m       行数(input 的行数)
//   ctx.n       列数(input 的列数)
//
// 注意:行与列在内存里的步长不同 —— input 的行距是 n,output 的行距是 m。

#include "ctx.h"

// ---------------------------------------------------------------------------
// 1) 写 kernel
// ---------------------------------------------------------------------------
__global__ void transpose(const float* input, float* output, int m, int n) {
    // TODO: 二维索引 + 边界判断,然后 out[c][r] = in[r][c]
    //
    // 朴素写法(也是本题基线的做法):
    //   int c = blockIdx.x * blockDim.x + threadIdx.x;   // 列
    //   int r = blockIdx.y * blockDim.y + threadIdx.y;   // 行
    //   if (r < m && c < n)
    //       output[(long long)c * m + r] = input[(long long)r * n + c];
    //
    // 这样写能跑对,但**慢**。原因:同一 warp 里相邻线程的 c 相邻,
    // 它们要写的位置相隔 m 个元素 —— 每次 warp 写都散成 32 个显存事务。
    //
    // 想拿高分要做分块 + 共享内存:
    //   1. 每个线程块负责 TILE×TILE 的一块(如 32×32)
    //   2. 合并读入共享内存:smem[ty][tx] = input[(row0+ty) * n + (col0+tx)]
    //      —— 读是合并的,因为相邻 tx 读相邻地址
    //   3. __syncthreads()
    //   4. 转置写出:output[(col0+ty) * m + (row0+tx)] = smem[tx][ty]
    //      —— 注意这里下标反过来了,让写也变成合并的
    //
    // 踩坑提示:
    //   * 别忘了 __syncthreads(),漏了会随机出错(racecheck 会抓到)
    //   * 共享内存数组建议声明成 [TILE][TILE+1],多出来的一列用来避开
    //     bank conflict —— 这一步能再快一截,值得亲手对比
    //   * 非方阵(m≠n)且不能被 TILE 整除时,共享内存的读和写都要判边界
}

// ---------------------------------------------------------------------------
// 2) 决定启动配置
// ---------------------------------------------------------------------------
void transpose_launch(LaunchCtx& ctx) {
    // TODO: 选块大小与网格大小
    //
    // 提示(朴素版):
    //   dim3 block(32, 8);
    //   dim3 grid((ctx.n + block.x - 1) / block.x,
    //             (ctx.m + block.y - 1) / block.y);
    //   transpose<<<grid, block>>>(ctx.input, ctx.output, ctx.m, ctx.n);
}

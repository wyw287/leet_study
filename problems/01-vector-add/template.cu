// 题目 01:向量加法
//
// 只在本文件里写代码。
// device 内存的分配、数据搬运、计时、校验、越界检测都由框架完成 ——
// 你只需要写 kernel 和决定启动配置。
//
//   ctx.a  const float*  device 指针,长度 ctx.n
//   ctx.b  const float*  device 指针,长度 ctx.n
//   ctx.c  float*        device 指针,长度 ctx.n,需要你写入
//   ctx.n  int           元素个数

#include "ctx.h"

// ---------------------------------------------------------------------------
// 1) 写 kernel
// ---------------------------------------------------------------------------
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    // TODO: 算出本线程负责的下标,判断边界后写入 c
    //
    // 提示:
    //   int i = blockIdx.x * blockDim.x + threadIdx.x;
    //   if (i < n) c[i] = a[i] + b[i];
    //
    // 注意:边界判断不能省。n 不整除线程总数时,越界的线程会写到别的
    // 缓冲上 —— 框架的哨兵区会抓到它。
}

// ---------------------------------------------------------------------------
// 2) 决定启动配置(这本身就是考点)
// ---------------------------------------------------------------------------
void vecadd_launch(LaunchCtx& ctx) {
    // TODO: 选线程块大小、算网格大小,然后启动 kernel
    //
    // 提示:
    //   int block = 256;
    //   int grid  = (ctx.n + block - 1) / block;
    //   vecadd<<<grid, block>>>(ctx.a, ctx.b, ctx.c, ctx.n);
}

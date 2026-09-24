// 题目 02:SAXPY
//
// SAXPY = "Scalar Alpha X Plus Y",是 BLAS 里最基础的一级运算:
//
//     out[i] = alpha * x[i] + y[i]        i = 0 .. n-1
//
//   ctx.x      const float*  device 指针,长度 n
//   ctx.y      const float*  device 指针,长度 n
//   ctx.out    float*        device 指针,长度 n,需要你写入
//   ctx.alpha  float         标量系数
//   ctx.n      int           元素个数
//
// 和向量加法的区别只有两点:多了一个标量参数,以及做了两次乘法/加法而不是一次。
// 这两点都**不影响**性能结论 —— 它依然是纯访存瓶颈。

#include "ctx.h"

// ---------------------------------------------------------------------------
// 1) 写 kernel
// ---------------------------------------------------------------------------
__global__ void saxpy(const float* x, const float* y, float* out,
                      float alpha, int n) {
    // TODO: 算下标 → 判边界 → 写入结果
    //
    // 提示:
    //   int i = blockIdx.x * blockDim.x + threadIdx.x;
    //   if (i < n) out[i] = alpha * x[i] + y[i];
    //
    // 注意 alpha 是从 host 传进来的**值**(不是指针):
    // CUDA 的 kernel 参数走一块专门的常量内存,传标量几乎零成本。
    // 不要为了"性能"把 alpha 塞进 device 内存再用指针访问 —— 那反而更慢。
}

// ---------------------------------------------------------------------------
// 2) 决定启动配置
// ---------------------------------------------------------------------------
void saxpy_launch(LaunchCtx& ctx) {
    // TODO: 选块大小与网格大小,然后把 ctx.alpha 一起传进去
    //
    // 提示:
    //   int block = 256;
    //   int grid  = (ctx.n + block - 1) / block;
    //   saxpy<<<grid, block>>>(ctx.x, ctx.y, ctx.out, ctx.alpha, ctx.n);
}

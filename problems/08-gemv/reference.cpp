// 题目 08 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 最直白的双重循环,但累加放在 double 里:y[r] 是 n 个 O(0.25) 量级的乘积之和,
// 4096 项 f32 顺序累加自身的舍入误差就能到 1e-4 量级,那样「对拍不过」就分不清
// 是选手写错了还是参考解自己不准。参考解不需要快,只需要**绝对可信**。
//
// 边界情况说明:GEMV 没有真正的边界情况 —— m、n 可以任意(1、质数、奇数都行),
// 参考解对每个 r 独立地从头到尾扫一遍,不依赖任何整除关系。求和顺序的差异
// (GPU 上是树形归约)由 verify 的容差兜住。

#include "ctx.h"

void reference(RefCtx& ctx) {
    const int m = ctx.m;
    const int n = ctx.n;

    for (int r = 0; r < m; ++r) {
        const float* arow = ctx.a + (size_t)r * (size_t)n;
        double acc = 0.0;
        for (int c = 0; c < n; ++c) {
            acc += (double)arow[c] * (double)ctx.x[c];
        }
        ctx.y[r] = (float)acc;
    }
}

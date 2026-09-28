// 题目 10 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 逐行从头累加到尾,一个标量循环。语义直白到没有出错的空间:
// inclusive 前缀和就是"维护一个累加器,每读一个数就先加再写"。
//
// 为什么累加器用 double:
//   本题的输入全是正数(x ∈ (0,1]),每行的和约等于 cols/2 ≈ 640。
//   参考解不需要快,只需要**绝对可信** —— 一旦它自己带 1e-3 量级的误差,
//   "对拍不过"就分不清是选手错了还是它自己不准。
//   用 double 之后,这点误差(约 1e-13)相对容差完全可以忽略。
//
// 边界情况说明:每一行都从头扫到尾,不依赖任何整除关系 ——
// cols 是质数(511)、不是 4 的倍数、比 blockDim 还小,对参考解都没有区别。

#include "ctx.h"

void reference(RefCtx& ctx) {
    const int rows = ctx.rows;
    const int cols = ctx.cols;

    for (int r = 0; r < rows; ++r) {
        const float* xr = ctx.x + (size_t)r * (size_t)cols;
        float* yr = ctx.y + (size_t)r * (size_t)cols;

        double acc = 0.0;
        for (int c = 0; c < cols; ++c) {
            acc += (double)xr[c];
            yr[c] = (float)acc;
        }
    }
}

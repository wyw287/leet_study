// 题目 06 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 三重循环,但顺序是 i-k-j 而不是教科书上的 i-j-k:
//   * 内层对 j 连续 —— 写 out 是顺序的,编译器也有机会向量化
//   * 累加在 double 里做 —— f32 累加 2048 项,自身的舍入误差就能到 1e-4 量级,
//     那样「对拍不过」就分不清是选手写错了还是参考解自己不准
//
// 参考解不需要快,只需要**绝对可信**。边界情况说明:矩阵乘法没有边界情况,
// 也不涉及求和顺序的语义分歧 —— 只有浮点结合顺序的差异,由 verify 的容差兜住。

#include "ctx.h"

#include <vector>

void reference(RefCtx& ctx) {
    const int n = ctx.n;
    std::vector<double> acc((size_t)n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) acc[j] = 0.0;

        const float* arow = ctx.a + (size_t)i * (size_t)n;
        for (int k = 0; k < n; ++k) {
            const double aik = (double)arow[k];
            const float* brow = ctx.b + (size_t)k * (size_t)n;
            for (int j = 0; j < n; ++j) acc[j] += aik * (double)brow[j];
        }

        float* crow = ctx.out + (size_t)i * (size_t)n;
        for (int j = 0; j < n; ++j) crow[j] = (float)acc[j];
    }
}

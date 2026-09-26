// 题目 07 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 直方图的语义没有任何浮点歧义:数数就是数数,CPU 和 GPU 必须一个不差。
// 所以框架对整数输出用**精确相等**比较(见 spec 的 atol/rtol 都填 0),
// 这里也不需要 double 之类的中转。
//
// 两条容易被忽略的语义,这里明确写出来:
//   1. bins 必须先清零 —— GPU 侧同样如此(而且那边更麻烦,见 baseline.cu 的注释)。
//   2. 落在 [0, nbins) 之外的值不计数。本题的输入是 randint(0..99)、nbins=128,
//      所以这条永远不会触发;写出来是为了让"边界"这件事有个明确答案,
//      而不是"恰好没发生"。

#include "ctx.h"

void reference(RefCtx& ctx) {
    for (int i = 0; i < ctx.nbins; ++i) {
        ctx.bins[i] = 0;
    }
    for (int i = 0; i < ctx.n; ++i) {
        int v = ctx.data[i];
        if (v >= 0 && v < ctx.nbins) {
            ctx.bins[v] += 1;
        }
    }
}

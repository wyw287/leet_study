// 题目 02 的参考解(CPU,ground truth)。

#include "ctx.h"

void reference(RefCtx& ctx) {
    for (int i = 0; i < ctx.n; ++i) {
        ctx.out[i] = ctx.alpha * ctx.x[i] + ctx.y[i];
    }
}

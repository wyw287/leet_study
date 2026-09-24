// 题目 01 的参考解(CPU,ground truth)—— 不暴露给做题者。
// 这里用最直白的标量循环,保证语义清晰、不会出错。
//
// 边界情况说明:浮点加法本身是可结合的,对拍用 atol/rtol 容差即可,
// 不要求和 GPU 的归约顺序一致。

#include "ctx.h"

void reference(RefCtx& ctx) {
    for (int i = 0; i < ctx.n; ++i) {
        ctx.c[i] = ctx.a[i] + ctx.b[i];
    }
}

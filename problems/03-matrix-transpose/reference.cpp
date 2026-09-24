// 题目 03 的参考解(CPU,ground truth)。
// 直白的二重循环:out[c][r] = in[r][c],即 out[c * m + r] = in[r * n + c]。

#include "ctx.h"

void reference(RefCtx& ctx) {
    for (int r = 0; r < ctx.m; ++r) {
        for (int c = 0; c < ctx.n; ++c) {
            // input 是 m×n(row-major),output 是 n×m(row-major)
            ctx.output[(long long)c * ctx.m + r] = ctx.input[(long long)r * ctx.n + c];
        }
    }
}

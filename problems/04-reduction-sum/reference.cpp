// 题目 04 的参考解(CPU,ground truth)。
//
// 用 double 累加:输入有 400 万个值,float32 顺序累加本身就会累积可观的舍入误差。
// 参考解要尽量接近真值,容差才有意义。

#include "ctx.h"

void reference(RefCtx& ctx) {
    double acc = 0.0;
    for (int i = 0; i < ctx.n; ++i) {
        acc += (double)ctx.input[i];
    }
    ctx.output[0] = (float)acc;
}

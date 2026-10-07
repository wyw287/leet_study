// 题目 15 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 这里是 O(n) 的朴素单趟循环:一遍遍历同时更新三个量。它同时也是本题
// 想教的东西的**最直白形态** —— GPU 上要做的事,本质上就是把这段循环
// 拆给几万个线程并行执行,而不是让每个统计量各扫一遍数组。
//
// 求和的中间结果用 double:CPU 上顺序累加 6700 万个 float,如果用 float
// 累加,光是舍入误差就能到 1e-2 量级,那样"参考解"本身就不够准,
// 拿它当判分基准会让正确实现被冤枉。min/max 无舍入,用什么类型都一样。

#include "ctx.h"

void reference(RefCtx& ctx) {
    double s = 0.0;
    float lo = ctx.x[0];
    float hi = ctx.x[0];
    for (int i = 0; i < ctx.n; ++i) {
        const float v = ctx.x[i];
        s += (double)v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    ctx.sum[0]     = (float)s;
    ctx.min_val[0] = lo;
    ctx.max_val[0] = hi;
}

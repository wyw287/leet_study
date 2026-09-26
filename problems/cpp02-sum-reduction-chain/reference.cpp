// CPU 参考解(oracle)—— 判分的基准,必须完全正确。
//
// 用 double 累加:x 是 4M 个 f32,和约为 2.1e6。float 只有 24 位尾数,
// 在这个量级上相邻可表示数的间隔是 0.125 —— 串行 float 累加实测会在
// big 用例上判错(超过允许误差 21)。double 的误差是 ~1e-9。
// 参考解的职责是"接近真值",所以这里必须用 double,即使它比 float 慢。
//
// 注意:这个函数**故意**写成最朴素的串行形式 —— 它不参与性能评分,
// 只负责给出正确答案。不要照抄它的写法去交题(那正是 1.00x 的基线)。

#include "ctx.h"

void reference(Ctx& ctx) {
    const int64_t n = (int64_t)ctx.n;
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += (double)ctx.x[i];
    }
    ctx.out[0] = s;
}

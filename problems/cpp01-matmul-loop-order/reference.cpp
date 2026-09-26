// CPU 参考解(oracle)—— 判分的基准,必须完全正确。
//
// 用 double 累加:矩阵乘是 384 项求和的累加,用 float 累加会有 ~1e-5 的相对误差,
// 而参考解的职责是"接近真值",不是"和某个特定实现一致"。
// 用户实现与它之间的偏差由 spec 里的 atol/rtol 吸收。

#include "ctx.h"

void reference(Ctx& ctx) {
    const int n = ctx.n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) {
                s += (double)ctx.A[i * n + k] * (double)ctx.B[k * n + j];
            }
            ctx.C[i * n + j] = (float)s;
        }
    }
}

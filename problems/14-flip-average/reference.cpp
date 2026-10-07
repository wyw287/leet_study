// 题目 14 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 语义:8 个翻转位置的算术平均
//   out[b][i][j] = (1/8) * Σ x[b 或 d-1-b][i 或 h-1-i][j 或 w-1-j]
//
// 用 double 累加:8 个 [-1,1) 的 float 相加,真值要用 double 才不会被舍入
// 污染。oracle 只负责"对",不负责"快"。
//
// 边界:三个维度都可以是奇数。d-1-b / h-1-i / w-1-j 在奇数维度的正中间会
// **等于自己**(d=1021 时 b=510 ⇒ mb=510)。参考解不做任何特殊处理:那个位置
// 在 8 项里出现两次、两次取到的是同一个值,平均下来还是它自己。
// 这不是"顺手写对了",而是**语义的一部分** —— 做题者的快版本必须复现它,
// 包括"同一层/同一行被同一个线程写两次"这种看起来诡异的写法。

#include "ctx.h"

void reference(RefCtx& ctx) {
    const int d = ctx.d, h = ctx.h, w = ctx.w;
    const long long hw = (long long)h * w;

    for (int b = 0; b < d; ++b) {
        const int mb = d - 1 - b;
        const float* p0 = ctx.x   + (long long)b  * hw;   // 第 b  层
        const float* p1 = ctx.x   + (long long)mb * hw;   // 第 mb 层(镜像层)
        float*       o  = ctx.out + (long long)b  * hw;

        for (int i = 0; i < h; ++i) {
            const int mi = h - 1 - i;

            // 四个"行首":(b,i) (b,mi) (mb,i) (mb,mi)。
            // 翻转的三个轴各自独立,所以 8 项就是这四个行首 × {j, mj}。
            const float* r0 = p0 + (long long)i  * w;
            const float* q0 = p0 + (long long)mi * w;
            const float* r1 = p1 + (long long)i  * w;
            const float* q1 = p1 + (long long)mi * w;
            float*       orow = o + (long long)i * w;

            for (int j = 0; j < w; ++j) {
                const int mj = w - 1 - j;
                double s = (double)r0[j] + (double)r0[mj]
                         + (double)q0[j] + (double)q0[mj]
                         + (double)r1[j] + (double)r1[mj]
                         + (double)q1[j] + (double)q1[mj];
                orow[j] = (float)(s * 0.125);   // 0.125 是 2 的幂,double 下精确
            }
        }
    }
}

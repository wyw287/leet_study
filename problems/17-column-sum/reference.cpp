// 题目 17 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 按**行**遍历、用 cols 个累加器(而不是"一列一个循环"):两者结果逐位相同
// (见下面那段),但行优先的走法对 CPU 缓存友好 —— 67M 次跨 4KB 步长的访存
// 会让这份参考解自己变成瓶颈,而它只是判分基准,不该拖慢整个流程。
//
// 为什么这题**不需要容差**:
//   x 的每个元素都是 0..99 的整数(randint 填充),最大可能的和是
//   rows_max * 99 = 131072 * 99 = 12,976,128,远小于 f32 的整数精度边界
//   2^24 = 16,777,216。
//   于是每一步部分和都是精确可表示的整数,浮点加法在这里**不产生舍入** ——
//   换任何累加顺序(顺序、树形、atomicAdd 的乱序)结果都逐位相同。
//   中间用 double 只是习惯,不是必需。
//
//   反过来说:如果值域改成 [-1,1) 的浮点,这条性质立刻消失(有舍入、有对消),
//   这道题就必须给容差,而且容差还得给得足够宽 —— 见 15 题里那段关于
//   "容差为什么必要"的讨论。

#include "ctx.h"

#include <vector>

void reference(RefCtx& ctx) {
    std::vector<double> acc(ctx.cols, 0.0);
    for (int i = 0; i < ctx.rows; ++i) {
        const float* row = ctx.x + (long)i * ctx.cols;
        for (int j = 0; j < ctx.cols; ++j) {
            acc[j] += (double)row[j];
        }
    }
    for (int j = 0; j < ctx.cols; ++j) {
        ctx.out[j] = (float)acc[j];
    }
}

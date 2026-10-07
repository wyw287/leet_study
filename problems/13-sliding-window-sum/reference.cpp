// 题目 13 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 语义:out[i] = x[max(0, i-w+1)] + ... + x[i-1] + x[i]
//                (窗口长度 = min(i+1, w),数组开头那 w-1 个是被裁过的)
//
// 这里用 O(n) 的滚动和,而不是"每个窗口重新加一遍"(O(n*w))。
// 两者在整数上**逐位相同**(整数加法可结合、没有舍入),但 n=16.7M、w=512 时
// 朴素写法要在 CPU 上跑 85 亿次加法 —— oracle 每跑一次就要几秒,太贵。
// oracle 只负责"对",不负责"和 GPU 用同一个算法"。
//
// 用 long long 累加:窗口最长 w=512 个 0..99 的数,和最大 50688,int 放得下;
// 但这里不省这点事 —— oracle 的正确性不该依赖"参数恰好不大"。
//
// 边界:前 w-1 个输出没有东西可减(i >= w 才减 x[i-w]),所以循环里那个
// 条件判断就是"窗口被裁"的全部逻辑。

#include "ctx.h"

void reference(RefCtx& ctx) {
    long long acc = 0;
    for (int i = 0; i < ctx.n; ++i) {
        acc += ctx.x[i];
        if (i >= ctx.w) {
            acc -= ctx.x[i - ctx.w];
        }
        ctx.out[i] = (int)acc;
    }
}

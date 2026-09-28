// 题目 12 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 这题的真值特别好写:输入是 0..99 的整数,排序的结果就是
// 「把每个值数一遍,然后按值从小到大原样铺开」。没有比较、没有交换、
// 没有任何出错的空间 —— 这段代码的正确性一眼可见。
//
// 为什么不用 std::sort:
//   ① 它慢。33.5M 个元素、每行 1024 个,std::sort 每行要做约 1 万次比较,
//      整道题要 3 亿多次 —— 参考解会在每次对拍时白白花掉一两秒。
//   ② 它会引入一个**不必要的假设**:std::sort 不关心值域,而这里值域是
//      题目契约(x 由 randint 填充,恒在 0..99)。用契约换 O(n) 是划算的。
//   ③ 数一遍再铺开,和「排好序」是同一件事 —— 这就是真值的定义本身。
//
// 边界情况说明:这段代码不依赖 rows / cols 的任何整除关系,
// 也不依赖 blockDim —— cols 是质数、比 block 小、比 block 大,对参考解都没有区别。

#include "ctx.h"

#define NBINS 100

void reference(RefCtx& ctx) {
    int hist[NBINS];

    for (int r = 0; r < ctx.rows; ++r) {
        const int* xr = ctx.x + (long)r * ctx.cols;
        int* yr = ctx.y + (long)r * ctx.cols;

        for (int v = 0; v < NBINS; ++v) hist[v] = 0;
        for (int c = 0; c < ctx.cols; ++c) ++hist[xr[c]];

        int k = 0;
        for (int v = 0; v < NBINS; ++v) {
            for (int n = 0; n < hist[v]; ++n) yr[k++] = v;
        }
    }
}

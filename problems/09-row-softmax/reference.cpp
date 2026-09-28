// 题目 09 的参考解(CPU,ground truth)—— 不暴露给做题者。
//
// 逐行三趟:求该行最大值 → 求 exp 之和 → 归一化。语义直白到没有出错的空间。
//
// 为什么中间量全用 double:
//   softmax 的输出在 [0,1],看起来 f32 也够;但参考解不需要快,只需要**绝对可信**。
//   一旦参考解自己带了 1e-6 量级的误差,"对拍不过"就分不清是选手错了还是它自己不准。
//
// 关于 scale:语义是"先缩放、再 softmax"。scale=100 时 exp 的参数最大到 100 ——
// 这正是本题要暴露的东西:不减去最大值,expf(100) 就是 inf。
// 参考解在 double 里先缩放、再减最大值,不存在溢出问题。
//
// 边界情况说明:每一行都从头扫到尾,不依赖任何整除关系 ——
// cols 是质数、比 blockDim 小、不整除 4,对参考解都没有区别。

#include "ctx.h"

#include <cmath>
#include <vector>

void reference(RefCtx& ctx) {
    const int rows = ctx.rows;
    const int cols = ctx.cols;
    const double s = (double)ctx.scale;

    std::vector<double> e((size_t)cols);

    for (int r = 0; r < rows; ++r) {
        const float* xr = ctx.x + (size_t)r * (size_t)cols;
        float* yr = ctx.y + (size_t)r * (size_t)cols;

        // ① 缩放 + 求最大值
        double mx = -1e308;
        for (int c = 0; c < cols; ++c) {
            e[c] = s * (double)xr[c];
            if (e[c] > mx) mx = e[c];
        }

        // ② exp(v - mx) 与求和。先减去 mx,参数必 <= 0,不可能上溢。
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            e[c] = std::exp(e[c] - mx);
            sum += e[c];
        }

        // ③ 归一化。1/sum 只算一次。
        const double inv = 1.0 / sum;
        for (int c = 0; c < cols; ++c) {
            yr[c] = (float)(e[c] * inv);
        }
    }
}

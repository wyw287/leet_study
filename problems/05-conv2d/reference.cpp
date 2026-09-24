// 题目 05 的参考解(CPU,ground truth)。
//
// 语义:valid 卷积,不补零。out[oy][ox] = Σ_{ky,kx} image[oy+ky][ox+kx] * kernel[ky][kx],
// 其中 0 <= oy < oh、0 <= ox < ow,且 oh = h-kh+1、ow = w-kw+1,
// 所以每个 out 像素的 kh*kw 个采样点都必然落在 image 内部(不需要任何边界判断)。
//
// 累加用 double:这是判分的基准,必须比任何 GPU 实现都更接近真值。
// 若这里也用 float,参考解自身就带了 1e-5 量级的噪声,容差就失去意义了。

#include "ctx.h"

void reference(RefCtx& ctx) {
    for (int oy = 0; oy < ctx.oh; ++oy) {
        for (int ox = 0; ox < ctx.ow; ++ox) {
            double acc = 0.0;
            for (int ky = 0; ky < ctx.kh; ++ky) {
                for (int kx = 0; kx < ctx.kw; ++kx) {
                    double v = (double)ctx.image[(long long)(oy + ky) * ctx.w + (ox + kx)];
                    double k = (double)ctx.kernel[ky * ctx.kw + kx];
                    acc += v * k;
                }
            }
            ctx.out[(long long)oy * ctx.ow + ox] = (float)acc;
        }
    }
}

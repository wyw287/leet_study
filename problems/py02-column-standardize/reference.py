"""题目 py02 的参考解(CPU,ground truth)。

与 py01 同理:参考解必须自身不可能有 GPU 侧的错误,所以跑在 CPU 上、用 float64。

语义(逐列,沿 dim=0):
    mu_c  = (1/rows) * Σ_r x[r, c]                      有偏均值
    var_c = (1/rows) * Σ_r (x[r, c] - mu_c)^2           有偏方差(= correction=0)
    out[r, c] = (x[r, c] - mu_c) / sqrt(var_c + eps)

注意方差是**有偏**定义(除以 rows,不是 rows-1)—— 这是题目规定的一部分,
不是笔误。参考解在这里写死 correction=0,学习者用别的约定会在对拍里被抓住。
"""
import torch


def reference(ctx):
    x = ctx.x.to(torch.float64)                          # (rows, cols)

    mu = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, correction=0, keepdim=True)
    out = (x - mu) / torch.sqrt(var + float(ctx.eps))

    ctx.out.copy_(out.to(ctx.x.dtype))

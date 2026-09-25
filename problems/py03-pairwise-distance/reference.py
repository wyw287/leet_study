"""题目 py03 的参考解(CPU,ground truth)—— 不暴露给做题者。

语义:把 (rows, d) 的批次切成两半(前 n 行是 A、后 m 行是 B),求

    out[i, j] = Σ_d (A[i, d] - B[j, d])^2          平方欧氏距离(不开方)

参考解必须自身不可能有 GPU 侧的错误,所以跑在 CPU 上、用 float64。

但这里**不能**照定义把 (n, m, d) 的差张量物化出来:n=m=4096、d=32 时它是
4096*4096*32*8B = 4.3GB,CPU 内存同样放不下 —— 这正是这道题要教的教训,
参考解自己也躲不过。所以用数学上等价的展开式:

    Σ_d (a_d - b_d)^2  =  Σ_d a_d^2  -  2·Σ_d a_d·b_d  +  Σ_d b_d^2

三项分别是 (n,1)、(n,m)、(1,m),中间没有 d 这一维。float64 下展开式与逐元素
定义的差在 1e-13 量级(灾难性抵消被 52 位尾数压住了),比题目容差 atol=1e-3
小 10 个数量级 —— 作为 oracle 足够精确。
"""
import torch


def reference(ctx):
    x = ctx.x.to(torch.float64)                       # (rows, d)
    a = x[:ctx.n]                                     # (n, d)
    b = x[ctx.n:ctx.n + ctx.m]                        # (m, d)

    a2 = (a * a).sum(dim=1, keepdim=True)             # (n, 1)
    b2 = (b * b).sum(dim=1, keepdim=True)             # (m, 1)
    out = a2 - 2.0 * (a @ b.t()) + b2.t()             # (n, m)

    ctx.out.copy_(out.to(ctx.x.dtype))

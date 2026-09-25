"""题目 py02 的性能基线:直接按公式的四个量写出来,一个算子一个算子地算。

这份基线**不是**"故意写烂"的 —— 它每一步都用了正确的 API(mean / var / 减法 /
开方 / 除法),数值上也完全正确。它慢的原因是 **eager 模式下每个算子都是一次
独立的 GPU 内核,每次都要把整块数据从头读一遍、再把结果写回去**。

沿着 dim=0 数一遍访存(单位 = 一个 268MB 的张量):

    x.mean(dim=0)              读 1               -> 1
    x.var(dim=0)               读 1               -> 1
    ctx.x - mu                 读 1 + 写 1        -> 2
    var + eps / sqrt            (cols 个元素,忽略)
    (x - mu) / sqrt(var+eps)   读 1 + 写 1        -> 2
                              --------------------------
                              合计 4 读 + 2 写 = 6 遍

而这道题的"理论下限"是 2 遍(读一遍 x、写一遍 out)——
中间的 4 遍全是中间张量的物化开销。

所以这道题的解法不是"把某个算子写得更快",而是
**换掉算子的组合方式,把遍数压下去**。
"""
import torch


def forward(ctx):
    mu = ctx.x.mean(dim=0, keepdim=True)
    var = ctx.x.var(dim=0, correction=0, keepdim=True)
    return (ctx.x - mu) / torch.sqrt(var + ctx.eps)

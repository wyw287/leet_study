"""题目 py02 的参考解 —— 把访存遍数从 6 压到 3

实测(8192×8192,268MB):
    基线                                  1.715 ms
    本实现(var_mean + rsqrt + addcmul)   0.849 ms   → 2.02x
    tall 用例                             1.760 ms → 0.880 ms  → 2.00x

核心思路:**把遍数压下去,而不是把某个算子写快。**

数一数基线的账:

    mu  = x.mean(dim=0)         读 x                     ①
    var = x.var(dim=0)          读 x                     ②
    (x - mu)                    读 x + 写 temp1         ③④
    / sqrt(var + eps)           读 temp1 + 写 out        ⑤⑥

6 遍 × 268MB = 1.61GB。而理论下限是 3 遍(读 x 求统计量、再读 x 写出结果)——
也就是说这道题的加速比天花板约 2.0x,我们拿到了 2.02x(略超是因为部分数据
落在 L2 里)。

一个反直觉的点:**这里的归约并不是瓶颈**。沿 dim=0 归约虽然跨行,
但 TensorIterator 会沿**连续的内层维**向量化,所以它跑得并不慢
(实测把归约改成"转置后再归约"完全没有收益,仍是 1.00x)。
真正的浪费在逐元素那几步的中间张量上。
"""

import torch


def forward(ctx):
    # 第一步:两个归约合并成一趟。
    # var_mean 用 Welford 在线算法,一趟同时算出均值与方差(6 遍 → 5 遍)。
    # 注意返回顺序是 (var, mean) 不是 (mean, var) —— 常见的翻车点。
    var, mu = torch.var_mean(ctx.x, dim=0, correction=0, keepdim=True)

    # 第二步:把逐元素链收敛成一趟。
    #
    #   (x - mu) / sqrt(var + eps)   是一个**仿射变换**:
    #
    #       out = x * s + b       其中 s = 1/sqrt(var+eps),b = -mu * s
    #
    #   s 和 b 各自只有 cols 个元素,算它们几乎不花钱;贵的是遍历整块 (rows, cols)
    #   的那一次。而 addcmul 恰好是「一趟读三个输入、写一个输出」的算子:
    #
    #       addcmul(input, t1, t2) = input + t1 * t2
    #
    #   令 input = -mu*s、t1 = x、t2 = s,一次就得到 out = x*s - mu*s。
    #
    # 用 rsqrt 而不是 1/sqrt:前者是一趟的倒平方根,后者要算开方再做除法。
    s = torch.rsqrt(var + ctx.eps)
    return torch.addcmul(-mu * s, ctx.x, s)


# ---------------------------------------------------------------------------
# 为什么不能更快了
#
# 纯 PyTorch 的地板就是 3 遍:统计量必须先把 x 读完才能算出来,
# 而结果又必须再遍历一遍 x 才能写出。想突破只有一条路 ——
# 把整个标准化融进一个自定义 kernel(online 统计 + 单趟写出),
# 那需要写 CUDA。这是另一道题。
#
# 【试过的死路,供参考】
#
#   * 转置后再归约:x.t().mean(dim=1) —— 实测 1.00x,毫无收益。
#     归约本来就不是瓶颈,改了也白改。
#
#   * 写进预分配的 ctx.out 再原地除:
#         torch.sub(ctx.x, mu, out=ctx.out); ctx.out.mul_(rsqrt(var+eps))
#     实测 1.20x —— 只做到 var_mean 那一步的收益,因为 sub 和 mul_ 仍是两趟
#     读+写 out。addcmul 把这两趟合成了一趟,这才是差距所在。
# ---------------------------------------------------------------------------

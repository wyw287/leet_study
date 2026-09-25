"""题目 py01 的参考解(CPU,ground truth)。

与 CUDA 侧同理:参考解必须自身不可能有 GPU 侧的错误,所以跑在 CPU 上。
用 float64 计算以保证接近真值 —— 输入有上亿个元素,float32 顺序累加本身
就会累积可观的舍入误差,容差才有意义。
"""
import torch


def reference(ctx):
    x = ctx.x                       # (rows, cols) CPU float32

    # 数值稳定的 softmax:先减去每列的最大值再取 exp,避免上溢
    xd = x.to(torch.float64)
    m = xd.max(dim=0, keepdim=True).values
    e = (xd - m).exp()
    ctx.out.copy_((e / e.sum(dim=0, keepdim=True)).to(x.dtype))

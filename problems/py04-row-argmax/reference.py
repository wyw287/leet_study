"""题目 py04 的参考解(CPU,ground truth)—— 不暴露给做题者。

语义:每行取最大值的列下标。

    out[i] = argmax_j x[i, j]

并列(同一行有多个相同的最大值)时取**下标最小者**。

参考解必须自身不可能有 GPU 侧的错误,所以跑在 CPU 上。这里用 float64 是为了
让「最大值」与输入逐位可比(其实 f32 也一样:最大值本身就是输入里的某个元素,
不是算出来的)。整道题不涉及任何浮点累加,所以答案在数学上是精确的 ——
这也是本题目容差可以收到 atol=1e-5 的原因。

注意:本题的随机填充是均匀分布,一行里出现并列最大值的概率极低,但「并列取最小
下标」这条规则仍然要写清楚 —— 规则的模糊比精度的误差更容易让人栽跟头。
"""
import torch


def reference(ctx):
    x = ctx.x.to(torch.float64)                      # (rows, cols)
    idx = torch.argmax(x, dim=1)                     # 并列时取第一个(下标最小)
    ctx.out.copy_(idx.to(torch.int32))

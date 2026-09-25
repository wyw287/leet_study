"""题目 py01 的性能基线:直接对跨步维调用 PyTorch 自带的 softmax。

注意这个基线的性质 —— 它**不是**一份"写得差的实现",而是**用对了 API、
但撞上了坏的内存布局**:

    x 是 (rows, cols) 按行连续的张量,内存里同一列的元素相隔 cols 个 float。
    沿 dim=0 归约意味着每个线程要跨着取数据,几乎每次访问都吃不满一个
    cache line。PyTorch 的 softmax 内核本身写得很好,但它改变不了
    "归约维不连续"这件事。

所以这道题的解法不是"写个更好的 softmax",而是**重新安排内存布局,
让 PyTorch 的快路径能用上**。这是 PyTorch 使用者与 PyTorch 熟练者之间
最典型的一道分水岭。
"""
import torch


def forward(ctx):
    # 直接指定 dim=0 —— 这是最自然的写法,也是性能陷阱所在:
    # 对按行连续的张量来说,沿 dim=0 归约意味着跨步访存。
    return torch.softmax(ctx.x, dim=0)

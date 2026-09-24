# 02 SAXPY

> 难度 ●○○○○ · 标签 `elementwise` `memory-bound` `scalar-param` · 考点:标量参数传递、访存模式

## 题目

SAXPY 是 BLAS 里最基础的一级运算,名字就是它的公式:
**S**calar **A**lpha **X** **P**lus **Y**。

```cuda
out[i] = alpha * x[i] + y[i]        i = 0 .. n-1
```

```cuda
__global__ void saxpy(const float* x, const float* y, float* out,
                      float alpha, int n);
void saxpy_launch(LaunchCtx& ctx);
```

`ctx.alpha` 是标量,`ctx.x` / `ctx.y` / `ctx.out` 是 device 指针。

## 和向量加法的区别 —— 以及为什么它们都不重要

看起来多了两样东西,但**对性能结论毫无影响**:

1. **多了个标量 `alpha`**。它作为 kernel 参数从 host 传进来,CUDA 会把它放进一块
   专门的**常量内存**(constant memory),所有线程读同一个值是天然广播的,几乎零成本。

   > 顺手纠正一个常见误解:不要为了"优化"把 `alpha` 先 `cudaMemcpy` 到 device 内存、
    > 再在 kernel 里通过指针访问 —— 那反而更慢,而且要多一次分配和搬运。

2. **多了一次乘法和一次加法**。但每个元素依然只搬 12 字节(读 `x`、读 `y`、写 `out`),
   算力需求远低于带宽供给。这就是所谓**算术强度**(arithmetic intensity)低 ——
   计算被访存完全掩盖。

所以本题和向量加法一样:**朴素实现已经是理论最优**,不可能更快。评分也看
有效带宽占峰值的百分比,而不是加速比。

## 提示

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = alpha * x[i] + y[i];
}
```

边界判断不能省 —— 用例 `ragged` 的 `n = 16777259` 是质数,不会被块大小整除。

一个能让带宽再上一台阶的进阶写法:**向量化访存**。用 `float4` 让每个线程一次搬
16 字节,指令数减少到 1/4。在访存已经饱和的题上这通常**不会**提升带宽
(带宽还是那个带宽),但它能降低指令开销 —— 值得亲手试一次,看看数字到底变不变。

## 评分

| 评级 | 带宽占峰值 | 意味着 |
|---|---|---|
| B | ≥ 50% | 访存是合并的 |
| A | ≥ 70% | 基本吃满带宽 |
| S | ≥ 85% | 逼近上限 |

小用例(总耗时不到 20µs)不参与评级。

## 思考和动手

1. **把下标改成跨步访问试试**:`int i = threadIdx.x * gridDim.x + blockIdx.x;`
   带宽会掉到多少?为什么?(这就是 coalescing 失效的样子,值得亲手感受一次。)

2. **`float4` 版本**:让每个线程处理 4 个连续元素。带宽会变吗?指令数变了吗?
   如果带宽没变,那优化的收益从哪里体现?

3. **`__restrict__` 关键字**:把参数声明成 `const float* __restrict__ x`。
   它向编译器承诺这两个指针不会指向同一块内存,于是编译器可以更自由地重排访存。
   对这道题有影响吗?对更复杂的 kernel 呢?

4. **in-place 版本**:如果把 `out` 直接写成 `ctx.y`(即 `y[i] = alpha*x[i] + y[i]`),
   结果还对吗?为什么这类"原地更新"在更复杂的算法里(比如前缀和)会出问题?

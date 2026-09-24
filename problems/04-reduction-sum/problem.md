# 04 归约求和

> 难度 ●●●○○ · 标签 `reduction` `shared-memory` `warp-shuffle` · 考点:树形归约、同步与竞态、warp shuffle、多级归约

## 题目

把 `n` 个 `float` 全部加起来,输出**一个** `float`。

```cuda
__global__ void reduce_sum(const float* input, float* output, int n);
void reduce_sum_launch(LaunchCtx& ctx);
```

注意 `ctx.output` 是**长度为 1** 的标量缓冲 —— 整个 kernel 里只应该有一个线程
(或一组通过原子操作协作的线程)写它。

## 为什么这道题是三颗星

前面两道题(向量加法、转置)的难点都在**访存模式**。这道题的难点变了,
它同时考三件事,每一件都是 CUDA 里绕不过去的:

1. **并行度**:基线只用了一个线程块。整张卡 128 个 SM,127 个在闲着。
   你要把它改成多块,于是就必须处理**块间归约**这个新问题。
2. **同步**:块内归约要靠共享内存传递中间结果,而"一个线程写、另一个线程读"
   之间必须有 `__syncthreads()`。漏掉就是竞态 —— 它最阴险的地方在于
   **有时对有时错**,靠跑一次是发现不了的。
3. **浮点误差**:求和顺序变了,结果就不再逐位相同。这在这类题里是正常的,
   但你得理解**为什么**容差是必要的,而不是以为程序写错了。

## 分三步走

### 第一步:让所有 SM 都干活

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
for (; i < n; i += blockDim.x * gridDim.x) sum += input[i];
```

跨步读取(第 `k` 轮时线程 `t` 读 `input[t + k * gridDim.x * blockDim.x]`)保证了
全局读仍然是合并的。

### 第二步:块内归约 —— 用 warp shuffle 而不是共享内存

树形归约的教科书写法是靠共享内存 + `log2(blockDim)` 轮 `__syncthreads()`。
但在现代 GPU 上,一个 warp 内的 32 个线程可以用**寄存器直传**完成归约,
既不需要共享内存也不需要同步:

```cuda
for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, off);
// 此时每个 warp 的 lane 0 持有本 warp 的部分和
```

一个 256 线程的块 = 8 个 warp。把 8 个 warp 的部分和写进共享内存,
再让第一个 warp 对这 8 个值做一次 shuffle 归约 —— 总共只需要**一次**
`__syncthreads()`,而不是 8 次。

```cuda
__shared__ float warp_sums[BLOCK / 32];
if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
__syncthreads();                    // ← 只需要这一次
if (tid < 32) { /* 对 warp_sums 再做一次 shuffle 归约 */ }
```

### 第三步:块间归约

每个块算出一个部分和,还得把它们合起来。两条路:

- **`atomicAdd(output, block_sum)`** —— 简单,适合入门。注意 `output` 必须先清零:
  框架填的是毒值(NaN),所以 launcher 里要 `cudaMemset(ctx.output, 0, sizeof(float))`。
- **两级缓冲** —— 每个块写自己的部分和到临时数组,再启一个 kernel 归约。
  多一次 kernel 启动,但避免了原子操作的串行化。

## 陷阱提醒

- **漏 `__syncthreads()`**。这是本题最容易犯的错,也是「结果不稳定」的典型来源。
  框架会把每个用例重复跑 5 次来抓它,racecheck 也能精确报出 hazard 的读写在哪个位置。
- **`atomicAdd` 和普通写不要混用**。`atomicAdd(output, v)` 是安全的;
  但只要有一个线程写成 `output[0] += v`,整个结果就不可预测了。
- **别忘边界**。用例 `odd` 的 `n = 25000001`,不整除线程总数。

## 评分

按**相对基线的加速比**评级:

| 评级 | 加速比 | 意味着 |
|---|---|---|
| B | ≥ 3x | 用上了多个线程块 |
| A | ≥ 10x | 块内归约也优化过了 |
| S | ≥ 20x | 两级归约都接近带宽上限 |

小用例(总耗时不到 20µs)不参与评级。本题里 `tiny` 专门用于竞态检查和边界考校。

## 关于容差

参考解在 CPU 上用 `double` 顺序累加,结果非常接近真值。你的 GPU 版本用的是树形
归约,求和顺序完全不同 —— 32M 个数的加法,哪怕每次只差 1 个 ulp,累积起来也会有
可观测的偏差。框架给的容差是 `atol=1e-2, rtol=1e-4`,实测优秀的实现能到 `1e-7`
量级的相对误差。

**这不是 bug,是浮点数的本性。** 想深究的话可以看:为什么两两配对(树形)累加
比顺序累加的误差小?这和你用 Kahan 求和能进一步减小误差是同一个道理。

## 思考题

1. 把块数从"覆盖全部数据"改成一个固定值(比如 `grid = 128`),配合跨步循环。
   块数更多就一定更快吗?什么时候块间归约的开销会超过并行度带来的收益?

2. `__shfl_down_sync` 的第一个参数是掩码 `0xffffffff`。如果只想让前 16 个线程参与,
   掩码该写什么?写错了会发生什么?

3. 试着把 `atomicAdd` 换成两级缓冲,性能会变好吗?在什么规模下会变好?

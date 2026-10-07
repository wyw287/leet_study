// 题目 15 的参考解:一趟遍历 + 两级归约。
//
// 这道题**只有一个洞察**,剩下的都是工程细节:
//
//   基线的三次遍历不是"算法不同",而是"同一份数据被搬了三遍"。
//   显存带宽是这题唯一的成本 —— 读一遍 256MiB 要 266µs,而三个统计量
//   全部的算术加起来只要 2µs。所以把三遍并成一遍,就是把 800µs 变成 266µs,
//   而 266µs 是**物理下限**,没有更快的可能(除非少读元素,那是错的)。
//
// 实现上就两件事:
//   ① 一个线程三个累加器(和 / 最小 / 最大),数据只在循环里读一次;
//   ② 块内共享内存树形归约 → 每块写一个三元组到 partial → 单块 kernel 收尾。
//
// 实测(2026-09-29,4090,框架自动清 L2,big 用例 n = 2^26 = 256MiB 读):
//     基线(三遍遍历)      0.855 ms   314 GB/s   31%
//     本实现(一趟)        0.289 ms   930 GB/s   92%     → 2.96x
//   92% 已经贴着"读一遍"这条物理上限(1008 GB/s → 266µs;剩下那 8% 里,
//   有一半是收尾 kernel 的固定开销,另一半是 DRAM 本身跑不满标称值)。
//
// 顺带一个必须记住的事实:**块数在这题上几乎不敏感**。把 grid 从 8192 压到
// 128(每 SM 一块),实测 907 GB/s(90%)—— 因为主 kernel 的瓶颈从头到尾都是
// 流量,不是并行度。真正能把分数吃掉的不是主 kernel,是**收尾**(见文件末尾)。

#include "ctx.h"

#define BLOCK 256
#define FINAL_BLOCK 256

// 第一级:一趟遍历,每块产出 (sum, min, max)。
// partial 布局:[blockIdx.x * 3 + {0:sum, 1:min, 2:max}]。
__global__ void array_stats(const float* __restrict__ x,
                            float* __restrict__ partial,
                            int n) {
    __shared__ float ss[BLOCK], sn[BLOCK], sx[BLOCK];
    const int tid = threadIdx.x;

    // ---- 三个累加器,一次遍历 ----
    // 这就是整个题目的答案:同一份数据只从显存来一次,三个操作共享它。
    float a  = 0.f;
    float lo = __int_as_float(0x7f800000);    // +inf
    float hi = -__int_as_float(0x7f800000);   // -inf
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = x[i];
        a  += v;
        lo  = fminf(lo, v);
        hi  = fmaxf(hi, v);
    }

    // ---- 块内归约 ----
    // 三个量共用同一轮同步:一次 __syncthreads 管三个归约。
    // 同步写在 if 外面 —— 写在里面就是"部分线程执行 barrier",未定义行为。
    ss[tid] = a; sn[tid] = lo; sx[tid] = hi;
    __syncthreads();
    for (int k = BLOCK / 2; k > 0; k >>= 1) {
        if (tid < k) {
            ss[tid] += ss[tid + k];
            sn[tid] = fminf(sn[tid], sn[tid + k]);
            sx[tid] = fmaxf(sx[tid], sx[tid + k]);
        }
        __syncthreads();
    }

    // ---- 每块写一个三元组 ----
    // 注意这是**无条件**写的:只要 grid 开得比 ceil(n / 块大小) 还大(比如图省事
    // 固定开满 max_blocks 块),就会有块一个元素都领不到 —— 它们也必须写下自己的
    // 单位元(sum=0 / min=+inf / max=-inf)。不写的话第二级 kernel 会读到框架
    // 清零后的 0,把最小值污染成 0,而且只在"块比数据多"的用例上暴露。
    if (tid == 0) {
        float* p = partial + 3 * blockIdx.x;
        p[0] = ss[0];
        p[1] = sn[0];
        p[2] = sx[0];
    }
}

// 第二级:单块,把 grid 个三元组合成三个标量。
// 这里是**覆盖写**,不是累加 —— 所以不需要 cudaMemset 去清毒值,
// 也不依赖 partial 的初值。
__global__ void array_stats_final(const float* __restrict__ partial,
                                  float* __restrict__ sum,
                                  float* __restrict__ min_val,
                                  float* __restrict__ max_val,
                                  int nblocks) {
    __shared__ float ss[FINAL_BLOCK], sn[FINAL_BLOCK], sx[FINAL_BLOCK];
    const int tid = threadIdx.x;

    float a  = 0.f;
    float lo = __int_as_float(0x7f800000);
    float hi = -__int_as_float(0x7f800000);
    for (int b = tid; b < nblocks; b += FINAL_BLOCK) {
        a  += partial[3 * b + 0];
        lo  = fminf(lo, partial[3 * b + 1]);
        hi  = fmaxf(hi, partial[3 * b + 2]);
    }

    ss[tid] = a; sn[tid] = lo; sx[tid] = hi;
    __syncthreads();
    for (int k = FINAL_BLOCK / 2; k > 0; k >>= 1) {
        if (tid < k) {
            ss[tid] += ss[tid + k];
            sn[tid] = fminf(sn[tid], sn[tid + k]);
            sx[tid] = fmaxf(sx[tid], sx[tid + k]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum[0]     = ss[0];
        min_val[0] = sn[0];
        max_val[0] = sx[0];
    }
}

void array_stats_launch(LaunchCtx& ctx) {
    const int block = BLOCK;

    // 块数:4090 有 128 个 SM,块数低于这个数就有 SM 在闲着。
    // 上限是 partial 的容量(见 spec 的 max_blocks),这里直接给满 8192 ——
    // 每个块分到的元素多一个少一个都无所谓,归约的固定开销在 µs 量级。
    int grid = (ctx.n + block - 1) / block;
    if (grid > ctx.max_blocks) grid = ctx.max_blocks;
    if (grid < 1) grid = 1;

    array_stats<<<grid, block, 0, ctx.stream>>>(ctx.x, ctx.partial, ctx.n);
    array_stats_final<<<1, FINAL_BLOCK, 0, ctx.stream>>>(ctx.partial, ctx.sum,
                                                         ctx.min_val, ctx.max_val, grid);
}

// ---------------------------------------------------------------------------
// 值得亲手验证的四件事(都真跑过,数字是实测的):
//
// 1. **让第二级 kernel 只用一个线程串行扫 partial 会怎样?**
//    这是本题最值钱的坑。8192 块时那一趟要花 ~0.47ms —— **比主 kernel 的
//    0.29ms 还贵**,总时间 0.762ms,只比三趟的基线快 12%(35% 峰值,评级 C)。
//    而同一个串行收尾,块数封在 1024 时只要 35µs(总 0.324ms,82%)。
//    同一个错误,块数差 8 倍,代价差 13 倍 —— 因为串行收尾的成本是 O(块数),
//    主 kernel 的成本是 O(n)。**"融合"和"每一级都并行"是两件独立的事。**
//
// 2. **float4 有用吗?** 让每个线程一次搬 16 字节、做 4 组更新,访存指令数
//    降到 1/4。实测 0.289 ms → 0.289 ms,**逐位相同**:瓶颈在 DRAM 那堵墙上,
//    不在指令发射上(和 01 题的结论一样)。它唯一的副作用是 n 不被 4 整除时
//    多出一个尾巴要处理 —— 而 ragged 用例正是为此设的。
//
// 3. **再往前一步:把 sum 换成 sum + sumsq(平方和),成本增加多少?**
//    答案是"几乎为零" —— 多一个累加器、多一个 FMA,但**一个字节都不多读**。
//    这就是融合的全部意义:在已经付过路费的车上多带一件行李。
//    反过来,如果在第二个 kernel 里算 sumsq,你就要再付一次 266µs。
//
// 4. **块数从 8192 降到 128 会怎样?** 实测 907 GB/s(90%),只差 2 个百分点。
//    每块分到的元素从 32 个变成 512 个,循环变长,但访存依然合并、并行度对
//    128 个 SM 来说刚好。**这题的瓶颈从头到尾都不是并行度,而是流量。**
// ---------------------------------------------------------------------------

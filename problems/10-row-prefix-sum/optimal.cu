// 题目 10 的参考解
//
// 一句话:**把并行度从"行间"挪到"行内",把访问从"跨行"变成"连续"。**
//
// 一个 block 负责一行,三步(段与段之间用 __syncthreads 隔开):
//
//   ① 合并地把这一行搬进共享内存(`c = t; c += BLOCK` 是跨步写法,
//      同一 warp 的 32 个线程地址连续 —— 基线缺的正是这个)
//   ② 每个线程把自己那**一段连续**的元素串行扫掉(段内累加),
//      顺便记下这一段的总和 → 得到 BLOCK 个"段总和"
//   ③ 对这 BLOCK 个段总和做一次块内 inclusive scan,减去自身 = exclusive
//      偏移;每个线程把偏移加回自己那一段,合并地写回 DRAM
//
// 为什么第②步要"每段串行"?这正是并行 scan 的核心技巧:
//   **串行 prefix sum 的全部代价是那条依赖链,而不是那 n 次加法。**
//   把 n 个元素切成 BLOCK 段,每段内部的链长就只有 K = n/BLOCK;
//   段与段之间的先后关系由一个只有 BLOCK 个元素的 scan 解决(代价可忽略)。
//   于是总功 O(n),关键路径 O(n/BLOCK + log BLOCK)。
//
// 对比 Hillis-Steele(模板注释里的路 A):它每轮步长翻倍,log2(cols) 轮完成,
// 关键路径更短,但**总功是 O(cols·log cols)** —— 比本解多做了 log2(cols) ≈ 10 倍
// 的加法与共享内存读写。那为什么两者实测几乎一样快?见下面"两个实测事实"。
//
// 访存账(square 用例 32768×1024 = 134MB):
//
//   DRAM     : 读 x 一遍 + 写 y 一遍 = 8 字节/元素 = 268MB → 下限约 270µs
//   共享内存 : 写 4 + (读 4 + 写 4) + 读 4 = 16 字节/元素
//
// 共享内存看着是 DRAM 的两倍,但它是**片上**的:4090 每个 SM 每周期 128 字节,
// 128 个 SM × 2.5GHz ≈ 41TB/s,是 DRAM 峰值(1TB/s)的 40 倍。
// 所以那 16 字节/元素折合下来不到 1 个周期,而省掉的 DRAM 往返是几百微秒。
//
// 另一个细节:第②步里线程 t 访问的是 row[t*K + j]。同一个 warp 内相邻线程的
// 地址差 K 个 float —— K=4 时是 4 路 bank conflict。要在意吗?不用:
// 4 路冲突只是把那 16 字节/元素的片上流量乘 4,仍然远低于 DRAM 那份。
// (真要在意,就把共享内存的段长补成奇数 row[t*(K|1) + j],那样冲突就没了。
//  这属于"片上带宽也成了瓶颈"时才需要操心的优化。)

#include "ctx.h"

#define BLOCK 256

// ---------------------------------------------------------------------------
// 块内 inclusive scan:只有 BLOCK 个数,代价可以忽略,所以直接用最直白的
// Hillis-Steele。写法上唯一要小心的是**原地扫描的读写竞争**:
// 本轮线程 A 写下的 red[c] 不能被本轮线程 B 当成 red[c-d] 读走,
// 所以拆成"读进寄存器 → 同步 → 写回 → 同步"。少一个 __syncthreads,
// 正确性对拍可能照样过(它只是"有时对"),racecheck 一定会报。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_scan_incl(float v, float* red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    #pragma unroll
    for (int d = 1; d < BLOCK; d <<= 1) {
        const float x = red[t] + ((t >= d) ? red[t - d] : 0.f);
        __syncthreads();
        red[t] = x;
        __syncthreads();
    }
    return red[t];
}

__global__ void row_scan(const float* __restrict__ x,
                         float* __restrict__ y,
                         int rows, int cols) {
    extern __shared__ float sh[];
    float* row = sh;          // 前 cols 个 float:这一行的数据
    float* red = sh + cols;   // 后面 BLOCK 个 float:块内 scan 的暂存区

    const int r = blockIdx.x;              // 一个 block 负责一行
    if (r >= rows) return;                 // 网格就是 rows 个 block,这句只是保险
    const int t = threadIdx.x;
    const float* xr = x + (size_t)r * (size_t)cols;
    float*       yr = y + (size_t)r * (size_t)cols;

    // ① 合并地搬进共享内存
    for (int c = t; c < cols; c += BLOCK) row[c] = xr[c];
    __syncthreads();

    // ② 每线程扫自己那一段连续区间。段长 K 由运行时决定;
    //    cols 不整除 BLOCK 时最后一段会被 hi 截断,领不到元素的线程
    //    (lo >= cols)一次循环都不进,acc 保持 0,不参与结果也不影响别人。
    const int K  = (cols + BLOCK - 1) / BLOCK;
    const int lo = t * K;
    const int hi = (lo + K < cols) ? (lo + K) : cols;

    float acc = 0.f;
    for (int c = lo; c < hi; ++c) {
        acc += row[c];
        row[c] = acc;                      // 段内 inclusive
    }

    // ③ 段间:块内 scan 段总和,减去自己这一段 = exclusive 偏移。
    //    空段的 acc 是 0,它在 scan 里的位置仍然正确(偏移 = 它前面所有段的和)。
    const float off = block_scan_incl(acc, red) - acc;

    // ④ 加偏移写回。仍然是"每线程一段连续",warp 内相邻线程差 K 个元素 ——
    //    K×32 = 128 字节正好是一条 cache line,所以每个 store 指令还是只碰
    //    一条线,合并效率没有损失。
    for (int c = lo; c < hi; ++c) yr[c] = row[c] + off;
}

void row_scan_launch(LaunchCtx& ctx) {
    // 一行一个 block。动态共享内存 = 行数据 + 块内 scan 暂存区。
    // 每块 5~6KB,256 线程 —— 一个 SM 上能同时驻留 8 个 block(受线程数限制),
    // 占用率是满的,显存延迟有足够多的 warp 去遮盖。
    const size_t shbytes = (size_t)(ctx.cols + BLOCK) * sizeof(float);
    row_scan<<<ctx.rows, BLOCK, shbytes>>>(ctx.x, ctx.y, ctx.rows, ctx.cols);
}

// ---------------------------------------------------------------------------
// 四个实测数字(2026-09-27,4090,框架自动清 L2;square 用例 = 134MB)——
// 每一个都值得你亲手验一遍,因为它们很可能和你的直觉不一样:
//
//   实现                                          耗时       加速比
//   把 scan 做在全局内存上(Hillis-Steele,10 趟)   1.230 ms    0.80x
//   基线:一个线程扫一行                            0.983 ms    1.00x
//   半成品:搬进共享内存,但块内只让线程 0 串行扫     0.253 ms    3.89x
//   本解(两段式,块内也并行)                      0.241 ms    4.08x
//
// 三条结论:
//
// 1. **值钱的是"把整行搬进共享内存 + 访存合并",不是"块内并行"。**
//    第三行和第四行只差 5% —— 块内那条 cols 长的串行依赖链,被同一个 SM 上
//    同时驻留的 8 个 block 之间的并行度遮盖掉了:每周期 DRAM 只搬得动约
//    3 字节/SM(2.6 周期/元素),而 8 条独立串行链合起来每 4 周期能推进 8 个元素
//    (0.5 周期/元素)。差 5 倍,所以遮得住。
//    这条结论有前提:**行装得下、block 驻留数够多**。行再长一倍,共享内存
//    把驻留数压到 1~2 个,串行链就遮不住了,那时两段式才会明显领先。
//    所以别背结论,要会算 —— 这就是"先测量再优化"的意思。
//
// 2. **在全局内存上做 scan 比"一个线程扫一行"还慢(0.80x)。**
//    它每轮读 2n、写 n,共 log₂(cols)=10 轮 → 120 字节/元素的 DRAM 流量;
//    而基线虽然访存不合并(每个 warp 的 32 个线程扫 32 个不同的行),
//    流量也只有约 32 字节/元素。算法对 ≠ 快。
//
// 3. 这道题的加速比是**台阶**,不是斜坡:1.0x 和 3.9x 之间基本是空的。
//    因为 DRAM 流量被题目定死了(读 4 + 写 4 = 8 字节/元素),任何"合并访存 +
//    并行度足够"的实现都会撞到同一堵墙(268MB / 1TB/s ≈ 270µs)。
//    这也解释了为什么门槛在 3.5x 而不是 2x —— 中间地带只有"修了一半"的实现。
//
// 想把"跨 block 的全局 scan"也拿下(本题一个 block 只碰一行,不需要跨 block
// 通信),看题面最后的思考题 3。
// ---------------------------------------------------------------------------

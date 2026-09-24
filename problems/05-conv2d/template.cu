// 题目 05:二维卷积(valid 模式)
//
// 语义 —— 每个输出像素是输入图上一块 kh×kw 窗口与卷积核的加权和:
//
//     out[oy][ox] = Σ_{ky<kh, kx<kw} image[oy+ky][ox+kx] * kernel[ky][kx]
//
// image 是 h 行 w 列、kernel 是 kh 行 kw 列、out 是 oh 行 ow 列(row-major),
// 其中 oh = h-kh+1、ow = w-kw+1 —— 是「valid」卷积,不补零,
// 所以每个 out 像素的 kh*kw 个采样点都必然落在 image 内部。
//
//   ctx.image   const float*  device 指针,长度 h*w
//   ctx.kernel  const float*  device 指针,长度 kh*kw
//   ctx.out     float*        device 指针,长度 oh*ow,需要你写入
//   ctx.h/ctx.w        输入图的尺寸
//   ctx.kh/ctx.kw      卷积核的尺寸(运行时才知道,不是一个固定常数)
//   ctx.oh/ctx.ow      输出的尺寸
//
// 这题的数据量换算:1 个输出像素 = kh*kw 次乘加。k7 用例里
// 4090² × 49 ≈ 8.2 亿次乘加,也就是 8.2 亿次采样点读取 × 4B ≈ 3.3GB 的访存流量,
// 而输入图只有 64MB。多出来的 3.2GB 全是重复读。你要做的就是把它们去掉。

#include "ctx.h"

// ---------------------------------------------------------------------------
// 1) 写 kernel
// ---------------------------------------------------------------------------
__global__ void conv2d(const float* __restrict__ image,
                       const float* __restrict__ kernel,
                       float* __restrict__ out,
                       int h, int w, int kh, int kw, int oh, int ow) {
    // TODO: 算出本线程负责的输出像素 (oy, ox),判断边界,累加 kh*kw 个乘加后写入 out。
    //
    // 朴素写法(也是本题基线的做法)是这样的:
    //
    //     int ox = blockIdx.x * blockDim.x + threadIdx.x;
    //     int oy = blockIdx.y * blockDim.y + threadIdx.y;
    //     if (oy >= oh || ox >= ow) return;
    //     float acc = 0.f;
    //     for (int ky = 0; ky < kh; ++ky)
    //         for (int kx = 0; kx < kw; ++kx)
    //             acc += image[(long long)(oy+ky)*w + ox+kx] * kernel[ky*kw+kx];
    //     out[(long long)oy*ow + ox] = acc;
    //
    // 它能跑对,但每个线程要发 kh*kw 次全局访存,而相邻线程的窗口是重叠的 ——
    // 数据被反复从 L1 搬进寄存器,片上的访存吞吐被这份冗余吃光了。
    //
    // 思路(两道坎,第二道才是分水岭):
    //
    //   ① 共享内存分块:每个线程块负责输出图上 TILE×TILE 的一块。
    //      它需要的输入是 (TILE+kh-1) × (TILE+kw-1) 的一块 —— 多出来的
    //      kh-1 / kw-1 就是分块卷积里的 halo(光晕)。
    //      协作把这一整块(含 halo)搬进共享内存,__syncthreads(),
    //      然后每个线程只从共享内存里取自己那 kh*kw 个采样点。
    //      冗余从「每像素 kh*kw 次全局读」降到「每像素约 1 次全局读」。
    //
    //      注意:kh/kw 是运行时参数,所以共享内存的大小要动态算
    //      (extern __shared__ float sm[]; 启动时把字节数作为第三个参数传进去),
    //      或者用一个足够大的静态数组 + 上限判断。
    //
    //   ② 寄存器分块:只做①的话,内层循环仍然是 kh*kw 次**共享内存**读 ——
    //      共享内存和 L1 在芯片上是同一块 SRAM,吞吐一样,所以①拿到的收益
    //      只是把「全局冗余读」变成「共享内存读」,次数没变!
    //      真正的加速来自让一个线程算**多个相邻的输出**(比如 x 方向连续 4 个):
    //      一行窗口里读 (4+kw-1) 个值,就能算出 4 个输出的这一行贡献,
    //      于是每读 1 个值摊到 ~4 次 FMA 上,而不是 1 次。
    //
    // 还有个容易被忽略的细节:合并访问。一个 warp 32 个 float = 128B,
    // 只有当起始地址 128B 对齐时才是「一次」访存;窗口平移 kx 格之后,
    // 大部分 tap 的起始地址是错位的,一次 warp 访存要拆成两次。
    // 共享内存没有这个问题(它按 bank 寻址,32 个连续 float 恰好铺满 32 个 bank),
    // 这本身也是一部分收益。
    //
    // 踩坑提示:
    //   * 填 halo 的时候必须判断边界:row0+r < h、col0+c < w,越界的位置填 0。
    //     越界写会被框架的哨兵区抓到,越界读则可能读到别的缓冲上。
    //   * __syncthreads() 不能漏,也不能放在分支里(会死锁或读到没填完的数据)。
    //   * 最后一个 block 的 tile 会超出 oh/ow:算出来的输出位置要判断再写。
    //   * 共享内存数组别写成 [TILE][TILE]:halo 要求它是
    //     [TILE+kh-1][TILE+kw-1];并且行距建议留个 padding,
    //     否则内层按列取数时容易撞 bank。
}

// ---------------------------------------------------------------------------
// 2) 决定启动配置
// ---------------------------------------------------------------------------
void conv2d_launch(LaunchCtx& ctx) {
    // TODO: 选块大小与网格大小,然后启动 kernel
    //
    // 朴素版提示:
    //   dim3 block(32, 8);
    //   dim3 grid((ctx.ow + block.x - 1) / block.x,
    //             (ctx.oh + block.y - 1) / block.y);
    //   conv2d<<<grid, block>>>(ctx.image, ctx.kernel, ctx.out,
    //                           ctx.h, ctx.w, ctx.kh, ctx.kw, ctx.oh, ctx.ow);
    //
    // 分块版还要在第三个启动参数里给出动态共享内存的字节数。
}

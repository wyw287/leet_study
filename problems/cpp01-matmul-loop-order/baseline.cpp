// 性能基线 —— 也是学习者拿到的起点代码。
//
// 这是**最自然的写法**:照着 C[i][j] = sum_k A[i][k]*B[k][j] 的定义直译。
// 它完全正确,数值上也挑不出毛病 —— 问题只出在访存顺序上。
//
// 具体地说,内层循环 k 递增时:
//   A[i*n+k] —— 连续访问,一个 cache line 装 16 个 float,摊下来每元素 0.25 次访存
//   B[k*n+j] —— **跨步访问**,步长 n*4 = 1536 字节,每读一个元素就换一条 cache line
//
// 于是内层循环每算一个乘加,就要从内存里搬一条 64 字节的 cache line,
// 而其中只有 4 个字节被用上 —— 有效利用率 1/16。

#include "ctx.h"

void matmul(Ctx& ctx) {
    const int n = ctx.n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k) {
                s += ctx.A[i * n + k] * ctx.B[k * n + j];
            }
            ctx.C[i * n + j] = s;
        }
    }
}

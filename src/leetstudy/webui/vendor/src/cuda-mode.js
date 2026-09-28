// CUDA 模式 = clike 的 cpp 模式 + 一层 CUDA 词表前置拦截。
//
// 为什么不写完整的 tokenizer:clike 已经把 C++ 那部分(注释、字符串、数字、
// 预处理、括号配对、缩进)做对了,而且它支持 `state.tokenize` 覆盖 —— 这正是
// 给 C++ 加方言的标准挂载点。我们只在它前面拦一次「整个标识符」,
// 命中 CUDA 词表就返回对应样式,否则原样交还给 clike。
//
// 样式名用的是 CodeMirror 5 legacy 字符串,StreamLanguage 会经 tokenTable
// 映射成 Tag。可用的名字见 @codemirror/language 的 defaultTokenTable。
import { cpp } from "@codemirror/legacy-modes/mode/clike"

// 归为 typeName(红)—— 与 Pygments 的 Token.Keyword.Type 对齐。
// 注意存储/访问限定符也在这里:Pygments 的 CudaLexer 就是这么分的
// (__shared__ / __restrict__ 是 kt,而 __global__ 是 kr)。
const CUDA_TYPES = `
  float2 float3 float4 double2 double3 double4 half2 half4
  char2 char4 uchar2 uchar4 short2 short4 ushort2 ushort4
  int2 int3 int4 uint2 uint3 uint4 long2 long4 ulong2 ulong4
  longlong2 ulonglong2 dim3
  __shared__ __constant__ __managed__ __restrict__ __align__
  cudaStream_t cudaEvent_t cudaError_t cudaDeviceProp cudaMemcpyKind
  cudaTextureObject_t cudaSurfaceObject_t
`

// 归为 builtin(绿,Pygments 的 nb)—— 这五个是「内核自带的变量」,
// 单独一类是因为 monokai 暗色主题下 nb(#A6E22E)和 k(#66D9EF)不是同一个颜色,
// 合进 keyword 会在暗色下露馅。
const CUDA_BUILTINS = `threadIdx blockIdx blockDim gridDim warpSize`

// 归为 keyword(绿)—— 与 Pygments 的 k / kr 对齐(默认配色里这俩都是绿色粗体)。
// 这一类比 Pygments 的 CUDA lexer 覆盖得广:Pygments 只认 __global__/__forceinline__
// 等少数几个,把 atomicAdd / cudaMalloc / __syncwarp 都当成普通名字(不着色)。
// 这里补齐 —— 同样是绿,只是让 CUDA 自己的东西都亮起来。
const CUDA_KEYWORDS = `
  __global__ __device__ __host__ __forceinline__ __noinline__ __inline__
  __launch_bounds__ __grid_constant__
  __syncthreads __syncthreads_count __syncthreads_and __syncthreads_or
  __syncwarp __threadfence __threadfence_block __threadfence_system
  __shfl_sync __shfl_up_sync __shfl_down_sync __shfl_xor_sync
  __shfl __shfl_up __shfl_down __shfl_xor
  __ballot_sync __all_sync __any_sync __activemask __match_any_sync
  __popc __popcll __ffs __ffsll __clz __clzll __brev __brevll
  __mul24 __umul24 __mulhi __umulhi __sad __usad
  __ldg __ldcv __ldca __stcg __stcs
  __expf __exp10f __logf __log2f __powf __sinf __cosf __sincosf __tanf
  __fdividef __frcp_rn __fsqrt_rn __saturatef fmaf
  atomicAdd atomicSub atomicExch atomicMin atomicMax atomicInc atomicDec
  atomicCAS atomicAnd atomicOr atomicXor
  cudaMalloc cudaMallocManaged cudaFree cudaMemcpy cudaMemcpyAsync cudaMemset
  cudaDeviceSynchronize cudaGetLastError cudaPeekAtLastError cudaGetErrorString
  cudaStreamCreate cudaStreamDestroy cudaStreamSynchronize
  cudaEventCreate cudaEventRecord cudaEventElapsedTime cudaEventSynchronize
  cudaOccupancyMaxActiveBlocksPerMultiprocessor
`

const TYPES = new Set(CUDA_TYPES.trim().split(/\s+/))
const BUILTINS = new Set(CUDA_BUILTINS.trim().split(/\s+/))
const KEYWORDS = new Set(CUDA_KEYWORDS.trim().split(/\s+/))

export const cuda = {
  name: "cuda",

  // 这四个原样转发给 clike —— 尤其是 indent/copyState,
  // 状态对象里有 clike 自己的 context 栈,自己实现会漏。
  startState: cpp.startState,
  copyState: cpp.copyState,
  indent: cpp.indent,
  languageData: cpp.languageData,

  token(stream, state) {
    // state.tokenize 非空 = 正处在注释/字符串/预处理指令里,
    // 那些上下文里再认关键字就会把注释里的 atomicAdd 也点亮。
    if (!state.tokenize) {
      const save = stream.pos
      stream.eatWhile(/[\w$]/)
      const word = stream.current()
      if (word) {
        if (TYPES.has(word)) return "typeName"
        if (BUILTINS.has(word)) return "builtin"
        if (KEYWORDS.has(word)) return "keyword"
      }
      stream.pos = save // 没命中就完全交还给 clike,不留痕迹
    }
    const style = cpp.token(stream, state)

    // 指针/引用的 `*` `&` 紧跟在类型后面时,clike 会把它们也算进类型
    // (`float*` 里的星号拿到的是 "type")。Pygments 是按运算符着色的
    // (kt 后面跟 o),这里改判 —— 340 字符的样本里这是最后一处配色差异。
    //
    // 用改判而不是「把 star 退回给下一次调用」:后者要回退 stream.pos,
    // 而那个位置已经被 clike 用来更新 state 了,两边会错位。
    if (style === "type" && /^[*&]+$/.test(stream.current())) return "operator"
    return style
  },
}

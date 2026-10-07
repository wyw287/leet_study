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

// ---------------------------------------------------------------------------
// 词表
//
// **一份数据,两处用途**:编辑器上色(token() 按 kind 定样式名),补全下拉
// (按 kind 给图标、按组的说明给 detail)。合成一张表是为了两边不漂移 ——
// 加一个词只需要加一行。
//
// kind 只有三类,因为样式名只有三类可映射:
//   type    → "typeName"  (红)   Pygments 的 Token.Keyword.Type
//   builtin → "builtin"   (绿)   Pygments 的 Name.Builtin(nb)
//   kw      → "keyword"   (绿)   Pygments 的 k / kr
//
// 分组只影响补全下拉里的 detail,不影响上色 —— 所以可以按语义随便分。
// ---------------------------------------------------------------------------

// ---- 类型(红)----
const T = `
  float2 float3 float4 double2 double3 double4 half2 half4
  char2 char4 uchar2 uchar4 short2 short4 ushort2 ushort4
  int2 int3 int4 uint2 uint3 uint4 long2 long4 ulong2 ulong4
  longlong2 ulonglong2 dim3
  __shared__ __constant__ __managed__ __restrict__ __align__
  cudaStream_t cudaEvent_t cudaError_t cudaDeviceProp cudaMemcpyKind
  cudaTextureObject_t cudaSurfaceObject_t
`

// ---- 内建变量(绿,nb)—— 单独一类是因为 monokai 暗色下 nb(#A6E22E)和
//      k(#66D9EF)不是一个颜色,合进 keyword 会在暗色主题露馅。
const BUILTIN = `threadIdx blockIdx blockDim gridDim warpSize`

// ---- 关键字:按语义分组,补全时用组名当说明 ----
const GROUPS = [
  ["限定符", `
    __global__ __device__ __host__ __forceinline__ __noinline__ __inline__
    __launch_bounds__ __grid_constant__
  `],
  ["块内/设备同步", `
    __syncthreads __syncthreads_count __syncthreads_and __syncthreads_or
    __syncwarp __threadfence __threadfence_block __threadfence_system
  `],
  ["warp 交换/投票", `
    __shfl_sync __shfl_up_sync __shfl_down_sync __shfl_xor_sync
    __shfl __shfl_up __shfl_down __shfl_xor
    __ballot_sync __all_sync __any_sync __activemask __match_any_sync
  `],
  ["原子操作", `
    atomicAdd atomicSub atomicExch atomicMin atomicMax atomicInc atomicDec
    atomicCAS atomicAnd atomicOr atomicXor
  `],
  ["位运算", `__popc __popcll __ffs __ffsll __clz __clzll __brev __brevll
              __mul24 __umul24 __mulhi __umulhi __sad __usad`],
  ["快速数学", `
    __expf __exp10f __logf __log2f __powf __sinf __cosf __sincosf __tanf
    __fdividef __frcp_rn __fsqrt_rn __saturatef fmaf
  `],
  ["缓存/访存提示", `__ldg __ldcv __ldca __stcg __stcs`],
  ["运行时 API", `
    cudaMalloc cudaMallocManaged cudaFree cudaMemcpy cudaMemcpyAsync cudaMemset
    cudaDeviceSynchronize cudaGetLastError cudaPeekAtLastError cudaGetErrorString
    cudaStreamCreate cudaStreamDestroy cudaStreamSynchronize
    cudaEventCreate cudaEventRecord cudaEventElapsedTime cudaEventSynchronize
    cudaOccupancyMaxActiveBlocksPerMultiprocessor
  `],
]

const split = s => s.trim().split(/\s+/)
const TYPES = new Set(split(T))
const BUILTINS = new Set(split(BUILTIN))

// 词 → 说明(组名)。同一个词出现在多组时以**先出现的**为准。
const DETAIL = new Map()
for (const [group, words] of GROUPS)
  for (const w of split(words))
    if (!DETAIL.has(w)) DETAIL.set(w, group)
const KEYWORDS = new Set(DETAIL.keys())

// token() 的返回值:legacy 样式名。只有这三类。
const STYLE = {type: "typeName", builtin: "builtin", kw: "keyword"}

// ---------------------------------------------------------------------------
// 补全
//
// 两类来源,CM6 的 `autocompletion()` 会把它们合起来(见文件末尾的
// languageData —— 它和 clike 自带的 C++ 关键字表**并列**,不是替换):
//
//   ① 这张词表 + clike 的 C++ 关键字  → 静态,`completeFromList`
//   ② 文档里已经出现过的标识符         → `completeAnyWord`(前端挂,见 index.html)
//
// 注意 ② 是**词形匹配**,不是语义补全:它能把 `ctx`、`n`、`block` 这些你已经
// 写过的名字补出来,但**不认识类型** —— 所以 `ctx.` 之后不会列出结构体成员。
// 那需要真正的语义分析(知道 ctx 是 LaunchCtx),CM6 的流式分词器做不到。
//
// ⚠️ **两路之间不去重,这是已知且接受的行为。**
// 打 `blo` 时 `blockDim` 会出现两次:一次来自这张表(带「内建变量」说明),
// 一次来自 completeAnyWord(文档里出现过 `blockDim.x`,没说明)。
// 想过三个办法,选了最省事的:
//   · 自己扫文档、合并成一路    —— 能去重,但要多写 20 行,还得自己维护
//   · 把内建从这张表里删掉      —— 新文件(还没写过 threadIdx)就补不出来了
//   · 接受重复                  ← 现在的选择
// 重复项的说明文字不一样,看起来更像"两个来源"而不是"坏了"。真要治的话,
// 上面第一条是正路。
// ---------------------------------------------------------------------------

// clike 自带的 C++ 关键字/类型/字面量,拍平成一个字符串数组。
const CPP_WORDS = cpp.languageData.autocomplete || []

const COMPLETIONS = [
  ...[...TYPES].map(label => ({label, type: "type", detail: "CUDA 类型"})),
  ...[...BUILTINS].map(label => ({label, type: "variable", detail: "内建变量"})),
  ...[...KEYWORDS].map(label => ({label, type: "function", detail: DETAIL.get(label)})),
  ...CPP_WORDS.map(label => ({label, type: "keyword"})),
]

export const cuda = {
  name: "cuda",

  // 这四个原样转发给 clike —— 尤其是 indent/copyState,
  // 状态对象里有 clike 自己的 context 栈,自己实现会漏。
  startState: cpp.startState,
  copyState: cpp.copyState,
  indent: cpp.indent,

  // autocomplete 这一项**追加**在 clike 的后面:两边的值都会被
  // languageDataAt 收集,再各自 asSource() 成一路补全来源。
  languageData: {...cpp.languageData, autocomplete: COMPLETIONS},

  token(stream, state) {
    // state.tokenize 非空 = 正处在注释/字符串/预处理指令里,
    // 那些上下文里再认关键字就会把注释里的 atomicAdd 也点亮。
    if (!state.tokenize) {
      const save = stream.pos
      stream.eatWhile(/[\w$]/)
      const word = stream.current()
      if (word) {
        if (TYPES.has(word)) return STYLE.type
        if (BUILTINS.has(word)) return STYLE.builtin
        if (KEYWORDS.has(word)) return STYLE.kw
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

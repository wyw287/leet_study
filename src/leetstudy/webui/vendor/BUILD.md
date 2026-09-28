# vendor/ —— CodeMirror 6 打包产物

`codemirror.js` 是**提交进仓库的产物**,前端直接 `<script type="module">` 引它。
`src/` 下是它的全部构建输入 —— 产物是压缩过的,不把输入留下的话,
那张 CUDA 词表和模式代码就没人改得动了。

## 为什么要打包

CodeMirror 6 是一堆 ESM 小包(`@codemirror/state` / `view` / `language` …),
浏览器里裸 `import` 会打出几十个请求。esbuild 把它们合成一个文件。

## 重新构建

```bash
# 在一个临时目录里装依赖(别把 node_modules 弄进仓库)
mkdir -p /tmp/cm6build && cd /tmp/cm6build
cp <仓库>/src/leetstudy/webui/vendor/src/* .
PATH="$PWD/../.venv/bin:$PATH" npm install          # 见下:代理
./node_modules/.bin/esbuild entry.js --bundle --format=esm --minify \
    --outfile=codemirror.js --log-level=warning
cp codemirror.js <仓库>/src/leetstudy/webui/vendor/codemirror.js
```

**npm 要走代理**,和 pip 的分流规则相反(见 CLAUDE.md 的「环境约束」):

```bash
npm config set proxy http://127.0.0.1:11451
npm config set https-proxy http://127.0.0.1:11451
```

## 改 CUDA 模式

`src/cuda-mode.js`。它是 `clike` 的 `cpp` 模式加一层前置拦截 ——
只有一张词表(CUDA_TYPES / CUDA_BUILTINS / CUDA_KEYWORDS)加两条改判规则,
其余全部转发给 clike。改完必须重新构建。

`entry.js` 只导出前端真正用到的东西,多导一个没用到的模块会让 esbuild
没法摇掉它 —— 现在 346 KB(113 KB gzip)。

## 配色不能自己写死

前端的 HighlightStyle 里**没有一个是硬编码的颜色**,全部绑到 `--py-*` CSS 变量。
那些变量由 `server.py` 的 `_highlight_css()` 从 Pygments 自己生成的 CSS 里
正则抠出来(和「参考解」页共用同一份定义)。所以换 Pygments 主题或版本时,
编辑器会跟着变,不会漂移。

变量名到 Pygments token 的对应见 `server.py` 的 `_CSS_VARS`。

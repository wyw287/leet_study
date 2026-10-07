// 打包入口:只导出前端真正要用的东西,方便 esbuild 摇掉其余部分。
export { EditorView, keymap, lineNumbers, highlightActiveLine,
         highlightActiveLineGutter, drawSelection, placeholder } from "@codemirror/view"
export { EditorState, Compartment } from "@codemirror/state"
export { StreamLanguage, HighlightStyle, syntaxHighlighting,
         bracketMatching, indentUnit, indentOnInput } from "@codemirror/language"
export { defaultKeymap, history, historyKeymap, indentWithTab,
         indentMore, indentLess, toggleComment } from "@codemirror/commands"
export { searchKeymap, highlightSelectionMatches } from "@codemirror/search"
// 补全:autocompletion() 是扩展,completeAnyWord 是"文档里出现过的词"那一路,
// completionKeymap 提供 Ctrl-Space 手动触发和上下键选择。
// closeBrackets:打左括号自动补右括号。三套括号配置不用我们写 ——
// clike 和 python 两个模式**各自在 languageData 里声明了**
// (`( [ { ' "`,python 还多了三引号),CM6 会自己读。
export { autocompletion, completeAnyWord, completionKeymap,
         closeBrackets, closeBracketsKeymap } from "@codemirror/autocomplete"
export { Tag, tags } from "@lezer/highlight"
// 三个科目各一个模式,和 server.py 的 _LEXER_BY_EXT 一一对应。
// cuda 是自己写的(见 cuda-mode.js),另外两个用 legacy-modes 现成的。
export { python } from "@codemirror/legacy-modes/mode/python"
export { cpp } from "@codemirror/legacy-modes/mode/clike"
export { cuda } from "./cuda-mode.js"

"""调用本地 claude CLI(headless)完成出题与讲评。

为什么用 `claude -p` 而不是自己调 API:
  * 复用用户已经配好的登录与模型接入(本机是自定义模型,不能硬编码 --model)
  * 让 claude 自己用 Read/Write/Edit/Bash 工具去迭代 —— 出题是一个
    「写 → 编译 → 跑 → 看结果 → 改」的循环,交给它自己闭环比我们在外面
    编排要自然得多

权限策略:用白名单 + acceptEdits,只放开「读写题目文件」与「跑编译/判题命令」,
不用 --dangerously-skip-permissions。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from ..config import Config

# 默认放开的工具。Bash 只放开与本框架相关的命令(前缀匹配),不放开通配的 Bash ——
# 出题需要编译和自验证,这些命令必须能用,但没理由让它跑任意 shell。
DEFAULT_ALLOWED_TOOLS = [
    "Read", "Write", "Edit", "Glob", "Grep",
    "Bash(leet:*)",
    "Bash(.venv/bin/leet:*)",
    "Bash(nvcc:*)",
    "Bash(compute-sanitizer:*)",
]

EventFn = Callable[[str], None]


@dataclass
class AgentResult:
    ok: bool
    text: str = ""
    error: str = ""
    returncode: int = 0
    seconds: float = 0.0
    log: List[str] = field(default_factory=list)


def claude_path(cfg: Config) -> Optional[str]:
    p = shutil.which(cfg.claude_bin)
    if p:
        return p
    if Path(cfg.claude_bin).is_file():
        return cfg.claude_bin
    return None


def _describe_tool(block: dict) -> str:
    """把一次工具调用描述成一行人类可读的进度。"""
    name = block.get("name") or "?"
    args = block.get("input") or {}

    def short(p: object) -> str:
        s = str(p or "")
        # 只显示相对题库/仓库根的路径,免得刷屏
        for marker in ("problems/", "solutions/", "build/"):
            idx = s.find(marker)
            if idx >= 0:
                return s[idx:]
        return Path(s).name if "/" in s else s

    if name in ("Write", "Edit", "Read", "NotebookEdit"):
        verb = {"Write": "写入", "Edit": "修改", "Read": "读取"}.get(name, name)
        return f"{verb} {short(args.get('file_path'))}"
    if name == "Bash":
        cmd = str(args.get("command") or "").strip().replace("\n", " ")
        return f"执行 {cmd[:110]}"
    if name in ("Glob", "Grep"):
        return f"{name} {str(args.get('pattern') or '')[:60]}"
    return f"{name}"


def run(
    cfg: Config,
    prompt: str,
    on_event: Optional[EventFn] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Path] = None,
) -> AgentResult:
    """跑一次 headless claude,把过程流式汇报给 on_event。"""
    binary = claude_path(cfg)
    if binary is None:
        return AgentResult(
            ok=False,
            error=f"找不到 claude CLI({cfg.claude_bin})。"
                  f"出题与讲评需要它;可用 LEETSTUDY_CLAUDE_BIN 指定路径。",
        )

    cmd: List[str] = [
        binary,
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "acceptEdits",
        "--add-dir", str(cfg.root),
    ]
    if cfg.claude_model:
        cmd += ["--model", cfg.claude_model]
    tools = cfg.claude_allowed_tools or DEFAULT_ALLOWED_TOOLS
    if tools:
        # --allowedTools 是变参选项(<tools...>),拆成多个 argv 会一路吞掉后面的
        # 参数。这里按官方示例压成单个逗号分隔的字符串。
        cmd += ["--allowedTools", ",".join(tools)]
    cmd.extend(cfg.claude_extra_args)

    # prompt 走 stdin,而不是 argv。两个理由:
    #   1. 任何变参选项都不可能"吃掉"它 —— 位置无关,不会因为将来加选项而回归
    #   2. 出题的提示词有十几 KB,不受 ARG_MAX 限制
    # 注意:claude -p 无位置参数时会读 stdin,这比 "--" 之类的分隔符可靠。
    prompt_via_stdin = True

    env = dict(__import__("os").environ)
    # 把项目 venv 的 bin 放进 PATH,这样出题者可以直接跑 `leet validate`
    # 而不必去猜 `.venv/bin/leet` 这个路径。
    venv_bin = cfg.root / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env.setdefault("VIRTUAL_ENV", str(cfg.root / ".venv"))
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if prompt_via_stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd or cfg.root),
            env=env,
        )
    except OSError as exc:
        return AgentResult(ok=False, error=f"无法启动 claude:{exc}")

    if prompt_via_stdin and proc.stdin is not None:
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            return AgentResult(ok=False, error=f"写入提示词失败:{exc}")

    log: List[str] = []
    state = {"final": "", "error": ""}

    def emit(msg: str) -> None:
        log.append(msg)
        if on_event:
            on_event(msg)

    def reader() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "assistant":
                for blk in (ev.get("message") or {}).get("content") or []:
                    btype = blk.get("type")
                    if btype == "text":
                        text = (blk.get("text") or "").strip()
                        if text:
                            first = text.splitlines()[0][:120]
                            emit(f"思考 {first}")
                    elif btype == "tool_use":
                        emit(_describe_tool(blk))
            elif kind == "result":
                state["final"] = ev.get("result") or ""
                if ev.get("is_error"):
                    state["error"] = str(ev.get("result") or "未知错误")

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    limit = timeout if timeout is not None else cfg.author_timeout
    thread.join(limit)

    if thread.is_alive():
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return AgentResult(
            ok=False, log=log, seconds=time.time() - t0,
            error=f"claude 执行超时(>{limit}s)。可用 LEETSTUDY_AUTHOR_TIMEOUT 调大。",
        )

    # 收尾不要用 proc.communicate() —— stdin 已经手动关闭,communicate 会尝试
    # flush 它并抛 ValueError。stderr 由单独的线程排空,再用 wait() 收尸。
    err_chunks: List[str] = []

    def drain_stderr() -> None:
        assert proc.stderr is not None
        try:
            for chunk in proc.stderr:
                err_chunks.append(chunk)
        except (ValueError, OSError):
            pass

    err_thread = threading.Thread(target=drain_stderr, daemon=True)
    err_thread.start()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    err_thread.join(5)

    stderr = "".join(err_chunks)
    rc = proc.returncode or 0
    if rc != 0 and not state["error"]:
        state["error"] = (stderr or "").strip()[:800] or f"claude 退出码 {rc}"

    return AgentResult(
        ok=rc == 0 and not state["error"],
        text=state["final"],
        error=state["error"],
        returncode=rc,
        seconds=time.time() - t0,
        log=log,
    )

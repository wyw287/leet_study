#!/usr/bin/env python3
"""leet_study 的本地 web 界面 —— **独立工具,不 import 框架代码**。

设计立场
--------
这个工具**以用户身份**驱动 `leet` CLI,而不是 import `leetstudy`。原因是解耦:
框架内部 API(`judge.Verdict`、`subjects.Subject` …)可以随便改,只要 `leet`
命令还能用,这个界面就不用跟着动。

通常「以用户身份调 CLI」的脆弱点是**解析给人看的终端输出**。这个项目不存在
那个问题 —— web 需要的数据全都有文件级的结构化通道:

    problems/*/spec.yaml           题目定义(YAML,项目自称"唯一驱动代码生成的文件")
    problems/*/problem.md          题面
    progress.json                  完成状态与最佳成绩
    build/*/last_verdict.json      判题结果(逐用例误差/guards/perf/评级 + 中文 hints)
    solutions/*/solution.*         学习者的代码

所以:**数据一律读文件,"动作"才走 CLI**。唯一的一处例外见 `_read_spec`。

长任务
------
`leet test` 要 11~21 秒(CUDA 上下文初始化就占 4.4 秒),`leet bench` 更久。
所以不能同步处理 HTTP 请求 —— 提交进**单工作线程的队列**,界面轮询进度。

单工作线程不是偷懒:框架的 `build/<题号>/`、`progress.json` 都是单份的,
同一道题并发跑两次会互相覆盖编译产物。串行执行把这个问题从根上消掉。

安全
----
这个工具的本质是**跑任意学习者代码**(nvcc/g++ 编译并执行解答),那就是
RCE by design。所以默认只绑 127.0.0.1,不要暴露到网络。

用法
----
作为框架的子命令(常用):

    leet web                    # http://127.0.0.1:8765
    leet web --port 9000 --open

也可以单独跑(调试本模块时方便,不需要装 leet 命令):

    python3 src/leetstudy/webui/server.py

**这个模块不 import 框架的其它部分** —— 只读文件 + 调 `leet` CLI。
仓库根与 `leet` 可执行文件的位置都由调用方传进来(见 `serve`)。
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("缺 PyYAML。用项目的 venv 跑:`.venv/bin/python -m leetstudy.webui.server`")

HERE = Path(__file__).resolve().parent          # 本模块所在目录(模板也在这)

# ---------------------------------------------------------------------------
# 运行位置(仓库根 / leet 可执行文件)
#
# 由 `serve()` 在启动时设一次,之后只读。放模块级而不是层层传参,是因为
# **一个进程只有一个 server**,而且这些值在第一个请求到达之前就定下来了 ——
# 为它把参数穿透十几层调用不划算。`_require_root()` 保证没人能读到 None。
# ---------------------------------------------------------------------------
ROOT: Optional[Path] = None
LEET: Optional[Path] = None
PROBLEMS_DIR: Optional[Path] = None
SOLUTIONS_DIR: Optional[Path] = None
BUILD_DIR: Optional[Path] = None
PROGRESS: Optional[Path] = None

#: 允许跨域访问的来源白名单。**默认为空 = 不开 CORS**(和以前一样)。
#:
#: 为什么是白名单而不是 `*`:这个 API 能改 `solutions/` 下的文件、能触发 `leet test`
#: 去**编译并运行**那些文件 —— 开了通配,你浏览器里访问的任何一个网页都可以先
#: 覆盖你的 solution.cu、再 POST 一个 test 把它跑起来。那就是一条 drive-by RCE。
#: CORS 正是挡住这件事的那道墙,拆墙之前得知道墙后面是什么。
#:
#: 所以:要用就明确写出前端的来源(`http://localhost:3000` 之类)。
_ALLOW_ORIGINS: List[str] = []


def _require_root() -> Path:
    if ROOT is None:
        raise RuntimeError("webui 还没初始化:请通过 `leet web` 或 serve() 启动")
    return ROOT


def _bind_root(root: Path, leet_bin: Path) -> None:
    global ROOT, LEET, PROBLEMS_DIR, SOLUTIONS_DIR, BUILD_DIR, PROGRESS
    ROOT = Path(root).resolve()
    LEET = Path(leet_bin).resolve()
    PROBLEMS_DIR = ROOT / "problems"
    SOLUTIONS_DIR = ROOT / "solutions"
    BUILD_DIR = ROOT / "build"
    PROGRESS = ROOT / "progress.json"


def find_root(start: Optional[Path] = None) -> Optional[Path]:
    """从 start 往上找带 `problems/` 的目录。供独立运行时兜底(不走框架)。"""
    d = (start or Path.cwd()).resolve()
    for cand in [d, *d.parents]:
        if (cand / "problems").is_dir():
            return cand
    return None


def find_leet(root: Path) -> Optional[Path]:
    """找 `leet` 可执行文件:先 PATH,再项目 venv。供独立运行时兜底。"""
    found = shutil.which("leet")
    if found:
        return Path(found)
    for cand in (root / ".venv" / "bin" / "leet", root / ".venv/bin/leet.exe"):
        if cand.is_file():
            return cand
    return None


#: 单个任务保留的输出上限(防止 `leet new` 那种长任务把内存吃光)
MAX_OUTPUT_CHARS = 400_000

#: 允许通过界面触发的动作 → CLI 子命令。
#: **白名单**,界面传什么都不能越出这几个。
ACTIONS = {
    "test": ["test"],
    "bench": ["bench"],
    "start": ["start"],
    "solution": ["solution"],
}


# --------------------------------------------------------------------------- #
# 数据读取(全部走文件,不解析终端输出)
# --------------------------------------------------------------------------- #

def _read_spec(problem_id: str) -> Optional[Dict[str, Any]]:
    """读一道题的 spec.yaml。

    这是本工具**唯一**直接依赖框架数据格式的地方 —— 但它依赖的是 spec.yaml,
    而不是 Python API。spec.yaml 是出题文档里公开的契约(「机器可读定义」),
    比 import 内部类稳定得多。
    """
    path = PROBLEMS_DIR / problem_id / "spec.yaml"
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None


def _safe_id(problem_id: str) -> Optional[str]:
    """把 id 解析成题库里真实存在的目录名(挡掉路径穿越)。"""
    if not problem_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", problem_id):
        return None
    return problem_id if (PROBLEMS_DIR / problem_id).is_dir() else None


def _solution_path(problem_id: str) -> Optional[Path]:
    """找到解答文件。用 glob 而不是查表 —— 这样不必知道各科目的文件名约定。"""
    d = SOLUTIONS_DIR / problem_id
    if not d.is_dir():
        return None
    for p in sorted(d.iterdir()):
        if p.is_file() and p.name.startswith("solution"):
            return p
    return None


def _read_progress() -> Dict[str, Any]:
    if not PROGRESS.is_file():
        return {}
    try:
        return (json.loads(PROGRESS.read_text(encoding="utf-8")) or {}).get("problems") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_verdict(problem_id: str) -> Optional[Dict[str, Any]]:
    """读上次判题结果。

    `leet test` 每次都会刷新它;里面已经有评级、加速比、逐用例误差与 guards,
    还有框架生成好的中文 hints —— 界面直接用,不重算、不解析。
    """
    path = BUILD_DIR / problem_id / "last_verdict.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _list_problems() -> List[Dict[str, Any]]:
    progress = _read_progress()
    out: List[Dict[str, Any]] = []
    if not PROBLEMS_DIR.is_dir():
        return out
    for d in sorted(PROBLEMS_DIR.iterdir()):
        if not d.is_dir() or not (d / "spec.yaml").is_file():
            continue
        spec = _read_spec(d.name) or {}
        pr = progress.get(d.name) or {}
        out.append({
            "id": d.name,
            "title": spec.get("title") or d.name,
            "subject": spec.get("subject") or "cuda",
            "difficulty": spec.get("difficulty"),
            "tags": spec.get("tags") or [],
            "concepts": spec.get("concepts") or [],
            "cases": len(spec.get("cases") or []),
            "metric": (spec.get("perf") or {}).get("metric"),
            "required_grade": (spec.get("perf") or {}).get("required_grade"),
            "solved": bool(pr.get("solved")),
            "best_grade": pr.get("best_grade"),
            "best_metric": pr.get("best_metric"),
            "metric_name": pr.get("metric_name"),
            "attempts": pr.get("attempts"),
            "started": _solution_path(d.name) is not None,
        })
    return out


# --------------------------------------------------------------------------- #
# 任务队列
# --------------------------------------------------------------------------- #

class Job:
    __slots__ = ("id", "problem_id", "action", "argv", "status", "output",
                 "returncode", "created", "started", "finished", "error")

    def __init__(self, problem_id: str, action: str, argv: List[str]):
        self.id = uuid.uuid4().hex[:12]
        self.problem_id = problem_id
        self.action = action
        self.argv = argv
        self.status = "queued"           # queued | running | done | failed
        self.output = ""
        self.returncode: Optional[int] = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.error: Optional[str] = None

    def append(self, text: str) -> None:
        self.output += text
        if len(self.output) > MAX_OUTPUT_CHARS:      # 只留尾巴,长任务不涨内存
            self.output = self.output[-MAX_OUTPUT_CHARS:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "problem_id": self.problem_id, "action": self.action,
            "status": self.status, "output": self.output,
            "returncode": self.returncode, "error": self.error,
            "seconds": round((self.finished or time.time())
                             - (self.started or self.created), 2),
        }


class JobQueue:
    """单工作线程的串行队列 —— 见模块 docstring 里"为什么串行"。"""

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._current: Optional[Job] = None
        t = threading.Thread(target=self._run, daemon=True, name="leet-worker")
        t.start()

    def submit(self, problem_id: str, action: str) -> Job:
        argv = list(ACTIONS[action]) + [problem_id]
        job = Job(problem_id, action, argv)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            running = self._current.id if self._current else None
            pending = self._q.qsize()
        return {"running": running, "pending": pending}

    def _run(self) -> None:
        while True:
            job = self._q.get()
            with self._lock:
                self._current = job
            job.status = "running"
            job.started = time.time()
            try:
                self._exec(job)
            except Exception as exc:                       # noqa: BLE001
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
                if job.status == "running":
                    job.status = "done" if job.returncode == 0 else "failed"
                with self._lock:
                    self._current = None

    def _exec(self, job: Job) -> None:
        env = dict(os.environ)
        # CLI 要靠 PATH 找自己(nvcc / compute-sanitizer / ninja 等)
        env["PATH"] = f"{LEET.parent}:{env.get('PATH', '')}"
        proc = subprocess.Popen(
            [str(LEET)] + job.argv,
            cwd=str(self.cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:                 # 逐行读,界面能看到实时进度
            job.append(line)
        job.returncode = proc.wait()


JOBS: Optional[JobQueue] = None


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class _Server(ThreadingHTTPServer):
    """服务器本体。两个属性都**必须在这里**设,不能构造完再赋值:

    · `request_queue_size` 只在 `server_bind()` 里被读一次去调 `listen()`,
      所以构造之后再改已经晚了(`socketserver` 的默认值是 5)。
    · 两者都是**服务器类**的属性 —— 写在 Handler 上完全不生效。这个坑我踩过:
      改完一看 `ss -ltn` 的 backlog 还是 5。

    backlog 5 对于"浏览器一次并发开 4~6 条连接 + 外面还套一层转发"是偏小的,
    表现是连接被内核丢掉、客户端一直转圈、而**服务端日志里什么都没有**。
    """

    request_queue_size = 128
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "leet-webui/1.0"

    # ---- 工具 ---- #
    def _cors_origin(self) -> Optional[str]:
        """这个请求的来源允许吗?允许就返回该回的头值,否则 None。

        只有**请求带了 Origin 且它在白名单里**才回 CORS 头 —— 同源请求和
        命令行工具(curl)本来就不需要,回了反而是把口子开得比必要的大。
        """
        if not _ALLOW_ORIGINS:
            return None
        origin = self.headers.get("Origin")
        if not origin:
            return None
        if "*" in _ALLOW_ORIGINS:
            return "*"
        return origin if origin in _ALLOW_ORIGINS else None

    def _cors_headers(self) -> List[tuple]:
        o = self._cors_origin()
        if o is None:
            return []
        # Vary: Origin —— 同一个 URL 对不同来源会给出不同的头,
        # 不声明的话中间缓存可能把给 A 站的响应喂给 B 站。
        return [("Access-Control-Allow-Origin", o), ("Vary", "Origin")]

    def _send_json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in self._cors_headers():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, code: int = 200,
                   ctype: str = "text/plain; charset=utf-8",
                   no_store: bool = False) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        for k, v in self._cors_headers():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # ★ 默认是 HTTP/1.0,每响应一个请求就关连接。浏览器打开一个页面会并发发
    # 4~6 条请求,于是每次都得重新握手 —— 在共享服务器 + SSH 转发这种链路上
    # 很容易撞上 accept 队列。改 1.1 后连接会被复用。
    #
    # 开 1.1 的前提是**每条响应都必须带准确的 Content-Length**(否则客户端会
    # 一直等连接关,表现就是"转圈、空白")。下面三个发送路径都带了,加新的
    # 响应路径时别忘了。
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:      # 安静一点
        if not os.environ.get("LEET_WEBUI_VERBOSE"):
            return
        # 带上 User-Agent —— 默认格式只有 IP,而这是一台**共享服务器**,
        # 光看 127.0.0.1 分不出请求是你自己发的还是别人的进程。
        # (实测踩过:服务端日志里一直有 /api/problems 在轮询,查了半天才发现
        #  来源是个已经跑了两周的、属于另一个用户的 Firefox。)
        super().log_message(fmt + '  UA="%s"', *args,
                            self.headers.get("User-Agent", "-"))

    # ---- 路由 ---- #
    def do_OPTIONS(self) -> None:                               # noqa: N802
        """预检。

        PUT + `Content-Type: application/json` 和 POST 都会先发一个 OPTIONS,
        浏览器拿到不允许的答复就不发真正的请求。所以这里是跨域的实际闸门 ——
        之前没有这个处理,一律 501,等于天然禁止跨域。
        """
        o = self._cors_origin()
        if o is None:
            return self._send_json({"error": "跨域未获允许"}, 403)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", o)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")   # 600 秒内不用重复预检
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:                                  # noqa: N802
        path = self.path.split("?", 1)[0]

        # ★ no-store:这个服务**一个缓存头都不发**的话,浏览器会启发式缓存,
        # 而这两样东西每次改代码都会变(index.html 和 392 KB 的编辑器产物)。
        # 实测踩过:换了编辑器包之后浏览器还在用旧的,新页面调一个旧包里没有的
        # 导出 —— 页面挂掉,而服务端日志里一切正常,极难归因。
        # 本地端口上重新下载 392 KB 是毫秒级,不值得为它冒这个险。
        if path in ("/", "/index.html"):
            html = (HERE / "index.html").read_text(encoding="utf-8")
            return self._send_text(html, ctype="text/html; charset=utf-8",
                                   no_store=True)

        # 前端编辑器(CodeMirror 6)的打包产物。**提交进仓库**,不是运行时生成的 ——
        # 重装/升级的办法见 vendor/BUILD.md。缺了它页面还能用,只是编辑区退化成
        # 纯文本(前端的 import 失败会走那条路)。
        if path == "/vendor/codemirror.js":
            f = HERE / "vendor" / "codemirror.js"
            if not f.is_file():
                return self._send_json(
                    {"error": "前端编辑器产物缺失 —— 见 webui/vendor/BUILD.md"}, 404)
            return self._send_text(f.read_text(encoding="utf-8"),
                                   ctype="text/javascript; charset=utf-8",
                                   no_store=True)

        if path == "/api/problems":
            return self._send_json({"problems": _list_problems(),
                                    "queue": JOBS.snapshot() if JOBS else {}})

        if path == "/api/highlight.css":
            return self._send_text(_highlight_css(), ctype="text/css; charset=utf-8",
                                   no_store=True)

        m = re.fullmatch(r"/api/problems/([^/]+)", path)
        if m:
            pid = _safe_id(m.group(1))
            if not pid:
                return self._send_json({"error": "没有这道题"}, 404)
            spec = _read_spec(pid) or {}
            stmt = PROBLEMS_DIR / pid / (spec.get("statement") or "problem.md")
            sol = _solution_path(pid)
            opt = PROBLEMS_DIR / pid / _optimal_name(pid)
            opt_src = opt.read_text(encoding="utf-8") if opt.is_file() else ""
            sol_src = sol.read_text(encoding="utf-8") if sol else ""
            stmt_src = stmt.read_text(encoding="utf-8") if stmt.is_file() else ""
            return self._send_json({
                "id": pid, "spec": spec,
                "statement": stmt_src,
                # 题面的 HTML(服务端用 markdown-it-py 渲染,代码块交给 Pygments)。
                # 拿不到 markdown-it-py 时是 null,前端退回纯文本渲染。
                "statement_html": _render_markdown(stmt_src),
                "solution_path": str(sol.relative_to(ROOT)) if sol else None,
                "solution": sol_src,
                "optimal": opt_src,
                # 高亮后的 HTML(服务端渲染)。拿不到 pygments 时是 null,
                # 前端退回纯文本 —— 见 _highlight 的说明。
                "optimal_html": _highlight(opt_src, _lang_of(opt) if opt.is_file() else None),
                "solution_lang": _lang_of(sol),
                "verdict": _read_verdict(pid),
            })

        if path == "/api/queue":
            return self._send_json(JOBS.snapshot() if JOBS else {})

        m = re.fullmatch(r"/api/jobs/([^/]+)", path)
        if m:
            job = JOBS.get(m.group(1)) if JOBS else None
            if not job:
                return self._send_json({"error": "没有这个任务"}, 404)
            return self._send_json(job.to_dict())

        return self._send_json({"error": "没有这个路径"}, 404)

    def do_PUT(self) -> None:                                   # noqa: N802
        m = re.fullmatch(r"/api/problems/([^/]+)/solution", self.path.split("?", 1)[0])
        if not m:
            return self._send_json({"error": "没有这个路径"}, 404)
        pid = _safe_id(m.group(1))
        if not pid:
            return self._send_json({"error": "没有这道题"}, 404)

        payload = self._body_json()
        src = payload.get("source")
        if not isinstance(src, str):
            return self._send_json({"error": "缺 source 字段"}, 400)

        path = _solution_path(pid)
        if path is None:
            return self._send_json(
                {"error": "这道题还没有工作区,先点「开始做题」(leet start)"}, 409)
        try:
            path.write_text(src, encoding="utf-8")
        except OSError as exc:
            return self._send_json({"error": f"写不进去:{exc}"}, 500)
        return self._send_json({"ok": True, "path": str(path.relative_to(ROOT))})

    def do_POST(self) -> None:                                  # noqa: N802
        path = self.path.split("?", 1)[0]

        # 这里原本有一个 POST /api/highlight(前端防抖后把源码发过来换高亮 HTML)。
        # 编辑区换成 CodeMirror 之后**没有任何调用方了** —— 上色改在浏览器里同步做,
        # 所以删掉。服务端的 _highlight() 本身还在用(参考解页、题面代码块)。
        #
        # 当初之所以让编辑器走服务端高亮,是为了和参考解页共用同一套着色规则。
        # 现在这个目标由**配色变量**达成:server.py 的 _theme_vars() 把 Pygments
        # 自己的 CSS 反解成 --py-* 变量,编辑器绑的就是它们 —— 共用的是定义,
        # 而不是每次渲染的结果,反而更不容易漂移。

        m = re.fullmatch(r"/api/problems/([^/]+)/(\w+)", path)
        if not m:
            return self._send_json({"error": "没有这个路径"}, 404)
        pid = _safe_id(m.group(1))
        action = m.group(2)
        if not pid:
            return self._send_json({"error": "没有这道题"}, 404)
        if action not in ACTIONS:
            return self._send_json(
                {"error": f"不允许的动作 {action};可用:{' / '.join(ACTIONS)}"}, 400)

        job = JOBS.submit(pid, action) if JOBS else None
        if job is None:
            return self._send_json({"error": "任务队列没起来"}, 500)
        return self._send_json({"job": job.to_dict()}, 202)


def _optimal_name(problem_id: str) -> str:
    """参考解的文件名。

    按科目推断(框架里各科目自己声明 optimal_filename)。这里做一份最小映射,
    而不是 import 框架 —— 代价是新科目要在这里补一行,收益是 web 工具完全不依赖
    框架的 Python API。
    """
    spec = _read_spec(problem_id) or {}
    return {"cuda": "optimal.cu", "cpp": "optimal.cpp",
            "pytorch": "optimal.py"}.get(spec.get("subject") or "cuda", "optimal.cu")


# --------------------------------------------------------------------------- #
# 语法高亮
#
# 用 **Pygments**,而且刻意用它而不是前端库:
#
# * 它**已经在 venv 里** —— `rich` 依赖它,所以零新增依赖。
# * 它**能区分语言**,而这不是可有可无的:`cuda` / `cpp` / `python` 三个 lexer
#   正好覆盖本项目三种解答(`.cu` / `.cpp` / `.py`)。CUDA 的 `__global__`、
#   `<<< >>>` 这些只有 cuda lexer 认得出来。
# * 服务端渲染意味着**不依赖 CDN** —— 这台机器访问国际站点要挂代理,
#   网页里引 CDN 会让界面在离线/没代理时直接废掉。
# * 够快:实测 47 行的解答 1.8ms、103 行的参考解 4.1ms,所以连"边打字边高亮"
#   都撑得住(见 index.html 里的输入防抖)。
#
# Pygments 是 pygments 的传递依赖,不是我们直接声明的。真掉了要能退化,
# 所以 `_highlight` 在拿不到它时返回 None,调用方退回纯文本。
# --------------------------------------------------------------------------- #

#: 按扩展名选 lexer。和文件内容无关 —— 扩展名在这里就是权威(框架自己也是这么分的)。
_LEXER_BY_EXT = {
    ".cu": "cuda", ".cuh": "cuda",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
    ".py": "python",
}


def _lang_of(path: Optional[Path]) -> Optional[str]:
    return _LEXER_BY_EXT.get(path.suffix.lower()) if path else None


def _highlight(src: str, lang: Optional[str]) -> Optional[str]:
    """把源码渲染成带 `<span class="...">` 的高亮 HTML(nowrap,由外层套 <pre>)。

    拿不到 pygments 或语言不认识时返回 None —— 调用方原样走纯文本路径。
    """
    if not src or not lang:
        return None
    try:
        from pygments import highlight as _hl
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name
    except ImportError:
        return None
    try:
        return _hl(src, get_lexer_by_name(lang), HtmlFormatter(nowrap=True))
    except Exception:                                   # noqa: BLE001
        return None


def _highlight_css() -> str:
    """Pygments 的配色,按深浅色模式给两套。

    只出 token 的颜色规则(nowrap),**背景留给页面自己的 CSS** ——
    这样代码块的底色跟其余界面一致,不会突然出现一块 Pygments 风格的白/黑底。

    末尾会附上 `--py-*` 变量(见 `_theme_vars`),那是给前端编辑器用的:
    编辑器的配色全绑在这些变量上,于是**编辑器、题面、参考解页共用一份定义**。
    """
    try:
        from pygments.formatters import HtmlFormatter
    except ImportError:
        return ""
    light = HtmlFormatter(style="default").get_style_defs(".hl")
    # 深色用 dracula 而不是 monokai —— 实测 monokai 的 k(关键字)和 kt(类型)
    # 是**同一个颜色** #66D9EF,于是 `__global__` 和 `void` 长得一模一样。
    # 这两个恰恰是 CUDA 里最常并排出现的两类 token。
    # 备选里一个都不完美(见 docs/webui.md 的对照表),dracula 是撞得最轻的:
    # 它让 k 与 kt 分开了(粉 / 青),代价是 kt 与 nb 同色(青),
    # 而 nb(threadIdx 这类)出现得远比 kt 少。
    dark = HtmlFormatter(style="dracula").get_style_defs(".hl")
    return (light
            + "\n:root{" + _theme_vars(light) + "}"
            + "\n@media (prefers-color-scheme: dark) {\n"
            + dark
            + _override_rules(".hl", _DARK_OVERRIDES)
            + "\n:root{" + _theme_vars(dark, _DARK_OVERRIDES) + "}\n}")


def _override_rules(scope: str, overrides: Dict[str, str]) -> str:
    """把改开的颜色**也**写进 Pygments 那套类名里。

    只导 --py-* 是不够的:「参考解」页和题面里的代码块是**服务端渲染好的
    HTML**,用的是 Pygments 原样输出的 `.hl .nb` 这类类名,它们不读变量。
    不补这几条,编辑器里 threadIdx 是绿的面参考解页还是青的 ——
    那正是这套设计要避免的「三处不一致」。

    放在 `dark` 之后(同级选择器、后来者胜),所以能盖掉 Pygments 的原值。
    """
    by_var = dict(_CSS_VARS)
    return "".join("\n%s .%s{color:%s}" % (scope, by_var[v], c)
                   for v, c in overrides.items())


# --------------------------------------------------------------------------- #
# 编辑器配色变量
#
# 前端的高亮不再走服务端往返(编辑区换成 CodeMirror,在浏览器里同步算),
# 但配色必须和 Pygments 那套**完全一致** —— 否则同一个词在题面、参考解、
# 编辑器三处会是三个颜色。
#
# 做法是把上面生成的 CSS **反过来解析**一遍,抠出 --py-* 变量。不手抄颜色:
# 换 Pygments 版本或主题时,编辑器自动跟着变,不会漂移。
# (实测 340 字符的样本逐字符对照,只剩 1 处差异 —— 预处理指令里的文件名,
#  Pygments 单独染成注释色,而编辑器把整行 `#include "x.h"` 当一个 token。)
# --------------------------------------------------------------------------- #

# 变量名 → Pygments 的 token 简写。名字取短的,写 HighlightStyle 时好认。
_CSS_VARS = (
    ("k",   "k"),     # Keyword           关键字
    ("kt",  "kt"),    # Keyword.Type      类型
    ("c",   "c"),     # Comment           注释
    ("s",   "s"),     # String            字符串
    ("num", "m"),     # Number            数字
    ("o",   "o"),     # Operator          运算符
    ("nb",  "nb"),    # Name.Builtin      内建(threadIdx 这类)
    ("nf",  "nf"),    # Name.Function     函数名
    ("cp",  "cp"),    # Comment.Preproc   预处理指令
    ("kc",  "kc"),    # Keyword.Constant  true / false / NULL
)


def _css_rule(css: str, cls: str) -> str:
    """抠出 `.hl .<cls> { … }` 的声明体,没有就是空串。"""
    m = re.search(r"^\.hl \.%s \{([^}]*)\}" % re.escape(cls), css, re.M)
    return m.group(1) if m else ""


def _css_color(css: str, cls: str) -> str:
    m = re.search(r"color:\s*([^;]+)", _css_rule(css, cls))
    return m.group(1).strip() if m else "inherit"


def _css_base(css: str) -> str:
    """Pygments 在 `.hl` 上设的**基准正文色**。

    default 主题不设(于是继承页面色),dracula 会设成 #f8f8f2。编辑器如果不
    跟着设,暗色下整屏代码会比参考解页暗一档(230 vs 248),看着像失焦。
    """
    m = re.search(r"^\.hl \{([^}]*)\}", css, re.M)
    if m:
        c = re.search(r"color:\s*([^;]+)", m.group(1))
        if c:
            return c.group(1).strip()
    return "inherit"


def _theme_vars(css: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """从一份 Pygments CSS 里导出编辑器要用的全部 --py-* 变量。

    `overrides` 用来改开主题自身的撞色,键是 `_CSS_VARS` 里的短名。
    这不破坏「颜色只定义一次」——覆盖发生在这里,而 --py-* 正是编辑器、题面、
    参考解页**共用**的那一份,所以三处依然一致。它破的只是「必须原样照抄
    Pygments 主题」,而那本身不是目的。
    """
    overrides = overrides or {}
    out = ["--py-base:%s;" % _css_base(css)]
    for var, cls in _CSS_VARS:
        rule = _css_rule(css, cls)
        color = overrides.get(var) or _css_color(css, cls)
        out.append("--py-%s:%s;" % (var, color))
        out.append("--py-%s-w:%s;" % (var, "bold" if "font-weight: bold" in rule else "normal"))
        out.append("--py-%s-s:%s;" % (var, "italic" if "font-style: italic" in rule else "normal"))
    return "".join(out)


# 深色主题(dracula)自身有三处不够用,这里按 dracula 自己调色板里没用到的
# 颜色改开(注释那条是提亮同色相):
#
#   kt(类型)与 nb(内建)都是青  → nb 改用 dracula 的绿 #50FA7B
#      不分开的话 `int i = blockIdx.x * blockDim.x + threadIdx.x;` 整行同色
#   k(关键字)与 o(运算符)都是粉 → o 改用 dracula 的紫 #BD93F9
#      operator 在题库语料里出现 1042 次,仅次于变量名
#   c(注释)对 --code-bg 只有 3.9:1 → 提亮成 #828FC2,到 5.8:1
#      **这条是本项目特有的**:题面和参考解的内容基本都在注释里,
#      注释读着累等于整个工具读着累。dracula 原色是给代码配的,不是给讲义配的。
#
# 扫过 7 个现成的深色主题(monokai / native / one-dark / dracula / material /
# gruvbox-dark / solarized-dark),**没有一个四类 token 两两可辨** —— 见
# docs/webui.md 的对照表。所以与其换来换去,不如就用最接近的再改几个值。
#
# ⚠️ 这几个色值是照 dracula 的调色板挑的,换主题时要一起换。
_DARK_OVERRIDES = {"nb": "#50FA7B", "o": "#BD93F9", "c": "#828FC2"}


# --------------------------------------------------------------------------- #
# 题面渲染
#
# 用 **markdown-it-py** —— 和 Pygments 一样,它**已经在 venv 里**
# (`rich` 的依赖),所以零新增依赖、不碰 CDN。选它而不是继续用前端那版手写的
# 正则渲染器,理由:
#
# * 手写版只覆盖了「题面看起来会用到」的子集,边界情况(列表嵌套、表格对齐、
#   转义)全靠猜。CommonMark 的规则比看上去多得多。
# * 它能把围栏代码块**交给 Pygments**,于是题面里的 ```cuda / ```python
#   和编辑器、参考解页共用同一套配色,不会两处不一样。
#
# ⚠️ 声明式地说明这里的一个前提:**题面里不能有原始 HTML**。preset 用
#   `commonmark` 且不开 `html` 选项,所以 `<div>` 之类会被转义成文本 ——
#   这对本项目的题库是对的(题面由出题规范约束,不该嵌 HTML)。
# --------------------------------------------------------------------------- #

#: MarkdownIt 实例。**模块级建一次** —— 它是有状态的解析器,每次重建很浪费。
#: 建失败(没装 markdown-it-py)时留 None,调用方退回纯文本。
_MD = None
_MD_FAILED = False


def _md_instance():
    global _MD, _MD_FAILED
    if _MD is None and not _MD_FAILED:
        try:
            from markdown_it import MarkdownIt
            _MD = (MarkdownIt("commonmark", {"highlight": _md_highlight})
                   .enable("table").enable("strikethrough"))
        except ImportError:
            _MD_FAILED = True
    return _MD


def _md_highlight(code: str, lang: str, attrs: str = "") -> str:
    """给围栏代码块上色。**返回空串 = 不进高亮**,交给 markdown-it 自己转义。

    没标语言的围栏(题面里有 90 多处,多是公式和输出样例)走的正是这条路 ——
    它们不需要语法着色,但仍然是一个等宽代码块。
    """
    lang = (lang or "").strip().split()[0] if lang else ""
    if not lang:
        return ""
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name
        return highlight(code, get_lexer_by_name(lang), HtmlFormatter(nowrap=True))
    except Exception:                                   # noqa: BLE001
        return ""                                       # 语言不认识 -> 普通代码块


def _render_markdown(text: str) -> Optional[str]:
    """题面 -> HTML。拿不到 markdown-it-py 时返回 None(前端退回纯文本)。"""
    md = _md_instance()
    if md is None or not text:
        return None
    try:
        html = md.render(text)
    except Exception:                                   # noqa: BLE001
        return None
    # 代码块复用编辑器那套 Pygments 配色(样式规则是按 .hl 定义的);
    # markdown-it 输出的就是裸 <pre><code>,这里加个类名即可。
    return html.replace("<pre>", '<pre class="hl">')


def serve(root: Path, leet_bin: Path, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = False, allow_origins: Optional[List[str]] = None) -> None:
    """起服务,阻塞直到 Ctrl-C。

    仓库根与 `leet` 可执行文件由调用方给(`leet web` 从框架拿,独立运行时自己找)——
    这样本模块不必知道框架怎么定位仓库。
    """
    global JOBS, _ALLOW_ORIGINS
    _bind_root(root, leet_bin)
    _ALLOW_ORIGINS = [o.strip() for o in (allow_origins or []) if o.strip()]

    if not PROBLEMS_DIR.is_dir():
        sys.exit(f"找不到题库目录 {PROBLEMS_DIR}")
    if not LEET.is_file():
        sys.exit(f"找不到 {LEET}\n先建好 venv:python3 -m venv .venv && .venv/bin/pip install -e .")

    JOBS = JobQueue(ROOT)
    url = f"http://{host}:{port}/"
    print(f"leet web  →  {url}")
    print(f"  仓库:{ROOT}")
    print(f"  CLI :{LEET}")
    if _ALLOW_ORIGINS:
        # 开了就一定要说清楚:这个 API 能改文件、还能编译并运行那些文件。
        print(f"  跨域:{', '.join(_ALLOW_ORIGINS)}")
        print("  [注意] 被允许的来源可以读写你的解答、并触发编译运行 ——"
              "\n         等于把本机代码执行权交给了那些页面。"
              "确认那是你自己的前端再继续。")
    print("  Ctrl-C 退出")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    srv = _Server((host, port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n再见")


def main() -> None:
    """独立运行时的入口(不走框架)。

    直接 `python3 src/leetstudy/webui/server.py` 时,仓库根和 leet 都得自己找 ——
    平时不用这条路,调试本模块时方便。
    """
    ap = argparse.ArgumentParser(description="leet_study 的本地 web 界面")
    ap.add_argument("--host", default="127.0.0.1",
                    help="绑定地址。默认只绑本机 —— 这个工具会执行任意解答代码")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    ap.add_argument("--allow-origin", action="append", default=[],
                    metavar="ORIGIN",
                    help="允许跨域访问的来源(可重复)。默认不开 CORS")
    args = ap.parse_args()

    root = find_root()
    if root is None:
        sys.exit("找不到仓库根(往上找不到 problems/ 目录)。"
                 "用 `leet web` 启动,或在一个题目仓库里运行。")
    leet = find_leet(root)
    if leet is None:
        sys.exit(f"在 PATH 和 {root/'.venv/bin/leet'} 里都没找到 `leet`。"
                 "先装好:python3 -m venv .venv && .venv/bin/pip install -e .")
    serve(root, leet, host=args.host, port=args.port, open_browser=args.open,
          allow_origins=args.allow_origin)


if __name__ == "__main__":
    main()

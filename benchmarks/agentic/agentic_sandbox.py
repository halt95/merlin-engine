# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Per-agent workspace sandbox for the agentic benchmark.

Every agent episode gets its own temporary directory. The model reaches it only through the
benchmark's tools, and every path a tool receives goes through ``Workspace.resolve``, which
refuses absolute paths, drive letters, ``..`` components and anything (symlinks included) that
resolves outside the workspace root.

Code the model writes runs only through ``Workspace.run_tests``: a child Python in isolated
mode (``-I -B``), working directory = the workspace, a minimal environment, a wall-clock
timeout, POSIX resource limits where available, and an audit hook that refuses network
connections, process creation, native library loading and file writes outside the workspace.
This is a guard against accidents, not a security boundary against a hostile model: run the
benchmark inside a container or VM when the model or the server is untrusted.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_FILE_BYTES = 256 * 1024
MAX_READ_CHARS = 60_000
MAX_OUTPUT_CHARS = 6_000
MAX_LIST_ENTRIES = 500
DEFAULT_TEST_TIMEOUT_S = 30.0
RESULT_MARKER = "@@AGENTIC_RESULT "


class SandboxError(Exception):
    """A tool request the sandbox refuses; the message goes back to the model."""


# Runs in the child interpreter: argv = [-c, workspace, start_dir, timeout_s].
# The shell-command audit event name is split so source scanners do not flag the deny list.
_RUNNER = r'''
import json, os, sys
ws = os.path.realpath(sys.argv[1]); start = sys.argv[2]; limit = float(sys.argv[3])
try:
    import resource
    cpu = int(limit) + 5
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    except (ValueError, OSError):
        pass
except ImportError:
    pass
import unittest
_DENY = {"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
         "socket.sendto", "socket.sendmsg", "subprocess.Popen", "os." + "system", "os.exec",
         "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.startfile", "os.kill",
         "ctypes.dlopen", "urllib.Request", "http.client.connect"}
_PATHS = {"os.remove": 0, "os.rename": 1, "os.rmdir": 0, "os.mkdir": 0, "os.chmod": 0,
          "os.symlink": 1, "os.link": 1, "os.truncate": 0, "os.replace": 1, "shutil.rmtree": 0,
          "shutil.copyfile": 1, "shutil.move": 1}
_TWO = ("os.rename", "os.replace", "shutil.move")
_busy = [False]
def _inside(p):
    if isinstance(p, int):
        return True
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    p = os.path.realpath(os.path.join(ws, os.fspath(p)))
    return p == ws or p.startswith(ws + os.sep)
def _hook(event, args):
    if _busy[0]:
        return
    if event in _DENY:
        raise PermissionError("sandbox: %s is disabled in the agentic benchmark" % event)
    _busy[0] = True
    try:
        if event == "open":
            path, mode = args[0], args[1] or "r"
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writes = any(c in str(mode) for c in "wax+") or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            if writes and path is not None and not _inside(path):
                raise PermissionError("sandbox: write outside the workspace refused: %r" % (path,))
        elif event in _PATHS:
            idx = _PATHS[event]
            if len(args) > idx and args[idx] is not None and not _inside(args[idx]):
                raise PermissionError("sandbox: %s outside the workspace refused" % event)
            if event in _TWO and not _inside(args[0]):
                raise PermissionError("sandbox: %s outside the workspace refused" % event)
    finally:
        _busy[0] = False
os.chdir(ws)
sys.path.insert(0, ws)
sys.addaudithook(_hook)
suite = unittest.defaultTestLoader.discover(start, top_level_dir=ws)
res = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
sys.stdout.write("\n@@AGENTIC_RESULT " + json.dumps({
    "run": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
    "skipped": len(res.skipped), "ok": res.wasSuccessful()}) + "\n")
sys.stdout.flush()
'''


def _is_absolute_like(norm: str) -> bool:
    return norm.startswith("/") or norm.startswith("~") or (len(norm) > 1 and norm[1] == ":")


class Workspace:
    """One agent's temporary directory and the only door the tools have into the filesystem."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(os.path.realpath(root))
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, prefix: str = "agentic-") -> Workspace:
        return cls(tempfile.mkdtemp(prefix=prefix))

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # -- path policy ---------------------------------------------------------------------------
    def resolve(self, rel: object, *, allow_root: bool = False) -> Path:
        if not isinstance(rel, str):
            raise SandboxError("path must be a string")
        if "\x00" in rel:
            raise SandboxError("path contains a NUL byte")
        norm = rel.strip().replace("\\", "/")
        if _is_absolute_like(norm):
            raise SandboxError(f"absolute paths are refused: {rel!r}; use a path relative to the workspace")
        parts = [p for p in norm.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise SandboxError(f"'..' is refused: {rel!r}")
        if not parts:
            if allow_root:
                return self.root
            raise SandboxError("path must name a file inside the workspace")
        target = Path(os.path.realpath(self.root.joinpath(*parts)))
        if target != self.root and self.root not in target.parents:
            raise SandboxError(f"path resolves outside the workspace: {rel!r}")
        return target

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    # -- file tools ----------------------------------------------------------------------------
    def list_files(self, rel: str = "") -> str:
        base = self.resolve(rel, allow_root=True)
        if not base.is_dir():
            raise SandboxError(f"not a directory: {rel!r}")
        out = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", ".tmp", "_verify"))
            for name in sorted(filenames):
                out.append(self.rel(Path(dirpath) / name))
                if len(out) >= MAX_LIST_ENTRIES:
                    return "\n".join(out + ["... (listing truncated)"])
        return "\n".join(out) if out else "(empty)"

    def read_file(self, rel: str) -> str:
        p = self.resolve(rel)
        if not p.is_file():
            raise SandboxError(f"no such file: {rel!r}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n... (file truncated)"
        return text

    def write_file(self, rel: str, content: object) -> str:
        if not isinstance(content, str):
            raise SandboxError("content must be a string")
        data = content.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise SandboxError(f"file too large ({len(data)} bytes; limit {MAX_FILE_BYTES})")
        p = self.resolve(rel)
        if p.is_dir():
            raise SandboxError(f"is a directory: {rel!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
        return f"wrote {len(data)} bytes to {self.rel(p)}"

    def replace_in_file(self, rel: str, old: object, new: object) -> str:
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise SandboxError("old and new must be strings and old must not be empty")
        text = self.read_file(rel)
        n = text.count(old)
        if n != 1:
            raise SandboxError(f"old text occurs {n} times in {rel!r}; it must occur exactly once")
        return self.write_file(rel, text.replace(old, new, 1))

    # -- test execution ------------------------------------------------------------------------
    def run_tests(self, start: str = "tests", timeout_s: float = DEFAULT_TEST_TIMEOUT_S) -> dict:
        start_dir = self.resolve(start)
        if not start_dir.is_dir():
            raise SandboxError(f"test directory not found: {start!r}")
        tmp = self.root / ".tmp"
        tmp.mkdir(exist_ok=True)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.root), "TMPDIR": str(tmp),
               "TEMP": str(tmp), "TMP": str(tmp), "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8",
               "no_proxy": "*", "NO_PROXY": "*"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        cmd = [sys.executable, "-I", "-B", "-c", _RUNNER, str(self.root), self.rel(start_dir), str(timeout_s)]
        kw: dict = {"cwd": str(self.root), "env": env, "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
        if os.name == "posix":
            kw["start_new_session"] = True
        proc = subprocess.Popen(cmd, **kw)
        timed_out = False
        try:
            raw, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            raw, _ = proc.communicate()
        return parse_runner_output(raw.decode("utf-8", errors="replace"), proc.returncode, timed_out)


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        proc.kill()


def parse_runner_output(text: str, returncode: int | None, timed_out: bool) -> dict:
    """Turn the child's stdout into a result dict; the last marker line wins."""
    res = {"ok": False, "run": 0, "failures": 0, "errors": 0, "skipped": 0,
           "timed_out": timed_out, "returncode": returncode}
    body = []
    for line in text.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                res.update(json.loads(line[len(RESULT_MARKER):]))
            except ValueError:
                pass
        else:
            body.append(line)
    if timed_out or returncode != 0:
        res["ok"] = False
    out = "\n".join(body).strip()
    if len(out) > MAX_OUTPUT_CHARS:
        out = "... (output truncated)\n" + out[-MAX_OUTPUT_CHARS:]
    res["output"] = out
    return res


def format_test_result(res: dict) -> str:
    head = "TIMED OUT" if res.get("timed_out") else ("PASSED" if res.get("ok") else "FAILED")
    return (f"{head}: ran {res.get('run', 0)}, failures {res.get('failures', 0)}, "
            f"errors {res.get('errors', 0)}\n{res.get('output', '')}")

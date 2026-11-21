# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The three offline workloads of the agentic benchmark.

tool_loop  generated workspaces with N stub functions; per function the agent edits the code, adds a
           test and runs the suite. Checked by hidden tests generated from the same seed.
repo_fix   small bundled repositories with a bug report and failing tests; checked by running the
           original tests (restored before the check, so editing them does not help).
research   questions about a bundled fictional corpus, read through local search and fetch tools;
           checked against an answer key.

Everything is generated from bundled files and a seed: no network and no downloads.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agentic_sandbox import SandboxError, Workspace, format_test_result

DATA = Path(__file__).resolve().parent / "data"
WORKLOADS = ("tool_loop", "repo_fix", "research")


def fn_tool(name: str, description: str, props: dict | None = None, required: list | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}


_STR = {"type": "string"}
FILE_TOOLS = [
    fn_tool("list_files", "List the files in the workspace, or under a sub-directory of it.",
            {"path": {"type": "string", "description": "Directory relative to the workspace; empty for the root."}}),
    fn_tool("read_file", "Read a text file from the workspace.", {"path": _STR}, ["path"]),
    fn_tool("write_file", "Create or overwrite a text file in the workspace with the given content.",
            {"path": _STR, "content": _STR}, ["path", "content"]),
    fn_tool("replace_in_file", "Replace one exact occurrence of old with new in a workspace file.",
            {"path": _STR, "old": _STR, "new": _STR}, ["path", "old", "new"]),
    fn_tool("run_tests", "Run the unittest suite in tests/ and return the result."),
    fn_tool("finish", "Call when the task is complete.", {"summary": _STR}),
]
RESEARCH_TOOLS = [
    fn_tool("search", "Search the document library. Returns matching document ids, titles and snippets.",
            {"query": _STR}, ["query"]),
    fn_tool("fetch", "Fetch the full text of one document by its id.", {"doc_id": _STR}, ["doc_id"]),
    fn_tool("submit_answer", "Submit the final answer. Keep it short: the value or name asked for.",
            {"answer": _STR}, ["answer"]),
]


@dataclass
class Episode:
    """One agent episode: a first user message, a tool dispatcher and a checker."""
    task_id: str
    user_message: str
    workspace: Workspace | None = None
    done: bool = False
    tool_counts: dict = field(default_factory=dict)

    def call_tool(self, name: str, args: dict) -> str:
        raise NotImplementedError

    def verify(self) -> dict:
        raise NotImplementedError

    def close(self, keep: bool = False) -> None:
        if self.workspace is not None and not keep:
            self.workspace.cleanup()

    def _count(self, name: str) -> None:
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1


class FileEpisode(Episode):
    test_timeout_s: float = 30.0

    def call_tool(self, name: str, args: dict) -> str:
        self._count(name)
        ws = self.workspace
        if name == "list_files":
            return ws.list_files(args.get("path") or "")
        if name == "read_file":
            return ws.read_file(args.get("path"))
        if name == "write_file":
            return ws.write_file(args.get("path"), args.get("content"))
        if name == "replace_in_file":
            return ws.replace_in_file(args.get("path"), args.get("old"), args.get("new"))
        if name == "run_tests":
            return format_test_result(ws.run_tests("tests", self.test_timeout_s))
        if name == "finish":
            self.done = True
            return "finished"
        raise SandboxError(f"unknown tool: {name!r}")


# -- tool loop -------------------------------------------------------------------------------------

@dataclass
class FuncSpec:
    name: str
    params: str
    doc: str
    body: str
    cases: list  # [(args tuple, expected)]

    def source(self, body: str | None = None) -> str:
        return f'def {self.name}({self.params}):\n    """{self.doc}"""\n    {body or self.body}\n'


# Each template returns (key, parameter list, docstring, reference body, reference callable, argument sets).
# The reference body is the source a correct agent could write; the callable computes the expected values.

def _affine(rng):
    a, b = rng.randint(2, 9), rng.randint(-20, 20)
    return ("affine", "x", f"Return x multiplied by {a}, plus {b}.", f"return x * {a} + {b}",
            lambda x: x * a + b, [(0,), (1,), (-3,), (7,), (12,)])


def _clamp(rng):
    lo, hi = rng.randint(-10, 0), rng.randint(5, 20)
    return ("clamp", "x", f"Return x clamped to the closed range [{lo}, {hi}].", f"return max({lo}, min({hi}, x))",
            lambda x: max(lo, min(hi, x)), [(-50,), (lo,), (0,), (3,), (hi,), (99,)])


def _count_char(rng):
    c = rng.choice("aeinorst")
    return ("count_char", "s", f"Return how many times the letter '{c}' occurs in s, ignoring case.",
            f"return s.lower().count('{c}')", lambda s: s.lower().count(c),
            [("",), ("Banana Split",), ("STREET ART",), ("a quiet morning in the station",)])


def _sum_divisible(rng):
    k = rng.randint(2, 7)
    return ("sum_divisible", "xs", f"Return the sum of the integers in the list xs that are divisible by {k}.",
            f"return sum(v for v in xs if v % {k} == 0)", lambda xs: sum(v for v in xs if v % k == 0),
            [([],), ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],), ([-k, 3 * k, 1],), ([0, 13, 21, 35],)])


def _join_reversed(rng):
    sep = rng.choice(["-", "+", " / ", ", ", "::"])
    return ("join_reversed", "words", f"Return the strings in the list words in reverse order, joined by {sep!r}.",
            f"return {sep!r}.join(reversed(words))", lambda words: sep.join(reversed(words)),
            [([],), (["a"],), (["x", "y", "z"],), (["one", "two"],)])


def _digit_sum(rng):
    m = rng.randint(2, 5)
    return ("digit_sum", "n", f"Return the sum of the decimal digits of abs(n), multiplied by {m}.",
            f"return sum(int(d) for d in str(abs(n))) * {m}", lambda n: sum(int(d) for d in str(abs(n))) * m,
            [(0,), (7,), (1234,), (-905,), (99999,)])


def _window_max(rng):
    k = rng.randint(2, 4)
    return ("window_max", "xs",
            f"Return a list with the maximum of every run of {k} consecutive items of the list xs, in order; "
            f"return an empty list when xs has fewer than {k} items.",
            f"return [max(xs[i:i + {k}]) for i in range(len(xs) - {k} + 1)]",
            lambda xs: [max(xs[i:i + k]) for i in range(len(xs) - k + 1)],
            [([],), ([5],), ([1, 3, 2, 5, 4, 1, 7],), ([9, 8, 7, 6, 5],)])


def _shift(s: str, k: int) -> str:
    return "".join(chr((ord(ch) - 97 + k) % 26 + 97) if "a" <= ch <= "z" else ch for ch in s)


def _caesar(rng):
    k = rng.randint(1, 25)
    return ("caesar", "s",
            f"Shift every lowercase ASCII letter in s forward by {k} places in the alphabet, wrapping from 'z' "
            f"to 'a'; leave all other characters unchanged.",
            f"return ''.join(chr((ord(ch) - 97 + {k}) % 26 + 97) if 'a' <= ch <= 'z' else ch for ch in s)",
            lambda s: _shift(s, k), [("",), ("abc",), ("xyz",), ("Hello, world!",)])


TEMPLATES = (_affine, _clamp, _count_char, _sum_divisible, _join_reversed, _digit_sum, _window_max, _caesar)


def generate_funcs(seed: str, n: int) -> list[FuncSpec]:
    """Deterministic function specs for one tool-loop workspace."""
    rng = random.Random(f"tool_loop:{seed}")
    offset = rng.randrange(len(TEMPLATES))
    specs = []
    for j in range(n):
        key, params, doc, body, fn, arg_sets = TEMPLATES[(offset + j) % len(TEMPLATES)](rng)
        cases = [(args, fn(*args)) for args in arg_sets]
        specs.append(FuncSpec(f"{key}_{j + 1:02d}", params, doc, body, cases))
    return specs


def render_stub_module(specs: list[FuncSpec], solved: bool = False) -> str:
    head = '"""Functions to implement. Each docstring is the specification."""\n\n\n'
    return head + "\n\n".join(s.source(None if solved else "raise NotImplementedError") for s in specs)


def render_checker(specs: list[FuncSpec]) -> str:
    out = ["import unittest", "", "import funcs", "", ""]
    for s in specs:
        out.append(f"class Check_{s.name}(unittest.TestCase):")
        out.append("    def test_cases(self):")
        for args, expected in s.cases:
            call = ", ".join(repr(a) for a in args)
            out.append(f"        self.assertEqual(funcs.{s.name}({call}), {expected!r})")
        out.append("")
    return "\n".join(out) + "\n"


class ToolLoopEpisode(FileEpisode):
    def __init__(self, task_id: str, seed: str, n_funcs: int, test_timeout_s: float):
        self.specs = generate_funcs(seed, n_funcs)
        ws = Workspace.create(prefix="agentic-tool-")
        ws.write_file("funcs.py", render_stub_module(self.specs))
        ws.write_file("tests/__init__.py", "")
        ws.write_file("tests/test_import.py",
                      "import unittest\n\nimport funcs\n\n\nclass ImportTest(unittest.TestCase):\n"
                      "    def test_module_imports(self):\n        self.assertTrue(hasattr(funcs, '__doc__'))\n")
        names = ", ".join(s.name for s in self.specs)
        msg = (f"Task id: {task_id}\n"
               f"The workspace contains funcs.py with {n_funcs} stub functions: {names}. Each docstring is the "
               "specification. Work through the functions in order. For each function:\n"
               "1. implement it in funcs.py;\n"
               "2. add tests/test_<function name>.py with at least one unittest test case for it;\n"
               "3. call run_tests and fix any failure before moving on.\n"
               "When every function is implemented and the suite passes, call finish.")
        super().__init__(task_id=task_id, user_message=msg, workspace=ws)
        self.test_timeout_s = test_timeout_s

    def verify(self) -> dict:
        ws = self.workspace
        shutil.rmtree(ws.root / "_verify", ignore_errors=True)
        ws.write_file("_verify/__init__.py", "")
        ws.write_file("_verify/test_checker.py", render_checker(self.specs))
        res = ws.run_tests("_verify", self.test_timeout_s)
        n = len(self.specs)
        passed = max(0, res["run"] - res["failures"] - res["errors"]) if res["run"] == n else 0
        tests_written = sum((ws.root / "tests" / f"test_{s.name}.py").is_file() for s in self.specs)
        ok = bool(res["ok"]) and res["run"] == n and tests_written == n
        return {"success": ok, "score": passed / n if n else 0.0, "functions_passed": passed,
                "functions": n, "tests_written": tests_written, "checker_timed_out": res["timed_out"]}


# -- repo fix ----------------------------------------------------------------------------------------

def repo_tasks() -> list[dict]:
    tasks = []
    for d in sorted((DATA / "repo_tasks").iterdir()):
        if (d / "task.json").is_file():
            meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
            meta["dir"] = d
            tasks.append(meta)
    return tasks


def _copy_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())


def count_test_methods(test_dir: Path) -> int:
    return sum(len(re.findall(r"^\s+def test_", p.read_text(encoding="utf-8"), re.M))
               for p in test_dir.rglob("test*.py"))


class RepoFixEpisode(FileEpisode):
    def __init__(self, task_id: str, task: dict, test_timeout_s: float):
        self.task = task
        ws = Workspace.create(prefix="agentic-repo-")
        _copy_tree(task["dir"] / "repo", ws.root)
        msg = (f"Task id: {task_id}\n"
               f"Bug report: {task['title']}\n\n{task['issue']}\n\n"
               "Fix the code in this repository so that every test in tests/ passes. Do not change the tests. "
               "Use list_files and read_file to look around, edit the code, call run_tests to check, and call "
               "finish when the suite passes.")
        super().__init__(task_id=task_id, user_message=msg, workspace=ws)
        self.test_timeout_s = test_timeout_s

    def verify(self) -> dict:
        ws = self.workspace
        test_dir = self.task.get("test_dir", "tests")
        pristine = self.task["dir"] / "repo" / test_dir
        dst = ws.root / test_dir
        shutil.rmtree(dst, ignore_errors=True)
        _copy_tree(pristine, dst)
        res = ws.run_tests(test_dir, self.test_timeout_s)
        expected = count_test_methods(pristine)
        passed = max(0, res["run"] - res["failures"] - res["errors"])
        ok = bool(res["ok"]) and res["run"] == expected
        return {"success": ok, "score": passed / expected if expected else 0.0, "tests_passed": passed,
                "tests": expected, "checker_timed_out": res["timed_out"]}


# -- research ----------------------------------------------------------------------------------------

def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA / "corpus").glob("*.md"))}


def load_questions() -> list[dict]:
    return json.loads((DATA / "research_questions.json").read_text(encoding="utf-8"))


_WORD = re.compile(r"[a-z0-9]+")


def normalise_answer(text: str) -> str:
    t = text.lower().replace("-", " ").replace("–", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(".!?;:, ")


def answer_matches(answer: object, accept: list[str], max_len: int = 300) -> bool:
    """True when a normalised accepted answer appears as a whole token in the submitted answer."""
    if not isinstance(answer, str) or not answer.strip() or len(answer) > max_len:
        return False
    norm = normalise_answer(answer)
    return any(re.search(r"(?<![\w.])" + re.escape(normalise_answer(acc)) + r"(?!\w)", norm) for acc in accept)


def search_corpus(corpus: dict[str, str], query: object, limit: int = 5) -> str:
    if not isinstance(query, str) or not query.strip():
        raise SandboxError("query must be a non-empty string")
    terms = set(_WORD.findall(query.lower()))
    scored = []
    for doc_id, text in corpus.items():
        words = _WORD.findall(text.lower())
        score = sum(words.count(t) for t in terms) + 5 * sum(t in doc_id for t in terms)
        if score:
            scored.append((-score, doc_id))
    if not scored:
        return "no matching documents"
    lines = []
    for _, doc_id in sorted(scored)[:limit]:
        text = corpus[doc_id]
        title = text.splitlines()[0].lstrip("# ").strip()
        body = " ".join(" ".join(text.splitlines()[1:]).split())[:160]
        lines.append(f"{doc_id}: {title} -- {body}...")
    return "\n".join(lines)


class ResearchEpisode(Episode):
    def __init__(self, task_id: str, question: dict, corpus: dict[str, str]):
        self.question = question
        self.corpus = corpus
        self.answer = None
        listing = ", ".join(corpus)
        msg = (f"Task id: {task_id}\n"
               f"Question: {question['question']}\n\n"
               "Answer from the document library only; the facts are not known outside it, so do not guess. "
               f"Documents in the library: {listing}. Use search and fetch to read what you need, then call "
               "submit_answer with just the value or name asked for.")
        super().__init__(task_id=task_id, user_message=msg, workspace=None)

    def call_tool(self, name: str, args: dict) -> str:
        self._count(name)
        if name == "search":
            return search_corpus(self.corpus, args.get("query"))
        if name == "fetch":
            doc_id = args.get("doc_id")
            key = doc_id.strip().removesuffix(".md") if isinstance(doc_id, str) else None
            if key not in self.corpus:
                raise SandboxError(f"no such document: {doc_id!r}; known ids: {', '.join(self.corpus)}")
            return self.corpus[key]
        if name == "submit_answer":
            self.answer = args.get("answer")
            self.done = True
            return "answer recorded"
        raise SandboxError(f"unknown tool: {name!r}")

    def verify(self) -> dict:
        ok = answer_matches(self.answer, self.question["accept"])
        return {"success": ok, "score": 1.0 if ok else 0.0, "answer": self.answer}


# -- registry ----------------------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "tool_loop": "You are a careful software engineer working in a sandboxed Python workspace. You can act only "
                 "through the provided tools. Paths are relative to the workspace root. The test runner is "
                 "Python's unittest; tests live in tests/.",
    "repo_fix": "You are a careful software engineer fixing a bug in a small Python repository in a sandboxed "
                "workspace. You can act only through the provided tools. Paths are relative to the repository "
                "root. The test runner is Python's unittest; tests live in tests/.",
    "research": "You are a research assistant. You answer questions using only a local document library that "
                "you reach through the search and fetch tools. Read the documents you need before answering, "
                "check arithmetic, and give a short final answer through submit_answer.",
}


def tools_for(workload: str) -> list[dict]:
    return RESEARCH_TOOLS if workload == "research" else FILE_TOOLS


def make_episode(workload: str, index: int, seed: int, n_funcs: int = 6, test_timeout_s: float = 30.0) -> Episode:
    """Episode number ``index`` of a workload; the same (workload, index, seed) gives the same task."""
    if workload == "tool_loop":
        tag = f"s{seed}-t{index}"
        return ToolLoopEpisode(f"tool_loop/{tag}", tag, n_funcs, test_timeout_s)
    if workload == "repo_fix":
        tasks = repo_tasks()
        task = tasks[(index + seed) % len(tasks)]
        return RepoFixEpisode(f"repo_fix/{task['id']}", task, test_timeout_s)
    if workload == "research":
        qs = load_questions()
        q = qs[(index + seed) % len(qs)]
        return ResearchEpisode(f"research/{q['id']}", q, load_corpus())
    raise ValueError(f"unknown workload {workload!r}; choose from {', '.join(WORKLOADS)}")


def solve_tool_loop(ep: ToolLoopEpisode) -> None:
    """Write the reference implementations and one test per function (self-check and tests only)."""
    ep.workspace.write_file("funcs.py", render_stub_module(ep.specs, solved=True))
    for s in ep.specs:
        args, expected = s.cases[-1]
        call = ", ".join(repr(a) for a in args)
        ep.workspace.write_file(
            f"tests/test_{s.name}.py",
            f"import unittest\n\nimport funcs\n\n\nclass Test_{s.name}(unittest.TestCase):\n"
            f"    def test_case(self):\n        self.assertEqual(funcs.{s.name}({call}), {expected!r})\n")


def apply_repo_solution(ep: RepoFixEpisode) -> None:
    """Copy the bundled reference fix over the workspace (self-check and tests only)."""
    _copy_tree(ep.task["dir"] / "solution", ep.workspace.root)


def self_check(n_funcs: int = 6, test_timeout_s: float = 60.0) -> list[dict]:
    """Offline check of the bundled tasks: each must fail as shipped and pass with its reference fix."""
    rows = []
    for task in repo_tasks():
        ep = RepoFixEpisode(f"repo_fix/{task['id']}", task, test_timeout_s)
        try:
            before = ep.verify()
            apply_repo_solution(ep)
            after = ep.verify()
        finally:
            ep.close()
        rows.append({"task": ep.task_id, "fails_as_shipped": not before["success"],
                     "passes_with_fix": after["success"]})
    ep = ToolLoopEpisode("tool_loop/self-check", "self-check", n_funcs, test_timeout_s)
    try:
        before = ep.verify()
        solve_tool_loop(ep)
        after = ep.verify()
    finally:
        ep.close()
    rows.append({"task": ep.task_id, "fails_as_shipped": not before["success"], "passes_with_fix": after["success"]})
    corpus, qs = load_corpus(), load_questions()
    keyed = bool(corpus) and all(answer_matches(q["accept"][0], q["accept"]) for q in qs)
    rows.append({"task": "research/answer-key", "fails_as_shipped": not answer_matches("unknown", ["1896"]),
                 "passes_with_fix": keyed})
    return rows

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Check the Merlin engine wheel (stdlib only).

  wheel.py audit  WHEEL --engine ENGINE [--forbid TEXT ...]
                                              release gate run by build-wheel.sh
  wheel.py verify WHEEL RECORD ENGINE         install gate run by install-env.sh
  wheel.py native-ids ENGINE                  native inputs at HEAD
  wheel.py native-snapshot ENGINE             inputs and tracked paths for exported installs
  wheel.py flashinfer-ids PURELIB v1|v2        verified installed patch targets

The wheel's local version names the commit it was built from, and the release when the
build was given one (0.33.0+merlin3.0.2.2.<commit>; 0.33.0+merlin.<commit> without);
RECORD (release/merlin-wheel.sha256) holds "<sha256>  <filename>" for the one wheel
this checkout accepts. release/merlin-wheel.native, next to RECORD, holds one
"<git object id>  <path>" line per NATIVE_SOURCES path, taken at that commit:
verify accepts the wheel only while `git rev-parse HEAD:<path>` in the checkout
gives the same ids, so the build commit itself need not be in the checkout's
history.
"""

import argparse
import base64
import collections
import csv
import email.parser
import hashlib
import io
import json
import re
import runpy
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import NamedTuple

# merlin<release>.<commit>: the release is optional (dotted digits, no leading dot);
# merlin.<commit> is the form the wheels before 3.0.2.2 carry and stays valid.
NAME_RE = re.compile(
    r"vllm-0\.33\.0\+merlin(?P<release>[0-9]+(?:\.[0-9]+)*)?\.(?P<commit>[0-9a-f]{10})"
    r"-cp313-cp313-linux_x86_64\.whl"
)
REQUIRED = (
    "vllm/_C_stable_libtorch.abi3.so",
    "vllm/_moe_C_stable_libtorch.abi3.so",
    "vllm/cumem_allocator.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
)
# A change under these paths changes the compiled products (CUDA/C++ extensions and the Rust frontend).
# release/merlin-wheel.native records their git object ids at the wheel's build commit.
NATIVE_SOURCES = (
    "csrc", "cmake", "CMakeLists.txt", "setup.py",
    "rust", "rust-toolchain.toml", "tools/build_rust.py",
)
# Authorship and project fields must be exactly upstream vLLM's plus the Merlin modifications author
# (pyproject.toml): the fork adds no maintainer or URL. The package carries LICENSE and the Merlin NOTICE.
METADATA_EXPECTED = {
    "Author": ["vLLM Team, halt95 (Merlin modifications)"],
    "License-File": ["LICENSE", "NOTICE"],
    "Author-email": [],
    "Maintainer": [],
    "Maintainer-email": [],
    "Description-Content-Type": ["text/markdown"],
    "Description": [],  # the description is the message body, never a header
    "Summary": ["A high-throughput and memory-efficient inference and serving engine for LLMs"],
    "Home-page": [],
    "Project-URL": [
        "Homepage, https://github.com/vllm-project/vllm",
        "Documentation, https://docs.vllm.ai/en/latest/",
        "Slack, https://slack.vllm.ai/",
    ],
}
# The description is the short fixed text in pyproject.toml ([project] readme, inline, with a link to the
# README at a release tag), never the README file: a documentation-only commit must not change the wheel.
MAX_DESCRIPTION = 2000
TAGGED_README = re.compile(
    r"https://github\.com/halt95/merlin-engine/blob/merlin-v[0-9]+(?:\.[0-9]+)*/README\.md"
)
# The release wheel is built for CUDA 13 (the lock's toolkit wheels). Requirement lines setup.py drops there:
SKIPPED_REQUIREMENTS = ("vllm-flash-attn", "flashinfer-cubin", "flashinfer-jit-cache")
REQUIREMENTS_FILE = "cuda.txt"
MARKER_TOKEN = re.compile(
    r"""[ \t]+|(?P<quote>"[^"\\\n\r]*"|'[^'\\\n\r]*')|(?P<paren>[()])|(?P<op>===|==|!=|<=|>=|~=|<|>)"""
    r"""|(?P<word>[A-Za-z0-9_.*+-]+)|(?P<other>.)""",
    re.S,
)
# The members every wheel carries besides REQUIRED: the Rust frontend and, in its one .dist-info directory,
# these files and the licence files METADATA_EXPECTED names (under licenses/).
FRONTEND = "vllm/vllm-rs"
DIST_INFO_FILES = ("METADATA", "WHEEL", "RECORD")
# Build-host paths no member may contain. Callers add needles (--forbid); they never replace these.
FORBIDDEN = ("/home/", "/root/")
# Reviewed upstream lines that legitimately contain a FORBIDDEN needle, as (member, exact line without its
# line ending). An entry excuses that one line of a text member from the FORBIDDEN scan only: needles a
# caller adds and the native members are never excused. Empty while no upstream line needs it.
EXEMPT: frozenset[tuple[str, bytes]] = frozenset()


class WheelError(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class WheelName(NamedTuple):
    commit: str
    release: str | None  # None for the old merlin.<commit> form


def parse_name(wheel: Path) -> WheelName:
    m = NAME_RE.fullmatch(wheel.name)
    if not m:
        raise WheelError(
            f"{wheel.name} is not a Merlin engine wheel name "
            "(vllm-0.33.0+merlin[<release>].<commit>-cp313-cp313-linux_x86_64.whl)"
        )
    return WheelName(m["commit"], m["release"])


def members(wheel: Path) -> list[zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(wheel) as z:
            return z.infolist()
    except zipfile.BadZipFile as e:
        raise WheelError(f"{wheel.name} is not a zip archive: {e}") from e


def version(commit: str, release: str | None = None) -> str:
    return f"0.33.0+merlin{release or ''}.{commit}"


def is_native(name: str) -> bool:
    return name.endswith(".so") or ".so." in name or name == FRONTEND


def check_members(
    names: Iterable[str], commit: str | None = None, release: str | None = None
) -> None:
    """The member list is complete; with commit, the .dist-info directory is the one the wheel name gives."""
    names = list(names)
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        raise WheelError(f"wheel lacks {', '.join(missing)}")
    cached = sorted(n for n in names if n.endswith(".pyc") or "/__pycache__/" in n)
    if cached:
        raise WheelError(f"wheel carries bytecode: {cached[0]} (+{len(cached) - 1})")
    twice = sorted(n for n, count in collections.Counter(names).items() if count > 1)
    if twice:
        raise WheelError(f"wheel lists a member twice: {twice[0]} (+{len(twice) - 1})")
    dists = sorted({n.split("/", 1)[0] for n in names if n.split("/", 1)[0].endswith(".dist-info")})
    if len(dists) != 1:
        raise WheelError(f"wheel must carry exactly one .dist-info directory, has {dists}")
    dist = dists[0]
    if commit is not None:
        want = f"vllm-{version(commit, release)}.dist-info"
        if dist != want:
            raise WheelError(f"wheel carries {dist}, its name gives {want}")
    need = [f"{dist}/{f}" for f in DIST_INFO_FILES]
    need += [f"{dist}/licenses/{f}" for f in METADATA_EXPECTED["License-File"]]
    need.append(FRONTEND)
    missing = [n for n in need if n not in names]
    if missing:
        raise WheelError(f"wheel lacks {', '.join(missing)}")


def runpaths(so: Path) -> list[str]:
    out = subprocess.run(
        ["readelf", "-d", str(so)], capture_output=True, text=True, check=True
    ).stdout
    return re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[(.*)\]", out)


def record_problems(name: str, raw: bytes, digests: dict[str, tuple[str, int]]) -> list[str]:
    """RECORD against the archive: digests holds (sha256, size) for every file member but RECORD itself."""
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as e:
        raise WheelError(f"{name} is not a UTF-8 CSV file: {e}") from e
    problems = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            problems.append(f"{name}: malformed row {row!r}")
            continue
        path, digest, size = row
        if path in seen:
            problems.append(f"{name}: lists {path} twice")
            continue
        seen.add(path)
        if path == name:
            if digest or size:
                problems.append(f"{name}: its own row must have no hash and no size")
        elif path not in digests:
            problems.append(f"{name}: lists {path}, which is not in the wheel")
        else:
            want_digest, want_size = digests[path]
            if digest != f"sha256={want_digest}":
                problems.append(f"{name}: hash of {path} is {digest!r}, the member has 'sha256={want_digest}'")
            if size != str(want_size):
                problems.append(f"{name}: size of {path} is {size!r}, the member has {want_size}")
    problems += [f"{name}: does not list {path}" for path in (*digests, name) if path not in seen]
    return problems


def text_problems(name: str, data: bytes, base: list[bytes], added: list[bytes]) -> list[str]:
    """Needles in a non-native member; EXEMPT excuses whole reviewed lines from the base needles only."""
    lines = [line for line in data.splitlines() if (name, line) not in EXEMPT]
    counts = [(n, sum(line.count(n) for line in lines)) for n in base]
    counts += [(n, data.count(n)) for n in added]
    return [f"{name}: contains {n.decode()!r} ({count}x)" for n, count in counts if count]


def project_table(engine: Path) -> dict:
    """The [project] table of the engine tree's pyproject.toml."""
    import tomllib  # the audit runs on the build venv's Python (3.13); the install gate never loads this

    try:
        return tomllib.loads((engine / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    except (OSError, ValueError, KeyError) as e:
        raise WheelError(f"cannot read the [project] table of {engine / 'pyproject.toml'}: {e}") from e


def expected_description(project: dict) -> bytes:
    """The wheel's description: pyproject.toml's inline readme text, short, linking the README at a release tag."""
    readme = project.get("readme")
    if not isinstance(readme, dict) or set(readme) != {"text", "content-type"}:
        raise WheelError("pyproject.toml: [project] readme must be inline text, not the README file")
    text = readme["text"]
    if readme["content-type"] != "text/markdown" or not isinstance(text, str):
        raise WheelError("pyproject.toml: [project] readme must be text/markdown text")
    if len(text.encode()) > MAX_DESCRIPTION or not TAGGED_README.search(text):
        raise WheelError(
            f"pyproject.toml: the readme text must be at most {MAX_DESCRIPTION} bytes and link the README at a release tag"
        )
    return text.encode()


def marker_tokens(marker: str) -> list[tuple[str, str]]:
    """(kind, text) tokens of an environment marker; quoted values keep their content, quotes become double."""
    tokens = []
    for m in MARKER_TOKEN.finditer(marker):
        kind = m.lastgroup
        if kind is None:
            continue  # whitespace between tokens
        text = m.group()
        if kind == "quote" and '"' not in text[1:-1]:
            text = '"' + text[1:-1] + '"'
        tokens.append((kind, text))
    return tokens


def normalize_marker(marker: str) -> str:
    """The marker without the whitespace between tokens (the space inside a quoted value is kept)."""
    out = ""
    for kind, text in marker_tokens(marker):
        if kind == "word" and text in ("and", "or", "in", "not"):
            out = out.rstrip(" ") + f" {text} "
        else:
            out += text
    return out.strip()


EXTRA_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


def canonical_extra(name: str) -> str:
    """An extra's name as pip compares it: runs of - _ . become one -, case folded."""
    return re.sub(r"[-_.]+", "-", name).lower()


# The environment marker variables PEP 508 and packaging define (the legacy dotted names included).
MARKER_VARIABLES = frozenset({
    "python_version", "python_full_version", "os_name", "sys_platform", "platform_release", "platform_system",
    "platform_version", "platform_machine", "platform_python_implementation", "implementation_name",
    "implementation_version", "extra", "extras", "dependency_groups", "os.name", "sys.platform",
    "platform.version", "platform.machine", "platform.python_implementation", "python_implementation",
})


# Comparisons every installer evaluates for any value (the others can raise on a value that is not a version).
PLAIN_OPERATORS = frozenset({"==", "!=", "<", "<=", ">", ">=", "in", "not in"})


class _NotAMarker(Exception):
    pass


def marker_in_base(marker: str) -> tuple[bool, list[str]]:
    """(can the marker hold on a plain install, the extra names it compares).

    A plain install evaluates a marker with extra set to the empty string and every other variable
    unknown, so only an `extra == "<name>"` clause is decided: false for a valid non-empty name, true for
    an empty or blank value, and only for ==. Every other comparison of the other variables with a literal
    (==, !=, ordering, in, not in) and extra != counts as possibly true, as does an invalid extra name. A
    marker this parser cannot read, or that an installer can refuse to evaluate, raises _NotAMarker: unbalanced
    brackets, an escaped or unterminated quote, a name that is not a marker variable, whitespace other than
    space and tab, two literals or two variables compared, === or ~=, extras and dependency_groups, extra with
    any operator but == and !=, no marker."""
    tokens = marker_tokens(marker)
    names: list[str] = []
    pos = 0

    def peek(n=0):
        return tokens[pos + n] if pos + n < len(tokens) else (None, None)

    def take():
        nonlocal pos
        if pos >= len(tokens):
            raise _NotAMarker
        pos += 1
        return tokens[pos - 1]

    def keyword(word):
        return peek() == ("word", word)

    def operand():
        kind, text = take()
        if kind == "quote":
            return ("str", text[1:-1])
        if kind == "word" and text in MARKER_VARIABLES:
            return ("var", text)
        raise _NotAMarker

    def comparator():
        kind, text = take()
        if kind == "op" or (kind == "word" and text == "in"):
            return text
        if kind == "word" and text == "not" and take() == ("word", "in"):
            return "not in"
        raise _NotAMarker

    def atom():
        if peek() == ("paren", "("):
            take()
            value = expr()
            if take() != ("paren", ")"):
                raise _NotAMarker
            return value
        left, op, right = operand(), comparator(), operand()
        if left[0] == right[0] == "str":
            raise _NotAMarker  # two literals: the installer cannot evaluate it
        if left[0] == right[0] == "var":
            raise _NotAMarker  # a variable against a variable: not a form the installer's rules are mirrored for
        var, other = (left, right) if left[0] == "var" else (right, left)
        if var[1] in ("extras", "dependency_groups"):
            raise _NotAMarker  # set-valued variables: only `in` / `not in` evaluate, and not on every installer
        if var[1] != "extra":
            if op not in PLAIN_OPERATORS:
                raise _NotAMarker  # === and ~= raise when the value is not a version
            return True
        if op == "!=":
            return True  # extra is never the compared name on a plain install, so this holds
        if op != "==":
            raise _NotAMarker
        value = other[1]
        if not value.strip():
            return True  # extra is the empty string on a plain install
        if not EXTRA_NAME.fullmatch(value):
            return True  # not an extra name; compare the line
        names.append(canonical_extra(value))
        return False

    def conjunction():
        value = atom()
        while keyword("and"):
            take()
            value = atom() and value  # both sides are read, so every extra name is collected
        return value

    def expr():
        value = conjunction()
        while keyword("or"):
            take()
            value = conjunction() or value
        return value

    result = expr()
    if pos != len(tokens):
        raise _NotAMarker
    return result, names


def marker_is_valid(marker: str) -> bool:
    """True if the marker follows the PEP 508 grammar: an or of ands of comparisons between a marker variable
    and a quoted string (either order, or two of either), brackets balanced. Whether it can be evaluated is
    a separate question (see marker_in_base)."""
    tokens = marker_tokens(marker)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def take():
        nonlocal pos
        if pos >= len(tokens):
            raise _NotAMarker
        pos += 1
        return tokens[pos - 1]

    def operand():
        kind, text = take()
        if kind == "quote" or (kind == "word" and text in MARKER_VARIABLES):
            return
        raise _NotAMarker

    def atom():
        if peek() == ("paren", "("):
            take()
            expr()
            if take() != ("paren", ")"):
                raise _NotAMarker
            return
        operand()
        kind, text = take()
        if not (kind == "op" or (kind == "word" and text == "in")
                or (kind == "word" and text == "not" and take() == ("word", "in"))):
            raise _NotAMarker
        operand()

    def expr():
        atom()
        while peek() in (("word", "and"), ("word", "or")):
            take()
            atom()

    try:
        expr()
    except _NotAMarker:
        return False
    return pos == len(tokens)


def extra_only(marker: str, declared: set[str]) -> bool:
    """True only if the line cannot install on a plain install and every extra it names is one the wheel
    declares in Provides-Extra (canonical names). A line for an undeclared extra is compared instead, so the
    audit reports it as an added requirement."""
    try:
        possible, names = marker_in_base(marker)
    except _NotAMarker:
        return False
    return not possible and all(name in declared for name in names)


# PEP 508 pieces. Whitespace inside a requirement is space or tab, and only where the grammar allows it.
_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_EXTRAS = rf"\[[ \t]*(?:{_NAME}(?:[ \t]*,[ \t]*{_NAME})*[ \t]*)?\]"
def _version(segments: str, local: bool) -> str:
    """A PEP 440 version (case-insensitive): epoch, release, pre, post, dev and optionally a local part."""
    return (
        rf"(?i:v?(?:[0-9]+!)?{segments}(?:[-_.]?(?:alpha|a|beta|b|preview|pre|c|rc)[-_.]?[0-9]*)?"
        r"(?:-[0-9]+|[-_.]?(?:post|rev|r)[-_.]?[0-9]*)?(?:[-_.]?dev[-_.]?[0-9]*)?"
        + (r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?)" if local else ")")
    )


_V_LOCAL = _version(r"[0-9]+(?:\.[0-9]+)*", True)
_V_PLAIN = _version(r"[0-9]+(?:\.[0-9]+)*", False)
_V_COMPAT = _version(r"[0-9]+(?:\.[0-9]+)+", False)
# a prefix match (== and != only): epoch and release segments, then .* as the last segment, no suffix
_V_PREFIX = r"(?i:v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*\.\*)"
_SPEC = (
    rf"(?:(?:==|!=)[ \t]*(?:{_V_LOCAL}|{_V_PREFIX})"
    rf"|(?:<=|>=|<|>)[ \t]*{_V_PLAIN}"
    rf"|~=[ \t]*{_V_COMPAT}"
    r"|===[ \t]*[^ \t,;()]+)"
)
_SPECS = rf"{_SPEC}(?:[ \t]*,[ \t]*{_SPEC})*(?:[ \t]*,)?"  # packaging accepts one trailing comma
REQUIREMENT = re.compile(
    rf"(?P<name>{_NAME})[ \t]*(?P<extras>{_EXTRAS})?[ \t]*(?:(?P<specs>{_SPECS})|\([ \t]*(?P<paren>{_SPECS})[ \t]*\))?"
)
URL_REQUIREMENT = re.compile(
    rf"(?P<name>{_NAME})[ \t]*(?P<extras>{_EXTRAS})?[ \t]*@[ \t]*(?P<url>[^ \t]+)(?:[ \t]+;(?P<marker>.*))?", re.S
)
SPEC_CLAUSE = re.compile(r"(===|==|!=|<=|>=|~=|<|>)[ \t]*([^ \t,()]+)")


def is_url_requirement(text: str) -> bool:
    """A direct URL requirement (name @ url): its URL runs to the next space or tab, so a ';' glued to the
    URL belongs to the URL and the first ';' of the line is not the start of a marker."""
    return "@" in text.partition(";")[0]


def skipped_requirement(text: str, declared: set[str]) -> bool:
    """True if the audit ignores this Requires-Dist line: no direct URL, and it cannot install without a declared extra."""
    spec, _, marker = text.partition(";")
    return "@" not in spec and extra_only(marker, declared)


def normalize_requirement(text: str) -> str:
    """One requirement in a form both requirements/*.txt lines and METADATA Requires-Dist lines reduce to.

    The requirement is validated first (PEP 508 name, extras and version specifiers, whitespace only where the
    grammar allows it), so a malformed line is refused and never reduced to a valid one by case folding or by
    deleting whitespace."""
    text = text.strip()
    invalid = WheelError(f"Requires-Dist is not a valid requirement: {text}")
    if is_url_requirement(text):
        m = URL_REQUIREMENT.fullmatch(text)
        if not m or (m["marker"] is not None and not m["marker"].strip(" \t")):
            raise invalid
        # the URL is compared exactly: nothing in it is sorted, lowered or stripped
        specs, marker = m["url"], m["marker"]
        sep = "@"
    else:
        spec, semicolon, marker = text.partition(";")
        m = REQUIREMENT.fullmatch(spec.rstrip(" \t"))
        if not m or (semicolon and not marker.strip(" \t")):
            raise invalid
        clauses = SPEC_CLAUSE.findall(m["specs"] or m["paren"] or "")
        specs, sep = ",".join(sorted(f"{op}{version}" for op, version in clauses)), ""
    name = re.sub(r"[-_.]+", "-", m["name"]).lower()
    extras = ""
    if m["extras"]:
        names = [canonical_extra(e.strip(" \t")) for e in m["extras"][1:-1].split(",") if e.strip(" \t")]
        extras = "[" + ",".join(sorted(names)) + "]" if names else ""
    if marker:
        if not marker_is_valid(marker):
            raise invalid
        marker = normalize_marker(marker)
    return f"{name}{extras}{sep}{specs}" + (f";{marker}" if marker else "")


def requirement_outcome(line: str, declared: set[str]) -> tuple[str, str]:
    """What the audit does with one Requires-Dist value: ("invalid", message), ("ignored", "") or
    ("compared", normal form). The requirement is validated first, so a line that is not a valid requirement is
    never ignored for its marker and never reduced to a valid one."""
    try:
        normal = normalize_requirement(line)
    except WheelError as e:
        return "invalid", str(e)
    if skipped_requirement(line.strip(), declared):
        return "ignored", ""
    return "compared", normal


def normalize_specifier(text: str) -> str:
    """A version specifier set without whitespace and in sorted order, as setuptools writes it."""
    return ",".join(sorted(re.sub(r"\s+", "", text).split(",")))


def unfold(value: str) -> str:
    """A header value as one line: a folded header carries its line breaks inside the value."""
    return " ".join(value.split())


def read_requirements(directory: Path, filename: str) -> list[str]:
    """The requirement lines setup.py reads from requirements/<filename>, with -r files followed."""
    lines = []
    for line in (directory / filename).read_text(encoding="utf-8").strip().split("\n"):
        if line.startswith("-r "):
            lines += read_requirements(directory, line.split()[1])
        elif line.strip() and not line.startswith(("--", "#")):
            lines.append(line)
    return lines


def expected_requires(engine: Path) -> list[str]:
    """The non-extra Requires-Dist set the release wheel carries, from the tree's requirements."""
    lines = read_requirements(engine / "requirements", REQUIREMENTS_FILE)
    kept = []
    for line in lines:
        text = normalize_requirement(re.split(r"\s#", line)[0])
        # setup.py drops these by substring; only the exact (canonical) names are dropped here, so a
        # different name that merely contains one is still expected in the wheel
        if re.match(r"[a-z0-9-]+", text).group() not in SKIPPED_REQUIREMENTS:
            kept.append(text)
    return sorted(kept)


def metadata_problems(name: str, data: bytes, commit: str, project: dict, description: bytes,
                      requires: list[str], release: str | None = None) -> list[str]:
    """The checks on one wheel's METADATA against the engine tree's pyproject.toml and requirements."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return [f"{name}: METADATA is not valid UTF-8 ({e.reason} at byte {e.start})"]
    problems = []
    msg = email.parser.Parser().parsestr(text)
    expected = {
        "Name": ["vllm"], "Version": [version(commit, release)], **METADATA_EXPECTED,
    }
    for field, want in expected.items():
        have = [unfold(v) for v in msg.get_all(field, [])]
        if have != want:
            problems.append(f"{name}: {field} is {have!r}, expected {want!r}")
    # the Python floor is the tree's, as a specifier set (setuptools writes it sorted)
    floor = [normalize_specifier(v) for v in msg.get_all("Requires-Python", [])]
    want_floor = [normalize_specifier(str(project.get("requires-python")))]
    if floor != want_floor:
        problems.append(f"{name}: Requires-Python is {floor!r}, expected {want_floor!r}")
    # the dependency set is the tree's requirements; a line is ignored only when it cannot install
    # without a declared extra (extra lines are not compared), and a direct URL is always compared
    declared = {canonical_extra(v) for v in msg.get_all("Provides-Extra", [])}
    have = []
    for line in msg.get_all("Requires-Dist", []):
        outcome, detail = requirement_outcome(line, declared)
        if outcome == "invalid":
            problems.append(f"{name}: {detail}")
        elif outcome == "compared":
            have.append(detail)
    have.sort()
    for label, gone in (("lacks", sorted(set(requires) - set(have))),
                        ("adds", sorted(set(have) - set(requires)))):
        if gone:
            problems.append(f"{name}: Requires-Dist {label} {', '.join(gone)}")
    if have != requires and set(have) == set(requires):
        problems.append(f"{name}: Requires-Dist lists a requirement a different number of times")
    # the description is pyproject.toml's fixed text (the public-tree scan covers it)
    lf = data.replace(b"\r\n", b"\n")
    body = lf.split(b"\n\n", 1)[1] if b"\n\n" in lf else b""
    if body.rstrip() != description.replace(b"\r\n", b"\n").rstrip():
        problems.append(f"{name}: the description is not the text in pyproject.toml")
    return problems


def audit(wheel: Path, forbid: list[str], engine: Path) -> list[str]:
    commit, release = parse_name(wheel)
    infos = [i for i in members(wheel) if not i.is_dir()]
    check_members([i.filename for i in infos], commit, release)
    dist = f"vllm-{version(commit, release)}.dist-info"
    base = [t.encode() for t in FORBIDDEN]
    added = [t.encode() for t in forbid]
    project = project_table(engine)
    description = expected_description(project)
    requires = expected_requires(engine)
    problems = []
    digests: dict[str, tuple[str, int]] = {}
    record = b""
    with zipfile.ZipFile(wheel) as z, tempfile.TemporaryDirectory() as tmp:
        for info in infos:
            name = info.filename
            data = z.read(info)
            if name == f"{dist}/RECORD":
                # scanned like any text member (its rows carry every member's path); it has no row hash itself
                record = data
                problems += text_problems(name, data, base, added)
                continue
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            digests[name] = (digest.decode(), len(data))
            if name == f"{dist}/METADATA":
                problems += metadata_problems(
                    name, data, commit, project, description, requires, release
                )
            if name == f"{dist}/WHEEL":
                # the tags the installer trusts must be the ones the file name gives
                want = [wheel.name.removesuffix(".whl").split("-", 2)[2]]
                have = email.parser.BytesParser().parsebytes(data).get_all("Tag", [])
                if have != want:
                    problems.append(f"{name}: Tag is {have!r}, expected {want!r}")
            if not is_native(name):
                problems += text_problems(name, data, base, added)
                continue
            for needle in (*base, *added):
                if needle in data:
                    problems.append(
                        f"{name}: contains {needle.decode()!r} ({data.count(needle)}x)"
                    )
            so = Path(tmp) / "lib.so"
            so.write_bytes(data)
            for rp in runpaths(so):
                if any(p and not p.startswith("$ORIGIN") for p in rp.split(":")):
                    problems.append(f"{name}: RPATH/RUNPATH {rp}")
    problems += record_problems(f"{dist}/RECORD", record, digests)
    return problems


def git(engine: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(engine), *args], capture_output=True, text=True
    )


OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
TRACKED_PREFIX = "# tracked-files-v1 "


def read_tracked(native: Path) -> list[str] | None:
    """The exact tracked input list saved at setup, independent of Git ignore rules."""
    rows = [line[len(TRACKED_PREFIX):] for line in native.read_text(encoding="utf-8").splitlines()
            if line.startswith(TRACKED_PREFIX)]
    if not rows:
        return None  # snapshots made before tracked paths were recorded
    try:
        paths = json.loads(rows[0])
    except ValueError as error:
        raise WheelError(f"{native}: malformed tracked native file list") from error
    if len(rows) != 1 or not isinstance(paths, list) or not paths:
        raise WheelError(f"{native}: malformed tracked native file list")
    seen = set()
    for path in paths:
        if not isinstance(path, str):
            raise WheelError(f"{native}: malformed tracked native file path")
        parts = PurePosixPath(path)
        if (not path or parts.is_absolute() or ".." in parts.parts or "\\" in path
                or parts.as_posix() != path or path in seen
                or not any(path == root or path.startswith(root + "/") for root in NATIVE_SOURCES)):
            raise WheelError(f"{native}: invalid or duplicate tracked native file {path!r}")
        seen.add(path)
    for path in paths:
        if any(parent.as_posix() in seen for parent in PurePosixPath(path).parents):
            raise WheelError(f"{native}: overlapping tracked native file {path!r}")
    if not all(any(path == root or path.startswith(root + "/") for path in paths) for root in NATIVE_SOURCES):
        raise WheelError(f"{native}: tracked file list omits a native source path")
    return paths


def native_snapshot(engine: Path) -> str:
    """Keep the tracked list inside .build-native, so setup rollback restores both."""
    ids = native_ids(engine)
    cp = git(engine, "ls-tree", "-rz", "--name-only", "HEAD", "--", *NATIVE_SOURCES)
    if cp.returncode:
        raise WheelError(f"cannot list tracked native inputs: {cp.stderr.strip()}")
    paths = cp.stdout.rstrip("\0").split("\0")
    return ("".join(f"{digest}  {path}\n" for path, digest in ids.items())
            + TRACKED_PREFIX + json.dumps(paths, ensure_ascii=True) + "\n")


def read_native(native: Path) -> dict[str, str]:
    """{path: git object id} from release/merlin-wheel.native, one entry per NATIVE_SOURCES path."""
    if not native.is_file():
        raise WheelError(
            f"{native} is missing; it records the native source tree ids the wheel was built from"
        )
    ids: dict[str, str] = {}
    for n, line in enumerate(native.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith(TRACKED_PREFIX):
            continue
        fields = line.split()
        if len(fields) != 2 or not OBJECT_ID_RE.fullmatch(fields[0]):
            raise WheelError(f"{native}:{n} must be one '<git object id>  <path>' line")
        if fields[1] in ids:
            raise WheelError(f"{native}:{n} lists {fields[1]} twice")
        ids[fields[1]] = fields[0]
    missing = [p for p in NATIVE_SOURCES if p not in ids]
    extra = sorted(set(ids) - set(NATIVE_SOURCES))
    if missing or extra:
        raise WheelError(
            f"{native} must list exactly the native source paths {', '.join(NATIVE_SOURCES)}"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; unexpected {', '.join(extra)}" if extra else "")
        )
    read_tracked(native)  # a damaged snapshot must never fall back to legacy behavior
    return ids


def native_ids(engine: Path, ref: str = "HEAD") -> dict[str, str]:
    """The native input identities at a commit, independently of the wheel manifest."""
    ids = {}
    for path in NATIVE_SOURCES:
        cp = git(engine, "rev-parse", "--verify", "--quiet", f"{ref}:{path}")
        if cp.returncode or not OBJECT_ID_RE.fullmatch(cp.stdout.strip()):
            detail = cp.stderr.strip() or "git returned no valid object ID"
            raise WheelError(f"cannot read native input {path} at {ref} in {engine}: {detail}")
        ids[path] = cp.stdout.strip()
    return ids


def flashinfer_targets(variant: str) -> tuple[str, ...]:
    """Only files changed by the installed series, from its authenticated table."""
    series = runpy.run_path(str(Path(__file__).with_name("apply-flashinfer-workload-patch.py")))
    if variant not in ("v1", "v2"):
        raise WheelError(f"unknown FlashInfer patch variant: {variant}")
    return tuple(
        f"flashinfer/{name}"
        for name, before, after in zip(
            series["TARGETS"], series["STATE_SHA"]["pristine"], series["STATE_SHA"][variant]
        )
        if before != after
    )


def flashinfer_ids(purelib: Path, variant: str) -> dict[str, str]:
    """Hash the selected patch targets after install-time verification has passed."""
    return {path: sha256(purelib / path) for path in flashinfer_targets(variant)}


def check_native(engine: Path, native: Path, commit: str) -> None:
    """The checkout's native sources (at HEAD) must be the ones the wheel was built from."""
    want = read_native(native)
    for path in NATIVE_SOURCES:
        have = git(engine, "rev-parse", "--verify", "--quiet", f"HEAD:{path}")
        if have.returncode:
            raise WheelError(
                f"cannot read {path} at HEAD in {engine} (is it a git checkout with {path} committed?); "
                "build from source instead"
            )
        got = have.stdout.strip()
        if got != want[path]:
            raise WheelError(
                f"native sources differ from the ones wheel commit {commit} was built from: "
                f"{path} is {got} in this checkout, {want[path]} in {native.name}; "
                "build from source or use the wheel built from this checkout"
            )


def verify(wheel: Path, record: Path, engine: Path) -> str:
    if not record.is_file() or not record.read_text(encoding="utf-8").strip():
        raise WheelError(
            f"{record} records no release wheel for this checkout; "
            "build from source instead"
        )
    fields = record.read_text(encoding="utf-8").split()
    if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise WheelError(f"{record} must hold one '<sha256>  <filename>' line")
    want, name = fields
    if wheel.name != name:
        raise WheelError(f"wheel is {wheel.name}; this checkout accepts {name}")
    got = sha256(wheel)
    if got != want:
        raise WheelError(f"{wheel.name}: sha256 {got} does not match the recorded {want}")
    commit, release = parse_name(wheel)
    check_native(engine, record.with_name("merlin-wheel.native"), commit)
    check_members([i.filename for i in members(wheel)], commit, release)
    return commit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("wheel", type=Path)
    a.add_argument("--forbid", action="append", default=[])
    a.add_argument(
        "--engine", type=Path, required=True,
        help="the engine tree whose pyproject.toml and requirements the wheel's metadata must match",
    )
    v = sub.add_parser("verify")
    v.add_argument("wheel", type=Path)
    v.add_argument("record", type=Path)
    v.add_argument("engine", type=Path)
    n = sub.add_parser("native-ids")
    n.add_argument("engine", type=Path)
    snapshot = sub.add_parser("native-snapshot")
    snapshot.add_argument("engine", type=Path)
    f = sub.add_parser("flashinfer-ids")
    f.add_argument("purelib", type=Path)
    f.add_argument("variant", choices=("v1", "v2"))
    args = ap.parse_args(argv)
    try:
        if args.cmd == "audit":
            problems = audit(args.wheel, args.forbid, args.engine)
            for p in problems:
                print(f"wheel audit: {p}", file=sys.stderr)
            if problems:
                return 1
            print(f"WHEEL AUDIT OK: {args.wheel.name}")
        elif args.cmd == "native-snapshot":
            print(native_snapshot(args.engine), end="")
        elif args.cmd in ("native-ids", "flashinfer-ids"):
            ids = native_ids(args.engine) if args.cmd == "native-ids" else flashinfer_ids(args.purelib, args.variant)
            for path, digest in ids.items():
                print(f"{digest}  {path}")
        else:
            commit = verify(args.wheel, args.record, args.engine)
            print(f"WHEEL OK: {args.wheel.name} (commit {commit})")
    except (WheelError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as e:
        print(f"wheel: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""CLI for ast_distance — cross-language AST comparison and porting tools."""

from __future__ import annotations

import sys
import os
import stat
import time
import math
import fcntl
import subprocess
from pathlib import Path
from dataclasses import dataclass

from .ast_parser import ASTParser, Language
from .codebase import Codebase, CodebaseComparator, SourceFile
from .node_types import NodeType, node_type_name
from .porting_utils import PortingAnalyzer
from .similarity import ASTSimilarity
from .task_manager import TaskManager, TaskStatus, PortTask


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_language(s: str) -> Language:
    try:
        return Language(s)
    except ValueError:
        raise SystemExit(f"Unknown language: {s} (use rust, kotlin, cpp, or python)")


_LANG_NAMES = {
    Language.RUST: "Rust",
    Language.KOTLIN: "Kotlin",
    Language.CPP: "C++",
    Language.PYTHON: "Python",
}

_EXT_MAP: dict[str, str] = {
    ".rs": ".kt", ".kt": ".rs",
    ".cpp": ".py", ".hpp": ".py",
    ".py": ".cpp",
}


def _cpp_ast_distance_exe() -> str:
    # Repo-relative default when running from source checkout.
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "build" / "ast_distance"
    if candidate.exists():
        return str(candidate)
    return "ast_distance"


def _run_cpp_ast_distance(args: list[str]) -> int:
    exe = _cpp_ast_distance_exe()
    proc = subprocess.run([exe, *args], check=False)
    return proc.returncode


def _target_path_from_source(rel_path: str, src_lang: str, tgt_lang: str) -> str:
    """Generate expected target path from a source relative path."""
    p = Path(rel_path)
    ext_map = {
        ("rust", "kotlin"): ".kt",
        ("rust", "python"): ".py",
        ("kotlin", "rust"): ".rs",
        ("cpp", "python"): ".py",
        ("python", "cpp"): ".cpp",
    }
    new_ext = ext_map.get((src_lang, tgt_lang), p.suffix)

    # Kotlin filename convention for ports: snake_case Rust filenames become PascalCase Kotlin files.
    # Example: `typing/ty.rs` -> `typing/Ty.kt`, `my_file.rs` -> `MyFile.kt`.
    if src_lang == "rust" and tgt_lang == "kotlin":
        stem = p.stem
        out_name = "".join(
            (ch.upper() if i == 0 or stem[i - 1] == "_" else ch)
            for i, ch in enumerate(stem)
            if ch != "_"
        )
        p = p.with_name(out_name + new_ext)
        result = str(p)
    else:
        result = str(p.with_suffix(new_ext))

    if result.startswith("src/"):
        result = result[4:]
    return result


# ── Guardrails (Swarm Mode) ────────────────────────────────────────────────

@dataclass
class _GuardrailsContext:
    active: bool = False
    task_file: str = ""
    agent: int = 0
    override_mode: bool = False


_GUARD = _GuardrailsContext()


def _refuse_piped_stdio() -> None:
    """Refuse to run when stdout or stderr is piped via a shell pipeline.

    Piping has caused model-driven wrappers to truncate output silently.
    """

    def _is_fifo(fd: int) -> bool:
        try:
            return stat.S_ISFIFO(os.fstat(fd).st_mode)
        except OSError:
            return False

    out_fifo = _is_fifo(sys.stdout.fileno())
    err_fifo = _is_fifo(sys.stderr.fileno())
    if not (out_fifo or err_fifo):
        return

    def _detect_pipeline_peer_process() -> bool:
        # If ast_distance is part of a shell pipeline (cmd | other),
        # there will be a sibling process in our process group that's neither
        # our ancestor nor our child.
        #
        # We intentionally allow non-terminal stdout when the caller captures output
        # directly (e.g. a wrapper process reading from a pipe), because that is not
        # the same failure mode as shell filtering commands like `sed`/`grep`.
        self_pid = os.getpid()
        try:
            pgid = os.getpgrp()
        except Exception:
            return True

        try:
            proc = subprocess.run(
                ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "pgid="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return True
        if proc.returncode != 0:
            return True

        rows: list[tuple[int, int, int]] = []
        ppid_by_pid: dict[int, int] = {}
        for line in proc.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
                pg = int(parts[2])
            except ValueError:
                continue
            rows.append((pid, ppid, pg))
            ppid_by_pid[pid] = ppid

        ancestors: set[int] = set()
        cur = self_pid
        for _ in range(64):
            parent = ppid_by_pid.get(cur, 0)
            if parent <= 0:
                break
            if parent in ancestors:
                break
            ancestors.add(parent)
            cur = parent

        for pid, ppid, pg in rows:
            if pg != pgid:
                continue
            if pid == self_pid:
                continue
            if pid in ancestors:
                continue
            if ppid == self_pid:
                continue  # our own child
            return True

        return False

    if not _detect_pipeline_peer_process():
        return

    msg = (
        "Error: {which} is piped to another program.\n"
        "ast_distance does not support piping (|).\n"
        "Run it directly in a terminal.\n"
    )
    # Mirror the C++ behavior: write to the opposite stream when possible.
    if out_fifo and not err_fifo:
        os.write(sys.stderr.fileno(), msg.format(which="stdout").encode())
    elif err_fifo and not out_fifo:
        os.write(sys.stdout.fileno(), msg.format(which="stderr").encode())
    else:
        os.write(sys.stderr.fileno(), msg.format(which="stdout/stderr").encode())
    raise SystemExit(2)


def _agents_dir(task_file: str) -> Path:
    base = Path(task_file).resolve().parent
    if not str(base):
        base = Path.cwd()
    return base / ".cache" / "ast_distance" / "agents"


def _agent_lock_path(task_file: str, agent: int) -> Path:
    return _agents_dir(task_file) / f"agent_{agent}.lock"


def _agent_notice_path(task_file: str, agent: int) -> Path:
    return _agents_dir(task_file) / f"agent_{agent}.notice"


def _agent_state_path(task_file: str, agent: int) -> Path:
    return _agents_dir(task_file) / f"agent_{agent}.state"


@dataclass
class _AgentState:
    last_used_epoch: int = 0
    last_score_epoch: int = 0
    last_score: float = -1.0  # -1 = unknown


def _load_agent_state(path: Path) -> _AgentState:
    st = _AgentState()
    try:
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "last_used_epoch":
                st.last_used_epoch = int(float(v))
            elif k == "last_score_epoch":
                st.last_score_epoch = int(float(v))
            elif k == "last_score":
                st.last_score = float(v)
    except OSError:
        pass
    return st


def _save_agent_state(path: Path, st: _AgentState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"last_used_epoch={st.last_used_epoch}\n"
        f"last_score_epoch={st.last_score_epoch}\n"
        f"last_score={st.last_score}\n"
    )


def _now_epoch_seconds() -> int:
    return int(time.time())


def _activity_score_from_idle_minutes(idle_minutes: int) -> float:
    if idle_minutes <= 15:
        return 100.0
    if idle_minutes >= 60:
        return 0.0
    t = (idle_minutes - 15) / 45.0
    return 100.0 * (1.0 - t)


def _format_idle_minutes(idle_minutes: int) -> str:
    if idle_minutes < 0:
        idle_minutes = 0
    h = idle_minutes // 60
    m = idle_minutes % 60
    if h == 0:
        return f"{m}m"
    return f"{h}h{m:02d}m"


class _AgentLock:
    def __init__(self, task_file: str, agent: int, override_wait: bool) -> None:
        self._fd: int | None = None
        self._locked = False
        self._holder_pid = 0
        self._path = _agent_lock_path(task_file, agent)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o666)

        flags = fcntl.LOCK_EX
        if not override_wait:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(self._fd, flags)
            self._locked = True
            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, str(os.getpid()).encode())
            os.fsync(self._fd)
        except BlockingIOError:
            self._locked = False
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                data = os.read(self._fd, 64).decode(errors="replace").strip()
                self._holder_pid = int(data) if data.isdigit() else 0
            except OSError:
                self._holder_pid = 0

    def locked(self) -> bool:
        return self._locked

    def holder_pid(self) -> int:
        return self._holder_pid

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            if self._locked:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            self._locked = False

    def __enter__(self) -> _AgentLock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _write_agent_notice(task_file: str, agent: int, msg: str) -> None:
    p = _agent_notice_path(task_file, agent)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def _print_and_clear_agent_notice(task_file: str, agent: int) -> None:
    p = _agent_notice_path(task_file, agent)
    try:
        s = p.read_text()
    except OSError:
        return
    if s.strip():
        print(f"\n=== NOTICE (Agent #{agent}) ===\n", file=sys.stderr)
        print(s.rstrip() + "\n", file=sys.stderr)
    try:
        p.unlink()
    except OSError:
        pass


def _touch_agent_last_used(task_file: str, agent: int) -> None:
    st = _load_agent_state(_agent_state_path(task_file, agent))
    st.last_used_epoch = _now_epoch_seconds()
    _save_agent_state(_agent_state_path(task_file, agent), st)


def _print_agent_activity_section(tm: TaskManager, task_file: str, current_agent: int) -> None:
    print("\n=== Agent Activity ===\n")
    print(f"You are agent #{current_agent}\n")

    now = _now_epoch_seconds()

    rows: list[dict] = []
    for t in tm.tasks:
        if t.status != TaskStatus.ASSIGNED:
            continue
        try:
            agent = int(t.assigned_to)
        except Exception:
            continue

        st = _load_agent_state(_agent_state_path(task_file, agent))
        idle_s = (now - st.last_used_epoch) if st.last_used_epoch > 0 else 60 * 60
        idle_min = max(0, int((idle_s + 30) // 60))
        score = _activity_score_from_idle_minutes(idle_min)

        trend = 0.0
        has_trend = False
        if st.last_score >= 0.0:
            trend = score - st.last_score
            has_trend = True

        st.last_score = score
        st.last_score_epoch = now
        _save_agent_state(_agent_state_path(task_file, agent), st)

        rows.append(
            {
                "agent": agent,
                "task": t.source_qualified,
                "idle_min": idle_min,
                "score": score,
                "trend": trend,
                "has_trend": has_trend,
            }
        )

    if not rows:
        print("No agents currently assigned.")
        return

    rows.sort(key=lambda r: r["agent"])

    print(f"{'Agent':<8}{'Working On':<34}{'Idle':<10}{'Active':<10}Trend")
    print("-" * 76)
    for r in rows:
        d = int(round(r["trend"])) if r["has_trend"] else None
        trend_s = "-" if d is None else (f"+{d}" if d > 0 else str(d))
        print(
            f"#{r['agent']:<7}"
            f"{r['task'][:32]:<34}"
            f"{_format_idle_minutes(r['idle_min']):<10}"
            f"{int(round(r['score'])):>3}%{'':<6}"
            f"{trend_s}"
        )


def _parse_global_flags(args: list[str]) -> tuple[int, str, bool, list[str]]:
    """Parse global guardrail flags and return (agent, task_file, override, remaining_args)."""
    agent = 0
    task_file = ""
    override_mode = False

    rest: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--agent" and i + 1 < len(args):
            agent = int(args[i + 1])
            i += 2
        elif a == "--task-file" and i + 1 < len(args):
            task_file = args[i + 1]
            i += 2
        elif a == "--override":
            override_mode = True
            i += 1
        else:
            rest.append(a)
            i += 1
    return agent, task_file, override_mode, rest


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_compare(file1: str, lang1: str, file2: str, lang2: str) -> None:
    """Compare two files and print AST similarity report."""
    parser = ASTParser()
    l1 = _parse_language(lang1)
    l2 = _parse_language(lang2)

    def _file_contains_macro_rules(filepath: str) -> bool:
        try:
            return "macro_rules!" in Path(filepath).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    macro_friendly = ((l1 == Language.RUST and _file_contains_macro_rules(file1)) or
                      (l2 == Language.RUST and _file_contains_macro_rules(file2)))

    print(f"Parsing {_LANG_NAMES[l1]} file: {file1}")
    tree1 = parser.parse_file(file1, l1)

    print(f"Parsing {_LANG_NAMES[l2]} file: {file2}")
    tree2 = parser.parse_file(file2, l2)

    report = ASTSimilarity.compare(tree1, tree2, macro_friendly=macro_friendly)
    print()
    print(f"=== AST Comparison Report ===")
    print(f"Histogram cosine: {report['cosine_sim']:.4f}")
    print(f"Jaccard:          {report['jaccard_sim']:.4f}")
    print(f"Structure:        {report['structure_sim']:.4f}")
    print(f"Edit distance:    {report['edit_distance_sim']:.4f}")
    print(f"Combined:         {report['combined_score']:.4f}")

    ids1 = parser.extract_identifiers_from_file(file1, l1)
    ids2 = parser.extract_identifiers_from_file(file2, l2)
    content_score = ASTSimilarity.combined_similarity_with_content(tree1, tree2, ids1, ids2)

    print("\n=== Identifier Content Analysis ===")
    print(f"Identifiers:          {ids1.total_identifiers} vs {ids2.total_identifiers}")
    print(f"Unique (raw):         {len(ids1.identifier_freq)} vs {len(ids2.identifier_freq)}")
    print(f"Unique (canonical):   {len(ids1.canonical_freq)} vs {len(ids2.canonical_freq)}")
    print(f"Raw cosine:           {ids1.identifier_cosine_similarity(ids2):.4f}")
    print(f"Canonical cosine:     {ids1.canonical_cosine_similarity(ids2):.4f}")
    print(f"Canonical jaccard:    {ids1.canonical_jaccard_similarity(ids2):.4f}")

    file1_stubs = parser.has_stub_bodies_in_files([file1], l1)
    file2_stubs = parser.has_stub_bodies_in_files([file2], l2)
    # IMPORTANT: stubs in the *target* indicate incomplete transliteration and should gate completion.
    # Stubs in the *source* are treated as baseline and should not force similarity to zero.
    if file2_stubs:
        content_score = 0.0
        print("\n*** STUB DETECTED ***")
        if file1_stubs:
            print(f"  {file1} has TODO/stub/placeholder in function bodies")
        print(f"  {file2} has TODO/stub/placeholder in function bodies")
        print("  Content-Aware Score forced to 0.0000")
    else:
        if file1_stubs:
            print(f"\nNote: {file1} contains TODO/stub/placeholder markers in function bodies (source baseline).")
        print(f"\nContent-Aware Score:  {content_score:.4f}")

    num_types = int(NodeType.NUM_TYPES)
    if tree1:
        print(f"\n=== {_LANG_NAMES[l1]} AST Histogram ===")
        _print_histogram(tree1.node_type_histogram(num_types))
    if tree2:
        print(f"\n=== {_LANG_NAMES[l2]} AST Histogram ===")
        _print_histogram(tree2.node_type_histogram(num_types))

    # Comments comparison
    comments1 = parser.extract_comments_from_file(file1, l1)
    comments2 = parser.extract_comments_from_file(file2, l2)

    print(f"\n=== Documentation Comparison ===")
    print(f"Doc comment count: {comments1.doc_comment_count} vs {comments2.doc_comment_count}")
    print(f"Doc lines:         {comments1.total_doc_lines} vs {comments2.total_doc_lines}")

    doc_cosine = comments1.doc_cosine_similarity(comments2)
    doc_jaccard = comments1.doc_jaccard_similarity(comments2)
    doc_line_cov = comments1.doc_line_coverage_capped(comments2)
    doc_line_bal = comments1.doc_line_balance(comments2)
    doc_weighted = 0.5 * doc_cosine + 0.5 * doc_line_cov
    print(f"Doc text cosine:  {doc_cosine * 100:.0f}%")
    print(f"Doc text jaccard: {doc_jaccard * 100:.0f}%")
    print(f"Doc line coverage: {doc_line_cov * 100:.0f}%")
    print(f"Doc line balance:  {doc_line_bal * 100:.0f}%")
    print(f"Doc weighted (eq): {doc_weighted * 100:.0f}%")


def cmd_dump(filepath: str, lang_str: str) -> None:
    """Dump AST structure of a file."""
    parser = ASTParser()
    lang = _parse_language(lang_str)

    print(f"Parsing {filepath} as {lang_str}...\n")
    tree = parser.parse_file(filepath, lang)
    if tree is None:
        raise SystemExit(f"Failed to parse {filepath}")

    print("AST Structure:")
    _print_tree(tree)

    print()
    hist = tree.node_type_histogram(int(NodeType.NUM_TYPES))
    _print_histogram(hist)

    print(f"\nTree Statistics:")
    print(f"  Size:  {tree.size()} nodes")
    print(f"  Depth: {tree.depth()}")


def cmd_scan(directory: str, lang: str) -> None:
    """Scan directory and show file list with import counts."""
    cb = Codebase(directory, lang)
    cb.scan()
    cb.extract_imports()

    print(f"=== Scanned {len(cb.files)} {lang} files ===\n")
    print(f"{'Qualified Name':<40} {'Imports':<8} Path")
    print("-" * 80)
    for path, sf in cb.files.items():
        print(f"{sf.qualified_name[:38]:<40} {len(sf.imports):<8} {sf.relative_path}")


def cmd_deps(directory: str, lang: str) -> None:
    """Build and show dependency graph."""
    cb = Codebase(directory, lang)
    cb.scan()
    cb.extract_imports()
    cb.build_dependency_graph()

    print(cb.print_summary())

    print("\n=== Files by Dependent Count ===\n")
    print(f"{'File':<40} {'Deps':<10} {'DepBy':<10} Status")
    print("-" * 70)

    for sf in cb.ranked_by_dependents():
        status = ""
        if sf.dependent_count >= 5:
            status = "CORE"
        elif sf.dependent_count == 0:
            status = "leaf"
        print(f"{sf.qualified_name[:38]:<40} {sf.dependency_count:<10} {sf.dependent_count:<10} {status}")

    roots = cb.root_files(3)
    if roots:
        print("\n=== Core Files (most dependents) ===")
        for sf in roots:
            print(f"\n{sf.qualified_name} ({sf.dependent_count} dependents):")
            print("  Imported by:")
            for i, dep in enumerate(sf.imported_by):
                if i >= 5:
                    print(f"    ... and {len(sf.imported_by) - 5} more")
                    break
                print(f"    - {cb.files[dep].qualified_name}")


def cmd_rank(src_dir: str, src_lang: str, tgt_dir: str, tgt_lang: str) -> None:
    """Rank files by porting priority (dependents + similarity)."""
    source = _scan_codebase(src_dir, src_lang, build_deps=True)
    target = _scan_codebase(tgt_dir, tgt_lang, build_deps=True)

    comp = CodebaseComparator(source, target)
    comp.find_matches()
    comp.compute_similarities()
    print(comp.format_report())


def cmd_deep(src_dir: str, src_lang: str, tgt_dir: str, tgt_lang: str) -> dict:
    """Full analysis: AST + deps + TODOs + lint + line ratios.

    Returns structured metrics dict in addition to printing human-readable output.
    """
    print(f"=== Deep Analysis: {src_dir} ({src_lang}) -> {tgt_dir} ({tgt_lang}) ===\n")

    print(f"Scanning source codebase ({src_lang})...")
    source = _scan_codebase(src_dir, src_lang, build_deps=True)
    print(source.print_summary())

    print(f"Scanning target codebase ({tgt_lang})...")
    target = _scan_codebase(tgt_dir, tgt_lang, build_deps=True, porting_data=True)
    print(target.print_summary())

    print("Comparing codebases...")
    comp = CodebaseComparator(source, target)
    comp.find_matches()

    print("Computing AST similarities...")
    comp.compute_similarities()

    # Agent-scoped dashboard (guardrails): when a task system is initialized, lock the
    # dashboard view to the assigned file and its direct imports.
    if _GUARD.active and _GUARD.agent > 0 and not _GUARD.override_mode:
        tm = TaskManager(_GUARD.task_file)
        if tm.load():
            _print_agent_activity_section(tm, _GUARD.task_file, _GUARD.agent)

            agent_id = str(_GUARD.agent)
            focus = next(
                (t for t in tm.tasks if t.status == TaskStatus.ASSIGNED and t.assigned_to == agent_id),
                None,
            )

            print("\n=== Agent Scope Dashboard ===\n")
            print(f"Task file: {_GUARD.task_file}")
            print(f"You are agent #{_GUARD.agent}\n")

            if focus is None:
                print(f"No task assigned to agent #{_GUARD.agent}.")
                print(f"Run: ast_distance --assign {_GUARD.task_file} {_GUARD.agent}\n")
                print("Note: full project dashboard is locked in agent scope.")
                print("Use --override if you need full-project output.")
                return {"error": "no_task_assigned", "agent": _GUARD.agent, "task_file": _GUARD.task_file}

            # Build relative_path -> logical key maps for locating the focus file.
            src_rel_to_key = {sf.relative_path: key for key, sf in source.files.items()}
            tgt_rel_to_key = {sf.relative_path: key for key, sf in target.files.items()}

            focus_src_key = src_rel_to_key.get(focus.source_path, "")
            focus_tgt_key = tgt_rel_to_key.get(focus.target_path, "")

            focus_source_path = Path(tm.source_root) / focus.source_path
            focus_target_path = Path(tm.target_root) / focus.target_path

            print(f"Assigned task: {focus.source_qualified}")
            print(f"Source: {focus_source_path}{' (missing?)' if not focus_source_path.exists() else ''}")
            print(f"Target: {focus_target_path} ({'exists' if focus_target_path.exists() else 'missing'})\n")

            scope_source_keys: set[str] = set()
            scope_target_keys: set[str] = set()
            if focus_src_key:
                scope_source_keys.add(focus_src_key)
                scope_source_keys |= set(source.files[focus_src_key].depends_on)
            if focus_tgt_key:
                scope_target_keys.add(focus_tgt_key)
                scope_target_keys |= set(target.files[focus_tgt_key].depends_on)

            print(
                f"Scope: {len(scope_source_keys)} source files (focus + imports), "
                f"{len(scope_target_keys)} target files (focus + imports)"
            )

            assigned_to_agent: dict[str, int] = {}
            for t in tm.tasks:
                if t.status != TaskStatus.ASSIGNED:
                    continue
                try:
                    assigned_to_agent[t.source_qualified] = int(t.assigned_to)
                except Exception:
                    continue

            blocked_by: dict[str, int] = {}
            for key in scope_source_keys:
                if key == focus_src_key:
                    continue
                sf = source.files[key]
                other = assigned_to_agent.get(sf.qualified_name)
                if other is not None and other != _GUARD.agent:
                    blocked_by[key] = other

            if blocked_by:
                print("\nBlocked imports (assigned to other agents):")
                for key, other in sorted(blocked_by.items(), key=lambda kv: kv[1]):
                    print(f"  - {source.files[key].qualified_name} (agent #{other})")

            match_by_source = {m.source_path: m for m in comp.matches}
            unmatched_source = set(comp.unmatched_source)

            def _print_row(key: str, is_focus: bool) -> None:
                sf = source.files[key]
                m = match_by_source.get(key)
                missing = key in unmatched_source

                status: list[str] = []
                if is_focus:
                    status.append("FOCUS")
                if missing:
                    status.append("MISSING_FILE")
                if m is not None and m.is_stub:
                    status.append("STUB(FAIL)")
                if m is not None and m.todo_count > 0:
                    status.append("TODO(FAIL)")
                if key in blocked_by:
                    status.append(f"BLOCKED(agent #{blocked_by[key]})")

                sim = "-" if (m is None or missing) else f"{m.similarity:.2f}"
                todos = 0 if m is None else m.todo_count
                lint = 0 if m is None else m.lint_count

                print(
                    f"{sf.qualified_name[:28]:<30}"
                    f"{sf.dependent_count:<11}"
                    f"{sim:<11}"
                    f"{todos:<8}"
                    f"{lint:<6}"
                    f"{','.join(status)}"
                )

            print("\n=== Scope Work Items ===\n")
            print(f"{'File':<30}{'Dependents':<11}{'Similarity':<11}{'TODOs':<8}{'Lint':<6}Status")
            print("-" * 90)

            if focus_src_key:
                _print_row(focus_src_key, True)
            else:
                print(
                    f"{focus.source_qualified[:28]:<30}"
                    f"{focus.dependent_count:<11}"
                    f"{'-':<11}"
                    f"{0:<8}"
                    f"{0:<6}"
                    f"FOCUS,UNKNOWN_SOURCE_PATH"
                )

            others = [k for k in scope_source_keys if k and k != focus_src_key]
            others.sort(key=lambda k: source.files[k].dependent_count, reverse=True)
            for k in others:
                _print_row(k, False)

            print("\n=== Next Action (Focus) ===\n")
            focus_match = match_by_source.get(focus_src_key) if focus_src_key else None
            if focus_match is None or focus_src_key in unmatched_source:
                print("Target file is missing. Create it and port the full implementation.")
            elif focus_match.is_stub:
                print("Replace the stub with a real implementation (stubs are a failure mode).")
            elif focus_match.todo_count > 0:
                print("Remove TODOs and finish the implementation (TODOs are a failure mode).")
            elif focus_match.similarity < 0.85:
                print("Improve similarity to >= 0.85 (whole-file + identifier-content + parity).")
            else:
                print("Looks complete. If you have validated behavior and tests, mark it complete.")

            print("\nMark complete with:")
            print(f"  ast_distance --agent {_GUARD.agent} --complete {_GUARD.task_file} {focus.source_qualified}")
            print("\nNote: full project dashboard is locked in agent scope.")
            print("Use --override to view the full project view.")
            return {"mode": "agent_scope", "agent": _GUARD.agent, "focus": focus.source_qualified}

    print(comp.format_report())

    # Porting quality summary
    print("\n=== Porting Quality Summary ===\n")
    total_todos = sum(m.todo_count for m in comp.matches)
    total_lint = sum(m.lint_count for m in comp.matches)
    stub_count = sum(1 for m in comp.matches if m.is_stub)
    header_matched = sum(1 for m in comp.matches if m.matched_by_header)
    total_matched = len(comp.matches)
    ranked = comp.ranked_for_porting()
    incomplete = sum(1 for m in ranked if m.similarity < 0.6)
    missing_count = len(comp.unmatched_source)

    print(f"Matched by header:     {header_matched} / {total_matched}")
    print(f"Matched by name:       {total_matched - header_matched} / {total_matched}")
    print(f"Total TODOs in target: {total_todos}")
    print(f"Total lint errors:     {total_lint}")
    print(f"Stub files:            {stub_count}")

    # Files with issues
    issues = [m for m in ranked if m.todo_count > 0 or m.lint_count > 0 or m.is_stub or m.similarity < 0.6]
    if issues:
        print("\n=== Files with Issues ===\n")
        print(f"{'File':<30} {'Sim':>6} {'Ratio':>6} {'TODOs':>5} {'Lint':>5} Status")
        print("-" * 70)
        for m in issues[:20]:
            status = ""
            if m.is_stub:
                status = "STUB"
            elif m.similarity < 0.4:
                status = "LOW_SIM"
            elif m.lint_count > 0:
                status = "LINT"
            elif m.todo_count > 0:
                status = "TODO"
            ratio = m.target_lines / m.source_lines if m.source_lines > 0 else 0.0
            print(f"{m.target_qualified[:28]:<30} {m.similarity:>5.2f} {ratio:>6.2f} {m.todo_count:>5} {m.lint_count:>5} {status}")
        if len(issues) > 20:
            print(f"... and {len(issues) - 20} more files")

    # Recommendations
    print("\n=== Porting Recommendations ===\n")
    print(f"Incomplete ports (similarity < 60%): {incomplete}")
    print(f"Missing files: {missing_count}")

    if incomplete > 0:
        print("\nTop priority to complete:")
        for m in [m for m in ranked if m.similarity < 0.6][:10]:
            flags = []
            if m.is_stub:
                flags.append("[STUB]")
            if m.todo_count > 0:
                flags.append(f"[{m.todo_count} TODOs]")
            print(f"  {m.source_qualified:<30} sim={m.similarity:.2f} deps={m.source_dependents} {' '.join(flags)}")

    if comp.unmatched_source:
        print("\nTop priority to create:")
        missing_sorted = sorted(
            [source.files[p] for p in comp.unmatched_source],
            key=lambda s: s.dependent_count, reverse=True,
        )
        for sf in missing_sorted[:10]:
            print(f"  {sf.qualified_name:<30} deps={sf.dependent_count}")

    # Documentation gaps
    doc_gaps = [
        (m.doc_gap_ratio(), m)
        for m in comp.matches
        if m.doc_gap_ratio() > 0.2 and m.source_doc_lines > 5
    ]
    doc_gaps.sort(key=lambda x: x[0] * x[1].source_doc_lines, reverse=True)

    total_src_doc = sum(m.source_doc_lines for m in comp.matches)
    total_tgt_doc = sum(m.target_doc_lines for m in comp.matches)

    print("\n=== Documentation Gaps ===\n")
    pct = (100.0 * total_tgt_doc / total_src_doc) if total_src_doc > 0 else None
    docs_missing = pct is not None and pct < 85.0
    if docs_missing:
        print("There is missing documentation that is hurting overall scoring.")
    if total_src_doc > 0 and pct is not None:
        print(f"Documentation coverage: {total_tgt_doc} / {total_src_doc} lines ({pct:.0f}%)")
    else:
        print("Documentation coverage: N/A (source has no docs)")
    print(f"Files with >20% doc gap: {len(doc_gaps)}\n")

    if doc_gaps:
        print(f"{'File':<30} {'Src Docs':>10} {'Tgt Docs':>10} {'Gap %':>8} {'DocSim':>8} {'DocAmt':>8} {'DocEq':>8}")
        print("-" * 98)
        for gap, m in doc_gaps[:25]:
            print(
                f"{m.source_qualified[:28]:<30} {m.source_doc_lines:>10} {m.target_doc_lines:>10} "
                f"{int(gap * 100):>7}% {m.doc_similarity:>7.2f} {m.doc_coverage:>7.2f} {m.doc_weighted:>7.2f}"
            )
        if len(doc_gaps) > 25:
            print(f"... and {len(doc_gaps) - 25} more files with doc gaps")
    else:
        print("No significant documentation gaps found.")

    # Return structured metrics for programmatic use.
    hdr_ratio = header_matched / total_matched if total_matched > 0 else 0.0
    return {
        "matched_by_header": {"count": header_matched, "total": total_matched},
        "matched_by_name": {"count": total_matched - header_matched, "total": total_matched},
        "matched_by_header_ratio": hdr_ratio,
        "total_todos": total_todos,
        "total_lint_errors": total_lint,
        "stub_files": stub_count,
        "missing_files": missing_count,
        "incomplete_ports": incomplete,
    }


def cmd_missing(src_dir: str, src_lang: str, tgt_dir: str, tgt_lang: str) -> dict:
    """Show files missing from target, ranked by importance.

    Returns dict with missing file list for programmatic use.
    """
    source = _scan_codebase(src_dir, src_lang, build_deps=True)
    target = Codebase(tgt_dir, tgt_lang)
    target.scan()

    comp = CodebaseComparator(source, target)
    comp.find_matches()

    print(f"=== Missing from {tgt_lang} (ranked by dependents) ===\n")
    print(f"{'Source File':<40} {'Deps':<10} Path")
    print("-" * 80)

    missing = sorted(
        [source.files[p] for p in comp.unmatched_source],
        key=lambda s: s.dependent_count, reverse=True,
    )
    for sf in missing:
        print(f"{sf.qualified_name[:38]:<40} {sf.dependent_count:<10} {sf.relative_path}")
    print(f"\nTotal: {len(missing)} files missing")

    return {
        "missing": [
            {"qualified_name": sf.qualified_name, "path": sf.relative_path,
             "dependent_count": sf.dependent_count}
            for sf in missing
        ],
        "total": len(missing),
    }


def cmd_todos(directory: str, verbose: bool = True) -> None:
    """Scan for TODO comments."""
    print(f"Scanning for TODOs in: {directory}\n")
    all_todos = []
    for entry in Path(directory).rglob("*"):
        if entry.is_file() and entry.suffix in (".rs", ".kt", ".kts", ".cpp", ".hpp", ".h", ".py"):
            todos = PortingAnalyzer.scan_todos(entry)
            all_todos.extend(todos)
    print(f"Found {len(all_todos)} TODOs")
    if verbose:
        for todo in all_todos:
            print(f"\n  {todo.file_path}:{todo.line_num}")
            if todo.tag:
                print(f"  Tag: {todo.tag}")
            print(f"  {todo.message}")
            for ctx in todo.context:
                print(f"  {ctx}")


def cmd_lint(directory: str) -> None:
    """Run lint checks."""
    print(f"Running lint checks on: {directory}\n")
    lint_errors = []
    for entry in Path(directory).rglob("*"):
        if entry.is_file() and entry.suffix in (".rs", ".kt", ".kts", ".cpp", ".hpp", ".h", ".py"):
            lint_errors.extend(PortingAnalyzer.lint_file(entry))

    if not lint_errors:
        print("No lint errors found.")
        return

    print(f"Found {len(lint_errors)} lint error(s):")
    for err in lint_errors:
        print(f"  {err.file_path}:{err.line_num} [{err.type}] {err.message}")


def _proxy_to_cpp(args: list[str]) -> None:
    rc = _run_cpp_ast_distance(args)
    if rc != 0:
        raise SystemExit(rc)


def cmd_stats(directory: str) -> None:
    """Show file statistics."""
    print(f"=== File Statistics: {directory} ===\n")
    print(f"{'File':<40} {'Lines':>6} {'Code':>6} {'Stub':>5} {'TODOs':>6}")
    print("-" * 70)
    for entry in sorted(Path(directory).rglob("*")):
        if entry.is_file() and entry.suffix in (".rs", ".kt", ".kts", ".cpp", ".hpp", ".h", ".py"):
            stats = PortingAnalyzer.analyze_file(entry)
            stub_str = "STUB" if stats.is_stub else ""
            print(f"{entry.name:<40} {stats.line_count:>6} {stats.code_lines:>6} {stub_str:>5} {len(stats.todos):>6}")


def cmd_init_tasks(src_dir: str, src_lang: str, tgt_dir: str, tgt_lang: str,
                   task_file: str, agents_md: str = "") -> dict:
    """Generate task file from missing/incomplete ports.

    Returns dict with task count and top priorities for programmatic use.
    """
    print("=== Initializing Task File ===\n")

    source = _scan_codebase(src_dir, src_lang, build_deps=True)
    target = _scan_codebase(tgt_dir, tgt_lang, build_deps=True, porting_data=True)

    comp = CodebaseComparator(source, target)
    comp.find_matches()
    print("Computing AST similarities...")
    comp.compute_similarities()

    tm = TaskManager(task_file)
    tm.source_root = src_dir
    tm.target_root = tgt_dir
    tm.source_lang = src_lang
    tm.target_lang = tgt_lang
    tm.agents_md_path = agents_md

    # Collect pending files: missing + incomplete or invalid ports.
    # NOTE: TODOs/stubs are a failure mode for "complete" status, so they must stay in the queue.
    pending_keys: set[str] = set(comp.unmatched_source)
    for m in comp.matches:
        if m.similarity < 0.85 or m.todo_count > 0 or m.lint_count > 0 or m.is_stub:
            pending_keys.add(m.source_path)

    pending_files = [source.files[k] for k in pending_keys]
    pending_files.sort(key=lambda s: s.dependent_count, reverse=True)

    for sf in pending_files:
        task = PortTask(
            source_path=sf.relative_path,
            source_qualified=sf.qualified_name,
            target_path=_target_path_from_source(sf.relative_path, src_lang, tgt_lang),
            dependent_count=sf.dependent_count,
            dependency_count=sf.dependency_count,
            dependencies=[imp.module_path for imp in sf.imports],
        )
        tm.tasks.append(task)

    if not tm.save():
        raise SystemExit(f"Failed to write task file: {task_file}")

    print(f"Generated {len(tm.tasks)} tasks")
    print(f"Task file: {task_file}")

    print("\nTop 10 priority tasks:")
    for t in tm.tasks[:10]:
        print(f"  {t.source_qualified:<30} deps={t.dependent_count}")

    return {
        "task_count": len(tm.tasks),
        "task_file": task_file,
        "top_tasks": [
            {"source_qualified": t.source_qualified, "dependent_count": t.dependent_count}
            for t in tm.tasks[:10]
        ],
    }


def cmd_tasks(task_file: str, agent: int) -> None:
    """Show task status summary."""
    tm = TaskManager(task_file)
    if not tm.load():
        raise SystemExit(f"Error: Could not load task file: {task_file}")

    stats = tm.get_stats()

    _print_agent_activity_section(tm, task_file, agent)

    print("=== Task Status ===\n")
    print(f"Task file: {task_file}")
    print(f"Source root: {tm.source_root}")
    print(f"Target root: {tm.target_root}\n")
    print(f"Status Summary:")
    print(f"  Pending:   {stats['pending']}")
    print(f"  Assigned:  {stats['assigned']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Blocked:   {stats['blocked']}")
    print(f"  Total:     {len(tm.tasks)}\n")

    assigned = [t for t in tm.tasks if t.status == TaskStatus.ASSIGNED]
    if assigned:
        print("Currently Assigned:")
        for t in assigned:
            print(f"  {t.source_qualified:<30} -> {t.assigned_to} (since {t.assigned_at})")
        print()

    print("Pending Tasks (by priority):")
    print(f"{'Source':<35} {'Deps':<10} Target Path")
    print("-" * 70)
    pending = [t for t in tm.tasks if t.status == TaskStatus.PENDING]
    for t in pending[:20]:
        print(f"{t.source_qualified[:33]:<35} {t.dependent_count:<10} {t.target_path}")
    if len(pending) > 20:
        print(f"... and {len(pending) - 20} more")


def cmd_assign(task_file: str, requested_agent: int, override_mode: bool) -> str:
    """Assign highest-priority pending task to an agent session.

    If requested_agent <= 0, allocates a new monotonic agent number.
    Returns the assignment instructions text for programmatic use.
    """
    tm = TaskManager(task_file)
    if not tm.load():
        raise SystemExit(f"Error: Could not load task file: {task_file}")

    agent = requested_agent
    if agent <= 0:
        # Allocate a new agent number (monotonic) to avoid collisions/reuse confusion.
        max_agent = 0
        for t in tm.tasks:
            if t.status != TaskStatus.ASSIGNED:
                continue
            try:
                max_agent = max(max_agent, int(t.assigned_to))
            except Exception:
                pass

        dirp = _agents_dir(task_file)
        if dirp.exists():
            for entry in dirp.iterdir():
                name = entry.name
                if not name.startswith("agent_"):
                    continue
                num_str = name[len("agent_") :].split(".", 1)[0]
                if num_str.isdigit():
                    max_agent = max(max_agent, int(num_str))

        agent = max_agent + 1

    if agent <= 0:
        raise SystemExit("Error: Could not allocate agent number")

    with _AgentLock(task_file, agent, override_mode) as agent_lock:
        if not agent_lock.locked():
            holder = agent_lock.holder_pid()
            msg = f"Agent #{agent} appears to be in use"
            if holder > 0:
                msg += f" by pid {holder}"
            msg += ".\nRe-run with --override to wait, or pick a different agent number.\n"
            _write_agent_notice(
                task_file,
                agent,
                "Conflict: another PID attempted to use this agent number.\n" + msg,
            )
            print("Error: " + msg, file=sys.stderr)
            raise SystemExit(3)

        _print_and_clear_agent_notice(task_file, agent)
        _touch_agent_last_used(task_file, agent)

        agent_id = str(agent)

        # Check if agent already has an assigned task
        for t in tm.tasks:
            if t.status == TaskStatus.ASSIGNED and t.assigned_to == agent_id:
                print(
                    f"Agent #{agent} already has an assigned task: {t.source_qualified}",
                    file=sys.stderr,
                )
                print(
                    f"Complete it with: ast_distance --agent {agent} --complete {task_file} {t.source_qualified}",
                    file=sys.stderr,
                )
                print(
                    f"Or release it with: ast_distance --agent {agent} --release {task_file} {t.source_qualified}",
                    file=sys.stderr,
                )
                return ""

        task = tm.assign_next(agent_id)
        if task is None:
            msg = "No pending tasks available."
            stats = tm.get_stats()
            msg += (
                f"\nStatus: {stats['completed']}/{len(tm.tasks)} completed, "
                f"{stats['assigned']} assigned, {stats['pending']} pending"
            )
            print(msg)
            return msg

        _print_agent_activity_section(tm, task_file, agent)

        text = tm.format_assignment(task, agent)
        print(text)
        return text


def cmd_complete(task_file: str, source_qualified: str, agent: int, override_mode: bool) -> None:
    """Mark a task as completed, with strict anti-stub guardrails."""
    tm = TaskManager(task_file)
    if not tm.load():
        raise SystemExit(f"Error: Could not load task file: {task_file}")

    task = next((t for t in tm.tasks if t.source_qualified == source_qualified), None)
    if task is None:
        raise SystemExit(f"Task not found: {source_qualified}")

    if agent > 0 and task.status == TaskStatus.ASSIGNED and task.assigned_to != str(agent) and not override_mode:
        raise SystemExit(
            f"Error: Agent #{agent} cannot complete a task assigned to agent {task.assigned_to}\n"
            "Use --override only if you are explicitly taking over the task."
        )

    source_path = Path(tm.source_root) / task.source_path
    target_path = Path(tm.target_root) / task.target_path

    if not target_path.exists() and not override_mode:
        raise SystemExit(
            f"Error: Cannot complete task - target file does not exist: {target_path}\n"
            "Create the file and add a port-lint header first."
        )

    if target_path.exists():
        stats = PortingAnalyzer.analyze_file(target_path)
        if stats.todos and not override_mode:
            raise SystemExit(
                "Error: Cannot complete task - target file contains TODO markers\n"
                "TODOs are an automatic failure mode for porting completeness."
            )
        if stats.is_stub and not override_mode:
            raise SystemExit(
                "Error: Cannot complete task - target file is detected as a stub\n"
                "Complete the real implementation first."
            )

        src_lang = _parse_language(tm.source_lang)
        tgt_lang = _parse_language(tm.target_lang)

        parser = ASTParser()
        has_stubs = parser.has_stub_bodies_in_files([str(target_path)], tgt_lang)
        if has_stubs and not override_mode:
            raise SystemExit(
                "Error: Cannot complete task - target file contains stub/TODO markers in function bodies\n"
                "The code is fake. Complete the real implementation first."
            )

        src_tree = parser.parse_file(str(source_path), src_lang)
        tgt_tree = parser.parse_file(str(target_path), tgt_lang)
        if src_tree is None or tgt_tree is None:
            raise SystemExit(
                "Error: Cannot parse files for comparison\n"
                "This usually means the target file has syntax errors."
            )

        src_ids = parser.extract_identifiers_from_file(str(source_path), src_lang)
        tgt_ids = parser.extract_identifiers_from_file(str(target_path), tgt_lang)
        file_sim = ASTSimilarity.combined_similarity_with_content(src_tree, tgt_tree, src_ids, tgt_ids)

        # Parity penalty: missing functions should reduce score.
        from collections import Counter

        def _function_name_coverage_ratio(src_paths: list[str], tgt_paths: list[str]) -> float:
            try:
                src_bytes = parser._read_combined_files(src_paths)
                tgt_bytes = parser._read_combined_files(tgt_paths)
            except OSError:
                return 1.0

            src_funcs = parser.extract_functions_bytes(src_bytes, src_lang)
            tgt_funcs = parser.extract_functions_bytes(tgt_bytes, tgt_lang)

            src_names = [
                SourceFile.normalize_name(name)
                for name, _tree in src_funcs
                if name and name != "<anonymous>"
            ]
            if not src_names:
                return 1.0

            tgt_counts = Counter(
                SourceFile.normalize_name(name)
                for name, _tree in tgt_funcs
                if name and name != "<anonymous>"
            )

            matched = 0
            for n in src_names:
                if tgt_counts.get(n, 0) > 0:
                    matched += 1
                    tgt_counts[n] -= 1
            return matched / len(src_names)

        fn_cov = _function_name_coverage_ratio([str(source_path)], [str(target_path)])
        similarity = file_sim * fn_cov

        if similarity < 0.85 and not override_mode:
            raise SystemExit(
                f"Error: Cannot complete task with low similarity: {similarity:.4f}\n"
                "Port is incomplete. Increase similarity and function parity first."
            )

    if not tm.complete_task(source_qualified):
        raise SystemExit(f"Task not found: {source_qualified}")

    print(f"Marked as completed: {source_qualified}")

    # Rescan to update priorities
    if not tm.source_root or not tm.source_lang:
        print("Warning: Task file missing source/target info, cannot rescan.")
        return

    print("Rescanning codebases to update priorities...")
    source = _scan_codebase(tm.source_root, tm.source_lang, build_deps=True)
    target = _scan_codebase(tm.target_root, tm.target_lang, build_deps=True, porting_data=True)

    comp = CodebaseComparator(source, target)
    comp.find_matches()
    comp.compute_similarities()

    # Preserve assignment and completion states
    assigned_map: dict[str, tuple[str, str]] = {}
    completed_map: dict[str, str] = {}
    for t in tm.tasks:
        if t.status == TaskStatus.ASSIGNED:
            assigned_map[t.source_qualified] = (t.assigned_to, t.assigned_at)
        elif t.status == TaskStatus.COMPLETED:
            completed_map[t.source_qualified] = t.completed_at

    # Rebuild task list
    tm.tasks.clear()
    pending_keys: set[str] = set(comp.unmatched_source)
    for m in comp.matches:
        if m.similarity < 0.85 or m.todo_count > 0 or m.lint_count > 0 or m.is_stub:
            pending_keys.add(m.source_path)

    pending_files = [source.files[k] for k in pending_keys]
    pending_files.sort(key=lambda s: s.dependent_count, reverse=True)

    for sf in pending_files:
        if sf.qualified_name in completed_map:
            continue
        task = PortTask(
            source_path=sf.relative_path,
            source_qualified=sf.qualified_name,
            target_path=_target_path_from_source(sf.relative_path, tm.source_lang, tm.target_lang),
            dependent_count=sf.dependent_count,
            dependency_count=sf.dependency_count,
            dependencies=[imp.module_path for imp in sf.imports],
        )
        if sf.qualified_name in assigned_map:
            task.status = TaskStatus.ASSIGNED
            task.assigned_to, task.assigned_at = assigned_map[sf.qualified_name]
        tm.tasks.append(task)

    # Re-add completed tasks
    for qualified, completed_at in completed_map.items():
        t = PortTask(source_qualified=qualified, status=TaskStatus.COMPLETED, completed_at=completed_at)
        for _path, sf in source.files.items():
            if sf.qualified_name == qualified:
                t.source_path = sf.relative_path
                t.dependent_count = sf.dependent_count
                break
        else:
            t.source_path = qualified
        tm.tasks.append(t)

    tm.save()
    stats = tm.get_stats()
    total = stats["pending"] + stats["assigned"] + stats["completed"]
    print(f"Progress: {stats['completed']}/{total} completed")
    print(f"Remaining: {stats['pending']} pending, {stats['assigned']} assigned")

    print("\nUpdated top priorities:")
    pending = [t for t in tm.tasks if t.status == TaskStatus.PENDING]
    for t in pending[:5]:
        print(f"  {t.source_qualified:<30} deps={t.dependent_count}")


def cmd_release(task_file: str, source_qualified: str, agent: int, override_mode: bool) -> None:
    """Release an assigned task back to pending, with anti-stub guardrails."""
    tm = TaskManager(task_file)
    if not tm.load():
        raise SystemExit(f"Error: Could not load task file: {task_file}")

    task = next((t for t in tm.tasks if t.source_qualified == source_qualified), None)
    if task is None or task.status != TaskStatus.ASSIGNED:
        raise SystemExit(f"Task not found or not assigned: {source_qualified}")

    if agent > 0 and task.assigned_to != str(agent) and not override_mode:
        raise SystemExit(
            f"Error: Agent #{agent} cannot release a task assigned to agent {task.assigned_to}\n"
            "Use --override only if you are explicitly taking over the task."
        )

    target_path = Path(tm.target_root) / task.target_path
    if target_path.exists():
        source_path = Path(tm.source_root) / task.source_path

        if not override_mode:
            stats = PortingAnalyzer.analyze_file(target_path)
            if stats.todos:
                raise SystemExit(
                    "Error: Cannot release task - target file contains TODO markers\n"
                    "Complete the real implementation or delete the file to release."
                )
            if stats.is_stub:
                raise SystemExit(
                    "Error: Cannot release task - target file is detected as a stub\n"
                    "Complete the real implementation or delete the file to release."
                )

        src_lang = _parse_language(tm.source_lang)
        tgt_lang = _parse_language(tm.target_lang)
        parser = ASTParser()

        has_stubs = parser.has_stub_bodies_in_files([str(target_path)], tgt_lang)
        if has_stubs and not override_mode:
            raise SystemExit(
                "Error: Cannot release task - target file contains stub/TODO markers in function bodies\n"
                "The code is fake. Complete the real implementation or delete the file."
            )

        src_tree = parser.parse_file(str(source_path), src_lang)
        tgt_tree = parser.parse_file(str(target_path), tgt_lang)
        if src_tree is None or tgt_tree is None:
            raise SystemExit(
                "Error: Cannot parse files for comparison\n"
                "This usually means the target file has syntax errors.\n"
                "Fix the errors or delete the file to release."
            )

        src_ids = parser.extract_identifiers_from_file(str(source_path), src_lang)
        tgt_ids = parser.extract_identifiers_from_file(str(target_path), tgt_lang)
        file_sim = ASTSimilarity.combined_similarity_with_content(src_tree, tgt_tree, src_ids, tgt_ids)

        # Parity penalty: missing functions should reduce score.
        from collections import Counter

        def _function_name_coverage_ratio(src_paths: list[str], tgt_paths: list[str]) -> float:
            try:
                src_bytes = parser._read_combined_files(src_paths)
                tgt_bytes = parser._read_combined_files(tgt_paths)
            except OSError:
                return 1.0

            src_funcs = parser.extract_functions_bytes(src_bytes, src_lang)
            tgt_funcs = parser.extract_functions_bytes(tgt_bytes, tgt_lang)

            src_names = [
                SourceFile.normalize_name(name)
                for name, _tree in src_funcs
                if name and name != "<anonymous>"
            ]
            if not src_names:
                return 1.0

            tgt_counts = Counter(
                SourceFile.normalize_name(name)
                for name, _tree in tgt_funcs
                if name and name != "<anonymous>"
            )

            matched = 0
            for n in src_names:
                if tgt_counts.get(n, 0) > 0:
                    matched += 1
                    tgt_counts[n] -= 1
            return matched / len(src_names)

        fn_cov = _function_name_coverage_ratio([str(source_path)], [str(target_path)])
        similarity = file_sim * fn_cov

        if similarity < 0.50 and not override_mode:
            raise SystemExit(
                f"Error: Cannot release task with low similarity: {similarity:.4f}\n"
                "Target file exists but identifier content doesn't match source.\n"
                "Either complete the port or delete the target file to release."
            )

        print(
            f"Warning: Releasing with partial port (similarity {similarity:.4f})",
            file=sys.stderr,
        )
        print("Consider completing it instead (use --complete).", file=sys.stderr)

    if not tm.release_task(source_qualified):
        raise SystemExit(f"Task not found or not assigned: {source_qualified}")
    print(f"Released task: {source_qualified}")


# ── Shared helpers ──────────────────────────────────────────────────────────

def _scan_codebase(root: str, lang: str, *, build_deps: bool = False,
                   porting_data: bool = False) -> Codebase:
    cb = Codebase(root, lang)
    cb.scan()
    cb.extract_imports()
    if build_deps:
        cb.build_dependency_graph()
    if porting_data:
        cb.extract_porting_data()
    return cb


def _print_tree(tree, indent: int = 0) -> None:
    from .node_types import node_type_name, NodeType
    pad = "  " * indent
    name = node_type_name(NodeType(tree.node_type))
    suffix = " [leaf]" if tree.is_leaf() else ""
    label = f" ({tree.label})" if tree.label else ""
    print(f"{pad}{name}{label}{suffix}")
    for child in tree.children:
        _print_tree(child, indent + 1)


def _print_histogram(hist: list[int]) -> None:
    print("Node Type Histogram:")
    for i, count in enumerate(hist):
        if count > 0:
            name = node_type_name(NodeType(i))
            print(f"  {name:<15}: {count}")


# ── Usage and dispatch ──────────────────────────────────────────────────────

def print_usage() -> None:
    prog = "ast_distance"
    print(f"""AST Distance - Cross-language AST comparison and porting analysis

Usage:
  {prog} [--agent <number>] [--task-file <tasks.json>] [--override] <command>
      Guardrails: when a task system is initialized, commands require --agent.

  {prog} <file1> <lang1> <file2> <lang2>
      Compare AST similarity between two files

  {prog} --dump <file> <rust|kotlin|cpp|python>
      Dump AST structure of a file

  {prog} --scan <directory> <rust|kotlin|cpp|python>
      Scan directory and show file list with import counts

  {prog} --deps <directory> <rust|kotlin|cpp|python>
      Build and show dependency graph

  {prog} --rank <src_dir> <src_lang> <tgt_dir> <tgt_lang>
      Rank files by porting priority (dependents + similarity)

  {prog} --deep <src_dir> <src_lang> <tgt_dir> <tgt_lang>
      Full analysis: AST + deps + TODOs + lint + line ratios

  {prog} --missing <src_dir> <src_lang> <tgt_dir> <tgt_lang>
      Show files missing from target, ranked by importance

  {prog} --todos <directory>
      Scan for TODO comments with tags and context

  {prog} --lint <directory>
      Run lint checks (unused params, missing guards)

  {prog} --stats <directory>
      Show file statistics (line counts, stubs, TODOs)

Symbol Analysis:
  {prog} --symbols <kotlin_root> <cpp_root>
      Run symbol analysis (duplicates + stubs)

  {prog} --symbols-duplicates <kotlin_root> <cpp_root>
      Show duplicate class/struct definitions

  {prog} --symbols-stubs <kotlin_root> <cpp_root>
      Show stub files/classes

  {prog} --symbols-symbol <kotlin_root> <cpp_root> <symbol> [--json]
      Analyze a specific symbol (optionally output JSON)

  {prog} --symbol-parity <rust_root> <kotlin_root> [options]
      Rust->Kotlin symbol parity analysis

  {prog} --import-map <kotlin_root> [options]
      Build type registry and show missing imports per file

  {prog} --compiler-fixup <kotlin_root> <error_file> [options]
      Parse compiler errors and suggest import fixes

Swarm Task Management:
  {prog} --init-tasks <src_dir> <src_lang> <tgt_dir> <tgt_lang> <task_file>
      Generate task file from missing/incomplete ports

  {prog} --tasks <task_file>
      Show task status summary

  {prog} --assign <task_file> [agent_number]
      Assign highest-priority pending task to an agent session
      If agent_number is omitted, a new one is allocated and printed

  {prog} --complete <task_file> <source_qualified>
      Mark a task as completed

  {prog} --release <task_file> <source_qualified>
      Release an assigned task back to pending

  Languages: rust, kotlin, cpp, python""", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    raw_args = argv if argv is not None else sys.argv[1:]

    try:
        _refuse_piped_stdio()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2

    agent, task_file_flag, override_mode, args = _parse_global_flags(list(raw_args))
    if not args:
        print_usage()
        return 1

    mode = args[0]

    # Preserve global flags for C++ proxy calls.
    proxy_prefix: list[str] = []
    if agent > 0:
        proxy_prefix += ["--agent", str(agent)]
    if task_file_flag:
        proxy_prefix += ["--task-file", task_file_flag]
    if override_mode:
        proxy_prefix += ["--override"]

    proxy_to_cpp = mode in {
        "--symbols",
        "--symbols-duplicates",
        "--symbols-stubs",
        "--symbols-symbol",
        "--symbol-parity",
        "--import-map",
        "--compiler-fixup",
    }

    # Detect whether a task system is initialized for this invocation.
    task_file = ""
    if task_file_flag:
        task_file = task_file_flag
    elif mode in {"--tasks", "--assign", "--complete", "--release"} and len(args) >= 2:
        task_file = args[1]
    elif mode == "--init-tasks" and len(args) >= 6:
        task_file = args[5]
    elif Path("tasks.json").exists():
        task_file = "tasks.json"

    guard_active = False
    if task_file:
        tm_check = TaskManager(task_file)
        guard_active = tm_check.load()

    _GUARD.active = guard_active
    _GUARD.task_file = task_file
    _GUARD.agent = agent
    _GUARD.override_mode = override_mode

    # Guardrails: once a task system exists, require --agent for all commands (except --assign).
    agent_lock: _AgentLock | None = None
    if guard_active and mode != "--assign" and not proxy_to_cpp:
        if agent <= 0:
            print(f"Error: task system detected ({task_file}).", file=sys.stderr)
            print("All commands require an agent session number.", file=sys.stderr)
            print(f"Get one with: ast_distance --assign {task_file}", file=sys.stderr)
            print("Then re-run with: ast_distance --agent <number> ...", file=sys.stderr)
            return 2

        agent_lock = _AgentLock(task_file, agent, override_mode)
        if not agent_lock.locked():
            holder = agent_lock.holder_pid()
            msg = f"Agent #{agent} appears to be in use"
            if holder > 0:
                msg += f" by pid {holder}"
            msg += ".\nRe-run with --override to wait, or pick a different agent number.\n"
            _write_agent_notice(
                task_file,
                agent,
                "Conflict: another PID attempted to use this agent number.\n" + msg,
            )
            print("Error: " + msg, file=sys.stderr)
            agent_lock.close()
            return 3

        _print_and_clear_agent_notice(task_file, agent)
        _touch_agent_last_used(task_file, agent)

    try:
        if mode == "--scan" and len(args) >= 3:
            cmd_scan(args[1], args[2])

        elif mode == "--deps" and len(args) >= 3:
            cmd_deps(args[1], args[2])

        elif mode == "--rank" and len(args) >= 5:
            cmd_rank(args[1], args[2], args[3], args[4])

        elif mode == "--deep" and len(args) >= 5:
            cmd_deep(args[1], args[2], args[3], args[4])

        elif mode == "--missing" and len(args) >= 5:
            cmd_missing(args[1], args[2], args[3], args[4])

        elif mode == "--todos" and len(args) >= 2:
            verbose = not (len(args) >= 3 and args[2] == "--summary")
            cmd_todos(args[1], verbose)

        elif mode == "--lint" and len(args) >= 2:
            cmd_lint(args[1])

        elif mode == "--stats" and len(args) >= 2:
            cmd_stats(args[1])

        elif proxy_to_cpp:
            _proxy_to_cpp(proxy_prefix + args)

        elif mode == "--init-tasks" and len(args) >= 6:
            agents_md = args[6] if len(args) >= 7 else ""
            cmd_init_tasks(args[1], args[2], args[3], args[4], args[5], agents_md)

        elif mode == "--tasks" and len(args) >= 2:
            cmd_tasks(args[1], agent)

        elif mode == "--assign" and len(args) >= 2:
            requested_agent = 0
            if len(args) >= 3:
                requested_agent = int(args[2])
            elif agent > 0:
                requested_agent = agent
            cmd_assign(args[1], requested_agent, override_mode)

        elif mode == "--complete" and len(args) >= 3:
            cmd_complete(args[1], args[2], agent, override_mode)

        elif mode == "--release" and len(args) >= 3:
            cmd_release(args[1], args[2], agent, override_mode)

        elif mode == "--dump" and len(args) >= 3:
            cmd_dump(args[1], args[2])

        elif not mode.startswith("-") and len(args) >= 4:
            cmd_compare(args[0], args[1], args[2], args[3])

        else:
            print_usage()
            return 1

    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        if isinstance(e.code, str) and e.code:
            print(e.code, file=sys.stderr)
        return code
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if agent_lock is not None:
            agent_lock.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

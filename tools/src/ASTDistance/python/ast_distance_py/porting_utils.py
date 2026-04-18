"""Porting utilities for kotlinx.coroutines C++ port analysis.

Features:
    - TODO scanning with tag extraction
    - Lint checks (unused parameters)
    - Line counting and ratio analysis
    - "Transliterated from:" header parsing
    - Stub detection
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Set


IGNORED_KEYWORDS: Set[str] = {
    "if", "while", "for", "switch", "catch", "when", "return",
    "sizeof", "alignof", "decltype", "static_assert", "constexpr", "template",
    "void", "int", "bool", "float", "double", "char", "short", "long", "unsigned",
    "auto", "const", "static", "virtual", "override", "final", "explicit",
    "inline", "noexcept", "nullptr", "true", "false", "this", "new", "delete",
    # Kotlin keywords that look like function calls
    "check", "require", "assert"
}


class TodoItem:
    """Represents a TODO comment found in source code."""
    def __init__(self):
        self.file_path: str = ""
        self.line_num: int = 0
        self.tag: str = ""
        self.message: str = ""
        self.context: List[str] = []
        self.kt_line_start: int = 0
        self.kt_line_end: int = 0

    def print(self, verbose: bool = True) -> None:
        tag_display = self.tag if self.tag else "untagged"
        print(f"{self.file_path}:{self.line_num}: TODO({tag_display}): {self.message}")

        if verbose and self.context:
            print("  Context:")
            for line in self.context:
                print(f"    {line}")


class LintError:
    """Represents a lint error found in source code."""
    def __init__(self):
        self.file_path: str = ""
        self.line_num: int = 0
        self.type: str = ""
        self.message: str = ""

    def print(self) -> None:
        print(f"{self.file_path}:{self.line_num}: {self.type}: {self.message}")


class FileStats:
    """File statistics for porting analysis."""
    def __init__(self):
        self.path: str = ""
        self.relative_path: str = ""
        self.line_count: int = 0
        self.code_lines: int = 0
        self.comment_lines: int = 0
        self.blank_lines: int = 0
        self.is_stub: bool = False
        self.has_header_guard: bool = False
        self.transliterated_from: str = ""
        self.todos: List[TodoItem] = []
        self.lint_errors: List[LintError] = []

    def code_ratio(self, kt_lines: int) -> float:
        if kt_lines == 0:
            return 0.0
        return float(self.line_count) / float(kt_lines)

    def print(self) -> None:
        print(f"File: {self.path}")
        print(f"  Lines: {self.line_count} (code: {self.code_lines}, comments: {self.comment_lines}, blank: {self.blank_lines})")
        if self.transliterated_from:
            print(f"  Transliterated from: {self.transliterated_from}")
        if self.is_stub:
            print("  WARNING: Appears to be a stub")
        if not self.has_header_guard:
            print("  WARNING: Missing header guard")
        print(f"  TODOs: {len(self.todos)}, Lint errors: {len(self.lint_errors)}")


class PortingAnalyzer:
    """Porting analysis utilities."""

    @staticmethod
    def scan_todos(filepath: str, context_lines: int = 3) -> List[TodoItem]:
        """Scan a file for TODO comments."""
        todos: List[TodoItem] = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return todos

        todo_re = re.compile(r'//\s*TODO(\([^)]*\))?:\s*(.+)')
        line_ref_re = re.compile(r'Line\s+(\d+)(?:-(\d+))?', re.IGNORECASE)

        for i, line in enumerate(lines):
            match = todo_re.search(line)
            if match:
                todo = TodoItem()
                todo.file_path = filepath
                todo.line_num = i + 1

                tag_part = match.group(1) or ""
                if tag_part and len(tag_part) > 2:
                    todo.tag = tag_part[1:-1]

                todo.message = match.group(2)

                line_match = line_ref_re.search(todo.message)
                if line_match:
                    todo.kt_line_start = int(line_match.group(1))
                    if line_match.group(2):
                        todo.kt_line_end = int(line_match.group(2))
                    else:
                        todo.kt_line_end = todo.kt_line_start

                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                for j in range(start, end):
                    prefix = ">>> " if j == i else "    "
                    todo.context.append(f"{prefix}{j + 1:4d}: {lines[j].rstrip()}")

                todos.append(todo)

        return todos

    @staticmethod
    def extract_transliterated_from(filepath: str) -> str:
        """Extract source path header from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return ""

        trans_re = re.compile(r'Transliterated from:\s*(.+)', re.IGNORECASE)
        portlint_re = re.compile(r'port-lint:\s*(?:source|tests)\s+(.+)', re.IGNORECASE)

        for line in lines[:50]:
            match = trans_re.search(line)
            if match:
                result = match.group(1).strip()
                return result

            match = portlint_re.search(line)
            if match:
                result = match.group(1).strip()
                if result.startswith("codex-rs/"):
                    result = result[9:]
                return result

        return ""

    @staticmethod
    def analyze_file(filepath: str) -> FileStats:
        """Analyze file statistics (line counts, stub detection, header guards)."""
        stats = FileStats()
        stats.path = filepath

        p = Path(filepath)
        stats.relative_path = p.name
        ext = p.suffix

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return stats

        lines = content.split('\n')
        in_block_comment = False

        for line in lines:
            stats.line_count += 1

            trimmed = line.lstrip()

            if not trimmed:
                stats.blank_lines += 1
            elif trimmed.startswith("//"):
                stats.comment_lines += 1
            elif "/*" in trimmed:
                stats.comment_lines += 1
                if "*/" not in trimmed:
                    in_block_comment = True
            elif in_block_comment:
                stats.comment_lines += 1
                if "*/" in trimmed:
                    in_block_comment = False
            else:
                stats.code_lines += 1

        if ext in ('.hpp', '.h'):
            stats.has_header_guard = ('#pragma once' in content or 
                                       '#ifndef' in content)
        else:
            stats.has_header_guard = True

        clean = content
        clean = re.sub(r'//[^\n]*', '', clean)
        clean = re.sub(r'/\*[\s\S]*?\*/', '', clean)
        clean = re.sub(r'#include[^\n]*', '', clean)
        clean = re.sub(r'namespace[^\{]*\{?', '', clean)
        clean = re.sub(r'#pragma[^\n]*', '', clean)
        clean = re.sub(r'package\s+[^\n]*', '', clean)
        clean = re.sub(r'import\s+[^\n]*', '', clean)
        clean = re.sub(r'use\s+[^\n]*', '', clean)
        clean = re.sub(r'mod\s+\w+\s*;', '', clean)
        clean = re.sub(r'from\s+[^\n]*', '', clean)
        clean = re.sub(r'Copyright[^\n]*', '', clean)
        clean = re.sub(r'Licensed under[^\n]*', '', clean)
        clean = re.sub(r'Apache License[^\n]*', '', clean)
        clean = re.sub(r'\s+', '', clean)

        stats.is_stub = len(clean) < 100

        stats.transliterated_from = PortingAnalyzer.extract_transliterated_from(filepath)
        stats.todos = PortingAnalyzer.scan_todos(filepath)

        return stats

    @staticmethod
    def is_kotlin_file(filepath: str) -> bool:
        """Check if a file is a Kotlin source file."""
        return filepath.endswith('.kt')

    @staticmethod
    def extract_kotlin_param_names(args_str: str) -> List[str]:
        """Extract parameter names from a Kotlin function parameter list.

        In Kotlin, parameters are declared as `name: Type` (name before colon),
        unlike C++ where the type comes first. This extracts the identifier
        immediately before each `:` separator, skipping annotations and modifiers.
        """
        params: List[str] = []

        # Split by commas (respecting angle brackets for generics)
        segments: List[str] = []
        angle_depth = 0
        current = ""
        for c in args_str:
            if c == '<':
                angle_depth += 1
                current += c
            elif c == '>':
                angle_depth -= 1
                current += c
            elif c == ',' and angle_depth == 0:
                segments.append(current)
                current = ""
            else:
                current += c
        if current:
            segments.append(current)

        # For each segment, find `name:` pattern (Kotlin param syntax)
        kotlin_param_re = re.compile(r'\b(\w+)\s*:')
        for seg in segments:
            # Strip default value (find '=' not inside angle brackets)
            s = seg
            adepth = 0
            eq_pos = -1
            for i, ch in enumerate(s):
                if ch == '<':
                    adepth += 1
                elif ch == '>':
                    adepth -= 1
                elif ch == '=' and adepth == 0:
                    eq_pos = i
                    break
            if eq_pos >= 0:
                s = s[:eq_pos]

            # Find the last `name:` pattern (to skip modifiers like vararg)
            last_name = ""
            for m in kotlin_param_re.finditer(s):
                last_name = m.group(1)

            if (last_name and
                    last_name not in IGNORED_KEYWORDS and
                    not last_name.startswith('_')):
                params.append(last_name)

        return params

    @staticmethod
    def extract_cpp_param_names(args_str: str) -> List[str]:
        """Extract parameter names from a C/C++ function parameter list.

        In C/C++, parameters are declared as `Type name` (name is last token),
        so we extract the last identifier from each comma-separated segment.
        """
        params: List[str] = []

        for param in args_str.split(','):
            eq_pos = param.find('=')
            if eq_pos != -1:
                param = param[:eq_pos]

            tokens = re.findall(r'\b(\w+)\b', param)
            if tokens:
                last_token = tokens[-1]
                if (last_token not in IGNORED_KEYWORDS and
                        not last_token.startswith('_')):
                    params.append(last_token)

        return params

    @staticmethod
    def check_unused_params(filepath: str) -> List[LintError]:
        """Check for unused parameters in functions."""
        errors: List[LintError] = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return errors

        kotlin = PortingAnalyzer.is_kotlin_file(filepath)

        # For Kotlin files, require 'fun' keyword before the function name.
        # For C/C++ files, use the original heuristic pattern.
        if kotlin:
            func_re = re.compile(r'\bfun\s+(?:<[^>]*>\s+)?(\w+)\s*\(([^)]*)\)\s*(?::\s*\w+(?:<[^>]*>)?\s*)?\{')
        else:
            func_re = re.compile(r'(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?\{')

        for match in func_re.finditer(content):
            func_name = match.group(1)
            args_str = match.group(2)

            if func_name in IGNORED_KEYWORDS:
                continue

            start_pos = match.end()
            depth = 1
            idx = start_pos

            while idx < len(content) and depth > 0:
                if content[idx] == '{':
                    depth += 1
                elif content[idx] == '}':
                    depth -= 1
                idx += 1

            if depth != 0:
                continue

            body = content[start_pos:idx - 1]

            if not args_str or args_str.startswith("void"):
                continue

            params = (PortingAnalyzer.extract_kotlin_param_names(args_str)
                      if kotlin
                      else PortingAnalyzer.extract_cpp_param_names(args_str))

            for p in params:
                usage_re = re.compile(r'\b' + re.escape(p) + r'\b')
                if not usage_re.search(body):
                    void_cast1 = "(void)" + p
                    void_cast2 = "(void) " + p
                    if void_cast1 not in body and void_cast2 not in body:
                        err = LintError()
                        err.file_path = filepath
                        err.line_num = content[:match.start()].count('\n') + 1
                        err.type = "unused_param"
                        err.message = f"Unused parameter '{p}' in function '{func_name}'"
                        errors.append(err)

        return errors

    @staticmethod
    def lint_file(filepath: str) -> List[LintError]:
        """Run all lint checks on a file."""
        errors = PortingAnalyzer.check_unused_params(filepath)

        if filepath.endswith('.hpp') or filepath.endswith('.h'):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                return errors

            if '#pragma once' not in content and '#ifndef' not in content:
                err = LintError()
                err.file_path = filepath
                err.line_num = 1
                err.type = "missing_guard"
                err.message = "Missing header guard (#pragma once or #ifndef)"
                errors.append(err)

        return errors

    @staticmethod
    def analyze_directory(directory: str) -> List[FileStats]:
        """Scan a directory for source files and analyze them."""
        results: List[FileStats] = []

        path = Path(directory)

        if path.is_file():
            ext = path.suffix
            if ext in ('.hpp', '.cpp', '.h', '.kt', '.kts', '.rs'):
                stats = PortingAnalyzer.analyze_file(directory)
                stats.lint_errors = PortingAnalyzer.lint_file(directory)
                results.append(stats)
            return results

        supported_exts = ('.hpp', '.cpp', '.h', '.kt', '.kts', '.rs')
        skip_dirs = ('/vendor/', '/build/', '/tmp/', '/.git/')

        for root, dirs, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)

                if any(skip in filepath for skip in skip_dirs):
                    continue

                ext = Path(filename).suffix
                if ext not in supported_exts:
                    continue

                stats = PortingAnalyzer.analyze_file(filepath)
                stats.lint_errors = PortingAnalyzer.lint_file(filepath)
                results.append(stats)

        return results

    @staticmethod
    def group_todos_by_tag(todos: List[TodoItem]) -> Dict[str, List[TodoItem]]:
        """Group TODOs by tag."""
        grouped: Dict[str, List[TodoItem]] = {}
        for todo in todos:
            tag = todo.tag if todo.tag else "untagged"
            if tag not in grouped:
                grouped[tag] = []
            grouped[tag].append(todo)
        return grouped

    @staticmethod
    def print_todo_report(todos: List[TodoItem], verbose: bool = True) -> None:
        """Print a TODO report."""
        if not todos:
            print("No TODOs found.")
            return

        print("\n================================================================================")
        print(f"TODO REPORT - Found {len(todos)} TODO(s)")
        print("================================================================================\n")

        grouped = PortingAnalyzer.group_todos_by_tag(todos)

        print("Summary by tag:")
        for tag, items in grouped.items():
            print(f"  {tag}: {len(items)}")
        print("\n")

        if not verbose:
            for todo in todos:
                todo.print(False)
            return

        for todo in todos:
            print("--------------------------------------------------------------------------------")
            print(f"FILE: {todo.file_path}")
            print(f"LINE: {todo.line_num}")
            print(f"TAG:  {todo.tag if todo.tag else 'none'}")
            print(f"MSG:  {todo.message}")

            if todo.kt_line_start > 0:
                if todo.kt_line_end > todo.kt_line_start:
                    print(f"KT:   Lines {todo.kt_line_start}-{todo.kt_line_end}")
                else:
                    print(f"KT:   Line {todo.kt_line_start}")

            print("\nContext:")
            for line in todo.context:
                print(f"  {line}")
            print("\n")

    @staticmethod
    def print_lint_report(errors: List[LintError]) -> None:
        """Print a lint report."""
        if not errors:
            print("No lint errors found.")
            return

        print("\n================================================================================")
        print(f"LINT REPORT - Found {len(errors)} error(s)")
        print("================================================================================\n")

        grouped: Dict[str, List[LintError]] = {}
        for err in errors:
            if err.type not in grouped:
                grouped[err.type] = []
            grouped[err.type].append(err)

        print("Summary by type:")
        for etype, items in grouped.items():
            print(f"  {etype}: {len(items)}")
        print("\n")

        for err in errors:
            err.print()

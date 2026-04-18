"""AST Parser using tree-sitter — parses source files into normalized Trees.

Faithful transliteration of include/ast_parser.hpp from the C++ ast_distance tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import tree_sitter
import tree_sitter_rust
import tree_sitter_kotlin
import tree_sitter_cpp
import tree_sitter_python as tree_sitter_py

from .node_types import (
    NodeType,
    rust_node_to_type,
    kotlin_node_to_type,
    cpp_node_to_type,
    python_node_to_type,
)
from .tree import Tree


class Language(Enum):
    """Supported programming languages for AST parsing."""
    RUST = "rust"
    KOTLIN = "kotlin"
    CPP = "cpp"
    PYTHON = "python"


def _make_language(capsule: object, name: str) -> tree_sitter.Language:
    """Construct a Language from a tree-sitter-* package capsule."""
    try:
        return tree_sitter.Language(capsule)  # type: ignore[arg-type]
    except TypeError as e:
        raise RuntimeError(
            "ast-distance requires the modern 'tree-sitter' Python bindings (tree-sitter>=0.23) "
            "which accept language capsules."
        ) from e


_LANGUAGES: dict[Language, tree_sitter.Language] = {
    Language.RUST: _make_language(tree_sitter_rust.language(), "rust"),
    Language.KOTLIN: _make_language(tree_sitter_kotlin.language(), "kotlin"),
    Language.CPP: _make_language(tree_sitter_cpp.language(), "cpp"),
    Language.PYTHON: _make_language(tree_sitter_py.language(), "python"),
}

_NODE_MAP_FN = {
    Language.RUST: rust_node_to_type,
    Language.KOTLIN: kotlin_node_to_type,
    Language.CPP: cpp_node_to_type,
    Language.PYTHON: python_node_to_type,
}

# Stop-words for doc tokenizer
_STOP_WORDS = frozenset({"the", "and", "for", "this", "that", "with"})

# Operator classification sets
_ARITH_OPS = frozenset({"+", "-", "*", "/", "%", "**"})
_CMP_OPS = frozenset({"==", "!=", "<", ">", "<=", ">=", "===", "!=="})
_LOGIC_OPS = frozenset({"&&", "||", "!"})
_BIT_OPS = frozenset({"&", "|", "^", "~", "<<", ">>"})
_ASSIGN_OPS = frozenset({"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="})


# ══════════════════════════════════════════════════════════════════════════════
# IdentifierStats - Statistics about identifiers (function names, variables, etc.)
# Faithful transliteration of IdentifierStats from ast_parser.hpp:37-189
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IdentifierStats:
    """Statistics about identifiers (function names, variable names, etc.)
    
    Used for Phase 3: Token histogram analysis to detect naming divergences.
    """

    identifier_freq: dict[str, int] = field(default_factory=dict)
    canonical_freq: dict[str, int] = field(default_factory=dict)
    total_identifiers: int = 0

    @staticmethod
    def canonicalize(name: str) -> str:
        """Canonicalize an identifier for cross-language comparison.
        
        "foo_bar" and "fooBar" and "FooBar" all become "foobar".
        This lets snake_case Rust match camelCase Kotlin.
        
        Also normalizes cross-language equivalents:
          self/this → "this", Option/nullable → "option",
          Vec/List/MutableList → "list", etc.
        
        Faithful transliteration of canonicalize() from ast_parser.hpp:51-114.
        """
        # First: lowercase + strip underscores.
        #
        # NOTE: Identifier overlap is the dominant signal for Rust→Kotlin ports. Kotlin ports often
        # introduce "plumbing" identifiers (stdlib helpers, Result helpers, package/import path
        # components, and tiny temp vars) which can cause false negatives even for faithful
        # transliterations. We normalize or ignore a small, explicit set of such identifiers to keep
        # the similarity metric focused on semantic names.
        result = "".join(ch.lower() for ch in name if ch != "_")

        # Low-signal tokens to ignore entirely (do not count toward canonical overlap).
        #
        # These are intentionally explicit (not heuristic) to avoid hiding real drift.
        ignore = {
            # Explicit receiver tokens (Rust uses `self` heavily; Kotlin often omits `this`).
            "self",
            "this",
            # Rust path components / package/import noise.
            "crate",
            "io",
            "github",
            "com",
            "kotlin",
            "kotlinmania",
            "starlarkkotlin",
            # Kotlin keyword-ish noise that can appear as identifiers in some parses.
            "val",
            "var",
            "true",
            "false",
            # Very common tiny temp vars introduced by tuple destructuring / iterator glue.
            "xk",
            "xv",
            "xy",
            "yk",
            "yv",
            # Iterator/local glue frequently introduced by Kotlin ports (keep explicit).
            "xsiter",
            "xsbasics",
            "rest",
            "merged",
            "minlen",
            "ch",
            "sb",
        }
        if result in ignore:
            return ""

        # Cross-language equivalents (applied after lowering).
        equivalents = {
            # Keywords / visibility
            "super": "super",
            # Visibility: Rust `pub(crate)` most closely matches Kotlin `internal`
            "internal": "public",
            # Collections
            "vec": "list",
            "mutablelist": "list",
            "arraylist": "list",
            "mutablelistof": "list",
            "listof": "list",
            "tomutablelist": "list",
            "tolist": "list",
            "hashmap": "map",
            "mutablemap": "map",
            "mutablemapof": "map",
            "mapof": "map",
            "hashset": "set",
            "mutableset": "set",
            "mutablesetof": "set",
            "setof": "set",
            "btreemap": "map",
            "btreeset": "set",
            # Types
            "option": "nullable",
            "some": "notnull",
            "none": "null",
            "box": "boxed",
            "arc": "arc",
            "string": "string",
            "str": "string",
            "i32": "int",
            "i64": "long",
            "u32": "uint",
            "u64": "ulong",
            "usize": "uint",
            "isize": "int",
            "f32": "float",
            "f64": "double",
            "bool": "boolean",
            "kclass": "typeid",
            "pair": "tuple",
            "triple": "tuple",
            "unit": "void",
            # Error handling
            "result": "result",
            "freezeresult": "result",
            "err": "error",
            "ok": "success",
            "failure": "error",
            # Kotlin Result helper names often appear in faithful ports.
            "getorthrow": "unwrap",
            "exceptionornull": "error",
            "issuccess": "ok",
            "isfailure": "err",
            # Rust trait methods -> Kotlin equivalents
            "fmt": "tostring",  # Display::fmt -> toString
            "eq": "equals",  # PartialEq::eq -> equals
            "partialeq": "equals",
            "cmp": "compareto",  # Ord::cmp -> compareTo
            "partialcmp": "compareto",  # PartialOrd::partial_cmp -> compareTo
            "hash": "hashcode",  # Hash::hash -> hashCode
            "clone": "copy",  # Clone::clone -> copy (data class)
            "default": "invoke",  # Default::default -> companion invoke
            "fromstr": "parse",  # FromStr -> parse
            "intoiter": "iterator",  # IntoIterator::into_iter -> iterator
            "intoiterator": "iterator",
            "hasnext": "next",
            "next": "next",  # Iterator::next (same name)
            "serialize": "serialize",
            "deserialize": "deserialize",
            # Kotlin string builders are commonly used where Rust uses `String`.
            "stringbuilder": "string",
            "buildstring": "string",
            "append": "push",
            "substring": "split",
            "deref": "get",  # Deref::deref -> get/value
            "drop": "close",  # Drop::drop -> close/Closeable
            "freeze": "freeze",  # project-specific
            "trace": "trace",  # project-specific
            # Common prefixes
            "fn": "fun",
            "impl": "class",
            "pub": "public",
            "mut": "var",
            "let": "val",
            # Common operations in ports
            "len": "size",
            "size": "len",
            "push": "add",
            "add": "push",
            # List slice adapters often differ by language surface
            "slice": "asslice",
            # Kotlin toString vs Rust format!/Debug
            "tostring": "display",
            "formatdebug": "format",
        }

        return equivalents.get(result, result)

    def add_identifier(self, name: str) -> None:
        """Add an identifier to the statistics.
        
        Faithful transliteration of add_identifier() from ast_parser.hpp:116-122.
        """
        if not name:
            return
        self.identifier_freq[name] = self.identifier_freq.get(name, 0) + 1
        c = self.canonicalize(name)
        if c:
            self.canonical_freq[c] = self.canonical_freq.get(c, 0) + 1
        self.total_identifiers += 1

    @staticmethod
    def _cosine_similarity_of(a: dict[str, int], b: dict[str, int]) -> float:
        """Compute cosine similarity between two frequency maps.
        
        Faithful transliteration from ast_parser.hpp:165-188.
        """
        if not a or not b:
            return 0.0

        all_ids = set(a) | set(b)
        dot = norm1 = norm2 = 0.0
        for ident in all_ids:
            f1 = a.get(ident, 0)
            f2 = b.get(ident, 0)
            dot += f1 * f2
            norm1 += f1 * f1
            norm2 += f2 * f2

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))

    def identifier_cosine_similarity(self, other: IdentifierStats) -> float:
        """Compute cosine similarity of identifier frequencies.
        
        Uses raw identifiers — detects naming divergences when same language.
        Faithful transliteration from ast_parser.hpp:128-130.
        """
        return self._cosine_similarity_of(self.identifier_freq, other.identifier_freq)

    def canonical_cosine_similarity(self, other: IdentifierStats) -> float:
        """Compute cosine similarity using canonicalized identifiers.
        
        This is the key metric for cross-language porting: "foo_bar" matches "fooBar".
        A file full of placeholder stubs will score near 0 because the *names* are wrong,
        even if the AST shape looks similar.
        Faithful transliteration from ast_parser.hpp:138-140.
        """
        return self._cosine_similarity_of(self.canonical_freq, other.canonical_freq)

    def canonical_jaccard_similarity(self, other: IdentifierStats) -> float:
        """Jaccard similarity of canonicalized identifier sets (ignoring frequency).
        
        What fraction of identifiers are shared between source and target?
        Faithful transliteration from ast_parser.hpp:146-162.
        """
        if not self.canonical_freq and not other.canonical_freq:
            return 1.0
        if not self.canonical_freq or not other.canonical_freq:
            return 0.0

        ids1 = set(self.canonical_freq)
        ids2 = set(other.canonical_freq)

        intersection = len(ids1 & ids2)
        union_size = len(ids1) + len(ids2) - intersection

        if union_size == 0:
            return 1.0
        return intersection / union_size


# ══════════════════════════════════════════════════════════════════════════════
# CommentStats - Statistics about comments/documentation in source code
# Faithful transliteration from ast_parser.hpp:194-270
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CommentStats:
    """Statistics about comments/documentation in source code.
    
    Faithful transliteration of CommentStats from ast_parser.hpp:194-270.
    """
    doc_comment_count: int = 0
    line_comment_count: int = 0
    block_comment_count: int = 0
    total_comment_lines: int = 0
    total_doc_lines: int = 0
    doc_texts: list[str] = field(default_factory=list)
    word_freq: dict[str, int] = field(default_factory=dict)

    def print(self) -> None:
        """Print comment statistics (faithful transliteration of print() from ast_parser.hpp:203-211)."""
        print("Comment Statistics:")
        print(f"  Doc comments:      {self.doc_comment_count}")
        print(f"  Line comments:     {self.line_comment_count}")
        print(f"  Block comments:   {self.block_comment_count}")
        print(f"  Total comment lines: {self.total_comment_lines}")
        print(f"  Doc comment lines:   {self.total_doc_lines}")
        print(f"  Unique doc words:    {len(self.word_freq)}")

    def doc_coverage_ratio(self) -> float:
        """Compute ratio of doc comment lines to total comment lines.
        
        Faithful transliteration from ast_parser.hpp:213-216.
        """
        if self.total_comment_lines == 0:
            return 0.0
        return self.total_doc_lines / self.total_comment_lines

    def doc_line_coverage_capped(self, other: CommentStats) -> float:
        """Asymmetric doc amount coverage (target/source doc lines, capped at 1.0).

        If `other` has more docs than `self`, treat as full coverage (1.0) rather than a penalty.
        """
        if self.total_doc_lines <= 0:
            return 1.0
        return min(1.0, other.total_doc_lines / self.total_doc_lines)

    def doc_line_balance(self, other: CommentStats) -> float:
        """Symmetric doc amount balance (min/max of doc lines)."""
        max_lines = max(self.total_doc_lines, other.total_doc_lines)
        if max_lines <= 0:
            return 1.0
        return min(self.total_doc_lines, other.total_doc_lines) / max_lines

    def doc_cosine_similarity(self, other: CommentStats) -> float:
        """Compute cosine similarity of doc word frequencies.
        
        Returns 0.0 to 1.0 where 1.0 = identical vocabulary distribution.
        Faithful transliteration from ast_parser.hpp:222-248.
        """
        if not self.word_freq or not other.word_freq:
            return 0.0

        all_words = set(self.word_freq) | set(other.word_freq)
        dot = norm1 = norm2 = 0.0
        for w in all_words:
            f1 = self.word_freq.get(w, 0)
            f2 = other.word_freq.get(w, 0)
            dot += f1 * f2
            norm1 += f1 * f1
            norm2 += f2 * f2

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))

    def doc_jaccard_similarity(self, other: CommentStats) -> float:
        """Jaccard similarity of doc word sets (ignoring frequency).
        
        Faithful transliteration from ast_parser.hpp:253-269.
        """
        if not self.word_freq and not other.word_freq:
            return 1.0
        if not self.word_freq or not other.word_freq:
            return 0.0

        words1 = set(self.word_freq)
        words2 = set(other.word_freq)

        intersection = len(words1 & words2)
        union_size = len(words1) + len(words2) - intersection

        if union_size == 0:
            return 1.0
        return intersection / union_size


# ══════════════════════════════════════════════════════════════════════════════
# FunctionInfo - Function metadata extracted from source code
# Faithful transliteration from ast_parser.hpp:277-282
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FunctionInfo:
    """Function metadata extracted from source code.
    
    The AST is kept as the function body (not the whole declaration)
    so that stub checks and identifier matching are aligned with behavior.
    Faithful transliteration of FunctionInfo from ast_parser.hpp:277-282.
    """
    name: str = ""
    body_tree: Optional[Tree] = None
    identifiers: IdentifierStats = field(default_factory=IdentifierStats)
    has_stub_markers: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# ASTParser - Main parser class using tree-sitter
# Faithful transliteration from ast_parser.hpp:288-600+
# ══════════════════════════════════════════════════════════════════════════════

class ASTParser:
    """AST Parser using tree-sitter.
    
    Parses source files into normalized Tree structures.
    Faithful transliteration of class ASTParser from ast_parser.hpp.
    """

    def __init__(self) -> None:
        self._parsers: dict[Language, tree_sitter.Parser] = {}

    def _get_parser(self, lang: Language) -> tree_sitter.Parser:
        if lang not in self._parsers:
            p = tree_sitter.Parser(_LANGUAGES[lang])
            self._parsers[lang] = p
        return self._parsers[lang]

    # ════════════════════════════════════════════════════════════════════════════
    # Parsing methods
    # ════════════════════════════════════════════════════════════════════════════

    def _read_combined_files(self, filepaths: Iterable[str | Path]) -> bytes:
        """Read multiple files and combine them into a single byte string."""
        chunks: list[bytes] = []
        for filepath in filepaths:
            chunks.append(Path(filepath).read_bytes())
            chunks.append(b"\n\n")
        return b"".join(chunks)

    def parse_file(
        self, filepath: str | Path | Iterable[str | Path], lang: Language
    ) -> Tree:
        """Parse a source file into a normalized AST.
        
        Faithful transliteration of parse_file() from ast_parser.hpp:312-329.
        """
        if isinstance(filepath, (str, Path)):
            source = Path(filepath).read_bytes()
        else:
            source = self._read_combined_files(filepath)
        return self.parse_bytes(source, lang)

    def parse_string(self, source: str, lang: Language) -> Tree:
        """Parse a source code string into a normalized AST.
        
        Faithful transliteration of parse_string() from ast_parser.hpp:334-362.
        """
        return self.parse_bytes(source.encode(), lang)

    def parse_bytes(self, source: bytes, lang: Language) -> Tree:
        """Parse source code bytes into a normalized AST.
        
        Faithful transliteration of parse_string() from ast_parser.hpp:334-362.
        """
        parser = self._get_parser(lang)
        ts_tree = parser.parse(source)
        root = ts_tree.root_node
        return self._convert_node(root, source, lang)

    # ════════════════════════════════════════════════════════════════════════════
    # Comment extraction
    # ════════════════════════════════════════════════════════════════════════════

    def extract_comments(self, source: str, lang: Language) -> CommentStats:
        """Extract comment statistics from source code.
        
        Uses tree-sitter to find comment nodes.
        Faithful transliteration from ast_parser.hpp:368-395.
        """
        return self.extract_comments_bytes(source.encode(), lang)

    def extract_comments_bytes(self, source: bytes, lang: Language) -> CommentStats:
        """Extract comment statistics from source bytes."""
        parser = self._get_parser(lang)
        ts_tree = parser.parse(source)
        stats = CommentStats()
        self._extract_comments_recursive(ts_tree.root_node, source, lang, stats)
        return stats

    def extract_comments_from_file(
        self,
        filepath: str | Path | Iterable[str | Path],
        lang: Language,
    ) -> CommentStats:
        """Extract comment statistics from a file or files.
        
        Faithful transliteration from ast_parser.hpp:400-416.
        """
        try:
            if isinstance(filepath, (str, Path)):
                source = Path(filepath).read_bytes()
            else:
                source = self._read_combined_files(filepath)
        except OSError:
            return CommentStats()
        return self.extract_comments_bytes(source, lang)

    # ════════════════════════════════════════════════════════════════════════════
    # Identifier extraction
    # ════════════════════════════════════════════════════════════════════════════

    def extract_identifiers(self, source: str, lang: Language) -> IdentifierStats:
        """Extract identifier statistics from source code.
        
        Phase 3: Token histogram for detecting naming divergences.
        Faithful transliteration from ast_parser.hpp:422-451.
        """
        return self.extract_identifiers_bytes(source.encode(), lang)

    def extract_identifiers_bytes(self, source: bytes, lang: Language) -> IdentifierStats:
        """Extract identifier statistics from source bytes."""
        parser = self._get_parser(lang)
        ts_tree = parser.parse(source)
        stats = IdentifierStats()
        self._extract_identifiers_recursive(ts_tree.root_node, stats)
        return stats

    def extract_identifiers_from_file(
        self,
        filepath: str | Path | Iterable[str | Path],
        lang: Language,
    ) -> IdentifierStats:
        """Extract identifier statistics from a file or files.
        
        Faithful transliteration from ast_parser.hpp:456-472.
        """
        try:
            if isinstance(filepath, (str, Path)):
                source = Path(filepath).read_bytes()
            else:
                source = self._read_combined_files(filepath)
        except OSError:
            return IdentifierStats()
        return self.extract_identifiers_bytes(source, lang)

    # ════════════════════════════════════════════════════════════════════════════
    # Function extraction
    # ════════════════════════════════════════════════════════════════════════════

    def extract_functions(
        self, source: str, lang: Language
    ) -> list[tuple[str, Tree]]:
        """Parse and extract only function bodies for comparison.
        
        Faithful transliteration from ast_parser.hpp:477-488.
        """
        return self.extract_functions_bytes(source.encode(), lang)

    def extract_functions_bytes(
        self, source: bytes, lang: Language
    ) -> list[tuple[str, Tree]]:
        """Parse and extract function bodies from bytes."""
        parser = self._get_parser(lang)
        ts_tree = parser.parse(source)
        functions: list[tuple[str, Tree]] = []
        self._extract_functions_recursive(ts_tree.root_node, source, lang, functions)
        return functions

    def extract_function_infos(
        self, source: str, lang: Language
    ) -> list[FunctionInfo]:
        """Extract function metadata and body ASTs from source.
        
        Faithful transliteration from ast_parser.hpp:493-522.
        """
        return self.extract_function_infos_bytes(source.encode(), lang)

    def extract_function_infos_bytes(
        self, source: bytes, lang: Language
    ) -> list[FunctionInfo]:
        """Extract function metadata and body ASTs from bytes.
        
        Faithful transliteration from ast_parser.hpp:493-522.
        """
        parser = self._get_parser(lang)
        ts_tree = parser.parse(source)
        functions: list[FunctionInfo] = []
        self._extract_function_infos_recursive(ts_tree.root_node, source, lang, functions)
        return functions

    def extract_function_infos_from_file(
        self, filepath: str, lang: Language
    ) -> list[FunctionInfo]:
        """Extract function metadata for one source file.
        
        Faithful transliteration from ast_parser.hpp:527-535.
        """
        try:
            source = Path(filepath).read_bytes()
        except OSError:
            return []
        return self.extract_function_infos_bytes(source, lang)

    def extract_function_infos_from_files(
        self, filepaths: list[str], lang: Language
    ) -> list[FunctionInfo]:
        """Extract function metadata for multiple source files.
        
        Faithful transliteration from ast_parser.hpp:540-554.
        """
        try:
            source = self._read_combined_files(filepaths)
        except OSError:
            return []
        if not source:
            return []
        return self.extract_function_infos_bytes(source, lang)

    # ════════════════════════════════════════════════════════════════════════════
    # Stub detection
    # ════════════════════════════════════════════════════════════════════════════

    @staticmethod
    def text_has_stub_markers(text: str) -> bool:
        """Check if a string contains stub/TODO markers.
        
        Faithful transliteration from ast_parser.hpp:559-593.
        """
        lower = text.lower()

        def is_word(ch: str) -> bool:
            return ch.isalnum() or ch == "_"

        def has_word(word: str) -> bool:
            pos = lower.find(word)
            while pos != -1:
                left_ok = pos == 0 or not is_word(lower[pos - 1])
                end = pos + len(word)
                right_ok = end >= len(lower) or not is_word(lower[end])
                if left_ok and right_ok:
                    return True
                pos = lower.find(word, pos + 1)
            return False

        return (
            has_word("todo")
            or has_word("stub")
            or has_word("placeholder")
            or has_word("fixme")
            or "not yet implemented" in lower
            or "not implemented" in lower
            or has_word("unimplemented")
            or has_word("notimplemented")
        )

    @staticmethod
    def comment_has_stub_markers(text: str) -> bool:
        """For comment nodes, be stricter: only treat as stub when comment
        itself starts with TODO/FIXME/STUB.
        
        Faithful transliteration from ast_parser.hpp:595-600+.
        """
        lower = text.lower()

        i = 0
        if lower.startswith("//"):
            i = 2
            while i < len(lower) and lower[i] == "/":
                i += 1
        elif lower.startswith("#"):
            i = 1
            while i < len(lower) and lower[i] == "#":
                i += 1
        elif lower.startswith("/*"):
            i = 2
            if i < len(lower) and lower[i] == "*":
                i += 1

        while i < len(lower) and lower[i].isspace():
            i += 1
        while i < len(lower) and lower[i] == "*":
            i += 1
            while i < len(lower) and lower[i].isspace():
                i += 1

        def is_word(ch: str) -> bool:
            return ch.isalnum() or ch == "_"

        def starts_with_word(word: str) -> bool:
            if not lower.startswith(word, i):
                return False
            end = i + len(word)
            if end >= len(lower):
                return True
            return not is_word(lower[end])

        if (
            starts_with_word("todo")
            or starts_with_word("fixme")
            or starts_with_word("stub")
            or starts_with_word("placeholder")
            or starts_with_word("unimplemented")
            or starts_with_word("notimplemented")
        ):
            return True

        if lower.startswith("not implemented", i) or lower.startswith("not yet implemented", i):
            return True

        return False

    # ════════════════════════════════════════════════════════════════════════════
    # Internal conversion methods
    # ════════════════════════════════════════════════════════════════════════════

    def _convert_node(
        self, node: tree_sitter.Node, source: bytes, lang: Language
    ) -> Tree:
        """Convert a tree-sitter node to our normalized Tree structure."""
        map_fn = _NODE_MAP_FN[lang]
        normalized_type = map_fn(node.type)
        tree_node = Tree(node_type=int(normalized_type), label=node.type)

        if node.child_count == 0:
            text = node.text
            if text:
                tree_node.label = text.decode(errors="replace")
        else:
            for child in node.children:
                if child.is_named:
                    tree_node.add_child(self._convert_node(child, source, lang))
                else:
                    op = child.type
                    op_type = self._classify_operator(op)
                    if op_type is not None:
                        op_node = Tree(node_type=int(op_type), label=op)
                        tree_node.add_child(op_node)

        return tree_node

    @staticmethod
    def _classify_operator(op: str) -> NodeType | None:
        """Classify an operator into a NodeType."""
        if op in _ARITH_OPS:
            return NodeType.ARITHMETIC_OP
        if op in _CMP_OPS:
            return NodeType.COMPARISON_OP
        if op in _LOGIC_OPS:
            return NodeType.LOGICAL_OP
        if op in _BIT_OPS:
            return NodeType.BITWISE_OP
        if op in _ASSIGN_OPS:
            return NodeType.ASSIGNMENT_OP
        return None

    def _extract_comments_recursive(
        self,
        node: tree_sitter.Node,
        source: bytes,
        lang: Language,
        stats: CommentStats,
    ) -> None:
        """Recursively extract comments from AST."""
        type_s = node.type
        is_comment = False
        is_doc = False

        if lang == Language.KOTLIN:
            is_comment = type_s in ("line_comment", "multiline_comment")
        elif lang == Language.CPP:
            is_comment = type_s == "comment"
        elif lang == Language.RUST:
            is_comment = type_s in ("line_comment", "block_comment")
        elif lang == Language.PYTHON:
            is_comment = type_s == "comment"

        if is_comment:
            text = (node.text or b"").decode(errors="replace")
            lines = text.count("\n") + 1
            stats.total_comment_lines += lines

            if lang == Language.KOTLIN:
                is_doc = text.startswith("/**")
            elif lang == Language.CPP:
                is_doc = text.startswith("/**") or text.startswith("///") or text.startswith("//!")
            elif lang == Language.RUST:
                is_doc = text.startswith("///") or text.startswith("//!") or text.startswith("/**")

            if is_doc:
                stats.doc_comment_count += 1
                stats.total_doc_lines += lines
                stats.doc_texts.append(text)
                _tokenize_doc(text, stats.word_freq)
            elif text.startswith("/*"):
                stats.block_comment_count += 1
            else:
                stats.line_comment_count += 1

        for child in node.children:
            self._extract_comments_recursive(child, source, lang, stats)

    def _extract_identifiers_recursive(
        self, node: tree_sitter.Node, stats: IdentifierStats
    ) -> None:
        """Recursively extract identifiers from AST."""
        if node.type in (
            "identifier",
            "simple_identifier",
            "type_identifier",
            "field_identifier",
            "property_identifier",
        ):
            text = (node.text or b"").decode(errors="replace")
            if len(text) > 1 and text not in ("it", "this"):
                stats.add_identifier(text)

        for child in node.children:
            self._extract_identifiers_recursive(child, stats)

    def _extract_functions_recursive(
        self,
        node: tree_sitter.Node,
        source: bytes,
        lang: Language,
        functions: list[tuple[str, Tree]],
    ) -> None:
        """Recursively extract functions from AST."""
        is_func = False
        type_s = node.type
        if lang == Language.RUST:
            is_func = type_s == "function_item"
        elif lang == Language.KOTLIN:
            is_func = type_s == "function_declaration"
        elif lang == Language.CPP:
            is_func = type_s in ("function_definition", "function_declarator")
        elif lang == Language.PYTHON:
            is_func = type_s == "function_definition"

        if is_func:
            func_name = ""
            for child in node.children:
                ct = child.type
                if (
                    (lang == Language.RUST and ct == "identifier")
                    or (lang == Language.KOTLIN and ct == "simple_identifier")
                    or (lang == Language.CPP and ct in ("identifier", "field_identifier"))
                    or (lang == Language.PYTHON and ct == "identifier")
                ):
                    func_name = (child.text or b"").decode(errors="replace")
                    break
            func_tree = self._convert_node(node, source, lang)
            functions.append((func_name, func_tree))

        for child in node.children:
            self._extract_functions_recursive(child, source, lang, functions)

    def _extract_function_infos_recursive(
        self,
        node: tree_sitter.Node,
        source: bytes,
        lang: Language,
        functions: list[FunctionInfo],
    ) -> None:
        """Recursively extract function infos from AST."""
        is_func = False
        type_s = node.type
        if lang == Language.RUST:
            is_func = type_s == "function_item"
        elif lang == Language.KOTLIN:
            is_func = type_s == "function_declaration"
        elif lang == Language.CPP:
            is_func = type_s in ("function_definition", "function_declarator")
        elif lang == Language.PYTHON:
            is_func = type_s == "function_definition"

        if is_func:
            func_name = ""
            for child in node.children:
                ct = child.type
                if (
                    (lang == Language.RUST and ct == "identifier")
                    or (lang == Language.KOTLIN and ct == "simple_identifier")
                    or (lang == Language.CPP and ct in ("identifier", "field_identifier"))
                    or (lang == Language.PYTHON and ct == "identifier")
                ):
                    func_name = (child.text or b"").decode(errors="replace")
                    break
            
            func_tree = self._convert_node(node, source, lang)
            
            # Extract identifiers from the function
            ids = IdentifierStats()
            self._extract_identifiers_recursive(node, ids)
            
            # Check for stub markers in the function body
            body_node = self._extract_function_body(node, lang)
            has_stub = False
            if body_node:
                body_text = (body_node.text or b"").decode(errors="replace")
                has_stub = self.text_has_stub_markers(body_text)
            
            info = FunctionInfo(
                name=func_name,
                body_tree=func_tree,
                identifiers=ids,
                has_stub_markers=has_stub,
            )
            functions.append(info)

        for child in node.children:
            self._extract_function_infos_recursive(child, source, lang, functions)

    @staticmethod
    def _extract_function_body(node: tree_sitter.Node, lang: Language) -> tree_sitter.Node | None:
        """Extract function body node.
        
        Prefer body field when available.
        """
        try:
            body_node = node.child_by_field_name("body")
            if body_node and body_node.type:
                return body_node
        except Exception:
            pass

        # Fallback for grammars where body has a dedicated node.
        for child in node.children:
            if (
                (lang == Language.CPP and child.type in {"compound_statement", "function_body"})
                or (
                    lang in (Language.RUST, Language.KOTLIN, Language.PYTHON)
                    and child.type in {"body", "block", "function_body"}
                )
            ):
                return child
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize_doc(text: str, word_freq: dict[str, int]) -> None:
    """Tokenize documentation text into words for frequency analysis."""
    current: list[str] = []
    for ch in text:
        if ch.isalnum():
            current.append(ch.lower())
        else:
            if len(current) >= 3:
                word = "".join(current)
                if word not in _STOP_WORDS:
                    word_freq[word] = word_freq.get(word, 0) + 1
            current.clear()
    if len(current) >= 3:
        word = "".join(current)
        if word not in _STOP_WORDS:
            word_freq[word] = word_freq.get(word, 0) + 1


def parse_language(s: str) -> Language:
    """Parse a language string into a Language enum.
    
    Faithful transliteration from ast_parser.hpp.
    """
    s = s.lower().strip()
    for lang in Language:
        if lang.value == s:
            return lang
    raise ValueError(f"Unknown language: {s!r}. Expected one of: rust, kotlin, cpp, python")

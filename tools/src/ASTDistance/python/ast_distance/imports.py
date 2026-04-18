"""Import extraction from source files using tree-sitter.

Faithful transliteration of include/imports.hpp from the C++ ast_distance tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter
import tree_sitter_rust
import tree_sitter_kotlin
import tree_sitter_cpp
import tree_sitter_python as tree_sitter_py


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# Faithful transliteration from imports.hpp:26-106
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PackageDecl:
    """Represents a package/namespace declaration.
    
    Faithful transliteration of PackageDecl from imports.hpp:26-79.
    """
    raw: str = ""
    path: str = ""
    parts: list[str] = field(default_factory=list)

    def last(self) -> str:
        """Get the last component (usually the module/class name context).
        
        Faithful transliteration of last() from imports.hpp:32-34.
        """
        return self.parts[-1] if self.parts else ""

    def parent(self) -> str:
        """Get without the last component (parent package).
        
        Faithful transliteration of parent() from imports.hpp:37-45.
        """
        if len(self.parts) <= 1:
            return ""
        return ".".join(self.parts[:-1])

    @staticmethod
    def normalize(s: str) -> str:
        """Normalize for comparison (lowercase, no underscores).
        
        Faithful transliteration of normalize() from imports.hpp:48-55.
        """
        return "".join(c.lower() for c in s if c not in ("_", "-"))

    def similarity_to(self, other: PackageDecl) -> float:
        """Check if this package path matches another (fuzzy).
        
        Faithful transliteration of similarity_to() from imports.hpp:58-78.
        """
        if not self.parts or not other.parts:
            return 0.0

        # Count matching parts from the end (most specific first)
        matches = 0
        min_len = min(len(self.parts), len(other.parts))

        for i in range(min_len):
            a = self.normalize(self.parts[-(i + 1)])
            b = self.normalize(other.parts[-(i + 1)])
            if a == b:
                matches += 1
            elif a in b or b in a:
                # Substring match counts
                matches += 1
            else:
                # Stop at first mismatch
                break

        return matches / min_len


@dataclass
class Import:
    """Represents an import/use statement.
    
    Faithful transliteration of Import from imports.hpp:84-106.
    """
    raw: str = ""
    module_path: str = ""
    item: str = ""
    is_wildcard: bool = False

    def to_file_path(self) -> str:
        """Convert module path to potential file path.
        
        e.g., "ratatui::style::Color" -> "ratatui/style" or "ratatui/style/color"
        
        Faithful transliteration of to_file_path() from imports.hpp:91-105.
        """
        path = self.module_path
        # Replace :: with /
        path = path.replace("::", "/")
        # Replace . with /
        path = path.replace(".", "/")
        return path


# ══════════════════════════════════════════════════════════════════════════════
# Language setup
# ══════════════════════════════════════════════════════════════════════════════

_LANGUAGES: dict[str, tree_sitter.Language] = {
    "rust": tree_sitter.Language(tree_sitter_rust.language()),
    "kotlin": tree_sitter.Language(tree_sitter_kotlin.language()),
    "cpp": tree_sitter.Language(tree_sitter_cpp.language()),
    "python": tree_sitter.Language(tree_sitter_py.language()),
}

_EXT_TO_LANG: dict[str, str] = {
    ".rs": "rust",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".h": "cpp",
    ".py": "python",
}


# ══════════════════════════════════════════════════════════════════════════════
# ImportExtractor class
# Faithful transliteration from imports.hpp:111-200+
# ══════════════════════════════════════════════════════════════════════════════

class ImportExtractor:
    """Extract imports from source files using tree-sitter.
    
    Faithful transliteration of class ImportExtractor from imports.hpp.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, tree_sitter.Parser] = {}

    def _get_parser(self, lang: str) -> tree_sitter.Parser:
        if lang not in self._parsers:
            self._parsers[lang] = tree_sitter.Parser(_LANGUAGES[lang])
        return self._parsers[lang]

    def _parse(self, source: bytes, lang: str) -> Optional[tree_sitter.Node]:
        parser = self._get_parser(lang)
        tree = parser.parse(source)
        return tree.root_node if tree else None

    # ════════════════════════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════════════════════════

    def extract_from_file(self, filepath: str | Path) -> list[Import]:
        """Extract all imports from a source file.
        
        Faithful transliteration of extract_*_imports methods from imports.hpp.
        """
        p = Path(filepath)
        lang = _EXT_TO_LANG.get(p.suffix)
        if not lang:
            return []
        try:
            source = p.read_bytes()
        except OSError:
            return []
        return self._extract(source, lang)

    def extract_rust_imports(self, source: str) -> list[Import]:
        """Extract all imports from a Rust file.
        
        Faithful transliteration from imports.hpp:130-145.
        """
        return self._extract(source.encode(), "rust")

    def extract_kotlin_imports(self, source: str) -> list[Import]:
        """Extract all imports from a Kotlin file.
        
        Faithful transliteration from imports.hpp:150-165.
        """
        return self._extract(source.encode(), "kotlin")

    def extract_cpp_imports(self, source: str) -> list[Import]:
        """Extract all imports from a C++ file.
        
        Faithful transliteration from imports.hpp:170-185.
        """
        return self._extract(source.encode(), "cpp")

    def extract_python_imports(self, source: str) -> list[Import]:
        """Extract all imports from a Python file.
        
        Faithful transliteration from imports.hpp:190-200+.
        """
        return self._extract(source.encode(), "python")

    def extract_package_from_file(self, filepath: str | Path) -> PackageDecl:
        """Extract package declaration from a source file."""
        p = Path(filepath)
        lang = _EXT_TO_LANG.get(p.suffix)
        if not lang:
            return PackageDecl()
        try:
            source = p.read_bytes()
        except OSError:
            return PackageDecl()
        return self._extract_package(source, lang, str(filepath))

    # ════════════════════════════════════════════════════════════════════════════
    # Internal dispatch
    # ════════════════════════════════════════════════════════════════════════════

    def _extract(self, source: bytes, lang: str) -> list[Import]:
        root = self._parse(source, lang)
        if root is None:
            return []
        imports: list[Import] = []
        if lang == "rust":
            self._extract_rust_imports(root, source, imports)
        elif lang == "kotlin":
            self._extract_kotlin_imports(root, source, imports)
        elif lang == "cpp":
            self._extract_cpp_imports(root, source, imports)
        elif lang == "python":
            self._extract_python_imports(root, source, imports)
        return imports

    def _extract_package(self, source: bytes, lang: str, filepath: str) -> PackageDecl:
        if lang == "rust":
            return _derive_rust_module(filepath)
        elif lang == "kotlin":
            root = self._parse(source, lang)
            if root is None:
                return PackageDecl()
            pkg = PackageDecl()
            self._extract_kotlin_package(root, source, pkg)
            return pkg
        elif lang == "cpp":
            root = self._parse(source, lang)
            pkg = PackageDecl()
            if root:
                self._extract_cpp_namespace(root, source, pkg)
            if not pkg.parts:
                return _derive_from_path(filepath, skip=("src", "include"))
            return pkg
        elif lang == "python":
            return _derive_python_module(filepath)
        return PackageDecl()

    # ════════════════════════════════════════════════════════════════════════════
    # Rust extraction
    # ════════════════════════════════════════════════════════════════════════════

    def _extract_rust_imports(
        self, node: tree_sitter.Node, source: bytes, imports: list[Import]
    ) -> None:
        """Extract imports from Rust AST.
        
        Faithful transliteration from imports.hpp recursive methods.
        """
        if node.type == "use_declaration":
            raw = _text(node, source)
            imp = Import(raw=raw, is_wildcard="::*" in raw)
            for child in node.children:
                if child.type in (
                    "scoped_identifier",
                    "identifier",
                    "use_wildcard",
                    "use_list",
                    "scoped_use_list",
                ):
                    imp.module_path = _text(child, source)
                    if imp.module_path.startswith("use "):
                        imp.module_path = imp.module_path[4:]
                    imp.module_path = imp.module_path.rstrip(";")
                    break
            sep = imp.module_path.rfind("::")
            imp.item = imp.module_path[sep + 2 :] if sep >= 0 else imp.module_path
            if imp.module_path:
                imports.append(imp)
        for child in node.children:
            self._extract_rust_imports(child, source, imports)

    # ════════════════════════════════════════════════════════════════════════════
    # Kotlin extraction
    # ════════════════════════════════════════════════════════════════════════════

    def _extract_kotlin_imports(
        self, node: tree_sitter.Node, source: bytes, imports: list[Import]
    ) -> None:
        """Extract imports from Kotlin AST.
        
        Faithful transliteration from imports.hpp recursive methods.
        """
        if node.type == "import_header":
            raw = _text(node, source)
            imp = Import(raw=raw, is_wildcard=".*" in raw)
            for child in node.children:
                if child.type == "identifier":
                    imp.module_path = _text(child, source)
                    break
            if not imp.module_path:
                imp.module_path = raw
            if imp.module_path.startswith("import "):
                imp.module_path = imp.module_path[7:]
            imp.module_path = imp.module_path.strip()
            dot = imp.module_path.rfind(".")
            imp.item = imp.module_path[dot + 1 :] if dot >= 0 else imp.module_path
            if imp.module_path:
                imports.append(imp)
        for child in node.children:
            self._extract_kotlin_imports(child, source, imports)

    def _extract_kotlin_package(
        self, node: tree_sitter.Node, source: bytes, pkg: PackageDecl
    ) -> None:
        """Extract package declaration from Kotlin AST."""
        if node.type == "package_header":
            pkg.raw = _text(node, source)
            for child in node.children:
                if child.type == "identifier":
                    pkg.path = _text(child, source)
                    break
            if not pkg.path:
                pkg.path = pkg.raw
            if pkg.path.startswith("package "):
                pkg.path = pkg.path[8:]
            pkg.path = pkg.path.strip().rstrip(";")
            pkg.parts = [p for p in pkg.path.split(".") if p]
            return
        for child in node.children:
            if pkg.path:
                return
            self._extract_kotlin_package(child, source, pkg)

    # ════════════════════════════════════════════════════════════════════════════
    # C++ extraction
    # ════════════════════════════════════════════════════════════════════════════

    def _extract_cpp_imports(
        self, node: tree_sitter.Node, source: bytes, imports: list[Import]
    ) -> None:
        """Extract imports from C++ AST."""
        if node.type == "preproc_include":
            raw = _text(node, source)
            imp = Import(raw=raw, is_wildcard=False)
            for child in node.children:
                if child.type in ("string_literal", "system_lib_string"):
                    path = _text(child, source)
                    if len(path) >= 2 and path[0] in ('"', "<"):
                        path = path[1:-1]
                    imp.module_path = path.replace("/", "::")
                    for ext in (".hpp", ".h"):
                        if imp.module_path.endswith(ext):
                            imp.module_path = imp.module_path[: -len(ext)]
                            break
                    break
            sep = imp.module_path.rfind("::")
            imp.item = imp.module_path[sep + 2 :] if sep >= 0 else imp.module_path
            if imp.module_path:
                imports.append(imp)
        for child in node.children:
            self._extract_cpp_imports(child, source, imports)

    def _extract_cpp_namespace(
        self, node: tree_sitter.Node, source: bytes, pkg: PackageDecl
    ) -> None:
        """Extract namespace from C++ AST."""
        if node.type == "namespace_definition":
            for child in node.children:
                if child.type in ("namespace_identifier", "identifier"):
                    name = _text(child, source)
                    if name:
                        pkg.parts.append(name)
                        pkg.path = ".".join(pkg.parts)
                    break
            for child in node.children:
                if child.type == "declaration_list":
                    self._extract_cpp_namespace(child, source, pkg)
                    return
            return
        for child in node.children:
            if pkg.path:
                return
            self._extract_cpp_namespace(child, source, pkg)

    # ════════════════════════════════════════════════════════════════════════════
    # Python extraction
    # ════════════════════════════════════════════════════════════════════════════

    def _extract_python_imports(
        self, node: tree_sitter.Node, source: bytes, imports: list[Import]
    ) -> None:
        """Extract imports from Python AST."""
        if node.type in ("import_statement", "import_from_statement"):
            raw = _text(node, source)
            _parse_python_import_raw(raw, imports)
        for child in node.children:
            self._extract_python_imports(child, source, imports)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# Faithful transliteration from imports.hpp helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _text(node: tree_sitter.Node, source: bytes) -> str:
    """Get text content of a node."""
    return (node.text or b"").decode(errors="replace")


def _derive_rust_module(filepath: str) -> PackageDecl:
    """Derive Rust module from file path."""
    p = Path(filepath)
    parts = [s for s in p.parent.parts if s not in (".", "src", "lib") and s]
    stem = p.stem
    if stem not in ("mod", "lib"):
        parts.append(stem)
    return PackageDecl(parts=parts, path=".".join(parts))


def _derive_python_module(filepath: str) -> PackageDecl:
    """Derive Python module from file path."""
    p = Path(filepath)
    parts = [s for s in p.parent.parts if s not in (".", "src", "lib") and s]
    stem = p.stem
    if stem and stem != "__init__":
        parts.append(stem)
    return PackageDecl(parts=parts, path=".".join(parts))


def _derive_from_path(filepath: str, skip: tuple[str, ...] = ()) -> PackageDecl:
    """Derive module from file path (generic)."""
    p = Path(filepath)
    parts = [s for s in p.parent.parts if s not in (".", *skip) and s]
    stem = p.stem
    if stem:
        parts.append(stem)
    return PackageDecl(parts=parts, path=".".join(parts))


def _parse_python_import_raw(raw: str, imports: list[Import]) -> None:
    """Parse Python import statement."""
    s = raw.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    if s.startswith("import "):
        rest = s[7:]
        for part in _split_csv(rest):
            as_pos = part.find(" as ")
            if as_pos >= 0:
                part = part[:as_pos].strip()
            if not part:
                continue
            dot = part.rfind(".")
            item = part if dot < 0 else part[dot + 1 :]
            imports.append(Import(raw=raw, module_path=part, item=item, is_wildcard=False))
    elif s.startswith("from "):
        ip = s.find(" import ")
        if ip < 0:
            return
        module = s[5:ip].strip()
        items_s = s[ip + 8 :].strip()
        if items_s.startswith("(") and items_s.endswith(")"):
            items_s = items_s[1:-1].strip()
        if items_s == "*":
            imports.append(Import(raw=raw, module_path=module, item="*", is_wildcard=True))
            return
        for part in _split_csv(items_s):
            as_pos = part.find(" as ")
            if as_pos >= 0:
                part = part[:as_pos].strip()
            if not part:
                continue
            full = f"{module}.{part}"
            imports.append(Import(raw=raw, module_path=full, item=part, is_wildcard=False))


def _split_csv(s: str) -> list[str]:
    """Split comma-separated values, handling parentheses."""
    out: list[str] = []
    cur: list[str] = []
    paren = 0
    for ch in s:
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        if ch == "," and paren == 0:
            t = "".join(cur).strip()
            if t:
                out.append(t)
            cur.clear()
        else:
            cur.append(ch)
    t = "".join(cur).strip()
    if t:
        out.append(t)
    return out

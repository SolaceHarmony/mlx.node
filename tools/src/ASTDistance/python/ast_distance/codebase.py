"""Codebase management — transliterated from codebase.hpp."""

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from .imports import PackageDecl, Import, ImportExtractor
from .ast_parser import ASTParser, Language, FunctionInfo
from .porting_utils import PortingAnalyzer, FileStats, TodoItem, LintError
from .similarity import ASTSimilarity


def _has_valid_ext(path: str, language: str) -> bool:
    """Check if path has a valid extension for the given language."""
    if language == "rust":
        return path.endswith(".rs")
    elif language == "kotlin":
        return path.endswith(".kt") or path.endswith(".kts")
    elif language == "cpp":
        return (path.endswith(".cpp") or path.endswith(".hpp") or
                path.endswith(".cc") or path.endswith(".h"))
    elif language == "python":
        return path.endswith(".py")
    return False


def _is_header_file(ext: str) -> bool:
    """Check if extension is a header file."""
    return ext in (".hpp", ".h", ".hxx", ".hh")


class SourceFile:
    """Represents a source file with its metadata."""
    
    def __init__(self):
        self.paths: List[str] = []
        self.relative_path: str = ""
        self.filename: str = ""
        self.stem: str = ""
        self.qualified_name: str = ""
        self.extension: str = ""
        
        self.package = PackageDecl()
        self.imports: List[Import] = []
        self.imported_by: Set[str] = set()
        self.depends_on: Set[str] = set()
        
        self.dependent_count: int = 0
        self.dependency_count: int = 0
        
        self.similarity_score: float = 0.0
        self.matched_file: str = ""
        
        self.transliterated_from: str = ""
        self.line_count: int = 0
        self.code_lines: int = 0
        self.is_stub: bool = False
        self.todos: List[TodoItem] = []
        self.lint_errors: List[LintError] = []

    def identity(self) -> str:
        """Get the 'identity' for matching - last part of package + filename."""
        if self.package.parts:
            return self.package.path
        return self.qualified_name

    @staticmethod
    def make_qualified_name(rel_path: str) -> str:
        """Compute qualified name from path."""
        p = Path(rel_path)
        
        parts = []
        for part in p.parts[:-1]:
            if part and part != "." and part != "src":
                parts.append(part)
        
        stem = p.stem
        
        if parts:
            return parts[-1] + "." + stem
        return stem

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize name for matching (snake_case <-> PascalCase)."""
        result = ""
        prev_lower = False
        
        for c in name:
            if c == '_':
                continue
            
            if c.isupper() and prev_lower and result:
                result += c.lower()
            else:
                result += c.lower()
            
            prev_lower = c.islower()
        
        return result

    @staticmethod
    def to_pascal_case(name: str) -> str:
        """Convert snake_case to PascalCase for Kotlin filename generation.
        
        Example: "value" -> "Value", "my_file_name" -> "MyFileName"
        """
        result = ""
        capitalize_next = True
        
        for c in name:
            if c == '_':
                capitalize_next = True
                continue
            
            if capitalize_next:
                result += c.upper()
                capitalize_next = False
            else:
                result += c
        
        return result


class Codebase:
    """Manages a codebase - scans files, extracts imports, builds dependency graph."""
    
    def __init__(self, root: str, lang: str):
        self.root_path = root
        self.language = lang
        self.files: Dict[str, SourceFile] = {}
        self.by_stem: Dict[str, List[str]] = {}
        self.by_qualified: Dict[str, str] = {}

    def scan(self):
        """Scan directory and build file list."""
        root = Path(self.root_path)
        
        if root.is_file():
            if _has_valid_ext(str(root), self.language):
                sf = SourceFile()
                sf.paths.append(str(root))
                sf.relative_path = root.name
                sf.filename = root.name
                sf.stem = root.stem
                sf.extension = root.suffix
                sf.qualified_name = SourceFile.make_qualified_name(sf.relative_path)
                
                self.files[sf.relative_path] = sf
                self.by_stem[sf.stem] = [sf.relative_path]
                self.by_qualified[sf.qualified_name] = sf.relative_path
            return
        
        suffixes = [".common", ".concurrent", ".native", ".common_native", ".darwin", ".apple"]
        
        for entry in root.rglob("*"):
            if not entry.is_file():
                continue
            
            path = str(entry)
            if not _has_valid_ext(path, self.language):
                continue
            
            if any(skip in path for skip in ("/target/", "/build/", "/build_", "/_deps/")):
                continue
            
            rel_path = str(Path(path).relative_to(root))
            stem = entry.stem
            filename = entry.name
            extension = entry.suffix
            
            directory = str(Path(rel_path).parent)
            normalized_stem = stem
            
            for suffix in suffixes:
                if (len(normalized_stem) > len(suffix) and 
                    normalized_stem.endswith(suffix)):
                    normalized_stem = normalized_stem[:-len(suffix)]
                    break
            
            logical_key = directory if directory and directory != "." else normalized_stem
            if "/" not in logical_key and "\\" not in logical_key:
                logical_key = normalized_stem
            
            if logical_key in self.files:
                self.files[logical_key].paths.append(path)
                if extension in (".hpp", ".h"):
                    self.files[logical_key].filename = filename
                    self.files[logical_key].extension = extension
                    self.files[logical_key].relative_path = rel_path
            else:
                sf = SourceFile()
                sf.paths.append(path)
                sf.relative_path = rel_path
                sf.filename = filename
                sf.stem = stem
                sf.extension = extension
                sf.qualified_name = SourceFile.make_qualified_name(rel_path)
                
                self.files[logical_key] = sf
                if sf.stem not in self.by_stem:
                    self.by_stem[sf.stem] = []
                self.by_stem[sf.stem].append(logical_key)
                self.by_qualified[sf.qualified_name] = logical_key
        
        for stem, paths in self.by_stem.items():
            if len(paths) > 1:
                paths.sort(key=lambda p: (
                    not _is_header_file(self.files[p].extension),
                    len(p)
                ))
                
                seen_qualified = set()
                for path in paths:
                    sf = self.files[path]
                    if sf.qualified_name in seen_qualified:
                        p = Path(sf.relative_path)
                        full_qualified = ""
                        for part in p.parts[:-1]:
                            if part and part != "." and part != "src":
                                if full_qualified:
                                    full_qualified += "."
                                full_qualified += part
                        if full_qualified:
                            full_qualified += "."
                        full_qualified += sf.stem
                        sf.qualified_name = full_qualified
                    seen_qualified.add(sf.qualified_name)
                    self.by_qualified[sf.qualified_name] = path

    def extract_imports(self):
        """Extract imports and packages from all files."""
        extractor = ImportExtractor()
        
        for path, sf in self.files.items():
            for p in sf.paths:
                file_imports = extractor.extract_from_file(p)
                sf.imports.extend(file_imports)
                
                if self.language == "python":
                    if not sf.package.parts:
                        sf.package = self._derive_python_module(sf.relative_path)
                elif not sf.package.parts:
                    sf.package = extractor.extract_package_from_file(p)
            
            sf.dependency_count = len(sf.imports)

    def extract_porting_data(self):
        """Extract porting analysis data."""
        for path, sf in self.files.items():
            all_parts_stub = len(sf.paths) > 0
            
            for p in sf.paths:
                if not sf.transliterated_from:
                    sf.transliterated_from = PortingAnalyzer.extract_transliterated_from(p)
                
                stats = PortingAnalyzer.analyze_file(p)
                sf.line_count += stats.line_count
                sf.code_lines += stats.code_lines
                sf.todos.extend(stats.todos)
                
                lints = PortingAnalyzer.lint_file(p)
                sf.lint_errors.extend(lints)
                
                all_parts_stub = all_parts_stub and stats.is_stub
            
            sf.is_stub = all_parts_stub and sf.code_lines <= 100

    def transliteration_map(self) -> Dict[str, str]:
        """Build map of transliterated_from paths to files."""
        result = {}
        for path, sf in self.files.items():
            if sf.transliterated_from:
                result[sf.transliterated_from] = path
        return result

    def build_dependency_graph(self):
        """Build dependency graph - resolve imports to actual files."""
        for path, sf in self.files.items():
            for imp in sf.imports:
                resolved = self._resolve_import(imp)
                if resolved and resolved != path:
                    sf.depends_on.add(resolved)
                    self.files[resolved].imported_by.add(path)
        
        for path, sf in self.files.items():
            sf.dependent_count = len(sf.imported_by)

    def ranked_by_dependents(self) -> List[SourceFile]:
        """Get files sorted by dependent count (most depended-on first)."""
        result = list(self.files.values())
        result.sort(key=lambda sf: sf.dependent_count, reverse=True)
        return result

    def leaf_files(self) -> List[SourceFile]:
        """Get leaf files (no dependents - safe to port first)."""
        return [sf for sf in self.files.values() if sf.dependent_count == 0]

    def root_files(self, min_dependents: int = 3) -> List[SourceFile]:
        """Get root files (many dependents - core infrastructure)."""
        result = [sf for sf in self.files.values() if sf.dependent_count >= min_dependents]
        result.sort(key=lambda sf: sf.dependent_count, reverse=True)
        return result

    def print_summary(self):
        """Print codebase summary."""
        print(f"Codebase: {self.root_path} ({self.language})")
        print(f"  Files: {len(self.files)}")
        
        total_imports = 0
        max_dependents = 0
        most_depended = ""
        
        for sf in self.files.values():
            total_imports += len(sf.imports)
            if sf.dependent_count > max_dependents:
                max_dependents = sf.dependent_count
                most_depended = sf.qualified_name
        
        print(f"  Total imports: {total_imports}")
        if most_depended:
            print(f"  Most depended: {most_depended} ({max_dependents} dependents)")

    def _derive_python_module(self, rel_path: str) -> PackageDecl:
        """Derive Python package from relative path."""
        pkg = PackageDecl()
        p = Path(rel_path)
        
        parts = []
        for part in p.parts[:-1]:
            if part and part != "." and part != "src" and part != "lib":
                parts.append(part)
        
        stem = p.stem
        if stem and stem != "__init__":
            parts.append(stem)
        
        pkg.parts = parts
        pkg.path = ".".join(parts)
        
        return pkg

    def _resolve_import(self, imp: Import) -> str:
        """Resolve import to a file."""
        module = imp.module_path
        item = imp.item
        
        if item == "*":
            sep = "::" if self.language == "rust" else "."
            last_sep = module.rfind(sep)
            if last_sep != -1:
                item = module[last_sep + (2 if self.language == "rust" else 1):]
        
        normalized = SourceFile.normalize_name(item)
        
        for stem, paths in self.by_stem.items():
            if SourceFile.normalize_name(stem) == normalized:
                return paths[0]
        
        return ""


class CodebaseComparator:
    """Compare two codebases and find matches."""
    
    class FunctionComparisonResult:
        def __init__(self):
            self.score: float = -1.0
            self.matched_pairs: int = 0
            self.source_total: int = 0
            self.target_total: int = 0
            self.unmatched_source: int = 0
            self.unmatched_target: int = 0
            self.has_source_stub: bool = False
            self.has_target_stub: bool = False

    class Match:
        def __init__(self):
            self.source_path: str = ""
            self.target_path: str = ""
            self.source_qualified: str = ""
            self.target_qualified: str = ""
            self.similarity: float = 0.0
            self.source_dependents: int = 0
            self.target_dependents: int = 0
            self.source_lines: int = 0
            self.target_lines: int = 0
            self.todo_count: int = 0
            self.lint_count: int = 0
            self.is_stub: bool = False
            self.matched_by_header: bool = False
            
            self.source_function_count: int = 0
            self.target_function_count: int = 0
            self.matched_function_count: int = 0
            self.function_coverage: float = 1.0
            
            self.source_doc_lines: int = 0
            self.target_doc_lines: int = 0
            self.source_doc_comments: int = 0
            self.target_doc_comments: int = 0
            self.doc_similarity: float = 0.0
            self.doc_coverage: float = 1.0
            self.doc_weighted: float = 0.0

        def doc_gap_ratio(self) -> float:
            if self.source_doc_lines == 0:
                return 0.0
            if self.target_doc_lines == 0:
                return 1.0
            ratio = 1.0 - (float(self.target_doc_lines) / float(self.source_doc_lines))
            return max(0.0, ratio)

    def __init__(self, src: Codebase, tgt: Codebase):
        self.source = src
        self.target = tgt
        self.matches: List[CodebaseComparator.Match] = []
        self.unmatched_source: List[str] = []
        self.unmatched_target: List[str] = []

    @staticmethod
    def is_header_file(file: SourceFile) -> bool:
        return _is_header_file(file.extension)

    @staticmethod
    def name_match_score(src: SourceFile, tgt: SourceFile) -> float:
        src_norm = SourceFile.normalize_name(src.stem)
        tgt_norm = SourceFile.normalize_name(tgt.stem)
        src_qual_norm = SourceFile.normalize_name(src.qualified_name)
        tgt_qual_norm = SourceFile.normalize_name(tgt.qualified_name)
        
        header_boost = 0.02 if CodebaseComparator.is_header_file(tgt) else 0.0
        
        if src_qual_norm == tgt_qual_norm:
            return 1.0 + header_boost
        
        src_dot = src.qualified_name.rfind('.')
        tgt_dot = tgt.qualified_name.rfind('.')
        src_parent = src.qualified_name[:src_dot] if src_dot != -1 else ""
        tgt_parent = tgt.qualified_name[:tgt_dot] if tgt_dot != -1 else ""
        
        if src_norm == tgt_norm and src_parent and tgt_parent:
            src_parent_norm = SourceFile.normalize_name(src_parent)
            tgt_parent_norm = SourceFile.normalize_name(tgt_parent)
            if src_parent_norm == tgt_parent_norm:
                return 0.95 + header_boost
        
        if src_norm == tgt_norm:
            return 0.7 + header_boost
        
        if tgt_norm.find(src_norm) != -1:
            ratio = float(len(src_norm)) / float(len(tgt_norm))
            return 0.5 + 0.2 * ratio + header_boost
        if src_norm.find(tgt_norm) != -1:
            ratio = float(len(tgt_norm)) / float(len(src_norm))
            return 0.5 + 0.2 * ratio + header_boost
        
        if src.package.parts and tgt.package.parts:
            pkg_sim = src.package.similarity_to(tgt.package)
            if pkg_sim > 0.5:
                return pkg_sim * 0.6 + header_boost
        
        if src.package.parts:
            src_last = PackageDecl.normalize(src.package.last())
            if src_last == tgt_norm or tgt_norm.find(src_last) != -1:
                return 0.5 + header_boost
        if tgt.package.parts:
            tgt_last = PackageDecl.normalize(tgt.package.last())
            if tgt_last == src_norm or src_norm.find(tgt_last) != -1:
                return 0.5 + header_boost
        
        if (SourceFile.normalize_name(src_parent) == SourceFile.normalize_name(tgt_parent) 
            and src_parent):
            return 0.4 + header_boost
        
        return 0.0

    def find_matches(self):
        """Find matching files between codebases."""
        matched_sources: Set[str] = set()
        matched_targets: Set[str] = set()
        
        header_candidates: List[Tuple[float, str, str]] = []
        
        for tgt_path, tgt_file in self.target.files.items():
            if not tgt_file.transliterated_from:
                continue
            
            for src_path, src_file in self.source.files.items():
                match_score = 0.0
                
                if tgt_file.transliterated_from.find(src_file.relative_path) != -1:
                    match_score = 1.0
                elif (tgt_file.transliterated_from.endswith("/" + src_file.filename) or
                      tgt_file.transliterated_from == src_file.filename):
                    tgt_dir = tgt_file.qualified_name
                    src_dir = src_file.qualified_name
                    tgt_dot = tgt_dir.rfind('.')
                    src_dot = src_dir.rfind('.')
                    if tgt_dot != -1:
                        tgt_dir = tgt_dir[:tgt_dot]
                    if src_dot != -1:
                        src_dir = src_dir[:src_dot]
                    
                    if (SourceFile.normalize_name(tgt_dir) == 
                        SourceFile.normalize_name(src_dir)):
                        match_score = 0.9
                    else:
                        match_score = 0.5
                elif any(tgt_file.transliterated_from.endswith("/" + src_file.stem + ext) 
                         for ext in (".kt", ".rs", ".py", ".cpp", ".cc", ".hpp", ".h")):
                    match_score = 0.3
                
                if match_score > 0.0:
                    header_candidates.append((match_score, src_path, tgt_path))
        
        header_candidates.sort(key=lambda x: (-x[0], 
            not CodebaseComparator.is_header_file(self.target.files.get(x[2], SourceFile())),
            len(x[2])))
        
        for score, src_path, tgt_path in header_candidates:
            if src_path in matched_sources or tgt_path in matched_targets:
                continue
            
            src_file = self.source.files[src_path]
            tgt_file = self.target.files[tgt_path]
            
            m = CodebaseComparator.Match()
            m.source_path = src_path
            m.target_path = tgt_path
            m.source_qualified = src_file.qualified_name
            m.target_qualified = tgt_file.qualified_name
            m.similarity = 0.0
            m.source_dependents = src_file.dependent_count
            m.target_dependents = tgt_file.dependent_count
            m.source_lines = src_file.line_count
            m.target_lines = tgt_file.line_count
            m.todo_count = len(tgt_file.todos)
            m.lint_count = len(tgt_file.lint_errors)
            m.is_stub = tgt_file.is_stub
            
            if not m.is_stub and src_file.code_lines > 20 and tgt_file.code_lines > 0:
                ratio = float(tgt_file.code_lines) / float(src_file.code_lines)
                if ratio < 0.30:
                    m.is_stub = True
            
            m.matched_by_header = True
            
            self.matches.append(m)
            matched_sources.add(src_path)
            matched_targets.add(tgt_path)
        
        candidates: List[Tuple[float, str, str]] = []
        
        for src_path, src_file in self.source.files.items():
            if src_path in matched_sources:
                continue
            
            for tgt_path, tgt_file in self.target.files.items():
                if tgt_path in matched_targets:
                    continue
                
                score = CodebaseComparator.name_match_score(src_file, tgt_file)
                if score > 0.4:
                    candidates.append((score, src_path, tgt_path))
        
        candidates.sort(key=lambda x: -x[0])
        
        for score, src_path, tgt_path in candidates:
            if src_path in matched_sources or tgt_path in matched_targets:
                continue
            
            src_file = self.source.files[src_path]
            tgt_file = self.target.files[tgt_path]
            
            m = CodebaseComparator.Match()
            m.source_path = src_path
            m.target_path = tgt_path
            m.source_qualified = src_file.qualified_name
            m.target_qualified = tgt_file.qualified_name
            m.similarity = 0.0
            m.source_dependents = src_file.dependent_count
            m.target_dependents = tgt_file.dependent_count
            m.source_lines = src_file.line_count
            m.target_lines = tgt_file.line_count
            m.todo_count = len(tgt_file.todos)
            m.lint_count = len(tgt_file.lint_errors)
            m.is_stub = tgt_file.is_stub
            
            if not m.is_stub and src_file.code_lines > 20 and tgt_file.code_lines > 0:
                ratio = float(tgt_file.code_lines) / float(src_file.code_lines)
                if ratio < 0.30:
                    m.is_stub = True
            
            m.matched_by_header = False
            
            self.matches.append(m)
            matched_sources.add(src_path)
            matched_targets.add(tgt_path)
        
        for src_path in self.source.files:
            if src_path not in matched_sources:
                self.unmatched_source.append(src_path)
        for tgt_path in self.target.files:
            if tgt_path not in matched_targets:
                self.unmatched_target.append(tgt_path)

    @staticmethod
    def _string_to_language(lang: str) -> Language:
        if lang == "rust":
            return Language.RUST
        if lang == "kotlin":
            return Language.KOTLIN
        if lang == "cpp":
            return Language.CPP
        if lang == "python":
            return Language.PYTHON
        return Language.KOTLIN

    @staticmethod
    def compare_function_sets(source_functions: List[FunctionInfo],
                              target_functions: List[FunctionInfo]) -> 'CodebaseComparator.FunctionComparisonResult':
        result = CodebaseComparator.FunctionComparisonResult()
        result.source_total = len(source_functions)
        result.target_total = len(target_functions)
        
        for func in source_functions:
            if func.has_stub_markers:
                result.has_source_stub = True
        for func in target_functions:
            if func.has_stub_markers:
                result.has_target_stub = True
        
        if not source_functions or not target_functions:
            return result
        
        candidates = []
        
        for i, source_func in enumerate(source_functions):
            for j, target_func in enumerate(target_functions):
                sim = 0.0
                # Guardrail: treat stub markers as a failure only when they appear in the
                # target function but not in the source function. Rust source may contain
                # legitimate TODO/FIXME comments without implying an incomplete port.
                if not (target_func.has_stub_markers and not source_func.has_stub_markers):
                    sim = ASTSimilarity.combined_similarity_with_content(
                        source_func.body_tree,
                        target_func.body_tree,
                        source_func.identifiers,
                        target_func.identifiers)
                
                candidates.append((sim, i, j))
        
        candidates.sort(key=lambda x: -x[0])
        
        source_used = [False] * len(source_functions)
        target_used = [False] * len(target_functions)
        
        total_score = 0.0
        for score, src_idx, tgt_idx in candidates:
            if source_used[src_idx] or target_used[tgt_idx]:
                continue
            source_used[src_idx] = True
            target_used[tgt_idx] = True
            total_score += score
            result.matched_pairs += 1
        
        result.unmatched_source = result.source_total - result.matched_pairs
        result.unmatched_target = result.target_total - result.matched_pairs
        
        denominator = max(result.source_total, result.target_total)
        if denominator > 0:
            result.score = total_score / float(denominator)
        
        return result

    @staticmethod
    def function_name_coverage(source_functions: List[FunctionInfo],
                              target_functions: List[FunctionInfo]) -> Dict[str, int]:
        from .ast_parser import IdentifierStats
        
        cov = {"source_total": 0, "target_total": 0, "matched": 0, "ratio": 1.0}
        
        tgt_names = []
        for f in target_functions:
            if f.name and f.name != "<anonymous>":
                tgt_names.append(IdentifierStats.canonicalize(f.name))
                cov["target_total"] += 1
        
        for f in source_functions:
            if f.name and f.name != "<anonymous>":
                cov["source_total"] += 1
                key = IdentifierStats.canonicalize(f.name)
                if key in tgt_names:
                    cov["matched"] += 1
                    tgt_names.remove(key)
        
        if cov["source_total"] > 0:
            cov["ratio"] = float(cov["matched"]) / float(cov["source_total"])
        
        return cov

    def compute_similarities(self):
        """Compute AST similarity for all matches."""
        parser = ASTParser()
        
        for m in self.matches:
            try:
                src_file = self.source.files[m.source_path]
                tgt_file = self.target.files[m.target_path]
                src_lang = CodebaseComparator._string_to_language(self.source.language)
                tgt_lang = CodebaseComparator._string_to_language(self.target.language)
                
                has_stubs = parser.has_stub_bodies_in_files(tgt_file.paths, tgt_lang)
                
                if has_stubs:
                    m.similarity = 0.0
                    m.is_stub = True
                else:
                    src_tree = parser.parse_file(src_file.paths, src_lang)
                    tgt_tree = parser.parse_file(tgt_file.paths, tgt_lang)
                    
                    from .node_types import NodeType
                    if src_tree:
                        src_tree.flatten_node_type(int(NodeType.PACKAGE))
                    if tgt_tree:
                        tgt_tree.flatten_node_type(int(NodeType.PACKAGE))
                    
                    src_ids = parser.extract_identifiers_from_file(src_file.paths, src_lang)
                    tgt_ids = parser.extract_identifiers_from_file(tgt_file.paths, tgt_lang)
                    
                    file_sim = ASTSimilarity.combined_similarity_with_content(
                        src_tree, tgt_tree, src_ids, tgt_ids)
                    
                    source_functions = parser.extract_function_infos_from_files(
                        src_file.paths, src_lang)
                    target_functions = parser.extract_function_infos_from_files(
                        tgt_file.paths, tgt_lang)
                    
                    fn_cov = CodebaseComparator.function_name_coverage(
                        source_functions, target_functions)
                    
                    m.source_function_count = fn_cov["source_total"]
                    m.target_function_count = fn_cov["target_total"]
                    m.matched_function_count = fn_cov["matched"]
                    m.function_coverage = fn_cov["ratio"]
                    
                    m.similarity = file_sim * fn_cov["ratio"]
                
                src_docs = parser.extract_comments_from_file(src_file.paths, src_lang)
                tgt_docs = parser.extract_comments_from_file(tgt_file.paths, tgt_lang)
                
                m.source_doc_lines = src_docs.total_doc_lines
                m.target_doc_lines = tgt_docs.total_doc_lines
                m.source_doc_comments = src_docs.doc_comment_count
                m.target_doc_comments = tgt_docs.doc_comment_count
                m.doc_similarity = src_docs.doc_cosine_similarity(tgt_docs)
                m.doc_coverage = src_docs.doc_line_coverage_capped(tgt_docs)
                m.doc_weighted = 0.5 * m.doc_similarity + 0.5 * m.doc_coverage
                
            except Exception:
                m.similarity = -1.0

    def ranked_for_porting(self) -> List[Match]:
        """Get matches sorted by priority for porting."""
        result = self.matches[:]
        result.sort(key=lambda m: m.source_dependents * (1.0 - m.similarity), reverse=True)
        return result

    def format_report(self) -> str:
        """Return the same output as `print_report`, as a string.

        ProjectHub and other callers use this to embed the report in higher-level summaries.
        """
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.print_report()
        return buf.getvalue()

    def print_report(self):
        """Print comparison report."""
        print("\n=== Codebase Comparison Report ===\n")
        
        print(f"Source: {self.source.root_path} ({len(self.source.files)} files)")
        print(f"Target: {self.target.root_path} ({len(self.target.files)} files)")
        print()
        
        print(f"Matched:   {len(self.matches)} files")
        print(f"Unmatched: {len(self.unmatched_source)} source, {len(self.unmatched_target)} target\n")
        
        if self.matches:
            print("=== Matched Files (by porting priority) ===\n")
            print(f"{'Source':<30}{'Target':<30}{'Similarity':<10}{'Dependents':<11}{'FunctionParity':<14}{'Priority':<10}")
            print("-" * 110)
            
            ranked = self.ranked_for_porting()
            for m in ranked:
                funcs = "-"
                if m.source_function_count > 0:
                    funcs = f"{m.matched_function_count}/{m.source_function_count}"
                priority = m.source_dependents * (1.0 - m.similarity)
                
                print(f"{m.source_qualified[:28]:<30}{m.target_qualified[:28]:<30}"
                      f"{m.similarity:<10.2f}{m.source_dependents:<11}"
                      f"{funcs:<14}{priority:<10.1f}")
        
        if self.unmatched_source:
            print("\n=== Missing from Target (need to port) ===\n")
            missing = [self.source.files[p] for p in self.unmatched_source]
            missing.sort(key=lambda sf: sf.dependent_count, reverse=True)
            
            print(f"{'File':<30}{'Deps':<8}Path")
            print("-" * 78)
            
            shown = 0
            for sf in missing:
                if shown >= 20:
                    print(f"... and {len(missing) - 20} more missing files")
                    break
                print(f"{sf.qualified_name[:28]:<30}{sf.dependent_count:<8}{sf.relative_path}")
                shown += 1

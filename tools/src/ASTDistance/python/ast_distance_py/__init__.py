"""ast_distance_py — faithful transliteration of ast_distance C++ to Python.

This package is a line-by-line port of the C++ ast_distance tool,
maintaining semantic parity while providing idiomatic Python APIs.

Modules:
    node_types: Normalized AST node types mapping across languages
    tree: Tree structure for AST representation
    ast_parser: AST parsing using tree-sitter
    imports: Import extraction and package declarations
    similarity: AST similarity metrics
    codebase: Codebase management and comparison
    porting_utils: TODO scanning, lint checks, stub detection
    task_manager: Task management for swarm agents
"""

from .node_types import (
    NodeType,
    node_type_name,
    rust_node_to_type,
    kotlin_node_to_type,
    cpp_node_to_type,
    python_node_to_type,
)
from .tree import Tree
from .ast_parser import (
    ASTParser,
    Language,
    IdentifierStats,
    CommentStats,
    FunctionInfo,
    parse_language,
)
from .imports import (
    PackageDecl,
    Import,
    ImportExtractor,
)
from .similarity import (
    ASTSimilarity,
    ComparisonReport,
    get_node_weight,
    histogram_cosine_similarity,
    node_type_jaccard,
    structure_similarity,
    combined_similarity,
    combined_similarity_with_content,
    tree_edit_distance,
    normalized_edit_distance,
)
from .codebase import (
    SourceFile,
    Codebase,
    CodebaseComparator,
)
from .porting_utils import (
    TodoItem,
    LintError,
    FileStats,
    PortingAnalyzer,
)
from .task_manager import (
    TaskStatus,
    FileLock,
    PortTask,
    TaskManager,
)

__all__ = [
    "NodeType",
    "node_type_name",
    "rust_node_to_type",
    "kotlin_node_to_type",
    "cpp_node_to_type",
    "python_node_to_type",
    "Tree",
    "ASTParser",
    "Language",
    "IdentifierStats",
    "CommentStats",
    "FunctionInfo",
    "parse_language",
    "PackageDecl",
    "Import",
    "ImportExtractor",
    "ASTSimilarity",
    "ComparisonReport",
    "get_node_weight",
    "histogram_cosine_similarity",
    "node_type_jaccard",
    "structure_similarity",
    "combined_similarity",
    "combined_similarity_with_content",
    "tree_edit_distance",
    "normalized_edit_distance",
    "SourceFile",
    "Codebase",
    "CodebaseComparator",
    "TodoItem",
    "LintError",
    "FileStats",
    "PortingAnalyzer",
    "TaskStatus",
    "FileLock",
    "PortTask",
    "TaskManager",
]

"""AST similarity metrics - transliterated from similarity.hpp."""

import math
from typing import List, Optional, TYPE_CHECKING
from .node_types import NodeType, node_type_name
from .tree import Tree

if TYPE_CHECKING:
    from .ast_parser import IdentifierStats


NUM_NODE_TYPES = int(NodeType.NUM_TYPES)


def get_node_weight(node_type: int) -> float:
    """Get semantic weight for a node type.
    
    Structural nodes (CLASS, FUNCTION, IF) have higher weight.
    Common nodes (VARIABLE, VAR_DECL) have lower weight.
    """
    # Clamp to valid range - tree-sitter can return types outside our enum
    if node_type < 0 or node_type >= NUM_NODE_TYPES:
        return 1.0  # Default weight for unknown types
    
    try:
        nt = NodeType(node_type)
    except ValueError:
        return 1.0  # Default weight for unknown types

    # Critical structural elements: weight 5.0
    if nt in (NodeType.CLASS, NodeType.STRUCT,
              NodeType.FUNCTION, NodeType.IF,
              NodeType.WHILE, NodeType.FOR,
              NodeType.SWITCH, NodeType.TRY,
              NodeType.INTERFACE, NodeType.ENUM):
        return 5.0

    # Important operations: weight 2.0
    if nt in (NodeType.CALL, NodeType.METHOD_CALL,
              NodeType.RETURN, NodeType.THROW,
              NodeType.LAMBDA, NodeType.COMPARISON_OP,
              NodeType.LOGICAL_OP):
        return 2.0

    # Operators: weight 1.5 (semantic awareness improvement)
    if nt in (NodeType.ARITHMETIC_OP, NodeType.BITWISE_OP,
              NodeType.ASSIGNMENT_OP):
        return 1.5

    # Common/boilerplate: weight 0.5
    if nt in (NodeType.VARIABLE, NodeType.VAR_DECL):
        return 0.5

    # Default weight
    return 1.0


def histogram_cosine_similarity(tree1: Tree, tree2: Tree) -> float:
    """Weighted cosine similarity based on node type histogram.
    
    Uses semantic importance weighting to prioritize structural differences.
    """
    hist1 = tree1.node_type_histogram(NUM_NODE_TYPES)
    hist2 = tree2.node_type_histogram(NUM_NODE_TYPES)

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for i in range(NUM_NODE_TYPES):
        weight = get_node_weight(i)
        weighted1 = weight * float(hist1[i])
        weighted2 = weight * float(hist2[i])

        dot += weighted1 * weighted2
        norm1 += weighted1 * weighted1
        norm2 += weighted2 * weighted2

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def histogram_cosine_similarity_macro(tree1: Tree, tree2: Tree) -> float:
    """Macro-friendly cosine similarity.
    
    Rust macro-heavy files (e.g. `macro_rules!`) often do not produce comparable AST shapes
    across languages because the Rust parser represents macro bodies as token trees.
    For such files we compute a cosine similarity over a small subset of node types
    (VARIABLE + UNKNOWN), which better captures whether the port contains a similar set of
    identifiers/tokens without over-penalizing language-specific structure.
    """
    hist1 = tree1.node_type_histogram(NUM_NODE_TYPES)
    hist2 = tree2.node_type_histogram(NUM_NODE_TYPES)

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for i in range(NUM_NODE_TYPES):
        nt = NodeType(i)
        weight = 0.0
        if nt in (NodeType.VARIABLE, NodeType.UNKNOWN):
            weight = 1.0

        weighted1 = weight * float(hist1[i])
        weighted2 = weight * float(hist2[i])

        dot += weighted1 * weighted2
        norm1 += weighted1 * weighted1
        norm2 += weighted2 * weighted2

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def node_type_jaccard(tree1: Tree, tree2: Tree) -> float:
    """Jaccard similarity of node type sets."""
    hist1 = tree1.node_type_histogram(NUM_NODE_TYPES)
    hist2 = tree2.node_type_histogram(NUM_NODE_TYPES)

    intersection = 0
    union_count = 0
    for i in range(NUM_NODE_TYPES):
        intersection += min(hist1[i], hist2[i])
        union_count += max(hist1[i], hist2[i])

    if union_count == 0:
        return 1.0
    return float(intersection) / float(union_count)


def structure_similarity(tree1: Tree, tree2: Tree) -> float:
    """Structure similarity based on tree shape.
    
    Compares depth, size, and branching factor.
    """
    size1 = tree1.size()
    size2 = tree2.size()
    depth1 = tree1.depth()
    depth2 = tree2.depth()

    # Size similarity (normalized)
    max_size = max(size1, size2)
    if max_size == 0:
        size_sim = 1.0
    else:
        size_sim = 1.0 - abs(size1 - size2) / float(max_size)

    # Depth similarity
    max_depth = max(depth1, depth2)
    if max_depth == 0:
        depth_sim = 1.0
    else:
        depth_sim = 1.0 - abs(depth1 - depth2) / float(max_depth)

    # Combine
    return 0.5 * size_sim + 0.5 * depth_sim


def combined_similarity(tree1: Tree, tree2: Tree,
                         hist_weight: float = 0.5,
                         struct_weight: float = 0.3,
                         jaccard_weight: float = 0.2) -> float:
    """Combined similarity score using multiple metrics.
    
    SHAPE-ONLY version (no identifier content). Used as a structural baseline.

    WARNING: This metric cannot distinguish real code from placeholder stubs.
    A file of `fun x() = null` scores the same as `fun computeHash() = ...`.
    Use combined_similarity_with_content() for real porting assessment.
    """
    hist_sim = histogram_cosine_similarity(tree1, tree2)
    struct_sim = structure_similarity(tree1, tree2)
    jaccard_sim = node_type_jaccard(tree1, tree2)

    return (hist_weight * hist_sim +
            struct_weight * struct_sim +
            jaccard_weight * jaccard_sim)


def combined_similarity_with_content(tree1: Tree, tree2: Tree,
                                       ids1: 'IdentifierStats',
                                       ids2: 'IdentifierStats') -> float:
    """Content-aware combined similarity.
    
    Uses identifier comparison as the DOMINANT signal at the function level.

    A real port reuses the same identifiers (with naming convention changes).
    A file of stubs has completely different identifiers and scores near 0.

    Weights:
      0.50 - Canonical identifier cosine (the killer metric)
      0.15 - Canonical identifier jaccard (set overlap)
      0.15 - AST histogram cosine (structural shape)
      0.10 - Node type jaccard
      0.10 - Structure similarity (tree size/depth)
    """
    id_cosine = ids1.canonical_cosine_similarity(ids2)
    id_jaccard = ids1.canonical_jaccard_similarity(ids2)
    hist_sim = histogram_cosine_similarity(tree1, tree2)
    jaccard_sim = node_type_jaccard(tree1, tree2)
    struct_sim = structure_similarity(tree1, tree2)

    return (0.50 * id_cosine +
            0.15 * id_jaccard +
            0.15 * hist_sim +
            0.10 * jaccard_sim +
            0.10 * struct_sim)


MAX_EDIT_DISTANCE_NODES = 2000


def edit_distance_dp(nodes1: List[Tree], nodes2: List[Tree]) -> int:
    """Core DP for edit distance on node-type sequences.
    
    Uses two-row rolling DP: O(m) memory instead of O(n*m).
    """
    n = len(nodes1)
    m = len(nodes2)

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if nodes1[i-1].node_type == nodes2[j-1].node_type else 1
            curr[j] = min(
                prev[j] + 1,      # Delete
                curr[j-1] + 1,    # Insert
                prev[j-1] + cost  # Replace/Match
            )
        prev, curr = curr, prev

    return prev[m]


def tree_edit_distance(tree1: Tree, tree2: Tree) -> int:
    """Tree edit distance with OOM protection.
    
    For trees with more than MAX_EDIT_DISTANCE_NODES nodes, samples
    every Kth node to fit within budget, then scales the result.
    Two-row DP keeps memory at O(min(n,m)) regardless.
    """
    nodes1 = []
    nodes2 = []

    def collect1(n: Tree):
        nodes1.append(n)
    def collect2(n: Tree):
        nodes2.append(n)

    tree1.traverse_postorder(collect1)
    tree2.traverse_postorder(collect2)

    full_n = len(nodes1)
    full_m = len(nodes2)

    if full_n <= MAX_EDIT_DISTANCE_NODES and full_m <= MAX_EDIT_DISTANCE_NODES:
        return edit_distance_dp(nodes1, nodes2)

    # Strided sampling
    stride1 = (full_n + MAX_EDIT_DISTANCE_NODES - 1) // MAX_EDIT_DISTANCE_NODES
    stride2 = (full_m + MAX_EDIT_DISTANCE_NODES - 1) // MAX_EDIT_DISTANCE_NODES
    stride = max(stride1, stride2)

    s1 = nodes1[::stride]
    s2 = nodes2[::stride]

    return edit_distance_dp(s1, s2) * stride


def normalized_edit_distance(tree1: Tree, tree2: Tree) -> float:
    """Normalized tree edit distance (0 to 1, where 1 = identical)."""
    dist = tree_edit_distance(tree1, tree2)
    max_size = max(tree1.size(), tree2.size())
    if max_size == 0:
        return 1.0
    return 1.0 - float(dist) / float(max_size)


class ComparisonReport:
    """Detailed comparison report."""
    def __init__(self):
        self.cosine_sim: float = 0.0
        self.structure_sim: float = 0.0
        self.jaccard_sim: float = 0.0
        self.edit_distance_sim: float = 0.0
        self.combined_score: float = 0.0

        self.size1: int = 0
        self.size2: int = 0
        self.depth1: int = 0
        self.depth2: int = 0

        self.hist1: List[int] = []
        self.hist2: List[int] = []

    def print(self) -> None:
        print("=== AST Similarity Report ===")
        print(f"Tree 1: size={self.size1}, depth={self.depth1}")
        print(f"Tree 2: size={self.size2}, depth={self.depth2}")
        print("\nSimilarity Metrics:")
        print(f"  Cosine (histogram):    {self.cosine_sim:.4f}")
        print(f"  Structure:             {self.structure_sim:.4f}")
        print(f"  Jaccard:               {self.jaccard_sim:.4f}")
        print(f"  Edit Distance (norm):  {self.edit_distance_sim:.4f}")
        print(f"  Combined Score:        {self.combined_score:.4f}")


class ASTSimilarity:
    """Compute various similarity metrics between ASTs."""
    
    NUM_NODE_TYPES = NUM_NODE_TYPES
    
    @staticmethod
    def get_node_weight(node_type: int) -> float:
        return get_node_weight(node_type)
    
    @staticmethod
    def histogram_cosine_similarity(tree1: Tree, tree2: Tree) -> float:
        return histogram_cosine_similarity(tree1, tree2)
    
    @staticmethod
    def histogram_cosine_similarity_macro(tree1: Tree, tree2: Tree) -> float:
        return histogram_cosine_similarity_macro(tree1, tree2)
    
    @staticmethod
    def node_type_jaccard(tree1: Tree, tree2: Tree) -> float:
        return node_type_jaccard(tree1, tree2)
    
    @staticmethod
    def structure_similarity(tree1: Tree, tree2: Tree) -> float:
        return structure_similarity(tree1, tree2)
    
    @staticmethod
    def combined_similarity(tree1: Tree, tree2: Tree,
                           hist_weight: float = 0.5,
                           struct_weight: float = 0.3,
                           jaccard_weight: float = 0.2) -> float:
        return combined_similarity(tree1, tree2, hist_weight, struct_weight, jaccard_weight)
    
    @staticmethod
    def combined_similarity_with_content(tree1: Tree, tree2: Tree,
                                         ids1: 'IdentifierStats',
                                         ids2: 'IdentifierStats') -> float:
        return combined_similarity_with_content(tree1, tree2, ids1, ids2)
    
    MAX_EDIT_DISTANCE_NODES = MAX_EDIT_DISTANCE_NODES
    
    @staticmethod
    def tree_edit_distance(tree1: Tree, tree2: Tree) -> int:
        return tree_edit_distance(tree1, tree2)
    
    @staticmethod
    def normalized_edit_distance(tree1: Tree, tree2: Tree) -> float:
        return normalized_edit_distance(tree1, tree2)
    
    @staticmethod
    def compare(tree1: Tree, tree2: Tree, macro_friendly: bool = False) -> ComparisonReport:
        return compare(tree1, tree2, macro_friendly)


def compare(tree1: Tree, tree2: Tree, macro_friendly: bool = False) -> ComparisonReport:
    """Generate detailed comparison report."""
    report = ComparisonReport()

    report.size1 = tree1.size()
    report.size2 = tree2.size()
    report.depth1 = tree1.depth()
    report.depth2 = tree2.depth()

    report.hist1 = tree1.node_type_histogram(NUM_NODE_TYPES)
    report.hist2 = tree2.node_type_histogram(NUM_NODE_TYPES)

    if macro_friendly:
        report.cosine_sim = histogram_cosine_similarity_macro(tree1, tree2)
    else:
        report.cosine_sim = histogram_cosine_similarity(tree1, tree2)
    report.structure_sim = structure_similarity(tree1, tree2)
    report.jaccard_sim = node_type_jaccard(tree1, tree2)
    report.edit_distance_sim = normalized_edit_distance(tree1, tree2)

    report.combined_score = (0.3 * report.cosine_sim +
                            0.2 * report.structure_sim +
                            0.2 * report.jaccard_sim +
                            0.3 * report.edit_distance_sim)

    return report

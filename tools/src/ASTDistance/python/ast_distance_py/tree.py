"""Tree structure for AST representation.

Faithful transliteration of include/tree.hpp from the C++ ast_distance tool.
Ported from Stanford TreeLSTM (Lua) to C++, now to Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Tree:
    """A basic tree structure for AST representation.

    This class provides:
    - Parent/child relationships
    - Cached size and depth computations
    - Traversal methods (pre-order, post-order)
    - Binary tree conversion (left-child right-sibling)
    - Leaf collection and node type histogram

    Faithful transliteration of struct Tree from include/tree.hpp:18-141.
    """

    parent: Optional[Tree] = None
    children: list[Tree] = field(default_factory=list)

    # Node type (normalized across languages)
    node_type: int = 0

    # For leaf nodes: index into input embeddings
    leaf_idx: int = -1

    # Optional: original node label for debugging
    label: str = ""

    # Cached computations (mutable in C++, so we track invalidation)
    _cached_size: int = field(default=-1, init=False, repr=False)
    _cached_depth: int = field(default=-1, init=False, repr=False)

    def add_child(self, child: Tree) -> None:
        """Add a child node to this tree.

        Transliteration of add_child() from tree.hpp:39-43.
        """
        child.parent = self
        self.children.append(child)
        self._cached_size = -1  # Invalidate cache

    def num_children(self) -> int:
        """Return the number of direct children.

        Transliteration of num_children() from tree.hpp:45-47.
        """
        return len(self.children)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children).

        Transliteration of is_leaf() from tree.hpp:49-51.
        """
        return len(self.children) == 0

    def size(self) -> int:
        """Get the total number of nodes in the tree (with caching).

        Transliteration of size() from tree.hpp:53-61.
        """
        if self._cached_size >= 0:
            return self._cached_size
        s = 1
        for child in self.children:
            s += child.size()
        self._cached_size = s
        return s

    def depth(self) -> int:
        """Get the depth of the tree (with caching).

        Transliteration of depth() from tree.hpp:63-72.
        """
        if self._cached_depth >= 0:
            return self._cached_depth
        d = 0
        for child in self.children:
            d = max(d, child.depth())
        if self.children:
            d += 1
        self._cached_depth = d
        return d

    def traverse_preorder(self, fn: Callable[[Tree], None]) -> None:
        """Depth-first pre-order traversal.

        Transliteration of traverse_preorder() from tree.hpp:75-80.
        """
        fn(self)
        for child in self.children:
            child.traverse_preorder(fn)

    def traverse_postorder(self, fn: Callable[[Tree], None]) -> None:
        """Depth-first post-order traversal (needed for bottom-up Tree-LSTM).

        Transliteration of traverse_postorder() from tree.hpp:83-88.
        """
        for child in self.children:
            child.traverse_postorder(fn)
        fn(self)

    def to_binary(self) -> Tree:
        """Convert to left-child right-sibling binary tree format.

        Transliteration of to_binary() from tree.hpp:144-170.

        Returns a binary tree where:
        - children[0] is the first child (left child)
        - children[1] (if exists) is the next sibling (right child)
        """
        binary = Tree(node_type=self.node_type, label=self.label)
        binary.leaf_idx = self.leaf_idx

        if not self.children:
            return binary

        # First child becomes left child
        binary_child = self.children[0].to_binary()
        binary_child.parent = binary
        binary.children.append(binary_child)

        # Remaining siblings chain as right children
        current = binary_child
        for i in range(1, len(self.children)):
            sibling = self.children[i].to_binary()
            sibling.parent = current
            current.children.append(sibling)

            if len(current.children) == 1:
                # Add placeholder for left child (nullptr in C++)
                # This maintains the binary tree structure where
                # index 0 = first child, index 1 = sibling
                current.children.insert(0, None)  # type: ignore[arg-type]

            current = sibling

        return binary

    def get_leaves(self) -> list[Tree]:
        """Collect all leaf nodes.

        Transliteration of get_leaves() from tree.hpp:94-102.
        """
        leaves: list[Tree] = []

        def collector(node: Tree) -> None:
            if node.is_leaf():
                leaves.append(node)

        self.traverse_preorder(collector)
        return leaves

    def node_type_histogram(self, num_types: int) -> list[int]:
        """Count nodes by type.

        Transliteration of node_type_histogram() from tree.hpp:105-113.

        Args:
            num_types: The number of node types (typically NodeType.NUM_TYPES)

        Returns:
            A histogram list where hist[i] = count of nodes with node_type == i
        """
        hist = [0] * num_types

        def counter(node: Tree) -> None:
            if 0 <= node.node_type < num_types:
                hist[node.node_type] += 1

        self.traverse_preorder(counter)
        return hist

    def flatten_node_type(self, type_to_flatten: int) -> None:
        """Flatten nodes of specific type (replacing them with their children).

        Transliteration of flatten_node_type() from tree.hpp:116-140.

        Args:
            type_to_flatten: The node type to dissolve
        """
        if not self.children:
            return

        new_children: list[Tree] = []

        for child in self.children:
            # Recurse first (bottom-up flattening)
            child.flatten_node_type(type_to_flatten)

            if child.node_type == type_to_flatten:
                # Dissolve this node, append its children to current parent
                for grand_child in child.children:
                    grand_child.parent = self
                    new_children.append(grand_child)
            else:
                new_children.append(child)

        self.children = new_children
        self._cached_size = -1
        self._cached_depth = -1

    def invalidate_cache(self) -> None:
        """Invalidate cached size and depth values."""
        self._cached_size = -1
        self._cached_depth = -1

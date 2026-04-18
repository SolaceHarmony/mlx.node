// port-lint: source include/tree.hpp
import { NodeType } from "./types.js";

/**
 * A basic tree structure for AST representation.
 * Ported from Stanford TreeLSTM (Lua) to C++.
 */
export class Tree {
  parent: Tree | null = null;
  children: Tree[] = [];

  // Node type (normalized across languages)
  nodeType: NodeType = NodeType.UNKNOWN;

  // For leaf nodes: index into input embeddings
  leafIdx: number = -1;

  // Optional: original node label for debugging
  label: string = "";

  // Position tracking (from ast-parser)
  startPosition = { row: 0, column: 0 };
  endPosition = { row: 0, column: 0 };

  // Cached computations
  private cachedSize: number = -1;
  private cachedDepth: number = -1;

  constructor(type: NodeType = NodeType.UNKNOWN, lbl: string = "") {
    this.nodeType = type;
    this.label = lbl;
  }

  addChild(child: Tree): void {
    child.parent = this;
    this.children.push(child);
    this.cachedSize = -1; // Invalidate cache
  }

  numChildren(): number {
    return this.children.length;
  }

  isLeaf(): boolean {
    return this.children.length === 0;
  }

  size(): number {
    if (this.cachedSize >= 0) return this.cachedSize;
    let s = 1;
    for (const child of this.children) {
      s += child.size();
    }
    this.cachedSize = s;
    return s;
  }

  depth(): number {
    if (this.cachedDepth >= 0) return this.cachedDepth;
    let d = 0;
    for (const child of this.children) {
      d = Math.max(d, child.depth());
    }
    if (this.children.length > 0) d += 1;
    this.cachedDepth = d;
    return d;
  }

  // Depth-first pre-order traversal
  traversePreorder(fn: (node: Tree) => void): void {
    fn(this);
    for (const child of this.children) {
      child.traversePreorder(fn);
    }
  }

  // Depth-first post-order traversal (needed for bottom-up Tree-LSTM)
  traversePostorder(fn: (node: Tree) => void): void {
    for (const child of this.children) {
      child.traversePostorder(fn);
    }
    fn(this);
  }

  // Convert to left-child right-sibling binary tree format
  toBinary(): Tree {
    const binary = new Tree(this.nodeType, this.label);
    binary.leafIdx = this.leafIdx;

    if (this.children.length === 0) {
      return binary;
    }

    // First child becomes left child
    binary.children.push(this.children[0]!.toBinary());
    binary.children[0]!.parent = binary;

    // Remaining siblings chain as right children
    let current = binary.children[0]!;
    for (let i = 1; i < this.children.length; ++i) {
      const sibling = this.children[i]!.toBinary();
      sibling.parent = current;
      current.children.push(sibling);
      if (current.children.length === 1) {
        // Add placeholder for left child if needed
        // (In TS, we can just use an array, but to match binary tree logic)
        current.children.unshift(null as any);
      }
      current = current.children[current.children.length - 1]!;
    }

    return binary;
  }

  // Collect all leaf nodes
  getLeaves(): Tree[] {
    const leaves: Tree[] = [];
    this.traversePreorder((node) => {
      if (node.isLeaf()) {
        leaves.push(node);
      }
    });
    return leaves;
  }

  // Count nodes by type
  nodeTypeHistogram(numTypes: number): number[] {
    const hist = new Array(numTypes).fill(0);
    this.traversePreorder((node) => {
      if (node.nodeType >= 0 && (node.nodeType as number) < numTypes) {
        hist[node.nodeType as number]++;
      }
    });
    return hist;
  }

  // Flatten nodes of specific type (replacing them with their children)
  flattenNodeType(typeToFlatten: NodeType): void {
    if (this.children.length === 0) return;

    const newChildren: Tree[] = [];

    for (let child of this.children) {
      // Recurse first (bottom-up flattening)
      child.flattenNodeType(typeToFlatten);

      if (child.nodeType === typeToFlatten) {
        // Dissolve this node, append its children to current parent
        for (let grandChild of child.children) {
          grandChild.parent = this;
          newChildren.push(grandChild);
        }
      } else {
        newChildren.push(child);
      }
    }

    this.children = newChildren;
    this.cachedSize = -1;
    this.cachedDepth = -1;
  }
}

// Alias for parity with existing code
export { Tree as TreeNode };

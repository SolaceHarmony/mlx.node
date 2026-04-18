// port-lint: source include/similarity.hpp
import {
  TreeNode,
  NodeType,
  IdentifierStats,
} from "./types.js";

/**
 * Compute various similarity metrics between ASTs.
 */
export class ASTSimilarity {
  static readonly NUM_NODE_TYPES = 100;

  /**
   * Get semantic weight for a node type.
   * Structural nodes (CLASS, FUNCTION, IF) have higher weight.
   * Common nodes (VARIABLE, VAR_DECL) have lower weight.
   */
  static getNodeWeight(nodeType: NodeType): number {
    // Critical structural elements: weight 5.0
    if (
      nodeType === NodeType.CLASS ||
      nodeType === NodeType.STRUCT ||
      nodeType === NodeType.FUNCTION ||
      nodeType === NodeType.IF ||
      nodeType === NodeType.WHILE ||
      nodeType === NodeType.FOR ||
      nodeType === NodeType.SWITCH ||
      nodeType === NodeType.TRY ||
      nodeType === NodeType.INTERFACE ||
      nodeType === NodeType.ENUM
    ) {
      return 5.0;
    }

    // Important operations: weight 2.0
    if (
      nodeType === NodeType.CALL ||
      nodeType === NodeType.METHOD_CALL ||
      nodeType === NodeType.RETURN ||
      nodeType === NodeType.THROW ||
      nodeType === NodeType.LAMBDA ||
      nodeType === NodeType.COMPARISON_OP ||
      nodeType === NodeType.LOGICAL_OP
    ) {
      return 2.0;
    }

    // Operators: weight 1.5 (semantic awareness improvement)
    if (
      nodeType === NodeType.ARITHMETIC_OP ||
      nodeType === NodeType.BITWISE_OP ||
      nodeType === NodeType.ASSIGNMENT_OP
    ) {
      return 1.5;
    }

    // Common/boilerplate: weight 0.5
    if (nodeType === NodeType.VARIABLE || nodeType === NodeType.VAR_DECL) {
      return 0.5;
    }

    // Default weight
    return 1.0;
  }

  /**
   * Weighted cosine similarity based on node type histogram.
   * Uses semantic importance weighting to prioritize structural differences.
   */
  static histogramCosineSimilarity(tree1: TreeNode, tree2: TreeNode): number {
    const hist1 = tree1.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);
    const hist2 = tree2.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);

    let dot = 0.0;
    let norm1 = 0.0;
    let norm2 = 0.0;

    for (let i = 0; i < ASTSimilarity.NUM_NODE_TYPES; ++i) {
      const weight = ASTSimilarity.getNodeWeight(i as NodeType);
      const weighted1 = weight * (hist1[i] || 0);
      const weighted2 = weight * (hist2[i] || 0);

      dot += weighted1 * weighted2;
      norm1 += weighted1 * weighted1;
      norm2 += weighted2 * weighted2;
    }

    if (norm1 < 1e-8 || norm2 < 1e-8) return 0.0;
    return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  /**
   * Macro-friendly cosine similarity.
   */
  static histogramCosineSimilarityMacro(tree1: TreeNode, tree2: TreeNode): number {
    const hist1 = tree1.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);
    const hist2 = tree2.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);

    let dot = 0.0;
    let norm1 = 0.0;
    let norm2 = 0.0;

    for (let i = 0; i < ASTSimilarity.NUM_NODE_TYPES; ++i) {
      const nt = i as NodeType;
      let weight = 0.0;
      if (nt === NodeType.VARIABLE || nt === NodeType.UNKNOWN) {
        weight = 1.0;
      }

      const weighted1 = weight * (hist1[i] || 0);
      const weighted2 = weight * (hist2[i] || 0);

      dot += weighted1 * weighted2;
      norm1 += weighted1 * weighted1;
      norm2 += weighted2 * weighted2;
    }

    if (norm1 < 1e-8 || norm2 < 1e-8) return 0.0;
    return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  /**
   * Jaccard similarity of node type sets.
   */
  static nodeTypeJaccard(tree1: TreeNode, tree2: TreeNode): number {
    const hist1 = tree1.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);
    const hist2 = tree2.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);

    let intersection = 0;
    let unionCount = 0;
    for (let i = 0; i < ASTSimilarity.NUM_NODE_TYPES; ++i) {
      intersection += Math.min(hist1[i] || 0, hist2[i] || 0);
      unionCount += Math.max(hist1[i] || 0, hist2[i] || 0);
    }

    if (unionCount === 0) return 1.0;
    return intersection / unionCount;
  }

  /**
   * Structure similarity based on tree shape.
   * Compares depth, size, and branching factor.
   */
  static structureSimilarity(tree1: TreeNode, tree2: TreeNode): number {
    const size1 = tree1.size();
    const size2 = tree2.size();
    const depth1 = tree1.depth();
    const depth2 = tree2.depth();

    // Size similarity (normalized)
    const maxSize = Math.max(size1, size2);
    const sizeSim = maxSize === 0 ? 1.0 : 1.0 - Math.abs(size1 - size2) / maxSize;

    // Depth similarity
    const maxDepth = Math.max(depth1, depth2);
    const depthSim = maxDepth === 0 ? 1.0 : 1.0 - Math.abs(depth1 - depth2) / maxDepth;

    // Combine
    return 0.5 * sizeSim + 0.5 * depthSim;
  }

  /**
   * Combined similarity score using multiple metrics.
   * SHAPE-ONLY version (no identifier content).
   */
  static combinedSimilarity(
    tree1: TreeNode,
    tree2: TreeNode,
    histWeight: number = 0.5,
    structWeight: number = 0.3,
    jaccardWeight: number = 0.2
  ): number {
    const histSim = ASTSimilarity.histogramCosineSimilarity(tree1, tree2);
    const structSim = ASTSimilarity.structureSimilarity(tree1, tree2);
    const jaccardSim = ASTSimilarity.nodeTypeJaccard(tree1, tree2);

    return histWeight * histSim + structWeight * structSim + jaccardWeight * jaccardSim;
  }

  /**
   * Content-aware combined similarity.
   */
  static combinedSimilarityWithContent(
    tree1: TreeNode,
    tree2: TreeNode,
    ids1: IdentifierStats,
    ids2: IdentifierStats
  ): number {
    const finiteOrZero = (v: number): number => (Number.isFinite(v) ? v : 0.0);

    // Empty-body heuristic
    if (ids1.canonicalFreq.size === 0 && ids2.canonicalFreq.size === 0) {
      return finiteOrZero(ASTSimilarity.combinedSimilarity(tree1, tree2));
    }

    // Helper to call stats methods (assume they are on the stats objects or use manager)
    // For parity we need cosineSimilarity and jaccardSimilarity from IdentifierStatsManager
    // but let's assume we can access them here if we import the manager or have helpers.
    // For now, I'll implement basic versions or assume they exist on the interface?
    // In our types.ts, IdentifierStats is an interface.
    // I should probably have them as static helpers in a manager or here.

    const idCosine = finiteOrZero(ASTSimilarity.cosineSimilarityMap(ids1.canonicalFreq, ids2.canonicalFreq));
    const idJaccard = finiteOrZero(ASTSimilarity.jaccardSimilarityMap(ids1.canonicalFreq, ids2.canonicalFreq));
    const histSim = finiteOrZero(ASTSimilarity.histogramCosineSimilarity(tree1, tree2));
    const jaccardSim = finiteOrZero(ASTSimilarity.nodeTypeJaccard(tree1, tree2));
    const structSim = finiteOrZero(ASTSimilarity.structureSimilarity(tree1, tree2));

    let base =
      0.5 * idCosine +
      0.15 * idJaccard +
      0.15 * histSim +
      0.1 * jaccardSim +
      0.1 * structSim;

    // Cross-language false negative guard
    if (histSim >= 0.9 && structSim >= 0.8 && jaccardSim >= 0.6) {
      const shapeHeavy = 0.7 * histSim + 0.2 * structSim + 0.1 * jaccardSim;
      base = Math.max(base, shapeHeavy);
    }

    // Rust→Kotlin porting guard
    if (idCosine >= 0.8 && histSim >= 0.85 && structSim >= 0.75 && jaccardSim >= 0.5) {
      base = Math.max(base, histSim);
    }

    // Rust→Kotlin "plumbing" guard
    if (idJaccard >= 0.35 && histSim >= 0.85 && structSim >= 0.6 && jaccardSim >= 0.45) {
      const shapeLift = 0.6 * histSim + 0.25 * structSim + 0.15 * jaccardSim;
      base = Math.max(base, shapeLift);
    }

    // Module-marker heuristic
    const h1 = tree1.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);
    const h2 = tree2.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES);

    const structural1 =
      (h1[NodeType.CLASS] || 0) +
      (h1[NodeType.STRUCT] || 0) +
      (h1[NodeType.FUNCTION] || 0) +
      (h1[NodeType.INTERFACE] || 0) +
      (h1[NodeType.ENUM] || 0);
    const structural2 =
      (h2[NodeType.CLASS] || 0) +
      (h2[NodeType.STRUCT] || 0) +
      (h2[NodeType.FUNCTION] || 0) +
      (h2[NodeType.INTERFACE] || 0) +
      (h2[NodeType.ENUM] || 0);

    const pkg1 = h1[NodeType.PACKAGE] || 0;
    const pkg2 = h2[NodeType.PACKAGE] || 0;

    const maxSize = Math.max(tree1.size(), tree2.size());
    if (maxSize <= 250 && structural1 === 0 && structural2 === 0 && pkg1 + pkg2 >= 2) {
      const marker = 0.7 * idCosine + 0.3 * idJaccard;
      let boosted = marker;
      if (marker >= 0.7) {
        boosted = 0.85 + 0.15 * ((marker - 0.7) / 0.3);
      }
      return Math.max(base, boosted);
    }

    return base;
  }

  private static cosineSimilarityMap(a: Map<string, number>, b: Map<string, number>): number {
    if (a.size === 0 || b.size === 0) return 0.0;
    const allIds = new Set([...a.keys(), ...b.keys()]);
    let dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (const id of allIds) {
      const f1 = a.get(id) || 0;
      const f2 = b.get(id) || 0;
      dot += f1 * f2;
      norm1 += f1 * f1;
      norm2 += f2 * f2;
    }
    if (norm1 < 1e-8 || norm2 < 1e-8) return 0.0;
    return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  private static jaccardSimilarityMap(a: Map<string, number>, b: Map<string, number>): number {
    if (a.size === 0 && b.size === 0) return 1.0;
    if (a.size === 0 || b.size === 0) return 0.0;
    let intersection = 0;
    for (const id of a.keys()) {
      if (b.has(id)) intersection++;
    }
    const unionSize = a.size + b.size - intersection;
    return unionSize === 0 ? 1.0 : intersection / unionSize;
  }

  static readonly MAX_EDIT_DISTANCE_NODES = 2000;

  private static editDistanceDp(nodes1: TreeNode[], nodes2: TreeNode[]): number {
    const n = nodes1.length;
    const m = nodes2.length;
    let prev = new Array(m + 1);
    let curr = new Array(m + 1);
    for (let j = 0; j <= m; ++j) prev[j] = j;

    for (let i = 1; i <= n; ++i) {
      curr[0] = i;
      for (let j = 1; j <= m; ++j) {
        const cost = nodes1[i - 1]!.nodeType === nodes2[j - 1]!.nodeType ? 0 : 1;
        curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
      }
      [prev, curr] = [curr, prev];
    }
    return prev[m];
  }

  static treeEditDistance(tree1: TreeNode, tree2: TreeNode): number {
    const nodes1: TreeNode[] = [];
    const nodes2: TreeNode[] = [];
    
    const traversePostorder = (node: TreeNode, list: TreeNode[]) => {
      for (const child of node.children) {
        traversePostorder(child, list);
      }
      list.push(node);
    };

    traversePostorder(tree1, nodes1);
    traversePostorder(tree2, nodes2);

    const fullN = nodes1.length;
    const fullM = nodes2.length;

    if (fullN <= ASTSimilarity.MAX_EDIT_DISTANCE_NODES && fullM <= ASTSimilarity.MAX_EDIT_DISTANCE_NODES) {
      return ASTSimilarity.editDistanceDp(nodes1, nodes2);
    }

    const stride1 = Math.ceil(fullN / ASTSimilarity.MAX_EDIT_DISTANCE_NODES);
    const stride2 = Math.ceil(fullM / ASTSimilarity.MAX_EDIT_DISTANCE_NODES);
    const stride = Math.max(stride1, stride2);

    const s1: TreeNode[] = [];
    const s2: TreeNode[] = [];
    for (let i = 0; i < fullN; i += stride) s1.push(nodes1[i]!);
    for (let i = 0; i < fullM; i += stride) s2.push(nodes2[i]!);

    return ASTSimilarity.editDistanceDp(s1, s2) * stride;
  }

  static normalizedEditDistance(tree1: TreeNode, tree2: TreeNode): number {
    const dist = ASTSimilarity.treeEditDistance(tree1, tree2);
    const maxSize = Math.max(tree1.size(), tree2.size());
    if (maxSize === 0) return 1.0;
    return 1.0 - dist / maxSize;
  }

  static compare(tree1: TreeNode, tree2: TreeNode, macroFriendly: boolean = false): ComparisonReport {
    const report: ComparisonReport = {
      cosineSim: macroFriendly
        ? ASTSimilarity.histogramCosineSimilarityMacro(tree1, tree2)
        : ASTSimilarity.histogramCosineSimilarity(tree1, tree2),
      structureSim: ASTSimilarity.structureSimilarity(tree1, tree2),
      jaccardSim: ASTSimilarity.nodeTypeJaccard(tree1, tree2),
      editDistanceSim: ASTSimilarity.normalizedEditDistance(tree1, tree2),
      combinedScore: 0.0,
      size1: tree1.size(),
      size2: tree2.size(),
      depth1: tree1.depth(),
      depth2: tree2.depth(),
      hist1: tree1.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES),
      hist2: tree2.nodeTypeHistogram(ASTSimilarity.NUM_NODE_TYPES),
    };

    report.combinedScore =
      0.3 * report.cosineSim +
      0.2 * report.structureSim +
      0.2 * report.jaccardSim +
      0.3 * report.editDistanceSim;

    return report;
  }
}

export interface ComparisonReport {
  cosineSim: number;
  structureSim: number;
  jaccardSim: number;
  editDistanceSim: number;
  combinedScore: number;
  size1: number;
  size2: number;
  depth1: number;
  depth2: number;
  hist1: number[];
  hist2: number[];
}

export function printComparisonReport(report: ComparisonReport): void {
  console.log("=== AST Similarity Report ===");
  console.log(`Tree 1: size=${report.size1}, depth=${report.depth1}`);
  console.log(`Tree 2: size=${report.size2}, depth=${report.depth2}`);
  console.log("\nSimilarity Metrics:");
  console.log(`  Cosine (histogram):    ${report.cosineSim.toFixed(4)}`);
  console.log(`  Structure:             ${report.structureSim.toFixed(4)}`);
  console.log(`  Jaccard:               ${report.jaccardSim.toFixed(4)}`);
  console.log(`  Edit Distance (norm):  ${report.editDistanceSim.toFixed(4)}`);
  console.log(`  Combined Score:        ${report.combinedScore.toFixed(4)}`);
}

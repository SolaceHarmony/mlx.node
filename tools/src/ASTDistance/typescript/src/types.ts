/**
 * Language support for AST parsing
 */
export enum Language {
  RUST = "rust",
  KOTLIN = "kotlin",
  CPP = "cpp",
  PYTHON = "python",
  TYPESCRIPT = "typescript",
  C = "c",
}

/**
 * Normalized AST node types across languages (Sync with node_types.hpp)
 */
export enum NodeType {
  BLOCK = 0,
  IF = 1,
  FOR = 2,
  WHILE = 3,
  SWITCH = 4,
  RETURN = 5,
  OTHER = 6,
  CONTINUE = 7,
  BREAK = 8,
  THROW = 9,
  TRY = 10,
  TERNARY = 11,

  // Operations
  ASSIGN = 20,
  INDEX = 21,
  FIELD_ACCESS = 22,
  CALL = 23,
  METHOD_CALL = 24,
  CAST = 25,
  LAMBDA = 26,

  // Operators (unnamed tokens)
  ARITHMETIC_OP = 30,
  COMPARISON_OP = 31,
  LOGICAL_OP = 32,
  BITWISE_OP = 33,
  ASSIGNMENT_OP = 34,

  // Literals
  VARIABLE = 40,
  NUMBER = 41,
  STRING = 42,
  BOOLEAN = 43,
  NULL_LIT = 44,
  CHAR = 45,

  // Types
  TYPE_REF = 50,
  ARRAY_TYPE = 51,
  GENERIC_TYPE = 52,
  NULLABLE_TYPE = 53,
  POINTER_TYPE = 54,
  FUNC_TYPE = 55,

  // Declarations
  FUNCTION = 60,
  CLASS = 61,
  STRUCT = 62,
  ENUM = 63,
  INTERFACE = 64,
  VAR_DECL = 65,
  PARAM = 66,
  TYPE_PARAM = 67,
  ANNOTATION = 68,
  IMPORT = 69,
  PACKAGE = 70,

  // Other
  COMMENT = 80,
  MODIFIER = 81,
  RANGE = 82,
  GOTO = 83,

  // Operators (specific for name matching)
  EQ = 90,
  NE = 91,
  GT = 92,
  LT = 93,
  ADD = 94,
  SUB = 95,
  MUL = 96,
  DIV = 97,

  // Special
  UNKNOWN = 99,
}

import { Tree, TreeNode } from "./tree.js";
export { Tree, TreeNode };

import { TodoItem, LintError } from "./porting-utils.js";
export { TodoItem, LintError };

/**
 * Statistics about identifiers (function names, variable names, etc.)
 */
export interface IdentifierStats {
  identifierFreq: Map<string, number>;
  canonicalFreq: Map<string, number>;
  totalIdentifiers: number;
}

/**
 * Metadata for a function extracted from source code.
 */
export interface FunctionInfo {
  name: string;
  bodyTree: TreeNode;
  identifiers: IdentifierStats;
  hasStubMarkers: boolean;
  isTest: boolean;
}

/**
 * Statistics about comments/documentation
 */
export interface CommentStats {
  docCommentCount: number;
  lineCommentCount: number;
  blockCommentCount: number;
  totalCommentLines: number;
  totalDocLines: number;
  docTexts: string[];
  wordFreq: Map<string, number>;
}

/**
 * AST parse result with metadata
 */
export interface ParseResult {
  tree: TreeNode;
  filename: string;
  language: Language;
  commentStats: CommentStats;
  identifierStats: IdentifierStats;
  functions: FunctionInfo[];
  hasStubBodies: boolean;
  nodeTypes: Map<NodeType, number>; // Histogram
  importPaths: string[];
  exportPaths: string[];
  portLintHeader?: PortLintHeader;
}


/**
 * Similarity metrics between two ASTs
 */
export interface SimilarityMetrics {
  cosineHistogram: number;
  structure: number;
  jaccard: number;
  combined: number;
}

/**
 * Documentation comparison
 */
export interface DocumentationComparison {
  docCommentDifference: number;
  docLineDifference: number;
  docTextCosine: number;
}

/**
 * Port-lint header information
 */
export interface PortLintHeader {
  sourcePath: string;
  matchedBy: "header" | "name";
  confidence: number;
}

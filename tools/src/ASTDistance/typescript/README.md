# AST Distance TypeScript

Cross-language AST similarity measurement and porting analysis tool.

**Ported from the C++ implementation for use in Node.js environments.**

## Features

- **Tree-sitter parsing** for TypeScript, Rust, Kotlin, and C++ ASTs
- **Normalized node types** that map across languages
- **Multiple similarity metrics**:
  - Cosine similarity of node type histograms
  - Structure similarity (size, depth)
  - Jaccard similarity of node sets
  - Normalized tree edit distance
- **Codebase-level analysis**:
  - Dependency graph building
  - File matching by port-lint headers or name similarity
  - Porting priority ranking
  - Documentation gap detection
- **Quality checks**:
  - TODO scanning with context
  - Lint error detection
  - Stub file identification

## Installation

```bash
npm install @ast-distance/typescript
```

## Usage

### Command Line

```bash
# Compare two files
ast-distance <file1> <lang1> <file2> <lang2>

# Dump AST structure
ast-distance dump <file> <lang>

# Scan directory
ast-distance scan <directory> <lang>

# Deep codebase analysis
ast-distance deep <src_dir> <src_lang> <tgt_dir> <tgt_lang>

# Find missing files
ast-distance missing <src_dir> <src_lang> <tgt_dir> <tgt_lang>

# Scan for TODOs
ast-distance todos <directory>
```

### Programmatic API

```typescript
import {
  parseFile,
  computeSimilarity,
  printSimilarityReport,
  parseDirectory,
  buildDepGraph,
  rankPortingPriority,
  Language,
} from "@ast-distance/typescript";

// Compare two files
const result1 = await parseFile("source.ts", Language.TYPESCRIPT);
const result2 = await parseFile("target.rs", Language.RUST);

const metrics = computeSimilarity(result1, result2);
console.log(`Similarity: ${metrics.combined}`);

// Analyze codebase
const files = parseDirectory("./src", Language.TYPESCRIPT);
const depGraph = buildDepGraph(files);
const rankings = rankPortingPriority(srcFiles, tgtFiles, depGraph);
```

## Port-Lint Headers

Add a header comment to each ported file to enable accurate source tracking:

```typescript
// port-lint: source core/src/config.rs
package com.example.config

data class Config(...)
```

The header must appear in the first 50 lines. When present, the tool will:

- Match files explicitly instead of by name similarity
- Compare documentation coverage between source and target
- Report "Matched by header" vs "Matched by name" statistics

## Supported Languages

- **TypeScript** (`.ts`, `.tsx`, `.mts`, `.cts`)
- **Rust** (`.rs`) - _coming soon_
- **Kotlin** (`.kt`, `.kts`) - _coming soon_
- **C++** (`.cpp`, `.hpp`, `.cc`, `.h`) - _coming soon_

## Example Output

### File Comparison

```
=== AST Similarity Report ===
File 1: source.ts (TypeScript)
  Tree size: 142, depth: 7

File 2: target.rs (Rust)
  Tree size: 156, depth: 9

Similarity Metrics:
  Cosine (histogram):    0.92%
  Structure:             0.86%
  Combined Score:        0.89%

=== Documentation Comparison ===
  Doc count:    5 vs 4 (diff: -1)
  Doc lines:    24 vs 18 (diff: -6)
  Doc cosine:   87.42%
```

### Deep Analysis

```
=== Porting Quality Summary ===

Matched by header:    87 / 95
Matched by name:      8 / 95
Total TODOs in target: 42
Total lint errors:    287

=== Porting Priorities (Top 10) ===

  12 dependents  core/src/config.rs
  9 dependents   core/src/engine.rs
  7 dependents   core/src/parser.rs
  ...
```

## License

MIT

## Contributing

Contributions are welcome! Please see the main [ASTDistance repository](https://github.com/your-repo/ASTDistance) for guidelines.

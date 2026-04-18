# High Priority Ports - Action Plan

## Top 20 Files by Impact (Priority Score = Deps × (1 - Similarity))

| Rank | Source | Target | Similarity | Deps | Priority |
|------|--------|--------|------------|------|----------|
| 1 | `ast_parser` | `ast_parser` | 0.00 | 2 | 2.0 |
| 2 | `tree` | `tree` | 0.45 | 3 | 1.7 |
| 3 | `node_types` | `node_types` | 0.58 | 2 | 0.8 |
| 4 | `imports` | `imports` | 0.23 | 1 | 0.8 |
| 5 | `porting_utils` | `porting_utils` | 0.25 | 1 | 0.8 |
| 6 | `codebase` | `codebase` | 0.32 | 1 | 0.7 |
| 7 | `similarity` | `similarity` | 0.78 | 1 | 0.2 |
| 8 | `task_manager` | `task_manager` | 0.00 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **ast_parser** → `ast_parser`
  - Similarity: 0.00
  - Dependencies: 2

- **tree** → `tree`
  - Similarity: 0.45
  - Dependencies: 3

- **node_types** → `node_types`
  - Similarity: 0.58
  - Dependencies: 2

- **imports** → `imports`
  - Similarity: 0.23
  - Dependencies: 1

- **porting_utils** → `porting_utils`
  - Similarity: 0.25
  - Dependencies: 1

- **codebase** → `codebase`
  - Similarity: 0.32
  - Dependencies: 1

## Missing Files (Top by Dependents)

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `tensor` | 2 | `tensor.hpp` |
| 2 | `port_lint` | 0 | `port_lint.hpp` |
| 3 | `symbol_analysis` | 0 | `symbol_analysis.hpp` |
| 4 | `symbol_extraction` | 0 | `symbol_extraction.hpp` |
| 5 | `symbol_extractor` | 0 | `symbol_extractor.hpp` |
| 6 | `tree_lstm` | 0 | `tree_lstm.hpp` |


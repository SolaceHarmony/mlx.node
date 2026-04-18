# Code Port - Progress Report

**Generated:** 2026-03-30
**Source:** include
**Target:** python/ast_distance_py

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 14 | 100% |
| Target units (paired) | 9 | - |
| Target files (total) | 9 | - |
| Porting progress | 8 | 57.1% (matched) |
| Missing files | 6 | 42.9% |

## Port Quality Analysis

**Average Similarity:** 0.32

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 1 files (12.5% of matched)
- Critical (<0.60): 7 files (87.5% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `ast_parser` → `ast_parser` (0.00, 2 deps)
- `tree` → `tree` (0.45, 3 deps)
- `node_types` → `node_types` (0.58, 2 deps)
- `imports` → `imports` (0.23, 1 deps)
- `porting_utils` → `porting_utils` (0.25, 1 deps)
- `codebase` → `codebase` (0.32, 1 deps)
- `task_manager` → `task_manager` (0.00)

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `tensor` | 2 | `tensor.hpp` |
| 2 | `port_lint` | 0 | `port_lint.hpp` |
| 3 | `symbol_analysis` | 0 | `symbol_analysis.hpp` |
| 4 | `symbol_extraction` | 0 | `symbol_extraction.hpp` |
| 5 | `symbol_extractor` | 0 | `symbol_extractor.hpp` |
| 6 | `tree_lstm` | 0 | `tree_lstm.hpp` |

## Documentation Gaps

There is missing documentation that is hurting overall scoring.

**Documentation coverage:** 0 / 434 lines (0%)

Top documentation gaps (>20%):

- `ast_parser` - 100% gap (135 → 0 lines)
- `similarity` - 100% gap (71 → 0 lines)
- `porting_utils` - 100% gap (66 → 0 lines)
- `codebase` - 100% gap (57 → 0 lines)
- `task_manager` - 100% gap (43 → 0 lines)
- `imports` - 100% gap (39 → 0 lines)
- `node_types` - 100% gap (19 → 0 lines)


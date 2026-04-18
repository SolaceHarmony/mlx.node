# Code Port - Progress Report

**Generated:** 2026-04-17
**Source:** ./python/mlx
**Target:** ./node/src

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 30 | 100% |
| Target units (paired) | 36 | - |
| Target files (total) | 36 | - |
| Porting progress | 14 | 46.7% (matched) |
| Missing files | 16 | 53.3% |

## Port Quality Analysis

**Average Similarity:** 0.30

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 14 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `layers.linear` → `layers.linear` (0.31, 3 deps)
- `layers.dropout` → `layers.dropout` (0.57, 2 deps)
- `layers.embedding` → `layers.embedding` (0.22, 1 deps)
- `layers.transformer` → `layers.transformer` (0.44, 1 deps)
- `nn.losses` → `nn.losses` (0.59, 1 deps)
- `nn.__init__` → `nn.init` (0.11)
- `layers.activations` → `layers.activations` (0.40)
- `layers.base` → `layers.base` (0.31)
- `layers.convolution` → `layers.convolution` (0.48)
- `layers.normalization` → `layers.normalization` (0.17)
- `layers.pooling` → `layers.pooling` (0.25)
- `layers.recurrent` → `layers.recurrent` (0.28)
- `nn.init` → `utils.nn_init` (0.07)
- `layers.positional_encoding` → `streaming.encoding` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| _None detected_ | | | |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `layers.upsample` | 1 | `nn/layers/upsample.py` |
| 2 | `optimizers.optimizers` | 1 | `optimizers/optimizers.py` |
| 3 | `optimizers.schedulers` | 1 | `optimizers/schedulers.py` |
| 4 | `__main__` | 0 | `__main__.py` |
| 5 | `_os_warning` | 0 | `_os_warning.py` |
| 6 | `_reprlib_fix` | 0 | `_reprlib_fix.py` |
| 7 | `distributed_run` | 0 | `distributed_run.py` |
| 8 | `extension` | 0 | `extension.py` |
| 9 | `layers.__init__` | 0 | `nn/layers/__init__.py` |
| 10 | `layers.containers` | 0 | `nn/layers/containers.py` |
| 11 | `layers.convolution_transpose` | 0 | `nn/layers/convolution_transpose.py` |
| 12 | `layers.distributed` | 0 | `nn/layers/distributed.py` |
| 13 | `layers.quantized` | 0 | `nn/layers/quantized.py` |
| 14 | `nn.utils` | 0 | `nn/utils.py` |
| 15 | `optimizers.__init__` | 0 | `optimizers/__init__.py` |
| 16 | `utils` | 0 | `utils.py` |

## Documentation Gaps

**Documentation coverage:** 575 / 0 lines (N/A)

Top documentation gaps (>20%):

No significant documentation gaps found.


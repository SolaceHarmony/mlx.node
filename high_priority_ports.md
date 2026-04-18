# High Priority Ports - Action Plan

## Top 20 Files by Impact (Priority Score = Deps × (1 - Similarity))

| Rank | Source | Target | Similarity | Deps | Priority |
|------|--------|--------|------------|------|----------|
| 1 | `layers.linear` | `layers.linear` | 0.31 | 3 | 2.1 |
| 2 | `layers.dropout` | `layers.dropout` | 0.57 | 2 | 0.9 |
| 3 | `layers.embedding` | `layers.embedding` | 0.22 | 1 | 0.8 |
| 4 | `layers.transformer` | `layers.transformer` | 0.44 | 1 | 0.6 |
| 5 | `nn.losses` | `nn.losses` | 0.59 | 1 | 0.4 |
| 6 | `nn.__init__` | `nn.init` | 0.11 | 0 | 0.0 |
| 7 | `layers.activations` | `layers.activations` | 0.40 | 0 | 0.0 |
| 8 | `layers.base` | `layers.base` | 0.31 | 0 | 0.0 |
| 9 | `layers.convolution` | `layers.convolution` | 0.48 | 0 | 0.0 |
| 10 | `layers.normalization` | `layers.normalization` | 0.17 | 0 | 0.0 |
| 11 | `layers.pooling` | `layers.pooling` | 0.25 | 0 | 0.0 |
| 12 | `layers.recurrent` | `layers.recurrent` | 0.28 | 0 | 0.0 |
| 13 | `nn.init` | `utils.nn_init` | 0.07 | 0 | 0.0 |
| 14 | `layers.positional_encoding` | `streaming.encoding` | 0.00 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **layers.linear** → `layers.linear`
  - Similarity: 0.31
  - Dependencies: 3
  - Lint issues: 8

- **layers.dropout** → `layers.dropout`
  - Similarity: 0.57
  - Dependencies: 2
  - Lint issues: 3

- **layers.embedding** → `layers.embedding`
  - Similarity: 0.22
  - Dependencies: 1
  - Lint issues: 2

- **layers.transformer** → `layers.transformer`
  - Similarity: 0.44
  - Dependencies: 1
  - Lint issues: 1

- **nn.losses** → `nn.losses`
  - Similarity: 0.59
  - Dependencies: 1

## Missing Files (Top by Dependents)

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


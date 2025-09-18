# MLX → Node Porting Map

This document catalogs the upstream MLX sources we need to transliterate or
bind while porting to the Node + React 19 TypeScript stack. It complements
`node/CHECKLIST.md` by listing the concrete files behind each checklist item.

## Native C++ Sources (`mlx/`)

Currently linked into `mlx_array.node`:

- [x] `mlx/array.cpp`
- [x] `mlx/allocator.cpp`
- [x] `mlx/backend/no_gpu/allocator.cpp`
- [x] `mlx/dtype.cpp`
- [x] `mlx/dtype_utils.cpp`
- [x] `mlx/version.cpp`

Planned additions as coverage expands:

- [ ] `mlx/ops.cpp` — core tensor ops (reshape, transpose, arithmetic)
- [ ] `mlx/primitives.cpp` and `mlx/fast.cpp` — autodiff primitives & fast kernels
- [ ] `mlx/random.cpp`
- [ ] `mlx/transforms.cpp`
- [ ] `mlx/linalg.cpp`
- [ ] `mlx/fft.cpp`
- [ ] `mlx/einsum.cpp`
- [ ] `mlx/device.cpp`, `mlx/stream.cpp`, `mlx/scheduler.cpp`
- [ ] `mlx/export.cpp`, `mlx/utils.cpp`, `mlx/graph_utils.cpp`

## Python Binding References (`python/src/`)

Use these Pybind11 wrappers as the blueprint for our N-API layer.

- [ ] `python/src/array.cpp`
- [ ] `python/src/constants.cpp`
- [ ] `python/src/convert.cpp`
- [ ] `python/src/cuda.cpp`
- [ ] `python/src/device.cpp`
- [ ] `python/src/distributed.cpp`
- [ ] `python/src/export.cpp`
- [ ] `python/src/fast.cpp`
- [ ] `python/src/fft.cpp`
- [ ] `python/src/indexing.cpp`
- [ ] `python/src/linalg.cpp`
- [ ] `python/src/load.cpp`
- [ ] `python/src/memory.cpp`
- [ ] `python/src/metal.cpp`
- [ ] `python/src/mlx.cpp`
- [ ] `python/src/mlx_func.cpp`
- [ ] `python/src/ops.cpp`
- [ ] `python/src/random.cpp`
- [ ] `python/src/stream.cpp`
- [ ] `python/src/transforms.cpp`
- [ ] `python/src/trees.cpp`
- [ ] `python/src/utils.cpp`

Headers/helpers worth mirroring where appropriate:

- `python/src/buffer.h`
- `python/src/convert.h`
- `python/src/indexing.h`
- `python/src/load.h`
- `python/src/mlx_func.h`
- `python/src/small_vector.h`
- `python/src/trees.h`
- `python/src/utils.h`

## Python API Surface (`python/mlx/`)

Top-level utilities:

- [ ] `python/mlx/utils.py`
- [ ] `python/mlx/_os_warning.py`
- [ ] `python/mlx/distributed_run.py`

Neural-network stack (`python/mlx/nn/`):

- [ ] `__init__.py`
- [ ] `utils.py`
- [ ] `init.py`
- [ ] `losses.py`
- [ ] Layers under `python/mlx/nn/layers/`
- [ ] Positional encodings, transformer helpers, etc.

Optimizers and schedulers (`python/mlx/optimizers/`):

- [ ] `__init__.py`
- [ ] `optimizers.py`
- [ ] `schedulers.py`

Miscellaneous modules to keep in view:

- `python/mlx/tests/` (reference behaviour for parity tests)
- `python/mlx/nn/random/` and loaders (if present)

## Node Implementation Cross-Reference

- Native entrypoints delivered so far: array construction (`Array.from*`), dtype
  wrappers, scalar constructors (`zeros`, `ones`, `full`, and the `_like`
  variants).
- TypeScript surface in place: `src/core/array.ts`, `src/core/dtype.ts`,
  `src/index.ts`, supporting tests under `test/core/`.
- Progress tracking: `node/CHECKLIST.md` (high-level milestones) and this file
  (file-level source map).

## Immediate Opportunities

1. Pull `mlx/ops.cpp` (and dependencies) into the addon to unlock structural
   ops such as `reshape`, `transpose`, `moveaxis`, and arithmetic kernels.
2. Mirror the corresponding binding logic from `python/src/ops.cpp` into
   `node/src/native` once the symbols are available.
3. Start carving out the high-level Python modules (e.g. `mlx.nn`) following the
   checklist order, using the directory lists above to stage the work.

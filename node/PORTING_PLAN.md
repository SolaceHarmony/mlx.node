# MLX → Node Porting Map

This document catalogs the upstream MLX sources that are being transliterated or
bound while porting to the Node + React 19 TypeScript stack.

## Native C++ Sources (`mlx/`)

The following MLX core sources are used in the native addon (`mlx_array.node`):

- `mlx/array.cpp`
- `mlx/allocator.cpp`
- `mlx/backend/no_gpu/allocator.cpp`
- `mlx/dtype.cpp`
- `mlx/dtype_utils.cpp`
- `mlx/version.cpp`

Additional sources to be included as coverage expands:

- `mlx/ops.cpp` — core tensor ops (reshape, transpose, arithmetic)
- `mlx/primitives.cpp` and `mlx/fast.cpp` — autodiff primitives & fast kernels
- `mlx/random.cpp`
- `mlx/transforms.cpp`
- `mlx/linalg.cpp`
- `mlx/fft.cpp`
- `mlx/einsum.cpp`
- `mlx/device.cpp`, `mlx/stream.cpp`, `mlx/scheduler.cpp`
- `mlx/export.cpp`, `mlx/utils.cpp`, `mlx/graph_utils.cpp`

## Python Binding References (`python/src/`)

Pybind11 wrappers used as the blueprint for the N-API layer:

- `python/src/array.cpp`
- `python/src/constants.cpp`
- `python/src/convert.cpp`
- `python/src/cuda.cpp`
- `python/src/device.cpp`
- `python/src/distributed.cpp`
- `python/src/export.cpp`
- `python/src/fast.cpp`
- `python/src/fft.cpp`
- `python/src/indexing.cpp`
- `python/src/linalg.cpp`
- `python/src/load.cpp`
- `python/src/memory.cpp`
- `python/src/metal.cpp`
- `python/src/mlx.cpp`
- `python/src/mlx_func.cpp`
- `python/src/ops.cpp`
- `python/src/random.cpp`
- `python/src/stream.cpp`
- `python/src/transforms.cpp`
- `python/src/trees.cpp`
- `python/src/utils.cpp`

Headers and helpers to be mirrored:

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

- `python/mlx/utils.py`
- `python/mlx/_os_warning.py`
- `python/mlx/distributed_run.py`

Neural-network stack (`python/mlx/nn/`):

- `__init__.py`
- `utils.py`
- `init.py`
- `losses.py`
- Layers under `python/mlx/nn/layers/`
- Positional encodings, transformer helpers, etc.

Optimizers and schedulers (`python/mlx/optimizers/`):

- `__init__.py`
- `optimizers.py`
- `schedulers.py`

## React 19 / Next.js Integration Blueprint

- **Server orchestration**: Each MLX inference/training routine is exposed as a React Server Action that yields an async iterator of intermediate payloads.
- **Transport options**: Prefer native Server-Sent Events for low-overhead, one-way delivery. Fall back to `ReadableStream` when appropriate.
- **Client hydration**: React 19 wrappers open the SSE channel during server render and hydrate into client components via `EventSource`.
- **Payload schema**: Stream tensor headers first (shape, dtype, stride) followed by encoded binary chunks.
- **Library plumbing**: Streaming primitives live under `src/streaming` and `src/react`.

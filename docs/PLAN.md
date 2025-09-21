# Porting Plan / TODO (Python MLX 1:1 parity)

## Core ops (status)
- Implemented (dtype objects only, kw-only stream arg):
  - zeros, ones, full, reshape, transpose, moveaxis, swapaxes, add, multiply, where, matmul
  - Array class with shape(), dtype(), eval(), toTypedArray()
  - Streams: default_stream/new_stream/synchronize (exported under mlx.core)
  - Dtypes: exported as objects (mlx.float16/32/64, mlx.bfloat16, ints, bool, complex64);
    issubdtype under mlx.dtype (Python parity)

- In progress / next:
  - arange(stop, step=None, dtype=None, *, stream=None)
  - arange(start, stop, step=None, dtype=None, *, stream=None)
    - dtype default/promotion rules identical to Python (per python/src/ops.cpp)
  - array(x, dtype=None, *, stream=None) and asarray(x, dtype=None, *, stream=None)
    - Accept nested JS arrays (shape inference), scalars (rank‑0), TypedArray/ArrayBuffer/Buffer
    - dtype inference matches Python (float32 default; promote from inputs)
    - asarray no-copy for CPU TypedArray when dtype matches; otherwise copy

## Dtype & precision hardening
- Enforced dtype objects only in all ops (no string keys)
- Precision test harness (node/examples/dtype_precision.js) compares float16/bfloat16 bit patterns
  against IEEE rounding rules and validates int edges
- Next: wire a small test runner for repeatable checks (CPU/GPU)

## Streaming
- Exposed: default_stream, new_stream, synchronize
- Next: no extras beyond Python; ensure all ops accept kw-only stream and device

## Stability & tests
- Add unit tests per op (dtype/shape/stream/device paths)
- Regression tests for lazy Metal init and process teardown
- Concurrent first-call init tests (multiple async callers)

## Build & packaging
- Production build only: jit_embed → mlx_core.a → mlx.node (labs separate)
- Prebuild binaries for macOS arm64 (later)

## Docs & examples
- Examples: basic.js (sanity), dtype_precision.js (bit-accuracy)
- Next: README with strict 1:1 API references to Python MLX

- Streaming
  - Expose `default_stream`, `new_stream`, `synchronize` cleanly
  - Add helpers to wrap React 19 streaming (SSE) over incremental evaluations

- Stability & tests
  - Unit tests per op; regression tests for GPU init and teardown
  - Concurrent first-call init tests (multiple async callers)

- Build & packaging
  - Prebuild binaries for macOS arm64 (Metal)
  - Keep C++ JIT generator (labs/tools/jit_embed) wired in gyp

- Docs & examples
  - Short examples for Node consumers; Next.js quickstart (React 19)
  - Document stream semantics and default device behavior

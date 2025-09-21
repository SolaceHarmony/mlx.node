# Porting Plan / TODO

- Basic ops and tensor tooling (priority)
  - Cover: zeros/ones/full/reshape/transpose/moveaxis/swapaxes/add/multiply/where
  - Ensure every op accepts optional `stream` and runs on Metal by default
  - TypedArray interop in and out (large shapes, all common dtypes)

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

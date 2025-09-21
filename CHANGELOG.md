# Changelog

All notable changes to this project will be documented in this file.

## 2025-09-20 — Node 22 / Metal GPU checkpoint

- Node-API (N-API) stabilization for MLX on Node 22:
  - Added a small C++ prebuild tool (`labs/tools/jit_embed`) that generates the Metal
    JIT preamble sources from the vendored MLX kernel files.
  - Wired a gyp action to emit `node/generated/metal_jit_preambles.cpp` and compile it
    into `mlx_core.a` alongside MLX’s `compiled.cpp` and `jit_kernels.cpp`.
  - Ensures full JIT coverage (utils/unary_ops/binary_ops/ternary_ops and core
    kernels) without relying on MLX’s CMake pipeline.

- Metal GPU init fixes:
  - Introduced a runtime singleton (`node/src/native/runtime.mm`) to initialize Metal
    safely (scoped autorelease pool, default GPU device set) before first op.
  - Avoided premature calls that previously led to `device_info()` crashes.

- Addon lifecycle fix:
  - Removed duplicate deletion of addon instance data (double free) by relying on the
    `Env.SetInstanceData` default finalizer only. This resolves exit-time crashes.

- Build system updates:
  - Compiles distributed stubs (`no_mpi`, `no_nccl`, `no_ring`) so distributed symbols
    resolve cleanly.
  - Uses `FMT_HEADER_ONLY` for libfmt to avoid external dependency for the addon.
  - Added `labs/cpp/standalone_{cpu,gpu}.cpp` and a portable fmt-detection script to
    validate the static library in isolation.

- Namespacing and API shape (Python parity):
  - Export a single native module named `mlx` (binary: `mlx.node`).
  - Provide a top‑level namespace object: `mlx.core` hosts Array and ops.
  - Dtype values are real objects: pass `mlx.float32` etc (strings temporarily allowed
    for transition only).
  - Ops accept stream/device (CPU/GPU) in the final parameter.
  - Added a factory: `mlx.core.array(typedArray, shape[, dtype][, streamOrDevice])`.

- Labs and docs:
  - Established `labs/` for experiments (fast iteration without touching the main
    addon).
  - Added architecture and planning documents under `docs/`.
  - New `.gitignore` for generated/build artifacts.

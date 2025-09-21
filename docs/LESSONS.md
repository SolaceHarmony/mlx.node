# Lessons Learned: Porting MLX to Node 22 / Metal

This checkpoint captures what we learned wiring Apple MLX into a Node 22 (N-API) addon with full Metal GPU support and a Python-parity API shape.

- JIT preambles are generated, not “missing libs”
  - MLX’s Metal JIT expects C++ functions (utils(), unary_ops(), binary_ops(), …) that return the textual MSL preambles/kernels. Upstream CMake generates `jit/*.cpp` from the kernel sources.
  - We implemented a C++ prebuild generator (`labs/tools/jit_embed`) and a gyp action to emit `node/generated/metal_jit_preambles.cpp` so we can compile MLX’s `compiled.cpp` and `jit_kernels.cpp` without running CMake.

- libfmt and other vendors
  - Using `FMT_HEADER_ONLY` for the addon keeps the binding self-contained. For external tools (standalone labs), add robust detection (pkg-config, Homebrew, MacPorts, apt, conda) with a graceful header-only fallback.

- Distributed layer
  - Always compile the stubs (`no_mpi`, `no_nccl`, `no_ring`) so the distributed interfaces link cleanly on macOS where those backends aren’t present.

- Metal init and ObjC scope
  - Call Metal initialization (and device info) under a scoped autorelease pool and only when the first op needs it. Loader-time init can crash if the ObjC environment isn’t fully ready.

- Node-API lifecycle and finalizers
  - Prefer `Env.SetInstanceData` with the default finalizer. Avoid adding duplicate cleanup hooks for the same pointer; double finalization will crash at teardown.

- API parity and namespacing
  - Dtypes as real objects (e.g., `mlx.float32`) are a better UX than ad-hoc strings and align with Python MLX.
  - Export a single namespace (`mlx.core`) that mirrors Python’s shape and leaves room for `mlx.nn` and other subpackages.

- Build isolation and iteration speed
  - Keep experiments under `labs/`; wire only the minimal generator into the main build. Avoid triggering full-rebuilds of the vendor tree for every iteration.

- Next improvements
  - Add a JS loader for packaging (so `require('mlx')` works out-of-repo without path hacks).
  - Round out factories (nested arrays, dtype inference), extend `mlx.nn`.
  - Prebuild macOS arm64 binaries.

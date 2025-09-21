# Architecture Overview (Node 22 / N-API)

- Binding: pure Node-API via `node-addon-api` (ABI-stable), no NAN.
- MLX linkage: build a static `mlx_core.a` from the vendored MLX sources and link it into the addon.
- Metal JIT: at build time we generate one amalgamated source (`node/generated/metal_jit_preambles.cpp`) from the vendored kernel files using a small C++ helper (`labs/tools/jit_embed`). This defines the functions declared in `mlx/backend/metal/jit/includes.h` (e.g., `utils()`, `binary_ops()`, …) so MLX’s `compiled.cpp`/`jit_kernels.cpp` link and the Metal runtime can JIT kernels as needed.
- Device/runtime: `node/src/native/runtime.mm` encapsulates Metal initialization (scoped autorelease pool, default device) and is invoked lazily on the first op to avoid loader-time hazards.
- Streams: the addon exposes default/new streams and synchronisation to mirror MLX semantics. More stream-aware ops are added incrementally so API parity with Python is preserved where sensible.
- Lifetime: per-Environment instance data via `Env.SetInstanceData` (with the default finalizer), plus `napi_type_tag` where we need stronger type checks.

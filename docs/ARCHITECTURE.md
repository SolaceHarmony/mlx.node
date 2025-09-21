Architecture (Node MLX)
=======================

Layers
------
- Vendor MLX core (C++): `vendor/mlx/**`, including GPU backends and Metal JIT.
- Generated Metal JIT sources: `node/generated/jit/*.cpp` (produced at build time by
  MLX’s `make_compiled_preamble.sh`) and compiled into `mlx_core.a`.
- Native addon (Node-API): `node/src/native/*.cc|mm` built as `mlx.node`.
- JS loader: `node/index.js` exports a single top‑level namespace object `mlx` with
  `mlx.core` members.

Binding surface (Python parity)
-------------------------------
- Dtypes: real objects (e.g., `mlx.float32`), exported under both `mlx` and
  `mlx.core`. `issubdtype` is provided via `mlx.dtype.issubdtype`.
- Array class: `mlx.core.Array` with `shape()`, `dtype()`, `eval()`, `toTypedArray()`.
- Factories & ops (partial, expanding): `zeros`, `ones`, `full`, `reshape`,
  `transpose`, `moveaxis`, `swapaxes`, `add`, `multiply`, `matmul`, `where`, and
  `array(typed, shape[, dtype][, stream])`.
- Streams: `default_stream(device)`, `new_stream(device)`, `synchronize(stream)`, and
  `Stream`/`StreamContext` classes. Keyword‑only `stream` arg is last for every op.

Runtime
-------
- Metal init: `runtime.mm` creates a scoped autorelease pool, probes
  `MTLCreateSystemDefaultDevice()` and binds MLX’s GPU device. Default device is set
  to GPU on first use.
- Instance data: All callbacks receive `AddonData` (via `info.Data()`), mirroring the
  Python binding’s state pattern. Dtype parsing uses this, avoiding fragile env-globals.
- Memory & ArrayBuffer: data is copied into/out of MLX arrays; zero-copy paths for
  CPU TypedArray are implemented in Array factories (guarded by dtype/device).

Build pipeline
--------------
- `node-gyp` target `mlx_core` runs `scripts/gen_mlx_metal_jit.sh`, which invokes
  MLX’s `make_compiled_preamble.sh` for the same source set as CMake to generate JIT
  preambles.
- Generated sources compile into `mlx_core.a` along with MLX sources; the addon links
  to `mlx_core.a`.
- `METAL_PATH` is defined to the vendored `mlx.metallib`.

Error handling & cleanup
------------------------
- Node-API exceptions map to JS errors; GPU init failures surface clearly.
- AddonData is attached via `Env.SetInstanceData` with the default finalizer.

Design principles
-----------------
- Transliterate Python MLX exactly (signatures, dtype rules, stream kw-only). Adapt
  only where Node idioms require (TypedArray, Node-API lifecycle, cleanup hooks).
- GPU-first: no CPU-only shunts; Metal device is the default.


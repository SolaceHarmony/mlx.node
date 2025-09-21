Porting Notes (Python → Node)
=============================

Goals
-----
- Keep a strict 1:1 API parity with Python MLX (function names, signatures, defaults,
  dtype semantics, keyword-only stream).
- Run on Metal (GPU) by default; no CPU-only detours.
- Avoid inventing new semantics: transliterate python/src/ops.cpp and related helpers.

Key decisions
-------------
- Dtypes are objects (not strings). Node accepts only `mlx.core.Dtype` values
  (e.g., `mlx.float32`), matching Python MLX.
- Dtype unwrapping uses per-callback `AddonData` (like Python’s nanobind state) and
  `Dtype.key` to identify, avoiding env-global lookups.
- Upstream JIT preambles are generated with MLX’s `make_compiled_preamble.sh`.
  We do not maintain a custom JIT.
- Streams mirror Python’s `StreamOrDevice` arg semantics; every op accepts a kw-only
  `stream` as the last parameter.

Transliteration rules (apply directly from python/src/ops.cpp)
--------------------------------------------------------------
- zeros/ones: default dtype=float32 when omitted.
- full(shape, vals, dtype=None, *, stream=None): infer dtype from `vals` when dtype is
  None (scalar_to_dtype); broadcast arrays per MLX rules.
- arange:
  - (start, stop[, step], dtype=None): dtype=None ⇒ float32 if any arg is float,
    else int32.
  - (stop[, step], dtype=None): dtype=None ⇒ step ? promote_types(dtype(stop),
    dtype(step)) : dtype(stop).
- array/asarray: accept nested JS arrays, scalars, TypedArray/ArrayBuffer; dtype
  inference and promotion match Python MLX exactly; no-copy on CPU when safe.

Node-specific adaptations
-------------------------
- TypedArray/ArrayBuffer interop for inputs/outputs.
- Node-API environment cleanup via instance data finalizers; no custom global singletons.
- Objective-C++ autorelease pools around Metal initialization.

Directory notes
---------------
- `docs/ARCHITECTURE.md`: binding layers and lifecycle.
- `docs/BUILDING.md`: build steps and JIT generation.
- `labs/`: experiments/tools (not part of production build).

Open items
----------
- Implement transliterated `full`, `arange`, `array/asarray` exactly as Python.
- Remove temporary non-Python helpers from `mlx.core` once staging passes.

How‑To: Using the Node Port Today
---------------------------------

Quick start
-----------
```js
const mx = require('./node');

// Dtypes are objects (just like Python):
const f32 = mx.float32;        // or mx.core.float32
const f16 = mx.float16;

// Streams: pass as keyword‑last arg (Python parity)
const gpu = mx.core.default_stream('gpu');

// Basic ops
const a = mx.core.zeros([2, 2], f32);          // dtype default is float32 if omitted
const b = mx.core.ones([2, 2], f16, gpu);      // run on GPU stream
const c = mx.core.matmul(a, mx.core.transpose(b));

console.log(a.shape(), a.dtype());             // [2, 2], 'float32'
console.log(b.shape(), b.dtype());             // [2, 2], 'float16'
```

What works now (mirrors Python)
-------------------------------
- Namespacing: top‑level `mlx`, with `mlx.core` containing Array, dtypes, ops, streams.
- Dtypes: real objects only (e.g., `mx.float32`). Passing strings raises `TypeError`.
- Array class: `shape()`, `dtype()`, `eval()`, `toTypedArray()`.
- Ops (parity stage 1):
  - factories: `zeros`, `ones`, `full` (scalar value), `array(typed, shape[, dtype][, stream])`
  - transforms: `reshape`, `transpose`, `moveaxis`, `swapaxes`
  - arithmetic: `add`, `multiply`, `matmul`, `where`
- Streams: `default_stream(device)`, `new_stream(device)`, `synchronize(stream)`.
- GPU‑first: default device is GPU on first use; no CPU shunt.

In progress / not yet (transliteration next)
-------------------------------------------
- dtype inference parity for `full(shape, vals, dtype=None, *, stream=None)` — infer from `vals` exactly as Python (scalar_to_dtype; broadcast array inputs).
- `arange` overloads: dtype selection rules and promotion identical to Python.
- `array/asarray` (nested JS arrays, scalars, TypedArray/ArrayBuffer) with full Python promotion rules and no‑copy on CPU when safe.
- Removal of temporary non‑Python helpers under `mlx.core` (kept only in labs/).

Binding pattern (mirroring Python)
----------------------------------
- Per‑callback state: each N‑API callback receives `AddonData` via `info.Data()`,
  just like Python binds state via nanobind. This is used for safe Dtype/Stream
  unwrapping and avoids fragile env‑global lookups.
- Keyword‑only `stream` arg: every op accepts the stream/device as the last parameter
  (mirrors Python’s kw‑only `stream`).
- Dtypes as identities: Node exposes a `Dtype` class and instances for each builtin
  dtype (`mlx.float32`, `mlx.int32`, etc.). `issubdtype` lives under `mlx.dtype`.

TypedArray interop (JS ↔ MLX)
-----------------------------
- Outbound: `toTypedArray()` returns the exact typed view for the tensor’s dtype
  (e.g., float32→Float32Array, float16/bfloat16→Uint16Array of raw bits, int64→BigInt64Array).
- Inbound: `array(typedArray, shape[, dtype][, stream])` copies data in; CPU no‑copy
  paths are added where Python would allow them.

JIT & Metal (no custom work)
----------------------------
- We rely on MLX’s upstream `make_compiled_preamble.sh` to generate JIT preambles at
  build time. The addon compiles those sources into the core static library along with
  MLX vendor code and uses the vendored `mlx.metallib` for kernels.

FAQ
---
- “It accepts string dtypes?” → No. Like Python MLX, pass dtype objects, e.g., `mx.float32`.
- “Where do I put device/stream?” → Always the last argument: `op(..., *, stream=...)`.
- “Why does float16 come back as Uint16Array?” → Python mirrors raw 16‑bit pattern; so do we.

Lessons Learned
---------------
- Use upstream MLX JIT generation, not custom code.
  - MLX’s `make_compiled_preamble.sh` stitches MSL sources with the correct jit headers (e.g., `jit/bf16.h` selects the Metal‑specific `bf16.h`).
  - Naively concatenating kernel sources breaks include ordering and feature flags; stick to the MLX preprocessor path.

- Bind state per callback (info.Data), not via env globals.
  - Dtype/Stream constructors must be read from `AddonData` that’s passed into each binding. Accessing env instance data at arbitrary times led to null derefs in early bring‑up.
  - We now detect dtype via `Dtype.key` and unwrap using the constructor from `AddonData`.

- Dtypes are objects only.
  - Matching Python MLX, strings are not accepted for dtype. Export `mlx.float32`, etc., and treat them as identities.

- GPU is not “just CPU pointers.”
  - Even on UMA, Metal uses `MTLBuffer` resources with storage modes and explicit synchronization. A CPU pointer is not a GPU handle.
  - MLX abstracts this correctly; from Node, we must feed MLX arrays (wrapped/buffered) and let it manage command encoders and sync.

- Broadcasting: rely on MLX.
  - For `full(shape, vals)`, when `vals` is an array, pass it to MLX and let MLX broadcast. If shapes don’t match, MLX raises a clear error.
  - Don’t attempt to reimplement broadcast in the binding.

- TypedArray ingestion rules.
  - If shape is omitted, assume 1D `[len]`.
  - Nested JS arrays: flatten and infer shape (numbers only); promotion rules will be unified with Python’s `to_array`.
  - CPU no‑copy is possible only when dtype/layout match; on GPU we copy and schedule on a stream.

- API surface discipline.
  - Keep only Python MLX ops under `mlx.core`. Diagnostic helpers belong in `labs/` and should not leak into the public API.

- Array dtype reporting.
  - Python returns a `Dtype` object; Node currently returns a string but will be switched to return the `Dtype` object to match parity.

- Build realities & artifacts.
  - Generating JIT sources is heavy; we place them under `node/generated/jit/` and ignore them in git.
  - Vendored `mlx.metallib` is large; GitHub warns about file size. Consider LFS if we split vendors later.
  - Use `FMT_HEADER_ONLY` to avoid linking libfmt.

- Lazy evaluation expectations.
  - MLX arrays are lazy; reading `toTypedArray()` (or `data<T>()`) triggers evaluation and synchronization.
  - Keep evaluation at the MLX API boundary; bindings shouldn’t manually flush command buffers.

- Keyword‑only `stream` semantics.
  - Always treat stream/device as the last argument, mirroring Python’s kw‑only param.
  - This avoids ambiguous positional parsing and keeps 1:1 signatures.


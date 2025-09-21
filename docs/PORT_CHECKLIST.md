Port Checklist (Python → Node MLX)
==================================

Legend: [x] done · [~] in progress · [ ] pending · (notes)

Core surface (mlx.core)
-----------------------
- [x] Dtype objects: `mlx.float16/32/64`, `mlx.bfloat16`, ints, bool, complex64
  - [x] `mlx.dtype.issubdtype`
  - [x] Dtype.key/name/size/category; dtype objects only (no strings)
- [x] Streams: `default_stream(device)`, `new_stream(device)`, `synchronize(stream)`
- [x] Array class: `shape()`, `dtype()`, `eval()`, `toTypedArray()`
  - `dtype()` returns a Dtype object (e.g., `mlx.float32`) — Python parity
- [~] Factories/ops (parity stage 1)
  - [x] zeros(shape, dtype=float32, *, stream)
  - [x] ones(shape, dtype=float32, *, stream)
  - [x] zeros_like(a, *, stream)
  - [x] ones_like(a, *, stream)
  - [x] full(shape, vals, dtype=None, *, stream) — scalar + array/TypedArray vals (broadcast); scalar dtype inference
  - [x] arange(start, stop[, step], dtype=None, *, stream) — dtype rules per Python
  - [x] arange(stop[, step], dtype=None, *, stream)
  - [ ] linspace(start, stop, num=50, dtype=float32, *, stream)
- [~] array(x, dtype=None, *, stream) — unified conversion
  - DONE: scalars, TypedArray/ArrayBuffer, existing MLX Array; nested lists default to float32
  - TODO: mixed-type nested lists use promote_types (exact Python promotion)
- [~] asarray(x, dtype=None, *, stream)
  - DONE: same unified conversion; identity/astype/copy for MLX Array
  - TODO: CPU no‑copy when dtype/layout safe (blocked on allocator ownership)
  - [x] reshape(a, shape, *, stream)
  - [x] transpose(a[, axes], *, stream)
  - [x] moveaxis(a, source, destination, *, stream)
  - [x] swapaxes(a, axis1, axis2, *, stream)
  - [ ] flatten(a, start_axis=0, end_axis=-1, *, stream)
  - [ ] unflatten(a, axis, shape, *, stream)
  - [ ] expand_dims(a, axis, *, stream)
  - [ ] squeeze(a, axis=None, *, stream)
  - [x] add(a, b, *, stream)
  - [x] multiply(a, b, *, stream)
  - [x] matmul(a, b, *, stream)
  - [x] where(cond, x, y, *, stream)

Reductions & indexing
---------------------
- [ ] sum/mean/min/max/argmin/argmax (axes, keepdims, *, stream)
- [ ] take/put, gather/scatter family (match Python surface)
- [ ] sort/argsort/topk — export where present

Math & special ops
------------------
- [ ] unary ops (sign, abs, negative, etc.) — expose as in Python
- [ ] softmax/logsumexp
- [ ] scan
- [ ] fft/ifft

Random & initialization
-----------------------
- [ ] random.* coverage (if exposed in Python MLX)

Device & memory
---------------
- [x] Default Metal device; GPU init via runtime.mm (autorelease pool)
- [ ] copy/to_device/from_device — surface parity with Python if applicable

TypedArray interop
------------------
- [x] toTypedArray() returns exact JS typed array by dtype
- [ ] array/asarray CPU no‑copy when dtype/layout match (blocked; copy used for correctness)

Dtype inference & promotion
---------------------------
- [~] scalar_to_dtype, promote_types: implemented for full/arange; unify array/asarray (nested lists) next

Error handling & lifecycle
--------------------------
- [x] Per‑callback AddonData (info.Data) for safe unwrapping and ctor access
- [x] Exceptions mapped to JS; clear error messages for Metal init/JIT failures
- [x] InstanceData finalizer for addon cleanup (no double free)

Docs & examples
---------------
- [x] ARCHITECTURE.md, PORTING_NOTES.md (with how‑to), BUILDING.md
- [ ] Examples/tests: dtype-precision; array/asarray promotion; stream/device cases

Notes
-----
- Policy: transliterate Python MLX exactly. No semantics invention; only Node‑idiom
  adaptations (TypedArray, AddonData, cleanup hooks).
- Public surface is Python-only (diagnostics in labs/, not exported).
- GPU‑first: no CPU‑only fallbacks. All ops accept `stream` keyword last.

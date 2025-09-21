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


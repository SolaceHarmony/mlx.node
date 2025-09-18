# MLX Node Parity Checklist

Goal: feature-by-feature parity with Python's `mlx` API while adhering to the Node C++ bindings (no Python shim).

Reference source map: see `PORTING_PLAN.md` for the list of upstream C++ and Python files feeding each milestone.

## Core Infrastructure
- [x] Build native addon (`mlx_array.node`) via N-API with MLX core sources (`array.cpp`, `dtype.cpp`, allocators).
- [ ] Introduce shared build config covering additional MLX sources as we expand surface area (ops, evaluators, transforms).

## DType Surface (`mlx.core.dtype`)
- [x] Expose MLX dtype constants with key/size/category accessors.
- [x] Link native `mlx::core::issubdtype` helpers to unblock `dtype.category` (fix crash).
- [x] Surface dtype category enums and Node `issubdtype` parity helpers.
 - [x] Provide Python-parity helpers (e.g. key enumeration API mirroring `__dir__`, formatting introspection).
- [ ] Add comprehensive dtype tests (edge categories, comparisons, promotions) and ensure stability under parallel runs.

## Array Constructors (`mlx.core.array`)
- [x] Instantiate arrays from host buffers for baseline coverage (`float32`, `int32`, `bool`, `complex64`).
- [ ] Implement zero-copy/unified-memory backed constructors to avoid JS heap duplication for large model weights.
 - [x] Support scalar/shape utilities (e.g. broadcasting rules, `array.zeros_like`, `array.ones_like`).

## Array Operations
- [ ] Port structural ops: `reshape`, `transpose`, `swapaxes`, `moveaxis`.
- [ ] Implement indexing/slicing semantics consistent with Python `mlx.core.array`.
- [ ] Add arithmetic and comparison ops leveraging MLX primitives (`add`, `mul`, `where`, etc.).
- [ ] Verify gradient/autodiff compatibility once higher-level ops are in place.

## Memory & Performance
- [ ] Wire unified allocator entry points so tensors can reference MLX-managed memory without JS copies.
- [ ] Surface cache/limit controls (`mlx.core.set_memory_limit`, etc.) in Node API.
- [ ] Benchmark baseline ops to ensure parity with Python within expected overhead bounds.

## React/Next Integration
- [ ] Produce React 19/Next.js friendly wrappers ensuring async server components can orchestrate MLX workloads.
- [ ] Document hydration-safe patterns for tensor creation/usage in shared memory contexts.

## Tooling & Tests
- [x] Mocha + TypeScript test harness covering core entry points.
- [ ] Add targeted crash/regression tests for dtype/array interop (LLDB snippets, stress cases).
- [ ] Automate build/test via CI once the surface stabilizes.

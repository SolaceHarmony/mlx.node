# MLX Node Parity Checklist

Goal: feature-by-feature parity with Python's `mlx` API while adhering to the Node C++ bindings (no Python shim).

Reference source map: see `PORTING_PLAN.md` for the list of upstream C++ and Python files feeding each milestone.

## Core Infrastructure
- [x] Build native addon (`mlx_array.node`) via N-API with MLX core sources (`array.cpp`, `dtype.cpp`, allocators).
- [x] Introduce shared build config covering additional MLX sources as we expand surface area (ops, evaluators, transforms).

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
- [x] Port structural ops: `reshape`, `transpose`, `swapaxes`, `moveaxis`.
- [ ] Implement indexing/slicing semantics consistent with Python `mlx.core.array`.
- [x] Add arithmetic and comparison ops leveraging MLX primitives (`add`, `multiply`, `where`, etc.) with scalar support.
- [ ] Verify gradient/autodiff compatibility once higher-level ops are in place.

## React/Next Integration (high priority)
- [ ] Produce React 19/Next.js friendly wrappers ensuring async server components can orchestrate MLX workloads.
- [ ] Document hydration-safe patterns for tensor creation/usage in shared memory contexts.

## Memory & Performance
- [ ] Wire unified allocator entry points so tensors can reference MLX-managed memory without JS copies.
- [ ] Surface cache/limit controls (`mlx.core.set_memory_limit`, etc.) in Node API.
- [ ] Benchmark baseline ops to ensure parity with Python within expected overhead bounds.

## I/O & Model Loading (after core tensor ops)
- [ ] Restore GGUF + safetensors loaders (vendor gguflib + nlohmann/json) once native build plumbing is ready.
- [x] Stage upstream `mlx.metallib` and kernel sources in `node/vendor` so Metal runtime can initialize under Node.
- [ ] Automate metallib generation during `npm run build` once the Metal toolchain is available in CI/dev environments.

## Optimizers (`mlx.optimizers`)
- [x] Base `Optimizer` class with state management and tree-based parameter handling.
- [x] `SGD` optimizer (structure and validation complete, blocked on missing core ops).
- [ ] Complete `SGD.applySingle()` implementation (requires subtract, scalar ops, astype).
- [x] Add `Adam` optimizer (structure and validation complete, blocked on missing core ops).
- [ ] Complete `Adam.applySingle()` implementation (requires subtract, divide, square, sqrt, rsqrt, power).
- [x] Add `Adamax` optimizer (fully implemented with all required ops).
- [x] Add `Lion` optimizer (fully implemented with all required ops).
- [x] Add `RMSprop` optimizer (fully implemented with all required ops).
- [x] Add `AdamW` optimizer.
- [ ] Add `Adagrad` optimizer.
- [ ] Add scheduler support (`mlx.optimizers.schedulers`).

## Tooling & Tests
- [x] Mocha + TypeScript test harness covering core entry points.
- [x] Basic optimizer tests (constructor, validation, state management).
- [ ] Add targeted crash/regression tests for dtype/array interop (LLDB snippets, stress cases).
- [ ] Add integration tests for optimizers with gradient application.
- [ ] Automate build/test via CI once the surface stabilizes.

MLX Node Architecture (GPU‑first)
=================================

This note captures the shape of the Node bindings and how we initialise and
use MLX on Apple Silicon with Metal, React 19, and streaming.

Goals
-----

- ABI‑stable addon via Node‑API (no direct V8/nan dependencies).
- GPU‑first: default device is Metal GPU; CPU path is available but not the
  default.
- Context‑aware initialisation: no global statics that assume a single VM; all
  constructors and singletons are stored per‑Environment.
- React 19/Next‑ready streaming helpers (SSE) for progressive server rendering.

Layers
------

- Native (C++): `node/src/native/*`
  - `runtime.{h,cc}`: small singleton providing `EnsureMetalInit()` and
    a `WithMetalPool()` helper. Initialises GPU once under a scoped
    `NS::AutoreleasePool`, sets default device to GPU.
  - `array.cc`, `dtype.cc`, `stream.cc`: Node‑API class/function bindings.
  - Vendored MLX sources live under `node/vendor/mlx/*` and are linked into a
    static library target `mlx_core` used by the addon.

- JS/TS surface: `node/src/*`
  - `core/` wraps the native entrypoints with ergonomic TypeScript APIs.
  - `streaming/` exposes SSE utilities (`eventStreamResponse`, framing).
  - `react/` contains React 19 hooks and server helpers.

GPU bring‑up
------------

- On the first op (e.g. `zeros()`), the addon calls
  `Runtime::EnsureMetalInit()`: this creates an autorelease pool, acquires the
  Metal device via MLX (`metal::device(Device::gpu)`), and sets the default
  device to GPU.
- Vendored `metal::device_info()` is guarded to throw if the underlying
  Objective‑C selectors return nil (device/name/architecture), avoiding NIL
  dereferences.

Streams
-------

- Streams are surfaced via `stream.cc` and mirrored in `src/core/stream.ts`.
- All ops accept an optional `StreamOrDevice` value and default to the active
  device/stream if omitted. Server code may pin work to a stream when building
  SSE responses.

React 19 / Next.js
------------------

- Use `streaming.eventStreamResponse(asyncIterable)` in route handlers or
  server actions to emit tensor frames progressively (metadata + data chunks).
- Client components subscribe with `useTensorStream` and rehydrate into typed
  arrays for rendering.

Build notes
-----------

- The Node build links Apple frameworks (Metal, Accelerate, Foundation,
  QuartzCore). The Metal kernels are loaded via `METAL_PATH` defined in
  `binding.gyp`.
- We removed the legacy `nan_lab` target to keep the addon purely Node‑API and
  reduce rebuild surface.

Next steps
----------

- Expand array ops coverage (`indexing`, broadcasts, comparisons).
- Zero‑copy constructors for large weights.
- Add regression tests for GPU initialisation and stream correctness.


# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Changed
- Reworked the Node addon lifetime management to use environment-scoped
  instance data (`AddonData`) and cleanup hooks so our wrappers follow the
  Native Abstractions for Node.js guidance on context-aware initialisation.
- Node addon now defers GPU/Metal initialisation until first use and routes
  it through a small C++ runtime singleton. This mirrors MLX’s Python path
  while avoiding unsafe loader-time Metal calls.

### Fixed
- Normalized LAPACK pointer casting in the shared C++ backend so both the
  Node.js bindings and the Python package call Accelerate with compatible
  signatures on macOS (complex buffers and mutable workspace parameters).
- Hardened MLX Metal `device_info()` in the vendored sources to guard against
  unexpected nil returns from the underlying Metal APIs and emit descriptive
  exceptions instead of segfaults.

### Added
- Bound core tensor ops (`reshape`, `transpose`, `moveaxis`, `swapaxes`, `add`,
  `multiply`, `where`) and surfaced MLX stream/device helpers so React/Next.js
  consumers can mirror the Python API while streaming results over SSE.
- Exported `gpu_info()` from the Node addon for quick visibility into Metal
  device name, architecture, and resource limits during bring-up.
- Authored initial React 19/Next.js streaming architecture docs under
  `docs/src/dev/node_streaming.rst`.
- Removed the legacy `nan_lab` build target and lab sources; the addon is now
  pure Node‑API throughout.

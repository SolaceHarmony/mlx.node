# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Fixed
- Normalized LAPACK pointer casting in the shared C++ backend so both the
  Node.js bindings and the Python package call Accelerate with compatible
  signatures on macOS (complex buffers and mutable workspace parameters).

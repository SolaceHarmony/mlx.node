Labs
====

This folder is for experiments and helper tools that are not part of the
production build. Nothing in here is required at runtime for the `mlx` addon.

- tools/jit_embed.cpp: small C++ precompile tool that reads vendored MLX Metal
  kernel/preamble files and emits a single generated C++ source compiled into
  the core static library. This mirrors MLX’s CMake JIT preamble generation.

Do not vendor any compiled outputs from labs/ into git. Production binding.gyp
invokes the generator as a build action and compiles the generated sources.

Building the Node Addon
=======================

Prereqs
-------
- macOS with Xcode command line tools (Metal available).
- Node.js v22.x and node-gyp.

Steps
-----
1) From repo root:

   - Install JS deps if needed: `cd node && npm ci`.
   - Build the addon: `cd node && npx node-gyp configure build`.

2) The build will:

   - Run `scripts/gen_mlx_metal_jit.sh`, which invokes MLX’s upstream
     `mlx/backend/metal/make_compiled_preamble.sh` to generate
     `node/generated/jit/*.cpp` (Metal JIT preambles).
   - Compile generated JIT sources and MLX vendor sources into `mlx_core.a`.
   - Link the Node-API addon `mlx.node`.

3) Loading the module:

   ```
   const mlx = require('./node');
   const z = mlx.core.zeros([2,2], mlx.float32);
   const s = mlx.core.default_stream('gpu');
   const o = mlx.core.ones([2,2], mlx.float16, s);
   ```

Notes
-----
- Generated sources live under `node/generated/jit/` and are ignored by git.
- `METAL_PATH` points to the vendored `vendor/mlx/backend/metal/kernels/mlx.metallib`.
- Build times can be long on first compile (full MLX); subsequent builds are faster.


# mlx.node Implementation Progress

## Status: Core Ops Complete

220+ tests passing across all implemented modules.

## Implemented

### Batch 1: Core Math & Shape Ops (47 ops)
- **Unary math:** ceil, floor, round, isnan, isinf, isfinite, logical_not, sinh, cosh, arcsinh, arccosh, arctanh, degrees, radians, erfinv, expm1
- **Cumulative:** cumsum, cumprod, cummax, cummin
- **Binary:** floor_divide, remainder, logical_and, logical_or, bitwise_and/or/xor, left_shift, right_shift
- **Reduction:** all, any, array_equal
- **Shape/creation:** flatten, eye, identity, linspace, tril, triu, broadcast_to, repeat, tile, sort, argsort, diag, diagonal, topk

### Batch 2: Random, Device, Memory, FFT, Fast
- **Random:** seed, key, split, randint, uniform, normal, gumbel, laplace, truncated_normal, permutation, categorical
- **Device:** default_device, set_default_device, is_available
- **Memory:** get_active_memory, get_peak_memory, get_cache_memory, reset_peak_memory, set_cache_limit, set_memory_limit, clear_cache
- **FFT:** fft, ifft, rfft, irfft, fft2, ifft2, rfft2, irfft2, fftn, ifftn, rfftn, irfftn, fftshift, ifftshift
- **Fast:** rms_norm, layer_norm, rope

### Batch 3: Advanced Ops (50+ ops)
- **Math:** log2, log10, isposinf, isneginf, bitwise_invert, conjugate/conj, real, imag, stop_gradient
- **Products:** outer, inner, kron
- **Parameterized:** nan_to_num, allclose, isclose, contiguous, unflatten, partition, argpartition, roll, tri, view, hadamard_transform
- **Multi-array:** meshgrid, broadcast_arrays, atleast_1d/2d/3d, concat, divmod, permute_dims, slice_update, put_along_axis
- **Convolution:** conv_general, conv_transpose1d/2d/3d, convolve
- **Tensor:** einsum, tensordot, block_masked_mm, gather_mm, segmented_mm
- **Quantization:** quantize, dequantize, quantized_matmul, gather_qmm

### Batch 4: Linalg & TS-Only Ops (20 ops)
- **Linalg (CPU-default):** inv, pinv, solve, solve_triangular, cholesky, cholesky_inv, tri_inv, svd, qr, lu, lu_factor, eig, eigvals, eigh, eigvalsh, cross
- **TS compositions:** trunc, broadcast_shapes, convolve, einsum_path

### Pre-existing
- Array creation, dtype system, streams, basic arithmetic, trig, reduction, reshape, transpose, matmul, conv1d/2d/3d, pad, slice, take, split, stack, concatenate, arange, sigmoid, erf, tanh, softmax, logaddexp, logsumexp, clip, etc.

## Remaining (31 issues)

| Category | Issues | Priority | Notes |
|----------|--------|----------|-------|
| Eval | #175-176 | High | Next up - straightforward |
| IO | #150-155 | High | load/save/safetensors/gguf |
| Transforms | #166-174 | Critical | grad, vjp, jvp, vmap, compile - architecturally complex |
| Metal/CUDA | #225-227 | Low | Niche, CUDA deferred |
| Distributed | #249-257 | Low | Deferred until training loop works |
| Export | #271-273 | Low | Depends on eval |

## Known Limitations
- Metallib missing scatter JIT kernels (eye, identity, diag crash at eval)
- Metallib missing steel_gemm kernels (einsum, tensordot crash at eval)
- `array(Int32Array)` creates float32 not int32 (bitwise ops on such arrays fail)
- Linalg ops CPU-only (MLX GPU linalg not yet supported)

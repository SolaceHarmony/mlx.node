Port Checklist (Python → Node MLX)
==================================

Updated: 2026-03-31
Legend: [x] done · [ ] pending

Core surface (mlx.core)
-----------------------
- [x] Dtype objects: float16/32/64, bfloat16, ints, bool, complex64
  - [x] issubdtype, dtype.key/name/size/category
- [x] Streams: default_stream, new_stream, synchronize, stream context
- [x] Array class: shape, dtype, eval, toTypedArray/toArray
- [x] Factories: zeros, ones, full, arange, linspace, eye, identity, tri, tril, triu, meshgrid
- [x] array(x, dtype) — TypedArray, scalars, nested lists; dtype inferred from TypedArray type
- [x] Shape ops: reshape, transpose, moveaxis, swapaxes, flatten, unflatten, expand_dims,
      squeeze, split, stack, concat, broadcast_to, broadcast_arrays, repeat, tile, permute_dims

Indexing & slicing
------------------
- [x] diag, diagonal, take, take_along_axis, put_along_axis
- [x] slice, slice_update, view, as_strided
- [x] gather_mm, gather_qmm

Math & element-wise ops
-----------------------
- [x] Arithmetic: add, subtract, multiply, divide, floor_divide, remainder, divmod, power, negative
- [x] Unary: abs, sign, ceil, floor, round, trunc, sqrt, rsqrt, square, reciprocal, clip
- [x] Exp/log: exp, expm1, log, log2, log10, log1p, logaddexp, logsumexp, logcumsumexp
- [x] Trig: sin, cos, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh, arcsinh, arccosh, arctanh
- [x] Special: erf, erfinv, sigmoid, degrees, radians, nan_to_num, softmax
- [x] Comparison: equal, not_equal, less, less_equal, greater, greater_equal, maximum, minimum
- [x] Logical: logical_and, logical_or, logical_not, all, any, allclose, isclose, array_equal
- [x] Type check: isnan, isinf, isfinite, isposinf, isneginf
- [x] Bitwise: bitwise_and, bitwise_or, bitwise_xor, bitwise_invert, left_shift, right_shift
- [x] Complex: conj/conjugate, real, imag
- [x] Matmul: matmul, addmm, einsum, einsum_path, tensordot, inner, outer, kron, block_masked_mm

Reductions
----------
- [x] sum, mean, prod, min, max, std, variance, trace
- [x] argmin, argmax, all, any
- [x] cumsum, cumprod, cummax, cummin

Sorting & selection
-------------------
- [x] sort, argsort, topk, partition, argpartition, roll

Convolution
-----------
- [x] conv1d, conv2d, conv3d, conv_general, convolve
- [x] conv_transpose1d, conv_transpose2d, conv_transpose3d

Quantization
------------
- [x] quantize, dequantize, quantized_matmul, segmented_mm

Advanced
--------
- [x] hadamard_transform, contiguous, stop_gradient, pad
- [x] broadcast_shapes, number_of_elements

IO
--
- [x] load (npy, safetensors, gguf auto-detect)
- [x] save (npy), save_safetensors, save_gguf

Transforms
----------
- [x] eval, async_eval
- [x] grad, value_and_grad, vjp, jvp
- [x] vmap, compile, checkpoint, enable_compile, disable_compile

Export
------
- [x] export_function, import_function, export_to_dot

Linear algebra (mlx.linalg)
----------------------------
- [x] inv, solve, solve_triangular, tri_inv
- [x] cholesky, cholesky_inv, qr, lu, lu_factor, svd
- [x] eig, eigh, eigvals, eigvalsh
- [x] norm, cross, pinv
- Note: All linalg ops CPU-only (matches upstream MLX)

Random (mlx.random)
--------------------
- [x] seed, key, split
- [x] normal, uniform, randint, bernoulli
- [x] categorical, gumbel, laplace, truncated_normal, multivariate_normal
- [x] permutation

FFT (mlx.fft)
--------------
- [x] fft, ifft, rfft, irfft
- [x] fft2, ifft2, rfft2, irfft2
- [x] fftn, ifftn, rfftn, irfftn
- [x] fftshift, ifftshift

Fast ops (mlx.fast)
--------------------
- [x] rms_norm, layer_norm, rope, scaled_dot_product_attention
- [ ] metal_kernel — custom kernel API
- [ ] cuda_kernel, precompiled_cuda_kernel — N/A on Apple Silicon

Device & memory
---------------
- [x] default_device, set_default_device, is_available
- [x] default_stream, set_default_stream, new_stream, synchronize
- [x] get_active_memory, get_peak_memory, get_cache_memory
- [x] reset_peak_memory, set_cache_limit, set_memory_limit, set_wired_limit, clear_cache

Neural network (mlx.nn)
-----------------------
- [x] 25 activation classes (ReLU, GELU, SiLU, etc.)
- [x] 26 activation functions
- [x] Core layers: Linear, Bilinear, Conv1d/2d/3d, Embedding
- [x] Normalization: BatchNorm, GroupNorm, InstanceNorm, LayerNorm, RMSNorm
- [x] Regularization: Dropout, Dropout2d, Dropout3d
- [x] Pooling: AvgPool1d/2d/3d, MaxPool1d/2d/3d
- [x] Recurrent: RNN, LSTM, GRU
- [x] Transformer: MultiHeadAttention, Transformer, Encoder/Decoder layers
- [x] 14 loss functions (cross_entropy, mse_loss, etc.)
- [x] 6 initializers (glorot_normal/uniform, constant, normal, orthogonal, uniform)

Optimizers (mlx.optimizers)
---------------------------
- [x] SGD, Adam, AdamW, Adamax, Adagrad, AdaDelta
- [x] RMSprop, Lion, Muon, Adafactor
- [x] MultiOptimizer, Optimizer base

Distributed (mlx.distributed)
------------------------------
- [ ] all_gather, all_sum, all_max, all_min — requires MPI
- [ ] init, send, recv, recv_like — deferred

Notes
-----
- GPU-first: all ops run on Metal GPU by default via MLX lazy evaluation
- Linalg ops CPU-only (matches upstream MLX — no GPU linalg yet)
- JIT Metal kernel compilation enabled for scatter and steel_gemm specializations
- 250 tests passing, 0 pending, 0 failing

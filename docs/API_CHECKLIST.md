# MLX Node.js API Implementation Checklist

**Updated:** 2026-03-31
**Legend:** [x] Done | [ ] Not Started

---

## Core Operations (mlx.core) — 190+ functions

### Array Creation — 16 functions
- [x] `arange`
- [x] `atleast_1d`
- [x] `atleast_2d`
- [x] `atleast_3d`
- [x] `eye`
- [x] `full`
- [x] `identity`
- [x] `linspace`
- [x] `meshgrid`
- [x] `ones`
- [x] `ones_like`
- [x] `tri`
- [x] `tril`
- [x] `triu`
- [x] `zeros`
- [x] `zeros_like`

### Shape Manipulation — 17 functions
- [x] `as_strided`
- [x] `broadcast_arrays`
- [x] `broadcast_shapes`
- [x] `broadcast_to`
- [x] `concatenate` / `concat`
- [x] `expand_dims`
- [x] `flatten`
- [x] `moveaxis`
- [x] `permute_dims`
- [x] `repeat`
- [x] `reshape`
- [x] `split`
- [x] `squeeze`
- [x] `stack`
- [x] `swapaxes`
- [x] `transpose`
- [x] `unflatten`

### Indexing & Slicing — 10 functions
- [x] `diag`
- [x] `diagonal`
- [x] `gather_mm`
- [x] `gather_qmm`
- [x] `put_along_axis`
- [x] `slice`
- [x] `slice_update`
- [x] `take`
- [x] `take_along_axis`
- [x] `view`

### Math — Arithmetic — 15 functions
- [x] `add`
- [x] `addmm`
- [x] `divide`
- [x] `divmod`
- [x] `floor_divide`
- [x] `matmul`
- [x] `maximum`
- [x] `minimum`
- [x] `multiply`
- [x] `negative`
- [x] `outer`
- [x] `power`
- [x] `remainder`
- [x] `square`
- [x] `subtract`

### Math — Exponential & Logarithmic — 8 functions
- [x] `exp`
- [x] `expm1`
- [x] `log`
- [x] `log10`
- [x] `log1p`
- [x] `log2`
- [x] `logaddexp`
- [x] `logsumexp`

### Math — Trigonometric — 12 functions
- [x] `arccos`
- [x] `arccosh`
- [x] `arcsin`
- [x] `arcsinh`
- [x] `arctan`
- [x] `arctan2`
- [x] `arctanh`
- [x] `cos`
- [x] `cosh`
- [x] `sin`
- [x] `sinh`
- [x] `tan`
- [x] `tanh`

### Math — Rounding & Absolute — 10 functions
- [x] `abs`
- [x] `ceil`
- [x] `clip`
- [x] `floor`
- [x] `nan_to_num`
- [x] `reciprocal`
- [x] `round`
- [x] `rsqrt`
- [x] `sign`
- [x] `sqrt`
- [x] `trunc`

### Math — Special Functions — 5 functions
- [x] `degrees`
- [x] `erf`
- [x] `erfinv`
- [x] `radians`
- [x] `sigmoid`

### Logical Operations — 14 functions
- [x] `all`
- [x] `allclose`
- [x] `any`
- [x] `array_equal`
- [x] `equal`
- [x] `greater`
- [x] `greater_equal`
- [x] `isclose`
- [x] `less`
- [x] `less_equal`
- [x] `logical_and`
- [x] `logical_not`
- [x] `logical_or`
- [x] `not_equal`

### Type Checking — 6 functions
- [x] `isfinite`
- [x] `isinf`
- [x] `isnan`
- [x] `isneginf`
- [x] `isposinf`
- [x] `issubdtype`

### Reductions — 14 functions
- [x] `all`
- [x] `any`
- [x] `argmax`
- [x] `argmin`
- [x] `logcumsumexp`
- [x] `logsumexp`
- [x] `max`
- [x] `mean`
- [x] `min`
- [x] `prod`
- [x] `std`
- [x] `sum`
- [x] `trace`
- [x] `variance`

### Cumulative Operations — 4 functions
- [x] `cummax`
- [x] `cummin`
- [x] `cumprod`
- [x] `cumsum`

### Sorting & Selection — 6 functions
- [x] `argpartition`
- [x] `argsort`
- [x] `partition`
- [x] `roll`
- [x] `sort`
- [x] `topk`

### Bitwise Operations — 6 functions
- [x] `bitwise_and`
- [x] `bitwise_invert`
- [x] `bitwise_or`
- [x] `bitwise_xor`
- [x] `left_shift`
- [x] `right_shift`

### Complex Number Operations — 3 functions
- [x] `conj` / `conjugate`
- [x] `imag`
- [x] `real`

### Convolution Operations — 8 functions
- [x] `conv1d`
- [x] `conv2d`
- [x] `conv3d`
- [x] `conv_general`
- [x] `conv_transpose1d`
- [x] `conv_transpose2d`
- [x] `conv_transpose3d`
- [x] `convolve`

### Advanced Operations — 12 functions
- [x] `block_masked_mm`
- [x] `contiguous`
- [x] `einsum`
- [x] `einsum_path`
- [x] `hadamard_transform`
- [x] `inner`
- [x] `kron`
- [x] `pad`
- [x] `segmented_mm`
- [x] `softmax`
- [x] `stop_gradient`
- [x] `tensordot`
- [x] `tile`
- [x] `where`

### Quantization — 3 functions
- [x] `dequantize`
- [x] `quantize`
- [x] `quantized_matmul`

### Save/Load — 4 functions
- [x] `load`
- [x] `save`
- [x] `save_gguf`
- [x] `save_safetensors`

---

## Transforms (mlx.core) — 11 functions

- [x] `async_eval`
- [x] `checkpoint`
- [x] `compile`
- [x] `disable_compile`
- [x] `enable_compile`
- [x] `eval`
- [x] `grad`
- [x] `jvp`
- [x] `value_and_grad`
- [x] `vjp`
- [x] `vmap`

---

## Linear Algebra (mlx.linalg) — 17 functions

- [x] `cholesky`
- [x] `cholesky_inv`
- [x] `cross`
- [x] `eig`
- [x] `eigh`
- [x] `eigvals`
- [x] `eigvalsh`
- [x] `inv`
- [x] `lu`
- [x] `lu_factor`
- [x] `norm`
- [x] `pinv`
- [x] `qr`
- [x] `solve`
- [x] `solve_triangular`
- [x] `svd`
- [x] `tri_inv`

---

## Random (mlx.random) — 13 functions

- [x] `bernoulli`
- [x] `categorical`
- [x] `gumbel`
- [x] `key`
- [x] `laplace`
- [x] `multivariate_normal`
- [x] `normal`
- [x] `permutation`
- [x] `randint`
- [x] `seed`
- [x] `split`
- [x] `truncated_normal`
- [x] `uniform`

---

## FFT (mlx.fft) — 14 functions

- [x] `fft`
- [x] `fft2`
- [x] `fftn`
- [x] `fftshift`
- [x] `ifft`
- [x] `ifft2`
- [x] `ifftn`
- [x] `ifftshift`
- [x] `irfft`
- [x] `irfft2`
- [x] `irfftn`
- [x] `rfft`
- [x] `rfft2`
- [x] `rfftn`

---

## Fast Operations (mlx.fast) — 7 functions

- [ ] `cuda_kernel` — N/A on Apple Silicon
- [x] `layer_norm`
- [ ] `metal_kernel` — custom kernel API, not yet implemented
- [ ] `precompiled_cuda_kernel` — N/A on Apple Silicon
- [x] `rms_norm`
- [x] `rope`
- [x] `scaled_dot_product_attention`

---

## Device & Memory Management — 13 functions

- [x] `clear_cache`
- [x] `default_device`
- [x] `default_stream`
- [x] `get_active_memory`
- [x] `get_cache_memory`
- [x] `get_peak_memory`
- [x] `is_available`
- [x] `new_stream`
- [x] `reset_peak_memory`
- [x] `set_cache_limit`
- [x] `set_default_device`
- [x] `set_default_stream`
- [x] `set_memory_limit`
- [x] `set_wired_limit`
- [x] `synchronize`

---

## Export — 3 functions

- [x] `export_function`
- [x] `export_to_dot`
- [x] `import_function`

---

## Distributed (mlx.distributed) — 9 functions

- [ ] `all_gather`
- [ ] `all_max`
- [ ] `all_min`
- [ ] `all_sum`
- [ ] `init`
- [ ] `recv`
- [ ] `recv_like`
- [ ] `send`

---

## Neural Network Layers (mlx.nn) — 50+ classes

### Activations — 25 classes
- [x] `CELU`
- [x] `ELU`
- [x] `GELU`
- [x] `GLU`
- [x] `HardShrink`
- [x] `HardTanh`
- [x] `Hardswish`
- [x] `Identity`
- [x] `LeakyReLU`
- [x] `LogSigmoid`
- [x] `LogSoftmax`
- [x] `Mish`
- [x] `PReLU`
- [x] `ReLU`
- [x] `ReLU6`
- [x] `SELU`
- [x] `SiLU`
- [x] `Sigmoid`
- [x] `Softmax`
- [x] `Softmin`
- [x] `Softplus`
- [x] `Softshrink`
- [x] `Softsign`
- [x] `Step`
- [x] `Tanh`

### Core Layers — 7 classes
- [x] `Bilinear`
- [x] `Conv1d`
- [x] `Conv2d`
- [x] `Conv3d`
- [x] `Embedding`
- [x] `Linear`

### Normalization — 5 classes
- [x] `BatchNorm`
- [x] `GroupNorm`
- [x] `InstanceNorm`
- [x] `LayerNorm`
- [x] `RMSNorm`

### Regularization — 3 classes
- [x] `Dropout`
- [x] `Dropout2d`
- [x] `Dropout3d`

### Pooling — 6 classes
- [x] `AvgPool1d`
- [x] `AvgPool2d`
- [x] `AvgPool3d`
- [x] `MaxPool1d`
- [x] `MaxPool2d`
- [x] `MaxPool3d`

### Recurrent — 3 classes
- [x] `GRU`
- [x] `LSTM`
- [x] `RNN`

### Transformer — 5 classes
- [x] `MultiHeadAttention`
- [x] `Transformer`
- [x] `TransformerDecoder`
- [x] `TransformerDecoderLayer`
- [x] `TransformerEncoder`
- [x] `TransformerEncoderLayer`

---

## NN Functions (mlx.nn) — 40+ functions

### Activation Functions — 26 functions
- [x] `celu`
- [x] `elu`
- [x] `gelu`
- [x] `gelu_approx`
- [x] `gelu_fast_approx`
- [x] `glu`
- [x] `hard_shrink`
- [x] `hard_tanh`
- [x] `hardswish`
- [x] `leaky_relu`
- [x] `log_sigmoid`
- [x] `log_softmax`
- [x] `mish`
- [x] `prelu`
- [x] `relu`
- [x] `relu6`
- [x] `selu`
- [x] `sigmoid`
- [x] `silu`
- [x] `softmax`
- [x] `softmin`
- [x] `softplus`
- [x] `softshrink`
- [x] `softsign`
- [x] `step`
- [x] `tanh`

### Loss Functions — 14 functions
- [x] `binary_cross_entropy`
- [x] `cosine_similarity_loss`
- [x] `cross_entropy`
- [x] `gaussian_nll_loss`
- [x] `hinge_loss`
- [x] `huber_loss`
- [x] `kl_div_loss`
- [x] `l1_loss`
- [x] `log_cosh_loss`
- [x] `margin_ranking_loss`
- [x] `mse_loss`
- [x] `nll_loss`
- [x] `smooth_l1_loss`
- [x] `triplet_loss`

### Initialization — 6 functions
- [x] `constant`
- [x] `glorot_normal`
- [x] `glorot_uniform`
- [x] `normal`
- [x] `orthogonal`
- [x] `uniform`

---

## Optimizers (mlx.optimizers) — 12 classes

- [x] `AdaDelta`
- [x] `Adafactor`
- [x] `Adagrad`
- [x] `Adam`
- [x] `Adamax`
- [x] `AdamW`
- [x] `Lion`
- [x] `MultiOptimizer`
- [x] `Muon`
- [x] `Optimizer`
- [x] `RMSprop`
- [x] `SGD`

---

## Summary

| Category | Total | Done | % |
|----------|-------|------|---|
| Core Operations | 190+ | 190+ | 100% |
| Transforms | 11 | 11 | 100% |
| Linear Algebra | 17 | 17 | 100% |
| Random | 13 | 13 | 100% |
| FFT | 14 | 14 | 100% |
| Fast Ops | 7 | 4 | 57% |
| Device/Memory | 15 | 15 | 100% |
| Export | 3 | 3 | 100% |
| Distributed | 8 | 0 | 0% |
| NN Layers | 50+ | 50+ | 100% |
| NN Functions | 46 | 46 | 100% |
| Optimizers | 12 | 12 | 100% |
| **TOTAL** | **386+** | **375+** | **97%** |

### Not Implemented
- **Distributed ops** (8) — requires MPI, deferred
- **metal_kernel** — custom Metal kernel API
- **cuda_kernel / precompiled_cuda_kernel** — N/A on Apple Silicon

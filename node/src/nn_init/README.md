# mlx.nn.orthogonal() Implementation Status

## Summary

This document describes the implementation status of `mlx.nn.orthogonal()` for the MLX Node.js bindings.

## Current Status: Placeholder Implementation

The `orthogonal()` function has been implemented as a **placeholder** in TypeScript. The function signature and validation logic are complete, but the actual orthogonal matrix generation is not yet functional.

## What Has Been Implemented

### 1. Module Structure (node/src/nn_init/index.ts)
- ✅ Created new `nn_init` module following TypeScript patterns
- ✅ Defined `Initializer` type for functional composition
- ✅ Implemented `orthogonal()` function signature:
  - `gain: number = 1.0` - Scaling factor
  - `dtype: MLXDtype = float32` - Output data type
  - Returns an initializer function
- ✅ Added input validation (checks for 2D arrays)
- ✅ Complete JSDoc documentation with examples

### 2. Module Export (node/src/index.ts)
- ✅ Added `nn_init` to module exports
- ✅ Available as `mx.nn_init.orthogonal()`

### 3. Tests (node/test/nn_init.test.ts)
- ✅ Test suite with 7 test cases
- ✅ Tests for function existence and type
- ✅ Tests for input validation (1D, 2D, 3D arrays)
- ✅ Tests for parameter handling (gain, dtype)
- ✅ Tests verify current "not implemented" behavior

## What Remains to Be Done

### Required MLX Operations

The full implementation of `orthogonal()` requires the following MLX operations to be exposed in the Node.js bindings:

1. **`random.normal(shape, dtype?, key?, stream?)`**
   - Location: `mlx::core::random::normal`
   - Required for: Generating initial random matrix
   - Status: Not exposed in Node.js bindings

2. **`linalg.qr(array, stream?)`**
   - Location: `mlx::core::linalg::qr`
   - Required for: QR decomposition
   - Status: Not exposed in Node.js bindings

3. **`diag(array, k?, stream?)`**
   - Location: `mlx::core::diag`
   - Required for: Extracting diagonal of R matrix
   - Status: Not exposed in Node.js bindings

4. **Array slicing operations**
   - Required for: Extracting submatrix from Q
   - Status: May need to be added to MLXArray class

5. **`astype(dtype)`**
   - Required for: Converting final result to target dtype
   - Status: May need to be added to MLXArray class

### Implementation Steps

Once the above operations are available, the implementation would be:

```typescript
export function orthogonal(
  gain: number = 1.0,
  dtype: MLXDtype = float32
): Initializer {
  return (a: MLXArray): MLXArray => {
    // Validate input is 2D
    const shape = a.shape;
    if (shape.length !== 2) {
      throw new Error(
        `Orthogonal initialization requires a 2D array but got a ${shape.length}D array.`
      );
    }

    const [rows, cols] = shape;
    const n = Math.max(rows, cols);

    // Generate random normal matrix
    const rmat = mx.random.normal([n, n]);

    // Perform QR decomposition on CPU
    const [q, r] = mx.linalg.qr(rmat, { stream: mx.cpu });

    // Adjust the sign of Q using the diagonal of R
    const d = mx.diag(r);
    let qAdjusted = mx.multiply(q, mx.sign(d));

    // Slice Q to the desired shape
    qAdjusted = qAdjusted.slice([[0, rows], [0, cols]]);

    // Scale Q by gain
    qAdjusted = mx.multiply(qAdjusted, gain);

    // Convert to specified dtype
    return qAdjusted.astype(dtype);
  };
}
```

## Testing

The current tests verify:
- Function exists and has correct type
- Returns an initializer function
- Validates 2D input requirement
- Throws appropriate errors for current placeholder implementation
- Accepts custom gain and dtype parameters

Once implemented, additional tests should verify:
- Generated matrices are orthogonal (Q^T * Q ≈ I)
- Gain parameter correctly scales the matrix
- Dtype parameter correctly converts output
- Works with various matrix dimensions

## Python Reference

The Python implementation can be found at:
- `python/mlx/nn/init.py` lines 400-441

## Next Steps

1. **Priority 1**: Implement C++ bindings for required operations:
   - `random.normal` → `node/src/native/array.cc`
   - `linalg.qr` → `node/src/native/array.cc`
   - `diag` → `node/src/native/array.cc`

2. **Priority 2**: Add array methods to MLXArray:
   - `.slice()` method
   - `.astype()` method

3. **Priority 3**: Complete `orthogonal()` implementation:
   - Replace placeholder with full implementation
   - Update tests to verify orthogonality properties
   - Add performance tests

4. **Priority 4**: Documentation:
   - Add to API documentation
   - Create usage examples
   - Update completion checklist

## Related Issues

This implementation is part of achieving feature parity with Python MLX for the Node.js bindings.

---

**Created**: 2026-02-12  
**Status**: Placeholder - Requires dependency operations  
**Estimated effort**: Medium (depends on operation bindings)

/**
 * Neural network initialization functions for MLX.
 *
 * This module provides initializers for neural network parameters,
 * following the patterns from Python's mlx.nn.init module.
 */

import MLXArray, { full } from '../core/array';
import type { MLXDtype } from '../core/dtype';
import { float32 } from '../core/dtype';
import { eye, argsort, less, multiply, random } from '../core/ops';

/**
 * Initializer function type that takes an array and returns an initialized array.
 */
export type Initializer = (array: MLXArray) => MLXArray;

/**
 * Calculate the fan-in and fan-out for a given array shape.
 * Used for Glorot and He initialization.
 *
 * @param arr - Array to calculate fan values for
 * @returns Tuple of [fan_in, fan_out]
 */
function calculateFanInFanOut(arr: MLXArray): [number, number] {
  const shape = arr.shape;

  if (shape.length < 2) {
    throw new Error(
      `Glorot / He initialization requires at least 2 dimensional input but got ${shape.length} dimensions.`
    );
  }

  let fanIn = shape[shape.length - 1];
  let fanOut = shape[0];

  if (shape.length > 2) {
    let receptiveField = 1;
    for (let i = 1; i < shape.length - 1; i++) {
      receptiveField *= shape[i];
    }
    fanIn = fanIn * receptiveField;
    fanOut = fanOut * receptiveField;
  }

  return [fanIn, fanOut];
}

/**
 * An initializer that returns an array filled with `value`.
 *
 * @param value - The value to fill the array with
 * @param dtype - The data type of the array. Default: float32
 * @returns An initializer function that takes an array and returns a new array
 *          with the same shape filled with the constant value
 *
 * @example
 * ```typescript
 * const initFn = constant(0.5);
 * const result = initFn(zeros([2, 2]));
 * // result is array([[0.5, 0.5], [0.5, 0.5]], dtype=float32)
 * ```
 */
export function constant(
  value: number,
  dtype?: MLXDtype
): (a: MLXArray) => MLXArray {
  const dtypeToUse = dtype ?? float32;
  return (a: MLXArray): MLXArray => {
    return full(a.shape, value, dtypeToUse);
  };
}

/**
 * A Glorot uniform initializer.
 *
 * This initializer samples from a uniform distribution with a range
 * computed from the number of input (fan_in) and output (fan_out)
 * units according to:
 *
 *   limit = gain * sqrt(6.0 / (fan_in + fan_out))
 *
 * @param dtype - The data type of the array (default: float32)
 * @returns An initializer function that takes an array and optional gain parameter
 */
export function glorot_uniform(
  dtype: any = 'float32'
): (arr: MLXArray, gain?: number) => MLXArray {
  return function initializer(arr: MLXArray, gain: number = 1.0): MLXArray {
    const [fanIn, fanOut] = calculateFanInFanOut(arr);
    const limit = gain * Math.sqrt(6.0 / (fanIn + fanOut));
    return random.uniform(-limit, limit, arr.shape, { dtype });
  };
}

/**
 * A Glorot normal initializer.
 *
 * This initializer samples from a normal distribution with a standard
 * deviation computed from the number of input (fan_in) and output
 * (fan_out) units according to:
 *
 *   std = gain * sqrt(2.0 / (fan_in + fan_out))
 *
 * @param dtype - The data type of the array (default: float32)
 * @returns An initializer function that takes an array and optional gain parameter
 */
export function glorot_normal(
  dtype: any = 'float32'
): (arr: MLXArray, gain?: number) => MLXArray {
  return function initializer(arr: MLXArray, gain: number = 1.0): MLXArray {
    const [fanIn, fanOut] = calculateFanInFanOut(arr);
    const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
    return random.normal(arr.shape, { dtype, scale: std });
  };
}

/**
 * An initializer that returns samples from a normal distribution.
 *
 * @param mean - Mean of the normal distribution (default: 0.0)
 * @param std - Standard deviation of the normal distribution (default: 1.0)
 * @param dtype - The data type of the array (default: float32)
 * @returns An initializer function
 */
export function normal(
  mean: number = 0.0,
  std: number = 1.0,
  dtype: any = 'float32'
): (arr: MLXArray) => MLXArray {
  return function initializer(arr: MLXArray): MLXArray {
    return random.normal(arr.shape, { dtype, loc: mean, scale: std });
  };
}

/**
 * An initializer that returns samples from a uniform distribution.
 *
 * @param low - Lower bound of the uniform distribution (default: 0.0)
 * @param high - Upper bound of the uniform distribution (default: 1.0)
 * @param dtype - The data type of the array (default: float32)
 * @returns An initializer function
 */
export function uniform(
  low: number = 0.0,
  high: number = 1.0,
  dtype: any = 'float32'
): (arr: MLXArray) => MLXArray {
  return function initializer(arr: MLXArray): MLXArray {
    return random.uniform(low, high, arr.shape, { dtype });
  };
}

/**
 * An initializer that returns an identity matrix.
 *
 * @param dtype - The data type of the array. Default: float32
 * @returns An initializer function that takes a square 2D array and returns
 *          an identity matrix of the same shape.
 *
 * @throws {Error} If the input array is not a square 2D matrix.
 *
 * @example
 * ```typescript
 * const initFn = identity();
 * const result = initFn(zeros([2, 2]));
 * // result is eye(2)
 * ```
 */
export function identity(
  dtype: MLXDtype = float32
): (arr: MLXArray) => MLXArray {
  return (arr: MLXArray): MLXArray => {
    const shape = arr.shape;
    if (shape.length !== 2 || shape[0] !== shape[1]) {
      throw new Error(
        `The input array must be a square matrix but got shape [${shape}].`
      );
    }
    return eye(shape[0], { dtype });
  };
}

/**
 * A He normal (Kaiming normal) initializer.
 *
 * Samples from a normal distribution with a standard deviation computed
 * from the number of input (`fan_in`) or output (`fan_out`) units:
 *
 *   σ = gain / √fan
 *
 * where `fan` is either `fan_in` (default) or `fan_out`.
 *
 * Reference: Delving Deep into Rectifiers (He et al., 2015).
 *
 * @param dtype - The data type of the array. Default: float32
 * @returns An initializer function that optionally accepts `mode` and `gain`.
 */
export function he_normal(
  dtype: any = 'float32'
): (arr: MLXArray, mode?: 'fan_in' | 'fan_out', gain?: number) => MLXArray {
  return function initializer(
    arr: MLXArray,
    mode: 'fan_in' | 'fan_out' = 'fan_in',
    gain: number = 1.0
  ): MLXArray {
    const [fanIn, fanOut] = calculateFanInFanOut(arr);
    if (mode !== 'fan_in' && mode !== 'fan_out') {
      throw new Error(
        `Invalid mode: "${mode}". Valid modes are: "fan_in", "fan_out".`
      );
    }
    const fan = mode === 'fan_in' ? fanIn : fanOut;
    const std = gain / Math.sqrt(fan);
    return random.normal(arr.shape, { dtype, scale: std });
  };
}

/**
 * A He uniform (Kaiming uniform) initializer.
 *
 * Samples from a uniform distribution with a range computed from the
 * number of input (`fan_in`) or output (`fan_out`) units:
 *
 *   limit = gain * √(3 / fan)
 *
 * where `fan` is either `fan_in` (default) or `fan_out`.
 *
 * Reference: Delving Deep into Rectifiers (He et al., 2015).
 *
 * @param dtype - The data type of the array. Default: float32
 * @returns An initializer function that optionally accepts `mode` and `gain`.
 */
export function he_uniform(
  dtype: any = 'float32'
): (arr: MLXArray, mode?: 'fan_in' | 'fan_out', gain?: number) => MLXArray {
  return function initializer(
    arr: MLXArray,
    mode: 'fan_in' | 'fan_out' = 'fan_in',
    gain: number = 1.0
  ): MLXArray {
    const [fanIn, fanOut] = calculateFanInFanOut(arr);
    if (mode !== 'fan_in' && mode !== 'fan_out') {
      throw new Error(
        `Invalid mode: "${mode}". Valid modes are: "fan_in", "fan_out".`
      );
    }
    const fan = mode === 'fan_in' ? fanIn : fanOut;
    const limit = gain * Math.sqrt(3.0 / fan);
    return random.uniform(-limit, limit, arr.shape, { dtype });
  };
}

/**
 * An initializer that returns a sparse matrix.
 *
 * In each row the `ceil(sparsity * cols)` smallest elements (by their
 * position in a uniformly sorted permutation) are zeroed out, mirroring
 * the column-wise zeroing in the Python reference implementation.
 *
 * The Python reference uses indexed assignment
 * (`a[arange(rows).reshape(-1,1), order[:, :num_zeros]] = 0`) which is
 * not yet available in the Node.js bindings.  We replicate the same
 * effect with a binary mask: positions where `argsort(uniform) < num_zeros`
 * become 0; the remaining positions keep their normal-distributed values.
 *
 * @param sparsity - Fraction of elements to zero per row (0 ≤ sparsity ≤ 1).
 * @param mean - Mean of the normal distribution for non-zero values. Default: 0.0
 * @param std - Standard deviation of the normal distribution. Default: 1.0
 * @param dtype - The data type of the array. Default: float32
 * @returns An initializer function that takes a 2D array.
 *
 * @throws {Error} If the input array is not 2D.
 *
 * @example
 * ```typescript
 * const initFn = sparse(0.5);
 * const result = initFn(zeros([4, 4]));
 * // At least 50% of elements per row are zero.
 * ```
 */
export function sparse(
  sparsity: number,
  mean: number = 0.0,
  std: number = 1.0,
  dtype: any = float32
): (arr: MLXArray) => MLXArray {
  return (arr: MLXArray): MLXArray => {
    const shape = arr.shape;
    if (shape.length !== 2) {
      throw new Error(
        `Only tensors with 2 dim are supported but received a ${shape.length}D array.`
      );
    }
    const [rows, cols] = shape;
    const numZeros = Math.ceil(sparsity * cols);

    // Sort order generated from a uniform random matrix (row-wise argsort).
    // sortOrder[i][j] = rank of column j in row i when sorted by a uniform sample.
    const sortOrder = argsort(
      random.uniform(0, 1, [rows, cols]),
      { axis: 1 }
    );

    // keepMask[i][j] = 1 when rank > numZeros (element survives), 0 otherwise.
    // 'less' returns bool (0/1): less(numZeros, sortOrder) ↔ numZeros < sortOrder.
    // Rank == numZeros is the boundary: Python zeros the first numZeros ranked
    // columns, so ranks [0, numZeros) are zeroed. Rank numZeros is kept.
    const keepMask = less(numZeros, sortOrder);

    // Sample values from the normal distribution in the requested dtype.
    const values = random.normal([rows, cols], { dtype, loc: mean, scale: std });

    // Zero out the masked positions by element-wise multiplication.
    return multiply(values, keepMask);
  };
}


/**
 * An initializer that returns an orthogonal matrix.
 *
 * This implementation generates an orthogonal matrix using QR decomposition
 * of a random normal matrix.
 *
 * @param gain - Scaling factor for the orthogonal matrix. Default: 1.0
 * @param dtype - Data type of the array. Default: float32
 * @returns An initializer function that takes a 2D array and returns an orthogonal matrix
 *
 * @throws {Error} If the input array is not 2D
 * @throws {Error} If required MLX operations are not yet available
 *
 * **Current Status**: Placeholder implementation. The full implementation
 * requires MLX operations (random.normal, linalg.qr, diag) to be exposed
 * in the Node.js bindings first.
 */
export function orthogonal(
  gain: number = 1.0,
  dtype: MLXDtype = float32
): Initializer {
  return (a: MLXArray): MLXArray => {
    // Validate input is 2D
    const shape = a.shape;
    if (shape.length !== 2) {
      throw new Error(
        `orthogonal initialization requires a 2D array but got a ${shape.length}D array.`
      );
    }

    throw new Error(
      'orthogonal() is not yet fully implemented. ' +
      'It requires MLX operations (random.normal, linalg.qr, diag) ' +
      'that need to be exposed in the Node.js bindings first.'
    );
  };
}

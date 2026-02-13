/**
 * Neural network initialization functions for MLX.
 *
 * This module provides initializers for neural network parameters,
 * following the patterns from Python's mlx.nn.init module.
 */

import MLXArray, { full } from '../core/array';
import type { MLXDtype, DTypeKey } from '../core/dtype';
import { float32 } from '../core/dtype';
import * as random from '../random';

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
  dtype: DTypeKey = 'float32'
): (arr: MLXArray, gain?: number) => MLXArray {
  return function initializer(arr: MLXArray, gain: number = 1.0): MLXArray {
    const [fanIn, fanOut] = calculateFanInFanOut(arr);
    const limit = gain * Math.sqrt(6.0 / (fanIn + fanOut));
    return random.uniform(-limit, limit, arr.shape, dtype);
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
  dtype: DTypeKey = 'float32'
): (arr: MLXArray, gain?: number) => MLXArray {
  return function initializer(arr: MLXArray, gain: number = 1.0): MLXArray {
    const [fanIn, fanOut] = calculateFanInFanOut(arr);
    const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
    return random.normal(arr.shape, dtype, 0, std);
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
  dtype: DTypeKey = 'float32'
): (arr: MLXArray) => MLXArray {
  return function initializer(arr: MLXArray): MLXArray {
    return random.normal(arr.shape, dtype, mean, std);
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
  dtype: DTypeKey = 'float32'
): (arr: MLXArray) => MLXArray {
  return function initializer(arr: MLXArray): MLXArray {
    return random.uniform(low, high, arr.shape, dtype);
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

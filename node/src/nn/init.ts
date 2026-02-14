import MLXArray from '../core/array';
import { random } from '../core/ops';

/**
 * Type for initializer functions.
 * These functions take a shape array and return initialized values.
 */
export type InitializerFunction = (
  array: MLXArray,
  ...args: any[]
) => MLXArray;

/**
 * Calculate fan_in and fan_out for a given array shape.
 *
 * For weight initialization, fan_in is the number of input units
 * and fan_out is the number of output units.
 *
 * @param x - The array whose shape to analyze
 * @returns A tuple [fan_in, fan_out]
 */
function _calculateFanInFanOut(x: MLXArray): [number, number] {
  const shape = x.shape;

  if (shape.length < 2) {
    throw new Error(
      `Glorot / He initialization requires at least 2 dimensional input ` +
      `but input with ${shape.length} dimensions.`
    );
  }

  let fanIn = shape[shape.length - 1];
  let fanOut = shape[0];

  if (shape.length > 2) {
    let receptiveField = 1;
    for (let i = 1; i < shape.length - 1; i++) {
      receptiveField *= shape[i];
    }
    fanIn *= receptiveField;
    fanOut *= receptiveField;
  }

  return [fanIn, fanOut];
}

/**
 * A He uniform (Kaiming uniform) initializer.
 *
 * This initializer samples from a uniform distribution with a range
 * computed from the number of input (fan_in) or output (fan_out)
 * units according to:
 *
 *     limit = gain * sqrt(3.0 / fan)
 *
 * where `fan` is either the number of input units when the
 * mode is "fan_in" or output units when the mode is "fan_out".
 *
 * For more details see the original reference:
 * "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
 * https://arxiv.org/abs/1502.01852
 *
 * @param dtype - The data type of the array. Default: float32.
 * @returns An initializer function that returns an array with the same shape
 *          as the input, filled with samples from the He uniform distribution.
 *
 * @example
 * ```typescript
 * import { nn, zeros } from 'mlx';
 *
 * // Create initializer
 * const initFn = nn.init.he_uniform();
 *
 * // Initialize a 2x2 weight matrix (uses fan_in by default)
 * const weights = initFn(zeros([2, 2]));
 *
 * // Initialize with fan_out mode and custom gain
 * const weights2 = initFn(zeros([2, 2]), 'fan_out', 5);
 * ```
 */
export function he_uniform(dtype?: any): InitializerFunction {
  return (
    a: MLXArray,
    mode: 'fan_in' | 'fan_out' = 'fan_in',
    gain: number = 1.0
  ): MLXArray => {
    const [fanIn, fanOut] = _calculateFanInFanOut(a);

    let fan: number;
    if (mode === 'fan_in') {
      fan = fanIn;
    } else if (mode === 'fan_out') {
      fan = fanOut;
    } else {
      throw new Error(
        `Invalid mode: ${mode}. Valid modes are: fan_in, fan_out`
      );
    }

    const limit = gain * Math.sqrt(3.0 / fan);
    return random.uniform(-limit, limit, a.shape, { dtype });
  };
}

/**
 * A He normal initializer.
 *
 * This initializer samples from a normal distribution with a standard
 * deviation computed from the number of input (fan_in) or output
 * (fan_out) units according to:
 *
 *     std = gain / sqrt(fan)
 *
 * where `fan` is either the number of input units when the
 * mode is "fan_in" or output units when the mode is "fan_out".
 *
 * @param dtype - The data type of the array. Default: float32.
 * @returns An initializer function
 *
 * @example
 * ```typescript
 * import { nn, zeros } from 'mlx';
 *
 * const initFn = nn.init.he_normal();
 * const weights = initFn(zeros([2, 2]));
 * ```
 */
export function he_normal(dtype?: any): InitializerFunction {
  return (
    a: MLXArray,
    mode: 'fan_in' | 'fan_out' = 'fan_in',
    gain: number = 1.0
  ): MLXArray => {
    const [fanIn, fanOut] = _calculateFanInFanOut(a);

    let fan: number;
    if (mode === 'fan_in') {
      fan = fanIn;
    } else if (mode === 'fan_out') {
      fan = fanOut;
    } else {
      throw new Error(
        `Invalid mode: ${mode}. Valid modes are: fan_in, fan_out`
      );
    }

    const std = gain / Math.sqrt(fan);
    return random.normal(a.shape, { scale: std, dtype });
  };
}

/**
 * A Glorot uniform initializer.
 *
 * This initializer samples from a uniform distribution with a range
 * computed from the number of input (fan_in) and output (fan_out)
 * units according to:
 *
 *     limit = gain * sqrt(6.0 / (fan_in + fan_out))
 *
 * @param dtype - The data type of the array. Default: float32.
 * @returns An initializer function
 *
 * @example
 * ```typescript
 * import { nn, zeros } from 'mlx';
 *
 * const initFn = nn.init.glorot_uniform();
 * const weights = initFn(zeros([2, 2]));
 * ```
 */
export function glorot_uniform(dtype?: any): InitializerFunction {
  return (
    a: MLXArray,
    gain: number = 1.0
  ): MLXArray => {
    const [fanIn, fanOut] = _calculateFanInFanOut(a);
    const limit = gain * Math.sqrt(6.0 / (fanIn + fanOut));
    return random.uniform(-limit, limit, a.shape, { dtype });
  };
}

/**
 * A Glorot normal initializer.
 *
 * This initializer samples from a normal distribution with a standard
 * deviation computed from the number of input (fan_in) and output
 * (fan_out) units according to:
 *
 *     std = gain * sqrt(2.0 / (fan_in + fan_out))
 *
 * @param dtype - The data type of the array. Default: float32.
 * @returns An initializer function
 *
 * @example
 * ```typescript
 * import { nn, zeros } from 'mlx';
 *
 * const initFn = nn.init.glorot_normal();
 * const weights = initFn(zeros([2, 2]));
 * ```
 */
export function glorot_normal(dtype?: any): InitializerFunction {
  return (
    a: MLXArray,
    gain: number = 1.0
  ): MLXArray => {
    const [fanIn, fanOut] = _calculateFanInFanOut(a);
    const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
    return random.normal(a.shape, { scale: std, dtype });
  };
}

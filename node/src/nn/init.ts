import MLXArray, { full } from '../core/array';
import {
  random,
  linalg,
  where,
  greater,
  multiply,
  sign,
  diag,
  slice,
} from '../core/ops';

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

/**
 * An initializer that fills the array with a constant value.
 *
 * @param value - The constant value to fill the array with.
 * @param dtype - The data type of the array.
 * @returns An initializer function.
 */
export function constant(value: number, dtype?: any): InitializerFunction {
  return (a: MLXArray): MLXArray => {
    return full(a.shape, value, dtype);
  };
}

/**
 * An initializer that returns an orthogonal matrix.
 *
 * @param gain - Scaling factor for the orthogonal matrix.
 * @param dtype - The data type of the array.
 * @returns An initializer function.
 */
export function orthogonal(gain = 1.0, dtype?: any): InitializerFunction {
  return (a: MLXArray): MLXArray => {
    if (a.ndim !== 2) {
      throw new Error(
        `Orthogonal initialization requires a 2D array but got a ${a.ndim}D array.`
      );
    }

    const [rows, cols] = a.shape;
    const n = Math.max(rows, cols);

    const rmat = random.normal([n, n], { dtype });

    // Perform QR decomposition
    const [q, r] = linalg.qr(rmat);

    // Adjust the sign of Q using the diagonal of R
    const d = diag(r);
    let adjustedQ = multiply(q, sign(d));

    // Slice Q to the desired shape
    adjustedQ = slice(adjustedQ, [0, 0], [rows, cols]);

    // Scale Q by gain
    return multiply(adjustedQ, gain);
  };
}

/**
 * An initializer that returns a sparse matrix.
 *
 * @param sparsity - The fraction of elements in each column to be set to zero.
 * @param std - Standard deviation of the normal distribution.
 * @param dtype - The data type of the array.
 * @returns An initializer function.
 */
export function sparse(
  sparsity: number,
  std = 0.01,
  dtype?: any
): InitializerFunction {
  return (a: MLXArray): MLXArray => {
    if (a.ndim !== 2) {
      throw new Error(
        `Sparse initialization requires a 2D array but got a ${a.ndim}D array.`
      );
    }

    const cond = greater(random.uniform(0, 1, a.shape), sparsity);
    const values = random.normal(a.shape, { scale: std, dtype: dtype });

    return where(cond, values, full(a.shape, 0, dtype));
  };
}

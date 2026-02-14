import MLXArray from '../core/array';
import type { MLXDtype } from '../core/dtype';
import { random } from '../core/ops';

/**
 * Calculate fan_in and fan_out for an array.
 * 
 * Based on the Python implementation in mlx.nn.init._calculate_fan_in_fan_out
 * 
 * @param a - The array to calculate fan values for
 * @returns Tuple of [fan_in, fan_out]
 */
function calculateFanInFanOut(a: MLXArray): [number, number] {
  const shape = a.shape;
  
  if (shape.length < 1) {
    throw new Error('Array must have at least 1 dimension');
  }
  
  if (shape.length === 1) {
    return [shape[0], shape[0]];
  }
  
  let fanIn = shape[shape.length - 1];
  let fanOut = shape[0];
  
  if (shape.length > 2) {
    // For conv layers, multiply by receptive field size
    let receptiveField = 1;
    for (let i = 1; i < shape.length - 1; i++) {
      receptiveField *= shape[i];
    }
    fanIn *= receptiveField;
    fanOut *= receptiveField;
  }
  
  return [fanIn, fanOut];
}

export type FanMode = 'fan_in' | 'fan_out';

export interface HeNormalInitializer {
  /**
   * Initialize an array with He normal distribution.
   * 
   * @param a - The array whose shape determines the output shape
   * @param mode - Either 'fan_in' or 'fan_out'. Default: 'fan_in'
   * @param gain - Scaling factor for the standard deviation. Default: 1.0
   * @returns Array filled with samples from He normal distribution
   */
  (a: MLXArray, mode?: FanMode, gain?: number): MLXArray;
}

/**
 * Build a He normal initializer.
 *
 * This initializer samples from a normal distribution with a standard
 * deviation computed from the number of input (`fan_in`) or output
 * (`fan_out`) units according to:
 *
 * ```
 * σ = gain / sqrt(fan)
 * ```
 *
 * where `fan` is either the number of input units when the
 * `mode` is `"fan_in"` or output units when the `mode` is
 * `"fan_out"`.
 *
 * For more details see the original reference: [Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
 *
 * @param dtype - The data type of the array. Default: float32
 * @returns An initializer function that returns an array with the same shape as the input,
 *          filled with samples from the He normal distribution
 *
 * @example
 * ```typescript
 * import * as mx from 'mlx';
 *
 * // Create a He normal initializer
 * const initFn = mx.nn.heNormal();
 *
 * // Initialize a weight matrix (fan_in mode by default)
 * const weights = mx.core.zeros([64, 128]); // 128 input features, 64 output features
 * const initializedWeights = initFn(weights);
 *
 * // Use fan_out mode with custom gain
 * const weights2 = mx.core.zeros([64, 128]);
 * const initializedWeights2 = initFn(weights2, 'fan_out', 2.0);
 * ```
 */
export function heNormal(dtype: MLXDtype = { key: 'float32' } as MLXDtype): HeNormalInitializer {
  return (a: MLXArray, mode: FanMode = 'fan_in', gain: number = 1.0): MLXArray => {
    const [fanIn, fanOut] = calculateFanInFanOut(a);
    
    let fan: number;
    if (mode === 'fan_in') {
      fan = fanIn;
    } else if (mode === 'fan_out') {
      fan = fanOut;
    } else {
      throw new Error(`Invalid mode: ${mode}. Valid modes are: fan_in, fan_out`);
    }
    
    const std = gain / Math.sqrt(fan);
    return random.normal(a.shape, { scale: std, dtype });
  };
}

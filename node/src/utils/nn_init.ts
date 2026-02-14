/**
 * Neural network weight initialization utilities.
 * 
 * Provides various weight initialization strategies for neural networks,
 * including Glorot (Xavier) initialization.
 */

import { type MLXArray } from '../core/array';
import { random, type NormalOptions } from '../core/ops';
import { float32, type DTypeLike } from '../core/dtype';

/**
 * Calculate the fan-in and fan-out for a weight tensor.
 * 
 * For a 2D tensor (weight matrix): fan_in = input_size, fan_out = output_size
 * For higher dimensional tensors (e.g., conv kernels): accounts for receptive field
 * 
 * @param arr - The array to calculate fan-in/fan-out for
 * @returns Tuple of [fan_in, fan_out]
 */
function calculateFanInFanOut(arr: MLXArray): [number, number] {
  if (arr.ndim < 2) {
    throw new Error(
      `Glorot / He initialization requires at least 2 dimensional input ` +
      `but input with ${arr.ndim} dimensions.`
    );
  }

  let fanIn = arr.shape[arr.shape.length - 1];
  let fanOut = arr.shape[0];

  if (arr.ndim > 2) {
    let receptiveField = 1;
    // Multiply dimensions between first and last (e.g., for conv kernels)
    for (let i = 1; i < arr.shape.length - 1; i++) {
      receptiveField *= arr.shape[i];
    }
    fanIn = fanIn * receptiveField;
    fanOut = fanOut * receptiveField;
  }

  return [fanIn, fanOut];
}

/**
 * Options for Glorot normal initializer
 */
export interface GlorotNormalOptions {
  /**
   * Data type of the output array. Default: float32
   */
  dtype?: DTypeLike;
}

/**
 * Initializer function type that takes an array and optional gain parameter
 */
export type Initializer = (arr: MLXArray, gain?: number) => MLXArray;

/**
 * A Glorot normal initializer (also known as Xavier normal initialization).
 * 
 * This initializer samples from a normal distribution with a standard
 * deviation computed from the number of input (fan_in) and output
 * (fan_out) units according to:
 * 
 * σ = gain * sqrt(2.0 / (fan_in + fan_out))
 * 
 * For more details see the original reference: 
 * "Understanding the difficulty of training deep feedforward neural networks"
 * by Glorot and Bengio (2010)
 * https://proceedings.mlr.press/v9/glorot10a.html
 * 
 * @param options - Configuration options including dtype
 * @returns An initializer function that takes an array and optional gain parameter
 * 
 * @example
 * ```typescript
 * import * as mx from 'mlx';
 * import { glorotNormal } from 'mlx/utils';
 * 
 * // Create initializer
 * const initFn = glorotNormal();
 * 
 * // Initialize a weight matrix
 * const weights = mx.zeros([100, 50]);
 * const initialized = initFn(weights);
 * 
 * // With custom gain
 * const initializedWithGain = initFn(weights, 2.0);
 * ```
 */
export function glorotNormal(options?: GlorotNormalOptions): Initializer {
  const dtype = options?.dtype ?? float32;
  
  return (arr: MLXArray, gain: number = 1.0): MLXArray => {
    const [fanIn, fanOut] = calculateFanInFanOut(arr);
    const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
    
    const normalOpts: NormalOptions = {
      dtype,
      scale: std,
    };
    
    return random.normal(arr.shape, normalOpts);
  };
}

import MLXArray, { full } from '../core/array';
import type { DTypeLike } from '../core/dtype';
import { float32, type MLXDtype } from '../core/dtype';

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

import addon from '../internal/addon';
import MLXArray from '../core/array';
import type { DTypeKey } from '../core/dtype';
import type { StreamLike } from '../core/stream';

/**
 * Generate uniformly distributed random numbers.
 * 
 * @param low - Lower bound (inclusive) or shape if only one array argument
 * @param high - Upper bound (exclusive) 
 * @param shape - Shape of the output array
 * @param dtype - Data type of the output (default: float32)
 * @param stream - Stream to use for the operation
 * @returns Array of random numbers
 */
export function uniform(
  low: number | number[],
  high?: number | number[],
  shape?: number[],
  dtype?: DTypeKey,
  stream?: StreamLike
): MLXArray;

export function uniform(
  shape: number[],
  dtype?: DTypeKey,
  stream?: StreamLike
): MLXArray;

export function uniform(...args: any[]): MLXArray {
  // If first arg is an array, it's the shape-only variant
  if (Array.isArray(args[0]) && (args.length === 1 || typeof args[1] === 'string' || args[1] === undefined)) {
    const [shape, dtype, stream] = args;
    const nativeArgs: any[] = [shape];
    if (dtype !== undefined) nativeArgs.push(dtype);
    if (stream !== undefined) nativeArgs.push(stream);
    return addon.core.random.uniform(...nativeArgs);
  }
  
  // Otherwise it's the (low, high, shape) variant
  return addon.core.random.uniform(...args);
}

/**
 * Generate normally distributed random numbers.
 * 
 * @param shape - Shape of the output array
 * @param dtype - Data type of the output (default: float32)
 * @param loc - Mean of the distribution (default: 0)
 * @param scale - Standard deviation of the distribution (default: 1)
 * @param stream - Stream to use for the operation
 * @returns Array of random numbers
 */
export function normal(
  shape: number[],
  dtype?: DTypeKey,
  loc?: number,
  scale?: number,
  stream?: StreamLike
): MLXArray {
  const args: any[] = [shape];
  if (dtype !== undefined) args.push(dtype);
  if (loc !== undefined) args.push(loc);
  if (scale !== undefined) args.push(scale);
  if (stream !== undefined) args.push(stream);
  
  return addon.core.random.normal(...args);
}

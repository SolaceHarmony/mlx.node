import addon from '../internal/addon';
import MLXArray, { normalizeShapeInput } from './array';
import type { StreamLike } from './stream';
import { toNativeStreamArgument } from './stream';
import type { DTypeLike } from './dtype';

function toNativeHandle(tensor: MLXArray): any {
  return tensor.toNative();
}

function normalizeStream(stream?: StreamLike | null): any {
  if (stream == null) {
    return undefined;
  }
  return toNativeStreamArgument(stream);
}

function normalizeAxes(axes: readonly number[]): number[] {
  return Array.from(axes, (axis) => {
    if (!Number.isInteger(axis)) {
      throw new Error('Axis indices must be integers');
    }
    return Number(axis);
  });
}

function normalizeAxisSpec(value: number | readonly number[], name: string): number[] {
  if (Array.isArray(value)) {
    return normalizeAxes(value);
  }
  if (!Number.isInteger(value)) {
    throw new Error(`${name} must be an integer or array of integers`);
  }
  return [Number(value)];
}

function appendStreamArg(args: any[], stream?: StreamLike | null): void {
  const native = normalizeStream(stream);
  if (native !== undefined) {
    args.push(native);
  }
}

export interface StreamOptions {
  stream?: StreamLike;
}

export interface ReshapeOptions extends StreamOptions {}

export function reshape(
  tensor: MLXArray,
  shape: readonly number[],
  options?: ReshapeOptions,
): MLXArray {
  const normalizedShape = normalizeShapeInput(shape);
  const args: any[] = [toNativeHandle(tensor), normalizedShape];
  appendStreamArg(args, options?.stream);
  const handle = addon.reshape(...args);
  return MLXArray.fromHandle(handle);
}

export interface TransposeOptions extends StreamOptions {}

export function transpose(
  tensor: MLXArray,
  axesOrOptions?: readonly number[] | TransposeOptions | null,
  maybeOptions?: TransposeOptions,
): MLXArray {
  let axes: number[] | undefined;
  let options: TransposeOptions | undefined;

  if (Array.isArray(axesOrOptions)) {
    axes = normalizeAxes(axesOrOptions);
    options = maybeOptions ?? undefined;
  } else if (axesOrOptions && 'stream' in axesOrOptions) {
    options = axesOrOptions;
  } else if (axesOrOptions != null) {
    throw new Error('transpose axes must be an array of integers');
  } else {
    options = maybeOptions;
  }

  const args: any[] = [toNativeHandle(tensor)];
  if (axes) {
    args.push(axes);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.transpose(...args);
  return MLXArray.fromHandle(handle);
}

export interface MoveAxisOptions extends StreamOptions {}

export function moveaxis(
  tensor: MLXArray,
  source: number | readonly number[],
  destination: number | readonly number[],
  options?: MoveAxisOptions,
): MLXArray {
  const src = normalizeAxisSpec(source, 'source axes');
  const dst = normalizeAxisSpec(destination, 'destination axes');
  const args: any[] = [toNativeHandle(tensor), src, dst];
  appendStreamArg(args, options?.stream);
  const handle = addon.moveaxis(...args);
  return MLXArray.fromHandle(handle);
}

export interface SwapAxesOptions extends StreamOptions {}

export function swapaxes(
  tensor: MLXArray,
  axis1: number,
  axis2: number,
  options?: SwapAxesOptions,
): MLXArray {
  if (!Number.isInteger(axis1) || !Number.isInteger(axis2)) {
    throw new Error('swapaxes expects integer axis indices');
  }
  const args: any[] = [
    toNativeHandle(tensor),
    Number(axis1),
    Number(axis2),
  ];
  appendStreamArg(args, options?.stream);
  const handle = addon.swapaxes(...args);
  return MLXArray.fromHandle(handle);
}

export interface BinaryOpOptions extends StreamOptions {}

type ScalarOrArray = MLXArray | number | boolean | bigint;

function toNativeScalarOrArray(value: ScalarOrArray): any {
  if (value instanceof MLXArray) {
    return toNativeHandle(value);
  }
  return value;
}

function binaryOp(
  name: 'add' | 'multiply' | 'subtract',
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon[name](...args);
  return MLXArray.fromHandle(handle);
}

export function add(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  return binaryOp('add', a, b, options);
}

export function multiply(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  return binaryOp('multiply', a, b, options);
}

export function subtract(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  return binaryOp('subtract', a, b, options);
}

export interface WhereOptions extends StreamOptions {}

export function where(
  condition: ScalarOrArray,
  onTrue: ScalarOrArray,
  onFalse: ScalarOrArray,
  options?: WhereOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(condition),
    toNativeScalarOrArray(onTrue),
    toNativeScalarOrArray(onFalse),
  ];
  appendStreamArg(args, options?.stream);
  const handle = addon.where(...args);
  return MLXArray.fromHandle(handle);
}

export interface UnaryOpOptions extends StreamOptions {}

export function tan(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.tan(...args);
  return MLXArray.fromHandle(handle);
}

export function sin(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.sin(...args);
  return MLXArray.fromHandle(handle);
}

export function cos(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.cos(...args);
  return MLXArray.fromHandle(handle);
}

export function arcsin(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.arcsin(...args);
  return MLXArray.fromHandle(handle);
}

export function arccos(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.arccos(...args);
  return MLXArray.fromHandle(handle);
}

export function arctan(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.arctan(...args);
  return MLXArray.fromHandle(handle);
}

export function arctan2(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.arctan2(...args);
  return MLXArray.fromHandle(handle);
}

export function rsqrt(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.rsqrt(...args);
  return MLXArray.fromHandle(handle);
}

export function square(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.square(...args);
  return MLXArray.fromHandle(handle);
}

export function sign(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.sign(...args);
  return MLXArray.fromHandle(handle);
}

export function abs(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.abs(...args);
  return MLXArray.fromHandle(handle);
}

export function sqrt(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.sqrt(...args);
  return MLXArray.fromHandle(handle);
}

export function exp(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.exp(...args);
  return MLXArray.fromHandle(handle);
}

export function log(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.log(...args);
  return MLXArray.fromHandle(handle);
}

export function divide(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.divide(...args);
  return MLXArray.fromHandle(handle);
}

export function power(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.power(...args);
  return MLXArray.fromHandle(handle);
}

export function equal(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.equal(...args);
  return MLXArray.fromHandle(handle);
}

export function not_equal(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.not_equal(...args);
  return MLXArray.fromHandle(handle);
}

export function less(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.less(...args);
  return MLXArray.fromHandle(handle);
}

export function less_equal(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.less_equal(...args);
  return MLXArray.fromHandle(handle);
}

export function greater(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.greater(...args);
  return MLXArray.fromHandle(handle);
}

export function greater_equal(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.greater_equal(...args);
  return MLXArray.fromHandle(handle);
}

export function maximum(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.maximum(...args);
  return MLXArray.fromHandle(handle);
}

export function minimum(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.minimum(...args);
  return MLXArray.fromHandle(handle);
}

export interface ArangeOptions extends StreamOptions {
  dtype?: any;
}

/**
 * Generate evenly spaced values within a given interval.
 * 
 * @param start - The start of the interval (inclusive). If only one argument is provided, this argument is treated as `stop` and the interval starts at 0.
 * @param stop - The end of the interval (exclusive). Required if both `start` and `stop` are specified.
 * @param step - The spacing between values. Default is 1.
 * @param options - Optional parameters, including `dtype` and `stream`.
 * @returns An MLXArray of evenly spaced values.
 * 
 * @example
 * // Generate 0 to 9
 * arange(10)
 * 
 * @example
 * // Generate 5 to 9
 * arange(5, 10)
 * 
 * @example
 * // Generate 0 to 9 with step 2
 * arange(0, 10, 2)
 * 
 * @example
 * // Generate 0.0 to 9.5 with step 0.5
 * arange(0, 10, 0.5)
 */
export function arange(
  start: number,
  stop?: number,
  step?: number,
  options?: ArangeOptions,
): MLXArray {
  const args: any[] = [];
  
  // Handle overloaded signatures
  if (stop === undefined) {
    // Single argument: arange(stop)
    args.push(start);
  } else {
    // Two or three arguments: arange(start, stop[, step])
    args.push(start, stop);
    if (step !== undefined) {
      args.push(step);
    }
  }
  
  // Add optional dtype
  if (options?.dtype !== undefined) {
    args.push(options.dtype);
  }
  
  // Add optional stream
  appendStreamArg(args, options?.stream);
  
  const handle = addon.arange(...args);
  return MLXArray.fromHandle(handle);
}

/**
 * Matrix multiplication.
 * 
 * Performs matrix multiplication between two arrays.
 * 
 * @param a - The first input array
 * @param b - The second input array
 * @param options - Optional stream configuration
 * @returns The result of the matrix multiplication
 * 
 * @example
 * ```typescript
 * const a = mx.array([[1, 2], [3, 4]]);
 * const b = mx.array([[5, 6], [7, 8]]);
 * const result = mx.matmul(a, b);
 * ```
 */
export function matmul(
  a: MLXArray,
  b: MLXArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.matmul(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Fused linear algebra ops
// ---------------------------------------------------------------------------

export interface AddmmOptions extends StreamOptions {}

/**
 * Compute alpha * (a @ b) + beta * c.
 *
 * Fused add-multiply-multiply: useful as the forward pass of a linear layer.
 *
 * @param c - Bias array
 * @param a - First input matrix
 * @param b - Second input matrix
 * @param alpha - Scalar multiplier for a @ b (default 1.0)
 * @param beta - Scalar multiplier for c (default 1.0)
 * @param options - Optional stream configuration
 * @returns The result of alpha * (a @ b) + beta * c
 */
export function addmm(
  c: MLXArray,
  a: MLXArray,
  b: MLXArray,
  alpha?: number,
  beta?: number,
  options?: AddmmOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(c), toNativeHandle(a), toNativeHandle(b)];
  if (alpha !== undefined) args.push(alpha);
  if (beta !== undefined) args.push(beta);
  appendStreamArg(args, options?.stream);
  const handle = addon.addmm(...args);
  return MLXArray.fromHandle(handle);
}

export interface NormalOptions extends StreamOptions {
  dtype?: DTypeLike;
  loc?: number | MLXArray;
  scale?: number | MLXArray;
  key?: MLXArray;
}

/**
 * Random number generation functions (mlx.core.random).
 */
export namespace random {
  /**
   * Generate normally distributed random numbers.
   *
   * Generates samples from a normal (Gaussian) distribution with specified
   * mean and standard deviation. If `loc` and `scale` are not provided,
   * generates from the standard normal distribution (mean=0, std=1).
   *
   * @param shape - Shape of the output array
   * @param options - Optional configuration (dtype, loc, scale, key, stream)
   * @returns Array of normally distributed random numbers
   *
   * @example
   * ```typescript
   * const x = mx.random.normal([2, 3]);
   * const y = mx.random.normal([2, 3], { loc: 5, scale: 2 });
   * ```
   */
  export function normal(
    shape: readonly number[],
    options?: NormalOptions,
  ): MLXArray {
    const normalizedShape = normalizeShapeInput(shape);
    const args: any[] = [normalizedShape];

    if (options?.dtype !== undefined) {
      args.push(options.dtype);
    }

    if (options?.loc !== undefined) {
      if (options.loc instanceof MLXArray) {
        args.push(toNativeHandle(options.loc));
      } else {
        args.push(options.loc);
      }
    }

    if (options?.scale !== undefined) {
      if (options.scale instanceof MLXArray) {
        args.push(toNativeHandle(options.scale));
      } else {
        args.push(options.scale);
      }
    }

    if (options?.key !== undefined) {
      args.push(toNativeHandle(options.key));
    }

    appendStreamArg(args, options?.stream);

    const handle = addon.random.normal(...args);
    return MLXArray.fromHandle(handle);
  }

  export interface BernoulliOptions extends StreamOptions {
    key?: MLXArray;
  }

  /**
   * Generate Bernoulli random samples.
   *
   * @param p - Probability of a 1 (default 0.5). Can be a number or MLXArray.
   * @param shape - Optional shape of the output array. If not provided, uses p.shape.
   * @param options - Optional configuration (key, stream)
   * @returns Array of boolean Bernoulli samples
   */
  export function bernoulli(
    p?: number | MLXArray,
    shape?: readonly number[],
    options?: BernoulliOptions,
  ): MLXArray {
    const args: any[] = [];
    if (p !== undefined) {
      if (p instanceof MLXArray) {
        args.push(toNativeHandle(p));
      } else {
        args.push(p);
      }
    }
    if (shape !== undefined) {
      args.push(normalizeShapeInput(shape));
    }
    if (options?.key !== undefined) {
      args.push(toNativeHandle(options.key));
    }
    appendStreamArg(args, options?.stream);
    const handle = addon.random.bernoulli(...args);
    return MLXArray.fromHandle(handle);
  }

  export interface UniformOptions extends StreamOptions {
    dtype?: any;
  }

  /**
   * Generate uniform random numbers.
   *
   * Generates random numbers from a uniform distribution. Can be called in two ways:
   * 1. uniform(shape, options?) - generates random numbers in [0, 1)
   * 2. uniform(low, high, shape, options?) - generates random numbers in [low, high)
   *
   * @param lowOrShape - Either the lower bound (if high and shape provided) or the shape
   * @param highOrOptions - Either the upper bound (if low and shape mode) or options
   * @param shapeOrOptions - Either the shape (if low and high provided) or options
   * @param maybeOptions - Options (if using low, high, shape signature)
   * @returns An MLXArray of uniform random numbers
   *
   * @example
   * ```typescript
   * const a = mx.random.uniform([2, 3]);
   * const b = mx.random.uniform(-1, 1, [2, 3]);
   * ```
   */
  export function uniform(
    lowOrShape: number | readonly number[],
    highOrOptions?: number | UniformOptions,
    shapeOrOptions?: readonly number[] | UniformOptions,
    maybeOptions?: UniformOptions,
  ): MLXArray {
    let args: any[];
    let options: UniformOptions | undefined;

    if (typeof lowOrShape === 'number' && typeof highOrOptions === 'number') {
      const low = lowOrShape;
      const high = highOrOptions;
      const shape = normalizeShapeInput(shapeOrOptions as readonly number[]);
      options = maybeOptions;
      args = [low, high, shape];
    } else {
      let shape: readonly number[];
      if (typeof lowOrShape === 'number') {
        shape = normalizeShapeInput([lowOrShape]);
      } else {
        shape = normalizeShapeInput(lowOrShape);
      }
      options = highOrOptions as UniformOptions | undefined;
      args = [shape];
    }

    if (options?.dtype !== undefined) {
      args.push(options.dtype);
    }

    appendStreamArg(args, options?.stream);

    const handle = addon.random.uniform(...args);
    return MLXArray.fromHandle(handle);
  }

  export function seed(s: number): void {
    addon.random.seed(s);
  }

  export function key(s: number): MLXArray {
    return MLXArray.fromHandle(addon.random.key(s));
  }

  export interface SplitOptions extends StreamOptions {}

  export function split(k: MLXArray, num?: number, options?: SplitOptions): MLXArray | [MLXArray, MLXArray] {
    const args: any[] = [toNativeHandle(k)];
    if (num !== undefined) args.push(num);
    appendStreamArg(args, options?.stream);
    const result = addon.random.split(...args);
    if (Array.isArray(result)) {
      return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1])];
    }
    return MLXArray.fromHandle(result);
  }

  export interface RandintOptions extends StreamOptions {
    dtype?: DTypeLike;
    key?: MLXArray;
  }

  export function randint(low: number | MLXArray, high: number | MLXArray, shape: readonly number[], options?: RandintOptions): MLXArray {
    const args: any[] = [
      low instanceof MLXArray ? toNativeHandle(low) : low,
      high instanceof MLXArray ? toNativeHandle(high) : high,
      normalizeShapeInput(shape),
    ];
    if (options?.dtype !== undefined) args.push(options.dtype);
    if (options?.key !== undefined) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.randint(...args));
  }

  export interface CategoricalOptions extends StreamOptions {
    axis?: number;
    key?: MLXArray;
  }

  export function categorical(logits: MLXArray, options?: CategoricalOptions): MLXArray {
    const args: any[] = [toNativeHandle(logits)];
    if (options?.axis !== undefined) args.push(options.axis);
    if (options?.key !== undefined) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.categorical(...args));
  }

  export interface PermutationOptions extends StreamOptions {
    axis?: number;
    key?: MLXArray;
  }

  export function permutation(x: number | MLXArray, options?: PermutationOptions): MLXArray {
    const args: any[] = [x instanceof MLXArray ? toNativeHandle(x) : x];
    if (x instanceof MLXArray && options?.axis !== undefined) args.push(options.axis);
    if (options?.key !== undefined) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.permutation(...args));
  }

  export interface GumbelOptions extends StreamOptions {
    dtype?: DTypeLike;
    key?: MLXArray;
  }

  export function gumbel(shape: readonly number[], options?: GumbelOptions): MLXArray {
    const args: any[] = [normalizeShapeInput(shape)];
    if (options?.dtype !== undefined) args.push(options.dtype);
    if (options?.key !== undefined) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.gumbel(...args));
  }

  export interface LaplaceOptions extends StreamOptions {
    dtype?: DTypeLike;
    key?: MLXArray;
  }

  export function laplace(shape: readonly number[], options?: LaplaceOptions): MLXArray {
    const args: any[] = [normalizeShapeInput(shape)];
    if (options?.dtype !== undefined) args.push(options.dtype);
    if (options?.key !== undefined) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.laplace(...args));
  }

  export interface TruncatedNormalOptions extends StreamOptions {
    shape?: readonly number[];
    dtype?: DTypeLike;
    key?: MLXArray;
  }

  export function truncated_normal(lower: number | MLXArray, upper: number | MLXArray, options?: TruncatedNormalOptions): MLXArray {
    const args: any[] = [
      lower instanceof MLXArray ? toNativeHandle(lower) : lower,
      upper instanceof MLXArray ? toNativeHandle(upper) : upper,
    ];
    if (options?.shape !== undefined) args.push(normalizeShapeInput(options.shape));
    if (options?.dtype !== undefined) args.push(options.dtype);
    if (options?.key !== undefined) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.truncated_normal(...args));
  }

  export interface MultivariateNormalOptions extends StreamOptions {
    dtype?: DTypeLike;
    key?: MLXArray;
  }

  export function multivariate_normal(mean: MLXArray, cov: MLXArray, shape: readonly number[], options?: MultivariateNormalOptions): MLXArray {
    const args: any[] = [toNativeHandle(mean), toNativeHandle(cov), normalizeShapeInput(shape)];
    if (options?.dtype !== undefined) args.push(options.dtype);
    if (options?.key !== undefined) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.multivariate_normal(...args));
  }
}

/**
 * Import a function from a .mlxfn file.
 * 
 * Returns a callable function that can be invoked with MLX arrays.
 * The imported function accepts:
 * - Positional array arguments: fn(a, b, c)
 * - A single array of arrays: fn([a, b, c])
 * - A single object/dict of arrays: fn({x: a, y: b})
 * - Combined: fn([a, b], {x: c, y: d})
 * 
 * The returned function always returns an array of output MLX arrays.
 * 
 * @param file - Path to the .mlxfn file
 * @returns A callable function wrapping the imported MLX function
 * 
 * @example
 * ```typescript
 * import { import_function, array } from 'mlx';
 * 
 * // Import a previously exported function
 * const fn = import_function('function.mlxfn');
 * 
 * // Call with positional arguments
 * const input = array([1, 2, 3]);
 * const [output] = fn(input);
 * 
 * // Call with named arguments
 * const [result] = fn({ x: input });
 * ```
 */
export function import_function(file: string): (...args: any[]) => MLXArray[] {
  if (typeof file !== 'string') {
    throw new TypeError('import_function expects a string file path');
  }
  
  const nativeFunction = addon.import_function(file);
  
  return (...args: any[]): MLXArray[] => {
    // Convert MLXArray inputs to native handles
    const nativeArgs = args.map(arg => {
      if (arg instanceof MLXArray) {
        return arg.toNative();
      } else if (Array.isArray(arg)) {
        return arg.map(item => item instanceof MLXArray ? item.toNative() : item);
      } else if (arg && typeof arg === 'object') {
        const converted: Record<string, any> = {};
        for (const [key, value] of Object.entries(arg)) {
          converted[key] = value instanceof MLXArray ? value.toNative() : value;
        }
        return converted;
      }
      return arg;
    });
    
    // Call the native function
    const resultHandles = nativeFunction(...nativeArgs);
    
    // Convert result handles back to MLXArray objects
    // The C++ implementation always returns an array of results
    if (!Array.isArray(resultHandles)) {
      throw new Error('Internal error: import_function expected array result from native function');
    }
    
    return resultHandles.map(handle => MLXArray.fromHandle(handle));
  };
}

// ---------------------------------------------------------------------------
// Reduction ops
// ---------------------------------------------------------------------------

export type AxisSpec = number | readonly number[] | null;

export interface ReductionOptions extends StreamOptions {
  keepdims?: boolean;
}

function normalizeAxisArg(axis?: AxisSpec): number[] | undefined {
  if (axis === null || axis === undefined) {
    return undefined;
  }
  if (typeof axis === 'number') {
    return [axis];
  }
  return Array.from(axis);
}

export function sum(
  a: MLXArray,
  axis?: AxisSpec,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  args.push(nativeAxis ?? null);
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.sum(...args);
  return MLXArray.fromHandle(handle);
}

export function mean(
  a: MLXArray,
  axis?: AxisSpec,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  args.push(nativeAxis ?? null);
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.mean(...args);
  return MLXArray.fromHandle(handle);
}

export interface VarOptions extends ReductionOptions {
  ddof?: number;
}

/**
 * Compute the variance along the given axes.
 *
 * @param a - Input array
 * @param axis - Optional axis or axes along which to compute variance
 * @param options - Optional configuration (keepdims, ddof, stream)
 * @returns The variance array
 */
export function variance(
  a: MLXArray,
  axis?: AxisSpec,
  options?: VarOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  if (nativeAxis !== undefined) {
    args.push(nativeAxis);
  }
  if (options?.keepdims) {
    args.push(true);
  } else if (options?.ddof !== undefined) {
    args.push(false); // must push keepdims before ddof
  }
  if (options?.ddof !== undefined) {
    args.push(options.ddof);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.var(...args);
  return MLXArray.fromHandle(handle);
}

export function logsumexp(
  a: MLXArray,
  axis?: AxisSpec,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  args.push(nativeAxis ?? null);
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.logsumexp(...args);
  return MLXArray.fromHandle(handle);
}

/**
 * A min reduction over the given axes.
 *
 * @param a - Input array
 * @param axis - Optional axis or axes along which to compute min
 * @param options - Optional configuration (keepdims, stream)
 * @returns The min array
 */
export function min(
  a: MLXArray,
  axis?: AxisSpec,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  args.push(nativeAxis ?? null);
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.min(...args);
  return MLXArray.fromHandle(handle);
}

/**
 * A max reduction over the given axes.
 *
 * @param a - Input array
 * @param axis - Optional axis or axes along which to compute max
 * @param options - Optional configuration (keepdims, stream)
 * @returns The max array
 */
export function max(
  a: MLXArray,
  axis?: AxisSpec,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  args.push(nativeAxis ?? null);
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.max(...args);
  return MLXArray.fromHandle(handle);
}

/**
 * A product reduction over the given axes.
 *
 * @param a - Input array
 * @param axis - Optional axis or axes along which to compute product
 * @param options - Optional configuration (keepdims, stream)
 * @returns The product array
 */
export function prod(
  a: MLXArray,
  axis?: AxisSpec,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  args.push(nativeAxis ?? null);
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.prod(...args);
  return MLXArray.fromHandle(handle);
}

/**
 * Indices of the minimum values along the axis.
 *
 * @param a - Input array
 * @param axis - Optional axis along which to find argmin. If omitted, argmin over flattened array.
 * @param options - Optional configuration (keepdims, stream)
 * @returns The argmin indices array
 */
export function argmin(
  a: MLXArray,
  axis?: number | null,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (axis !== undefined && axis !== null) {
    args.push(axis);
  }
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.argmin(...args);
  return MLXArray.fromHandle(handle);
}

/**
 * Indices of the maximum values along the axis.
 *
 * @param a - Input array
 * @param axis - Optional axis along which to find argmax. If omitted, argmax over flattened array.
 * @param options - Optional configuration (keepdims, stream)
 * @returns The argmax indices array
 */
export function argmax(
  a: MLXArray,
  axis?: number | null,
  options?: ReductionOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (axis !== undefined && axis !== null) {
    args.push(axis);
  }
  if (options?.keepdims) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.argmax(...args);
  return MLXArray.fromHandle(handle);
}

export interface StdOptions extends ReductionOptions {
  ddof?: number;
}

/**
 * Compute the standard deviation along the given axes.
 *
 * @param a - Input array
 * @param axis - Optional axis or axes along which to compute std
 * @param options - Optional configuration (keepdims, ddof, stream)
 * @returns The standard deviation array
 */
export function std(
  a: MLXArray,
  axis?: AxisSpec,
  options?: StdOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  if (nativeAxis !== undefined) {
    args.push(nativeAxis);
  }
  if (options?.keepdims) {
    args.push(true);
  } else if (options?.ddof !== undefined) {
    args.push(false); // must push keepdims before ddof
  }
  if (options?.ddof !== undefined) {
    args.push(options.ddof);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.std(...args);
  return MLXArray.fromHandle(handle);
}

export interface LogCumSumExpOptions extends StreamOptions {
  reverse?: boolean;
}

/**
 * Return the cumulative logsumexp of the elements along the given axis.
 *
 * @param a - Input array
 * @param axis - Axis along which to compute (required)
 * @param options - Optional configuration (reverse, stream)
 * @returns The cumulative logsumexp array
 */
export function logcumsumexp(
  a: MLXArray,
  axis: number,
  options?: LogCumSumExpOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), axis];
  if (options?.reverse) {
    args.push(true);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.logcumsumexp(...args);
  return MLXArray.fromHandle(handle);
}

export interface TraceOptions extends StreamOptions {
  dtype?: any;
}

/**
 * Return the sum along a specified diagonal in the given array.
 *
 * @param a - Input array
 * @param offset - Offset of the diagonal from the main diagonal (default 0)
 * @param axis1 - First axis of the 2-D sub-arrays (default 0)
 * @param axis2 - Second axis of the 2-D sub-arrays (default 1)
 * @param options - Optional configuration (dtype, stream)
 * @returns The trace array
 */
export function trace(
  a: MLXArray,
  offset?: number,
  axis1?: number,
  axis2?: number,
  options?: TraceOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (offset !== undefined) args.push(offset);
  if (axis1 !== undefined) args.push(axis1);
  if (axis2 !== undefined) args.push(axis2);
  if (options?.dtype !== undefined) args.push(options.dtype);
  appendStreamArg(args, options?.stream);
  const handle = addon.trace(...args);
  return MLXArray.fromHandle(handle);
}

export interface SoftmaxOptions extends StreamOptions {}

export function softmax(
  a: MLXArray,
  axis?: AxisSpec,
  options?: SoftmaxOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  const nativeAxis = normalizeAxisArg(axis);
  args.push(nativeAxis ?? null);
  appendStreamArg(args, options?.stream);
  const handle = addon.softmax(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Additional math ops
// ---------------------------------------------------------------------------

export function logaddexp(
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon.logaddexp(...args);
  return MLXArray.fromHandle(handle);
}

export interface ClipOptions extends StreamOptions {}

export function clip(
  a: MLXArray,
  aMin?: ScalarOrArray | null,
  aMax?: ScalarOrArray | null,
  options?: ClipOptions,
): MLXArray {
  const args: any[] = [
    toNativeHandle(a),
    aMin != null ? toNativeScalarOrArray(aMin) : null,
    aMax != null ? toNativeScalarOrArray(aMax) : null,
  ];
  appendStreamArg(args, options?.stream);
  const handle = addon.clip(...args);
  return MLXArray.fromHandle(handle);
}

export function log1p(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.log1p(...args);
  return MLXArray.fromHandle(handle);
}

export function negative(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.negative(...args);
  return MLXArray.fromHandle(handle);
}

export function reciprocal(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.reciprocal(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

export interface ExpandDimsOptions extends StreamOptions {}

export function expand_dims(
  a: MLXArray,
  axis: number | readonly number[],
  options?: ExpandDimsOptions,
): MLXArray {
  const nativeAxis = typeof axis === 'number' ? axis : Array.from(axis);
  const args: any[] = [toNativeHandle(a), nativeAxis];
  appendStreamArg(args, options?.stream);
  const handle = addon.expand_dims(...args);
  return MLXArray.fromHandle(handle);
}

export interface SqueezeOptions extends StreamOptions {}

export function squeeze(
  a: MLXArray,
  axis?: number | readonly number[] | null,
  options?: SqueezeOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (axis != null) {
    const nativeAxis = typeof axis === 'number' ? axis : Array.from(axis);
    args.push(nativeAxis);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.squeeze(...args);
  return MLXArray.fromHandle(handle);
}

export interface ConcatenateOptions extends StreamOptions {}

export function concatenate(
  arrays: readonly MLXArray[],
  axis?: number | null,
  options?: ConcatenateOptions,
): MLXArray {
  const nativeArrays = arrays.map(toNativeHandle);
  const args: any[] = [nativeArrays];
  if (axis !== undefined) {
    args.push(axis);
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.concatenate(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

export interface StackOptions extends StreamOptions {}

/**
 * Stack arrays along a new axis.
 *
 * @param arrays - Arrays to stack (must have the same shape)
 * @param axis - Axis along which to stack (default 0)
 * @param options - Optional stream configuration
 * @returns Stacked array
 */
export function stack(
  arrays: readonly MLXArray[],
  axis?: number,
  options?: StackOptions,
): MLXArray {
  const nativeArrays = arrays.map(toNativeHandle);
  const args: any[] = [nativeArrays];
  if (axis !== undefined) args.push(axis);
  appendStreamArg(args, options?.stream);
  const handle = addon.stack(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Convolution ops
// ---------------------------------------------------------------------------

export interface Conv1dOptions extends StreamOptions {}

/**
 * 1-D convolution over an input signal composed of several channels.
 *
 * @param input - Input array of shape (N, L, C_in)
 * @param weight - Weight array of shape (C_out, K, C_in/groups)
 * @param stride - Stride of the convolution (default 1)
 * @param padding - Zero-padding added to both sides (default 0)
 * @param dilation - Spacing between kernel elements (default 1)
 * @param groups - Number of blocked connections (default 1)
 * @param options - Optional stream configuration
 */
export function conv1d(
  input: MLXArray,
  weight: MLXArray,
  stride?: number,
  padding?: number,
  dilation?: number,
  groups?: number,
  options?: Conv1dOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(input), toNativeHandle(weight)];
  if (stride !== undefined) args.push(stride);
  if (padding !== undefined) args.push(padding);
  if (dilation !== undefined) args.push(dilation);
  if (groups !== undefined) args.push(groups);
  appendStreamArg(args, options?.stream);
  const handle = addon.conv1d(...args);
  return MLXArray.fromHandle(handle);
}

export interface Conv2dOptions extends StreamOptions {}

/**
 * 2-D convolution.
 *
 * @param input - Input array of shape (N, H, W, C_in)
 * @param weight - Weight array of shape (C_out, KH, KW, C_in/groups)
 * @param stride - Stride (default [1,1])
 * @param padding - Padding (default [0,0])
 * @param dilation - Dilation (default [1,1])
 * @param groups - Number of blocked connections (default 1)
 * @param options - Optional stream configuration
 */
export function conv2d(
  input: MLXArray,
  weight: MLXArray,
  stride?: number | [number, number],
  padding?: number | [number, number],
  dilation?: number | [number, number],
  groups?: number,
  options?: Conv2dOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(input), toNativeHandle(weight)];
  if (stride !== undefined) args.push(stride);
  if (padding !== undefined) args.push(padding);
  if (dilation !== undefined) args.push(dilation);
  if (groups !== undefined) args.push(groups);
  appendStreamArg(args, options?.stream);
  const handle = addon.conv2d(...args);
  return MLXArray.fromHandle(handle);
}

export interface Conv3dOptions extends StreamOptions {}

/**
 * 3-D convolution.
 *
 * @param input - Input array of shape (N, D, H, W, C_in)
 * @param weight - Weight array of shape (C_out, KD, KH, KW, C_in)
 * @param stride - Stride (default [1,1,1])
 * @param padding - Padding (default [0,0,0])
 * @param dilation - Dilation (default [1,1,1])
 * @param groups - Number of groups (default 1)
 * @param options - Optional stream configuration
 */
export function conv3d(
  input: MLXArray,
  weight: MLXArray,
  stride?: number | [number, number, number],
  padding?: number | [number, number, number],
  dilation?: number | [number, number, number],
  groups?: number,
  options?: Conv3dOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(input), toNativeHandle(weight)];
  if (stride !== undefined) args.push(stride);
  if (padding !== undefined) args.push(padding);
  if (dilation !== undefined) args.push(dilation);
  if (groups !== undefined) args.push(groups);
  appendStreamArg(args, options?.stream);
  const handle = addon.conv3d(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

export interface TakeAlongAxisOptions extends StreamOptions {}

export function take_along_axis(
  a: MLXArray,
  indices: MLXArray,
  axis: number | null,
  options?: TakeAlongAxisOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(indices), axis];
  appendStreamArg(args, options?.stream);
  const handle = addon.take_along_axis(...args);
  return MLXArray.fromHandle(handle);
}

export interface SliceOptions extends StreamOptions {}

/**
 * Slice (sub-array extraction).
 *
 * @param a - Input array
 * @param start - Start indices for each axis
 * @param stop - Stop indices for each axis
 * @param strides - Optional strides for each axis
 * @param options - Optional stream configuration
 */
export interface PadOptions extends StreamOptions {}

/**
 * Pad an array with a constant value.
 *
 * @param a - Input array
 * @param padWidth - Array of [low, high] pairs, one per axis
 * @param padValue - Value to use for padding (default: 0)
 * @param options - Stream options
 */
export function pad(
  a: MLXArray,
  padWidth: readonly (readonly [number, number])[],
  padValue?: number | MLXArray,
  options?: PadOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), padWidth.map(p => [...p])];
  if (padValue !== undefined) {
    if (padValue instanceof MLXArray) {
      args.push(toNativeHandle(padValue));
    } else {
      args.push(padValue);
    }
  }
  appendStreamArg(args, options?.stream);
  const handle = addon.pad(...args);
  return MLXArray.fromHandle(handle);
}

export function slice(
  a: MLXArray,
  start: readonly number[],
  stop: readonly number[],
  strides?: readonly number[],
  options?: SliceOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), [...start], [...stop]];
  if (strides !== undefined) args.push([...strides]);
  appendStreamArg(args, options?.stream);
  const handle = addon.slice(...args);
  return MLXArray.fromHandle(handle);
}

export interface AsStridedOptions extends StreamOptions {}

/**
 * Create a view of the array with the given shape, strides, and offset.
 *
 * @param a - Input array
 * @param shape - Output shape
 * @param strides - Strides in elements
 * @param offset - Offset in elements from start of data
 * @param options - Optional stream configuration
 */
export function as_strided(
  a: MLXArray,
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  options?: AsStridedOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), [...shape], [...strides], offset];
  appendStreamArg(args, options?.stream);
  const handle = addon.as_strided(...args);
  return MLXArray.fromHandle(handle);
}

export interface NumberOfElementsOptions extends StreamOptions {}

/**
 * Returns the number of elements along the given axes as a scalar array.
 *
 * @param a - Input array
 * @param axes - Axes to count
 * @param inverted - If true, count elements NOT along these axes
 * @param options - Optional stream configuration
 */
export function number_of_elements(
  a: MLXArray,
  axes: readonly number[],
  inverted: boolean = false,
  options?: NumberOfElementsOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), [...axes], inverted];
  appendStreamArg(args, options?.stream);
  const handle = addon.number_of_elements(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Fast namespace
// ---------------------------------------------------------------------------

export const fast = {
  /**
   * Scaled dot product attention.
   *
   * @param queries - Query array of shape (..., L, D)
   * @param keys - Key array of shape (..., S, D)
   * @param values - Value array of shape (..., S, Dv)
   * @param scale - Scale factor (typically 1/sqrt(D))
   * @param mask - Optional additive attention mask
   */
  scaled_dot_product_attention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: number,
    mask?: MLXArray,
  ): MLXArray {
    const args: any[] = [
      toNativeHandle(queries),
      toNativeHandle(keys),
      toNativeHandle(values),
      scale,
    ];
    if (mask !== undefined) args.push(toNativeHandle(mask));
    const handle = addon.fast.scaled_dot_product_attention(...args);
    return MLXArray.fromHandle(handle);
  },

  rms_norm(x: MLXArray, weight: MLXArray | null, eps: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(x), weight ? toNativeHandle(weight) : null, eps];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fast.rms_norm(...args));
  },

  layer_norm(x: MLXArray, weight: MLXArray | null, bias: MLXArray | null, eps: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(x), weight ? toNativeHandle(weight) : null, bias ? toNativeHandle(bias) : null, eps];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fast.layer_norm(...args));
  },

  rope(x: MLXArray, dims: number, traditional: boolean, base: number | null, scale: number, offset: number, freqs?: MLXArray, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(x), dims, traditional, base, scale, offset];
    if (freqs !== undefined) args.push(toNativeHandle(freqs));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fast.rope(...args));
  },
};

// ---------------------------------------------------------------------------
// Activation primitives
// ---------------------------------------------------------------------------

export function sigmoid(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.sigmoid(...args);
  return MLXArray.fromHandle(handle);
}

export function erf(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.erf(...args);
  return MLXArray.fromHandle(handle);
}

export function tanh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon.tanh(...args);
  return MLXArray.fromHandle(handle);
}

export interface SplitOptions extends StreamOptions {}

export function split(
  a: MLXArray,
  indicesOrSections: number | readonly number[],
  axis?: number,
  options?: SplitOptions,
): MLXArray[] {
  const nativeSplit = typeof indicesOrSections === 'number'
    ? indicesOrSections
    : Array.from(indicesOrSections);
  const args: any[] = [toNativeHandle(a), nativeSplit];
  if (axis !== undefined) {
    args.push(axis);
  }
  appendStreamArg(args, options?.stream);
  const handles: any[] = addon.split(...args);
  return handles.map((h: any) => MLXArray.fromHandle(h));
}

// ---------------------------------------------------------------------------
// take(a, indices, axis?, stream?) — gather elements by index
// ---------------------------------------------------------------------------

export interface TakeOptions extends StreamOptions {}

/**
 * Take elements from an array along an axis.
 *
 * @param a - Input array
 * @param indices - Integer indices to gather
 * @param axis - Optional axis along which to take. If omitted, array is flattened first.
 * @param options - Optional stream configuration
 * @returns Gathered elements
 */
export function take(
  a: MLXArray,
  indices: MLXArray,
  axis?: number,
  options?: TakeOptions,
): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(indices)];
  if (axis !== undefined) args.push(axis);
  appendStreamArg(args, options?.stream);
  const handle = addon.take(...args);
  return MLXArray.fromHandle(handle);
}

// ---------------------------------------------------------------------------
// Linear algebra (mlx.core.linalg)
// ---------------------------------------------------------------------------

export namespace linalg {
  export interface NormOptions extends StreamOptions {
    keepdims?: boolean;
  }

  /**
   * Compute vector or matrix norms.
   *
   * @param a - Input array.
   * @param ord - Order of the norm (number, 'fro', or 'nuc'). Default: 2-norm / Frobenius.
   * @param axis - Axis or axes along which to compute. Default: all.
   * @param options.keepdims - Keep normed axes as size-1 dims. Default: false.
   * @param options.stream - Stream to use.
   * @returns The norm(s).
   */
  export function norm(
    a: MLXArray,
    ord?: number | string | null,
    axis?: number | readonly number[] | null,
    options?: NormOptions,
  ): MLXArray {
    const nativeOrd = (ord === undefined || ord === null) ? null : ord;

    let nativeAxis: number | number[] | null = null;
    if (axis !== undefined && axis !== null) {
      if (typeof axis === 'number') {
        nativeAxis = axis;
      } else {
        nativeAxis = [...axis];
      }
    }

    const keepdims = options?.keepdims ?? false;

    const args: any[] = [
      toNativeHandle(a),
      nativeOrd,
      nativeAxis,
      keepdims,
    ];
    appendStreamArg(args, options?.stream);
    const handle = addon.linalg.norm(...args);
    return MLXArray.fromHandle(handle);
  }
}

// ---------------------------------------------------------------------------
// Batch: new unary math ops
// ---------------------------------------------------------------------------

export function ceil(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.ceil(...args));
}

export function floor(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.floor(...args));
}

export function round(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.round(...args));
}

export function isnan(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.isnan(...args));
}

export function isinf(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.isinf(...args));
}

export function isfinite(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.isfinite(...args));
}

export function logical_not(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.logical_not(...args));
}

export function sinh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.sinh(...args));
}

export function cosh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cosh(...args));
}

export function arcsinh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.arcsinh(...args));
}

export function arccosh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.arccosh(...args));
}

export function arctanh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.arctanh(...args));
}

export function degrees(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.degrees(...args));
}

export function radians(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.radians(...args));
}

export function erfinv(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.erfinv(...args));
}

export function expm1(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.expm1(...args));
}

// ---------------------------------------------------------------------------
// Batch: cumulative ops
// ---------------------------------------------------------------------------

export interface CumOpOptions extends StreamOptions {
  axis?: number;
  reverse?: boolean;
  inclusive?: boolean;
}

export function cumsum(a: MLXArray, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.axis !== undefined) args.push(options.axis);
  else args.push(0);
  if (options?.reverse !== undefined) args.push(options.reverse);
  if (options?.inclusive !== undefined) args.push(options.inclusive);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cumsum(...args));
}

export function cumprod(a: MLXArray, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.axis !== undefined) args.push(options.axis);
  else args.push(0);
  if (options?.reverse !== undefined) args.push(options.reverse);
  if (options?.inclusive !== undefined) args.push(options.inclusive);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cumprod(...args));
}

export function cummax(a: MLXArray, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.axis !== undefined) args.push(options.axis);
  else args.push(0);
  if (options?.reverse !== undefined) args.push(options.reverse);
  if (options?.inclusive !== undefined) args.push(options.inclusive);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cummax(...args));
}

export function cummin(a: MLXArray, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.axis !== undefined) args.push(options.axis);
  else args.push(0);
  if (options?.reverse !== undefined) args.push(options.reverse);
  if (options?.inclusive !== undefined) args.push(options.inclusive);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cummin(...args));
}

// ---------------------------------------------------------------------------
// Batch: new binary ops
// ---------------------------------------------------------------------------

export function floor_divide(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.floor_divide(...args));
}

export function remainder(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.remainder(...args));
}

export function logical_and(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.logical_and(...args));
}

export function logical_or(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.logical_or(...args));
}

export function bitwise_and(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.bitwise_and(...args));
}

export function bitwise_or(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.bitwise_or(...args));
}

export function bitwise_xor(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.bitwise_xor(...args));
}

export function left_shift(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.left_shift(...args));
}

export function right_shift(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.right_shift(...args));
}

// ---------------------------------------------------------------------------
// Batch: reduction & query ops
// ---------------------------------------------------------------------------

export function all(a: MLXArray, axis?: AxisSpec | null, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (axis !== undefined && axis !== null) {
    args.push(typeof axis === 'number' ? [axis] : [...axis]);
  }
  if (options?.keepdims) args.push(options.keepdims);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.all(...args));
}

export function any(a: MLXArray, axis?: AxisSpec | null, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (axis !== undefined && axis !== null) {
    args.push(typeof axis === 'number' ? [axis] : [...axis]);
  }
  if (options?.keepdims) args.push(options.keepdims);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.any(...args));
}

export function array_equal(a: MLXArray, b: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.array_equal(...args));
}

// ---------------------------------------------------------------------------
// Batch: shape & creation ops
// ---------------------------------------------------------------------------

export interface FlattenOptions extends StreamOptions {
  start_axis?: number;
  end_axis?: number;
}

export function flatten(a: MLXArray, options?: FlattenOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.start_axis !== undefined) args.push(options.start_axis);
  if (options?.end_axis !== undefined) args.push(options.end_axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.flatten(...args));
}

export interface EyeOptions extends StreamOptions {
  m?: number;
  k?: number;
  dtype?: DTypeLike;
}

export function eye(n: number, options?: EyeOptions): MLXArray {
  const args: any[] = [n];
  if (options?.m !== undefined) args.push(options.m);
  if (options?.k !== undefined) {
    if (options?.m === undefined) args.push(n); // must provide m before k
    args.push(options.k);
  }
  if (options?.dtype !== undefined) args.push(options.dtype);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.eye(...args));
}

export interface IdentityOptions extends StreamOptions {
  dtype?: DTypeLike;
}

export function identity(n: number, options?: IdentityOptions): MLXArray {
  const args: any[] = [n];
  if (options?.dtype !== undefined) args.push(options.dtype);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.identity(...args));
}

export interface LinspaceOptions extends StreamOptions {
  num?: number;
  dtype?: DTypeLike;
}

export function linspace(start: number, stop: number, options?: LinspaceOptions): MLXArray {
  const args: any[] = [start, stop];
  if (options?.num !== undefined) args.push(options.num);
  if (options?.dtype !== undefined) args.push(options.dtype);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.linspace(...args));
}

export interface TriOptions extends StreamOptions {
  k?: number;
}

export function tril(a: MLXArray, options?: TriOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.k !== undefined) args.push(options.k);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.tril(...args));
}

export function triu(a: MLXArray, options?: TriOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.k !== undefined) args.push(options.k);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.triu(...args));
}

export function broadcast_to(a: MLXArray, shape: readonly number[], options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), [...shape]];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.broadcast_to(...args));
}

export interface RepeatOptions extends StreamOptions {
  axis?: number;
}

export function repeat(a: MLXArray, repeats: number, options?: RepeatOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), repeats];
  if (options?.axis !== undefined) args.push(options.axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.repeat(...args));
}

export function tile(a: MLXArray, reps: number | readonly number[], options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), typeof reps === 'number' ? reps : [...reps]];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.tile(...args));
}

export interface SortOptions extends StreamOptions {
  axis?: number;
}

export function sort(a: MLXArray, options?: SortOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.axis !== undefined) args.push(options.axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.sort(...args));
}

export function argsort(a: MLXArray, options?: SortOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.axis !== undefined) args.push(options.axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.argsort(...args));
}

export interface DiagOptions extends StreamOptions {
  k?: number;
}

export function diag(a: MLXArray, options?: DiagOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.k !== undefined) args.push(options.k);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.diag(...args));
}

export interface DiagonalOptions extends StreamOptions {
  offset?: number;
  axis1?: number;
  axis2?: number;
}

export function diagonal(a: MLXArray, options?: DiagonalOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.offset !== undefined) args.push(options.offset);
  if (options?.axis1 !== undefined) args.push(options.axis1);
  if (options?.axis2 !== undefined) args.push(options.axis2);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.diagonal(...args));
}

export interface TopKOptions extends StreamOptions {
  axis?: number;
}

export function topk(a: MLXArray, k: number, options?: TopKOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), k];
  if (options?.axis !== undefined) args.push(options.axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.topk(...args));
}

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

export interface DeviceInfo {
  type: 'cpu' | 'gpu';
  index: number;
}

export function default_device(): DeviceInfo {
  return addon.default_device();
}

export function set_default_device(device: string | DeviceInfo): void {
  addon.set_default_device(device);
}

export function is_available(device: string | DeviceInfo): boolean {
  return addon.is_available(device);
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

export function clear_cache(): void {
  addon.clear_cache();
}

export function get_active_memory(): number {
  return addon.get_active_memory();
}

export function get_cache_memory(): number {
  return addon.get_cache_memory();
}

export function get_peak_memory(): number {
  return addon.get_peak_memory();
}

export function reset_peak_memory(): void {
  addon.reset_peak_memory();
}

export function set_cache_limit(limit: number): number {
  return addon.set_cache_limit(limit);
}

export function set_memory_limit(limit: number): number {
  return addon.set_memory_limit(limit);
}

export function set_wired_limit(limit: number): number {
  return addon.set_wired_limit(limit);
}

// ---------------------------------------------------------------------------
// FFT namespace
// ---------------------------------------------------------------------------

export namespace fft_ns {
  export function fft(a: MLXArray, n?: number, axis?: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push(n);
    if (axis !== undefined) args.push(axis);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fft(...args));
  }

  export function ifft(a: MLXArray, n?: number, axis?: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push(n);
    if (axis !== undefined) args.push(axis);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifft(...args));
  }

  export function fft2(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fft2(...args));
  }

  export function ifft2(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifft2(...args));
  }

  export function fftn(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fftn(...args));
  }

  export function ifftn(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifftn(...args));
  }

  export function rfft(a: MLXArray, n?: number, axis?: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push(n);
    if (axis !== undefined) args.push(axis);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.rfft(...args));
  }

  export function irfft(a: MLXArray, n?: number, axis?: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push(n);
    if (axis !== undefined) args.push(axis);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.irfft(...args));
  }

  export function rfft2(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.rfft2(...args));
  }

  export function irfft2(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.irfft2(...args));
  }

  export function rfftn(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.rfftn(...args));
  }

  export function irfftn(a: MLXArray, n?: readonly number[], axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (n !== undefined) args.push([...n]);
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.irfftn(...args));
  }

  export function fftshift(a: MLXArray, axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fftshift(...args));
  }

  export function ifftshift(a: MLXArray, axes?: readonly number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(a)];
    if (axes !== undefined) args.push([...axes]);
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifftshift(...args));
  }
}

// Re-export fft namespace as "fft" for use in index
export { fft_ns as fft };

// ---------------------------------------------------------------------------
// Batch 3: Additional core ops
// ---------------------------------------------------------------------------

// Simple unary ops
export function log2(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.log2(...args));
}

export function log10(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.log10(...args));
}

export function isposinf(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.isposinf(...args));
}

export function isneginf(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.isneginf(...args));
}

export function bitwise_invert(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.bitwise_invert(...args));
}

export function conjugate(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conjugate(...args));
}

/** Alias for conjugate */
export const conj = conjugate;

export function real(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.real(...args));
}

export function imag(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.imag(...args));
}

export function stop_gradient(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.stop_gradient(...args));
}

// Simple binary ops
export function outer(a: MLXArray, b: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.outer(...args));
}

export function inner(a: MLXArray, b: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.inner(...args));
}

export function kron(a: MLXArray, b: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.kron(...args));
}

// Parameterized ops

export interface NanToNumOptions extends StreamOptions {
  nan?: number;
  posinf?: number;
  neginf?: number;
}

export function nan_to_num(a: MLXArray, options?: NanToNumOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.nan !== undefined) args.push(options.nan);
  if (options?.posinf !== undefined) args.push(options.posinf);
  if (options?.neginf !== undefined) args.push(options.neginf);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.nan_to_num(...args));
}

export interface CloseOptions extends StreamOptions {
  rtol?: number;
  atol?: number;
  equal_nan?: boolean;
}

export function allclose(a: MLXArray, b: MLXArray, options?: CloseOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  if (options?.rtol !== undefined) args.push(options.rtol);
  if (options?.atol !== undefined) args.push(options.atol);
  if (options?.equal_nan !== undefined) args.push(options.equal_nan);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.allclose(...args));
}

export function isclose(a: MLXArray, b: MLXArray, options?: CloseOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  if (options?.rtol !== undefined) args.push(options.rtol);
  if (options?.atol !== undefined) args.push(options.atol);
  if (options?.equal_nan !== undefined) args.push(options.equal_nan);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.isclose(...args));
}

export function view(a: MLXArray, dtype: DTypeLike, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), dtype];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.view(...args));
}

export interface ContiguousOptions extends StreamOptions {
  allow_col_major?: boolean;
}

export function contiguous(a: MLXArray, options?: ContiguousOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (options?.allow_col_major !== undefined) args.push(options.allow_col_major);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.contiguous(...args));
}

export function hadamard_transform(a: MLXArray, scale?: number, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  if (scale !== undefined) args.push(scale);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.hadamard_transform(...args));
}

export function unflatten(a: MLXArray, axis: number, shape: readonly number[], options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), axis, [...shape]];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.unflatten(...args));
}

export interface PartitionOptions extends StreamOptions {
  axis?: number;
}

export function partition(a: MLXArray, kth: number, options?: PartitionOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), kth];
  if (options?.axis !== undefined) args.push(options.axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.partition(...args));
}

export function argpartition(a: MLXArray, kth: number, options?: PartitionOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), kth];
  if (options?.axis !== undefined) args.push(options.axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.argpartition(...args));
}

export function put_along_axis(a: MLXArray, indices: MLXArray, values: MLXArray, axis: number, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(indices), toNativeHandle(values), axis];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.put_along_axis(...args));
}

export interface RollOptions extends StreamOptions {
  axis?: number | readonly number[];
}

export function roll(a: MLXArray, shift: number | readonly number[], options?: RollOptions): MLXArray {
  const args: any[] = [
    toNativeHandle(a),
    Array.isArray(shift) ? [...shift] : shift,
  ];
  if (options?.axis !== undefined) {
    args.push(Array.isArray(options.axis) ? [...options.axis] : options.axis);
  }
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.roll(...args));
}

export interface TriCreateOptions extends StreamOptions {
  m?: number;
  k?: number;
  dtype?: DTypeLike;
}

export function tri(n: number, options?: TriCreateOptions): MLXArray {
  const args: any[] = [n];
  if (options?.m !== undefined) args.push(options.m);
  if (options?.k !== undefined) args.push(options.k);
  if (options?.dtype !== undefined) args.push(options.dtype);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.tri(...args));
}

export interface MeshgridOptions extends StreamOptions {
  sparse?: boolean;
  indexing?: 'xy' | 'ij';
}

export function meshgrid(arrays: readonly MLXArray[], options?: MeshgridOptions): MLXArray[] {
  const nativeArrays = arrays.map(a => toNativeHandle(a));
  const args: any[] = [nativeArrays];
  if (options?.sparse !== undefined) args.push(options.sparse);
  if (options?.indexing !== undefined) args.push(options.indexing);
  appendStreamArg(args, options?.stream);
  const result = addon.meshgrid(...args);
  return result.map((h: any) => MLXArray.fromHandle(h));
}

export function broadcast_arrays(arrays: readonly MLXArray[], options?: StreamOptions): MLXArray[] {
  const nativeArrays = arrays.map(a => toNativeHandle(a));
  const args: any[] = [nativeArrays];
  appendStreamArg(args, options?.stream);
  const result = addon.broadcast_arrays(...args);
  return result.map((h: any) => MLXArray.fromHandle(h));
}

export function atleast_1d(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.atleast_1d(...args));
}

export function atleast_2d(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.atleast_2d(...args));
}

export function atleast_3d(a: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.atleast_3d(...args));
}

export interface SliceUpdateOptions extends StreamOptions {
  strides?: readonly number[];
}

export function slice_update(src: MLXArray, update: MLXArray, start: readonly number[], stop: readonly number[], options?: SliceUpdateOptions): MLXArray {
  const args: any[] = [toNativeHandle(src), toNativeHandle(update), [...start], [...stop]];
  if (options?.strides !== undefined) args.push([...options.strides]);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.slice_update(...args));
}

export interface ConvGeneralOptions extends StreamOptions {
  stride?: readonly number[];
  padding_lo?: readonly number[];
  padding_hi?: readonly number[];
  kernel_dilation?: readonly number[];
  input_dilation?: readonly number[];
  groups?: number;
  flip?: boolean;
}

export function conv_general(input: MLXArray, weight: MLXArray, options?: ConvGeneralOptions): MLXArray {
  const args: any[] = [toNativeHandle(input), toNativeHandle(weight)];
  args.push(options?.stride ? [...options.stride] : []);
  args.push(options?.padding_lo ? [...options.padding_lo] : []);
  args.push(options?.padding_hi ? [...options.padding_hi] : []);
  args.push(options?.kernel_dilation ? [...options.kernel_dilation] : []);
  args.push(options?.input_dilation ? [...options.input_dilation] : []);
  if (options?.groups !== undefined) args.push(options.groups);
  if (options?.flip !== undefined) args.push(options.flip);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_general(...args));
}

export interface ConvTranspose1dOptions extends StreamOptions {
  stride?: number;
  padding?: number;
  dilation?: number;
  output_padding?: number;
  groups?: number;
}

export function conv_transpose1d(input: MLXArray, weight: MLXArray, options?: ConvTranspose1dOptions): MLXArray {
  const args: any[] = [toNativeHandle(input), toNativeHandle(weight)];
  if (options?.stride !== undefined) args.push(options.stride);
  if (options?.padding !== undefined) args.push(options.padding);
  if (options?.dilation !== undefined) args.push(options.dilation);
  if (options?.output_padding !== undefined) args.push(options.output_padding);
  if (options?.groups !== undefined) args.push(options.groups);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_transpose1d(...args));
}

export interface ConvTranspose2dOptions extends StreamOptions {
  stride?: [number, number] | number;
  padding?: [number, number] | number;
  dilation?: [number, number] | number;
  output_padding?: [number, number] | number;
  groups?: number;
}

export function conv_transpose2d(input: MLXArray, weight: MLXArray, options?: ConvTranspose2dOptions): MLXArray {
  const normPair = (v: [number, number] | number | undefined): [number, number] | undefined => {
    if (v === undefined) return undefined;
    return typeof v === 'number' ? [v, v] : v;
  };
  const args: any[] = [toNativeHandle(input), toNativeHandle(weight)];
  const s = normPair(options?.stride); if (s) args.push(s);
  const p = normPair(options?.padding); if (p) args.push(p);
  const d = normPair(options?.dilation); if (d) args.push(d);
  const op = normPair(options?.output_padding); if (op) args.push(op);
  if (options?.groups !== undefined) args.push(options.groups);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_transpose2d(...args));
}

export interface ConvTranspose3dOptions extends StreamOptions {
  stride?: [number, number, number] | number;
  padding?: [number, number, number] | number;
  dilation?: [number, number, number] | number;
  output_padding?: [number, number, number] | number;
  groups?: number;
}

export function conv_transpose3d(input: MLXArray, weight: MLXArray, options?: ConvTranspose3dOptions): MLXArray {
  const normTriple = (v: [number, number, number] | number | undefined): [number, number, number] | undefined => {
    if (v === undefined) return undefined;
    return typeof v === 'number' ? [v, v, v] : v;
  };
  const args: any[] = [toNativeHandle(input), toNativeHandle(weight)];
  const s = normTriple(options?.stride); if (s) args.push(s);
  const p = normTriple(options?.padding); if (p) args.push(p);
  const d = normTriple(options?.dilation); if (d) args.push(d);
  const op = normTriple(options?.output_padding); if (op) args.push(op);
  if (options?.groups !== undefined) args.push(options.groups);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_transpose3d(...args));
}

export function einsum(subscripts: string, operands: readonly MLXArray[], options?: StreamOptions): MLXArray {
  const nativeOps = operands.map(a => toNativeHandle(a));
  const args: any[] = [subscripts, nativeOps];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.einsum(...args));
}

export function tensordot(a: MLXArray, b: MLXArray, axes: number | [readonly number[], readonly number[]], options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  if (typeof axes === 'number') {
    args.push(axes);
  } else {
    args.push([[...axes[0]], [...axes[1]]]);
  }
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.tensordot(...args));
}

export interface BlockMaskedMMOptions extends StreamOptions {
  mask_out?: MLXArray | null;
  mask_lhs?: MLXArray | null;
  mask_rhs?: MLXArray | null;
}

export function block_masked_mm(a: MLXArray, b: MLXArray, block_size: number, options?: BlockMaskedMMOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b), block_size];
  args.push(options?.mask_out ? toNativeHandle(options.mask_out) : null);
  args.push(options?.mask_lhs ? toNativeHandle(options.mask_lhs) : null);
  args.push(options?.mask_rhs ? toNativeHandle(options.mask_rhs) : null);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.block_masked_mm(...args));
}

export interface GatherMMOptions extends StreamOptions {
  lhs_indices?: MLXArray | null;
  rhs_indices?: MLXArray | null;
  sorted_indices?: boolean;
}

export function gather_mm(a: MLXArray, b: MLXArray, options?: GatherMMOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b)];
  args.push(options?.lhs_indices ? toNativeHandle(options.lhs_indices) : null);
  args.push(options?.rhs_indices ? toNativeHandle(options.rhs_indices) : null);
  if (options?.sorted_indices !== undefined) args.push(options.sorted_indices);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.gather_mm(...args));
}

export function segmented_mm(a: MLXArray, b: MLXArray, segments: MLXArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeHandle(b), toNativeHandle(segments)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.segmented_mm(...args));
}

export interface QuantizeOptions extends StreamOptions {
  group_size?: number;
  bits?: number;
  mode?: string;
}

export function quantize(w: MLXArray, options?: QuantizeOptions): [MLXArray, MLXArray, MLXArray] {
  const args: any[] = [toNativeHandle(w)];
  if (options?.group_size !== undefined) args.push(options.group_size);
  if (options?.bits !== undefined) args.push(options.bits);
  if (options?.mode !== undefined) args.push(options.mode);
  appendStreamArg(args, options?.stream);
  const result = addon.quantize(...args);
  return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1]), MLXArray.fromHandle(result[2])];
}

export interface DequantizeOptions extends StreamOptions {
  biases?: MLXArray | null;
  group_size?: number;
  bits?: number;
  mode?: string;
}

export function dequantize(w: MLXArray, scales: MLXArray, options?: DequantizeOptions): MLXArray {
  const args: any[] = [toNativeHandle(w), toNativeHandle(scales)];
  args.push(options?.biases ? toNativeHandle(options.biases) : null);
  if (options?.group_size !== undefined) args.push(options.group_size);
  if (options?.bits !== undefined) args.push(options.bits);
  if (options?.mode !== undefined) args.push(options.mode);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.dequantize(...args));
}

export interface QuantizedMatmulOptions extends StreamOptions {
  biases?: MLXArray | null;
  transpose?: boolean;
  group_size?: number;
  bits?: number;
  mode?: string;
}

export function quantized_matmul(x: MLXArray, w: MLXArray, scales: MLXArray, options?: QuantizedMatmulOptions): MLXArray {
  const args: any[] = [toNativeHandle(x), toNativeHandle(w), toNativeHandle(scales)];
  args.push(options?.biases ? toNativeHandle(options.biases) : null);
  if (options?.transpose !== undefined) args.push(options.transpose);
  if (options?.group_size !== undefined) args.push(options.group_size);
  if (options?.bits !== undefined) args.push(options.bits);
  if (options?.mode !== undefined) args.push(options.mode);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.quantized_matmul(...args));
}

export interface GatherQMMOptions extends StreamOptions {
  biases?: MLXArray | null;
  lhs_indices?: MLXArray | null;
  rhs_indices?: MLXArray | null;
  transpose?: boolean;
  group_size?: number;
  bits?: number;
  mode?: string;
  sorted_indices?: boolean;
}

export function gather_qmm(x: MLXArray, w: MLXArray, scales: MLXArray, options?: GatherQMMOptions): MLXArray {
  const args: any[] = [toNativeHandle(x), toNativeHandle(w), toNativeHandle(scales)];
  args.push(options?.biases ? toNativeHandle(options.biases) : null);
  args.push(options?.lhs_indices ? toNativeHandle(options.lhs_indices) : null);
  args.push(options?.rhs_indices ? toNativeHandle(options.rhs_indices) : null);
  if (options?.transpose !== undefined) args.push(options.transpose);
  if (options?.group_size !== undefined) args.push(options.group_size);
  if (options?.bits !== undefined) args.push(options.bits);
  if (options?.mode !== undefined) args.push(options.mode);
  if (options?.sorted_indices !== undefined) args.push(options.sorted_indices);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.gather_qmm(...args));
}

// TS-only aliases and wrappers

/** Alias: concatenate arrays along an axis */
export function concat(arrays: readonly MLXArray[], axis?: number, options?: StreamOptions): MLXArray {
  const nativeArrays = arrays.map(a => toNativeHandle(a));
  const args: any[] = [nativeArrays];
  if (axis !== undefined) args.push(axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.concatenate(...args));
}

/** Alias: divmod returns [floor_divide(a,b), remainder(a,b)] */
export function divmod(a: MLXArray, b: MLXArray, options?: StreamOptions): [MLXArray, MLXArray] {
  return [floor_divide(a, b, options), remainder(a, b, options)];
}

/** Alias for transpose */
export function permute_dims(a: MLXArray, axes?: readonly number[], options?: StreamOptions): MLXArray {
  return transpose(a, axes, options);
}


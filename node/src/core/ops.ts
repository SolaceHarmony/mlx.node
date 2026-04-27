import addon from '../internal/addon';
import MLXArray, { normalizeShapeInput, from_js_array } from './array';
import type { StreamLike } from './stream';
import { toNativeStreamArgument } from './stream';
import type { DTypeLike } from './dtype';
import type { Device, DeviceLike } from './device';
import { normalizeDevice } from './device';
import { tree_flatten } from '../utils/tree';

function toNativeHandle(tensor: MLXArray): any {
  if (tensor && typeof tensor === 'object' && 'toNative' in (tensor as any)) {
    return (tensor as any).toNative();
  }
  return tensor;
}

function normalizeStream(stream?: StreamLike | null): any {
  if (stream == null) {
    return null;
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
  if (native !== undefined && native !== null) {
    args.push(native);
  }
}

export interface StreamOptions {
  stream?: StreamLike;
}

export type AxisSpec = number | readonly number[] | null;

function normalizeAxisArg(axis?: AxisSpec): number[] | undefined {
  if (axis === null || axis === undefined) {
    return undefined;
  }
  if (typeof axis === 'number') {
    return [axis];
  }
  return Array.from(axis);
}

export interface BinaryOpOptions extends StreamOptions {}
export interface UnaryOpOptions extends StreamOptions {}
export interface PadOptions extends StreamOptions {}
export interface SliceOptions extends StreamOptions {}
export interface TransposeOptions extends StreamOptions {}
export interface ReshapeOptions extends StreamOptions {}
export interface SplitOptions extends StreamOptions {}
export interface TileOptions extends StreamOptions {}
export interface RepeatOptions extends StreamOptions {}
export interface ConvOptions extends StreamOptions {}
export interface ConvGeneralOptions extends StreamOptions {}
export interface WhereOptions extends StreamOptions {}
export interface ClipOptions extends StreamOptions {}
export interface TopKOptions extends StreamOptions {}
export interface SortOptions extends StreamOptions {
  axis?: number;
}
export interface ArgSortOptions extends StreamOptions {
  axis?: number;
}

export interface AsStridedOptions extends StreamOptions {}
export interface NumberOfElementsOptions extends StreamOptions {
  axes?: number[];
  inverted?: boolean;
}
export interface StdOptions extends VarOptions {}
export interface SoftmaxOptions extends StreamOptions {}
export interface ExpandDimsOptions extends StreamOptions {}
export interface SqueezeOptions extends StreamOptions {}
export interface ConcatenateOptions extends StreamOptions {
  axis?: number;
}
export interface TakeAlongAxisOptions extends StreamOptions {}
export interface MoveAxisOptions extends StreamOptions {}
export interface SwapAxesOptions extends StreamOptions {}
export interface ArangeOptions extends StreamOptions {
  dtype?: DTypeLike;
}
export interface NormalOptions extends StreamOptions {
  loc?: number | MLXArray;
  scale?: number | MLXArray;
  key?: MLXArray;
  dtype?: DTypeLike;
}
export interface AddmmOptions extends StreamOptions {}
export interface LogCumSumExpOptions extends CumOpOptions {}
export interface TraceOptions extends StreamOptions {
  dtype?: DTypeLike;
}
export interface FlattenOptions extends StreamOptions {
  start_axis?: number;
  end_axis?: number;
}
export interface EyeOptions extends StreamOptions {
  dtype?: DTypeLike;
}
export interface IdentityOptions extends StreamOptions {
  dtype?: DTypeLike;
}
export interface LinspaceOptions extends StreamOptions {
  dtype?: DTypeLike;
}
export interface TriOptions extends StreamOptions {
  m?: number;
  k?: number;
  dtype?: DTypeLike;
}
export interface DiagOptions extends StreamOptions {}
export interface DiagonalOptions extends StreamOptions {}
export interface NanToNumOptions extends StreamOptions {
  nan?: number;
  posinf?: number;
  neginf?: number;
}
export interface CloseOptions extends StreamOptions {
  rtol?: number;
  atol?: number;
  equal_nan?: boolean;
}
export interface ContiguousOptions extends StreamOptions {}
export interface PartitionOptions extends StreamOptions {}
export interface RollOptions extends StreamOptions {}
export interface TriCreateOptions extends StreamOptions {
  dtype?: DTypeLike;
}
export interface MeshgridOptions extends StreamOptions {
  indexing?: 'xy' | 'ij';
}
export interface SliceUpdateOptions extends StreamOptions {}
export interface ConvTranspose1dOptions extends StreamOptions {}
export interface ConvTranspose2dOptions extends StreamOptions {}
export interface ConvTranspose3dOptions extends StreamOptions {}
export interface SaveSafetensorsOptions extends StreamOptions {}
export interface BlockMaskedMMOptions extends StreamOptions {}
export interface GatherMMOptions extends StreamOptions {}
export interface QuantizeOptions extends StreamOptions {}
export interface DequantizeOptions extends StreamOptions {}
export interface QuantizedMatmulOptions extends StreamOptions {}
export interface GatherQMMOptions extends StreamOptions {}

export type LoadResult = MLXArray | Record<string, MLXArray>;
export type DeviceInfo = Device;

type ScalarOrArray = MLXArray | number | boolean | bigint | number[] | bigint[];

function toNativeScalarOrArray(value: ScalarOrArray): any {
  if (value && typeof value === 'object' && !Array.isArray(value) && 'toNative' in (value as any)) {
    return (value as any).toNative();
  }
  if (Array.isArray(value)) {
    return toNativeHandle(from_js_array(value as any));
  }
  // Return raw scalar for native inference in Add, Multiply, etc.
  return value;
}

function binaryOp(
  name: string,
  a: ScalarOrArray,
  b: ScalarOrArray,
  options?: BinaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const handle = addon[name](...args);
  return MLXArray.fromHandle(handle);
}

function unaryOp(
  name: string,
  a: ScalarOrArray,
  options?: UnaryOpOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  const handle = addon[name](...args);
  return MLXArray.fromHandle(handle);
}

// ─────────────────────────────── BINARY OPS ────────────────────────────────

export function add(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('add', a, b, options); }
export function multiply(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('multiply', a, b, options); }
export function subtract(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('subtract', a, b, options); }
export function divide(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('divide', a, b, options); }
export function power(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('power', a, b, options); }
export function equal(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('equal', a, b, options); }
export function not_equal(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('not_equal', a, b, options); }
export function less(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('less', a, b, options); }
export function less_equal(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('less_equal', a, b, options); }
export function greater(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('greater', a, b, options); }
export function greater_equal(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('greater_equal', a, b, options); }
export function maximum(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('maximum', a, b, options); }
export function minimum(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('minimum', a, b, options); }
export function floor_divide(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('floor_divide', a, b, options); }
export function remainder(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('remainder', a, b, options); }
export function fmod(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('fmod', a, b, options); }
export function logaddexp(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('logaddexp', a, b, options); }

export function clip(
  a: ScalarOrArray,
  a_min?: ScalarOrArray | null,
  a_max?: ScalarOrArray | null,
  options?: ClipOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  args.push(a_min !== undefined && a_min !== null ? toNativeScalarOrArray(a_min) : null);
  args.push(a_max !== undefined && a_max !== null ? toNativeScalarOrArray(a_max) : null);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.clip(...args));
}
export function logical_and(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('logical_and', a, b, options); }
export function logical_or(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('logical_or', a, b, options); }
export function bitwise_and(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('bitwise_and', a, b, options); }
export function bitwise_or(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('bitwise_or', a, b, options); }
export function bitwise_xor(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('bitwise_xor', a, b, options); }
export function bitwise_shift_left(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('bitwise_shift_left', a, b, options); }
export function bitwise_shift_right(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('bitwise_shift_right', a, b, options); }
export function arctan2(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): MLXArray { return binaryOp('arctan2', a, b, options); }

export const left_shift = bitwise_shift_left;
export const right_shift = bitwise_shift_right;

export function divmod(a: ScalarOrArray, b: ScalarOrArray, options?: BinaryOpOptions): [MLXArray, MLXArray] {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  const result: any[] = addon.divmod(...args);
  return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1])];
}

// ─────────────────────────────── UNARY OPS ─────────────────────────────────

export function tan(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('tan', a, options); }
export function sin(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('sin', a, options); }
export function cos(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('cos', a, options); }
export function arcsin(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('arcsin', a, options); }
export function arccos(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('arccos', a, options); }
export function arctan(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('arctan', a, options); }
export function sinh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('sinh', a, options); }
export function cosh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('cosh', a, options); }
export function arcsinh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('arcsinh', a, options); }
export function arccosh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('arccosh', a, options); }
export function arctanh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('arctanh', a, options); }
export function rsqrt(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('rsqrt', a, options); }
export function square(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('square', a, options); }
export function sign(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('sign', a, options); }
export function abs(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('abs', a, options); }
export function sqrt(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('sqrt', a, options); }
export function exp(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('exp', a, options); }
export function log(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('log', a, options); }
export function log1p(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('log1p', a, options); }
export function log2(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('log2', a, options); }
export function log10(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('log10', a, options); }
export function expm1(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('expm1', a, options); }
export function sigmoid(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('sigmoid', a, options); }
export function erf(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('erf', a, options); }
export function erfinv(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('erfinv', a, options); }
export function negative(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('negative', a, options); }
export function floor(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('floor', a, options); }
export function ceil(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('ceil', a, options); }
export function round(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('round', a, options); }
export function trunc(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('trunc', a, options); }
export function isnan(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('isnan', a, options); }
export function isinf(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('isinf', a, options); }
export function isposinf(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('isposinf', a, options); }
export function isneginf(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('isneginf', a, options); }
export function isfinite(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('isfinite', a, options); }
export function logical_not(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('logical_not', a, options); }
export function bitwise_invert(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('bitwise_invert', a, options); }
export function degrees(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('degrees', a, options); }
export function radians(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('radians', a, options); }
export function reciprocal(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('reciprocal', a, options); }
export function real(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('real', a, options); }
export function imag(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('imag', a, options); }
export function stop_gradient(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('stop_gradient', a, options); }
export function tanh(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('tanh', a, options); }
export function conjugate(a: ScalarOrArray, options?: UnaryOpOptions): MLXArray { return unaryOp('conjugate', a, options); }
export const conj = conjugate;

export function load(file: string, options?: StreamOptions): LoadResult {
  const args: any[] = [file];
  appendStreamArg(args, options?.stream);
  const result = addon.load(...args);
  if (result.arrays) {
    // Dictionary format (safetensors, gguf)
    const wrappedArrays: Record<string, MLXArray> = {};
    for (const [key, handle] of Object.entries(result.arrays)) {
      wrappedArrays[key] = MLXArray.fromHandle(handle);
    }
    return wrappedArrays;
  }
  // Single array format (.npy, .npz)
  return MLXArray.fromHandle(result);
}

export function load_safetensors(file: string, options?: StreamOptions): { arrays: Record<string, MLXArray>, metadata: Record<string, string> } {
    const args: any[] = [file];
    appendStreamArg(args, options?.stream);
    const result = addon.load(...args);
    const wrappedArrays: Record<string, MLXArray> = {};
    for (const [key, handle] of Object.entries(result.arrays)) {
      wrappedArrays[key] = MLXArray.fromHandle(handle);
    }
    return {
        arrays: wrappedArrays,
        metadata: result.metadata
    };
}

export function load_gguf(file: string, options?: StreamOptions): { arrays: Record<string, MLXArray>, metadata: Record<string, any> } {
    const args: any[] = [file];
    appendStreamArg(args, options?.stream);
    const result = addon.load(...args);
    const wrappedArrays: Record<string, MLXArray> = {};
    for (const [key, handle] of Object.entries(result.arrays)) {
      wrappedArrays[key] = MLXArray.fromHandle(handle);
    }
    return {
        arrays: wrappedArrays,
        metadata: result.metadata
    };
}

export function save(file: string, a: ScalarOrArray, options?: StreamOptions): void {
  const args: any[] = [file, toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  addon.save(...args);
}

export function save_safetensors(file: string, a: any, options?: SaveSafetensorsOptions): void {
  const flattened = tree_flatten(a, { isLeaf: (v) => v && typeof v === 'object' && 'toNative' in (v as any) });
  const dict: Record<string, any> = {};
  if (Array.isArray(flattened)) {
      for (const [key, value] of flattened) {
          dict[key] = toNativeScalarOrArray(value as any);
      }
  } else {
      for (const [key, value] of Object.entries(flattened)) {
          dict[key] = toNativeScalarOrArray(value as any);
      }
  }
  const args: any[] = [file, dict];
  if (options?.metadata) {
      args.push(options.metadata);
  }
  appendStreamArg(args, options?.stream);
  addon.save_safetensors(...args);
}

export function save_gguf(file: string, a: any, options?: StreamOptions): void {
  const flattened = tree_flatten(a, { isLeaf: (v) => v && typeof v === 'object' && 'toNative' in (v as any) });
  const dict: Record<string, any> = {};
  if (Array.isArray(flattened)) {
      for (const [key, value] of flattened) {
          dict[key] = toNativeScalarOrArray(value as any);
      }
  } else {
      for (const [key, value] of Object.entries(flattened)) {
          dict[key] = toNativeScalarOrArray(value as any);
      }
  }
  const args: any[] = [file, dict];
  appendStreamArg(args, options?.stream);
  addon.save_gguf(...args);
}

export function import_function(name: string, options?: StreamOptions): void {
  const args: any[] = [name];
  appendStreamArg(args, options?.stream);
  addon.import_function(...args);
}

export function export_function(name: string, options?: StreamOptions): void {
  const args: any[] = [name];
  appendStreamArg(args, options?.stream);
  addon.export_function(...args);
}

// ─────────────────────────────── REDUCTIONS ────────────────────────────────

export interface ReductionOptions extends StreamOptions {
  keepdims?: boolean;
}

export function sum(a: ScalarOrArray, axis?: AxisSpec, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, options?.keepdims ?? false];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.sum(...args));
}

export function mean(a: ScalarOrArray, axis?: AxisSpec, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, options?.keepdims ?? false];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.mean(...args));
}

export interface VarOptions extends ReductionOptions {
  ddof?: number;
}

export function variance(a: ScalarOrArray, axis?: AxisSpec, options?: VarOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, options?.keepdims ?? false];
  if (options?.ddof !== undefined) args.push(options.ddof);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.variance(...args));
}

export function std(a: ScalarOrArray, axis?: AxisSpec, options?: StdOptions): MLXArray {
  return sqrt(variance(a, axis, options), options);
}

export function min(a: ScalarOrArray, axis?: AxisSpec, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, options?.keepdims ?? false];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.min(...args));
}

export function max(a: ScalarOrArray, axis?: AxisSpec, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, options?.keepdims ?? false];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.max(...args));
}

export function prod(a: ScalarOrArray, axis?: AxisSpec, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, options?.keepdims ?? false];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.prod(...args));
}

export function logsumexp(a: ScalarOrArray, axis?: AxisSpec, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, options?.keepdims ?? false];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.logsumexp(...args));
}

export function argmin(a: ScalarOrArray, axis?: number | null, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  if (axis !== undefined && axis !== null) args.push(axis);
  args.push(options?.keepdims ?? false);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.argmin(...args));
}

export function argmax(a: ScalarOrArray, axis?: number | null, options?: ReductionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  if (axis !== undefined && axis !== null) args.push(axis);
  args.push(options?.keepdims ?? false);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.argmax(...args));
}

export function softmax(a: ScalarOrArray, axis?: AxisSpec, options?: SoftmaxOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.softmax(...args));
}

export function all(a: ScalarOrArray, axisOrOptions?: AxisSpec | ReductionOptions, options?: ReductionOptions): MLXArray {
  let axis: AxisSpec = null;
  let opts = options;
  if (axisOrOptions && typeof axisOrOptions === 'object' && !('toNative' in (axisOrOptions as any)) && !Array.isArray(axisOrOptions)) {
      opts = axisOrOptions as ReductionOptions;
  } else {
      axis = axisOrOptions as AxisSpec;
  }
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, opts?.keepdims ?? false];
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.all(...args));
}

export function any(a: ScalarOrArray, axisOrOptions?: AxisSpec | ReductionOptions, options?: ReductionOptions): MLXArray {
  let axis: AxisSpec = null;
  let opts = options;
  if (axisOrOptions && typeof axisOrOptions === 'object' && !('toNative' in (axisOrOptions as any)) && !Array.isArray(axisOrOptions)) {
      opts = axisOrOptions as ReductionOptions;
  } else {
      axis = axisOrOptions as AxisSpec;
  }
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axis) ?? null, opts?.keepdims ?? false];
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.any(...args));
}

// ─────────────────────────────── SCANS ────────────────────────────────────

export interface CumOpOptions extends StreamOptions {
  reverse?: boolean;
  inclusive?: boolean;
}

export function cumsum(a: ScalarOrArray, axis: number, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), axis, options?.reverse ?? false, options?.inclusive ?? true];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cumsum(...args));
}

export function cumprod(a: ScalarOrArray, axis: number, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), axis, options?.reverse ?? false, options?.inclusive ?? true];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cumprod(...args));
}

export function cummax(a: ScalarOrArray, axis: number, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), axis, options?.reverse ?? false, options?.inclusive ?? true];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cummax(...args));
}

export function cummin(a: ScalarOrArray, axis: number, options?: CumOpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), axis, options?.reverse ?? false, options?.inclusive ?? true];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.cummin(...args));
}

export function logcumsumexp(a: ScalarOrArray, axis: number, options?: LogCumSumExpOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), axis, options?.reverse ?? false, options?.inclusive ?? true];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.logcumsumexp(...args));
}

// ─────────────────────────────── LINEAR ALGEBRA ───────────────────────────

export function matmul(a: ScalarOrArray, b: ScalarOrArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.matmul(...args));
}

export const dot = matmul;

export function addmm(c: ScalarOrArray, a: ScalarOrArray, b: ScalarOrArray, alpha = 1.0, beta = 1.0, options?: AddmmOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(c), toNativeScalarOrArray(a), toNativeScalarOrArray(b), alpha, beta];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.addmm(...args));
}

export function outer(a: ScalarOrArray, b: ScalarOrArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.outer(...args));
}

export function inner(a: ScalarOrArray, b: ScalarOrArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.inner(...args));
}

export function trace(a: ScalarOrArray, offset = 0, axis1 = 0, axis2 = 1, dtypeOrOptions?: DTypeLike | TraceOptions, maybeOptions?: TraceOptions): MLXArray {
  let dtype: DTypeLike | undefined;
  let options: TraceOptions | undefined;
  if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('toNative' in (dtypeOrOptions as any)) && !('key' in (dtypeOrOptions as any))) {
    options = dtypeOrOptions as TraceOptions;
    dtype = options.dtype;
  } else {
    dtype = dtypeOrOptions as DTypeLike;
    options = maybeOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), offset, axis1, axis2];
  if (dtype) {
    const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.trace(...args));
}

export function kron(a: ScalarOrArray, b: ScalarOrArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.kron(...args));
}

export function tensordot(a: ScalarOrArray, b: ScalarOrArray, axes: number | [readonly number[], readonly number[]] = 2, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  if (typeof axes === 'number') args.push(axes);
  else args.push([[...axes[0]], [...axes[1]]]);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.tensordot(...args));
}

export function einsum(subscripts: string, ...operands: (ScalarOrArray | StreamOptions)[]): MLXArray {
  let options: StreamOptions | undefined;
  let ops = operands;
  if (operands.length > 0) {
    const last = operands[operands.length - 1];
    if (last && typeof last === 'object' && !('toNative' in (last as any)) && !Array.isArray(last)) {
      options = last as StreamOptions;
      ops = operands.slice(0, -1);
    }
  }
  const nativeOps = (ops as ScalarOrArray[]).map(toNativeScalarOrArray);
  const args: any[] = [subscripts, nativeOps];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.einsum(...args));
}

export function einsum_path(subscripts: string, ...operands: ScalarOrArray[]): any {
  const nativeOps = operands.map(toNativeScalarOrArray);
  return addon.einsum_path(subscripts, nativeOps);
}

export function norm(a: ScalarOrArray, ord?: number | string | null, axis?: AxisSpec, keepdimsOrOptions?: boolean | ReductionOptions, options?: ReductionOptions): MLXArray {
    let keepdims = false;
    let opts = options;
    if (typeof keepdimsOrOptions === 'boolean') {
        keepdims = keepdimsOrOptions;
    } else if (keepdimsOrOptions && typeof keepdimsOrOptions === 'object') {
        keepdims = keepdimsOrOptions.keepdims ?? false;
        opts = keepdimsOrOptions;
    }
    const args: any[] = [toNativeScalarOrArray(a), ord ?? null, normalizeAxisArg(axis) ?? null, keepdims];
    appendStreamArg(args, opts?.stream);
    return MLXArray.fromHandle(addon.norm(...args));
}

// ─────────────────────────────── INDEXING ──────────────────────────────────

export function take(a: MLXArray, indices: ScalarOrArray, axis?: number, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeScalarOrArray(indices)];
  if (axis !== undefined) args.push(axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.take(...args));
}

export function take_along_axis(a: MLXArray, indices: ScalarOrArray, axis: number | null, options?: TakeAlongAxisOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeScalarOrArray(indices), axis];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.take_along_axis(...args));
}

export function put_along_axis(a: MLXArray, indices: ScalarOrArray, values: ScalarOrArray, axis: number, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeHandle(a), toNativeScalarOrArray(indices), toNativeScalarOrArray(values), axis];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.put_along_axis(...args));
}

export function slice(
  a: ScalarOrArray,
  start: readonly number[],
  stop: readonly number[],
  stridesOrOptions?: readonly number[] | SliceOptions,
  maybeOptions?: SliceOptions,
): MLXArray {
  let strides: readonly number[] | undefined;
  let options: SliceOptions | undefined;
  if (Array.isArray(stridesOrOptions)) {
    strides = stridesOrOptions;
    options = maybeOptions;
  } else {
    options = stridesOrOptions as SliceOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), [...start], [...stop]];
  if (strides) {
    args.push([...strides]);
  }
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.slice(...args));
}

export function astype(a: ScalarOrArray, dtype: DTypeLike, options?: StreamOptions): MLXArray {
  const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
  const args: any[] = [toNativeScalarOrArray(a), d];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.astype(...args));
}

export function pad(
  a: ScalarOrArray,
  padWidths: [number, number][],
  constantValueOrOptions: ScalarOrArray | PadOptions = 0,
  maybeOptions?: PadOptions,
): MLXArray {
  let constantValue: ScalarOrArray = 0;
  let options: PadOptions | undefined;
  if (constantValueOrOptions && typeof constantValueOrOptions === 'object' && !('toNative' in (constantValueOrOptions as any)) && !Array.isArray(constantValueOrOptions)) {
    options = constantValueOrOptions as PadOptions;
  } else {
    constantValue = constantValueOrOptions as ScalarOrArray;
    options = maybeOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), padWidths, toNativeScalarOrArray(constantValue)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.pad(...args));
}

export function slice_update(
  a: ScalarOrArray,
  update: ScalarOrArray,
  start: readonly number[],
  stop: readonly number[],
  stridesOrOptions?: readonly number[] | SliceUpdateOptions,
  maybeOptions?: SliceUpdateOptions,
): MLXArray {
  let strides: readonly number[] | undefined;
  let options: SliceUpdateOptions | undefined;
  if (Array.isArray(stridesOrOptions)) {
    strides = stridesOrOptions;
    options = maybeOptions;
  } else {
    options = stridesOrOptions as SliceUpdateOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(update), [...start], [...stop]];
  if (strides) {
    args.push([...strides]);
  }
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.slice_update(...args));
}

// ─────────────────────────────── SHAPE OPS ─────────────────────────────────

export function reshape(a: ScalarOrArray, shape: readonly number[], options?: ReshapeOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeShapeInput(shape)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.reshape(...args));
}

export function flatten(a: ScalarOrArray, start_axis = 0, end_axis = -1, options?: FlattenOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), start_axis, end_axis];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.flatten(...args));
}

export function transpose(a: ScalarOrArray, axesOrOptions?: readonly number[] | TransposeOptions, maybeOptions?: TransposeOptions): MLXArray {
  let axes: readonly number[] | undefined;
  let options: TransposeOptions | undefined;
  if (Array.isArray(axesOrOptions)) {
    axes = axesOrOptions;
    options = maybeOptions;
  } else {
    options = axesOrOptions as TransposeOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a)];
  if (axes) args.push(normalizeAxes(axes));
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.transpose(...args));
}

export const permute_dims = transpose;

export function moveaxis(a: ScalarOrArray, source: number | readonly number[], destination: number | readonly number[], options?: MoveAxisOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisSpec(source, 'source'), normalizeAxisSpec(destination, 'destination')];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.moveaxis(...args));
}

export function swapaxes(a: ScalarOrArray, axis1: number, axis2: number, options?: SwapAxesOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), axis1, axis2];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.swapaxes(...args));
}

export function expand_dims(a: ScalarOrArray, axis: number | readonly number[], options?: ExpandDimsOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeAxisSpec(axis, 'axis')];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.expand_dims(...args));
}

export function squeeze(a: ScalarOrArray, axis?: number | readonly number[] | null, options?: SqueezeOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  if (axis != null) args.push(normalizeAxisSpec(axis, 'axis'));
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.squeeze(...args));
}

export function concatenate(arrays: readonly ScalarOrArray[], axisOrOptions?: number | null | ConcatenateOptions, maybeOptions?: ConcatenateOptions): MLXArray {
  let axis: number | null = 0;
  let options: ConcatenateOptions | undefined;
  if (typeof axisOrOptions === 'number' || axisOrOptions === null) {
    axis = axisOrOptions;
    options = maybeOptions;
  } else if (axisOrOptions && typeof axisOrOptions === 'object') {
    options = axisOrOptions as ConcatenateOptions;
    axis = (options as any).axis ?? 0;
  } else {
    options = maybeOptions;
  }
  const args: any[] = [arrays.map(toNativeScalarOrArray), axis ?? 0];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.concatenate(...args));
}

export const concat = concatenate;

export function stack(arrays: readonly ScalarOrArray[], axis = 0, options?: StreamOptions): MLXArray {
  const args: any[] = [arrays.map(toNativeScalarOrArray), axis];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.stack(...args));
}

export function broadcast_to(a: ScalarOrArray, shape: readonly number[], options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), normalizeShapeInput(shape)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.broadcast_to(...args));
}

export function broadcast_arrays(arrays: readonly ScalarOrArray[], options?: StreamOptions): MLXArray[] {
  const args: any[] = [arrays.map(toNativeScalarOrArray)];
  appendStreamArg(args, options?.stream);
  const result: any[] = addon.broadcast_arrays(...args);
  return result.map(MLXArray.fromHandle);
}

export function atleast_1d(...arrays: ScalarOrArray[]): MLXArray | MLXArray[] {
  const result: any[] = addon.atleast_1d(arrays.map(toNativeScalarOrArray));
  const wrapped = result.map(MLXArray.fromHandle);
  return wrapped.length === 1 ? wrapped[0] : wrapped;
}

export function atleast_2d(...arrays: ScalarOrArray[]): MLXArray | MLXArray[] {
  const result: any[] = addon.atleast_2d(arrays.map(toNativeScalarOrArray));
  const wrapped = result.map(MLXArray.fromHandle);
  return wrapped.length === 1 ? wrapped[0] : wrapped;
}

export function atleast_3d(...arrays: ScalarOrArray[]): MLXArray | MLXArray[] {
  const result: any[] = addon.atleast_3d(arrays.map(toNativeScalarOrArray));
  const wrapped = result.map(MLXArray.fromHandle);
  return wrapped.length === 1 ? wrapped[0] : wrapped;
}

export function meshgrid(...arrays: (ScalarOrArray | MeshgridOptions)[]): MLXArray[] {
  let options: MeshgridOptions | undefined;
  let ops = arrays;
  if (arrays.length > 0) {
    const last = arrays[arrays.length - 1];
    if (last && typeof last === 'object' && !('toNative' in (last as any)) && !Array.isArray(last)) {
      options = last as MeshgridOptions;
      ops = arrays.slice(0, -1);
    }
  }
  const args: any[] = [(ops as ScalarOrArray[]).map(toNativeScalarOrArray), options?.indexing ?? 'xy'];
  appendStreamArg(args, options?.stream);
  const result: any[] = addon.meshgrid(...args);
  return result.map(MLXArray.fromHandle);
}

export function broadcast_shapes(...shapes: (readonly number[])[]): number[] {
  return addon.broadcast_shapes(...shapes.map((s) => normalizeShapeInput(s)));
}

export function as_strided(
  a: ScalarOrArray,
  shape: readonly number[],
  strides: readonly number[],
  offsetOrOptions: number | AsStridedOptions = 0,
  maybeOptions?: AsStridedOptions,
): MLXArray {
  let offset = 0;
  let options: AsStridedOptions | undefined;
  if (typeof offsetOrOptions === 'number') {
    offset = offsetOrOptions;
    options = maybeOptions;
  } else {
    options = offsetOrOptions as AsStridedOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), normalizeShapeInput(shape), [...strides], offset];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.as_strided(...args));
}

export function number_of_elements(a: ScalarOrArray, options?: NumberOfElementsOptions): number {
  if (a && typeof a === 'object' && 'toNative' in (a as any)) {
      if (options?.axes || options?.inverted) {
          return addon.number_of_elements(toNativeScalarOrArray(a), options.axes ?? null, options.inverted ?? false);
      }
      return (a as any).shape.reduce((acc: number, dim: number) => acc * dim, 1);
  }
  return 1;
}

export function unflatten(a: ScalarOrArray, axis: number, shape: readonly number[], options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), axis, normalizeShapeInput(shape)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.unflatten(...args));
}

export function roll(a: ScalarOrArray, shift: number | readonly number[], axisOrOptions?: number | readonly number[] | RollOptions, maybeOptions?: RollOptions): MLXArray {
  let axis: number | readonly number[] | undefined;
  let options: RollOptions | undefined;
  if (typeof axisOrOptions === 'number' || Array.isArray(axisOrOptions)) {
    axis = axisOrOptions as number | readonly number[];
    options = maybeOptions;
  } else {
    options = axisOrOptions as RollOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), Array.isArray(shift) ? [...shift] : shift];
  if (axis !== undefined) args.push(Array.isArray(axis) ? [...axis] : axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.roll(...args));
}

export function view(a: ScalarOrArray, dtype: DTypeLike, options?: StreamOptions): MLXArray {
  const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
  const args: any[] = [toNativeScalarOrArray(a), d];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.view(...args));
}

export function contiguous(a: ScalarOrArray, options?: ContiguousOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.contiguous(...args));
}

export function split(a: ScalarOrArray, numOrIndices: number | readonly number[], axis = 0, options?: SplitOptions): MLXArray[] {
  const args: any[] = [toNativeScalarOrArray(a), Array.isArray(numOrIndices) ? [...numOrIndices] : numOrIndices, axis];
  appendStreamArg(args, options?.stream);
  const result: any[] = addon.split(...args);
  return result.map(MLXArray.fromHandle);
}

export function tile(a: ScalarOrArray, reps: number | readonly number[], options?: TileOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), Array.isArray(reps) ? [...reps] : [reps]];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.tile(...args));
}

export function repeat(a: ScalarOrArray, repeats: number, axis?: number, options?: RepeatOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), repeats];
  if (axis !== undefined) args.push(axis);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.repeat(...args));
}

export function sort(a: ScalarOrArray, axisOrOptions?: number | SortOptions, options?: SortOptions): MLXArray {
  let axis = -1;
  let opts = options;
  if (typeof axisOrOptions === 'number') {
    axis = axisOrOptions;
  } else if (axisOrOptions && typeof axisOrOptions === 'object') {
    axis = (axisOrOptions as any).axis ?? -1;
    opts = axisOrOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), axis];
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.sort(...args));
}

export function argsort(a: ScalarOrArray, axisOrOptions?: number | ArgSortOptions, options?: ArgSortOptions): MLXArray {
  let axis = -1;
  let opts = options;
  if (typeof axisOrOptions === 'number') {
    axis = axisOrOptions;
  } else if (axisOrOptions && typeof axisOrOptions === 'object') {
    axis = (axisOrOptions as any).axis ?? -1;
    opts = axisOrOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), axis];
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.argsort(...args));
}

export function diag(a: ScalarOrArray, k = 0, options?: DiagOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), k];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.diag(...args));
}

export function diagonal(a: ScalarOrArray, offset = 0, axis1 = 0, axis2 = 1, options?: DiagonalOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), offset, axis1, axis2];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.diagonal(...args));
}

export function topk(a: ScalarOrArray, k: number, axisOrOptions?: number | TopKOptions, options?: TopKOptions): MLXArray {
  let axis = -1;
  let opts = options;
  if (typeof axisOrOptions === 'number') {
    axis = axisOrOptions;
  } else if (axisOrOptions && typeof axisOrOptions === 'object') {
    axis = (axisOrOptions as any).axis ?? -1;
    opts = axisOrOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), k, axis];
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.topk(...args));
}

export function nan_to_num(a: ScalarOrArray, nanOrOptions?: number | NanToNumOptions, posinf?: number, neginf?: number, options?: NanToNumOptions): MLXArray {
  let nan = 0.0;
  let opts = options;
  if (typeof nanOrOptions === 'number') {
    nan = nanOrOptions;
  } else if (nanOrOptions && typeof nanOrOptions === 'object') {
    nan = (nanOrOptions as any).nan ?? 0.0;
    posinf = (nanOrOptions as any).posinf;
    neginf = (nanOrOptions as any).neginf;
    opts = nanOrOptions;
  }
  const args: any[] = [toNativeScalarOrArray(a), nan];
  if (posinf !== undefined) args.push(posinf);
  if (neginf !== undefined) args.push(neginf);
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.nan_to_num(...args));
}

export function arange(
  start: number,
  stop?: number,
  stepOrOptions?: number | ArangeOptions,
  dtypeOrOptions?: DTypeLike | ArangeOptions,
  maybeOptions?: ArangeOptions,
): MLXArray {
  let stopVal: number | undefined;
  let step = 1.0;
  let dtype: DTypeLike | undefined;
  let options: ArangeOptions | undefined;

  if (stop === undefined) {
    stopVal = start;
    start = 0.0;
    if (stepOrOptions && typeof stepOrOptions === 'object') {
        options = stepOrOptions as ArangeOptions;
        dtype = options.dtype;
    } else if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('key' in (dtypeOrOptions as any))) {
        options = dtypeOrOptions as ArangeOptions;
        dtype = options.dtype;
    }
  } else {
    stopVal = stop;
    if (typeof stepOrOptions === 'number') {
        step = stepOrOptions;
        if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('key' in (dtypeOrOptions as any))) {
            options = dtypeOrOptions as ArangeOptions;
            dtype = options.dtype;
        } else if (dtypeOrOptions) {
            dtype = dtypeOrOptions as DTypeLike;
            options = maybeOptions;
        } else {
            options = maybeOptions;
        }
    } else if (stepOrOptions === undefined) {
        if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('key' in (dtypeOrOptions as any))) {
            options = dtypeOrOptions as ArangeOptions;
            dtype = options.dtype;
        } else if (dtypeOrOptions) {
            dtype = dtypeOrOptions as DTypeLike;
            options = maybeOptions;
        }
    } else if (stepOrOptions && typeof stepOrOptions === 'object' && 'key' in (stepOrOptions as any)) {
        dtype = stepOrOptions as DTypeLike;
        options = maybeOptions;
    } else if (stepOrOptions && typeof stepOrOptions === 'object') {
        options = stepOrOptions as ArangeOptions;
        dtype = options.dtype;
    }
  }

  const args: any[] = [start, stopVal, step];
  if (dtype) {
    const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.arange(...args));
}

export function full(shape: readonly number[], value: ScalarOrArray, dtypeOrOptions?: DTypeLike | StreamOptions, options?: StreamOptions): MLXArray {
  let dtype: DTypeLike | undefined;
  let opts = options;
  if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('key' in (dtypeOrOptions as any))) {
      opts = dtypeOrOptions as StreamOptions;
  } else {
      dtype = dtypeOrOptions as DTypeLike;
  }
  const args: any[] = [normalizeShapeInput(shape), toNativeScalarOrArray(value)];
  if (dtype) {
    const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.full(...args));
}

export function zeros(shape: readonly number[], dtypeOrOptions?: DTypeLike | StreamOptions, options?: StreamOptions): MLXArray {
  let dtype: DTypeLike | undefined;
  let opts = options;
  if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('key' in (dtypeOrOptions as any))) {
      opts = dtypeOrOptions as StreamOptions;
  } else {
      dtype = dtypeOrOptions as DTypeLike;
  }
  const args: any[] = [normalizeShapeInput(shape)];
  if (dtype) {
    const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.zeros(...args));
}

export function ones(shape: readonly number[], dtypeOrOptions?: DTypeLike | StreamOptions, options?: StreamOptions): MLXArray {
  let dtype: DTypeLike | undefined;
  let opts = options;
  if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('key' in (dtypeOrOptions as any))) {
      opts = dtypeOrOptions as StreamOptions;
  } else {
      dtype = dtypeOrOptions as DTypeLike;
  }
  const args: any[] = [normalizeShapeInput(shape)];
  if (dtype) {
    const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.ones(...args));
}

export function array_equal(a: ScalarOrArray, b: ScalarOrArray, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.array_equal(...args));
}

export function isclose(
  a: ScalarOrArray,
  b: ScalarOrArray,
  rtolOrOptions?: number | CloseOptions,
  atol?: number,
  equal_nan?: boolean,
  options?: CloseOptions,
): MLXArray {
  let rtol = 1e-5;
  let opts = options;
  if (typeof rtolOrOptions === 'number') {
    rtol = rtolOrOptions;
  } else if (rtolOrOptions && typeof rtolOrOptions === 'object') {
    opts = rtolOrOptions;
    rtol = opts.rtol ?? 1e-5;
    atol = atol ?? opts.atol;
    equal_nan = equal_nan ?? opts.equal_nan;
  }
  const args: any[] = [
    toNativeScalarOrArray(a),
    toNativeScalarOrArray(b),
    rtol,
    atol ?? 1e-8,
    equal_nan ?? false,
  ];
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.isclose(...args));
}

export function allclose(
  a: ScalarOrArray,
  b: ScalarOrArray,
  rtolOrOptions?: number | CloseOptions,
  atol?: number,
  equal_nan?: boolean,
  options?: CloseOptions,
): MLXArray {
  let rtol = 1e-5;
  let opts = options;
  if (typeof rtolOrOptions === 'number') {
    rtol = rtolOrOptions;
  } else if (rtolOrOptions && typeof rtolOrOptions === 'object') {
    opts = rtolOrOptions;
    rtol = opts.rtol ?? 1e-5;
    atol = atol ?? opts.atol;
    equal_nan = equal_nan ?? opts.equal_nan;
  }
  const args: any[] = [
    toNativeScalarOrArray(a),
    toNativeScalarOrArray(b),
    rtol,
    atol ?? 1e-8,
    equal_nan ?? false,
  ];
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.allclose(...args));
}

export function tri(n: number, mOrOptions?: number | TriCreateOptions, kOrOptions: number | TriCreateOptions = 0, options?: TriCreateOptions): MLXArray {
  let m: number | undefined;
  let k = 0;
  let opts = options;

  if (typeof mOrOptions === 'number') {
    m = mOrOptions;
    if (typeof kOrOptions === 'number') {
        k = kOrOptions;
    } else {
        opts = kOrOptions as TriCreateOptions;
    }
  } else {
    opts = mOrOptions as TriCreateOptions;
  }

  const args: any[] = [n, m ?? n, k];
  if (opts?.dtype) {
    const d = (typeof opts.dtype === 'string') ? opts.dtype : (opts.dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.tri(...args));
}

export function tril(a: ScalarOrArray, k = 0, options?: TriOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), k];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.tril(...args));
}

export function triu(a: ScalarOrArray, k = 0, options?: TriOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), k];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.triu(...args));
}

export function eye(n: number, mOrOptions?: number | EyeOptions, kOrOptions: number | EyeOptions = 0, options?: EyeOptions): MLXArray {
  let m: number | undefined;
  let k = 0;
  let opts = options;

  if (typeof mOrOptions === 'number') {
    m = mOrOptions;
    if (typeof kOrOptions === 'number') {
        k = kOrOptions;
    } else {
        opts = kOrOptions as EyeOptions;
    }
  } else {
    opts = mOrOptions as EyeOptions;
  }

  const args: any[] = [n, m ?? n, k];
  if (opts?.dtype) {
    const d = (typeof opts.dtype === 'string') ? opts.dtype : (opts.dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.eye(...args));
}

export function identity(n: number, options?: IdentityOptions): MLXArray {
  return eye(n, n, 0, options);
}

export function linspace(start: number, stop: number, numOrOptions: number | LinspaceOptions = 50, options?: LinspaceOptions): MLXArray {
  let num = 50;
  let opts = options;
  if (typeof numOrOptions === 'number') {
    num = numOrOptions;
  } else {
    opts = numOrOptions;
  }
  const args: any[] = [start, stop, num];
  if (opts?.dtype) {
    const d = (typeof opts.dtype === 'string') ? opts.dtype : (opts.dtype as any).key;
    if (d) args.push(d);
  }
  appendStreamArg(args, opts?.stream);
  return MLXArray.fromHandle(addon.linspace(...args));
}

export function where(condition: ScalarOrArray, x: ScalarOrArray, y: ScalarOrArray, options?: WhereOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(condition), toNativeScalarOrArray(x), toNativeScalarOrArray(y)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.where(...args));
}

// ─────────────────────────────── DEVICE OPS ─────────────────────────────────

export function default_device(): Device {
  return addon.default_device();
}

export function set_default_device(device: DeviceLike): void {
  addon.set_default_device(normalizeDevice(device));
}

export function is_available(): boolean {
  if (addon.is_available) {
    return addon.is_available();
  }
  return true;
}

// ─────────────────────────────── MEMORY OPS ─────────────────────────────────

export function get_active_memory(): number {
  return addon.get_active_memory();
}

export function get_peak_memory(): number {
  return addon.get_peak_memory();
}

export function get_cache_memory(): number {
  return addon.get_cache_memory();
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

export function clear_cache(): void {
  addon.clear_cache();
}

export function set_wired_limit(limit: number): number {
  return addon.set_wired_limit(limit);
}

// ─────────────────────────────── FFT OPS ───────────────────────────────────

export namespace fft {
  export function fft(a: ScalarOrArray, n?: number, axis = -1, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), n ?? null, axis];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fft(...args));
  }

  export function ifft(a: ScalarOrArray, n?: number, axis = -1, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), n ?? null, axis];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifft(...args));
  }

  export function rfft(a: ScalarOrArray, n?: number, axis = -1, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), n ?? null, axis];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.rfft(...args));
  }

  export function irfft(a: ScalarOrArray, n?: number, axis = -1, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), n ?? null, axis];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.irfft(...args));
  }

  export function fft2(a: ScalarOrArray, s?: [number, number], axes: [number, number] = [-2, -1], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, [...axes]];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fft2(...args));
  }

  export function ifft2(a: ScalarOrArray, s?: [number, number], axes: [number, number] = [-2, -1], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, [...axes]];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifft2(...args));
  }

  export function rfft2(a: ScalarOrArray, s?: [number, number], axes: [number, number] = [-2, -1], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, [...axes]];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.rfft2(...args));
  }

  export function irfft2(a: ScalarOrArray, s?: [number, number], axes: [number, number] = [-2, -1], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, [...axes]];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.irfft2(...args));
  }

  export function fftn(a: ScalarOrArray, s?: number[], axes?: number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, axes ?? null];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fftn(...args));
  }

  export function ifftn(a: ScalarOrArray, s?: number[], axes?: number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, axes ?? null];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifftn(...args));
  }

  export function rfftn(a: ScalarOrArray, s?: number[], axes?: number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, axes ?? null];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.rfftn(...args));
  }

  export function irfftn(a: ScalarOrArray, s?: number[], axes?: number[], options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), s ?? null, axes ?? null];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.irfftn(...args));
  }

  export function fftshift(a: ScalarOrArray, axes?: AxisSpec, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axes) ?? null];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.fftshift(...args));
  }

  export function ifftshift(a: ScalarOrArray, axes?: AxisSpec, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), normalizeAxisArg(axes) ?? null];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fft.ifftshift(...args));
  }
}

export const fft_ns = fft;

// ─────────────────────────────── LINALG OPS ─────────────────────────────────

export namespace linalg {
  export function inv(a: ScalarOrArray, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.inv(...args));
  }

  export function pinv(a: ScalarOrArray, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.pinv(...args));
  }

  export function solve(a: ScalarOrArray, b: ScalarOrArray, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.solve(...args));
  }

  export function solve_triangular(a: ScalarOrArray, b: ScalarOrArray, upper = false, left = true, unit_triangular = false, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b), upper, left, unit_triangular];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.solve_triangular(...args));
  }

  export function cholesky(a: ScalarOrArray, upper = false, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), upper];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.cholesky(...args));
  }

  export function cholesky_inv(a: ScalarOrArray, upper = false, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), upper];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.cholesky_inv(...args));
  }

  export function tri_inv(a: ScalarOrArray, upper = false, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), upper];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.tri_inv(...args));
  }

  export function svd(a: ScalarOrArray, options?: StreamOptions): [MLXArray, MLXArray, MLXArray] {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    const result: any[] = addon.linalg.svd(...args);
    return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1]), MLXArray.fromHandle(result[2])];
  }

  export function qr(a: ScalarOrArray, options?: StreamOptions): [MLXArray, MLXArray] {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    const result: any[] = addon.linalg.qr(...args);
    return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1])];
  }

  export function lu(a: ScalarOrArray, options?: StreamOptions): [MLXArray, MLXArray, MLXArray] {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    const result: any[] = addon.linalg.lu(...args);
    return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1]), MLXArray.fromHandle(result[2])];
  }

  export function lu_factor(a: ScalarOrArray, options?: StreamOptions): [MLXArray, MLXArray] {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    const result: any[] = addon.linalg.lu_factor(...args);
    return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1])];
  }

  export function norm(a: ScalarOrArray, ord?: number | string | null, axis?: AxisSpec, keepdims = false, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), ord ?? null, normalizeAxisArg(axis) ?? null, keepdims];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.norm(...args));
  }

  export function eig(a: ScalarOrArray, options?: StreamOptions): [MLXArray, MLXArray] {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    const result: any[] = addon.linalg.eig(...args);
    return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1])];
  }

  export function eigvals(a: ScalarOrArray, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a)];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.eigvals(...args));
  }

  export function eigh(a: ScalarOrArray, UPLO = 'L', options?: StreamOptions): [MLXArray, MLXArray] {
    const args: any[] = [toNativeScalarOrArray(a), UPLO];
    appendStreamArg(args, options?.stream);
    const result: any[] = addon.linalg.eigh(...args);
    return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1])];
  }

  export function eigvalsh(a: ScalarOrArray, UPLO = 'L', options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), UPLO];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.eigvalsh(...args));
  }

  export function cross(a: ScalarOrArray, b: ScalarOrArray, axis = -1, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b), axis];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.linalg.cross(...args));
  }
}

// ─────────────────────────────── TRANSFORM OPS ────────────────────────────────

export const eval_op = addon.eval;
export const async_eval = addon.async_eval;
export const grad = addon.grad;
export const value_and_grad = addon.value_and_grad;
export const vjp = addon.vjp;
export const jvp = addon.jvp;
export const vmap = addon.vmap;
export const compile_fn = addon.compile;
export const enable_compile = addon.enable_compile;
export const disable_compile = addon.disable_compile;
export const checkpoint = addon.checkpoint;
export const export_to_dot = addon.export_to_dot;

// ─────────────────────────────── FAST OPS ───────────────────────────────────

/** Fast path ops (Metal optimized) */
export namespace fast {
  export function scaled_dot_product_attention(
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
  }

  export function rms_norm(x: MLXArray, weight: MLXArray | null, eps: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(x), weight ? toNativeHandle(weight) : null, eps];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fast.rms_norm(...args));
  }

  export function layer_norm(x: MLXArray, weight: MLXArray | null, bias: MLXArray | null, eps: number, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(x), weight ? toNativeHandle(weight) : null, bias ? toNativeHandle(bias) : null, eps];
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fast.layer_norm(...args));
  }

  export function rope(x: MLXArray, dims: number, traditional: boolean, base: number | null, scale: number, offset: number, freqs?: MLXArray, options?: StreamOptions): MLXArray {
    const args: any[] = [toNativeHandle(x), dims, traditional, base, scale, offset];
    if (freqs !== undefined) args.push(toNativeHandle(freqs));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.fast.rope(...args));
  }
}

/** Random number generation functions (mlx.core.random). */
export namespace random {
  export function seed(s: number): void {
    addon.random.seed(s);
  }

  export function key(s: number): MLXArray {
    return MLXArray.fromHandle(addon.random.key(s));
  }

  export function split(k: MLXArray, num?: number, options?: StreamOptions): MLXArray | [MLXArray, MLXArray] {
    const args: any[] = [toNativeHandle(k)];
    if (num !== undefined) args.push(num);
    appendStreamArg(args, options?.stream);
    const result = addon.random.split(...args);
    if (Array.isArray(result)) {
      return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1])];
    }
    return MLXArray.fromHandle(result);
  }

  export function normal(
    shape: readonly number[],
    dtypeOrOptions?: DTypeLike | NormalOptions,
    maybeOptions?: NormalOptions,
  ): MLXArray {
    const args: any[] = [normalizeShapeInput(shape)];
    let dtype: DTypeLike | undefined;
    let opts: NormalOptions | undefined;

    if (dtypeOrOptions && typeof dtypeOrOptions === 'object' && !('toNative' in (dtypeOrOptions as any)) && !('key' in (dtypeOrOptions as any))) {
      opts = dtypeOrOptions as NormalOptions;
      dtype = opts.dtype;
    } else {
      dtype = dtypeOrOptions as DTypeLike;
      opts = maybeOptions;
    }

    if (dtype !== undefined) {
      const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
      if (d) args.push(d);
    }
    if (opts?.loc !== undefined) args.push(toNativeScalarOrArray(opts.loc));
    if (opts?.scale !== undefined) args.push(toNativeScalarOrArray(opts.scale));
    if (opts?.key) args.push(toNativeHandle(opts.key));
    appendStreamArg(args, opts?.stream);
    return MLXArray.fromHandle(addon.random.normal(...args));
  }

  export function uniform(
    lowOrShape: number | readonly number[],
    highOrOptions?: number | (StreamOptions & { dtype?: DTypeLike }),
    shapeOrOptions?: readonly number[] | (StreamOptions & { dtype?: DTypeLike }),
    maybeOptions?: { dtype?: DTypeLike } & StreamOptions,
  ): MLXArray {
    let args: any[];
    let opts: any;

    if (typeof lowOrShape === 'number' && typeof highOrOptions === 'number') {
      const low = lowOrShape;
      const high = highOrOptions;
      const shape = normalizeShapeInput(shapeOrOptions as readonly number[]);
      opts = maybeOptions;
      args = [low, high, shape];
    } else {
      let shape: readonly number[];
      if (typeof lowOrShape === 'number') {
        shape = normalizeShapeInput([lowOrShape]);
      } else {
        shape = normalizeShapeInput(lowOrShape);
      }
      opts = highOrOptions;
      args = [shape];
    }
    if (opts?.dtype !== undefined) {
      const d = (typeof opts.dtype === 'string') ? opts.dtype : (opts.dtype as any).key;
      if (d) args.push(d);
    }
    appendStreamArg(args, opts?.stream);
    return MLXArray.fromHandle(addon.random.uniform(...args));
  }

  export function bernoulli(
    p?: number | MLXArray,
    shape?: readonly number[],
    options?: { key?: MLXArray; stream?: StreamLike },
  ): MLXArray {
    const args: any[] = [];
    if (p !== undefined) args.push(toNativeScalarOrArray(p));
    if (shape) args.push(normalizeShapeInput(shape));
    if (options?.key) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.bernoulli(...args));
  }

  export function randint(
    low: number | MLXArray,
    high: number | MLXArray,
    shape: readonly number[],
    options?: { dtype?: DTypeLike; key?: MLXArray; stream?: StreamLike },
  ): MLXArray {
    const args: any[] = [toNativeScalarOrArray(low), toNativeScalarOrArray(high), normalizeShapeInput(shape)];
    if (options?.dtype !== undefined) {
      const d = (typeof options.dtype === 'string') ? options.dtype : (options.dtype as any).key;
      if (d) args.push(d);
    }
    if (options?.key) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.randint(...args));
  }

  export function categorical(
    logits: ScalarOrArray,
    axisOrOptions?: number | { axis?: number; key?: MLXArray; stream?: StreamLike },
    options?: { key?: MLXArray; stream?: StreamLike },
  ): MLXArray {
    let axis = -1;
    let opts = options;
    if (typeof axisOrOptions === 'number') {
      axis = axisOrOptions;
    } else if (axisOrOptions && typeof axisOrOptions === 'object') {
      axis = axisOrOptions.axis ?? -1;
      opts = axisOrOptions;
    }
    const args: any[] = [toNativeScalarOrArray(logits), axis];
    if (opts?.key) args.push(toNativeHandle(opts.key));
    appendStreamArg(args, opts?.stream);
    return MLXArray.fromHandle(addon.random.categorical(...args));
  }

  export function permutation(
    x: ScalarOrArray,
    axisOrOptions?: number | { axis?: number; key?: MLXArray; stream?: StreamLike },
    options?: { key?: MLXArray; stream?: StreamLike },
  ): MLXArray {
    let axis = 0;
    let opts = options;
    if (typeof axisOrOptions === 'number') {
      axis = axisOrOptions;
    } else if (axisOrOptions && typeof axisOrOptions === 'object') {
      axis = axisOrOptions.axis ?? 0;
      opts = axisOrOptions;
    }

    if (typeof x === 'number' || typeof x === 'bigint') {
      const args: any[] = [Number(x)];
      if (opts?.key) args.push(toNativeHandle(opts.key));
      appendStreamArg(args, opts?.stream);
      return MLXArray.fromHandle(addon.random.permutation(...args));
    } else {
      const args: any[] = [toNativeScalarOrArray(x), axis];
      if (opts?.key) args.push(toNativeHandle(opts.key));
      appendStreamArg(args, opts?.stream);
      return MLXArray.fromHandle(addon.random.permutation(...args));
    }
  }

  export function gumbel(shape: readonly number[], options?: { dtype?: DTypeLike; key?: MLXArray; stream?: StreamLike }): MLXArray {
    const args: any[] = [normalizeShapeInput(shape)];
    if (options?.dtype) {
      const d = (typeof options.dtype === 'string') ? options.dtype : (options.dtype as any).key;
      if (d) args.push(d);
    }
    if (options?.key) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.gumbel(...args));
  }

  export function laplace(shape: readonly number[], options?: { dtype?: DTypeLike; key?: MLXArray; stream?: StreamLike }): MLXArray {
    const args: any[] = [normalizeShapeInput(shape)];
    if (options?.dtype) {
      const d = (typeof options.dtype === 'string') ? options.dtype : (options.dtype as any).key;
      if (d) args.push(d);
    }
    if (options?.key) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.laplace(...args));
  }

  export function truncated_normal(
    lower: number | MLXArray,
    upper: number | MLXArray,
    options?: { shape?: readonly number[]; dtype?: DTypeLike; key?: MLXArray; stream?: StreamLike },
  ): MLXArray {
    const args: any[] = [toNativeScalarOrArray(lower), toNativeScalarOrArray(upper)];
    if (options?.shape) args.push(normalizeShapeInput(options.shape));
    if (options?.dtype) {
      const d = (typeof options.dtype === 'string') ? options.dtype : (options.dtype as any).key;
      if (d) args.push(d);
    }
    if (options?.key) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.truncated_normal(...args));
  }

  export function multivariate_normal(
    mean: MLXArray,
    cov: MLXArray,
    shape: readonly number[],
    options?: { dtype?: DTypeLike; key?: MLXArray; stream?: StreamLike },
  ): MLXArray {
    const args: any[] = [toNativeHandle(mean), toNativeHandle(cov), normalizeShapeInput(shape)];
    if (options?.dtype) {
      const d = (typeof options.dtype === 'string') ? options.dtype : (options.dtype as any).key;
      if (d) args.push(d);
    }
    if (options?.key) args.push(toNativeHandle(options.key));
    appendStreamArg(args, options?.stream);
    return MLXArray.fromHandle(addon.random.multivariate_normal(...args));
  }
}

export function hadamard_transform(a: ScalarOrArray, scale = 1.0, options?: StreamOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), scale];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.hadamard_transform(...args));
}

export function partition(a: ScalarOrArray, kth: number, axis = -1, options?: PartitionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), kth, axis];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.partition(...args));
}

export function argpartition(a: ScalarOrArray, kth: number, axis = -1, options?: PartitionOptions): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), kth, axis];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.argpartition(...args));
}

export function block_masked_mm(
  a: ScalarOrArray,
  b: ScalarOrArray,
  blockSize = 64,
  maskOut?: MLXArray,
  maskIn?: MLXArray,
  options?: BlockMaskedMMOptions
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(a),
    toNativeScalarOrArray(b),
    blockSize,
    maskOut ? toNativeHandle(maskOut) : null,
    maskIn ? toNativeHandle(maskIn) : null,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.block_masked_mm(...args));
}

export function gather_mm(
  a: ScalarOrArray,
  b: ScalarOrArray,
  lhsIndices?: MLXArray,
  rhsIndices?: MLXArray,
  options?: GatherMMOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b)];
  args.push(lhsIndices ? toNativeHandle(lhsIndices) : null);
  args.push(rhsIndices ? toNativeHandle(rhsIndices) : null);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.gather_mm(...args));
}

export function segmented_mm(
  a: ScalarOrArray,
  b: ScalarOrArray,
  segmentIds: MLXArray,
  options?: StreamOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(b), toNativeHandle(segmentIds)];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.segmented_mm(...args));
}

export function quantize(
  a: ScalarOrArray,
  groupSize = 64,
  bits = 4,
  options?: QuantizeOptions,
): [MLXArray, MLXArray, MLXArray] {
  const args: any[] = [toNativeScalarOrArray(a), groupSize, bits];
  appendStreamArg(args, options?.stream);
  const result: any[] = addon.quantize(...args);
  return [MLXArray.fromHandle(result[0]), MLXArray.fromHandle(result[1]), MLXArray.fromHandle(result[2])];
}

export function dequantize(
  a: ScalarOrArray,
  scales: ScalarOrArray,
  biases: ScalarOrArray,
  groupSize = 64,
  bits = 4,
  options?: DequantizeOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(scales), toNativeScalarOrArray(biases), groupSize, bits];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.dequantize(...args));
}

export function quantized_matmul(
  x: ScalarOrArray,
  w: ScalarOrArray,
  scales: ScalarOrArray,
  biases: ScalarOrArray,
  transpose = true,
  groupSize = 64,
  bits = 4,
  options?: QuantizedMatmulOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(x),
    toNativeScalarOrArray(w),
    toNativeScalarOrArray(scales),
    toNativeScalarOrArray(biases),
    transpose,
    groupSize,
    bits,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.quantized_matmul(...args));
}

export function gather_qmm(
  x: ScalarOrArray,
  w: ScalarOrArray,
  scales: ScalarOrArray,
  biases: ScalarOrArray,
  lhsIndices?: MLXArray,
  rhsIndices?: MLXArray,
  transpose = true,
  groupSize = 64,
  bits = 4,
  options?: GatherQMMOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(x),
    toNativeScalarOrArray(w),
    toNativeScalarOrArray(scales),
    toNativeScalarOrArray(biases),
  ];
  args.push(lhsIndices ? toNativeHandle(lhsIndices) : null);
  args.push(rhsIndices ? toNativeHandle(rhsIndices) : null);
  args.push(transpose, groupSize, bits);
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.gather_qmm(...args));
}

// ─────────────────────────────── CONVOLUTIONS ──────────────────────────────

export function conv1d(
  input: ScalarOrArray,
  weight: ScalarOrArray,
  stride = 1,
  padding = 0,
  dilation = 1,
  groups = 1,
  options?: ConvOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(input),
    toNativeScalarOrArray(weight),
    stride,
    padding,
    dilation,
    groups,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv1d(...args));
}

export function conv2d(
  input: ScalarOrArray,
  weight: ScalarOrArray,
  stride: number | [number, number] = 1,
  padding: number | [number, number] = 0,
  dilation: number | [number, number] = 1,
  groups = 1,
  options?: ConvOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(input),
    toNativeScalarOrArray(weight),
    Array.isArray(stride) ? [...stride] : stride,
    Array.isArray(padding) ? [...padding] : padding,
    Array.isArray(dilation) ? [...dilation] : dilation,
    groups,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv2d(...args));
}

export function conv3d(
  input: ScalarOrArray,
  weight: ScalarOrArray,
  stride: number | [number, number, number] = 1,
  padding: number | [number, number, number] = 0,
  dilation: number | [number, number, number] = 1,
  groups = 1,
  options?: ConvOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(input),
    toNativeScalarOrArray(weight),
    Array.isArray(stride) ? [...stride] : stride,
    Array.isArray(padding) ? [...padding] : padding,
    Array.isArray(dilation) ? [...dilation] : dilation,
    groups,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv3d(...args));
}

export function conv_general(
  input: ScalarOrArray,
  weight: ScalarOrArray,
  stride: number | number[] = 1,
  padding: number | number[] | [number[], number[]] = 0,
  dilation: number | number[] = 1,
  groups = 1,
  flip = false,
  options?: ConvGeneralOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(input),
    toNativeScalarOrArray(weight),
    Array.isArray(stride) ? [...stride] : stride,
    Array.isArray(padding)
      ? Array.isArray(padding[0])
        ? [[...(padding[0] as number[])], [...(padding[1] as number[])]]
        : [...(padding as number[])]
      : padding,
    Array.isArray(dilation) ? [...dilation] : dilation,
    groups,
    flip,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_general(...args));
}

export function conv_transpose1d(
  input: ScalarOrArray,
  weight: ScalarOrArray,
  stride = 1,
  padding = 0,
  dilation = 1,
  groups = 1,
  options?: ConvTranspose1dOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(input),
    toNativeScalarOrArray(weight),
    stride,
    padding,
    dilation,
    groups,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_transpose1d(...args));
}

export function conv_transpose2d(
  input: ScalarOrArray,
  weight: ScalarOrArray,
  stride: number | [number, number] = 1,
  padding: number | [number, number] = 0,
  dilation: number | [number, number] = 1,
  groups = 1,
  options?: ConvTranspose2dOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(input),
    toNativeScalarOrArray(weight),
    Array.isArray(stride) ? [...stride] : stride,
    Array.isArray(padding) ? [...padding] : padding,
    Array.isArray(dilation) ? [...dilation] : dilation,
    groups,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_transpose2d(...args));
}

export function conv_transpose3d(
  input: ScalarOrArray,
  weight: ScalarOrArray,
  stride: number | [number, number, number] = 1,
  padding: number | [number, number, number] = 0,
  dilation: number | [number, number, number] = 1,
  groups = 1,
  options?: ConvTranspose3dOptions,
): MLXArray {
  const args: any[] = [
    toNativeScalarOrArray(input),
    toNativeScalarOrArray(weight),
    Array.isArray(stride) ? [...stride] : stride,
    Array.isArray(padding) ? [...padding] : padding,
    Array.isArray(dilation) ? [...dilation] : dilation,
    groups,
  ];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.conv_transpose3d(...args));
}

export function convolve(
  a: ScalarOrArray,
  v: ScalarOrArray,
  mode: 'full' | 'valid' | 'same' = 'full',
  options?: StreamOptions,
): MLXArray {
  const args: any[] = [toNativeScalarOrArray(a), toNativeScalarOrArray(v), mode];
  appendStreamArg(args, options?.stream);
  return MLXArray.fromHandle(addon.convolve(...args));
}

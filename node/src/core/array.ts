import addon from '../internal/addon';
import type { DTypeKey } from './dtype';

export type DType = DTypeKey;

type NumericArray = readonly (number | bigint)[];
type SupportedTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array;

function normalizeShape(shape: readonly number[]): number[] {
  if (!Array.isArray(shape)) {
    throw new Error('Shape must be an array');
  }
  return shape.map((dim, index) => {
    if (!Number.isFinite(dim) || dim < 0) {
      throw new Error(`Invalid dimension at axis ${index}: ${dim}`);
    }
    return Math.trunc(dim);
  });
}

function elementCount(shape: readonly number[]): number {
  if (shape.length === 0) {
    return 1;
  }
  return shape.reduce((acc, dim) => acc * dim, 1);
}

function isTypedArray(input: unknown): input is SupportedTypedArray {
  return ArrayBuffer.isView(input) && !(input instanceof DataView);
}

function inferDTypeFromTypedArray(array: SupportedTypedArray): DType {
  if (array instanceof Float32Array) return 'float32';
  if (array instanceof Float64Array) return 'float64';
  if (array instanceof Int8Array) return 'int8';
  if (array instanceof Uint8Array || array instanceof Uint8ClampedArray)
    return 'uint8';
  if (array instanceof Int16Array) return 'int16';
  if (array instanceof Uint16Array) return 'uint16';
  if (array instanceof Int32Array) return 'int32';
  if (array instanceof Uint32Array) return 'uint32';
  if (array instanceof BigInt64Array) return 'int64';
  if (array instanceof BigUint64Array) return 'uint64';
  throw new Error('Unsupported TypedArray input');
}

function createTypedArrayFromNumbers(
  data: NumericArray,
  dtype: DType,
  elementCount: number,
): SupportedTypedArray {
  if (dtype === 'complex64') {
    if (data.length !== elementCount * 2) {
      throw new Error('Complex64 data length must be elementCount * 2');
    }
    const output = new Float32Array(data.length);
    data.forEach((value, index) => {
      output[index] = Number(value);
    });
    return output;
  }

  if (data.length !== elementCount) {
    throw new Error('Data length does not match the provided shape');
  }

  switch (dtype) {
    case 'float32': {
      const output = new Float32Array(elementCount);
      data.forEach((value, index) => {
        output[index] = Number(value);
      });
      return output;
    }
    case 'float64': {
      const output = new Float64Array(elementCount);
      data.forEach((value, index) => {
        output[index] = Number(value);
      });
      return output;
    }
    case 'int8': {
      const output = new Int8Array(elementCount);
      data.forEach((value, index) => {
        output[index] = Number(value);
      });
      return output;
    }
    case 'uint8':
    case 'bool': {
      const output = new Uint8Array(elementCount);
      data.forEach((value, index) => {
        const n = Number(value);
        output[index] = dtype === 'bool' ? (n ? 1 : 0) : n;
      });
      return output;
    }
    case 'int16': {
      const output = new Int16Array(elementCount);
      data.forEach((value, index) => {
        output[index] = Number(value);
      });
      return output;
    }
    case 'uint16': {
      const output = new Uint16Array(elementCount);
      data.forEach((value, index) => {
        output[index] = Number(value);
      });
      return output;
    }
    case 'int32': {
      const output = new Int32Array(elementCount);
      data.forEach((value, index) => {
        output[index] = Number(value);
      });
      return output;
    }
    case 'uint32': {
      const output = new Uint32Array(elementCount);
      data.forEach((value, index) => {
        output[index] = Number(value);
      });
      return output;
    }
    case 'int64': {
      const output = new BigInt64Array(elementCount);
      data.forEach((value, index) => {
        output[index] = BigInt(value);
      });
      return output;
    }
    case 'uint64': {
      const output = new BigUint64Array(elementCount);
      data.forEach((value, index) => {
        const big = BigInt(value);
        if (big < 0n) {
          throw new Error('uint64 values must be non-negative');
        }
        output[index] = big;
      });
      return output;
    }
    case 'float16':
    case 'bfloat16':
      throw new Error(
        `${dtype} requires a Uint16Array input. Provide encoded raw values.`,
      );
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

function castTypedArrayToDType(
  array: SupportedTypedArray,
  dtype: DType,
  elementCount: number,
): SupportedTypedArray {
  const currentDType = inferDTypeFromTypedArray(array);
  if (currentDType === dtype) {
    if (dtype === 'complex64' && array.length !== elementCount * 2) {
      throw new Error('Complex64 data length must be elementCount * 2');
    }
    if (dtype !== 'complex64' && array.length !== elementCount) {
      throw new Error('Data length does not match the provided shape');
    }
    return array;
  }
  const numericArray = Array.from(array as unknown as number[]);
  return createTypedArrayFromNumbers(numericArray, dtype, elementCount);
}

function ensureTypedArray(
  data: SupportedTypedArray | NumericArray,
  dtype: DType,
  elements: number,
): SupportedTypedArray {
  if (isTypedArray(data)) {
    return castTypedArrayToDType(data, dtype, elements);
  }
  return createTypedArrayFromNumbers(data, dtype, elements);
}

export type ArrayElement = number | boolean | [number, number];

export class MLXArray {
  private readonly handle: any;

  private constructor(handle: any) {
    this.handle = handle;
  }

  static fromHandle(handle: any): MLXArray {
    return new MLXArray(handle);
  }

  static from(
    data: SupportedTypedArray | NumericArray,
    shape: readonly number[],
    dtype: DType = 'float32',
  ): MLXArray {
    const normalizedShape = normalizeShape(shape);
    const elements = elementCount(normalizedShape);
    const typed = ensureTypedArray(data, dtype, elements);
    const handle = addon.Array.fromTypedArray(typed, normalizedShape, dtype);
    return MLXArray.fromHandle(handle);
  }

  static fromFloat32(
    data: Float32Array | readonly number[],
    shape: readonly number[],
  ): MLXArray {
    return MLXArray.from(data, shape, 'float32');
  }

  get shape(): number[] {
    return globalThis.Array.from(this.handle.shape() as number[]);
  }

  get dtype(): DType {
    return this.handle.dtype() as DType;
  }

  eval(): this {
    this.handle.eval();
    return this;
  }

  toTypedArray(): SupportedTypedArray {
    return this.handle.toTypedArray();
  }

  toFloat32Array(): Float32Array {
    if (this.dtype !== 'float32') {
      throw new Error('Array dtype is not float32');
    }
    return this.handle.toFloat32Array();
  }

  toNative(): any {
    return this.handle;
  }

  toArray(): ArrayElement[] {
    const data = this.toTypedArray();
    switch (this.dtype) {
      case 'bool':
        return Array.from(data as Uint8Array, (value) => value !== 0);
      case 'complex64': {
        const values = data as Float32Array;
        const output: ArrayElement[] = [];
        for (let i = 0; i < values.length; i += 2) {
          output.push([values[i], values[i + 1]]);
        }
        return output;
      }
      default:
        return Array.from(data as ArrayLike<number>);
    }
  }
}

export function array(
  data: SupportedTypedArray | NumericArray,
  shape: readonly number[],
  dtype: DType = 'float32',
): MLXArray {
  return MLXArray.from(data, shape, dtype);
}

export const normalizeShapeInput = (shape: readonly number[]): number[] =>
  Array.from(shape, (dim) => {
    if (!Number.isFinite(dim)) {
      throw new Error('Shape entries must be finite numbers');
    }
    return Math.trunc(dim);
  });

export function zeros(
  shape: readonly number[],
  dtype: DType = 'float32',
): MLXArray {
  return MLXArray.fromHandle(addon.zeros(normalizeShapeInput(shape), dtype));
}

export function ones(
  shape: readonly number[],
  dtype: DType = 'float32',
): MLXArray {
  return MLXArray.fromHandle(addon.ones(normalizeShapeInput(shape), dtype));
}

export function full(
  shape: readonly number[],
  value: number,
  dtype: DType = 'float32',
): MLXArray {
  return MLXArray.fromHandle(
    addon.full(normalizeShapeInput(shape), value, dtype),
  );
}

export function zeros_like(base: MLXArray): MLXArray {
  return MLXArray.fromHandle(addon.zeros_like(base.toNative()));
}

export function ones_like(base: MLXArray): MLXArray {
  return MLXArray.fromHandle(addon.ones_like(base.toNative()));
}

export default MLXArray;

import addon from '../internal/addon';
import type { DType, DTypeLike } from './dtype';
import { toDtypeObject } from './dtype';

/** @internal Symbol to identify MLXArray instances without circular dependencies. */
export const MLX_ARRAY_SYMBOL = Symbol.for('mlx.MLXArray');

/** Supported data types for mx.array construction. */
export type SupportedTypedArray =
  | Float32Array
  | Int32Array
  | Int16Array
  | Int8Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | BigInt64Array
  | BigUint64Array;

/** A single element value as returned by `MLXArray.toArray()`. */
export type ArrayElement = number | boolean | bigint | [number, number];

/**
 * An MLX multidimensional array.
 *
 * This class wraps a native `mlx::core::array` and provides zero-copy
 * views into its memory via `toTypedArray()`.
 *
 * For performance, prefer using the factory functions:
 *
 * 1. **Immediate** — `mx.array(data, shape?, dtype?)`.  Accepts a `TypedArray`
 *    and an optional shape.  Memory is copied once into the MLX allocator.
 *    ```typescript
 *    const a = mx.array(new Float32Array([1,2,3,4])); // 1D float32
 *    const c = mx.array(new Int32Array([1,2,3,4]), [2,2]); // 2D [2,2] int32
 *    ```
 *
 * 2. **Small-data escape hatch** — `mx.from_js_array(number[], dtype?)`.
 *    For small constant arrays (e.g., normalising weights).
 *
 * 3. **Streaming builder** — `mx.array_builder(dtype, shape)`.  Pre-allocates
 *    one MLX buffer for the full tensor.  The caller feeds rows one at a time
 *    via `builder.append_row(TypedArray)` and finalises with `builder.build()`.
 */
export class MLXArray {
  /** @internal Opaque native handle; never expose outside this module. */
  private readonly handle: any;

  private constructor(handle: any) {
    this.handle = handle;
    (this as any)[MLX_ARRAY_SYMBOL] = true;
  }

  /** @internal Create a wrapper from a native handle. */
  static fromHandle(handle: any): MLXArray {
    return new MLXArray(handle);
  }

  /** The array's shape as a JS array of non-negative integers. */
  get shape(): number[] {
    return globalThis.Array.from(this.handle.shape() as number[]);
  }

  /** The number of dimensions in the array. */
  get ndim(): number {
    return this.handle.shape().length;
  }

  /** The total number of elements in the array. */
  get size(): number {
    return this.handle.shape().reduce((a: number, b: number) => a * b, 1);
  }

  /** The array's element dtype as a string key. */
  get dtype(): DType {
    return this.handle.dtype().key as DType;
  }

  /** Return the underlying native handle. */
  toNative(): any {
    return this.handle;
  }

  /**
   * Cast the array to the given data type.
   */
  astype(dtype: DTypeLike, options?: { stream?: any }): MLXArray {
    const d = (typeof dtype === 'string') ? dtype : (dtype as any).key;
    const args: any[] = [this.handle, d];
    if (options?.stream) {
        // Direct addon call to avoid circular dependency with ops.ts
        const { toNativeStreamArgument } = require('./stream');
        args.push(toNativeStreamArgument(options.stream));
    }
    return MLXArray.fromHandle(addon.astype(...args));
  }

  /**
   * Materialise the tensor as a plain JS array.
   *
   * @remarks This allocates an intermediate TypedArray view (zero-copy) and
   * then converts it to a plain `ArrayElement[]`.  Intended for **debugging
   * and small tensors only**.  Use `toTypedArray()` directly when you need
   * to inspect raw numeric values.
   */
  toArray(): ArrayElement[] {
    const shape = this.shape;
    if (shape.length === 0) {
      const data = this.toTypedArray();
      if (this.dtype === 'bool') return [data[0] !== 0];
      if (this.dtype === 'complex64') return [[(data as Float32Array)[0], (data as Float32Array)[1]]];
      return [data[0] as number];
    }
    // Check if any dimension is zero
    if (shape.some((d) => d === 0)) {
      return [];
    }
    const data = this.toTypedArray();
    let flat: any[];

    switch (this.dtype) {
      case 'float16':
      case 'bfloat16':
        flat = globalThis.Array.from(this.astype('float32').toTypedArray() as Float32Array);
        break;
      case 'bool':
        flat = globalThis.Array.from(data as Uint8Array, (v) => v !== 0);
        break;
      case 'complex64': {
        const f = data as Float32Array;
        const out: any[] = [];
        for (let i = 0; i < f.length; i += 2) {
          out.push([f[i], f[i + 1]]);
        }
        flat = out;
        break;
      }
      default:
        flat = globalThis.Array.from(data as ArrayLike<number>);
    }

    if (shape.length === 1) {
        return flat;
    }

    // Helper to nest a flat array according to a shape
    function nest(flatArray: any[], dims: number[]): any[] {
        if (dims.length === 1) {
            return flatArray;
        }
        const result = [];
        const stride = dims.slice(1).reduce((a, b) => a * b, 1);
        for (let i = 0; i < dims[0]; i++) {
            result.push(nest(flatArray.slice(i * stride, (i + 1) * stride), dims.slice(1)));
        }
        return result;
    }

    return nest(flat, shape);
  }

  /**
   * Return a zero-copy TypedArray view into the array's memory.
   *
   * @remarks This is the fastest way to access MLX data from JavaScript.
   * The returned view points directly to the underlying MLX malloc'd memory.
   */
  toTypedArray(): SupportedTypedArray {
    const data = this.handle.toTypedArray();
    const { type } = this.handle.dtype();

    switch (this.dtype) {
      case 'float32':
        return new Float32Array(data);
      case 'int32':
        return new Int32Array(data);
      case 'int16':
        return new Int16Array(data);
      case 'int8':
        return new Int8Array(data);
      case 'uint32':
        return new Uint32Array(data);
      case 'uint16':
        return new Uint16Array(data);
      case 'uint8':
        return new Uint8Array(data);
      case 'int64':
        return new BigInt64Array(data);
      case 'uint64':
        return new BigUint64Array(data);
      case 'bool':
        return new Uint8Array(data);
      case 'complex64':
        return new Float32Array(data);
      case 'float16':
      case 'bfloat16':
        // Fallback for types without native TypedArray: return as raw bits (uint16)
        return new Uint16Array(data);
      default:
        throw new TypeError(`toTypedArray: Unsupported dtype: ${this.dtype}`);
    }
  }

  eval(): void {
    this.handle.eval();
  }

  toString(): string {
    return `array(shape=[${this.shape}], dtype=${this.dtype})`;
  }
}

// ---------------------------------------------------------------------------
// normalizeShapeInput (part of public API, used by zeros/ones/etc.)
// ---------------------------------------------------------------------------

/**
 * Standardise shape inputs from various JS formats to `number[]`.
 */
export function normalizeShapeInput(shape: number | readonly number[]): number[] {
  return typeof shape === 'number' ? [shape] : globalThis.Array.from(shape);
}

// ---------------------------------------------------------------------------
// from_js_array() — explicit small-data escape hatch
// ---------------------------------------------------------------------------

/** Maximum number of elements permitted via `from_js_array()`. */
const FROM_JS_ARRAY_MAX_ELEMENTS = 1024;

/**
 * Create an MLXArray from a potentially nested JS array.
 *
 * @remarks This function is limited to small arrays (< 1024 elements) and
 * uses a slow recursive scan.  For large data, use `mx.array(TypedArray)`.
 */
export function from_js_array(
  data: any[],
  dtype?: DTypeLike,
  shape?: readonly number[],
): MLXArray {
  const dtypeObj = toDtypeObject(dtype);
  const handle = addon.from_js_array(
    data,
    dtypeObj,
    shape ? normalizeShapeInput(shape) : undefined,
  );
  return MLXArray.fromHandle(handle);
}

/**
 * An alias for mx.array, following MLX Python naming.
 */
export const asarray = array;

/**
 * Create an MLXArray from a TypedArray or a scalar.
 */
export function array(
  data: SupportedTypedArray | number | boolean | bigint | any[],
  shapeOrDtype?: readonly number[] | DTypeLike,
  dtype?: DTypeLike,
): MLXArray {
  let resolvedShape: readonly number[] | undefined;
  let resolvedDtype: DTypeLike | undefined;

  if (globalThis.Array.isArray(shapeOrDtype)) {
    resolvedShape = shapeOrDtype;
    resolvedDtype = dtype;
  } else if (typeof shapeOrDtype === 'string' || (shapeOrDtype && typeof shapeOrDtype === 'object' && 'key' in (shapeOrDtype as any))) {
    resolvedDtype = shapeOrDtype as DTypeLike;
  } else {
    resolvedDtype = dtype;
  }

  const dtypeObj = toDtypeObject(resolvedDtype);

  // Scalar path
  if (
    typeof data === 'number' ||
    typeof data === 'boolean' ||
    typeof data === 'bigint'
  ) {
    const args: any[] = [data];
    if (dtypeObj !== undefined) args.push(dtypeObj);
    return MLXArray.fromHandle(addon.array(data, ...args.slice(1)));
  }

  // TypedArray path
  if (isTypedArray(data)) {
    const finalShape = resolvedShape !== undefined
      ? normalizeShapeInput(resolvedShape)
      : [(data as any).length];
    const args: any[] = [data, finalShape];
    if (dtypeObj !== undefined) args.push(dtypeObj);
    return MLXArray.fromHandle(addon.array(...args));
  }

  // Nested JS array path
  if (globalThis.Array.isArray(data)) {
    return from_js_array(data, resolvedDtype, resolvedShape);
  }

  throw new TypeError(
    'mx.array(): data must be a scalar (number | boolean | bigint), a ' +
    'TypedArray, or a nested list.  For large data, consider mx.array_builder().',
  );
}

function isTypedArray(obj: any): obj is SupportedTypedArray {
  return (
    obj instanceof Float32Array ||
    obj instanceof Int32Array ||
    obj instanceof Int16Array ||
    obj instanceof Int8Array ||
    obj instanceof Uint32Array ||
    obj instanceof Uint16Array ||
    obj instanceof Uint8Array ||
    obj instanceof BigInt64Array ||
    obj instanceof BigUint64Array
  );
}

/**
 * A builder for large MLXArrays from chunks.
 */
export class ArrayBuilder {
  private readonly handle: any;

  constructor(dtype: DTypeLike, shape: readonly number[]) {
    this.handle = new addon.ArrayBuilder(
      toDtypeObject(dtype),
      normalizeShapeInput(shape),
    );
  }

  append_row(data: SupportedTypedArray): void {
    this.handle.append_row(data);
  }

  build(): MLXArray {
    return MLXArray.fromHandle(this.handle.build());
  }
}

export function array_builder(
  dtype: DTypeLike,
  shape: readonly number[],
): ArrayBuilder {
  return new ArrayBuilder(dtype, shape);
}

/**
 * Return an array of given shape and type, filled with zeros.
 */
export function zeros(
  shape: number[],
  dtype: DTypeLike = 'float32',
): MLXArray {
  return MLXArray.fromHandle(
    addon.zeros(normalizeShapeInput(shape), toDtypeObject(dtype)),
  );
}

/**
 * Return an array of zeros with the same shape and type as the given array.
 */
export function zeros_like(a: MLXArray): MLXArray {
  return MLXArray.fromHandle(addon.zeros_like(a.toNative()));
}

/**
 * Return an array of given shape and type, filled with ones.
 */
export function ones(
  shape: number[],
  dtype: DTypeLike = 'float32',
): MLXArray {
  return MLXArray.fromHandle(
    addon.ones(normalizeShapeInput(shape), toDtypeObject(dtype)),
  );
}

/**
 * Return an array of ones with the same shape and type as the given array.
 */
export function ones_like(a: MLXArray): MLXArray {
  return MLXArray.fromHandle(addon.ones_like(a.toNative()));
}

/**
 * Return an array of given shape and type, filled with the given value.
 */
export function full(
  shape: number[],
  vals: number | boolean | bigint | MLXArray,
  dtype?: DTypeLike,
): MLXArray {
  const nativeVals = vals instanceof MLXArray ? vals.toNative() : vals;
  return MLXArray.fromHandle(
    addon.full(
      normalizeShapeInput(shape),
      nativeVals,
      toDtypeObject(dtype),
    ),
  );
}

export default MLXArray;

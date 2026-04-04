/**
 * @fileoverview MLX array construction, type definitions, and data-access
 * utilities for the Node.js binding.
 *
 * ## Memory model
 *
 * All large data must enter the system through one of three approved channels:
 *
 * 1. **File load** — `mx.load(path)`.  The C++ side reads the file into
 *    `mlx::core::allocator::malloc` memory directly.  JS never sees the bytes.
 *
 * 2. **Scalar / small TypedArray** — `mx.array(scalar)` or
 *    `mx.array(typedArray, shape?, dtype?)`.  Caller must provide data that is
 *    already in a TypedArray; the C++ bridge performs a single `memcpy` from
 *    the V8 backing-store into an MLX-owned buffer.  Use this path only for
 *    small, size-bounded tensors (embeddings, kernel weights, etc.).
 *
 * 3. **Streaming builder** — `mx.array_builder(dtype, shape)`.  Pre-allocates
 *    one MLX buffer for the full tensor.  The caller feeds rows one at a time
 *    via `builder.append_row(TypedArray)` and finalises with `builder.build()`.
 *    Each `append_row` is a bounded `memcpy` of one row.  No JS heap objects
 *    accumulate across iterations.
 *
 * **Do not** pass a plain `number[]` to the `array()` factory for data you
 * cannot fit on the V8 heap.  Use `from_js_array()` only for small,
 * explicitly bounded arrays (≤ 1024 elements enforced).
 *
 * ### Zero-copy read-back
 * `toTypedArray()` and `toFloat32Array()` return TypedArray views that point
 * **directly** at the MLX allocator buffer.  The underlying MLX array is kept
 * alive by a shared_ptr finalizer until the TypedArray is garbage collected.
 * Do **not** use these methods for bulk data export in production; prefer
 * `mx.save()` to write to disk.
 */

import addon from '../internal/addon';
import type { DTypeKey, MLXDtype } from './dtype';
import * as dtypeModule from './dtype';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** String key identifying an MLX dtype (e.g., `'float32'`, `'int8'`). */
export type DType = DTypeKey;

/** Either a dtype string key or an MLXDtype object. */
export type DTypeLike = MLXDtype | DType;

// ---------------------------------------------------------------------------
// Symbol-based coercion protocol (analogous to Python's __array__)
// ---------------------------------------------------------------------------

/**
 * A Symbol used to implement the MLX array coercion protocol.
 *
 * Any object that should be convertible to an `MLXArray` by `asarray()` may
 * implement this symbol as a zero-argument method that returns the array:
 *
 * ```ts
 * import { MLX_ARRAY_SYMBOL } from 'mlx';
 *
 * class MyDataSource {
 *   [MLX_ARRAY_SYMBOL](): MLXArray {
 *     // Must route through an approved memory channel (load, TypedArray,
 *     // or builder).  This is the adapter point, not a bypass.
 *     return mx.load(this.filepath);
 *   }
 * }
 * ```
 */
export const MLX_ARRAY_SYMBOL: unique symbol =
  Symbol.for('mlx.array') as any;

// ---------------------------------------------------------------------------
// Supported TypedArray union type
// ---------------------------------------------------------------------------

/** The set of TypedArrays the C++ bridge accepts as direct buffer inputs. */
export type SupportedTypedArray =
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

// ---------------------------------------------------------------------------
// Internal helpers (private, not exported)
// ---------------------------------------------------------------------------

/** Normalise a shape array: finite non-negative integers only. */
function normalizeShape(shape: readonly number[]): number[] {
  return shape.map((dim, i) => {
    if (!Number.isFinite(dim) || dim < 0) {
      throw new RangeError(`Invalid shape dimension at axis ${i}: ${dim}`);
    }
    return Math.trunc(dim);
  });
}

/** Convert a `DTypeLike` value to an `MLXDtype` object for native calls. */
function toDtypeObject(dtype: DTypeLike | undefined): MLXDtype | undefined {
  if (dtype === undefined) return undefined;
  if (typeof dtype === 'string') {
    return (dtypeModule as any)[dtype] as MLXDtype;
  }
  return dtype;
}

/** Return true if `x` is any TypedArray (excluding DataView). */
function isTypedArray(x: unknown): x is SupportedTypedArray {
  return ArrayBuffer.isView(x) && !(x instanceof DataView);
}

// ---------------------------------------------------------------------------
// Public-facing ArrayElement type (used by toArray())
// ---------------------------------------------------------------------------

/** A single element value as returned by `MLXArray.toArray()`. */
export type ArrayElement = number | boolean | [number, number];

// ---------------------------------------------------------------------------
// MLXArray class
// ---------------------------------------------------------------------------

/**
 * A handle on an `mlx::core::array`.  The array is reference-counted inside
 * the C++ addon; this JS object is a thin wrapper around the native handle.
 *
 * Conversions **to** MLXArray (large data):
 * - `mx.load(path)` — C++ reads from file, no JS heap involvement.
 * - `mx.array_builder(dtype, shape).append_row(...).build()` — streaming.
 *
 * Conversions **from** MLXArray (read-back):
 * - `toTypedArray()` / `toFloat32Array()` — zero-copy external buffer view.
 * - `toArray()` — for debugging only; allocates a JS array.
 */
export class MLXArray {
  /** @internal Opaque native handle; never expose outside this module. */
  private readonly handle: any;

  private constructor(handle: any) {
    this.handle = handle;
  }

  /**
   * Wrap a raw native handle in an `MLXArray`.
   * @internal Used by the adapter layer only.
   */
  static fromHandle(handle: any): MLXArray {
    return new MLXArray(handle);
  }

  /** The array's shape as a JS array of non-negative integers. */
  get shape(): number[] {
    return globalThis.Array.from(this.handle.shape() as number[]);
  }

  /** The array's element dtype as a string key. */
  get dtype(): DType {
    return this.handle.dtype().key as DType;
  }

  /**
   * Trigger eager evaluation of any pending lazy operations.
   * Idempotent; returns `this` for chaining.
   */
  eval(): this {
    this.handle.eval();
    return this;
  }

  /**
   * Return a zero-copy TypedArray view of the MLX buffer.
   *
   * The view points directly at the native `mlx::core::allocator::malloc`
   * memory.  The array is kept alive until V8 GCs the returned TypedArray.
   *
   * @remarks Use for debugging / inspection only.  Do **not** iterate over
   * billions of elements in JS — use `mx.save()` for large output.
   */
  toTypedArray(): SupportedTypedArray {
    return this.handle.toTypedArray();
  }

  /**
   * Return a zero-copy `Float32Array` view of the MLX float32 buffer.
   *
   * @throws `Error` if the array's dtype is not `float32`.
   * @remarks Same caveats as `toTypedArray()`.
   */
  toFloat32Array(): Float32Array {
    if (this.dtype !== 'float32') {
      throw new Error(`toFloat32Array(): dtype is '${this.dtype}', not 'float32'`);
    }
    return this.handle.toFloat32Array();
  }

  /**
   * Expose the raw native handle for direct C++ calls.
   * @internal
   */
  toNative(): any {
    return this.handle;
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
    const data = this.toTypedArray();
    switch (this.dtype) {
      case 'bool':
        return Array.from(data as Uint8Array, (v) => v !== 0);
      case 'complex64': {
        const f = data as Float32Array;
        const out: ArrayElement[] = [];
        for (let i = 0; i < f.length; i += 2) {
          out.push([f[i], f[i + 1]]);
        }
        return out;
      }
      default:
        return Array.from(data as ArrayLike<number>);
    }
  }
}

// ---------------------------------------------------------------------------
// normalizeShapeInput (part of public API, used by zeros/ones/etc.)
// ---------------------------------------------------------------------------

/**
 * Normalise a shape tuple: validates finite non-negative values and truncates
 * to integers.
 */
export const normalizeShapeInput = (shape: readonly number[]): number[] =>
  normalizeShape(shape);

// ---------------------------------------------------------------------------
// array() — scalars and small TypedArrays
// ---------------------------------------------------------------------------

/**
 * Construct an `MLXArray` from a scalar value or a TypedArray.
 *
 * This is the **small-data front door**.  Acceptable inputs:
 * - Scalars: `number`, `boolean`, `bigint` → rank-0 array inferred dtype.
 * - TypedArray: `Float32Array`, `Int32Array`, etc. — passed directly to C++
 *   as a single `memcpy` into an MLX-owned buffer.
 *
 * @param data  Scalar or TypedArray (not a plain `number[]` — use
 *   `from_js_array()` for that, subject to its size limit).
 * @param shape Optional shape; if omitted, inferred from `data.length` for
 *   TypedArrays or `[]` for scalars.
 * @param dtype Optional dtype override.  If omitted the C++ side infers it
 *   from the TypedArray element type or scalar JS type.
 *
 * @example
 * ```ts
 * const a = mx.array(1.5);                           // scalar float32
 * const b = mx.array(new Float32Array([1, 2, 3]));   // 1D [3] float32
 * const c = mx.array(new Int32Array([1,2,3,4]), [2,2]); // 2D [2,2] int32
 * ```
 */
export function array(
  data: SupportedTypedArray | number | boolean | bigint,
  shape?: readonly number[],
  dtype?: DTypeLike,
): MLXArray {
  const dtypeObj = toDtypeObject(dtype);

  // Scalar path — passes directly to the C++ scalar constructor.
  if (
    typeof data === 'number' ||
    typeof data === 'boolean' ||
    typeof data === 'bigint'
  ) {
    const args: any[] = [data];
    if (dtypeObj !== undefined) args.push(dtypeObj);
    return MLXArray.fromHandle(addon.array(data, ...args.slice(1)));
  }

  // TypedArray path — single memcpy from V8 backing-store into MLX malloc.
  if (isTypedArray(data)) {
    const resolvedShape = shape !== undefined
      ? normalizeShape([...shape])
      : [data.length];
    const args: any[] = [data, resolvedShape];
    if (dtypeObj !== undefined) args.push(dtypeObj);
    return MLXArray.fromHandle(addon.array(...args));
  }

  throw new TypeError(
    'mx.array(): data must be a scalar (number | boolean | bigint) or a ' +
    'TypedArray.  For plain number[], use mx.from_js_array() (max 1024 ' +
    'elements) or mx.array_builder() for large data.',
  );
}

// ---------------------------------------------------------------------------
// from_js_array() — explicit small-data escape hatch
// ---------------------------------------------------------------------------

/** Maximum number of elements permitted via `from_js_array()`. */
const FROM_JS_ARRAY_MAX_ELEMENTS = 1024;

/**
 * Construct an `MLXArray` from a plain JavaScript `number[]`.
 *
 * This is an **explicit small-data escape hatch**.  It exists to support test
 * harnesses and small initialiser arrays (e.g. shape tuples, tiny kernels).
 * The conversion path goes through a `std::vector<double>` inside the C++
 * addon (unavoidable for untyped JS arrays) and is capped at 1024 elements to
 * prevent accidental use for large tensors.
 *
 * For data larger than 1024 elements use:
 * - `mx.load(path)` — file-backed, C++ reads directly.
 * - `mx.array_builder(dtype, shape)` — streaming row-at-a-time construction.
 *
 * @param data  Array of numbers (must be ≤ {@link FROM_JS_ARRAY_MAX_ELEMENTS}).
 * @param dtype Optional dtype; defaults to `float32`.
 * @param shape Optional shape; defaults to `[data.length]`.
 *
 * @throws `RangeError` if `data.length > FROM_JS_ARRAY_MAX_ELEMENTS`.
 */
export function from_js_array(
  data: readonly (number | bigint)[],
  dtype?: DTypeLike,
  shape?: readonly number[],
): MLXArray {
  if (data.length > FROM_JS_ARRAY_MAX_ELEMENTS) {
    throw new RangeError(
      `from_js_array(): input has ${data.length} elements but the limit is ` +
      `${FROM_JS_ARRAY_MAX_ELEMENTS}.  Use mx.load() or mx.array_builder() ` +
      `for large tensors.`,
    );
  }
  // Delegate to the C++ array() factory which handles number[] via
  // ParseNestedNumberArray — no JS-side type coercion needed.
  const args: any[] = [Array.from(data)];
  if (shape !== undefined) {
    // Provide explicit shape as second argument only when shapes differ from 1D.
    // The C++ factory ignores the shape arg when it's an array.
    args.push(normalizeShape([...shape]));
  }
  const dtypeObj = toDtypeObject(dtype);
  if (dtypeObj !== undefined) args.push(dtypeObj);
  return MLXArray.fromHandle(addon.array(...args));
}

// ---------------------------------------------------------------------------
// ArrayBuilder — streaming row-at-a-time construction
// ---------------------------------------------------------------------------

/**
 * A handle on the native `ArrayBuilderWrapper`.
 *
 * Do **not** call the constructor directly; use `array_builder()`.
 */
export class ArrayBuilder {
  /** @internal */
  private readonly handle: any;

  /** @internal */
  constructor(handle: any) {
    this.handle = handle;
  }

  /**
   * Append one row of data into the pre-allocated MLX buffer.
   *
   * The TypedArray must have exactly `shape[last_dim]` elements (i.e. one row
   * of the innermost dimension).  The bytes are `memcpy`d directly from the
   * TypedArray's V8 backing-store into the MLX buffer at the current write
   * offset — no JS heap allocation occurs.
   *
   * @param row A TypedArray of the correct element type and length.
   * @throws `RangeError` if the length does not match the expected row stride,
   *   or if the buffer is already full.
   * @throws `Error` if `build()` has already been called.
   */
  append_row(row: SupportedTypedArray): this {
    this.handle.append_row(row);
    return this;
  }

  /**
   * Finalise the builder and return the completed `MLXArray`.
   *
   * Wraps the pre-allocated MLX buffer in an `mlx::core::array` with zero
   * additional memory allocations or copies.  May only be called once.
   *
   * @throws `Error` if the buffer has not been fully written.
   * @throws `Error` if called more than once.
   */
  build(): MLXArray {
    return MLXArray.fromHandle(this.handle.build());
  }
}

/**
 * Create a streaming `ArrayBuilder` for constructing large tensors without
 * allocating the full dataset in the V8 heap.
 *
 * Pre-allocates exactly one `mlx::core::allocator::malloc` buffer for the
 * full tensor.  Fill it row-by-row via `builder.append_row(TypedArray)`.
 * Finalise with `builder.build()`.
 *
 * @param dtype The dtype of the resulting array (string key or MLXDtype).
 * @param shape The full shape of the resulting tensor.  Element 0 is the
 *   number of rows; the last element is the columns per row (the stride each
 *   `append_row` call must match).
 *
 * @example
 * ```ts
 * const N = 100_000;
 * const D = 768;
 * const builder = mx.array_builder('float32', [N, D]);
 * for (const embedding of source) {            // Float32Array [D]
 *   builder.append_row(embedding);             // one memcpy of 768*4 bytes
 * }
 * const matrix = builder.build();              // MLXArray [N, D] float32
 * ```
 */
export function array_builder(
  dtype: DTypeLike,
  shape: readonly number[],
): ArrayBuilder {
  const dtypeObj = toDtypeObject(dtype);
  if (dtypeObj === undefined) {
    throw new TypeError('array_builder(): dtype must be a valid dtype');
  }
  const resolvedShape = normalizeShape([...shape]);
  const handle = addon.array_builder(dtypeObj, resolvedShape);
  return new ArrayBuilder(handle);
}

// ---------------------------------------------------------------------------
// asarray() — coercion with protocol support
// ---------------------------------------------------------------------------

/**
 * Coerce an arbitrary value to an `MLXArray`.
 *
 * Resolution order:
 * 1. Already an `MLXArray` → returned as-is.
 * 2. Implements `[MLX_ARRAY_SYMBOL]()` → calls the protocol method.
 * 3. Scalar (`number | boolean | bigint`) → `array(scalar)`.
 * 4. TypedArray → `array(typedArray)`.
 * 5. Small plain `number[]` (≤ 1024 elements) → `from_js_array()`.
 * 6. Anything else → `TypeError`.
 *
 * @param x The value to coerce.
 * @param dtype Optional dtype override applied after coercion.
 */
export function asarray(x: unknown, dtype?: DTypeLike): MLXArray {
  let result: MLXArray;

  if (x instanceof MLXArray) {
    result = x;
  } else if (
    x !== null &&
    x !== undefined &&
    typeof (x as any)[MLX_ARRAY_SYMBOL] === 'function'
  ) {
    result = (x as any)[MLX_ARRAY_SYMBOL]() as MLXArray;
    if (!(result instanceof MLXArray)) {
      throw new TypeError(
        `asarray(): [MLX_ARRAY_SYMBOL]() must return an MLXArray`,
      );
    }
  } else if (
    typeof x === 'number' ||
    typeof x === 'boolean' ||
    typeof x === 'bigint'
  ) {
    result = array(x, undefined, dtype);
    dtype = undefined; // already applied
  } else if (isTypedArray(x)) {
    result = array(x, undefined, dtype);
    dtype = undefined;
  } else if (Array.isArray(x)) {
    // Explicit small-data path — will throw if too large.
    result = from_js_array(x as number[], dtype);
    dtype = undefined;
  } else {
    throw new TypeError(
      'asarray(): cannot coerce value of type ' +
      (x === null ? 'null' : typeof x) +
      ' to MLXArray.  Use mx.load() for file-backed data or ' +
      'mx.array_builder() for large tensors.',
    );
  }

  // Apply dtype cast if requested (and not already applied above).
  if (dtype !== undefined) {
    const dtypeObj = toDtypeObject(dtype);
    if (dtypeObj !== undefined && result.dtype !== (dtypeObj as any).key) {
      return MLXArray.fromHandle(addon.asarray(result.toNative(), dtypeObj));
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Creation op wrappers
// ---------------------------------------------------------------------------

/**
 * Create an array of zeros with the given shape and dtype.
 *
 * @param shape  Array dimensions.
 * @param dtype  Element dtype (default: `float32`).
 */
export function zeros(shape: readonly number[], dtype?: DTypeLike): MLXArray {
  const args: any[] = [normalizeShape([...shape])];
  const dtypeObj = toDtypeObject(dtype);
  if (dtypeObj !== undefined) args.push(dtypeObj);
  return MLXArray.fromHandle(addon.zeros(...args));
}

/**
 * Create an array of ones with the given shape and dtype.
 *
 * @param shape  Array dimensions.
 * @param dtype  Element dtype (default: `float32`).
 */
export function ones(shape: readonly number[], dtype?: DTypeLike): MLXArray {
  const args: any[] = [normalizeShape([...shape])];
  const dtypeObj = toDtypeObject(dtype);
  if (dtypeObj !== undefined) args.push(dtypeObj);
  return MLXArray.fromHandle(addon.ones(...args));
}

/**
 * Create an array filled with a constant value.
 *
 * @param shape  Array dimensions.
 * @param value  Fill value: scalar, TypedArray (for complex64), or MLXArray.
 * @param dtype  Element dtype (default: inferred from `value`).
 */
export function full(
  shape: readonly number[],
  value: number | SupportedTypedArray | MLXArray,
  dtype?: DTypeLike,
): MLXArray {
  const normalizedShape = normalizeShape([...shape]);
  const args: any[] = [normalizedShape];

  if (value instanceof MLXArray) {
    args.push(value.toNative());
  } else if (isTypedArray(value)) {
    args.push(value);
  } else {
    args.push(value as number);
  }

  const dtypeObj = toDtypeObject(dtype);
  if (dtypeObj !== undefined) args.push(dtypeObj);
  return MLXArray.fromHandle(addon.full(...args));
}

/**
 * Create a zero array with the same shape and dtype as `base`.
 *
 * @param base  Source array.
 */
export function zeros_like(base: MLXArray): MLXArray {
  return MLXArray.fromHandle(addon.zeros_like(base.toNative()));
}

/**
 * Create a ones array with the same shape and dtype as `base`.
 *
 * @param base  Source array.
 */
export function ones_like(base: MLXArray): MLXArray {
  return MLXArray.fromHandle(addon.ones_like(base.toNative()));
}

export default MLXArray;

/**
 * @fileoverview Node.js parity tests for python/tests/test_array.py
 *
 * Covers: TestDtypes, TestEquality, TestInequality, TestArray, TestVersion.
 * Numpy-specific tests (buffer_protocol, dlpack, memoryview) are omitted —
 * no JS equivalent exists. TensorFlow tests are omitted.
 */

import { strict as assert } from 'assert';
import * as mx from '../../src';
import { from_js_array } from '../../src';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Assert two MLXArrays are the same shape, dtype, and values. */
function assertArrayEqual(a: mx.Array, b: mx.Array): void {
  assert.deepEqual(a.shape, b.shape, `shape mismatch: ${a.shape} vs ${b.shape}`);
  assert.equal(a.dtype, b.dtype, `dtype mismatch: ${a.dtype} vs ${b.dtype}`);
  assert.deepEqual(a.toArray(), b.toArray());
}

/** Assert every element of a float MLXArray is close to the corresponding value. */
function assertAllClose(
  a: mx.Array,
  expected: number[],
  atol = 1e-5,
  rtol = 1e-5,
): void {
  const vals = a.toArray() as number[];
  assert.equal(vals.length, expected.length);
  for (let i = 0; i < vals.length; i++) {
    const diff = Math.abs(vals[i] - expected[i]);
    const tol = atol + rtol * Math.abs(expected[i]);
    assert.ok(diff <= tol, `element ${i}: ${vals[i]} vs ${expected[i]}, diff=${diff}`);
  }
}

// ---------------------------------------------------------------------------
// TestDtypes
// ---------------------------------------------------------------------------

describe('TestDtypes', () => {
  it('dtype sizes', () => {
    // Parity: mx.bool_.size == 1, etc.
    assert.equal(mx.bool_.size, 1);
    assert.equal(mx.uint8.size, 1);
    assert.equal(mx.uint16.size, 2);
    assert.equal(mx.uint32.size, 4);
    assert.equal(mx.uint64.size, 8);
    assert.equal(mx.int8.size, 1);
    assert.equal(mx.int16.size, 2);
    assert.equal(mx.int32.size, 4);
    assert.equal(mx.int64.size, 8);
    assert.equal(mx.float16.size, 2);
    assert.equal(mx.float32.size, 4);
    assert.equal(mx.bfloat16.size, 2);
    assert.equal(mx.complex64.size, 8);
  });

  it('dtype key strings', () => {
    // In Python: str(mx.bool_) == 'mlx.core.bool'
    // In Node: dtype.key gives the string like 'bool', 'uint8', etc.
    assert.equal(mx.bool_.key, 'bool');
    assert.equal(mx.uint8.key, 'uint8');
    assert.equal(mx.uint16.key, 'uint16');
    assert.equal(mx.uint32.key, 'uint32');
    assert.equal(mx.uint64.key, 'uint64');
    assert.equal(mx.int8.key, 'int8');
    assert.equal(mx.int16.key, 'int16');
    assert.equal(mx.int32.key, 'int32');
    assert.equal(mx.int64.key, 'int64');
    assert.equal(mx.float16.key, 'float16');
    assert.equal(mx.float32.key, 'float32');
    assert.equal(mx.bfloat16.key, 'bfloat16');
    assert.equal(mx.complex64.key, 'complex64');
  });
});

// ---------------------------------------------------------------------------
// TestEquality
// ---------------------------------------------------------------------------

describe('TestEquality', () => {
  it('array == array elementwise', () => {
    const a = from_js_array([1, 2, 3], 'int32');
    const b = from_js_array([1, 2, 3], 'int32');
    const c = from_js_array([1, 2, 4], 'int32');
    assert.ok(mx.all(mx.equal(a, b)).toArray()[0] as boolean);
    assert.ok(!(mx.all(mx.equal(a, c)).toArray()[0] as boolean));
  });

  it('array == scalar elementwise', () => {
    const a = from_js_array([1, 2, 3], 'int32');
    assert.ok((mx.any(mx.equal(a, mx.array(1))).toArray()[0] as boolean));
    assert.ok(!(mx.all(mx.equal(a, mx.array(4))).toArray()[0] as boolean));
  });

  it('array != array elementwise', () => {
    const a = from_js_array([1, 2, 3], 'int32');
    const b = from_js_array([1, 2, 3], 'int32');
    const c = from_js_array([1, 2, 4], 'int32');
    assert.ok(!(mx.any(mx.not_equal(a, b)).toArray()[0] as boolean));
    assert.ok((mx.any(mx.not_equal(a, c)).toArray()[0] as boolean));
  });
});

// ---------------------------------------------------------------------------
// TestArray — core.array basics
// ---------------------------------------------------------------------------

describe('TestArray', () => {
  describe('array_basics', () => {
    it('scalar int32', () => {
      // mx.array(1) → int32, shape=[]
      const x = mx.array(1);
      assert.deepEqual(x.shape, []);
      assert.equal(x.dtype, 'int32');
      assert.deepEqual(x.toArray(), [1]);
    });

    it('scalar float32', () => {
      const x = mx.array(1.0);
      assert.deepEqual(x.shape, []);
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.toArray(), [1.0]);
    });

    it('scalar bool', () => {
      const x = mx.array(false);
      assert.deepEqual(x.shape, []);
      assert.equal(x.dtype, 'bool');
      assert.deepEqual(x.toArray(), [false]);
    });

    it('1D int32 from TypedArray', () => {
      const data = new Int32Array([0, 1, 2]);
      const x = mx.array(data);
      assert.equal(x.dtype, 'int32');
      assert.deepEqual(x.shape, [3]);
      assert.deepEqual(x.toArray(), [0, 1, 2]);
    });

    it('1D float32 from TypedArray', () => {
      const data = new Float32Array([0.0, 1.0, 2.0]);
      const x = mx.array(data);
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.shape, [3]);
      assert.deepEqual(x.toArray(), [0, 1, 2]);
    });

    it('bool TypedArray (Uint8Array) → bool dtype', () => {
      const data = new Uint8Array([1, 0, 1]);
      const x = mx.array(data, undefined, 'bool');
      assert.equal(x.dtype, 'bool');
      assert.deepEqual(x.shape, [3]);
      assert.deepEqual(x.toArray(), [true, false, true]);
    });

    it('int32 list via from_js_array', () => {
      const x = from_js_array([0, 1, 2], 'int32');
      assert.equal(x.dtype, 'int32');
      assert.deepEqual(x.shape, [3]);
    });

    it('float32 list via from_js_array', () => {
      const x = from_js_array([0.0, 1.0, 2.0], 'float32');
      assert.equal(x.dtype, 'float32');
    });

    it('complex64 interleaved Float32Array', () => {
      // Python: mx.array([1j, 1+0j]) → complex64
      const data = new Float32Array([0, 1, 1, 0]); // [0+1j, 1+0j] interleaved
      const x = mx.array(data, [2], 'complex64');
      assert.equal(x.dtype, 'complex64');
      assert.deepEqual(x.shape, [2]);
    });
  });

  describe('bool_conversion', () => {
    it('scalar true/false arrays', () => {
      // Python: bool(mx.array(True)) == True
      const t = mx.array(true);
      assert.ok(t.toArray()[0] as boolean);
      const f = mx.array(false);
      assert.ok(!(f.toArray()[0] as boolean));
    });
  });

  describe('int_type', () => {
    it('small int → int32', () => {
      const x = mx.array(1);
      assert.equal(x.dtype, 'int32');
    });

    it('scalar with explicit dtype', () => {
      const x = mx.array(1, undefined, 'uint32');
      assert.equal(x.dtype, 'uint32');
    });

    it('explicit int64', () => {
      const x = mx.array(1, undefined, 'int64');
      assert.equal(x.dtype, 'int64');
    });
  });

  describe('construction_from_small_list via from_js_array', () => {
    it('empty array', () => {
      // Python: mx.array([]) → shape=(0,), dtype=float32
      // from_js_array([]) also works
      const x = from_js_array([], 'float32');
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.shape, [0]);
    });

    it('bool list', () => {
      // Python: mx.array([True, False, 3]) → dtype=int32
      // JS: from_js_array([1, 0, 3], 'int32')
      const x = from_js_array([1, 0, 3], 'int32');
      assert.equal(x.dtype, 'int32');
    });

    it('mixed int/float list → float32', () => {
      const x = from_js_array([1, 2, 4.0], 'float32');
      assert.equal(x.dtype, 'float32');
    });
  });

  describe('array_type_cast (astype)', () => {
    it('float32 → int32 truncates', () => {
      const a = mx.array(new Float32Array([0.1, 2.3, -1.3]));
      const b = mx.core.astype(a, 'int32');
      assert.equal(b.dtype, 'int32');
      assert.deepEqual(b.toArray(), [0, 2, -1]);
    });

    it('int32 → float32', () => {
      const a = from_js_array([0, 2, -1], 'int32');
      const b = mx.core.astype(a, 'float32');
      assert.equal(b.dtype, 'float32');
    });
  });

  describe('array_comparison', () => {
    it('lt, le, gt, ge between arrays', () => {
      const a = mx.array(new Float32Array([0.0, 1.0, 5.0]));
      const b = mx.array(new Float32Array([-1.0, 2.0, 5.0]));
      assert.deepEqual(mx.less(a, b).toArray(), [false, true, false]);
      assert.deepEqual(mx.less_equal(a, b).toArray(), [false, true, true]);
      assert.deepEqual(mx.greater(a, b).toArray(), [true, false, false]);
      assert.deepEqual(mx.greater_equal(a, b).toArray(), [true, false, true]);
    });

    it('lt, le, gt, ge with scalar', () => {
      const a = mx.array(new Float32Array([0.0, 1.0, 5.0]));
      assert.deepEqual(mx.less(a, mx.array(5.0)).toArray(), [true, true, false]);
      assert.deepEqual(mx.greater(a, mx.array(1.0)).toArray(), [false, false, true]);
      assert.deepEqual(mx.greater_equal(a, mx.array(1.0)).toArray(), [false, true, true]);
    });
  });

  describe('array_neg', () => {
    it('negate float array', () => {
      const a = mx.array(new Float32Array([-1.0, 4.0, 0.0]));
      const neg = mx.negative(a);
      assert.deepEqual(neg.toArray(), [1.0, -4.0, -0.0]);
    });
  });

  describe('array_to_list (toArray)', () => {
    it('1D int values', () => {
      const vals = [1, 2, 3, 4];
      const x = from_js_array(vals, 'int32');
      assert.deepEqual(x.toArray(), vals);
    });

    it('1D float values', () => {
      const vals = [1.5, 2.5, 3.5, 4.5];
      const x = mx.array(new Float32Array(vals));
      assert.deepEqual(x.toArray(), vals);
    });

    it('bool dtype toArray → boolean[]', () => {
      const vals = [1, 0, 1, 0];
      const x = from_js_array(vals, 'bool');
      assert.deepEqual(x.toArray(), [true, false, true, false]);
    });

    it('complex64 toArray → [real, imag][] pairs', () => {
      // interleaved float32: [real0, imag0, real1, imag1, ...]
      const data = new Float32Array([0.5, 0, 1.5, 1.0, 2.5, 0]);
      const x = mx.array(data, [3], 'complex64');
      const list = x.toArray() as [number, number][];
      assert.equal(list.length, 3);
      assert.deepEqual(list[0], [0.5, 0]);
      assert.deepEqual(list[1], [1.5, 1.0]);
      assert.deepEqual(list[2], [2.5, 0]);
    });
  });

  describe('zeros / ones / full / _like', () => {
    it('zeros shape and dtype', () => {
      const x = mx.zeros([2, 3], 'float32');
      assert.deepEqual(x.shape, [2, 3]);
      assert.equal(x.dtype, 'float32');
      assert.ok((x.toArray() as number[]).every((v) => v === 0));
    });

    it('ones shape and dtype', () => {
      const x = mx.ones([2], 'float32');
      assert.deepEqual(x.shape, [2]);
      assert.equal(x.dtype, 'float32');
      assert.ok((x.toArray() as number[]).every((v) => v === 1));
    });

    it('full scalar fill', () => {
      const x = mx.full([2], 3.0, 'float32');
      assert.deepEqual(x.shape, [2]);
      assert.deepEqual(x.toArray(), [3.0, 3.0]);
    });

    it('full 2D', () => {
      const x = mx.full([2, 3], 2.0, 'float32');
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.shape, [2, 3]);
      assert.deepEqual(x.toArray(), [2, 2, 2, 2, 2, 2]);
    });

    it('zeros_like', () => {
      const base = mx.ones([4], 'float32');
      const z = mx.zeros_like(base);
      assert.deepEqual(z.shape, [4]);
      assert.equal(z.dtype, 'float32');
      assert.ok((z.toArray() as number[]).every((v) => v === 0));
    });

    it('ones_like', () => {
      const base = mx.zeros([4], 'int32');
      const o = mx.ones_like(base);
      assert.equal(o.dtype, 'int32');
      assert.ok((o.toArray() as number[]).every((v) => v === 1));
    });

    it('zeros/ones with bool dtype', () => {
      for (const dtype of ['bool', 'int32', 'float32'] as const) {
        const z = mx.zeros([2, 2], dtype);
        assert.equal(z.dtype, dtype);
        const o = mx.ones([2, 2], dtype);
        assert.equal(o.dtype, dtype);
      }
    });
  });

  describe('array_equal op', () => {
    it('equal arrays', () => {
      const x = from_js_array([1, 2, 3, 4], 'int32');
      const y = from_js_array([1, 2, 3, 4], 'int32');
      assert.ok(mx.array_equal(x, y).toArray()[0] as boolean);
    });

    it('unequal arrays', () => {
      const x = from_js_array([1, 2, 3, 4], 'int32');
      const y = from_js_array([1, 2, 4, 5], 'int32');
      assert.ok(!(mx.array_equal(x, y).toArray()[0] as boolean));
    });

    it('different lengths → not equal', () => {
      const x = from_js_array([1, 2, 3, 4], 'int32');
      const y = from_js_array([1, 2, 3], 'int32');
      assert.ok(!(mx.array_equal(x, y).toArray()[0] as boolean));
    });

    it('equal value different dtype', () => {
      const x = from_js_array([1, 2, 3, 4], 'int32');
      const y = mx.array(new Float32Array([1, 2, 3, 4]));
      assert.ok(mx.array_equal(x, y).toArray()[0] as boolean);
    });

    it('NaN not equal by default', () => {
      const x = mx.array(new Float32Array([0, NaN]));
      const y = mx.array(new Float32Array([0, NaN]));
      assert.ok(!(mx.array_equal(x, y).toArray()[0] as boolean));
    });

    it('NaN equal with equal_nan=true', () => {
      const x = mx.array(new Float32Array([0, NaN]));
      const y = mx.array(new Float32Array([0, NaN]));
      assert.ok(mx.array_equal(x, y, true).toArray()[0] as boolean);
    });
  });

  describe('logical overloads', () => {
    it('bitwise and on bool', () => {
      // Python: mx.array(True) & True → True
      const a = mx.array(true);
      const b = mx.array(true);
      const r = mx.logical_and(a, b);
      assert.ok(r.toArray()[0] as boolean);
    });

    it('bool AND false', () => {
      const a = mx.array(true);
      const f = mx.array(false);
      assert.ok(!(mx.logical_and(a, f).toArray()[0] as boolean));
    });

    it('bool OR false', () => {
      const a = mx.array(true);
      const f = mx.array(false);
      assert.ok(mx.logical_or(a, f).toArray()[0] as boolean);
    });

    it('bool NOT', () => {
      const a = mx.array(false);
      assert.ok(mx.logical_not(a).toArray()[0] as boolean);
    });
  });

  describe('arange', () => {
    it('arange basic', () => {
      // Python: mx.arange(5) → [0,1,2,3,4] int32
      const x = mx.arange(0, 5, 1, 'int32');
      assert.deepEqual(x.shape, [5]);
      assert.deepEqual(x.toArray(), [0, 1, 2, 3, 4]);
    });

    it('arange float step', () => {
      const x = mx.arange(0, 1, 0.25, 'float32');
      assertAllClose(x, [0, 0.25, 0.5, 0.75]);
    });
  });

  describe('reshape', () => {
    it('1D to 2D', () => {
      const x = mx.arange(0, 6, 1, 'int32');
      const y = mx.reshape(x, [2, 3]);
      assert.deepEqual(y.shape, [2, 3]);
      assert.deepEqual(y.toArray(), [0, 1, 2, 3, 4, 5]);
    });
  });

  describe('transpose', () => {
    it('2D no-args', () => {
      const x = from_js_array([0, 1, 1, 1, 0, 0], 'int32', [2, 3]);
      const t = mx.transpose(x);
      assert.deepEqual(t.shape, [3, 2]);
    });

    it('with axes', () => {
      const x = mx.arange(0, 24, 1, 'int32');
      const r = mx.reshape(x, [2, 3, 4]);
      const t = mx.transpose(r, [0, 2, 1]);
      assert.deepEqual(t.shape, [2, 4, 3]);
    });
  });

  describe('concatenate / split / stack', () => {
    it('concatenate along axis 0', () => {
      const a = from_js_array([1, 2, 3], 'int32');
      const b = from_js_array([4, 5, 6], 'int32');
      const c = mx.concatenate([a, b]);
      assert.deepEqual(c.shape, [6]);
      assert.deepEqual(c.toArray(), [1, 2, 3, 4, 5, 6]);
    });

    it('split', () => {
      const x = from_js_array([0, 1, 2, 3, 4, 5], 'int32');
      const parts = mx.split(x, 2);
      assert.equal(parts.length, 2);
      assert.deepEqual(parts[0].toArray(), [0, 1, 2]);
      assert.deepEqual(parts[1].toArray(), [3, 4, 5]);
    });

    it('stack', () => {
      const a = from_js_array([1, 2], 'int32');
      const b = from_js_array([3, 4], 'int32');
      const s = mx.stack([a, b], 0);
      assert.deepEqual(s.shape, [2, 2]);
      assert.deepEqual(s.toArray(), [1, 2, 3, 4]);
    });
  });

  describe('broadcast_to', () => {
    it('scalar to 1D', () => {
      const x = mx.array(1, undefined, 'float32');
      const b = mx.broadcast_to(x, [3]);
      assert.deepEqual(b.shape, [3]);
      assert.deepEqual(b.toArray(), [1, 1, 1]);
    });

    it('row to matrix', () => {
      const x = mx.arange(0, 3, 1, 'int32');
      const b = mx.broadcast_to(x, [2, 3]);
      assert.deepEqual(b.shape, [2, 3]);
    });
  });

  describe('array.real / array.imag', () => {
    it('real of real array', () => {
      const x = mx.array(new Float32Array([1.0]));
      const r = mx.real(x);
      assert.deepEqual(r.toArray(), [1.0]);
    });

    it('real and imag of complex64', () => {
      const data = new Float32Array([1.0, 1.0]); // 1+1j
      const x = mx.array(data, [1], 'complex64');
      const r = mx.real(x);
      const im = mx.imag(x);
      assert.deepEqual(r.toArray(), [1.0]);
      assert.deepEqual(im.toArray(), [1.0]);
    });
  });

  describe('squeeze', () => {
    it('squeeze axis', () => {
      const x = mx.zeros([10, 10, 1], 'float32');
      const y = mx.squeeze(x, [2]);
      assert.deepEqual(y.shape, [10, 10]);
    });
  });

  describe('moveaxis / swapaxes', () => {
    it('moveaxis', () => {
      const x = mx.zeros([2, 3, 4], 'float32');
      const y = mx.moveaxis(x, 0, 2);
      assert.deepEqual(y.shape, [3, 4, 2]);
    });

    it('swapaxes', () => {
      const x = mx.zeros([2, 3, 4], 'float32');
      const y = mx.swapaxes(x, 0, 2);
      assert.deepEqual(y.shape, [4, 3, 2]);
    });
  });

  describe('flatten', () => {
    it('flatten 3D → 1D', () => {
      const x = mx.arange(0, 24, 1, 'int32');
      const r = mx.reshape(x, [2, 3, 4]);
      const f = mx.flatten(r);
      assert.deepEqual(f.shape, [24]);
    });
  });

  describe('sum / prod / mean / var', () => {
    it('sum all elements', () => {
      const x = from_js_array([1, 2, 3, 3], 'int32', [2, 2]);
      const s = mx.sum(x);
      assert.equal((s.toArray() as number[])[0], 9);
    });

    it('sum along axis 0', () => {
      const x = from_js_array([1, 2, 3, 3], 'int32', [2, 2]);
      assert.deepEqual(mx.sum(x, 0).toArray(), [4, 5]);
    });

    it('sum along axis 1', () => {
      const x = from_js_array([1, 2, 3, 3], 'int32', [2, 2]);
      assert.deepEqual(mx.sum(x, 1).toArray(), [3, 6]);
    });

    it('prod all elements', () => {
      const x = from_js_array([1, 2, 3, 3], 'int32', [2, 2]);
      assert.equal((mx.prod(x).toArray() as number[])[0], 18);
    });

    it('mean', () => {
      const x = from_js_array([1, 2, 3, 4], 'float32', [2, 2]);
      assert.ok(Math.abs((mx.mean(x).toArray() as number[])[0] - 2.5) < 1e-6);
    });

    it('var', () => {
      const x = from_js_array([1, 2, 3, 4], 'float32', [2, 2]);
      assert.ok(Math.abs((mx.var_(x).toArray() as number[])[0] - 1.25) < 1e-6);
    });
  });

  describe('min / max / argmin / argmax', () => {
    it('min and max', () => {
      const x = from_js_array([1, 2, 3, 4], 'int32', [2, 2]);
      assert.equal((mx.min_(x).toArray() as number[])[0], 1);
      assert.equal((mx.max_(x).toArray() as number[])[0], 4);
    });

    it('argmin argmax no axis', () => {
      const x = from_js_array([4, 1, 3, 2], 'float32');
      assert.equal((mx.argmin(x).toArray() as number[])[0], 1);
      assert.equal((mx.argmax(x).toArray() as number[])[0], 0);
    });
  });

  describe('cumsum / cumprod', () => {
    it('cumsum', () => {
      const x = from_js_array([1, 2, 3, 4], 'int32');
      assert.deepEqual(mx.cumsum(x, 0).toArray(), [1, 3, 6, 10]);
    });

    it('cumprod', () => {
      const x = from_js_array([1, 2, 3, 4], 'int32');
      assert.deepEqual(mx.cumprod(x, 0).toArray(), [1, 2, 6, 24]);
    });
  });

  describe('unary ops via mx.*', () => {
    it('abs', () => {
      const x = mx.array(new Float32Array([-1, 2, -3]));
      assert.deepEqual(mx.abs(x).toArray(), [1, 2, 3]);
    });

    it('square', () => {
      const x = mx.array(new Float32Array([1, 2, 3]));
      assert.deepEqual(mx.square(x).toArray(), [1, 4, 9]);
    });

    it('sqrt', () => {
      const x = mx.array(new Float32Array([4, 9, 16]));
      assertAllClose(mx.sqrt(x), [2, 3, 4]);
    });

    it('exp', () => {
      const x = mx.array(new Float32Array([0, 1]));
      assertAllClose(mx.exp(x), [1, Math.E]);
    });

    it('log', () => {
      const x = mx.array(new Float32Array([1, Math.E]));
      assertAllClose(mx.log(x), [0, 1]);
    });

    it('sin and cos', () => {
      const x = mx.array(new Float32Array([0, Math.PI / 2]));
      assertAllClose(mx.sin(x), [0, 1], 1e-6);
      assertAllClose(mx.cos(x), [1, 0], 1e-6);
    });

    it('log1p', () => {
      const x = mx.array(new Float32Array([0, 1]));
      assertAllClose(mx.log1p(x), [0, Math.log(2)]);
    });

    it('reciprocal', () => {
      const x = mx.array(new Float32Array([2, 4]));
      assertAllClose(mx.reciprocal(x), [0.5, 0.25]);
    });
  });

  describe('all / any', () => {
    it('all on bool array', () => {
      const x = from_js_array([1, 1, 1], 'bool');
      assert.ok(mx.all(x).toArray()[0] as boolean);
    });

    it('all false on mixed bool', () => {
      const x = from_js_array([1, 0, 1], 'bool');
      assert.ok(!(mx.all(x).toArray()[0] as boolean));
    });

    it('any on bool array', () => {
      const x = from_js_array([0, 0, 1], 'bool');
      assert.ok(mx.any(x).toArray()[0] as boolean);
    });
  });

  describe('where', () => {
    it('mx.where selects from two arrays', () => {
      const cond = from_js_array([1, 0, 1], 'bool');
      const a = from_js_array([1, 2, 3], 'int32');
      const b = from_js_array([10, 20, 30], 'int32');
      const r = mx.where(cond, a, b);
      assert.deepEqual(r.toArray(), [1, 20, 3]);
    });
  });

  describe('diagonal / trace', () => {
    it('diagonal of 3×3', () => {
      const x = from_js_array([1,2,3,4,5,6,7,8,9], 'float32', [3, 3]);
      const d = mx.diagonal(x);
      assert.deepEqual(d.toArray(), [1, 5, 9]);
    });
  });

  describe('round', () => {
    it('round to even (banker\'s rounding)', () => {
      // Python: mx.round([0.5, -0.5, 1.5, -1.5]) → [0, 0, 2, -2]
      const x = mx.array(new Float32Array([0.5, -0.5, 1.5, -1.5]));
      assert.deepEqual(mx.round(x).toArray(), [0, -0, 2, -2]);
    });

    it('round to 1 decimal', () => {
      const x = mx.array(new Float32Array([1.537, 1.471]));
      const r = mx.round(x, 1);
      assertAllClose(r, [1.5, 1.5], 1e-4);
    });
  });
});

// ---------------------------------------------------------------------------
// TestOps (subset — full coverage in test/core/ops.test.ts)
// ---------------------------------------------------------------------------

describe('TestOps (subset from test_array.py)', () => {
  describe('add', () => {
    it('int array + int array', () => {
      const x = mx.array(1);
      const y = mx.array(1);
      assert.equal((mx.add(x, y).toArray() as number[])[0], 2);
    });

    it('float array + float scalar', () => {
      const x = mx.array(1.0);
      const z = mx.add(x, mx.array(3.0));
      assert.equal(z.dtype, 'float32');
      assert.ok(Math.abs((z.toArray() as number[])[0] - 4.0) < 1e-6);
    });
  });

  describe('subtract', () => {
    it('float - float', () => {
      const x = mx.array(new Float32Array([4.0]));
      const y = mx.array(new Float32Array([3.0]));
      const z = mx.subtract(x, y);
      assert.equal(z.dtype, 'float32');
      assert.ok(Math.abs((z.toArray() as number[])[0] - 1.0) < 1e-6);
    });
  });

  describe('multiply', () => {
    it('float * float', () => {
      const z = mx.multiply(mx.array(2.0), mx.array(3.0));
      assert.ok(Math.abs((z.toArray() as number[])[0] - 6.0) < 1e-6);
    });
  });

  describe('divide', () => {
    it('float / float', () => {
      const z = mx.divide(mx.array(2.0), mx.array(4.0));
      assert.ok(Math.abs((z.toArray() as number[])[0] - 0.5) < 1e-6);
    });

    it('int / int → float32', () => {
      const x = mx.array(5, undefined, 'int32');
      const y = mx.array(2, undefined, 'int32');
      const z = mx.divide(x, y);
      assert.equal(z.dtype, 'float32');
      assert.ok(Math.abs((z.toArray() as number[])[0] - 2.5) < 1e-6);
    });

    it('floor_divide', () => {
      const x = mx.array(5, undefined, 'int32');
      const y = mx.array(2, undefined, 'int32');
      const z = mx.floor_divide(x, y);
      assert.equal(z.dtype, 'int32');
      assert.equal((z.toArray() as number[])[0], 2);
    });
  });

  describe('minimum / maximum', () => {
    it('minimum', () => {
      const a = mx.array(new Float32Array([0.0, -5.0, 10.0]));
      const b = mx.array(new Float32Array([1.0, -7.0, 3.0]));
      assert.deepEqual(mx.minimum(a, b).toArray(), [0, -7, 3]);
    });

    it('maximum', () => {
      const a = mx.array(new Float32Array([0.0, -5.0, 10.0]));
      const b = mx.array(new Float32Array([1.0, -7.0, 3.0]));
      assert.deepEqual(mx.maximum(a, b).toArray(), [1, -5, 10]);
    });

    it('minimum with NaN propagates', () => {
      const a = mx.array(new Float32Array([NaN]));
      const b = mx.array(new Float32Array([0.0]));
      assert.ok(isNaN((mx.minimum(a, b).toArray() as number[])[0]));
    });
  });

  describe('floor / ceil', () => {
    it('floor', () => {
      const x = mx.array(new Float32Array([-22.03, 19.98, -27.0, 9.0, 0.0]));
      assert.deepEqual(mx.floor(x).toArray(), [-23, 19, -27, 9, 0]);
    });

    it('ceil', () => {
      const x = mx.array(new Float32Array([-22.03, 19.98, -27.0, 9.0, 0.0]));
      assert.deepEqual(mx.ceil(x).toArray(), [-22, 20, -27, 9, 0]);
    });
  });

  describe('isnan / isinf / isfinite', () => {
    it('isnan', () => {
      const x = mx.array(new Float32Array([0.0, NaN]));
      assert.deepEqual(mx.isnan(x).toArray(), [false, true]);
    });

    it('isinf', () => {
      const x = mx.array(new Float32Array([0.0, Infinity]));
      assert.deepEqual(mx.isinf(x).toArray(), [false, true]);
    });

    it('isfinite', () => {
      const x = mx.array(new Float32Array([0.0, Infinity, NaN]));
      assert.deepEqual(mx.isfinite(x).toArray(), [true, false, false]);
    });
  });

  describe('logsumexp', () => {
    it('basic logsumexp', () => {
      const x = from_js_array([1.0, 2.0, 3.0, 4.0], 'float32', [2, 2]);
      const r = mx.logsumexp(x);
      // Expected: log(e^1 + e^2 + e^3 + e^4) ≈ 4.4402
      expect: {
        const val = (r.toArray() as number[])[0];
        assert.ok(Math.abs(val - Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(4))) < 0.01);
      }
    });
  });

  describe('matmul', () => {
    it('2x2 matmul', () => {
      const a = from_js_array([1, 2, 3, 4], 'float32', [2, 2]);
      const b = from_js_array([1, 0, 0, 1], 'float32', [2, 2]);
      const c = mx.matmul(a, b);
      assert.deepEqual(c.shape, [2, 2]);
      assert.deepEqual(c.toArray(), [1, 2, 3, 4]);
    });
  });
});

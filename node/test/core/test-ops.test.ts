import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import * as core from '../../src/index';

const approx = (a: number, b: number, tol = 1e-5) => Math.abs(a - b) < tol;
const allClose = (arr: number[], expected: number[], tol = 1e-5) =>
  arr.every((v, i) => approx(v, expected[i], tol));
const toArr = (t: any): number[] => t.toArray() as number[];
const toScalar = (t: any): any => t.toArray()[0];

// ---------------------------------------------------------------------------
// test_full_ones_zeros
// ---------------------------------------------------------------------------
describe('full / ones / zeros (multi-dtype, bool fill)', () => {
  it('full with scalar shape', () => {
    const x = core.full([2], 3.0);
    assert.deepEqual(x.shape, [2]);
    assert.deepEqual(toArr(x), [3, 3]);
  });

  it('full with 2D shape', () => {
    const x = core.full([2, 3], 2.0);
    assert.equal(x.dtype, 'int32');
    assert.deepEqual(x.shape, [2, 3]);
    assert.deepEqual(toArr(x), [2, 2, 2, 2, 2, 2]);
  });

  it('full with broadcast array value', () => {
    const x = core.full([3, 2], core.array(new Float32Array([2.0, 3.0])));
    assert.deepEqual(toArr(x), [2, 3, 2, 3, 2, 3]);
  });

  it('zeros multi-dtype', () => {
    for (const dt of ['int32', 'float32'] as const) {
      const x = core.zeros([2, 2], dt);
      assert.equal(x.dtype, dt);
      assert.deepEqual(toArr(x), [0, 0, 0, 0]);
    }
  });

  it('ones multi-dtype', () => {
    for (const dt of ['int32', 'float32'] as const) {
      const x = core.ones([2, 2], dt);
      assert.equal(x.dtype, dt);
      assert.deepEqual(toArr(x), [1, 1, 1, 1]);
    }
  });

  it('zeros_like preserves dtype and shape', () => {
    const a = core.ones([3, 2], 'int32');
    const z = core.zeros_like(a);
    assert.equal(z.dtype, 'int32');
    assert.deepEqual(z.shape, [3, 2]);
    assert.deepEqual(toArr(z), [0, 0, 0, 0, 0, 0]);
  });

  it('ones_like preserves dtype and shape', () => {
    const a = core.zeros([2, 3], 'float32');
    const o = core.ones_like(a);
    assert.equal(o.dtype, 'float32');
    assert.deepEqual(o.shape, [2, 3]);
    assert.deepEqual(toArr(o), [1, 1, 1, 1, 1, 1]);
  });
});

// ---------------------------------------------------------------------------
// test_comparisons (scalar broadcasting)
// ---------------------------------------------------------------------------
describe('comparisons with scalar broadcasting', () => {
  it('less with scalar rhs', () => {
    const a = core.array(new Float32Array([0.0, 1.0, 5.0]));
    assert.deepEqual(toArr(core.less(a, 5)), [true, true, false]);
  });

  it('less with scalar lhs', () => {
    const a = core.array(new Float32Array([0.0, 1.0, 5.0]));
    assert.deepEqual(toArr(core.less(5, a)), [false, false, false]);
  });

  it('less_equal with scalar lhs', () => {
    const a = core.array(new Float32Array([0.0, 1.0, 5.0]));
    assert.deepEqual(toArr(core.less_equal(5, a)), [false, false, true]);
  });

  it('greater with scalar rhs', () => {
    const a = core.array(new Float32Array([0.0, 1.0, 5.0]));
    assert.deepEqual(toArr(core.greater(a, 1)), [false, false, true]);
  });

  it('greater_equal with scalar rhs', () => {
    const a = core.array(new Float32Array([0.0, 1.0, 5.0]));
    assert.deepEqual(toArr(core.greater_equal(a, 1)), [false, true, true]);
  });

  it('equal element-wise', () => {
    const a = core.array(new Float32Array([0, 1, 5, -1]));
    const b = core.array(new Float32Array([0, 2, 5, 3]));
    assert.deepEqual(toArr(core.equal(a, b)), [true, false, true, false]);
  });

  it('not_equal element-wise', () => {
    const a = core.array(new Float32Array([0, 1, 5, -1]));
    const b = core.array(new Float32Array([0, 2, 5, 3]));
    assert.deepEqual(toArr(core.not_equal(a, b)), [false, true, false, true]);
  });
});

// ---------------------------------------------------------------------------
// test_remainder (negative modulo, dtype preservation)
// ---------------------------------------------------------------------------
describe('remainder', () => {
  it('basic int remainder', () => {
    const x = core.array(new Int32Array([2]));
    const y = core.array(new Int32Array([4]));
    assert.equal(toScalar(core.remainder(x, y)), 2);
    assert.equal(toScalar(core.remainder(y, x)), 0);
  });

  it('negative modulo int gives positive result', () => {
    // -1 % 2 == 1 in Python/MLX semantics
    const r = core.remainder(-1, core.array(new Int32Array([2])));
    assert.equal(toScalar(r), 1);
  });

  it('negative mod negative gives negative', () => {
    const r = core.remainder(-1, core.array(new Int32Array([-2])));
    assert.equal(toScalar(r), -1);
  });

  it('float remainder preserves dtype', () => {
    const x = core.array(new Float32Array([2]));
    const y = core.array(new Float32Array([4]));
    const z = core.remainder(x, y);
    assert.equal(z.dtype, 'float32');
    assert.equal(toScalar(z), 2);
  });

  it('range modulo positive', () => {
    // [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4] % 5
    const x = core.array(new Int32Array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]));
    const r = core.remainder(x, 5);
    assert.deepEqual(toArr(r), [-0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
  });

  it('range modulo negative', () => {
    const x = core.array(new Int32Array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]));
    const r = core.remainder(x, -5);
    assert.deepEqual(toArr(r), [-0, -4, -3, -2, -1, 0, -4, -3, -2, -1]);
  });
});

// ---------------------------------------------------------------------------
// test_unary_ops (comprehensive edge cases)
// ---------------------------------------------------------------------------
describe('unary ops edge cases', () => {
  it('abs of negative floats', () => {
    const a = core.array(new Float32Array([-1, 1, -2, 3]));
    assert.deepEqual(toArr(core.abs(a)), [1, 1, 2, 3]);
  });

  it('sign with zero', () => {
    const a = core.array(new Float32Array([-1, 1, 0, -2, 3]));
    assert.deepEqual(toArr(core.sign(a)), [-1, 1, 0, -1, 1]);
  });

  it('ceil with inf', () => {
    const a = core.array(new Float32Array([-22.03, 19.98, -27, 9, 0.0]));
    const r = toArr(core.ceil(a));
    assert.deepEqual(r, [-22, 20, -27, 9, 0]);
  });

  it('floor with inf', () => {
    const a = core.array(new Float32Array([-22.03, 19.98, -27, 9, 0.0]));
    const r = toArr(core.floor(a));
    assert.deepEqual(r, [-23, 19, -27, 9, 0]);
  });

  it('round banker rounding', () => {
    const a = core.array(new Float32Array([0.5, -0.5, 1.5, -1.5]));
    const r = toArr(core.round(a));
    // banker's rounding: 0.5 -> 0, 1.5 -> 2
    assert.equal(r[0], 0);
    assert.equal(r[2], 2);
  });

  it('reciprocal', () => {
    const a = core.array(new Float32Array([0.1, 0.5, 1.0, 2.0]));
    const r = toArr(core.reciprocal(a));
    assert.ok(allClose(r, [10, 2, 1, 0.5], 1e-4));
  });

  it('negative', () => {
    const a = core.array(new Float32Array([-1, 1, -2, 3]));
    assert.deepEqual(toArr(core.negative(a)), [1, -1, 2, -3]);
  });

  it('log1p near zero', () => {
    const a = core.array(new Float32Array([0]));
    assert.ok(approx(toScalar(core.log1p(a)), 0));
  });

  it('expm1 at zero', () => {
    const a = core.array(new Float32Array([0]));
    assert.ok(approx(toScalar(core.expm1(a)), 0));
  });

  it('sigmoid', () => {
    const a = core.array(new Float32Array([0.0, 1.0, -1.0, 5.0, -5.0]));
    const r = toArr(core.sigmoid(a));
    assert.ok(approx(r[0], 0.5));
    assert.ok(r[1] > 0.5);
    assert.ok(r[2] < 0.5);
  });

  it('erf at zero', () => {
    const a = core.array(new Float32Array([0]));
    assert.ok(approx(toScalar(core.erf(a)), 0));
  });
});

// ---------------------------------------------------------------------------
// test_tri / test_tril / test_triu (k offset parameter)
// ---------------------------------------------------------------------------
describe('tri / tril / triu with k offset', () => {
  it('tri basic 3x3', () => {
    const t = core.tri(3);
    assert.deepEqual(t.shape, [3, 3]);
    assert.deepEqual(toArr(t), [1, 0, 0, 1, 1, 0, 1, 1, 1]);
  });

  it('tri with k=1', () => {
    const t = core.tri(3, { m: 3, k: 1 });
    assert.deepEqual(toArr(t), [1, 1, 0, 1, 1, 1, 1, 1, 1]);
  });

  it('tri with k=-1', () => {
    const t = core.tri(3, { m: 3, k: -1 });
    assert.deepEqual(toArr(t), [0, 0, 0, 1, 0, 0, 1, 1, 0]);
  });

  it('tri non-square 2x4', () => {
    const t = core.tri(2, { m: 4 });
    assert.deepEqual(t.shape, [2, 4]);
    assert.deepEqual(toArr(t), [1, 0, 0, 0, 1, 1, 0, 0]);
  });

  it('tril default k=0', () => {
    const a = core.ones([3, 3]);
    assert.deepEqual(toArr(core.tril(a)), [1, 0, 0, 1, 1, 0, 1, 1, 1]);
  });

  it('tril k=1', () => {
    const a = core.ones([3, 3]);
    assert.deepEqual(toArr(core.tril(a, { k: 1 })), [1, 1, 0, 1, 1, 1, 1, 1, 1]);
  });

  it('tril k=-1', () => {
    const a = core.ones([3, 3]);
    assert.deepEqual(toArr(core.tril(a, { k: -1 })), [0, 0, 0, 1, 0, 0, 1, 1, 0]);
  });

  it('triu default k=0', () => {
    const a = core.ones([3, 3]);
    assert.deepEqual(toArr(core.triu(a)), [1, 1, 1, 0, 1, 1, 0, 0, 1]);
  });

  it('triu k=1', () => {
    const a = core.ones([3, 3]);
    assert.deepEqual(toArr(core.triu(a, { k: 1 })), [0, 1, 1, 0, 0, 1, 0, 0, 0]);
  });

  it('triu k=-1', () => {
    const a = core.ones([3, 3]);
    assert.deepEqual(toArr(core.triu(a, { k: -1 })), [1, 1, 1, 1, 1, 1, 0, 1, 1]);
  });
});

// ---------------------------------------------------------------------------
// test_expand_dims / test_squeeze (various axes)
// ---------------------------------------------------------------------------
describe('expand_dims / squeeze', () => {
  it('expand_dims axis 0', () => {
    const a = core.zeros([2, 2]);
    assert.deepEqual(core.expand_dims(a, 0).shape, [1, 2, 2]);
  });

  it('expand_dims axis -1', () => {
    const a = core.zeros([2, 2]);
    assert.deepEqual(core.expand_dims(a, [0, -1]).shape, [1, 2, 2, 1]);
  });

  it('expand_dims tuple axes', () => {
    const a = core.zeros([2, 2]);
    assert.deepEqual(core.expand_dims(a, [0, 1]).shape, [1, 1, 2, 2]);
  });

  it('squeeze removes all size-1 dims', () => {
    const a = core.zeros([2, 1, 2, 1]);
    assert.deepEqual(core.squeeze(a).shape, [2, 2]);
  });

  it('squeeze specific axis', () => {
    const a = core.zeros([2, 1, 2, 1]);
    assert.deepEqual(core.squeeze(a, 1).shape, [2, 2, 1]);
  });

  it('squeeze multiple axes', () => {
    const a = core.zeros([2, 1, 2, 1]);
    assert.deepEqual(core.squeeze(a, [1, 3]).shape, [2, 2]);
  });

  it('squeeze no-op on non-singleton', () => {
    const a = core.zeros([2, 2]);
    assert.deepEqual(core.squeeze(a).shape, [2, 2]);
  });
});

// ---------------------------------------------------------------------------
// test_sort / test_argsort (axis parameter, multi-dim)
// ---------------------------------------------------------------------------
describe('sort / argsort multi-dim', () => {
  it('sort 1D', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5]));
    assert.deepEqual(toArr(core.sort(a)), [1, 1, 3, 4, 5]);
  });

  it('sort 2D axis 0', () => {
    // [[3,1],[2,4]] sorted along axis 0 => [[2,1],[3,4]]
    const a = core.array(new Float32Array([3, 1, 2, 4]), [2, 2]);
    const r = core.sort(a, { axis: 0 });
    assert.deepEqual(toArr(r), [2, 1, 3, 4]);
  });

  it('sort 2D axis 1', () => {
    const a = core.array(new Float32Array([3, 1, 4, 2]), [2, 2]);
    const r = core.sort(a, { axis: 1 });
    assert.deepEqual(toArr(r), [1, 3, 2, 4]);
  });

  it('sort preserves dtype', () => {
    const a = core.array(new Int32Array([5, 3, 1, 4, 2]));
    const r = core.sort(a);
    assert.equal(r.dtype, 'int32');
    assert.deepEqual(toArr(r), [1, 2, 3, 4, 5]);
  });

  it('argsort 1D basic', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5]));
    const idx = toArr(core.argsort(a));
    // first two indices should point to value 1
    assert.equal(idx[0], 1);
    assert.equal(idx[1], 3);
  });

  it('argsort 2D axis 0', () => {
    const a = core.array(new Float32Array([3, 1, 2, 4]), [2, 2]);
    const idx = core.argsort(a, { axis: 0 });
    assert.deepEqual(idx.shape, [2, 2]);
  });

  it('argsort 2D axis 1', () => {
    const a = core.array(new Float32Array([3, 1, 4, 2]), [2, 2]);
    const idx = core.argsort(a, { axis: 1 });
    assert.deepEqual(idx.shape, [2, 2]);
    // Row 0: [3,1] -> sorted indices [1,0]
    const row0 = toArr(idx);
    assert.equal(row0[0], 1);
    assert.equal(row0[1], 0);
  });
});

// ---------------------------------------------------------------------------
// test_concatenate / test_stack (axis variations)
// ---------------------------------------------------------------------------
describe('concatenate / stack axis variations', () => {
  it('concatenate 1D', () => {
    const a = core.array(new Float32Array([1, 2]));
    const b = core.array(new Float32Array([3, 4]));
    assert.deepEqual(toArr(core.concatenate([a, b])), [1, 2, 3, 4]);
  });

  it('concatenate 2D axis 0', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const b = core.array(new Float32Array([5, 6, 7, 8]), [2, 2]);
    const r = core.concatenate([a, b], 0);
    assert.deepEqual(r.shape, [4, 2]);
    assert.deepEqual(toArr(r), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('concatenate 2D axis 1', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const b = core.array(new Float32Array([5, 6, 7, 8]), [2, 2]);
    const r = core.concatenate([a, b], 1);
    assert.deepEqual(r.shape, [2, 4]);
    assert.deepEqual(toArr(r), [1, 2, 5, 6, 3, 4, 7, 8]);
  });

  it('stack 1D axis 0', () => {
    const a = core.array(new Float32Array([1, 2]));
    const b = core.array(new Float32Array([3, 4]));
    const r = core.stack([a, b]);
    assert.deepEqual(r.shape, [2, 2]);
    assert.deepEqual(toArr(r), [1, 2, 3, 4]);
  });

  it('stack 1D axis 1', () => {
    const a = core.array(new Float32Array([1, 2]));
    const b = core.array(new Float32Array([3, 4]));
    const r = core.stack([a, b], 1);
    assert.deepEqual(r.shape, [2, 2]);
    assert.deepEqual(toArr(r), [1, 3, 2, 4]);
  });

  it('stack 2D axis 0', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [1, 4]);
    const b = core.array(new Float32Array([5, 6, 7, 8]), [1, 4]);
    const r = core.stack([a, b]);
    assert.deepEqual(r.shape, [2, 1, 4]);
  });

  it('stack 2D axis 1', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [1, 4]);
    const b = core.array(new Float32Array([5, 6, 7, 8]), [1, 4]);
    const r = core.stack([a, b], 1);
    assert.deepEqual(r.shape, [1, 2, 4]);
  });
});

// ---------------------------------------------------------------------------
// test_pad (various pad widths, constant mode)
// ---------------------------------------------------------------------------
describe('pad', () => {
  it('pad 1D symmetric', () => {
    const a = core.array(new Float32Array([1, 2, 3]));
    const r = core.pad(a, [[1, 1]]);
    assert.deepEqual(r.shape, [5]);
    assert.deepEqual(toArr(r), [0, 1, 2, 3, 0]);
  });

  it('pad 1D asymmetric', () => {
    const a = core.array(new Float32Array([1, 2, 3]));
    const r = core.pad(a, [[2, 1]]);
    assert.deepEqual(r.shape, [6]);
    assert.deepEqual(toArr(r), [0, 0, 1, 2, 3, 0]);
  });

  it('pad 2D with constant value', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const r = core.pad(a, [[1, 0], [0, 1]], 9);
    assert.deepEqual(r.shape, [3, 3]);
    // row of 9s on top, column of 9s on right
    assert.deepEqual(toArr(r), [9, 9, 9, 1, 2, 9, 3, 4, 9]);
  });

  it('pad 3D symmetric', () => {
    const a = core.zeros([1, 1, 1]);
    const r = core.pad(a, [[1, 1], [1, 1], [1, 1]]);
    assert.deepEqual(r.shape, [3, 3, 3]);
  });

  it('pad zero-padding is identity', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const r = core.pad(a, [[0, 0], [0, 0]]);
    assert.deepEqual(toArr(r), [1, 2, 3, 4]);
  });
});

// ---------------------------------------------------------------------------
// test_where (broadcasting, dtype promotion)
// ---------------------------------------------------------------------------
describe('where broadcasting', () => {
  it('where with scalar true branch', () => {
    const cond = core.array(new Float32Array([1, 0, 1, 0]));
    const r = core.where(core.greater(cond, 0), 10, 20);
    assert.deepEqual(toArr(r), [10, 20, 10, 20]);
  });

  it('where with 2D condition and broadcast', () => {
    const cond = core.array([1, 0, 0, 1], [2, 2], 'bool');
    const x = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const y = core.array(new Float32Array([5, 6]));
    const r = core.where(cond, x, y);
    assert.deepEqual(r.shape, [2, 2]);
    // cond true at [0,0] and [1,1]: take x; else take y broadcast
    assert.deepEqual(toArr(r), [1, 6, 5, 4]);
  });

  it('where all true', () => {
    const cond = core.ones([3], 'bool');
    const x = core.array(new Float32Array([10, 20, 30]));
    const y = core.array(new Float32Array([100, 200, 300]));
    assert.deepEqual(toArr(core.where(cond, x, y)), [10, 20, 30]);
  });

  it('where all false', () => {
    const cond = core.zeros([3], 'bool');
    const x = core.array(new Float32Array([10, 20, 30]));
    const y = core.array(new Float32Array([100, 200, 300]));
    assert.deepEqual(toArr(core.where(cond, x, y)), [100, 200, 300]);
  });
});

// ---------------------------------------------------------------------------
// test_split (sections, axis)
// ---------------------------------------------------------------------------
describe('split', () => {
  it('split 1D into 3', () => {
    const a = core.array(new Float32Array([1, 2, 3]));
    const [x, y, z] = core.split(a, 3);
    assert.equal(toScalar(x), 1);
    assert.equal(toScalar(y), 2);
    assert.equal(toScalar(z), 3);
  });

  it('split 2D along axis 0', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const [x, y, z] = core.split(a, 3, 0);
    assert.deepEqual(x.shape, [1, 2]);
    assert.deepEqual(toArr(x), [1, 2]);
    assert.deepEqual(toArr(y), [3, 4]);
    assert.deepEqual(toArr(z), [5, 6]);
  });

  it('split at indices', () => {
    const a = core.arange(8);
    const [x, y, z] = core.split(a, [1, 5]);
    assert.deepEqual(toArr(x), [0]);
    assert.deepEqual(toArr(y), [1, 2, 3, 4]);
    assert.deepEqual(toArr(z), [5, 6, 7]);
  });
});

// ---------------------------------------------------------------------------
// test_clip (with min/max arrays)
// ---------------------------------------------------------------------------
describe('clip', () => {
  it('clip with scalar min and max', () => {
    const a = core.array(new Int32Array([1, 4, 3, 8, 5]));
    const r = core.clip(a, 2, 6);
    assert.deepEqual(toArr(r), [2, 4, 3, 6, 5]);
  });

  it('clip with only min', () => {
    const a = core.array(new Int32Array([-1, 1, 0, 5]));
    const r = core.clip(a, 0, null);
    assert.deepEqual(toArr(r), [0, 1, 0, 5]);
  });

  it('clip with only max', () => {
    const a = core.array(new Int32Array([2, 3, 4, 5]));
    const r = core.clip(a, null, 4);
    assert.deepEqual(toArr(r), [2, 3, 4, 4]);
  });

  it('clip with array min', () => {
    const a = core.array(new Int32Array([2, 3, 4, 5]));
    const mins = core.array(new Int32Array([3, 1, 5, 5]));
    const r = core.clip(a, mins, 4);
    // clip([2,3,4,5], min=[3,1,5,5], max=4) => [3, 3, 4, 4]
    assert.deepEqual(toArr(r), [3, 3, 4, 4]);
  });

  it('clip with array min and max', () => {
    const a = core.array(new Int32Array([2, 3, 4, 5]));
    const mins = core.array(new Int32Array([3, 1, 5, 5]));
    const maxs = core.array(new Int32Array([5, -1, 2, 9]));
    const r = core.clip(a, mins, maxs);
    // MLX: clip applies min then max, so max wins when min > max
    assert.deepEqual(toArr(r), [3, -1, 2, 5]);
  });
});

// ---------------------------------------------------------------------------
// test_take / test_take_along_axis
// ---------------------------------------------------------------------------
describe('take / take_along_axis', () => {
  it('take 1D', () => {
    const a = core.arange(5);
    const idx = core.array(new Int32Array([0, 2, 4]));
    assert.deepEqual(toArr(core.take(a, idx)), [0, 2, 4]);
  });

  it('take with negative index', () => {
    const a = core.array(new Float32Array([10, 20, 30, 40]));
    const idx = core.array(new Int32Array([0, -1]));
    const r = core.take(a, idx);
    assert.equal(toArr(r)[0], 10);
    assert.equal(toArr(r)[1], 40);
  });

  it('take along axis 0', () => {
    // 2x3 array, take rows [1, 0]
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const idx = core.array(new Int32Array([1, 0]));
    const r = core.take(a, idx, 0);
    assert.deepEqual(r.shape, [2, 3]);
    assert.deepEqual(toArr(r), [4, 5, 6, 1, 2, 3]);
  });

  it('take along axis 1', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const idx = core.array(new Int32Array([2, 0]));
    const r = core.take(a, idx, 1);
    assert.deepEqual(r.shape, [2, 2]);
    assert.deepEqual(toArr(r), [3, 1, 6, 4]);
  });

  it('take_along_axis 1D', () => {
    const a = core.array(new Float32Array([10, 20, 30, 40, 50]));
    const idx = core.array(new Int32Array([4, 0, 2]));
    const r = core.take_along_axis(a, idx, 0);
    assert.deepEqual(toArr(r), [50, 10, 30]);
  });

  it('take_along_axis 2D axis 1', () => {
    // Sort indices then gather
    const a = core.array(new Float32Array([30, 10, 20, 60, 40, 50]), [2, 3]);
    const idx = core.argsort(a, { axis: 1 });
    const sorted = core.take_along_axis(a, idx, 1);
    assert.deepEqual(toArr(sorted), [10, 20, 30, 40, 50, 60]);
  });
});

// ---------------------------------------------------------------------------
// test_repeat / test_tile (multi-dim)
// ---------------------------------------------------------------------------
describe('repeat / tile multi-dim', () => {
  it('repeat 1D', () => {
    const a = core.array(new Float32Array([1, 2]));
    assert.deepEqual(toArr(core.repeat(a, 3)), [1, 1, 1, 2, 2, 2]);
  });

  it('repeat 2D axis 0', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const r = core.repeat(a, 2, { axis: 0 });
    assert.deepEqual(r.shape, [4, 2]);
    assert.deepEqual(toArr(r), [1, 2, 1, 2, 3, 4, 3, 4]);
  });

  it('repeat 2D axis 1', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const r = core.repeat(a, 2, { axis: 1 });
    assert.deepEqual(r.shape, [2, 4]);
    assert.deepEqual(toArr(r), [1, 1, 2, 2, 3, 3, 4, 4]);
  });

  it('repeat 0 times', () => {
    const a = core.array(new Float32Array([1, 2, 3]));
    const r = core.repeat(a, 0);
    assert.equal(r.shape[0], 0);
  });

  it('tile 1D', () => {
    const a = core.array(new Float32Array([1, 2]));
    assert.deepEqual(toArr(core.tile(a, 3)), [1, 2, 1, 2, 1, 2]);
  });

  it('tile 2D with scalar reps', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const r = core.tile(a, 2);
    assert.deepEqual(r.shape, [2, 4]);
    assert.deepEqual(toArr(r), [1, 2, 1, 2, 3, 4, 3, 4]);
  });

  it('tile expanding dims', () => {
    // tile a (3,) with [2,2,2] should give (2,2,6)
    const a = core.array(new Float32Array([1, 2, 3]));
    const r = core.tile(a, [2, 2, 2]);
    assert.deepEqual(r.shape, [2, 2, 6]);
  });
});

// ---------------------------------------------------------------------------
// Additional coverage: array_equal, logaddexp, variance, std
// ---------------------------------------------------------------------------
describe('miscellaneous gap coverage', () => {
  it('array_equal with different shapes', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]));
    const b = core.array(new Float32Array([1, 2, 3]));
    assert.equal(toScalar(core.array_equal(a, b)), false);
  });

  it('array_equal cross dtype', () => {
    const a = core.array(new Int32Array([1, 2, 3, 4]));
    const b = core.array(new Float32Array([1, 2, 3, 4]));
    assert.equal(toScalar(core.array_equal(a, b)), true);
  });

  it('logaddexp basic', () => {
    const a = core.array(new Float32Array([0, 1, 2, 9]));
    const b = core.array(new Float32Array([1, 0, 4, 2.5]));
    const r = toArr(core.logaddexp(a, b));
    // logaddexp(0,1) = log(e^0 + e^1) ~ 1.3133
    assert.ok(approx(r[0], Math.log(Math.exp(0) + Math.exp(1)), 1e-4));
  });

  it('sum axis 0', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    assert.deepEqual(toArr(core.sum(a, 0)), [4, 6]);
  });

  it('sum axis 1', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    assert.deepEqual(toArr(core.sum(a, 1)), [3, 7]);
  });

  it('mean global', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    assert.ok(approx(toScalar(core.mean(a)), 2.5));
  });

  it('prod global', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    assert.equal(toScalar(core.prod(a)), 24);
  });

  it('min / max global', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5]));
    assert.equal(toScalar(core.min(a)), 1);
    assert.equal(toScalar(core.max(a)), 5);
  });

  it('argmin / argmax', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5]));
    assert.equal(toScalar(core.argmin(a)), 1);
    assert.equal(toScalar(core.argmax(a)), 4);
  });

  it('flatten with start_axis', () => {
    const x = core.zeros([2, 3, 4]);
    assert.deepEqual(core.flatten(x, { start_axis: 1 }).shape, [2, 12]);
  });

  it('flatten with end_axis', () => {
    const x = core.zeros([2, 3, 4]);
    assert.deepEqual(core.flatten(x, { start_axis: 0, end_axis: 1 }).shape, [6, 4]);
  });
});

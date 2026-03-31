import { strict as assert } from 'assert';
import { core } from '../../src';

const toScalar = (t: any) => t.toArray()[0];
const approx = (a: number, b: number, eps = 1e-5) => Math.abs(a - b) < eps;

describe('batch ops: unary math', () => {
  it('ceil', () => {
    const a = core.array(new Float32Array([1.2, 2.7, -1.5]), [3], 'float32');
    assert.deepEqual(core.ceil(a).toArray(), [2, 3, -1]);
  });

  it('floor', () => {
    const a = core.array(new Float32Array([1.9, 2.1, -0.5]), [3], 'float32');
    assert.deepEqual(core.floor(a).toArray(), [1, 2, -1]);
  });

  it('round', () => {
    const a = core.array(new Float32Array([1.5, 2.3, -0.6]), [3], 'float32');
    const r = core.round(a).toArray();
    assert.equal(r[0], 2);
    assert.equal(r[1], 2);
    assert.equal(r[2], -1);
  });

  it('isnan', () => {
    const a = core.array(new Float32Array([1.0, NaN, 3.0]), [3], 'float32');
    assert.deepEqual(core.isnan(a).toArray(), [false, true, false]);
  });

  it('isinf', () => {
    const a = core.array(new Float32Array([1.0, Infinity, -Infinity]), [3], 'float32');
    assert.deepEqual(core.isinf(a).toArray(), [false, true, true]);
  });

  it('isfinite', () => {
    const a = core.array(new Float32Array([1.0, Infinity, NaN]), [3], 'float32');
    assert.deepEqual(core.isfinite(a).toArray(), [true, false, false]);
  });

  it('logical_not', () => {
    const a = core.array([1, 0, 1, 0], [4], 'bool');
    assert.deepEqual(core.logical_not(a).toArray(), [false, true, false, true]);
  });

  it('sinh', () => {
    const a = core.array(new Float32Array([0]), [1], 'float32');
    assert.ok(approx(toScalar(core.sinh(a)), 0));
  });

  it('cosh', () => {
    const a = core.array(new Float32Array([0]), [1], 'float32');
    assert.ok(approx(toScalar(core.cosh(a)), 1));
  });

  it('arcsinh', () => {
    const a = core.array(new Float32Array([0]), [1], 'float32');
    assert.ok(approx(toScalar(core.arcsinh(a)), 0));
  });

  it('arccosh', () => {
    const a = core.array(new Float32Array([1]), [1], 'float32');
    assert.ok(approx(toScalar(core.arccosh(a)), 0));
  });

  it('arctanh', () => {
    const a = core.array(new Float32Array([0]), [1], 'float32');
    assert.ok(approx(toScalar(core.arctanh(a)), 0));
  });

  it('degrees', () => {
    const a = core.array(new Float32Array([Math.PI]), [1], 'float32');
    assert.ok(approx(toScalar(core.degrees(a)), 180));
  });

  it('radians', () => {
    const a = core.array(new Float32Array([180]), [1], 'float32');
    assert.ok(approx(toScalar(core.radians(a)), Math.PI));
  });

  it('expm1', () => {
    const a = core.array(new Float32Array([0]), [1], 'float32');
    assert.ok(approx(toScalar(core.expm1(a)), 0));
  });

  it('erfinv', () => {
    const a = core.array(new Float32Array([0]), [1], 'float32');
    assert.ok(approx(toScalar(core.erfinv(a)), 0));
  });
});

describe('batch ops: cumulative', () => {
  it('cumsum along axis 0', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [4], 'float32');
    assert.deepEqual(core.cumsum(a).toArray(), [1, 3, 6, 10]);
  });

  it('cumprod along axis 0', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [4], 'float32');
    assert.deepEqual(core.cumprod(a).toArray(), [1, 2, 6, 24]);
  });

  it('cummax along axis 0', () => {
    const a = core.array(new Float32Array([3, 1, 4, 2]), [4], 'float32');
    assert.deepEqual(core.cummax(a).toArray(), [3, 3, 4, 4]);
  });

  it('cummin along axis 0', () => {
    const a = core.array(new Float32Array([3, 1, 4, 2]), [4], 'float32');
    assert.deepEqual(core.cummin(a).toArray(), [3, 1, 1, 1]);
  });

  it('cumsum with reverse', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3], 'float32');
    assert.deepEqual(core.cumsum(a, { reverse: true }).toArray(), [6, 5, 3]);
  });
});

describe('batch ops: binary', () => {
  it('floor_divide', () => {
    const a = core.array(new Float32Array([7, 8, 9]), [3], 'float32');
    const b = core.array(new Float32Array([2, 3, 4]), [3], 'float32');
    assert.deepEqual(core.floor_divide(a, b).toArray(), [3, 2, 2]);
  });

  it('remainder', () => {
    const a = core.array(new Float32Array([7, 8, 9]), [3], 'float32');
    const b = core.array(new Float32Array([2, 3, 4]), [3], 'float32');
    assert.deepEqual(core.remainder(a, b).toArray(), [1, 2, 1]);
  });

  it('logical_and', () => {
    const a = core.array([1, 1, 0, 0], [4], 'bool');
    const b = core.array([1, 0, 1, 0], [4], 'bool');
    assert.deepEqual(core.logical_and(a, b).toArray(), [true, false, false, false]);
  });

  it('logical_or', () => {
    const a = core.array([1, 1, 0, 0], [4], 'bool');
    const b = core.array([1, 0, 1, 0], [4], 'bool');
    assert.deepEqual(core.logical_or(a, b).toArray(), [true, true, true, false]);
  });

  it('bitwise_and', () => {
    const a = core.array(new Int32Array([0b1100, 0b1010]), [2], 'int32');
    const b = core.array(new Int32Array([0b1010, 0b1100]), [2], 'int32');
    assert.deepEqual(core.bitwise_and(a, b).toArray(), [0b1000, 0b1000]);
  });

  it('bitwise_or', () => {
    const a = core.array(new Int32Array([0b1100, 0b1010]), [2], 'int32');
    const b = core.array(new Int32Array([0b1010, 0b1100]), [2], 'int32');
    assert.deepEqual(core.bitwise_or(a, b).toArray(), [0b1110, 0b1110]);
  });

  it('bitwise_xor', () => {
    const a = core.array(new Int32Array([0b1100, 0b1010]), [2], 'int32');
    const b = core.array(new Int32Array([0b1010, 0b1100]), [2], 'int32');
    assert.deepEqual(core.bitwise_xor(a, b).toArray(), [0b0110, 0b0110]);
  });

  it('left_shift', () => {
    const a = core.array(new Int32Array([1, 2, 3]), [3], 'int32');
    const b = core.array(new Int32Array([1, 2, 3]), [3], 'int32');
    assert.deepEqual(core.left_shift(a, b).toArray(), [2, 8, 24]);
  });

  it('right_shift', () => {
    const a = core.array(new Int32Array([8, 16, 32]), [3], 'int32');
    const b = core.array(new Int32Array([1, 2, 3]), [3], 'int32');
    assert.deepEqual(core.right_shift(a, b).toArray(), [4, 4, 4]);
  });
});

describe('batch ops: reduction & query', () => {
  it('all (global)', () => {
    const a = core.array([1, 1, 1, 1], [4], 'bool');
    assert.equal(toScalar(core.all(a)), true);
  });

  it('all with false', () => {
    const a = core.array([1, 0, 1, 1], [4], 'bool');
    assert.equal(toScalar(core.all(a)), false);
  });

  it('any (global)', () => {
    const a = core.array([0, 0, 0, 1], [4], 'bool');
    assert.equal(toScalar(core.any(a)), true);
  });

  it('any all false', () => {
    const a = core.array([0, 0, 0, 0], [4], 'bool');
    assert.equal(toScalar(core.any(a)), false);
  });

  it('array_equal', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3], 'float32');
    const b = core.array(new Float32Array([1, 2, 3]), [3], 'float32');
    assert.equal(toScalar(core.array_equal(a, b)), true);
  });

  it('array_equal (not equal)', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3], 'float32');
    const b = core.array(new Float32Array([1, 2, 4]), [3], 'float32');
    assert.equal(toScalar(core.array_equal(a, b)), false);
  });
});

describe('batch ops: shape & creation', () => {
  it('flatten', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3], 'float32');
    const f = core.flatten(a);
    assert.deepEqual(f.shape, [6]);
    assert.deepEqual(f.toArray(), [1, 2, 3, 4, 5, 6]);
  });

  it('eye', () => {
    const e = core.eye(3);
    assert.deepEqual(e.shape, [3, 3]);
    assert.deepEqual(e.toArray(), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('eye with m != n', () => {
    const e = core.eye(2, { m: 3 });
    assert.deepEqual(e.shape, [2, 3]);
    assert.deepEqual(e.toArray(), [1, 0, 0, 0, 1, 0]);
  });

  it('identity', () => {
    const id = core.identity(3);
    assert.deepEqual(id.shape, [3, 3]);
    assert.deepEqual(id.toArray(), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('linspace', () => {
    const l = core.linspace(0, 1, { num: 5 });
    assert.deepEqual(l.shape, [5]);
    const arr = l.toArray();
    assert.ok(approx(arr[0], 0));
    assert.ok(approx(arr[2], 0.5));
    assert.ok(approx(arr[4], 1));
  });

  it('tril', () => {
    const a = core.ones([3, 3]);
    const t = core.tril(a);
    assert.deepEqual(t.toArray(), [1, 0, 0, 1, 1, 0, 1, 1, 1]);
  });

  it('triu', () => {
    const a = core.ones([3, 3]);
    const t = core.triu(a);
    assert.deepEqual(t.toArray(), [1, 1, 1, 0, 1, 1, 0, 0, 1]);
  });

  it('broadcast_to', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [1, 3], 'float32');
    const b = core.broadcast_to(a, [2, 3]);
    assert.deepEqual(b.shape, [2, 3]);
    assert.deepEqual(b.toArray(), [1, 2, 3, 1, 2, 3]);
  });

  it('repeat', () => {
    const a = core.array(new Float32Array([1, 2]), [2], 'float32');
    const r = core.repeat(a, 3);
    assert.deepEqual(r.toArray(), [1, 1, 1, 2, 2, 2]);
  });

  it('tile', () => {
    const a = core.array(new Float32Array([1, 2]), [2], 'float32');
    const t = core.tile(a, 3);
    assert.deepEqual(t.toArray(), [1, 2, 1, 2, 1, 2]);
  });

  it('sort', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5]), [5], 'float32');
    assert.deepEqual(core.sort(a).toArray(), [1, 1, 3, 4, 5]);
  });

  it('argsort', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5]), [5], 'float32');
    const idx = core.argsort(a);
    // indices of sorted order
    const arr = idx.toArray();
    assert.equal(arr[0], 1); // value 1
    assert.equal(arr[1], 3); // value 1
    assert.equal(arr[2], 0); // value 3
  });

  it('diag', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3], 'float32');
    const d = core.diag(a);
    assert.deepEqual(d.shape, [3, 3]);
    assert.deepEqual(d.toArray(), [1, 0, 0, 0, 2, 0, 0, 0, 3]);
  });

  it('diagonal', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]), [3, 3], 'float32');
    const d = core.diagonal(a);
    assert.deepEqual(d.toArray(), [1, 5, 9]);
  });

  it('topk', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]), [8], 'float32');
    const t = core.topk(a, 3);
    const arr = t.toArray();
    // top 3 values sorted descending
    assert.ok(arr.includes(9));
    assert.ok(arr.includes(6));
    assert.ok(arr.includes(5));
  });
});

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import * as core from '../../src/index';

const approx = (a: number, b: number, tol = 1e-5) => Math.abs(a - b) < tol;

const toArr = (x: any): any => x.toArray();
const toFlat = (x: any): number[] => (x.toArray() as number[]).flat(Infinity) as number[];

describe('reduce: sum', () => {
  it('sum over entire array (no axis)', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s = core.sum(x);
    assert.ok(approx(toArr(s)[0], 21));
  });

  it('sum axis=0 on 2D', () => {
    // [[1,2,3],[4,5,6]] -> sum axis=0 -> [5,7,9]
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s = core.sum(x, 0);
    assert.deepEqual(s.shape, [3]);
    const r = toArr(s);
    assert.ok(approx(r[0], 5));
    assert.ok(approx(r[1], 7));
    assert.ok(approx(r[2], 9));
  });

  it('sum axis=1 on 2D', () => {
    // [[1,2,3],[4,5,6]] -> sum axis=1 -> [6,15]
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s = core.sum(x, 1);
    assert.deepEqual(s.shape, [2]);
    const r = toArr(s);
    assert.ok(approx(r[0], 6));
    assert.ok(approx(r[1], 15));
  });

  it('sum axis=-1 is equivalent to last axis', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s1 = core.sum(x, -1);
    const s2 = core.sum(x, 1);
    assert.deepEqual(toArr(s1), toArr(s2));
  });

  it('sum with keepdims=true', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s = core.sum(x, 0, { keepdims: true });
    assert.deepEqual(s.shape, [1, 3]);
    const r = toFlat(s);
    assert.ok(approx(r[0], 5));
    assert.ok(approx(r[1], 7));
    assert.ok(approx(r[2], 9));
  });

  it('sum with keepdims on axis=1', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s = core.sum(x, 1, { keepdims: true });
    assert.deepEqual(s.shape, [2, 1]);
  });

  it('sum over multiple axes simultaneously', () => {
    // 2x3x4 array
    const data = new Float32Array(24);
    for (let i = 0; i < 24; i++) data[i] = i + 1;
    const x = core.array(data, [2, 3, 4]);
    // Sum over axes [0, 2] -> shape [3]
    const s = core.sum(x, [0, 2]);
    assert.deepEqual(s.shape, [3]);
    // axis 0,2 collapse: row0 = sum of elements [1..4]+[13..16] = 10+58 = 68
    // Actually let's compute: slice [0,:,:] is 1..12, slice [1,:,:] is 13..24
    // [0,0,:] = [1,2,3,4] sum=10; [1,0,:] = [13,14,15,16] sum=58; total axis(0,2) for j=0: 68
    // [0,1,:] = [5,6,7,8] sum=26; [1,1,:] = [17,18,19,20] sum=74; total: 100
    // [0,2,:] = [9,10,11,12] sum=42; [1,2,:] = [21,22,23,24] sum=90; total: 132
    const r = toArr(s);
    assert.ok(approx(r[0], 68));
    assert.ok(approx(r[1], 100));
    assert.ok(approx(r[2], 132));
  });

  it('sum on int32 dtype', () => {
    const x = core.array(new Int32Array([10, 20, 30, 40]), [2, 2]);
    const s = core.sum(x);
    assert.equal(toArr(s)[0], 100);
  });

  it('sum single element array', () => {
    const x = core.array(new Float32Array([42]), [1]);
    const s = core.sum(x);
    assert.ok(approx(toArr(s)[0], 42));
  });

  it('sum all same values', () => {
    const data = new Float32Array(12).fill(3);
    const x = core.array(data, [3, 4]);
    const s = core.sum(x);
    assert.ok(approx(toArr(s)[0], 36));
  });
});

describe('reduce: mean', () => {
  it('mean over entire array', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const m = core.mean(x);
    assert.ok(approx(toArr(m)[0], 3.5));
  });

  it('mean axis=0', () => {
    // [[1,2,3],[4,5,6]] -> mean axis=0 -> [2.5, 3.5, 4.5]
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const m = core.mean(x, 0);
    assert.deepEqual(m.shape, [3]);
    const r = toArr(m);
    assert.ok(approx(r[0], 2.5));
    assert.ok(approx(r[1], 3.5));
    assert.ok(approx(r[2], 4.5));
  });

  it('mean axis=1', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const m = core.mean(x, 1);
    assert.deepEqual(m.shape, [2]);
    const r = toArr(m);
    assert.ok(approx(r[0], 2));
    assert.ok(approx(r[1], 5));
  });

  it('mean with keepdims', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const m = core.mean(x, 0, { keepdims: true });
    assert.deepEqual(m.shape, [1, 3]);
  });
});

describe('reduce: min', () => {
  it('min over entire array', () => {
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m = core.min(x);
    assert.ok(approx(toArr(m)[0], 1));
  });

  it('min axis=0', () => {
    // [[5,1,8],[3,9,2]] -> min axis=0 -> [3,1,2]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m = core.min(x, 0);
    assert.deepEqual(m.shape, [3]);
    const r = toArr(m);
    assert.ok(approx(r[0], 3));
    assert.ok(approx(r[1], 1));
    assert.ok(approx(r[2], 2));
  });

  it('min axis=1', () => {
    // [[5,1,8],[3,9,2]] -> min axis=1 -> [1,2]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m = core.min(x, 1);
    assert.deepEqual(m.shape, [2]);
    const r = toArr(m);
    assert.ok(approx(r[0], 1));
    assert.ok(approx(r[1], 2));
  });

  it('min axis=-1', () => {
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m1 = core.min(x, -1);
    const m2 = core.min(x, 1);
    assert.deepEqual(toArr(m1), toArr(m2));
  });

  it('min on int32', () => {
    const x = core.array(new Int32Array([10, -5, 3, 7]), [2, 2]);
    const m = core.min(x);
    assert.equal(toArr(m)[0], -5);
  });
});

describe('reduce: max', () => {
  it('max over entire array', () => {
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m = core.max(x);
    assert.ok(approx(toArr(m)[0], 9));
  });

  it('max axis=0', () => {
    // [[5,1,8],[3,9,2]] -> max axis=0 -> [5,9,8]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m = core.max(x, 0);
    assert.deepEqual(m.shape, [3]);
    const r = toArr(m);
    assert.ok(approx(r[0], 5));
    assert.ok(approx(r[1], 9));
    assert.ok(approx(r[2], 8));
  });

  it('max axis=1', () => {
    // [[5,1,8],[3,9,2]] -> max axis=1 -> [8,9]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m = core.max(x, 1);
    assert.deepEqual(m.shape, [2]);
    const r = toArr(m);
    assert.ok(approx(r[0], 8));
    assert.ok(approx(r[1], 9));
  });

  it('max with keepdims', () => {
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m = core.max(x, 1, { keepdims: true });
    assert.deepEqual(m.shape, [2, 1]);
    const r = toFlat(m);
    assert.ok(approx(r[0], 8));
    assert.ok(approx(r[1], 9));
  });

  it('max axis=-1', () => {
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const m1 = core.max(x, -1);
    const m2 = core.max(x, 1);
    assert.deepEqual(toArr(m1), toArr(m2));
  });
});

describe('reduce: prod', () => {
  it('prod over entire array', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const p = core.prod(x);
    assert.ok(approx(toArr(p)[0], 24));
  });

  it('prod axis=0', () => {
    // [[1,2],[3,4]] -> prod axis=0 -> [3,8]
    const x = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const p = core.prod(x, 0);
    assert.deepEqual(p.shape, [2]);
    const r = toArr(p);
    assert.ok(approx(r[0], 3));
    assert.ok(approx(r[1], 8));
  });

  it('prod axis=1', () => {
    // [[1,2],[3,4]] -> prod axis=1 -> [2,12]
    const x = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const p = core.prod(x, 1);
    assert.deepEqual(p.shape, [2]);
    const r = toArr(p);
    assert.ok(approx(r[0], 2));
    assert.ok(approx(r[1], 12));
  });

  it('prod with keepdims', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const p = core.prod(x, 0, { keepdims: true });
    assert.deepEqual(p.shape, [1, 2]);
    const r = toFlat(p);
    assert.ok(approx(r[0], 3));
    assert.ok(approx(r[1], 8));
  });

  it('prod on int32', () => {
    const x = core.array(new Int32Array([2, 3, 4, 5]), [2, 2]);
    const p = core.prod(x);
    assert.equal(toArr(p)[0], 120);
  });
});

describe('reduce: argmin / argmax', () => {
  it('argmin over flattened array', () => {
    // [5, 1, 8, 3, 9, 2] -> argmin = 1
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmin(x);
    assert.equal(toArr(idx)[0], 1);
  });

  it('argmax over flattened array', () => {
    // [5, 1, 8, 3, 9, 2] -> argmax = 4
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmax(x);
    assert.equal(toArr(idx)[0], 4);
  });

  it('argmin axis=0', () => {
    // [[5,1,8],[3,9,2]] -> argmin axis=0 -> [1,0,1]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmin(x, 0);
    assert.deepEqual(idx.shape, [3]);
    assert.deepEqual(toArr(idx), [1, 0, 1]);
  });

  it('argmin axis=1', () => {
    // [[5,1,8],[3,9,2]] -> argmin axis=1 -> [1,2]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmin(x, 1);
    assert.deepEqual(idx.shape, [2]);
    assert.deepEqual(toArr(idx), [1, 2]);
  });

  it('argmax axis=0', () => {
    // [[5,1,8],[3,9,2]] -> argmax axis=0 -> [0,1,0]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmax(x, 0);
    assert.deepEqual(idx.shape, [3]);
    assert.deepEqual(toArr(idx), [0, 1, 0]);
  });

  it('argmax axis=1', () => {
    // [[5,1,8],[3,9,2]] -> argmax axis=1 -> [2,1]
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmax(x, 1);
    assert.deepEqual(idx.shape, [2]);
    assert.deepEqual(toArr(idx), [2, 1]);
  });

  it('argmin with keepdims', () => {
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmin(x, 0, { keepdims: true });
    assert.deepEqual(idx.shape, [1, 3]);
    assert.deepEqual(toFlat(idx), [1, 0, 1]);
  });

  it('argmax with keepdims', () => {
    const x = core.array(new Float32Array([5, 1, 8, 3, 9, 2]), [2, 3]);
    const idx = core.argmax(x, 1, { keepdims: true });
    assert.deepEqual(idx.shape, [2, 1]);
    assert.deepEqual(toFlat(idx), [2, 1]);
  });

  it('argmin on int32', () => {
    const x = core.array(new Int32Array([10, -3, 7, 0]), [2, 2]);
    const idx = core.argmin(x);
    assert.equal(toArr(idx)[0], 1);
  });

  it('argmax on int32', () => {
    const x = core.array(new Int32Array([10, -3, 7, 0]), [2, 2]);
    const idx = core.argmax(x);
    assert.equal(toArr(idx)[0], 0);
  });
});

describe('reduce: 3D reductions', () => {
  // 2x3x4 array: values 1..24
  const make3D = () => {
    const data = new Float32Array(24);
    for (let i = 0; i < 24; i++) data[i] = i + 1;
    return core.array(data, [2, 3, 4]);
  };

  it('sum axis=0 on 3D', () => {
    const x = make3D();
    const s = core.sum(x, 0);
    assert.deepEqual(s.shape, [3, 4]);
    // [0,0,:] = [1,2,3,4], [1,0,:] = [13,14,15,16] -> sum = [14,16,18,20]
    const r = toFlat(s);
    assert.ok(approx(r[0], 14));
    assert.ok(approx(r[1], 16));
    assert.ok(approx(r[2], 18));
    assert.ok(approx(r[3], 20));
  });

  it('sum axis=2 on 3D', () => {
    const x = make3D();
    const s = core.sum(x, 2);
    assert.deepEqual(s.shape, [2, 3]);
    // [0,0,:] = [1,2,3,4] sum=10; [0,1,:] = [5,6,7,8] sum=26; [0,2,:] = [9,10,11,12] sum=42
    const r = toFlat(s);
    assert.ok(approx(r[0], 10));
    assert.ok(approx(r[1], 26));
    assert.ok(approx(r[2], 42));
  });

  it('mean over axes [0,2] on 3D', () => {
    const x = make3D();
    const m = core.mean(x, [0, 2]);
    assert.deepEqual(m.shape, [3]);
    // For j=0: elements are [1,2,3,4,13,14,15,16] mean = 68/8 = 8.5
    // For j=1: [5,6,7,8,17,18,19,20] mean = 100/8 = 12.5
    // For j=2: [9,10,11,12,21,22,23,24] mean = 132/8 = 16.5
    const r = toArr(m);
    assert.ok(approx(r[0], 8.5));
    assert.ok(approx(r[1], 12.5));
    assert.ok(approx(r[2], 16.5));
  });

  it('max over axes [1,2] on 3D', () => {
    const x = make3D();
    const m = core.max(x, [1, 2]);
    assert.deepEqual(m.shape, [2]);
    // slice 0: max of 1..12 = 12; slice 1: max of 13..24 = 24
    const r = toArr(m);
    assert.ok(approx(r[0], 12));
    assert.ok(approx(r[1], 24));
  });

  it('min over axes [1,2] on 3D', () => {
    const x = make3D();
    const m = core.min(x, [1, 2]);
    assert.deepEqual(m.shape, [2]);
    // slice 0: min of 1..12 = 1; slice 1: min of 13..24 = 13
    const r = toArr(m);
    assert.ok(approx(r[0], 1));
    assert.ok(approx(r[1], 13));
  });

  it('sum all axes with keepdims on 3D', () => {
    const x = make3D();
    // sum over all axes with keepdims should give shape [1,1,1]
    const s = core.sum(x, [0, 1, 2], { keepdims: true });
    assert.deepEqual(s.shape, [1, 1, 1]);
    // 1+2+...+24 = 300
    assert.ok(approx(toFlat(s)[0], 300));
  });
});

describe('reduce: edge cases', () => {
  it('reduction on 1-element array', () => {
    const x = core.array(new Float32Array([7.5]), [1, 1]);
    assert.ok(approx(toArr(core.sum(x))[0], 7.5));
    assert.ok(approx(toArr(core.mean(x))[0], 7.5));
    assert.ok(approx(toArr(core.min(x))[0], 7.5));
    assert.ok(approx(toArr(core.max(x))[0], 7.5));
    assert.ok(approx(toArr(core.prod(x))[0], 7.5));
  });

  it('all same values: min == max == value', () => {
    const data = new Float32Array(20).fill(5);
    const x = core.array(data, [4, 5]);
    assert.ok(approx(toArr(core.min(x))[0], 5));
    assert.ok(approx(toArr(core.max(x))[0], 5));
    assert.ok(approx(toArr(core.mean(x))[0], 5));
  });

  it('negative values reduce correctly', () => {
    const x = core.array(new Float32Array([-3, -1, -4, -1, -5, -9]), [2, 3]);
    assert.ok(approx(toArr(core.min(x))[0], -9));
    assert.ok(approx(toArr(core.max(x))[0], -1));
    // sum = -3-1-4-1-5-9 = -23
    assert.ok(approx(toArr(core.sum(x))[0], -23));
  });

  it('large 1D sum', () => {
    const n = 10000;
    const data = new Float32Array(n).fill(1);
    const x = core.array(data, [n]);
    const s = core.sum(x);
    assert.ok(approx(toArr(s)[0], n, 1));
  });

  it('prod of zeros is zero', () => {
    const x = core.array(new Float32Array([1, 0, 3, 4]), [2, 2]);
    assert.ok(approx(toArr(core.prod(x))[0], 0));
  });

  it('argmin / argmax on all-equal array returns 0', () => {
    const data = new Float32Array(6).fill(3);
    const x = core.array(data, [2, 3]);
    assert.equal(toArr(core.argmin(x))[0], 0);
    assert.equal(toArr(core.argmax(x))[0], 0);
  });

  it('sum keepdims preserves ndim', () => {
    const data = new Float32Array(60);
    for (let i = 0; i < 60; i++) data[i] = i;
    const x = core.array(data, [3, 4, 5]);
    const s = core.sum(x, 1, { keepdims: true });
    assert.equal(s.shape.length, 3);
    assert.deepEqual(s.shape, [3, 1, 5]);
  });
});

describe('reduce: dtype consistency', () => {
  it('sum on float32 returns float32', () => {
    const x = core.array(new Float32Array([1, 2, 3]), [3]);
    const s = core.sum(x);
    assert.equal(s.dtype, 'float32');
  });

  it('sum on int32 returns int32', () => {
    const x = core.array(new Int32Array([1, 2, 3]), [3]);
    const s = core.sum(x);
    assert.equal(s.dtype, 'int32');
  });

  it('mean on int32 returns float32', () => {
    const x = core.array(new Int32Array([1, 2, 3, 4]), [4]);
    const m = core.mean(x);
    assert.equal(m.dtype, 'float32');
  });

  it('argmin returns uint32', () => {
    const x = core.array(new Float32Array([3, 1, 2]), [3]);
    const idx = core.argmin(x);
    assert.equal(idx.dtype, 'uint32');
  });

  it('argmax returns uint32', () => {
    const x = core.array(new Float32Array([3, 1, 2]), [3]);
    const idx = core.argmax(x);
    assert.equal(idx.dtype, 'uint32');
  });
});

describe('reduce: array_equal verification', () => {
  it('sum results match via array_equal', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const s1 = core.sum(x, 0);
    const expected = core.array(new Float32Array([5, 7, 9]), [3]);
    const eq = core.array_equal(s1, expected);
    assert.equal(toArr(eq)[0], true);
  });

  it('prod axis=1 matches expected', () => {
    // [[2,3],[4,5]] -> prod axis=1 -> [6, 20]
    const x = core.array(new Float32Array([2, 3, 4, 5]), [2, 2]);
    const p = core.prod(x, 1);
    const expected = core.array(new Float32Array([6, 20]), [2]);
    const eq = core.array_equal(p, expected);
    assert.equal(toArr(eq)[0], true);
  });
});

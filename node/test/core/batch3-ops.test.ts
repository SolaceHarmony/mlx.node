import { describe, it } from 'mocha';
import * as assert from 'assert';
import * as core from '../../src/core';

describe('batch 3: unary math ops', () => {
  it('log2', () => {
    const r = core.log2(core.array(new Float32Array([1, 2, 4, 8]), [4]));
    const v = r.toArray() as number[];
    assert.deepStrictEqual(v.map(Math.round), [0, 1, 2, 3]);
  });

  it('log10', () => {
    const r = core.log10(core.array(new Float32Array([1, 10, 100, 1000]), [4]));
    const v = r.toArray() as number[];
    assert.deepStrictEqual(v.map(Math.round), [0, 1, 2, 3]);
  });

  it('isposinf', () => {
    const r = core.isposinf(core.array(new Float32Array([Infinity, -Infinity, 0, 1]), [4]));
    const v = r.toArray();
    assert.deepStrictEqual(v, [true, false, false, false]);
  });

  it('isneginf', () => {
    const r = core.isneginf(core.array(new Float32Array([Infinity, -Infinity, 0, 1]), [4]));
    const v = r.toArray();
    assert.deepStrictEqual(v, [false, true, false, false]);
  });

  it('bitwise_invert', () => {
    const a = core.array(new Int32Array([0, 1, -1, 255]));
    const r = core.bitwise_invert(a);
    assert.deepStrictEqual(r.toArray(), [-1, -2, 0, -256]);
  });

  it('conjugate / conj', () => {
    // conjugate of real array is identity
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const r = core.conjugate(a);
    assert.deepStrictEqual(r.toArray(), a.toArray());
    // conj alias
    const r2 = core.conj(a);
    assert.deepStrictEqual(r2.toArray(), a.toArray());
  });

  it('real and imag on real arrays', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const r = core.real(a);
    assert.deepStrictEqual(r.toArray(), [1, 2, 3]);
  });

  it('stop_gradient', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const r = core.stop_gradient(a);
    assert.deepStrictEqual(r.toArray(), [1, 2, 3]);
  });
});

describe('batch 3: binary ops', () => {
  it('outer product', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const b = core.array(new Float32Array([4, 5]), [2]);
    const r = core.outer(a, b);
    assert.deepStrictEqual(r.shape, [3, 2]);
    assert.deepStrictEqual(r.toArray(), [4, 5, 8, 10, 12, 15]);
  });

  it('inner product', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const b = core.array(new Float32Array([4, 5, 6]), [3]);
    const r = core.inner(a, b);
    const v = r.toArray() as number[];
    assert.deepStrictEqual(v, [32]); // 1*4 + 2*5 + 3*6
  });

  it('kron product', () => {
    const a = core.array(new Float32Array([1, 0, 0, 1]), [2, 2]);
    const b = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const r = core.kron(a, b);
    assert.deepStrictEqual(r.shape, [4, 4]);
    // identity kron b = block diagonal of b
    const v = r.toArray() as number[];
    assert.strictEqual(v[0], 1);
    assert.strictEqual(v[1], 2);
    assert.strictEqual(v[2], 0);
    assert.strictEqual(v[3], 0);
  });
});

describe('batch 3: parameterized ops', () => {
  it('nan_to_num', () => {
    const a = core.array(new Float32Array([NaN, Infinity, -Infinity, 1]), [4]);
    const r = core.nan_to_num(a, { nan: 0, posinf: 999, neginf: -999 });
    const v = r.toArray() as number[];
    assert.strictEqual(v[0], 0);
    assert.strictEqual(v[1], 999);
    assert.strictEqual(v[2], -999);
    assert.strictEqual(v[3], 1);
  });

  it('allclose', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const b = core.array(new Float32Array([1, 2, 3.00001]), [3]);
    const r = core.allclose(a, b, { atol: 1e-4 });
    const v = r.toArray();
    assert.deepStrictEqual(v, [true]);
  });

  it('isclose', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const b = core.array(new Float32Array([1, 2.1, 3]), [3]);
    const r = core.isclose(a, b);
    assert.deepStrictEqual(r.toArray(), [true, false, true]);
  });

  it('contiguous', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const r = core.contiguous(a);
    assert.deepStrictEqual(r.toArray(), a.toArray());
  });

  it('unflatten', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [6]);
    const r = core.unflatten(a, 0, [2, 3]);
    assert.deepStrictEqual(r.shape, [2, 3]);
  });

  it('partition', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]), [8]);
    const r = core.partition(a, 3);
    const v = r.toArray() as number[];
    // kth element (index 3) should be in its sorted position
    // Elements before index 3 should be <= v[3], elements after >= v[3]
    for (let i = 0; i < 3; i++) assert.ok(v[i] <= v[3]);
    for (let i = 4; i < 8; i++) assert.ok(v[i] >= v[3]);
  });

  it('argpartition', () => {
    const a = core.array(new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]), [8]);
    const r = core.argpartition(a, 3);
    assert.deepStrictEqual(r.shape, [8]);
  });

  it('roll', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5]), [5]);
    const r = core.roll(a, 2);
    assert.deepStrictEqual(r.toArray(), [4, 5, 1, 2, 3]);
  });

  it('tri', () => {
    const r = core.tri(3);
    assert.deepStrictEqual(r.shape, [3, 3]);
    const v = r.toArray() as number[];
    // Lower triangular matrix of ones
    assert.strictEqual(v[0], 1); // [0,0]
    assert.strictEqual(v[1], 0); // [0,1]
    assert.strictEqual(v[3], 1); // [1,0]
    assert.strictEqual(v[4], 1); // [1,1]
  });

  it('view', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [4]);
    // view float32 as int32
    const r = core.view(a, 'int32');
    assert.deepStrictEqual(r.shape, [4]);
    assert.strictEqual(r.dtype, 'int32');
  });

  it('hadamard_transform', () => {
    // Hadamard requires size to be power of 2
    const a = core.array(new Float32Array([1, 1, 1, 1]), [4]);
    const r = core.hadamard_transform(a);
    assert.deepStrictEqual(r.shape, [4]);
  });
});

describe('batch 3: multi-return and array ops', () => {
  it('meshgrid', () => {
    const x = core.array(new Float32Array([1, 2, 3]), [3]);
    const y = core.array(new Float32Array([4, 5]), [2]);
    const [X, Y] = core.meshgrid([x, y]);
    assert.deepStrictEqual(X.shape, [2, 3]);
    assert.deepStrictEqual(Y.shape, [2, 3]);
  });

  it('broadcast_arrays', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const b = core.array(new Float32Array([4]), [1]);
    const [A, B] = core.broadcast_arrays([a, b]);
    assert.deepStrictEqual(A.shape, [3]);
    assert.deepStrictEqual(B.shape, [3]);
  });

  it('atleast_1d / 2d / 3d', () => {
    const a = core.array(new Float32Array([5]), [1]);
    const r1 = core.atleast_1d(a);
    assert.ok(r1.shape.length >= 1);
    const r2 = core.atleast_2d(a);
    assert.ok(r2.shape.length >= 2);
    const r3 = core.atleast_3d(a);
    assert.ok(r3.shape.length >= 3);
  });

  it('concat', () => {
    const a = core.array(new Float32Array([1, 2]), [2]);
    const b = core.array(new Float32Array([3, 4]), [2]);
    const r = core.concat([a, b]);
    assert.deepStrictEqual(r.toArray(), [1, 2, 3, 4]);
  });

  it('divmod', () => {
    const a = core.array(new Float32Array([7, 10]), [2]);
    const b = core.array(new Float32Array([3, 4]), [2]);
    const [q, rem] = core.divmod(a, b);
    assert.deepStrictEqual(q.toArray(), [2, 2]);
    assert.deepStrictEqual(rem.toArray(), [1, 2]);
  });

  it('permute_dims', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const r = core.permute_dims(a, [1, 0]);
    assert.deepStrictEqual(r.shape, [3, 2]);
  });

  it('slice_update', () => {
    const src = core.zeros([4]);
    const update = core.ones([2]);
    const r = core.slice_update(src, update, [1], [3]);
    const v = r.toArray() as number[];
    assert.strictEqual(v[0], 0);
    assert.strictEqual(v[1], 1);
    assert.strictEqual(v[2], 1);
    assert.strictEqual(v[3], 0);
  });

  it('put_along_axis', () => {
    const a = core.zeros([5]);
    const idx = core.array(new Int32Array([1, 3]), [2]);
    const vals = core.array(new Float32Array([10, 20]), [2]);
    const r = core.put_along_axis(a, idx, vals, 0);
    assert.deepStrictEqual(r.shape, [5]);
  });
});

describe('batch 3: einsum and tensor ops', () => {
  it('einsum matrix multiply', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const b = core.array(new Float32Array([5, 6, 7, 8]), [2, 2]);
    const c = core.einsum('ij,jk->ik', [a, b]);
    assert.deepStrictEqual(c.shape, [2, 2]);
    const vals = c.toArray() as number[];
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert.deepStrictEqual(vals, [19, 22, 43, 50]);
  });

  it('tensordot', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const b = core.array(new Float32Array([5, 6, 7, 8]), [2, 2]);
    const c = core.tensordot(a, b, 1);
    assert.deepStrictEqual(c.shape, [2, 2]);
    const vals = c.toArray() as number[];
    assert.deepStrictEqual(vals, [19, 22, 43, 50]);
  });
});

describe('batch 3: quantize / dequantize', () => {
  it('quantize and dequantize roundtrip', () => {
    // Create a simple matrix that's quantizable (last dim divisible by group_size)
    const w = core.array(new Float32Array(128).fill(0).map((_, i) => (i % 7) - 3), [2, 64]);
    const [qw, scales, biases] = core.quantize(w);
    assert.ok(qw.shape.length >= 1);
    assert.ok(scales.shape.length >= 1);
    assert.ok(biases.shape.length >= 1);
    // dequantize back
    const dw = core.dequantize(qw, scales, { biases });
    assert.deepStrictEqual(dw.shape, w.shape);
  });
});

describe('batch 3: convolution ops', () => {
  it('conv_transpose1d', () => {
    // Input: [batch=1, length=4, channels_in=1]
    const input = core.array(new Float32Array([1, 2, 3, 4]), [1, 4, 1]);
    // Weight: [channels_out=1, kW=2, channels_in=1]
    const weight = core.array(new Float32Array([1, 1]), [1, 2, 1]);
    const r = core.conv_transpose1d(input, weight);
    assert.strictEqual(r.shape[0], 1); // batch
    assert.strictEqual(r.shape[2], 1); // channels out
    assert.ok(r.shape[1] >= 4); // length >= input length
  });
});

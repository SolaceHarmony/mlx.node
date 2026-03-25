import { describe, it } from 'mocha';
import * as assert from 'assert';
import * as core from '../../src/core';

describe('batch 4: TS-only ops', () => {
  it('trunc toward zero', () => {
    const a = core.array(new Float32Array([2.7, -2.7, 1.5, -1.5]), [4]);
    const r = core.trunc(a);
    assert.deepStrictEqual(r.toArray(), [2, -2, 1, -1]);
  });

  it('broadcast_shapes', () => {
    assert.deepStrictEqual(core.broadcast_shapes([3, 1], [1, 4]), [3, 4]);
    assert.deepStrictEqual(core.broadcast_shapes([5], [1]), [5]);
    assert.deepStrictEqual(core.broadcast_shapes([2, 3], [3]), [2, 3]);
    assert.throws(() => core.broadcast_shapes([3], [4]));
  });

  it('einsum_path returns path structure', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const [path, subs] = core.einsum_path('ij,jk->ik', a, a);
    assert.ok(Array.isArray(path));
    assert.strictEqual(subs, 'ij,jk->ik');
  });
});

describe('batch 4: linalg ops', () => {
  it('inv inverts a matrix', () => {
    // 2x2 identity should invert to itself
    const eye = core.array(new Float32Array([1, 0, 0, 1]), [2, 2]);
    const inv = core.linalg.inv(eye);
    assert.deepStrictEqual(inv.shape, [2, 2]);
    const v = inv.toArray() as number[];
    assert.ok(Math.abs(v[0] - 1) < 1e-5);
    assert.ok(Math.abs(v[3] - 1) < 1e-5);
  });

  it('solve Ax=b', () => {
    const A = core.array(new Float32Array([2, 1, 1, 3]), [2, 2]);
    const b = core.array(new Float32Array([5, 7]), [2]);
    const x = core.linalg.solve(A, b);
    assert.deepStrictEqual(x.shape, [2]);
  });

  it('cholesky decomposition', () => {
    // Symmetric positive definite matrix
    const A = core.array(new Float32Array([4, 2, 2, 3]), [2, 2]);
    const L = core.linalg.cholesky(A);
    assert.deepStrictEqual(L.shape, [2, 2]);
  });

  it('qr decomposition', () => {
    const A = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const [Q, R] = core.linalg.qr(A);
    assert.deepStrictEqual(Q.shape[0], 3);
    assert.deepStrictEqual(R.shape[1], 2);
  });

  it('svd decomposition', () => {
    const A = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const [U, S, Vt] = core.linalg.svd(A);
    assert.strictEqual(U.shape[0], 2);
    assert.strictEqual(S.shape[0], 2); // min(m,n)
    assert.strictEqual(Vt.shape[0], 3);
  });

  it('lu decomposition', () => {
    const A = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const result = core.linalg.lu(A);
    assert.ok(Array.isArray(result));
    assert.strictEqual(result.length, 3); // P, L, U
  });

  it('lu_factor', () => {
    const A = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const [lu, pivots] = core.linalg.lu_factor(A);
    assert.deepStrictEqual(lu.shape, [2, 2]);
  });

  it('eigvalsh for symmetric matrix', () => {
    const A = core.array(new Float32Array([2, 1, 1, 2]), [2, 2]);
    const vals = core.linalg.eigvalsh(A);
    assert.strictEqual(vals.shape[0], 2);
  });

  it('eigh for symmetric matrix', () => {
    const A = core.array(new Float32Array([2, 1, 1, 2]), [2, 2]);
    const [vals, vecs] = core.linalg.eigh(A);
    assert.strictEqual(vals.shape[0], 2);
    assert.deepStrictEqual(vecs.shape, [2, 2]);
  });

  it('cross product', () => {
    const a = core.array(new Float32Array([1, 0, 0]), [3]);
    const b = core.array(new Float32Array([0, 1, 0]), [3]);
    const r = core.linalg.cross(a, b);
    assert.deepStrictEqual(r.shape, [3]);
    const v = r.toArray() as number[];
    // i x j = k → [0, 0, 1]
    assert.ok(Math.abs(v[2] - 1) < 1e-5);
  });

  it('pinv pseudo-inverse', () => {
    const A = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const pA = core.linalg.pinv(A);
    assert.deepStrictEqual(pA.shape, [3, 2]);
  });

  it('tri_inv', () => {
    // Lower triangular matrix
    const L = core.array(new Float32Array([2, 0, 1, 3]), [2, 2]);
    const Linv = core.linalg.tri_inv(L);
    assert.deepStrictEqual(Linv.shape, [2, 2]);
  });

  it('solve_triangular', () => {
    const L = core.array(new Float32Array([2, 0, 1, 3]), [2, 2]);
    const b = core.array(new Float32Array([4, 7]), [2]);
    const x = core.linalg.solve_triangular(L, b);
    assert.deepStrictEqual(x.shape, [2]);
  });

  it('cholesky_inv', () => {
    const A = core.array(new Float32Array([4, 2, 2, 3]), [2, 2]);
    const Ainv = core.linalg.cholesky_inv(A);
    assert.deepStrictEqual(Ainv.shape, [2, 2]);
  });
});

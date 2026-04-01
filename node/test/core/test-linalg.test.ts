import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../../src/index';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const approx = (a: number, b: number, tol = 1e-4) => Math.abs(a - b) < tol;
const allClose = (arr: number[], expected: number[], tol = 1e-4) =>
  arr.every((v, i) => approx(v, expected[i], tol));

/** Assert two MLXArrays are element-wise close. */
function assertAllClose(
  a: core.core.MLXArray,
  b: core.core.MLXArray,
  opts: { atol?: number; rtol?: number } = {},
) {
  const atol = opts.atol ?? 1e-5;
  const rtol = opts.rtol ?? 1e-5;
  const r = core.allclose(a, b, { atol, rtol });
  const v = r.toArray() as boolean[];
  assert.ok(v[0], `allclose failed: shapes ${a.shape} vs ${b.shape}`);
}

/** Build a float32 MLXArray from a flat JS array and shape. */
function mx(data: number[], shape: number[]) {
  return core.array(new Float32Array(data), shape);
}

// ---------------------------------------------------------------------------
// norm
// ---------------------------------------------------------------------------

describe('linalg.norm', () => {
  it('default (L2) vector norm', () => {
    // sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ≈ 3.7417
    const x = mx([1, 2, 3], [3]);
    const n = core.linalg.norm(x);
    const v = (n.toArray() as number[])[0];
    assert.ok(approx(v, Math.sqrt(14), 1e-4));
  });

  it('L1 vector norm', () => {
    const x = mx([1, -2, 3], [3]);
    const n = core.linalg.norm(x, 1);
    const v = (n.toArray() as number[])[0];
    assert.ok(approx(v, 6, 1e-5));
  });

  it('L-inf vector norm', () => {
    const x = mx([1, -5, 3], [3]);
    const n = core.linalg.norm(x, Infinity);
    const v = (n.toArray() as number[])[0];
    assert.ok(approx(v, 5, 1e-5));
  });

  it('negative-inf vector norm', () => {
    const x = mx([1, -5, 3], [3]);
    const n = core.linalg.norm(x, -Infinity);
    const v = (n.toArray() as number[])[0];
    assert.ok(approx(v, 1, 1e-5)); // min(|1|, |5|, |3|) = 1
  });

  it('L0 "norm" (count of nonzeros)', () => {
    const x = mx([0, 2, 0, 4, 5], [5]);
    const n = core.linalg.norm(x, 0);
    const v = (n.toArray() as number[])[0];
    assert.ok(approx(v, 3, 1e-5));
  });

  it('Frobenius matrix norm (default for 2-D)', () => {
    // arange 1..6 reshaped to 2x3, fro norm = sqrt(sum of squares)
    const x = mx([1, 2, 3, 4, 5, 6], [2, 3]);
    const n = core.linalg.norm(x);
    const expected = Math.sqrt(1 + 4 + 9 + 16 + 25 + 36);
    const v = (n.toArray() as number[])[0];
    assert.ok(approx(v, expected, 1e-4));
  });

  it('Frobenius norm via ord="fro"', () => {
    const x = mx([1, 2, 3, 4, 5, 6], [2, 3]);
    const n = core.linalg.norm(x, 'fro');
    const expected = Math.sqrt(91);
    const v = (n.toArray() as number[])[0];
    assert.ok(approx(v, expected, 1e-4));
  });

  it('norm with axis parameter (row-wise L2)', () => {
    // 2x3 matrix, compute L2 norm along axis=1 (each row)
    const x = mx([1, 2, 3, 4, 5, 6], [2, 3]);
    const n = core.linalg.norm(x, 2, 1);
    const v = n.toArray() as number[];
    assert.ok(approx(v[0], Math.sqrt(14), 1e-4));
    assert.ok(approx(v[1], Math.sqrt(77), 1e-4));
  });

  it('norm with keepdims', () => {
    const x = mx([1, 2, 3, 4, 5, 6], [2, 3]);
    const n = core.linalg.norm(x, 2, 1, { keepdims: true });
    assert.deepStrictEqual(n.shape, [2, 1]);
  });
});

// ---------------------------------------------------------------------------
// qr
// ---------------------------------------------------------------------------

describe('linalg.qr', () => {
  it('Q @ R ≈ A for square matrix', () => {
    const A = mx([2, 3, 1, 2], [2, 2]);
    const [Q, R] = core.linalg.qr(A);
    const recon = core.matmul(Q, R);
    assertAllClose(recon, A);
  });

  it('Q is orthogonal (Q^T @ Q ≈ I)', () => {
    const A = mx([2, 3, 1, 2], [2, 2]);
    const [Q] = core.linalg.qr(A);
    const QtQ = core.matmul(core.transpose(Q), Q);
    assertAllClose(QtQ, core.eye(2), { atol: 1e-5 });
  });

  it('R is upper triangular', () => {
    const A = mx([2, 3, 1, 2], [2, 2]);
    const [, R] = core.linalg.qr(A);
    const lower = core.tril(R, { k: -1 });
    assertAllClose(lower, core.zeros_like(R), { atol: 1e-6 });
  });

  it('non-square tall matrix (4x2)', () => {
    const data = Array.from({ length: 8 }, (_, i) => i + 1);
    const A = mx(data, [4, 2]);
    const [Q, R] = core.linalg.qr(A);
    const recon = core.matmul(Q, R);
    assertAllClose(recon, A, { atol: 1e-4 });
  });

  it('non-square wide matrix (2x4)', () => {
    const data = Array.from({ length: 8 }, (_, i) => i + 1);
    const A = mx(data, [2, 4]);
    const [Q, R] = core.linalg.qr(A);
    const recon = core.matmul(Q, R);
    assertAllClose(recon, A, { atol: 1e-4 });
  });

  it('batch QR (stacked matrices)', () => {
    const A1 = mx([2, 3, 1, 2], [2, 2]);
    const A2 = mx([-1, 2, -4, 1], [2, 2]);
    const A = core.stack([A1, A2]);
    const [Q, R] = core.linalg.qr(A);
    // Verify each matrix in the batch
    for (let i = 0; i < 2; i++) {
      const qi = mx((Q.toArray() as number[]).slice(i * 4, (i + 1) * 4), [2, 2]);
      const ri = mx((R.toArray() as number[]).slice(i * 4, (i + 1) * 4), [2, 2]);
      const ai = mx(((i === 0 ? A1 : A2).toArray() as number[]), [2, 2]);
      const recon = core.matmul(qi, ri);
      assertAllClose(recon, ai, { atol: 1e-4 });
    }
  });
});

// ---------------------------------------------------------------------------
// svd
// ---------------------------------------------------------------------------

describe('linalg.svd', () => {
  it('U @ diag(S) @ Vt ≈ A', () => {
    const A = mx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    const [U, S, Vt] = core.linalg.svd(A);
    // U is 4x4, S is 3, Vt is 3x3 — need U[:, :3] @ diag(S) @ Vt
    // Extract first 3 columns of U (4x3)
    const Udata = U.toArray() as number[];
    const Uslice: number[] = [];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 3; c++) {
        Uslice.push(Udata[r * 4 + c]);
      }
    }
    const Utrim = mx(Uslice, [4, 3]);
    const diagS = core.diag(S);
    const recon = core.matmul(core.matmul(Utrim, diagS), Vt);
    assertAllClose(recon, A, { atol: 1e-4 });
  });

  it('singular values are non-negative', () => {
    const A = mx([1, 2, 3, 4, 5, 6], [2, 3]);
    const [, S] = core.linalg.svd(A);
    const sv = S.toArray() as number[];
    for (const v of sv) {
      assert.ok(v >= -1e-7, `singular value ${v} should be non-negative`);
    }
  });

  it('batch SVD', () => {
    const A = mx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
    const [U, S, Vt] = core.linalg.svd(A);
    assert.strictEqual(U.shape[0], 2);
    assert.strictEqual(S.shape[0], 2);
    assert.strictEqual(Vt.shape[0], 2);
  });
});

// ---------------------------------------------------------------------------
// cholesky
// ---------------------------------------------------------------------------

describe('linalg.cholesky', () => {
  it('L @ L^T ≈ A for positive definite matrix', () => {
    // Build A = sqrtA^T @ sqrtA / 81
    const sqrtA = mx([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const A = core.divide(core.matmul(core.transpose(sqrtA), sqrtA), 81);
    const L = core.linalg.cholesky(A);
    const recon = core.matmul(L, core.transpose(L));
    assertAllClose(recon, A, { atol: 1e-5 });
  });

  it('L is lower triangular', () => {
    const A = mx([4, 2, 2, 3], [2, 2]);
    const L = core.linalg.cholesky(A);
    const upper = core.triu(L, { k: 1 });
    assertAllClose(upper, core.zeros_like(upper), { atol: 1e-7 });
  });

  it('batch cholesky', () => {
    const A1 = mx([4, 2, 2, 3], [2, 2]);
    const A2 = mx([5, 3, 3, 4], [2, 2]);
    const AB = core.stack([A1, A2]);
    const Ls = core.linalg.cholesky(AB);
    assert.deepStrictEqual(Ls.shape, [2, 2, 2]);
    // Verify first matrix
    const Ldata = Ls.toArray() as number[];
    const L1 = mx(Ldata.slice(0, 4), [2, 2]);
    const recon = core.matmul(L1, core.transpose(L1));
    assertAllClose(recon, A1, { atol: 1e-5 });
  });
});

// ---------------------------------------------------------------------------
// inv
// ---------------------------------------------------------------------------

describe('linalg.inv', () => {
  it('A @ inv(A) ≈ I', () => {
    const A = mx([1, 2, 3, 6, -5, 4, -9, 8, 7], [3, 3]);
    const Ainv = core.linalg.inv(A);
    const prod = core.matmul(A, Ainv);
    assertAllClose(prod, core.eye(3), { atol: 1e-4 });
  });

  it('inv(I) ≈ I', () => {
    const I = core.eye(3);
    const Iinv = core.linalg.inv(I);
    assertAllClose(Iinv, I, { atol: 1e-6 });
  });

  it('batch inverse', () => {
    const A = mx([1, 2, 3, 6, -5, 4, -9, 8, 7], [3, 3]);
    const B = mx(
      [1 - 100, 2 - 100, 3 - 100, 6 - 100, -5 - 100, 4 - 100, -9 - 100, 8 - 100, 7 - 100],
      [3, 3],
    );
    const AB = core.stack([A, B]);
    const invs = core.linalg.inv(AB);
    // Verify first matrix
    const invData = invs.toArray() as number[];
    const inv0 = mx(invData.slice(0, 9), [3, 3]);
    const prod = core.matmul(A, inv0);
    assertAllClose(prod, core.eye(3), { atol: 1e-4 });
  });
});

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

describe('linalg.solve', () => {
  it('A @ x ≈ b for 1-D rhs', () => {
    const A = mx([3, 1, 2, 1, 8, 6, 9, 2, 5], [3, 3]);
    const b = mx([11, 35, 28], [3]);
    const x = core.linalg.solve(A, b);
    const Ax = core.matmul(A, x);
    assertAllClose(Ax, b, { atol: 1e-4 });
  });

  it('A @ x ≈ b for 2-D rhs (multiple columns)', () => {
    const A = mx([2, 1, 1, 3], [2, 2]);
    const b = mx([5, 7, 3, 11], [2, 2]); // two right-hand-side columns
    const x = core.linalg.solve(A, b);
    const Ax = core.matmul(A, x);
    assertAllClose(Ax, b, { atol: 1e-4 });
  });

  it('solve with symmetric positive definite A', () => {
    // A = [[5, 2], [2, 5]]  (SPD), b = [7, 7]
    const A = mx([5, 2, 2, 5], [2, 2]);
    const b = mx([7, 7], [2]);
    const x = core.linalg.solve(A, b);
    const Ax = core.matmul(A, x);
    assertAllClose(Ax, b, { atol: 1e-4 });
    // Known solution: x = [1, 1]
    const xv = x.toArray() as number[];
    assert.ok(allClose(xv, [1, 1], 1e-4));
  });
});

// ---------------------------------------------------------------------------
// eigh / eigvalsh
// ---------------------------------------------------------------------------

describe('linalg.eigh', () => {
  it('A @ V ≈ V @ diag(lambda) for symmetric 2x2', () => {
    const A = mx([2, 1, 1, 2], [2, 2]);
    const [vals, vecs] = core.linalg.eigh(A);
    // A @ vecs should equal vecs * vals (column-wise)
    const Av = core.matmul(A, vecs);
    // vecs * vals (broadcasting vals across rows)
    // vals shape [2], vecs shape [2,2]
    // In MLX: vals[..., None, :] * vecs — but we can verify A @ V ≈ V @ diag(vals)
    const diagV = core.diag(vals);
    const vd = core.matmul(vecs, diagV);
    assertAllClose(Av, vd, { atol: 1e-4 });
  });

  it('eigenvalues of [[2,1],[1,2]] are [1, 3]', () => {
    const A = mx([2, 1, 1, 2], [2, 2]);
    const [vals] = core.linalg.eigh(A);
    const v = vals.toArray() as number[];
    // eigh returns sorted eigenvalues
    assert.ok(approx(v[0], 1, 1e-4));
    assert.ok(approx(v[1], 3, 1e-4));
  });

  it('eigvalsh matches eigh eigenvalues', () => {
    const A = mx([4, 1, 1, 3], [2, 2]);
    const [vals] = core.linalg.eigh(A);
    const valsOnly = core.linalg.eigvalsh(A);
    assertAllClose(vals, valsOnly, { atol: 1e-5 });
  });

  it('larger symmetric matrix', () => {
    // Build symmetric 3x3: [[6, 2, 1], [2, 5, 3], [1, 3, 7]]
    const A = mx([6, 2, 1, 2, 5, 3, 1, 3, 7], [3, 3]);
    const [vals, vecs] = core.linalg.eigh(A);
    // Verify A @ V ≈ V @ diag(vals)
    const Av = core.matmul(A, vecs);
    const vd = core.matmul(vecs, core.diag(vals));
    assertAllClose(Av, vd, { atol: 1e-4 });
  });
});

// ---------------------------------------------------------------------------
// lu
// ---------------------------------------------------------------------------

describe('linalg.lu', () => {
  it('P, L, U decomposition: L[P,:] @ U ≈ A', () => {
    const A = mx([3, 1, 2, 1, 8, 6, 9, 2, 5], [3, 3]);
    const [P, L, U] = core.linalg.lu(A);
    // Reconstruct using L[P,:] @ U
    const Lperm = core.take_along_axis(L, core.expand_dims(P, -1), 0);
    const recon = core.matmul(Lperm, U);
    assertAllClose(recon, A, { atol: 1e-4 });
  });

  it('LU returns correct shapes', () => {
    const A = mx([1, 2, 3, 4, 5, 6], [3, 2]);
    const [P, L, U] = core.linalg.lu(A);
    // P should be a permutation vector of length min(m,n) or m
    assert.ok(P.shape.length >= 1);
    assert.ok(L.shape.length === 2);
    assert.ok(U.shape.length === 2);
  });

  it('non-square tall matrix (3x2)', () => {
    const A = mx([3, 1, 1, 8, 9, 2], [3, 2]);
    const [P, L, U] = core.linalg.lu(A);
    const Lperm = core.take_along_axis(L, core.expand_dims(P, -1), 0);
    const recon = core.matmul(Lperm, U);
    assertAllClose(recon, A, { atol: 1e-4 });
  });

  it('non-square wide matrix (2x3)', () => {
    const A = mx([3, 1, 2, 1, 8, 6], [2, 3]);
    const [P, L, U] = core.linalg.lu(A);
    const Lperm = core.take_along_axis(L, core.expand_dims(P, -1), 0);
    const recon = core.matmul(Lperm, U);
    assertAllClose(recon, A, { atol: 1e-4 });
  });
});

// ---------------------------------------------------------------------------
// cross
// ---------------------------------------------------------------------------

describe('linalg.cross', () => {
  it('i x j = k', () => {
    const a = mx([1, 0, 0], [3]);
    const b = mx([0, 1, 0], [3]);
    const r = core.linalg.cross(a, b);
    const v = r.toArray() as number[];
    assert.ok(allClose(v, [0, 0, 1], 1e-5));
  });

  it('j x i = -k', () => {
    const a = mx([0, 1, 0], [3]);
    const b = mx([1, 0, 0], [3]);
    const r = core.linalg.cross(a, b);
    const v = r.toArray() as number[];
    assert.ok(allClose(v, [0, 0, -1], 1e-5));
  });

  it('a x a = 0', () => {
    const a = mx([1, 2, 3], [3]);
    const r = core.linalg.cross(a, a);
    const v = r.toArray() as number[];
    assert.ok(allClose(v, [0, 0, 0], 1e-5));
  });

  it('known cross product', () => {
    // [1,2,3] x [4,5,6] = [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
    const a = mx([1, 2, 3], [3]);
    const b = mx([4, 5, 6], [3]);
    const r = core.linalg.cross(a, b);
    const v = r.toArray() as number[];
    assert.ok(allClose(v, [-3, 6, -3], 1e-5));
  });

  it('negative values', () => {
    // [-1,-2,-3] x [4,-5,6] = [(-2)*6-(-3)*(-5), (-3)*4-(-1)*6, (-1)*(-5)-(-2)*4]
    //                        = [-12-15, -12+6, 5+8] = [-27, -6, 13]
    const a = mx([-1, -2, -3], [3]);
    const b = mx([4, -5, 6], [3]);
    const r = core.linalg.cross(a, b);
    const v = r.toArray() as number[];
    assert.ok(allClose(v, [-27, -6, 13], 1e-5));
  });

  it('batch cross product (2D arrays)', () => {
    const a = mx([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = mx([4, 5, 6, 1, 2, 3], [2, 3]);
    const r = core.linalg.cross(a, b, { axis: 1 });
    const v = r.toArray() as number[];
    // row 0: [1,2,3] x [4,5,6] = [-3, 6, -3]
    // row 1: [4,5,6] x [1,2,3] = [3, -6, 3]
    assert.ok(allClose(v, [-3, 6, -3, 3, -6, 3], 1e-5));
  });
});

// ---------------------------------------------------------------------------
// pinv
// ---------------------------------------------------------------------------

describe('linalg.pinv', () => {
  it('A @ pinv(A) @ A ≈ A (Moore-Penrose property)', () => {
    const A = mx([1, 2, 3, 6, -5, 4, -9, 8, 7], [3, 3]);
    const Ap = core.linalg.pinv(A);
    const recon = core.matmul(core.matmul(A, Ap), A);
    assertAllClose(recon, A, { atol: 1e-4 });
  });

  it('pinv(A) @ A @ pinv(A) ≈ pinv(A)', () => {
    const A = mx([1, 2, 3, 6, -5, 4, -9, 8, 7], [3, 3]);
    const Ap = core.linalg.pinv(A);
    const recon = core.matmul(core.matmul(Ap, A), Ap);
    assertAllClose(recon, Ap, { atol: 1e-4 });
  });

  it('pinv of non-square matrix', () => {
    const A = mx([1, 2, 3, 4, 5, 6], [2, 3]);
    const Ap = core.linalg.pinv(A);
    assert.deepStrictEqual(Ap.shape, [3, 2]);
    // Verify Moore-Penrose property
    const recon = core.matmul(core.matmul(A, Ap), A);
    assertAllClose(recon, A, { atol: 1e-3 });
  });

  it('pinv of singular matrix', () => {
    // Rank-1 matrix: rows are multiples
    const A = mx([4, 1, 4, 1], [2, 2]);
    const Ap = core.linalg.pinv(A);
    const recon = core.matmul(core.matmul(A, Ap), A);
    assertAllClose(recon, A, { atol: 1e-4 });
  });

  it('batch pinv', () => {
    const A1 = mx([1, 2, 3, 4], [2, 2]);
    const A2 = mx([5, 6, 7, 8], [2, 2]);
    const AB = core.stack([A1, A2]);
    const pinvs = core.linalg.pinv(AB);
    assert.deepStrictEqual(pinvs.shape, [2, 2, 2]);
  });
});

// ---------------------------------------------------------------------------
// Additional decomposition tests
// ---------------------------------------------------------------------------

describe('linalg: additional coverage', () => {
  it('svd: norm(S) ≈ norm(A, "fro")', () => {
    const A = mx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    const [, S] = core.linalg.svd(A);
    const normS = core.linalg.norm(S);
    const normA = core.linalg.norm(A, 'fro');
    assertAllClose(normS, normA, { atol: 1e-4 });
  });

  it('cholesky_inv: A @ cholesky_inv(L) ≈ I', () => {
    const A = mx([4, 2, 2, 3], [2, 2]);
    const L = core.linalg.cholesky(A);
    const Ainv = core.linalg.cholesky_inv(L);
    const prod = core.matmul(A, Ainv);
    assertAllClose(prod, core.eye(2), { atol: 1e-4 });
  });

  it('tri_inv lower: L @ tri_inv(L) ≈ I', () => {
    const L = mx([1, 0, 0, 6, -5, 0, -9, 8, 7], [3, 3]);
    const Linv = core.linalg.tri_inv(L);
    const prod = core.matmul(L, Linv);
    assertAllClose(prod, core.eye(3), { atol: 1e-4 });
  });

  it('tri_inv upper: U @ tri_inv(U) ≈ I', () => {
    const U = mx([1, 6, -9, 0, -5, 8, 0, 0, 7], [3, 3]);
    const Uinv = core.linalg.tri_inv(U, { upper: true });
    const prod = core.matmul(U, Uinv);
    assertAllClose(prod, core.eye(3), { atol: 1e-4 });
  });

  it('solve_triangular lower: L @ x ≈ b', () => {
    const L = mx([4, 0, 0, 2, 3, 0, 1, -2, 5], [3, 3]);
    const b = mx([8, 14, 3], [3]);
    const x = core.linalg.solve_triangular(L, b, { upper: false });
    const Lx = core.matmul(L, x);
    assertAllClose(Lx, b, { atol: 1e-4 });
  });

  it('solve_triangular upper: U @ x ≈ b', () => {
    const U = mx([3, 2, 1, 0, 5, 4, 0, 0, 6], [3, 3]);
    const b = mx([13, 33, 18], [3]);
    const x = core.linalg.solve_triangular(U, b, { upper: true });
    const Ux = core.matmul(U, x);
    assertAllClose(Ux, b, { atol: 1e-4 });
  });
});

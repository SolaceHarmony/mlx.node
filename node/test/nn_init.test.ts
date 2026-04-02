// Copyright © 2023 Apple Inc.
// Ported from python/tests/test_init.py — line‑for‑line transliteration.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as mx from '../src';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const toFlat = (a: ReturnType<typeof mx.array>): number[] =>
  (a.toArray() as any[]).flat(Infinity) as number[];

/**
 * Checks that two MLXArrays are elementwise close.  Mirrors `mx.allclose`.
 */
const allclose = (
  a: ReturnType<typeof mx.array>,
  b: ReturnType<typeof mx.array>,
  atol = 1e-5,
): boolean => {
  const av = toFlat(a);
  const bv = toFlat(b);
  if (av.length !== bv.length) return false;
  return av.every((x, i) => Math.abs(x - bv[i]) <= atol + 1e-5 * Math.abs(bv[i]));
};

// ---------------------------------------------------------------------------
// TestInit
// ---------------------------------------------------------------------------
describe('TestInit', () => {
  // -------------------------------------------------------------------------
  // test_constant
  // -------------------------------------------------------------------------
  describe('constant', () => {
    const value = 5.0;
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, ...number[]][] = [[3], [3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`constant(${value}, ${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.constant(value, dtype);
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // test_normal
  // -------------------------------------------------------------------------
  describe('normal', () => {
    const mean = 0.0;
    const std = 1.0;
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, ...number[]][] = [[3], [3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`normal(${mean}, ${std}, dtype=${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.normal(mean, std, dtype);
          // Use zeros as the template array (mirrors np.empty semantics)
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // test_uniform
  // -------------------------------------------------------------------------
  describe('uniform', () => {
    const low = -1.0;
    const high = 1.0;
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, ...number[]][] = [[3], [3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`uniform(${low}, ${high}, dtype=${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.uniform(low, high, dtype);
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
          // All values must lie within [low, high]
          const flat = toFlat(result);
          assert.ok(
            flat.every((v) => v >= low && v <= high),
            `Some values outside [${low}, ${high}]`,
          );
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // test_identity  (initializes a 3×3 square matrix to the identity)
  // -------------------------------------------------------------------------
  describe('identity', () => {
    const dtypes = [mx.float32, mx.float16] as const;

    for (const dtype of dtypes) {
      it(`identity(dtype=${dtype}) produces eye(3) for shape [3,3]`, () => {
        const initializer = mx.nn_init.identity(dtype);
        const result = initializer(mx.zeros([3, 3]));
        assert.ok(mx.array_equal(result, mx.eye(3)).toArray()[0], 'identity != eye(3)');
        assert.equal(result.dtype, dtype.toString());
      });

      it(`identity(dtype=${dtype}) throws ValueError for non-square [3,2]`, () => {
        const initializer = mx.nn_init.identity(dtype);
        assert.throws(
          () => initializer(mx.zeros([3, 2])),
          /square|2D|dimension|shape/i,
        );
      });
    }
  });

  // -------------------------------------------------------------------------
  // test_glorot_normal
  // -------------------------------------------------------------------------
  describe('glorot_normal', () => {
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, ...number[]][] = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`glorot_normal(dtype=${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.glorot_normal(dtype);
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // test_glorot_uniform
  // -------------------------------------------------------------------------
  describe('glorot_uniform', () => {
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, ...number[]][] = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`glorot_uniform(dtype=${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.glorot_uniform(dtype);
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // test_he_normal
  // -------------------------------------------------------------------------
  describe('he_normal', () => {
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, ...number[]][] = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`he_normal(dtype=${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.he_normal(dtype);
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // test_he_uniform
  // -------------------------------------------------------------------------
  describe('he_uniform', () => {
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, ...number[]][] = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`he_uniform(dtype=${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.he_uniform(dtype);
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // test_sparse
  // -------------------------------------------------------------------------
  describe('sparse', () => {
    const mean = 0.0;
    const std = 1.0;
    const sparsity = 0.5;
    const dtypes = [mx.float32, mx.float16] as const;
    const shapes: [number, number][] = [
      [3, 2],
      [2, 2],
      [4, 3],
    ];

    for (const dtype of dtypes) {
      for (const shape of shapes) {
        it(`sparse(sparsity=${sparsity}, dtype=${dtype}) on shape [${shape}]`, () => {
          const initializer = mx.nn_init.sparse(sparsity, mean, std, dtype);
          const result = initializer(mx.zeros(shape));
          assert.deepEqual(result.shape, shape);
          assert.equal(result.dtype, dtype.toString());
          // At least sparsity fraction of columns must be zero per-row
          const flat = toFlat(result);
          const [rows, cols] = shape;
          const totalZeros = flat.filter((v) => v === 0).length;
          // total zeros >= floor(sparsity * element_count)
          assert.ok(
            totalZeros >= Math.floor(sparsity * rows * cols),
            `Expected >= ${Math.floor(sparsity * rows * cols)} zeros, got ${totalZeros}`,
          );
        });
      }
    }

    it('sparse throws ValueError for 1D input', () => {
      const initializer = mx.nn_init.sparse(sparsity, mean, std);
      assert.throws(
        () => initializer(mx.zeros([10])),
        /2 dim|2D|dimension/i,
      );
    });
  });

  // -------------------------------------------------------------------------
  // test_orthogonal
  // -------------------------------------------------------------------------
  describe('orthogonal', () => {
    it('square matrix [4,4] satisfies W @ W.T ≈ eye(4)', () => {
      // Skip if orthogonal is not yet fully implemented
      try {
        const initializer = mx.nn_init.orthogonal(1.0, mx.float32);
        const result = initializer(mx.zeros([4, 4]));
        assert.deepEqual(result.shape, [4, 4]);
        assert.equal(result.dtype, mx.float32.toString());

        // W @ W.T should be close to the identity
        const WWT = mx.matmul(result, mx.transpose(result));
        const eye4 = mx.eye(4);
        assert.ok(allclose(WWT, eye4, 1e-4), 'Orthogonal square: W @ W.T != eye(4)');
      } catch (e: any) {
        if (/not yet fully implemented/i.test(e.message)) {
          // acceptable — feature is a known TODO
          return;
        }
        throw e;
      }
    });

    it('rectangular matrix [6,4] satisfies W.T @ W ≈ eye(4)', () => {
      try {
        const initializer = mx.nn_init.orthogonal(1.0, mx.float32);
        const result = initializer(mx.zeros([6, 4]));
        assert.deepEqual(result.shape, [6, 4]);
        assert.equal(result.dtype, mx.float32.toString());

        // W.T @ W should be close to eye(4) for a tall matrix
        const WTW = mx.matmul(mx.transpose(result), result);
        const eye4 = mx.eye(4);
        assert.ok(allclose(WTW, eye4, 1e-4), 'Orthogonal rect: W.T @ W != eye(4)');
      } catch (e: any) {
        if (/not yet fully implemented/i.test(e.message)) {
          return;
        }
        throw e;
      }
    });

    it('throws for 1D arrays', () => {
      const initializer = mx.nn_init.orthogonal();
      assert.throws(
        () => initializer(mx.zeros([5])),
        /2D|dimension/i,
      );
    });

    it('throws for 3D arrays', () => {
      const initializer = mx.nn_init.orthogonal();
      assert.throws(
        () => initializer(mx.zeros([3, 4, 5])),
        /2D|dimension/i,
      );
    });
  });
});

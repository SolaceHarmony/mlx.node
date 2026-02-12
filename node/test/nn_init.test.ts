import { describe, it } from 'node:test';
import assert from 'node:assert';
import * as mx from '../src';

const toArray = (tensor: ReturnType<typeof mx.array>): number[] => tensor.toArray() as number[];

describe('mlx.nn_init', () => {
  describe('orthogonal', () => {
    it('should export orthogonal function', () => {
      assert.ok(mx.nn_init.orthogonal);
      assert.strictEqual(typeof mx.nn_init.orthogonal, 'function');
    });

    it('should return an initializer function', () => {
      const init = mx.nn_init.orthogonal(1.0, mx.float32);
      assert.strictEqual(typeof init, 'function');
    });

    it('should throw error for non-2D arrays', () => {
      const init = mx.nn_init.orthogonal();
      const array1d = mx.zeros([5]);

      assert.throws(
        () => init(array1d),
        /requires a 2D array/
      );
    });

    it('should throw error for 3D arrays', () => {
      const init = mx.nn_init.orthogonal();
      const array3d = mx.zeros([3, 4, 5]);

      assert.throws(
        () => init(array3d),
        /requires a 2D array/
      );
    });

    it('should accept 2D arrays but throw not implemented error', () => {
      const init = mx.nn_init.orthogonal();
      const array2d = mx.zeros([3, 5]);

      // Currently throws "not yet fully implemented" error
      assert.throws(
        () => init(array2d),
        /not yet fully implemented/
      );
    });

    it('should accept custom gain parameter', () => {
      const init = mx.nn_init.orthogonal(2.0);
      assert.strictEqual(typeof init, 'function');
    });

    it('should accept custom dtype parameter', () => {
      const init = mx.nn_init.orthogonal(1.0, mx.float16);
      assert.strictEqual(typeof init, 'function');
    });
  });

  describe('sparse', () => {
    it('should create a sparse matrix with correct shape', () => {
      const input = mx.zeros([3, 4], mx.float32);
      const result = mx.nn_init.sparse(input, 0.5);

      assert.deepEqual(result.shape, [3, 4]);
      assert.strictEqual(result.dtype.toString(), 'float32');
    });

    it('should zero out approximately the correct fraction of elements', () => {
      const rows = 10;
      const cols = 100;
      const sparsity = 0.5;
      const input = mx.zeros([rows, cols], mx.float32);
      const result = mx.nn_init.sparse(input, sparsity);

      const data = toArray(result);
      const zeroCount = data.filter(x => x === 0).length;

      const expectedZeros = rows * Math.ceil(cols * sparsity);
      assert.strictEqual(zeroCount, expectedZeros);
    });

    it('should apply sparsity per row (not globally)', () => {
      const rows = 5;
      const cols = 10;
      const sparsity = 0.3;
      const input = mx.zeros([rows, cols], mx.float32);
      const result = mx.nn_init.sparse(input, sparsity);

      const data = toArray(result);
      const expectedZerosPerRow = Math.ceil(cols * sparsity);

      for (let i = 0; i < rows; i++) {
        const row = data.slice(i * cols, (i + 1) * cols);
        const rowZeros = row.filter(x => x === 0).length;
        assert.strictEqual(rowZeros, expectedZerosPerRow, `Row ${i} should have ${expectedZerosPerRow} zeros`);
      }
    });

    it('should use custom mean and std', () => {
      const input = mx.zeros([10, 10], mx.float32);
      const mean = 5.0;
      const std = 2.0;
      const sparsity = 0.1;

      const result = mx.nn_init.sparse(input, sparsity, mean, std);
      const data = toArray(result);
      const nonZeroData = data.filter(x => x !== 0);

      const avg = nonZeroData.reduce((a, b) => a + b, 0) / nonZeroData.length;
      assert.ok(Math.abs(avg - mean) < std, `Average ${avg} should be close to mean ${mean}`);
    });

    it('should throw error for non-2D arrays', () => {
      const input1D = mx.zeros([10], mx.float32);
      assert.throws(() => {
        mx.nn_init.sparse(input1D, 0.5);
      }, /only tensors with 2 dimensions are supported/);

      const input3D = mx.zeros([2, 3, 4], mx.float32);
      assert.throws(() => {
        mx.nn_init.sparse(input3D, 0.5);
      }, /only tensors with 2 dimensions are supported/);
    });

    it('should validate sparsity range', () => {
      const input = mx.zeros([3, 4], mx.float32);

      assert.throws(() => {
        mx.nn_init.sparse(input, -0.1);
      }, /sparsity must be between 0 and 1/);

      assert.throws(() => {
        mx.nn_init.sparse(input, 1.1);
      }, /sparsity must be between 0 and 1/);
    });

    it('should work with sparsity 0 (no zeros)', () => {
      const input = mx.zeros([3, 4], mx.float32);
      const result = mx.nn_init.sparse(input, 0.0);

      const data = toArray(result);
      const zeroCount = data.filter(x => x === 0).length;
      assert.strictEqual(zeroCount, 0);
    });

    it('should work with sparsity 1 (all zeros)', () => {
      const input = mx.zeros([3, 4], mx.float32);
      const result = mx.nn_init.sparse(input, 1.0);

      const data = toArray(result);
      const zeroCount = data.filter(x => x === 0).length;
      assert.strictEqual(zeroCount, 3 * 4);
    });
  });
});

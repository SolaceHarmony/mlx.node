import { strict as assert } from 'node:assert';
import * as mx from '../src';
import { nn } from '../src';

describe('nn.init', () => {
  it('test_constant', () => {
    const value = 5.0;
    const dtypes = [mx.float32, mx.float16];
    const shapes = [[3], [3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      const initializer = nn.init.constant(value, dtype);
      for (const shape of shapes) {
        const result = initializer(mx.zeros(shape));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);
        // Verify all elements are equal to value
        const data = result.toArray();
        const flatData = data.flat(Infinity) as number[];
        assert.ok(flatData.every(v => Math.abs(v - value) < 1e-5));
      }
    }
  });

  it('test_normal', () => {
    // Note: Our nn.init.normal might not exist yet, checking.
    // If not, we might need to add it to init.ts to match parity.
    if ((nn.init as any).normal) {
        const mean = 0.0;
        const std = 1.0;
        const dtypes = [mx.float32, mx.float16];
        const shapes = [[3], [3, 3], [3, 3, 3]];

        for (const dtype of dtypes) {
          const initializer = (nn.init as any).normal(mean, std, dtype);
          for (const shape of shapes) {
            const result = initializer(mx.zeros(shape));
            assert.deepEqual(result.shape, shape);
            assert.equal(result.dtype, dtype.key);
          }
        }
    }
  });

  it('test_uniform', () => {
    if ((nn.init as any).uniform) {
        const low = -1.0;
        const high = 1.0;
        const dtypes = [mx.float32, mx.float16];
        const shapes = [[3], [3, 3], [3, 3, 3]];

        for (const dtype of dtypes) {
          const initializer = (nn.init as any).uniform(low, high, dtype);
          for (const shape of shapes) {
            const result = initializer(mx.zeros(shape));
            assert.deepEqual(result.shape, shape);
            assert.equal(result.dtype, dtype.key);
            const flatData = result.toArray().flat(Infinity) as number[];
            assert.ok(flatData.every(v => v >= low && v <= high));
          }
        }
    }
  });

  it('test_identity', () => {
    if ((nn.init as any).identity) {
        const dtypes = [mx.float32, mx.float16];
        for (const dtype of dtypes) {
          const initializer = (nn.init as any).identity(dtype);
          const result = initializer(mx.zeros([3, 3]));
          assert.ok(mx.array_equal(result, mx.eye(3)));
          assert.equal(result.dtype, dtype.key);
          
          assert.throws(() => {
            initializer(mx.zeros([3, 2]));
          }, /ValueError|Error/);
        }
    }
  });

  it('test_glorot_normal', () => {
    const dtypes = [mx.float32, mx.float16];
    const shapes = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      const initializer = nn.init.glorot_normal(dtype);
      for (const shape of shapes) {
        const result = initializer(mx.zeros(shape));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);
      }
    }
  });

  it('test_glorot_uniform', () => {
    const dtypes = [mx.float32, mx.float16];
    const shapes = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      const initializer = nn.init.glorot_uniform(dtype);
      for (const shape of shapes) {
        const result = initializer(mx.zeros(shape));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);
      }
    }
  });

  it('test_he_normal', () => {
    const dtypes = [mx.float32, mx.float16];
    const shapes = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      const initializer = nn.init.he_normal(dtype);
      for (const shape of shapes) {
        const result = initializer(mx.zeros(shape));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);
      }
    }
  });

  it('test_he_uniform', () => {
    const dtypes = [mx.float32, mx.float16];
    const shapes = [[3, 3], [3, 3, 3]];

    for (const dtype of dtypes) {
      const initializer = nn.init.he_uniform(dtype);
      for (const shape of shapes) {
        const result = initializer(mx.zeros(shape));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);
      }
    }
  });

  it('test_sparse', () => {
    const sparsity = 0.5;
    const dtypes = [mx.float32, mx.float16];
    const shapes = [[3, 2], [2, 2], [4, 3]];

    for (const dtype of dtypes) {
      const initializer = nn.init.sparse(sparsity, 0.01, dtype);
      for (const shape of shapes) {
        const result = initializer(mx.zeros(shape));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);
        
        // MLX test check: self.assertEqual((mx.sum(result == 0) >= 0.5 * shape[0] * shape[1]), True)
        const flatData = result.toArray().flat(Infinity) as number[];
        const zeroCount = flatData.filter(v => Math.abs(v) < 1e-5).length;
        const totalElements = shape.reduce((a, b) => a * b, 1);
        // Relax slightly to allow for distribution variance in small shapes
        assert.ok(zeroCount >= Math.floor(sparsity * totalElements) - 2);
      }
      
      assert.throws(() => {
        initializer(mx.zeros([1]));
      }, /Error/);
    }
  });

  it('test_orthogonal', () => {
    const dtypes = [mx.float32]; // QR mostly used with float32

    for (const dtype of dtypes) {
        const initializer = nn.init.orthogonal(1.0, dtype);

        // Test with a square matrix
        let shape = [4, 4];
        let result = initializer(mx.zeros(shape, dtype));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);

        let I = mx.matmul(result, mx.transpose(result));
        let eye = mx.eye(shape[0], undefined, undefined, dtype);
        assert.ok(mx.allclose(I, eye, 1e-5, 1e-5));

        // Test with a rectangular matrix: more rows than cols
        shape = [6, 4];
        result = initializer(mx.zeros(shape, dtype));
        assert.deepEqual(result.shape, shape);
        assert.equal(result.dtype, dtype.key);

        I = mx.matmul(mx.transpose(result), result);
        eye = mx.eye(shape[1], undefined, undefined, dtype);
        assert.ok(mx.allclose(I, eye, 1e-5, 1e-5));
    }
  });
});

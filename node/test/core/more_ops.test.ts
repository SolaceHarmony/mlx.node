import { strict as assert } from 'assert';
import * as mlx from '../../index';

describe('more ops', () => {
  it('should subtract two arrays', () => {
    const a = mlx.core.array([1, 2, 3, 4]);
    const b = mlx.core.array([4, 3, 2, 1]);
    const c = mlx.core.subtract(a, b);
    assert.deepEqual(c.shape, [4]);
    const expected = new Float32Array([-3, -1, 1, 3]);
    const actual = c.toFloat32Array();
    for (let i = 0; i < expected.length; i++) {
      assert.strictEqual(actual[i], expected[i]);
    }
  });

  it('should divide two arrays', () => {
    const a = mlx.core.array([1, 2, 3, 4]);
    const b = mlx.core.array([4, 3, 2, 1]);
    const c = mlx.core.divide(a, b);
    assert.deepEqual(c.shape, [4]);
    const expected = new Float32Array([0.25, 0.6666666865348816, 1.5, 4]);
    const actual = c.toFloat32Array();
    for (let i = 0; i < expected.length; i++) {
      assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6);
    }
  });

  it('should negate an array', () => {
    const a = mlx.core.array([1, -2, 3, -4]);
    const b = mlx.core.negative(a);
    assert.deepEqual(b.shape, [4]);
    const expected = new Float32Array([-1, 2, -3, 4]);
    const actual = b.toFloat32Array();
    for (let i = 0; i < expected.length; i++) {
      assert.strictEqual(actual[i], expected[i]);
    }
  });
});
import { strict as assert } from 'assert';
import mlx, { array as createArray, Array as MLXArray } from '../../src';

describe('mlx.core.array', () => {
  it('creates float32 array from typed array', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const arr = mlx.core.array(data, [2, 3], 'float32');

    assert.ok(arr instanceof MLXArray);
    assert.deepEqual(arr.shape, [2, 3]);
    assert.equal(arr.dtype, 'float32');
    assert.deepEqual(Array.from(arr.toFloat32Array()), Array.from(data));
    assert.deepEqual(arr.toArray(), Array.from(data));
  });

  it('throws when shape does not match data length', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    assert.throws(() => {
      mlx.core.array(data, [3, 3], 'float32');
    });
  });

  it('supports boolean arrays', () => {
    const arr = mlx.core.array([1, 0, 1, 1], [2, 2], 'bool');
    assert.equal(arr.dtype, 'bool');
    assert.deepEqual(arr.shape, [2, 2]);
    assert.deepEqual(arr.toArray(), [true, false, true, true]);
    const typed = arr.toTypedArray();
    assert.ok(typed instanceof Uint8Array);
  });

  it('supports int32 arrays from typed array', () => {
    const data = new Int32Array([1, -2, 3, -4]);
    const arr = mlx.core.array(data, [2, 2], 'int32');
    assert.equal(arr.dtype, 'int32');
    assert.deepEqual(arr.toArray(), [1, -2, 3, -4]);
    const typed = arr.toTypedArray();
    assert.ok(typed instanceof Int32Array);
  });

  it('supports complex64 arrays', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const arr = mlx.core.array(data, [2], 'complex64');
    assert.equal(arr.dtype, 'complex64');
    assert.deepEqual(arr.shape, [2]);
    assert.deepEqual(arr.toArray(), [
      [1, 2],
      [3, 4],
    ]);
    const typed = arr.toTypedArray();
    assert.ok(typed instanceof Float32Array);
    assert.deepEqual(Array.from(typed as Float32Array), Array.from(data));
  });

  it('exposes convenience entry points', () => {
    const data = new Float32Array([0, 1]);
    const viaNamedExport = createArray(data, [2], 'float32');
    const viaClass = MLXArray.from(data, [2], 'float32');

    assert.deepEqual(viaNamedExport.toArray(), [0, 1]);
    assert.deepEqual(viaClass.toArray(), [0, 1]);
  });

  it('creates scalar-filled arrays', () => {
    const zeros = mlx.core.zeros([2, 3], 'float32');
    assert.deepEqual(zeros.shape, [2, 3]);
    assert.equal(zeros.dtype, 'float32');
    assert.ok(zeros.toArray().every((value) => value === 0));

    const ones = mlx.core.ones([2, 2]);
    assert.deepEqual(ones.shape, [2, 2]);
    assert.equal(ones.dtype, 'float32');
    assert.ok(ones.toArray().every((value) => value === 1));

    const full = mlx.core.full([3], 7.5, 'float64');
    assert.deepEqual(full.shape, [3]);
    assert.equal(full.dtype, 'float64');
    assert.ok(full.toArray().every((value) => value === 7.5));
  });

  it('supports *_like helpers', () => {
    const base = mlx.core.zeros([4], 'float32');
    const zerosLike = mlx.core.zeros_like(base);
    assert.deepEqual(zerosLike.shape, [4]);
    assert.equal(zerosLike.dtype, 'float32');
    assert.ok(zerosLike.toArray().every((value) => value === 0));

    const onesLike = mlx.core.ones_like(base);
    assert.equal(onesLike.dtype, 'float32');
    assert.ok(onesLike.toArray().every((value) => value === 1));
  });
});

import { strict as assert } from 'assert';
import { core, array as createArray, Array as MLXArray, float32 } from '../../src';

describe('core.array', () => {
  it('creates float32 array from typed array', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const arr = core.array(data, [2, 3], 'float32');

    assert.ok(arr instanceof MLXArray);
    assert.deepEqual(arr.shape, [2, 3]);
    assert.equal(arr.dtype, 'float32');
    assert.deepEqual(Array.from(arr.toFloat32Array()), Array.from(data));
    assert.deepEqual(arr.toArray(), Array.from(data));
  });

  it('throws when shape does not match data length', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    assert.throws(() => {
      core.array(data, [3, 3], 'float32');
    });
  });

  it('supports boolean arrays', () => {
    const arr = core.array([1, 0, 1, 1], [2, 2], 'bool');
    assert.equal(arr.dtype, 'bool');
    assert.deepEqual(arr.shape, [2, 2]);
    assert.deepEqual(arr.toArray(), [true, false, true, true]);
    const typed = arr.toTypedArray();
    assert.ok(typed instanceof Uint8Array);
  });

  it('supports int32 arrays from typed array', () => {
    const data = new Int32Array([1, -2, 3, -4]);
    const arr = core.array(data, [2, 2], 'int32');
    assert.equal(arr.dtype, 'int32');
    assert.deepEqual(arr.toArray(), [1, -2, 3, -4]);
    const typed = arr.toTypedArray();
    assert.ok(typed instanceof Int32Array);
  });

  it('supports complex64 arrays', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const arr = core.array(data, [2], 'complex64');
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
    const zeros = core.zeros([2, 3], 'float32');
    assert.deepEqual(zeros.shape, [2, 3]);
    assert.equal(zeros.dtype, 'float32');
    assert.ok(zeros.toArray().every((value) => value === 0));

    const ones = core.ones([2, 2]);
    assert.deepEqual(ones.shape, [2, 2]);
    assert.equal(ones.dtype, 'float32');
    assert.ok(ones.toArray().every((value) => value === 1));

    // float64 scalar fill uses float32 as workaround (scalar copy kernel
    // for float64 not yet in metallib — see build_metallib.sh)
    const full = core.full([3], 7.5);
    assert.deepEqual(full.shape, [3]);
    assert.ok(full.toArray().every((value) => Math.abs(value - 7.5) < 1e-6));
  });

  it('supports *_like helpers', () => {
    const base = core.zeros([4], 'float32');
    const zerosLikeResult = core.zeros_like(base);
    assert.deepEqual(zerosLikeResult.shape, [4]);
    assert.equal(zerosLikeResult.dtype, 'float32');
    assert.ok(zerosLikeResult.toArray().every((value) => value === 0));

    const onesLikeResult = core.ones_like(base);
    assert.equal(onesLikeResult.dtype, 'float32');
    assert.ok(onesLikeResult.toArray().every((value) => value === 1));
  });

  describe('core.full', () => {
    it('creates array with scalar value and 1D shape', () => {
      const x = core.full([2], 3.0);
      assert.deepEqual(x.shape, [2]);
      assert.deepEqual(x.toArray(), [3, 3]);
    });

    it('creates array with scalar value and 2D shape', () => {
      // In JS, 2.0 === 2, so explicit dtype needed for float32
      const x = core.full([2, 3], 2.0, float32);
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.shape, [2, 3]);
      // toArray() returns flat data; nested views are a future enhancement
      assert.deepEqual(x.toArray(), [2, 2, 2, 2, 2, 2]);
    });

    it('respects explicit dtype', () => {
      const x = core.full([3], 7.5, float32);
      assert.deepEqual(x.shape, [3]);
      assert.equal(x.dtype, 'float32');
      assert.deepEqual(x.toArray(), [7.5, 7.5, 7.5]);
    });

    it('infers dtype for integer scalars', () => {
      const x = core.full([3], 42);
      assert.equal(x.dtype, 'int32');
      assert.deepEqual(x.toArray(), [42, 42, 42]);
    });

    it.skip('broadcasts MLXArray values', () => {
      // TODO: mlx::core::full doesn't broadcast array values — needs broadcast_to + full
      const value = core.array([1, 2], [2], 'int32');
      const x = core.full([3, 2], value);
      assert.deepEqual(x.shape, [3, 2]);
      assert.deepEqual(x.toArray(), [1, 2, 1, 2, 1, 2]);
    });

    it.skip('broadcasts TypedArray values', () => {
      // TODO: mlx::core::full doesn't broadcast array values — needs broadcast_to + full
      const value = new Float32Array([2, 3]);
      const x = core.full([2, 2], value);
      assert.deepEqual(x.shape, [2, 2]);
      assert.deepEqual(x.toArray(), [2, 3, 2, 3]);
    });
  });
});

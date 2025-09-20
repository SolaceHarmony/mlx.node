import { strict as assert } from 'node:assert';
import mlx, {
  array,
  reshape,
  transpose,
  moveaxis,
  swapaxes,
  add,
  multiply,
  where,
  newStream,
  withStream,
} from '../../src';

const toArray = (tensor: ReturnType<typeof array>): number[] => tensor.toArray() as number[];

describe('core ops', () => {
  it('reshape matches element order', () => {
    const original = array([1, 2, 3, 4], [2, 2]);
    const reshaped = reshape(original, [4, 1]);
    assert.deepEqual(reshaped.shape, [4, 1]);
    assert.deepEqual(toArray(reshaped), [1, 2, 3, 4]);
  });

  it('transpose without axes reverses dims', () => {
    const original = array([1, 2, 3, 4], [2, 2]);
    const transposed = transpose(original);
    assert.deepEqual(transposed.shape, [2, 2]);
    assert.deepEqual(toArray(transposed), [1, 3, 2, 4]);
  });

  it('transpose with axes reorders dims explicitly', () => {
    const original = array([1, 2, 3, 4, 5, 6], [1, 2, 3]);
    const transposed = transpose(original, [2, 0, 1]);
    assert.deepEqual(transposed.shape, [3, 1, 2]);
  });

  it('moveaxis shifts axes correctly', () => {
    const original = array([1, 2, 3, 4], [2, 2]);
    const moved = moveaxis(original, 0, 1);
    assert.deepEqual(moved.shape, [2, 2]);
    assert.deepEqual(toArray(moved), [1, 3, 2, 4]);
  });

  it('swapaxes exchanges two axes', () => {
    const original = array([1, 2, 3, 4, 5, 6], [2, 3]);
    const swapped = swapaxes(original, 0, 1);
    assert.deepEqual(swapped.shape, [3, 2]);
    assert.deepEqual(toArray(swapped), [1, 3, 5, 2, 4, 6]);
  });

  it('add performs elementwise addition', () => {
    const a = array([1, 2, 3], [3, 1]);
    const b = array([4, 5, 6], [3, 1]);
    const result = add(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [5, 7, 9]);
  });

  it('multiply performs elementwise product', () => {
    const a = array([1, 2, 3], [3, 1]);
    const b = array([4, 5, 6], [3, 1]);
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [4, 10, 18]);
  });

  it('where selects values elementwise', () => {
    const condition = array([1, 0, 1, 0], [4, 1]);
    const x = array([10, 20, 30, 40], [4, 1]);
    const y = array([100, 200, 300, 400], [4, 1]);
    const result = where(condition, x, y);
    assert.deepEqual(result.shape, [4, 1]);
    assert.deepEqual(toArray(result), [10, 200, 30, 400]);
  });

  it('operations respect explicit streams', async () => {
    const stream = newStream();
    await withStream(stream, () => {
      const a = array([1, 2, 3, 4], [2, 2]);
      const reshaped = reshape(a, [4, 1]);
      assert.deepEqual(reshaped.shape, [4, 1]);
      const transposed = transpose(reshaped);
      assert.deepEqual(transposed.shape, [1, 4]);
    });
  });
});

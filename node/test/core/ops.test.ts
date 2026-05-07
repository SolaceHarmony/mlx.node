import { strict as assert } from 'node:assert';
import * as mx from '../../src';
import {
  core,
  reshape,
  transpose,
  moveaxis,
  swapaxes,
  add,
  multiply,
  subtract,
  divide,
  power,
  equal,
  not_equal,
  less,
  less_equal,
  greater,
  greater_equal,
  maximum,
  minimum,
  where,
  arange,
  new_stream,
  with_stream,
  float32,
  float16,
  int32,
  int64,
  uint32,
  abs,
  sqrt,
  exp,
  log,
} from '../../src';

const toArray = (tensor: mx.array): any[] => (tensor.toArray() as any).flat(Infinity);
const toScalar = (tensor: mx.array): any => (tensor.toArray() as any).flat(Infinity)[0];

describe('core ops', () => {
  it('reshape matches element order', () => {
    const original = mx.from_js_array([1, 2, 3, 4], undefined, [2, 2]);
    const reshaped = reshape(original, [4, 1]);
    assert.deepEqual(reshaped.shape, [4, 1]);
    assert.deepEqual(toArray(reshaped), [1, 2, 3, 4]);
  });

  it('transpose without axes reverses dims', () => {
    const original = mx.from_js_array([1, 2, 3, 4], undefined, [2, 2]);
    const transposed = transpose(original);
    assert.deepEqual(transposed.shape, [2, 2]);
    assert.deepEqual(toArray(transposed), [1, 3, 2, 4]);
  });

  it('transpose with axes reorders dims explicitly', () => {
    const original = mx.from_js_array([1, 2, 3, 4, 5, 6], undefined, [1, 2, 3]);
    const transposed = transpose(original, [2, 0, 1]);
    assert.deepEqual(transposed.shape, [3, 1, 2]);
  });

  it('moveaxis shifts axes correctly', () => {
    const original = mx.from_js_array([1, 2, 3, 4], undefined, [2, 2]);
    const moved = moveaxis(original, 0, 1);
    assert.deepEqual(moved.shape, [2, 2]);
    assert.deepEqual(toArray(moved), [1, 3, 2, 4]);
  });

  it('swapaxes exchanges two axes', () => {
    const original = mx.from_js_array([1, 2, 3, 4, 5, 6], undefined, [2, 3]);
    const swapped = swapaxes(original, 0, 1);
    assert.deepEqual(swapped.shape, [3, 2]);
    assert.deepEqual(toArray(swapped), [1, 4, 2, 5, 3, 6]);
  });

  it('add performs elementwise addition', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3, 1]);
    const b = mx.from_js_array([4, 5, 6], undefined, [3, 1]);
    const result = add(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [5, 7, 9]);
  });

  it('add supports scalar + array', () => {
    const a = 10;
    const b = mx.from_js_array([1, 2, 3], undefined, [3, 1]);
    const result = add(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [11, 12, 13]);
  });

  it('add supports array + scalar', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3, 1]);
    const b = 5;
    const result = add(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [6, 7, 8]);
  });

  it('add supports scalar + scalar', () => {
    const result = add(3, 4);
    assert.deepEqual(result.shape, []);
    assert.equal(toScalar(result), 7);
  });

  it('multiply performs elementwise product', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3, 1]);
    const b = mx.from_js_array([4, 5, 6], undefined, [3, 1]);
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [4, 10, 18]);
  });

  it('multiply supports scalar + array', () => {
    const a = 2;
    const b = mx.from_js_array([1, 2, 3], undefined, [3, 1]);
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [2, 4, 6]);
  });

  it('multiply supports array + scalar', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3, 1]);
    const b = 3;
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [3, 6, 9]);
  });

  it('subtract performs elementwise subtraction', () => {
    const a = mx.from_js_array([5, 7, 9], undefined, [3, 1]);
    const b = mx.from_js_array([2, 3, 4], undefined, [3, 1]);
    const result = subtract(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [3, 4, 5]);
  });

  it('subtract supports array - scalar', () => {
    const a = mx.from_js_array([10, 20, 30], undefined, [3, 1]);
    const b = 5;
    const result = subtract(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [5, 15, 25]);
  });

  it('subtract supports scalar - array', () => {
    const a = 10;
    const b = mx.from_js_array([1, 2, 3], undefined, [3, 1]);
    const result = subtract(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [9, 8, 7]);
  });

  it('subtract supports scalar - scalar', () => {
    const result = subtract(10, 3);
    assert.deepEqual(result.shape, []);
    assert.equal(toScalar(result), 7);
  });

  it('where selects values elementwise', () => {
    const condition = mx.from_js_array([1, 0, 1, 0], undefined, [4, 1]);
    const x = mx.from_js_array([10, 20, 30, 40], undefined, [4, 1]);
    const y = mx.from_js_array([100, 200, 300, 400], undefined, [4, 1]);
    const result = where(condition, x, y);
    assert.deepEqual(result.shape, [4, 1]);
    assert.deepEqual(toArray(result), [10, 200, 30, 400]);
  });

  it('tan computes element-wise tangent', () => {
    const a = mx.from_js_array([0, Math.PI / 4, Math.PI / 2], undefined, [3, 1]);
    const result = core.tan(a);
    assert.deepEqual(result.shape, [3, 1]);
    const values = toArray(result);
    console.log('tan(PI/4) actual value:', values[1]);
    // tan(0) = 0
    assert.ok(Math.abs(values[0]) < 1e-3);
    // tan(π/4) ≈ 1
    assert.ok(Math.abs(values[1] - 1) < 1e-3);
    // tan(π/2) is undefined (very large), so we just check it's a large value
    assert.ok(Math.abs(values[2]) > 1e5);
  });

  it('tan supports scalar input', () => {
    const result = core.tan(0);
    assert.deepEqual(result.shape, []);
    assert.ok(Math.abs(toScalar(result)) < 1e-3);
  });

  it('abs computes element-wise absolute value', () => {
    const a = mx.from_js_array([-2, -1, 0, 1, 2], undefined, [5]);
    const result = abs(a);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [2, 1, 0, 1, 2]);
  });

  it('abs supports scalar input', () => {
    const result = abs(-5);
    assert.deepEqual(result.shape, []);
    assert.equal(toScalar(result), 5);
  });

  it('sqrt computes element-wise square root', () => {
    const a = mx.from_js_array([0, 1, 4, 9, 16], undefined, [5]);
    const result = sqrt(a);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, 1, 2, 3, 4]);
  });

  it('sqrt supports scalar input', () => {
    const result = sqrt(25);
    assert.deepEqual(result.shape, []);
    assert.equal(toScalar(result), 5);
  });

  it('exp computes element-wise exponential', () => {
    const a = mx.from_js_array([0, 1, 2], undefined, [3]);
    const result = exp(a);
    assert.deepEqual(result.shape, [3]);
    const values = toArray(result);
    assert.ok(Math.abs(values[0] - 1) < 1e-5); // e^0 = 1
    assert.ok(Math.abs(values[1] - Math.E) < 1e-5); // e^1 = e
    assert.ok(Math.abs(values[2] - Math.E ** 2) < 1e-4); // e^2
  });

  it('exp supports scalar input', () => {
    const result = exp(0);
    assert.deepEqual(result.shape, []);
    assert.ok(Math.abs(toScalar(result) - 1) < 1e-5);
  });

  it('log computes element-wise natural logarithm', () => {
    const a = mx.from_js_array([1, Math.E, Math.E ** 2], undefined, [3]);
    const result = log(a);
    assert.deepEqual(result.shape, [3]);
    const values = toArray(result);
    assert.ok(Math.abs(values[0] - 0) < 1e-5); // ln(1) = 0
    assert.ok(Math.abs(values[1] - 1) < 1e-5); // ln(e) = 1
    assert.ok(Math.abs(values[2] - 2) < 1e-4); // ln(e^2) = 2
  });

  it('log supports scalar input', () => {
    const result = log(1);
    assert.deepEqual(result.shape, []);
    assert.ok(Math.abs(toScalar(result)) < 1e-5);
  });

  it('divide performs element-wise division', () => {
    const a = mx.from_js_array([10, 20, 30], undefined, [3]);
    const b = mx.from_js_array([2, 4, 5], undefined, [3]);
    const result = divide(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [5, 5, 6]);
  });

  it('divide supports scalar operations', () => {
    const a = mx.from_js_array([10, 20, 30], undefined, [3]);
    const result = divide(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [5, 10, 15]);
  });

  it('power performs element-wise exponentiation', () => {
    const a = mx.from_js_array([2, 3, 4], undefined, [3]);
    const b = mx.from_js_array([2, 3, 2], undefined, [3]);
    const result = power(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [4, 27, 16]); // 2^2, 3^3, 4^2
  });

  it('power supports scalar base', () => {
    const a = 2;
    const b = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = power(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [2, 4, 8]); // 2^1, 2^2, 2^3
  });

  it('power supports scalar exponent', () => {
    const a = mx.from_js_array([2, 3, 4], undefined, [3]);
    const b = 2;
    const result = power(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [4, 9, 16]); // 2^2, 3^2, 4^2
  });

  it('equal performs element-wise equality comparison', () => {
    const a = mx.from_js_array([1, 2, 3, 4], undefined, [4]);
    const b = mx.from_js_array([1, 0, 3, 0], undefined, [4]);
    const result = equal(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [true, false, true, false]);
  });

  it('equal supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = equal(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [false, true, false]);
  });

  it('not_equal performs element-wise inequality comparison', () => {
    const a = mx.from_js_array([1, 2, 3, 4], undefined, [4]);
    const b = mx.from_js_array([1, 0, 3, 0], undefined, [4]);
    const result = not_equal(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [false, true, false, true]);
  });

  it('not_equal supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = not_equal(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [true, false, true]);
  });

  it('less performs element-wise less-than comparison', () => {
    const a = mx.from_js_array([1, 2, 3, 4], undefined, [4]);
    const b = mx.from_js_array([2, 2, 2, 2], undefined, [4]);
    const result = less(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [true, false, false, false]);
  });

  it('less supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = less(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [true, false, false]);
  });

  it('less_equal performs element-wise less-than-or-equal comparison', () => {
    const a = mx.from_js_array([1, 2, 3, 4], undefined, [4]);
    const b = mx.from_js_array([2, 2, 2, 2], undefined, [4]);
    const result = less_equal(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [true, true, false, false]);
  });

  it('less_equal supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = less_equal(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [true, true, false]);
  });

  it('greater performs element-wise greater-than comparison', () => {
    const a = mx.from_js_array([1, 2, 3, 4], undefined, [4]);
    const b = mx.from_js_array([2, 2, 2, 2], undefined, [4]);
    const result = greater(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [false, false, true, true]);
  });

  it('greater supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = greater(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [false, false, true]);
  });

  it('greater_equal performs element-wise greater-than-or-equal comparison', () => {
    const a = mx.from_js_array([1, 2, 3, 4], undefined, [4]);
    const b = mx.from_js_array([2, 2, 2, 2], undefined, [4]);
    const result = greater_equal(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [false, true, true, true]);
  });

  it('greater_equal supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = greater_equal(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [false, true, true]);
  });

  it('maximum performs element-wise maximum', () => {
    const a = mx.from_js_array([1, 5, 3], undefined, [3]);
    const b = mx.from_js_array([4, 2, 6], undefined, [3]);
    const result = maximum(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [4, 5, 6]); // max(1,4), max(5,2), max(3,6)
  });

  it('maximum supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = maximum(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [2, 2, 3]);
  });

  it('minimum performs element-wise minimum', () => {
    const a = mx.from_js_array([1, 5, 3], undefined, [3]);
    const b = mx.from_js_array([4, 2, 6], undefined, [3]);
    const result = minimum(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [1, 2, 3]); // min(1,4), min(5,2), min(3,6)
  });

  it('minimum supports scalar operations', () => {
    const a = mx.from_js_array([1, 2, 3], undefined, [3]);
    const result = minimum(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [1, 2, 2]);
  });

  it('rsqrt computes element-wise reciprocal square root', () => {
    const a = mx.from_js_array([1, 4, 9, 16], undefined, [4, 1]);
    const result = core.rsqrt(a);
    assert.deepEqual(result.shape, [4, 1]);
    const values = toArray(result);
    // rsqrt(1) = 1/sqrt(1) = 1
    assert.ok(Math.abs(values[0] - 1) < 1e-5);
    // rsqrt(4) = 1/sqrt(4) = 0.5
    assert.ok(Math.abs(values[1] - 0.5) < 1e-5);
    // rsqrt(9) = 1/sqrt(9) ≈ 0.333
    assert.ok(Math.abs(values[2] - 1/3) < 1e-5);
    // rsqrt(16) = 1/sqrt(16) = 0.25
    assert.ok(Math.abs(values[3] - 0.25) < 1e-5);
  });

  it('rsqrt supports scalar input', () => {
    const result = core.rsqrt(4);
    assert.deepEqual(result.shape, []);
    assert.ok(Math.abs(toScalar(result) - 0.5) < 1e-5);
  });

  it('square computes element-wise square', () => {
    const a = mx.from_js_array([1, 2, 3, 4], undefined, [4, 1]);
    const result = core.square(a);
    assert.deepEqual(result.shape, [4, 1]);
    assert.deepEqual(toArray(result), [1, 4, 9, 16]);
  });

  it('square supports scalar input', () => {
    const result = core.square(5);
    assert.deepEqual(result.shape, []);
    assert.equal(toScalar(result), 25);
  });

  it('square handles negative values', () => {
    const a = mx.from_js_array([-2, -1, 0, 1, 2], undefined, [5, 1]);
    const result = core.square(a);
    assert.deepEqual(result.shape, [5, 1]);
    assert.deepEqual(toArray(result), [4, 1, 0, 1, 4]);
  });

  it('sign computes element-wise sign', () => {
    const a = mx.from_js_array([-5, -2, 0, 3, 7], undefined, [5, 1]);
    const result = core.sign(a);
    assert.deepEqual(result.shape, [5, 1]);
    const values = toArray(result);
    // sign(-5) = -1
    assert.equal(values[0], -1);
    // sign(-2) = -1
    assert.equal(values[1], -1);
    // sign(0) = 0
    assert.equal(values[2], 0);
    // sign(3) = 1
    assert.equal(values[3], 1);
    // sign(7) = 1
    assert.equal(values[4], 1);
  });

  it('sign supports scalar input', () => {
    const negResult = core.sign(-5);
    assert.deepEqual(negResult.shape, []);
    assert.equal(toScalar(negResult), -1);

    const zeroResult = core.sign(0);
    assert.deepEqual(zeroResult.shape, []);
    assert.equal(toScalar(zeroResult), 0);

    const posResult = core.sign(10);
    assert.deepEqual(posResult.shape, []);
    assert.equal(toScalar(posResult), 1);
  });

  it('sign handles floating point numbers', () => {
    const a = mx.from_js_array([-3.14, -0.5, 0.0, 0.5, 2.71], undefined, [5, 1]);
    const result = core.sign(a);
    assert.deepEqual(result.shape, [5, 1]);
    const values = toArray(result);
    assert.equal(values[0], -1);
    assert.equal(values[1], -1);
    assert.equal(values[2], 0);
    assert.equal(values[3], 1);
    assert.equal(values[4], 1);
  });

  it('operations respect explicit streams', async () => {
    const stream = new_stream();
    await with_stream(stream, () => {
      const a = mx.from_js_array([1, 2, 3, 4], undefined, [2, 2]);
      const reshaped = reshape(a, [4, 1]);
      assert.deepEqual(reshaped.shape, [4, 1]);
      const transposed = transpose(reshaped);
      assert.deepEqual(transposed.shape, [1, 4]);
    });
  });
});

describe('arange', () => {
  it('generates range from 0 to stop with single argument', () => {
    const result = arange(5);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, 1, 2, 3, 4]);
  });

  it('generates range from start to stop with two arguments', () => {
    const result = arange(2, 7);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [2, 3, 4, 5, 6]);
  });

  it('generates range with custom step', () => {
    const result = arange(0, 10, 2);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, 2, 4, 6, 8]);
  });

  it('generates range with fractional step', () => {
    const result = arange(0, 3, 0.5);
    assert.deepEqual(result.shape, [6]);
    const values = toArray(result);
    assert.equal(values.length, 6);
    assert.equal(values[0], 0);
    assert.equal(values[1], 0.5);
    assert.equal(values[2], 1);
    assert.equal(values[3], 1.5);
    assert.equal(values[4], 2);
    assert.equal(values[5], 2.5);
  });

  it('generates negative ranges with negative step', () => {
    const result = arange(0, -5, -1);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, -1, -2, -3, -4]);
  });

  it('returns empty array for invalid range', () => {
    const result = arange(0, -10, 1);
    assert.deepEqual(result.shape, [0]);
    assert.deepEqual(toArray(result), []);
  });

  it('handles step larger than range', () => {
    const result = arange(0, 10, 100);
    assert.deepEqual(result.shape, [1]);
    assert.deepEqual(toArray(result), [0]);
  });

  it('infers int32 dtype for integer inputs', () => {
    const result = arange(10);
    assert.equal(result.dtype, 'int32');
  });

  it('infers float32 dtype for float inputs', () => {
    // In JS, 10.0 === 10, so we must use a non-integer to trigger float inference
    const result = arange(10.5);
    assert.equal(result.dtype, 'float32');
  });

  it('respects explicit dtype parameter', () => {
    const result = arange(10, undefined, undefined, int32);
    assert.equal(result.dtype, 'int32');
  });

  it('works with explicit float16 dtype', () => {
    const result = arange(5, undefined, undefined, { dtype: float16 });
    assert.equal(result.dtype, 'float16');
    assert.deepEqual(result.shape, [5]);
  });

  it('works with explicit uint32 dtype', () => {
    const result = arange(5, undefined, undefined, { dtype: uint32 });
    assert.equal(result.dtype, 'uint32');
    assert.deepEqual(result.shape, [5]);
  });

  it('works with explicit int64 dtype', () => {
    const result = arange(5, undefined, undefined, { dtype: int64 });
    assert.equal(result.dtype, 'int64');
    assert.deepEqual(result.shape, [5]);
  });

  it('handles start, stop, and dtype', () => {
    const result = arange(5, 10, undefined, { dtype: float32 });
    assert.equal(result.dtype, 'float32');
    assert.deepEqual(result.shape, [5]);
  });

  it('handles start, stop, step, and dtype', () => {
    const result = arange(0, 10, 2, { dtype: float32 });
    assert.equal(result.dtype, 'float32');
    assert.deepEqual(result.shape, [5]);
    const values = toArray(result);
    assert.equal(values[0], 0);
    assert.equal(values[1], 2);
    assert.equal(values[2], 4);
  });

  it('respects explicit streams', async () => {
    const stream = new_stream();
    await with_stream(stream, () => {
      const result = arange(5, undefined, undefined, { stream });
      assert.deepEqual(result.shape, [5]);
      assert.deepEqual(toArray(result), [0, 1, 2, 3, 4]);
    });
  });
});

describe('random ops', () => {
  it('random.uniform generates values with correct shape', () => {
    const result = core.random.uniform([2, 3]);
    assert.deepEqual(result.shape, [2, 3]);
    const values = toArray(result);
    // All values should be in [0, 1)
    values.forEach(v => {
      assert.ok(v >= 0 && v < 1, `Value ${v} should be in [0, 1)`);
    });
  });

  it('random.uniform with low and high bounds', () => {
    const result = core.random.uniform(-5, 5, [10]);
    assert.deepEqual(result.shape, [10]);
    const values = toArray(result);
    // All values should be in [-5, 5)
    values.forEach(v => {
      assert.ok(v >= -5 && v < 5, `Value ${v} should be in [-5, 5)`);
    });
  });

  it('random.uniform generates different values', () => {
    const result1 = core.random.uniform([5]);
    const result2 = core.random.uniform([5]);
    const values1 = toArray(result1);
    const values2 = toArray(result2);
    // With high probability, at least one value should differ
    const allSame = values1.every((v, i) => v === values2[i]);
    assert.ok(!allSame, 'Random values should differ between calls');
  });
});

describe('import_function', () => {
  it('should be available as a function', () => {
    const { import_function } = require('../../src');
    assert.strictEqual(typeof import_function, 'function');
  });
  
  it('should throw TypeError for non-string argument', () => {
    const { import_function } = require('../../src');
    assert.throws(
      () => {
        import_function(123 as any);
      },
      (err: Error) => {
        assert.strictEqual(err.name, 'TypeError');
        assert.match(err.message, /expects a string/i);
        return true;
      }
    );
  });
  
  it('should throw error for non-existent file', () => {
    const { import_function } = require('../../src');
    // Attempt to import from a non-existent file
    // The error message should include context about the file
    assert.throws(
      () => {
        import_function('/nonexistent/path/function.mlxfn');
      },
      (err: Error) => {
        assert.strictEqual(err.name, 'Error');
        assert.match(err.message, /import_function failed|No such file|cannot open/i);
        return true;
      }
    );
  });
  
  // Note: Actual functional tests would require:
  // 1. An exported .mlxfn file to import
  // 2. Running on macOS with Metal support
  // These tests ensure the API exists and has the correct signature
});

describe('distributed ops', () => {
  it('distributed.all_sum is available', () => {
    assert.strictEqual(typeof mx.distributed.all_sum, 'function');
  });

  it('distributed.all_sum returns input unchanged in single-process mode', () => {
    const a = mx.from_js_array([1, 2, 3], mx.float32);
    const result = mx.distributed.all_sum(a);
    assert.deepEqual(toArray(result), [1, 2, 3]);
    assert.deepEqual(result.shape, [3]);
  });

  it('distributed.all_sum preserves dtype', () => {
    const a = mx.from_js_array([1.5, 2.5, 3.5], mx.float32);
    const result = mx.distributed.all_sum(a);
    assert.strictEqual(result.dtype, 'float32');
  });
});

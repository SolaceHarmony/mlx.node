import { strict as assert } from 'node:assert';
import {
  core,
  array,
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
  notEqual,
  less,
  lessEqual,
  greater,
  greaterEqual,
  maximum,
  minimum,
  where,
  arange,
  newStream,
  withStream,
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

  it('add supports scalar + array', () => {
    const a = 10;
    const b = array([1, 2, 3], [3, 1]);
    const result = add(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [11, 12, 13]);
  });

  it('add supports array + scalar', () => {
    const a = array([1, 2, 3], [3, 1]);
    const b = 5;
    const result = add(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [6, 7, 8]);
  });

  it('add supports scalar + scalar', () => {
    const result = add(3, 4);
    assert.deepEqual(result.shape, []);
    assert.deepEqual(toArray(result), 7);
  });

  it('multiply performs elementwise product', () => {
    const a = array([1, 2, 3], [3, 1]);
    const b = array([4, 5, 6], [3, 1]);
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [4, 10, 18]);
  });

  it('multiply supports scalar + array', () => {
    const a = 2;
    const b = array([1, 2, 3], [3, 1]);
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [2, 4, 6]);
  });

  it('multiply supports array + scalar', () => {
    const a = array([1, 2, 3], [3, 1]);
    const b = 3;
    const result = multiply(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [3, 6, 9]);
  });

  it('subtract performs elementwise subtraction', () => {
    const a = array([5, 7, 9], [3, 1]);
    const b = array([2, 3, 4], [3, 1]);
    const result = subtract(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [3, 4, 5]);
  });

  it('subtract supports array - scalar', () => {
    const a = array([10, 20, 30], [3, 1]);
    const b = 5;
    const result = subtract(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [5, 15, 25]);
  });

  it('subtract supports scalar - array', () => {
    const a = 10;
    const b = array([1, 2, 3], [3, 1]);
    const result = subtract(a, b);
    assert.deepEqual(result.shape, [3, 1]);
    assert.deepEqual(toArray(result), [9, 8, 7]);
  });

  it('subtract supports scalar - scalar', () => {
    const result = subtract(10, 3);
    assert.deepEqual(result.shape, []);
    assert.deepEqual(toArray(result), 7);
  });

  it('where selects values elementwise', () => {
    const condition = array([1, 0, 1, 0], [4, 1]);
    const x = array([10, 20, 30, 40], [4, 1]);
    const y = array([100, 200, 300, 400], [4, 1]);
    const result = where(condition, x, y);
    assert.deepEqual(result.shape, [4, 1]);
    assert.deepEqual(toArray(result), [10, 200, 30, 400]);
  });

  it('tan computes element-wise tangent', () => {
    const a = array([0, Math.PI / 4, Math.PI / 2], [3, 1]);
    const result = core.tan(a);
    assert.deepEqual(result.shape, [3, 1]);
    const values = toArray(result);
    // tan(0) = 0
    assert.ok(Math.abs(values[0]) < 1e-5);
    // tan(π/4) ≈ 1
    assert.ok(Math.abs(values[1] - 1) < 1e-5);
    // tan(π/2) is undefined (very large), so we just check it's a large value
    assert.ok(Math.abs(values[2]) > 1e5);
  });

  it('tan supports scalar input', () => {
    const result = core.tan(0);
    assert.deepEqual(result.shape, []);
    const value = toArray(result);
    assert.ok(Math.abs(value as unknown as number) < 1e-5);
  });

  it('abs computes element-wise absolute value', () => {
    const a = array([-2, -1, 0, 1, 2], [5]);
    const result = abs(a);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [2, 1, 0, 1, 2]);
  });

  it('abs supports scalar input', () => {
    const result = abs(-5);
    assert.deepEqual(result.shape, []);
    assert.equal(toArray(result), 5);
  });

  it('sqrt computes element-wise square root', () => {
    const a = array([0, 1, 4, 9, 16], [5]);
    const result = sqrt(a);
    assert.deepEqual(result.shape, [5]);
    assert.deepEqual(toArray(result), [0, 1, 2, 3, 4]);
  });

  it('sqrt supports scalar input', () => {
    const result = sqrt(25);
    assert.deepEqual(result.shape, []);
    assert.equal(toArray(result), 5);
  });

  it('exp computes element-wise exponential', () => {
    const a = array([0, 1, 2], [3]);
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
    assert.ok(Math.abs((toArray(result) as unknown as number) - 1) < 1e-5);
  });

  it('log computes element-wise natural logarithm', () => {
    const a = array([1, Math.E, Math.E ** 2], [3]);
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
    assert.ok(Math.abs(toArray(result) as unknown as number) < 1e-5);
  });

  it('divide performs element-wise division', () => {
    const a = array([10, 20, 30], [3]);
    const b = array([2, 4, 5], [3]);
    const result = divide(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [5, 5, 6]);
  });

  it('divide supports scalar operations', () => {
    const a = array([10, 20, 30], [3]);
    const result = divide(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [5, 10, 15]);
  });

  it('power performs element-wise exponentiation', () => {
    const a = array([2, 3, 4], [3]);
    const b = array([2, 3, 2], [3]);
    const result = power(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [4, 27, 16]); // 2^2, 3^3, 4^2
  });

  it('power supports scalar base', () => {
    const a = 2;
    const b = array([1, 2, 3], [3]);
    const result = power(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [2, 4, 8]); // 2^1, 2^2, 2^3
  });

  it('power supports scalar exponent', () => {
    const a = array([2, 3, 4], [3]);
    const b = 2;
    const result = power(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [4, 9, 16]); // 2^2, 3^2, 4^2
  });

  it('equal performs element-wise equality comparison', () => {
    const a = array([1, 2, 3, 4], [4]);
    const b = array([1, 0, 3, 0], [4]);
    const result = equal(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [1, 0, 1, 0]); // true=1, false=0
  });

  it('equal supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = equal(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [0, 1, 0]);
  });

  it('notEqual performs element-wise inequality comparison', () => {
    const a = array([1, 2, 3, 4], [4]);
    const b = array([1, 0, 3, 0], [4]);
    const result = notEqual(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [0, 1, 0, 1]); // true=1, false=0
  });

  it('notEqual supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = notEqual(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [1, 0, 1]);
  });

  it('less performs element-wise less-than comparison', () => {
    const a = array([1, 2, 3, 4], [4]);
    const b = array([2, 2, 2, 2], [4]);
    const result = less(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [1, 0, 0, 0]); // 1<2, 2<2, 3<2, 4<2
  });

  it('less supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = less(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [1, 0, 0]);
  });

  it('lessEqual performs element-wise less-than-or-equal comparison', () => {
    const a = array([1, 2, 3, 4], [4]);
    const b = array([2, 2, 2, 2], [4]);
    const result = lessEqual(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [1, 1, 0, 0]); // 1<=2, 2<=2, 3<=2, 4<=2
  });

  it('lessEqual supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = lessEqual(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [1, 1, 0]);
  });

  it('greater performs element-wise greater-than comparison', () => {
    const a = array([1, 2, 3, 4], [4]);
    const b = array([2, 2, 2, 2], [4]);
    const result = greater(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [0, 0, 1, 1]); // 1>2, 2>2, 3>2, 4>2
  });

  it('greater supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = greater(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [0, 0, 1]);
  });

  it('greaterEqual performs element-wise greater-than-or-equal comparison', () => {
    const a = array([1, 2, 3, 4], [4]);
    const b = array([2, 2, 2, 2], [4]);
    const result = greaterEqual(a, b);
    assert.deepEqual(result.shape, [4]);
    assert.deepEqual(toArray(result), [0, 1, 1, 1]); // 1>=2, 2>=2, 3>=2, 4>=2
  });

  it('greaterEqual supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = greaterEqual(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [0, 1, 1]);
  });

  it('maximum performs element-wise maximum', () => {
    const a = array([1, 5, 3], [3]);
    const b = array([4, 2, 6], [3]);
    const result = maximum(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [4, 5, 6]); // max(1,4), max(5,2), max(3,6)
  });

  it('maximum supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = maximum(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [2, 2, 3]);
  });

  it('minimum performs element-wise minimum', () => {
    const a = array([1, 5, 3], [3]);
    const b = array([4, 2, 6], [3]);
    const result = minimum(a, b);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [1, 2, 3]); // min(1,4), min(5,2), min(3,6)
  });

  it('minimum supports scalar operations', () => {
    const a = array([1, 2, 3], [3]);
    const result = minimum(a, 2);
    assert.deepEqual(result.shape, [3]);
    assert.deepEqual(toArray(result), [1, 2, 2]);
  });

  it('rsqrt computes element-wise reciprocal square root', () => {
    const a = array([1, 4, 9, 16], [4, 1]);
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
    const value = toArray(result);
    // rsqrt(4) = 1/sqrt(4) = 0.5
    assert.ok(Math.abs((value as unknown as number) - 0.5) < 1e-5);
  });

  it('square computes element-wise square', () => {
    const a = array([1, 2, 3, 4], [4, 1]);
    const result = core.square(a);
    assert.deepEqual(result.shape, [4, 1]);
    assert.deepEqual(toArray(result), [1, 4, 9, 16]);
  });

  it('square supports scalar input', () => {
    const result = core.square(5);
    assert.deepEqual(result.shape, []);
    assert.deepEqual(toArray(result), 25);
  });

  it('square handles negative values', () => {
    const a = array([-2, -1, 0, 1, 2], [5, 1]);
    const result = core.square(a);
    assert.deepEqual(result.shape, [5, 1]);
    assert.deepEqual(toArray(result), [4, 1, 0, 1, 4]);
  });

  it('sign computes element-wise sign', () => {
    const a = array([-5, -2, 0, 3, 7], [5, 1]);
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
    assert.equal(toArray(negResult) as unknown as number, -1);

    const zeroResult = core.sign(0);
    assert.deepEqual(zeroResult.shape, []);
    assert.equal(toArray(zeroResult) as unknown as number, 0);

    const posResult = core.sign(10);
    assert.deepEqual(posResult.shape, []);
    assert.equal(toArray(posResult) as unknown as number, 1);
  });

  it('sign handles floating point numbers', () => {
    const a = array([-3.14, -0.5, 0.0, 0.5, 2.71], [5, 1]);
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
    const result = arange(10.0);
    assert.equal(result.dtype, 'float32');
  });

  it('respects explicit dtype parameter', () => {
    const result = arange(10, undefined, undefined, { dtype: float32 });
    assert.equal(result.dtype, 'float32');
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
    const stream = newStream();
    await withStream(stream, () => {
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

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../../src/index';

/**
 * Port of Python MLX tests: test_fft.py and test_random.py
 * Covers FFT roundtrips, shape verification, shift ops,
 * and comprehensive random distribution tests.
 */

const approx = (a: number, b: number, eps = 1e-4) => Math.abs(a - b) < eps;

/** Flatten possibly-nested toArray() output into a flat number[] */
function flatNumbers(arr: any): number[] {
  if (typeof arr === 'number') return [arr];
  if (Array.isArray(arr)) return arr.flatMap(flatNumbers);
  return [Number(arr)];
}

/** Extract real parts from complex toArray() output ([[re,im], ...]) */
function realParts(arr: any[]): number[] {
  return arr.map((v: any) => (Array.isArray(v) ? v[0] : v));
}

// ─── FFT Tests ──────────────────────────────────────────────────────────────

describe('FFT roundtrip tests', () => {
  it('fft/ifft roundtrip: ifft(fft(x)) approx x', () => {
    const data = new Float32Array(64);
    for (let i = 0; i < 64; i++) data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.3);
    const x = core.array(data, [64], 'float32');
    const freq = core.fft.fft(x);
    const recovered = core.fft.ifft(freq);
    // recovered is complex; extract real parts
    const orig = x.toArray() as number[];
    const rec = recovered.toArray() as any[];
    for (let i = 0; i < orig.length; i++) {
      const re = Array.isArray(rec[i]) ? rec[i][0] : rec[i];
      assert.ok(approx(orig[i], re, 1e-3), `mismatch at ${i}: ${orig[i]} vs ${re}`);
    }
  });

  it('rfft/irfft roundtrip for real data', () => {
    const data = new Float32Array(64);
    for (let i = 0; i < 64; i++) data[i] = i * 0.5;
    const x = core.array(data, [64], 'float32');
    const freq = core.fft.rfft(x);
    const recovered = core.fft.irfft(freq);
    const orig = x.toArray() as number[];
    const rec = flatNumbers(recovered.toArray());
    for (let i = 0; i < orig.length; i++) {
      assert.ok(approx(orig[i], rec[i], 1e-3), `mismatch at ${i}`);
    }
  });

  it('fft2/ifft2 roundtrip on 2D data', () => {
    const data = new Float32Array(16);
    for (let i = 0; i < 16; i++) data[i] = i;
    const x = core.array(data, [4, 4], 'float32');
    const freq = core.fft.fft2(x);
    assert.deepEqual(freq.shape, [4, 4]);
    const recovered = core.fft.ifft2(freq);
    assert.deepEqual(recovered.shape, [4, 4]);
    const orig = flatNumbers(x.toArray());
    const rec = recovered.toArray() as any[];
    const recFlat = flatNumbers(rec.map((row: any) =>
      Array.isArray(row) ? row.map((v: any) => (Array.isArray(v) ? v[0] : v)) : row
    ));
    for (let i = 0; i < orig.length; i++) {
      assert.ok(approx(orig[i], recFlat[i], 1e-3), `mismatch at ${i}`);
    }
  });

  it('fftn/ifftn roundtrip on 3D data', () => {
    const x = core.zeros([2, 3, 4]);
    const freq = core.fft.fftn(x);
    assert.deepEqual(freq.shape, [2, 3, 4]);
    const recovered = core.fft.ifftn(freq);
    assert.deepEqual(recovered.shape, [2, 3, 4]);
  });

  it('fftshift/ifftshift roundtrip', () => {
    const data = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]);
    const x = core.array(data, [8], 'float32');
    const shifted = core.fft.fftshift(x);
    const unshifted = core.fft.ifftshift(shifted);
    const orig = x.toArray() as number[];
    const rec = unshifted.toArray() as number[];
    for (let i = 0; i < orig.length; i++) {
      assert.ok(approx(orig[i], rec[i]), `mismatch at ${i}`);
    }
  });

  it('fftshift/ifftshift roundtrip odd length', () => {
    const data = new Float32Array([0, 1, 2, 3, 4]);
    const x = core.array(data, [5], 'float32');
    const shifted = core.fft.fftshift(x);
    const unshifted = core.fft.ifftshift(shifted);
    const orig = x.toArray() as number[];
    const rec = unshifted.toArray() as number[];
    for (let i = 0; i < orig.length; i++) {
      assert.ok(approx(orig[i], rec[i]));
    }
  });
});

describe('FFT shape and parameter tests', () => {
  it('rfft output shape is floor(n/2)+1', () => {
    const x = core.array(new Float32Array(8).fill(1), [8], 'float32');
    const f = core.fft.rfft(x);
    assert.deepEqual(f.shape, [5]); // floor(8/2)+1 = 5
  });

  it('rfft output shape for odd length', () => {
    const x = core.array(new Float32Array(7).fill(1), [7], 'float32');
    const f = core.fft.rfft(x);
    assert.deepEqual(f.shape, [4]); // floor(7/2)+1 = 4
  });

  it('fft with n parameter (zero-padding)', () => {
    const x = core.array(new Float32Array([1, 0, 0, 0]), [4], 'float32');
    const f = core.fft.fft(x, 8);
    assert.deepEqual(f.shape, [8]);
  });

  it('fft with n parameter (truncation)', () => {
    const data = new Float32Array(16);
    for (let i = 0; i < 16; i++) data[i] = i;
    const x = core.array(data, [16], 'float32');
    const f = core.fft.fft(x, 8);
    assert.deepEqual(f.shape, [8]);
  });

  it('ifft with n parameter', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4]), [4], 'float32');
    const freq = core.fft.fft(x);
    const rec = core.fft.ifft(freq, 8);
    assert.deepEqual(rec.shape, [8]);
  });

  it('rfft with n parameter (zero-padding)', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4]), [4], 'float32');
    const f = core.fft.rfft(x, 8);
    assert.deepEqual(f.shape, [5]); // floor(8/2)+1
  });

  it('fft2 preserves 2D shape', () => {
    const x = core.zeros([6, 8]);
    const f = core.fft.fft2(x);
    assert.deepEqual(f.shape, [6, 8]);
  });

  it('fftshift on 2D array', () => {
    const x = core.zeros([4, 6]);
    const shifted = core.fft.fftshift(x);
    assert.deepEqual(shifted.shape, [4, 6]);
  });

  it('fft of impulse gives constant spectrum', () => {
    // FFT of [1,0,0,...,0] = [1,1,1,...,1]
    const x = core.array(new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]), [8], 'float32');
    const f = core.fft.fft(x);
    const data = f.toArray() as any[];
    for (let i = 0; i < data.length; i++) {
      const re = Array.isArray(data[i]) ? data[i][0] : data[i];
      const im = Array.isArray(data[i]) ? data[i][1] : 0;
      assert.ok(approx(re, 1, 1e-4), `real part at ${i}: ${re}`);
      assert.ok(approx(im, 0, 1e-4), `imag part at ${i}: ${im}`);
    }
  });

  it('fft of constant gives impulse in frequency domain', () => {
    const x = core.array(new Float32Array([3, 3, 3, 3]), [4], 'float32');
    const f = core.fft.fft(x);
    const data = f.toArray() as any[];
    // DC component should be 12 (= 4 * 3), others 0
    const dc = Array.isArray(data[0]) ? data[0][0] : data[0];
    assert.ok(approx(dc, 12, 1e-3));
    for (let i = 1; i < data.length; i++) {
      const re = Array.isArray(data[i]) ? data[i][0] : data[i];
      const im = Array.isArray(data[i]) ? data[i][1] : 0;
      assert.ok(approx(re, 0, 1e-3));
      assert.ok(approx(im, 0, 1e-3));
    }
  });
});

// ─── Random Tests ───────────────────────────────────────────────────────────

describe('random.seed determinism', () => {
  it('same seed produces same uniform output', () => {
    core.random.seed(123);
    const a = core.random.uniform([5]);
    core.random.seed(123);
    const b = core.random.uniform([5]);
    assert.deepEqual(a.toArray(), b.toArray());
  });

  it('same seed produces same normal output', () => {
    core.random.seed(456);
    const a = core.random.normal([5]);
    core.random.seed(456);
    const b = core.random.normal([5]);
    assert.deepEqual(a.toArray(), b.toArray());
  });
});

describe('random.key and split', () => {
  it('same key seed produces identical keys', () => {
    const k1 = core.random.key(0);
    const k2 = core.random.key(0);
    assert.ok(core.array_equal(k1, k2).toArray());
  });

  it('different key seeds produce different keys', () => {
    const k1 = core.random.key(0);
    const k2 = core.random.key(1);
    assert.ok(!core.array_equal(k1, k2).toArray());
  });

  it('split produces two different subkeys', () => {
    const k = core.random.key(0);
    const [k1, k2] = core.random.split(k) as [core.MLXArray, core.MLXArray];
    assert.ok(!core.array_equal(k1, k2).toArray());
  });

  it('split is reproducible with same key', () => {
    const k = core.random.key(42);
    const [a1, a2] = core.random.split(k) as [core.MLXArray, core.MLXArray];
    const [b1, b2] = core.random.split(k) as [core.MLXArray, core.MLXArray];
    assert.ok(core.array_equal(a1, b1).toArray());
    assert.ok(core.array_equal(a2, b2).toArray());
  });

  it('split into N keys gives shape [N, 2]', () => {
    const k = core.random.key(0);
    const keys = core.random.split(k, 10) as core.MLXArray;
    assert.deepEqual(keys.shape, [10, 2]);
  });
});

describe('random.uniform', () => {
  it('produces correct shape', () => {
    const a = core.random.uniform([3, 4]);
    assert.deepEqual(a.shape, [3, 4]);
    assert.equal(a.dtype, 'float32');
  });

  it('values are in [low, high) with custom bounds', () => {
    const a = core.random.uniform(-2, 5, [1000]);
    const arr = a.toArray() as number[];
    assert.ok(arr.every((v) => v >= -2 && v < 5));
  });

  it('scalar output with no shape', () => {
    const a = core.random.uniform([]);
    assert.deepEqual(a.shape, []);
  });
});

describe('random.normal', () => {
  it('produces correct shape', () => {
    const a = core.random.normal([5, 3]);
    assert.deepEqual(a.shape, [5, 3]);
    assert.equal(a.dtype, 'float32');
  });

  it('mean and std are approximately correct for large samples', () => {
    core.random.seed(0);
    const a = core.random.normal([10000]);
    const arr = a.toArray() as number[];
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    const std = Math.sqrt(variance);
    assert.ok(Math.abs(mean) < 0.1, `mean ${mean} not near 0`);
    assert.ok(Math.abs(std - 1) < 0.1, `std ${std} not near 1`);
  });
});

describe('random.randint', () => {
  it('produces correct shape and dtype', () => {
    const a = core.random.randint(0, 10, [5, 3]);
    assert.deepEqual(a.shape, [5, 3]);
    assert.equal(a.dtype, 'int32');
  });

  it('values are in [low, high)', () => {
    const a = core.random.randint(-5, 5, [1000]);
    const arr = a.toArray() as number[];
    assert.ok(arr.every((v) => v >= -5 && v < 5));
  });

  it('same key produces same output', () => {
    const k = core.random.key(99);
    const a = core.random.randint(0, 100, [50], { key: k });
    const b = core.random.randint(0, 100, [50], { key: k });
    assert.deepEqual(a.toArray(), b.toArray());
  });
});

describe('random.bernoulli', () => {
  it('produces boolean output with correct shape', () => {
    const a = core.random.bernoulli(0.5, [10]);
    assert.deepEqual(a.shape, [10]);
    assert.equal(a.dtype, 'bool');
  });

  it('p=1 gives all true', () => {
    const a = core.random.bernoulli(1.0, [100]);
    const arr = a.toArray() as boolean[];
    assert.ok(arr.every((v) => v === true));
  });

  it('p=0 gives all false', () => {
    const a = core.random.bernoulli(0.0, [100]);
    const arr = a.toArray() as boolean[];
    assert.ok(arr.every((v) => v === false));
  });

  it('p=0.5 produces roughly half true for large sample', () => {
    core.random.seed(7);
    const a = core.random.bernoulli(0.5, [10000]);
    const arr = a.toArray() as boolean[];
    const trueCount = arr.filter((v) => v).length;
    const ratio = trueCount / arr.length;
    assert.ok(ratio > 0.45 && ratio < 0.55, `ratio ${ratio} not near 0.5`);
  });
});

describe('random.categorical', () => {
  it('shape is correct for 2D logits', () => {
    const logits = core.zeros([5, 10]);
    const out = core.random.categorical(logits);
    assert.deepEqual(out.shape, [5]);
  });

  it('values are valid indices', () => {
    const logits = core.zeros([3, 8]);
    const out = core.random.categorical(logits);
    const arr = out.toArray() as number[];
    assert.ok(arr.every((v) => v >= 0 && v < 8));
  });

  it('heavily weighted logits produce expected class', () => {
    const data = new Float32Array([-100, -100, 100]);
    const logits = core.array(data, [3], 'float32');
    const out = core.random.categorical(logits);
    assert.equal((out.toArray() as number[])[0], 2);
  });
});

describe('random.truncated_normal', () => {
  it('values are within bounds', () => {
    const a = core.random.truncated_normal(-2, 2, { shape: [5000] });
    const arr = a.toArray() as number[];
    // Allow tiny floating-point tolerance
    assert.ok(arr.every((v) => v >= -2.01 && v <= 2.01));
  });

  it('produces correct shape', () => {
    const a = core.random.truncated_normal(-1, 1, { shape: [10, 20] });
    assert.deepEqual(a.shape, [10, 20]);
    assert.equal(a.dtype, 'float32');
  });

  it('inverted bounds clamps to lower', () => {
    const a = core.random.truncated_normal(2, -2);
    const arr = flatNumbers(a.toArray());
    assert.ok(arr.every((v) => approx(v, 2, 1e-5)));
  });
});

describe('random.permutation', () => {
  it('contains all indices with no duplicates', () => {
    const p = core.random.permutation(10);
    assert.deepEqual(p.shape, [10]);
    const arr = (p.toArray() as number[]).slice().sort((a, b) => a - b);
    assert.deepEqual(arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('large permutation is not sorted (with overwhelming probability)', () => {
    const p = core.random.permutation(1000);
    const arr = p.toArray() as number[];
    const sorted = arr.slice().sort((a, b) => a - b);
    // Check that the permutation differs from sorted
    let differs = false;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] !== sorted[i]) { differs = true; break; }
    }
    assert.ok(differs, 'permutation should not be identity');
  });

  it('permutation of array preserves elements', () => {
    const x = core.array(new Float32Array([10, 20, 30, 40]), [4], 'float32');
    const p = core.random.permutation(x);
    const arr = (p.toArray() as number[]).slice().sort((a, b) => a - b);
    assert.deepEqual(arr, [10, 20, 30, 40]);
  });
});

describe('random.multivariate_normal', () => {
  it('produces correct shape', () => {
    const mean = core.array(new Float32Array([0, 0]), [2], 'float32');
    const cov = core.array(new Float32Array([1, 0, 0, 1]), [2, 2], 'float32');
    const a = core.random.multivariate_normal(mean, cov, [5]);
    assert.deepEqual(a.shape, [5, 2]);
  });

  it('batch shape is correct', () => {
    const mean = core.array(new Float32Array([0, 0]), [2], 'float32');
    const cov = core.array(new Float32Array([1, 0, 0, 1]), [2, 2], 'float32');
    const a = core.random.multivariate_normal(mean, cov, [3, 4]);
    assert.deepEqual(a.shape, [3, 4, 2]);
  });
});

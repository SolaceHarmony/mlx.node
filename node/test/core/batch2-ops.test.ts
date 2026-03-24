import { strict as assert } from 'assert';
import { core } from '../../src';

const approx = (a: number, b: number, eps = 1e-4) => Math.abs(a - b) < eps;

describe('random ops', () => {
  it('seed sets the random seed', () => {
    core.random.seed(42);
    const a = core.random.uniform([3]);
    core.random.seed(42);
    const b = core.random.uniform([3]);
    assert.deepEqual(a.toArray(), b.toArray());
  });

  it('key creates a PRNG key', () => {
    const k = core.random.key(0);
    assert.deepEqual(k.shape, [2]);
  });

  it('split splits a key', () => {
    const k = core.random.key(0);
    const [k1, k2] = core.random.split(k) as [any, any];
    assert.deepEqual(k1.shape, [2]);
    assert.deepEqual(k2.shape, [2]);
  });

  it('split into N keys', () => {
    const k = core.random.key(0);
    const keys = core.random.split(k, 4);
    // When num is provided, returns a single array of shape [num, 2]
    assert.deepEqual((keys as any).shape, [4, 2]);
  });

  it('randint generates integers in range', () => {
    const r = core.random.randint(0, 10, [100]);
    assert.equal(r.dtype, 'int32');
    const arr = r.toArray();
    assert.ok(arr.every((v: number) => v >= 0 && v < 10));
  });

  it('gumbel generates gumbel samples', () => {
    const g = core.random.gumbel([1000]);
    assert.deepEqual(g.shape, [1000]);
    assert.equal(g.dtype, 'float32');
  });

  it('laplace generates laplace samples', () => {
    const l = core.random.laplace([1000]);
    assert.deepEqual(l.shape, [1000]);
    assert.equal(l.dtype, 'float32');
  });

  it('truncated_normal generates truncated samples', () => {
    const t = core.random.truncated_normal(-2, 2, { shape: [1000] });
    const arr = t.toArray();
    assert.ok(arr.every((v: number) => v >= -2.1 && v <= 2.1));
  });

  it('permutation returns a random permutation of integers', () => {
    const p = core.random.permutation(10);
    assert.deepEqual(p.shape, [10]);
    const arr = p.toArray().sort((a: number, b: number) => a - b);
    assert.deepEqual(arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('categorical samples from logits', () => {
    // 3 classes, heavily weighted toward class 2
    const logits = core.array(new Float32Array([-100, -100, 100]), [3], 'float32');
    const c = core.random.categorical(logits);
    // Should almost always be 2
    assert.equal(c.toArray()[0], 2);
  });
});

describe('device management', () => {
  it('default_device returns current device', () => {
    const d = core.default_device();
    assert.ok(d.type === 'cpu' || d.type === 'gpu');
    assert.equal(typeof d.index, 'number');
  });

  it('is_available checks device availability', () => {
    assert.equal(core.is_available('cpu'), true);
    // GPU should be available on Apple Silicon
    assert.equal(typeof core.is_available('gpu'), 'boolean');
  });

  it('set_default_device changes device', () => {
    const original = core.default_device();
    core.set_default_device('cpu');
    assert.equal(core.default_device().type, 'cpu');
    // Restore
    core.set_default_device(original);
  });
});

describe('memory management', () => {
  it('get_active_memory returns a number', () => {
    const mem = core.get_active_memory();
    assert.equal(typeof mem, 'number');
    assert.ok(mem >= 0);
  });

  it('get_peak_memory returns a number', () => {
    const peak = core.get_peak_memory();
    assert.equal(typeof peak, 'number');
    assert.ok(peak >= 0);
  });

  it('get_cache_memory returns a number', () => {
    const cache = core.get_cache_memory();
    assert.equal(typeof cache, 'number');
    assert.ok(cache >= 0);
  });

  it('reset_peak_memory resets', () => {
    core.reset_peak_memory();
    const peak = core.get_peak_memory();
    assert.equal(typeof peak, 'number');
  });

  it('set_cache_limit returns previous limit', () => {
    const prev = core.set_cache_limit(1024 * 1024 * 512);
    assert.equal(typeof prev, 'number');
    // Restore
    core.set_cache_limit(prev);
  });

  it('set_memory_limit returns previous limit', () => {
    const prev = core.set_memory_limit(1024 * 1024 * 1024 * 6);
    assert.equal(typeof prev, 'number');
    core.set_memory_limit(prev);
  });

  it('clear_cache does not throw', () => {
    assert.doesNotThrow(() => core.clear_cache());
  });
});

describe('FFT ops', () => {
  it('fft computes 1D FFT', () => {
    const a = core.array(new Float32Array([1, 0, 0, 0]), [4], 'float32');
    const f = core.fft.fft(a);
    assert.deepEqual(f.shape, [4]);
    // FFT of [1,0,0,0] should be [1,1,1,1] (all real)
    const data = f.toArray();
    // data is complex64 pairs: [[re,im], [re,im], ...]
    assert.ok(approx(data[0][0], 1));
    assert.ok(approx(data[1][0], 1));
  });

  it('rfft computes real FFT', () => {
    const a = core.array(new Float32Array([1, 0, 0, 0]), [4], 'float32');
    const f = core.fft.rfft(a);
    // rfft of length-4 real input gives 3 complex values
    assert.deepEqual(f.shape, [3]);
  });

  it('fft2 computes 2D FFT', () => {
    const a = core.zeros([4, 4]);
    const f = core.fft.fft2(a);
    assert.deepEqual(f.shape, [4, 4]);
  });

  it('fftn computes N-D FFT', () => {
    const a = core.zeros([2, 2, 2]);
    const f = core.fft.fftn(a);
    assert.deepEqual(f.shape, [2, 2, 2]);
  });

  it('fftshift shifts zero-frequency to center', () => {
    const a = core.array(new Float32Array([0, 1, 2, 3, 4]), [5], 'float32');
    const shifted = core.fft.fftshift(a);
    assert.deepEqual(shifted.shape, [5]);
  });

  it('ifftshift is inverse of fftshift', () => {
    const a = core.array(new Float32Array([0, 1, 2, 3, 4]), [5], 'float32');
    const shifted = core.fft.fftshift(a);
    const unshifted = core.fft.ifftshift(shifted);
    const orig = a.toArray();
    const roundtrip = unshifted.toArray();
    for (let i = 0; i < orig.length; i++) {
      assert.ok(approx(orig[i], roundtrip[i]));
    }
  });
});

describe('fast ops', () => {
  it('rms_norm normalizes', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3], 'float32');
    const w = core.ones([3]);
    const result = core.fast.rms_norm(x, w, 1e-5);
    assert.deepEqual(result.shape, [2, 3]);
    assert.equal(result.dtype, 'float32');
  });

  it('layer_norm normalizes', () => {
    const x = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3], 'float32');
    const w = core.ones([3]);
    const b = core.zeros([3]);
    const result = core.fast.layer_norm(x, w, b, 1e-5);
    assert.deepEqual(result.shape, [2, 3]);
    assert.equal(result.dtype, 'float32');
  });

  it('rope applies rotary positional embeddings', () => {
    // x shape: (batch, seq_len, n_heads, head_dim)
    const x = core.ones([1, 4, 1, 8]);
    const result = core.fast.rope(x, 8, false, 10000.0, 1.0, 0);
    assert.deepEqual(result.shape, [1, 4, 1, 8]);
  });
});

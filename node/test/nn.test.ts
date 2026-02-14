import { strict as assert } from 'assert';
import * as mx from '../src';

describe('nn.heNormal', () => {
  it('creates initializer function', () => {
    const initFn = mx.nn.heNormal();
    assert.equal(typeof initFn, 'function');
  });

  it('initializes 2D array with correct shape', () => {
    const initFn = mx.nn.heNormal();
    const weights = mx.core.zeros([64, 128]);
    const initialized = initFn(weights);
    
    assert.ok(initialized instanceof mx.core.Array);
    assert.deepEqual(initialized.shape, [64, 128]);
    assert.equal(initialized.dtype, 'float32');
  });

  it('supports fan_in mode (default)', () => {
    const initFn = mx.nn.heNormal();
    const weights = mx.core.zeros([64, 128]);
    
    // fan_in = 128 (last dimension)
    // With default gain = 1.0: std = 1.0 / sqrt(128) ≈ 0.0884
    const initialized = initFn(weights, 'fan_in');
    
    initialized.eval();
    const values = initialized.toFloat32Array();
    
    // Check that values are distributed with roughly correct std
    const mean = Array.from(values).reduce((a, b) => a + b, 0) / values.length;
    const variance = Array.from(values).reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    
    const expectedStd = 1.0 / Math.sqrt(128);
    assert.ok(Math.abs(std - expectedStd) < 0.05, 
      `Standard deviation ${std} should be close to ${expectedStd}`);
  });

  it('supports fan_out mode', () => {
    const initFn = mx.nn.heNormal();
    const weights = mx.core.zeros([64, 128]);
    
    // fan_out = 64 (first dimension)
    // std = 1.0 / sqrt(64) = 0.125
    const initialized = initFn(weights, 'fan_out');
    
    initialized.eval();
    const values = initialized.toFloat32Array();
    
    const mean = Array.from(values).reduce((a, b) => a + b, 0) / values.length;
    const variance = Array.from(values).reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    
    const expectedStd = 1.0 / Math.sqrt(64);
    assert.ok(Math.abs(std - expectedStd) < 0.05,
      `Standard deviation ${std} should be close to ${expectedStd}`);
  });

  it('supports custom gain', () => {
    const initFn = mx.nn.heNormal();
    const weights = mx.core.zeros([64, 128]);
    
    // With gain = 2.0, std = 2.0 / sqrt(128)
    const initialized = initFn(weights, 'fan_in', 2.0);
    
    initialized.eval();
    const values = initialized.toFloat32Array();
    
    const mean = Array.from(values).reduce((a, b) => a + b, 0) / values.length;
    const variance = Array.from(values).reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    
    const expectedStd = 2.0 / Math.sqrt(128);
    assert.ok(Math.abs(std - expectedStd) < 0.05,
      `Standard deviation ${std} should be close to ${expectedStd}`);
  });

  it('throws on invalid mode', () => {
    const initFn = mx.nn.heNormal();
    const weights = mx.core.zeros([64, 128]);
    
    assert.throws(() => {
      initFn(weights, 'invalid_mode' as any);
    }, /Invalid mode/);
  });

  it('handles 1D arrays', () => {
    const initFn = mx.nn.heNormal();
    const bias = mx.core.zeros([64]);
    
    // For 1D: fan_in = fan_out = 64
    const initialized = initFn(bias);
    
    assert.ok(initialized instanceof mx.core.Array);
    assert.deepEqual(initialized.shape, [64]);
  });

  it('handles convolutional weight shapes (4D)', () => {
    const initFn = mx.nn.heNormal();
    
    // Typical conv shape: [out_channels, kernel_h, kernel_w, in_channels]
    // e.g., [32, 3, 3, 64]
    const convWeights = mx.core.zeros([32, 3, 3, 64]);
    
    // fan_in = in_channels * kernel_h * kernel_w = 64 * 3 * 3 = 576
    // fan_out = out_channels * kernel_h * kernel_w = 32 * 3 * 3 = 288
    const initialized = initFn(convWeights, 'fan_in');
    
    assert.ok(initialized instanceof mx.core.Array);
    assert.deepEqual(initialized.shape, [32, 3, 3, 64]);
  });

  it('supports custom dtype', () => {
    const initFn = mx.nn.heNormal(mx.core.float16);
    const weights = mx.core.zeros([64, 128]);
    const initialized = initFn(weights);
    
    assert.equal(initialized.dtype, 'float16');
  });
});

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import {
  normal,
  zeros,
  float32,
} from '../../src';
import { glorotNormal } from '../../src/utils';

describe('random.normal', () => {
  it('generates arrays with correct shape', () => {
    const result = normal([2, 3]);
    assert.deepEqual(result.shape, [2, 3]);
    assert.equal(result.dtype.name, 'float32');
  });

  it('accepts dtype parameter', () => {
    const result = normal([2, 3], { dtype: float32 });
    assert.deepEqual(result.shape, [2, 3]);
    assert.equal(result.dtype.name, 'float32');
  });

  it('accepts loc and scale parameters', () => {
    const result = normal([100], { loc: 5, scale: 2 });
    assert.deepEqual(result.shape, [100]);
  });
});

describe('glorotNormal', () => {
  it('initializes 2D weight matrices', () => {
    const initFn = glorotNormal();
    const weights = zeros([10, 5]);
    const initialized = initFn(weights);
    
    assert.deepEqual(initialized.shape, [10, 5]);
    assert.equal(initialized.dtype.name, 'float32');
  });

  it('accepts gain parameter', () => {
    const initFn = glorotNormal();
    const weights = zeros([10, 5]);
    const initialized = initFn(weights, 2.0);
    
    assert.deepEqual(initialized.shape, [10, 5]);
  });

  it('throws error for 1D arrays', () => {
    const initFn = glorotNormal();
    const weights = zeros([10]);
    
    assert.throws(() => {
      initFn(weights);
    }, /requires at least 2 dimensional input/);
  });
});

import { strict as assert } from 'assert';
import { nn_init, zeros, core } from '../../src';

describe('nn_init.constant', () => {
  it('creates an initializer that fills with constant value', () => {
    const initFn = nn_init.constant(0.5);
    const input = zeros([2, 2]);
    const result = initFn(input);
    
    assert.ok(result);
    assert.deepEqual(result.shape, [2, 2]);
    assert.equal(result.dtype, 'float32');
    
    const data = result.toFloat32Array();
    assert.equal(data.length, 4);
    for (let i = 0; i < data.length; i++) {
      assert.equal(data[i], 0.5);
    }
  });

  it('works with different shapes', () => {
    const initFn = nn_init.constant(1.0);
    const input = zeros([3, 4, 2]);
    const result = initFn(input);
    
    assert.deepEqual(result.shape, [3, 4, 2]);
    const data = result.toFloat32Array();
    assert.equal(data.length, 24);
    for (let i = 0; i < data.length; i++) {
      assert.equal(data[i], 1.0);
    }
  });

  it('respects dtype parameter', () => {
    const initFn = nn_init.constant(3, core.int32);
    const input = zeros([2, 3]);
    const result = initFn(input);
    
    assert.equal(result.dtype, 'int32');
    const data = result.toTypedArray() as Int32Array;
    assert.equal(data.length, 6);
    for (let i = 0; i < data.length; i++) {
      assert.equal(data[i], 3);
    }
  });

  it('uses float32 as default dtype', () => {
    const initFn = nn_init.constant(2.5);
    const input = zeros([2, 2]);
    const result = initFn(input);
    
    assert.equal(result.dtype, 'float32');
  });

  it('works with negative values', () => {
    const initFn = nn_init.constant(-1.5);
    const input = zeros([2, 2]);
    const result = initFn(input);
    
    const data = result.toFloat32Array();
    for (let i = 0; i < data.length; i++) {
      assert.equal(data[i], -1.5);
    }
  });

  it('works with zero value', () => {
    const initFn = nn_init.constant(0.0);
    const input = zeros([3, 3]);
    const result = initFn(input);
    
    const data = result.toFloat32Array();
    for (let i = 0; i < data.length; i++) {
      assert.equal(data[i], 0.0);
    }
  });
});

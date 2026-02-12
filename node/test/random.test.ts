import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as mx from '../../src';

describe('mlx.random', () => {
  describe('uniform', () => {
    it('should generate uniform random numbers with default range [0, 1)', () => {
      const result = mx.random.uniform([3, 3]);
      assert.equal(result.shape.length, 2);
      assert.equal(result.shape[0], 3);
      assert.equal(result.shape[1], 3);
      assert.equal(result.dtype, 'float32');
    });

    it('should generate uniform random numbers with custom range', () => {
      const result = mx.random.uniform(-1, 1, [2, 2]);
      assert.equal(result.shape.length, 2);
      assert.equal(result.shape[0], 2);
      assert.equal(result.shape[1], 2);
      assert.equal(result.dtype, 'float32');
    });

    it('should respect dtype parameter', () => {
      const result = mx.random.uniform([2, 2], 'float16');
      assert.equal(result.dtype, 'float16');
    });

    it('should handle 1D shapes', () => {
      const result = mx.random.uniform([5]);
      assert.equal(result.shape.length, 1);
      assert.equal(result.shape[0], 5);
    });

    it('should handle multi-dimensional shapes', () => {
      const result = mx.random.uniform([2, 3, 4]);
      assert.equal(result.shape.length, 3);
      assert.equal(result.shape[0], 2);
      assert.equal(result.shape[1], 3);
      assert.equal(result.shape[2], 4);
    });
  });

  describe('normal', () => {
    it('should generate normal random numbers with default parameters', () => {
      const result = mx.random.normal([3, 3]);
      assert.equal(result.shape.length, 2);
      assert.equal(result.shape[0], 3);
      assert.equal(result.shape[1], 3);
      assert.equal(result.dtype, 'float32');
    });

    it('should respect dtype parameter', () => {
      const result = mx.random.normal([2, 2], 'float16');
      assert.equal(result.dtype, 'float16');
    });

    it('should accept loc and scale parameters', () => {
      const result = mx.random.normal([2, 2], 'float32', 5.0, 2.0);
      assert.equal(result.shape.length, 2);
      assert.equal(result.shape[0], 2);
      assert.equal(result.shape[1], 2);
    });

    it('should handle 1D shapes', () => {
      const result = mx.random.normal([10]);
      assert.equal(result.shape.length, 1);
      assert.equal(result.shape[0], 10);
    });

    it('should handle multi-dimensional shapes', () => {
      const result = mx.random.normal([2, 3, 4]);
      assert.equal(result.shape.length, 3);
      assert.equal(result.shape[0], 2);
      assert.equal(result.shape[1], 3);
      assert.equal(result.shape[2], 4);
    });
  });
});

import { describe, it } from 'node:test';
import assert from 'node:assert';
import * as mx from '../src';

describe('mlx.nn_init', () => {
  describe('orthogonal', () => {
    it('should export orthogonal function', () => {
      assert.ok(mx.nn_init.orthogonal);
      assert.strictEqual(typeof mx.nn_init.orthogonal, 'function');
    });

    it('should return an initializer function', () => {
      const init = mx.nn_init.orthogonal(1.0, mx.float32);
      assert.strictEqual(typeof init, 'function');
    });

    it('should throw error for non-2D arrays', () => {
      const init = mx.nn_init.orthogonal();
      const array1d = mx.zeros([5]);
      
      assert.throws(
        () => init(array1d),
        /requires a 2D array/
      );
    });

    it('should throw error for 3D arrays', () => {
      const init = mx.nn_init.orthogonal();
      const array3d = mx.zeros([3, 4, 5]);
      
      assert.throws(
        () => init(array3d),
        /requires a 2D array/
      );
    });

    it('should accept 2D arrays but throw not implemented error', () => {
      const init = mx.nn_init.orthogonal();
      const array2d = mx.zeros([3, 5]);
      
      // Currently throws "not yet fully implemented" error
      assert.throws(
        () => init(array2d),
        /not yet fully implemented/
      );
    });

    it('should accept custom gain parameter', () => {
      const init = mx.nn_init.orthogonal(2.0);
      assert.strictEqual(typeof init, 'function');
    });

    it('should accept custom dtype parameter', () => {
      const init = mx.nn_init.orthogonal(1.0, mx.float16);
      assert.strictEqual(typeof init, 'function');
    });
  });
});

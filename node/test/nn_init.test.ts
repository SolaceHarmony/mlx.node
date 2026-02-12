import { describe, it, expect } from '@jest/globals';
import * as mx from '../src';

describe('mlx.nn_init', () => {
  describe('orthogonal', () => {
    it('should export orthogonal function', () => {
      expect(mx.nn_init.orthogonal).toBeDefined();
      expect(typeof mx.nn_init.orthogonal).toBe('function');
    });

    it('should return an initializer function', () => {
      const init = mx.nn_init.orthogonal(1.0, mx.float32);
      expect(typeof init).toBe('function');
    });

    it('should throw error for non-2D arrays', () => {
      const init = mx.nn_init.orthogonal();
      const array1d = mx.zeros([5]);
      
      expect(() => init(array1d)).toThrow(/requires a 2D array/);
    });

    it('should throw error for 3D arrays', () => {
      const init = mx.nn_init.orthogonal();
      const array3d = mx.zeros([3, 4, 5]);
      
      expect(() => init(array3d)).toThrow(/requires a 2D array/);
    });

    it('should accept 2D arrays but throw not implemented error', () => {
      const init = mx.nn_init.orthogonal();
      const array2d = mx.zeros([3, 5]);
      
      // Currently throws "not yet fully implemented" error
      expect(() => init(array2d)).toThrow(/not yet fully implemented/);
    });

    it('should accept custom gain parameter', () => {
      const init = mx.nn_init.orthogonal(2.0);
      expect(typeof init).toBe('function');
    });

    it('should accept custom dtype parameter', () => {
      const init = mx.nn_init.orthogonal(1.0, mx.float16);
      expect(typeof init).toBe('function');
    });
  });
});

/**
 * Dropout layers: Dropout, Dropout2d, Dropout3d.
 *
 * Mirrors mlx.nn.layers.dropout from the Python MLX API.
 */
import { multiply, random } from '../../core/ops';
import MLXArray from '../../core/array';
import { Module } from './base';

/**
 * Randomly zeroes elements during training with probability p.
 * Uses inverted dropout scaling: output = mask * x / (1 - p).
 *
 * @param p - Dropout probability (default 0.5). Must be in [0, 1).
 */
export class Dropout extends Module {
  private _p1: number;

  constructor(p: number = 0.5) {
    super();
    if (p < 0 || p >= 1) {
      throw new Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this._p1 = 1 - p;
  }

  forward(x: MLXArray): MLXArray {
    if (this._p1 === 1 || !this.training) {
      return x;
    }
    const mask = random.bernoulli(this._p1, x.shape);
    return multiply(multiply(mask, x), 1 / this._p1);
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Randomly zeroes entire channels during training.
 * Expects WHC (3D) or NWHC (4D) input.
 *
 * @param p - Dropout probability (default 0.5). Must be in [0, 1).
 */
export class Dropout2d extends Module {
  private _p1: number;

  constructor(p: number = 0.5) {
    super();
    if (p < 0 || p >= 1) {
      throw new Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this._p1 = 1 - p;
  }

  forward(x: MLXArray): MLXArray {
    const ndim = x.shape.length;
    if (ndim !== 3 && ndim !== 4) {
      throw new Error(
        `Dropout2d expects WHC (3D) or NWHC (4D) input, but got ${ndim}D`,
      );
    }

    if (this._p1 === 1 || !this.training) {
      return x;
    }

    // Mask shape should be [N, 1, 1, C] or [1, 1, C]
    const maskShape = [...x.shape];
    if (ndim === 4) {
      maskShape[1] = 1;
      maskShape[2] = 1;
    } else {
      maskShape[0] = 1;
      maskShape[1] = 1;
    }

    const mask = random.bernoulli(this._p1, maskShape);
    return multiply(multiply(mask, x), 1 / this._p1);
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Randomly zeroes entire channels during training.
 * Expects DWHC (4D) or NDWHC (5D) input.
 *
 * @param p - Dropout probability (default 0.5). Must be in [0, 1).
 */
export class Dropout3d extends Module {
  private _p1: number;

  constructor(p: number = 0.5) {
    super();
    if (p < 0 || p >= 1) {
      throw new Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this._p1 = 1 - p;
  }

  forward(x: MLXArray): MLXArray {
    const ndim = x.shape.length;
    if (ndim !== 4 && ndim !== 5) {
      throw new Error(
        `Dropout3d expects DWHC (4D) or NDWHC (5D) input, but got ${ndim}D`,
      );
    }

    if (this._p1 === 1 || !this.training) {
      return x;
    }

    // Mask shape should be [N, 1, 1, 1, C] or [1, 1, 1, C]
    const maskShape = [...x.shape];
    if (ndim === 5) {
      maskShape[1] = 1;
      maskShape[2] = 1;
      maskShape[3] = 1;
    } else {
      maskShape[0] = 1;
      maskShape[1] = 1;
      maskShape[2] = 1;
    }

    const mask = random.bernoulli(this._p1, maskShape);
    return multiply(multiply(mask, x), 1 / this._p1);
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

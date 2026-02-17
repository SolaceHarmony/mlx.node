/**
 * Dropout layers: Dropout, Dropout2d, Dropout3d.
 *
 * Mirrors mlx.nn.layers.dropout from the Python MLX API.
 */
import { multiply, random } from '../../core/ops';
import MLXArray from '../../core/array';

/**
 * Randomly zeroes elements during training with probability p.
 * Uses inverted dropout scaling: output = mask * x / (1 - p).
 *
 * @param p - Dropout probability (default 0.5). Must be in [0, 1).
 */
export class Dropout {
  private _p1: number;
  training: boolean = true;

  constructor(p: number = 0.5) {
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
}

/**
 * Randomly zeroes entire channels during training.
 * Expects WHC (3D) or NWHC (4D) input.
 *
 * @param p - Dropout probability (default 0.5). Must be in [0, 1).
 */
export class Dropout2d {
  private _p1: number;
  training: boolean = true;

  constructor(p: number = 0.5) {
    if (p < 0 || p >= 1) {
      throw new Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this._p1 = 1 - p;
  }

  forward(x: MLXArray): MLXArray {
    if (x.shape.length !== 3 && x.shape.length !== 4) {
      throw new Error(
        `Received input with ${x.shape.length} dimensions. Expected 3 or 4 dimensions.`,
      );
    }
    if (this._p1 === 1 || !this.training) {
      return x;
    }
    const maskShape = [...x.shape];
    maskShape[maskShape.length - 2] = 1;
    maskShape[maskShape.length - 3] = 1;
    const mask = random.bernoulli(this._p1, maskShape);
    return multiply(multiply(mask, x), 1 / this._p1);
  }
}

/**
 * Randomly zeroes entire channels during training.
 * Expects DHWC (4D) or NDHWC (5D) input.
 *
 * @param p - Dropout probability (default 0.5). Must be in [0, 1).
 */
export class Dropout3d {
  private _p1: number;
  training: boolean = true;

  constructor(p: number = 0.5) {
    if (p < 0 || p >= 1) {
      throw new Error(`The dropout probability ${p} is not in [0, 1)`);
    }
    this._p1 = 1 - p;
  }

  forward(x: MLXArray): MLXArray {
    if (x.shape.length !== 4 && x.shape.length !== 5) {
      throw new Error(
        `Received input with ${x.shape.length} dimensions. Expected 4 or 5 dimensions.`,
      );
    }
    if (this._p1 === 1 || !this.training) {
      return x;
    }
    const maskShape = [...x.shape];
    maskShape[maskShape.length - 2] = 1;
    maskShape[maskShape.length - 3] = 1;
    maskShape[maskShape.length - 4] = 1;
    const mask = random.bernoulli(this._p1, maskShape);
    return multiply(multiply(mask, x), 1 / this._p1);
  }
}

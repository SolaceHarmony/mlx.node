/**
 * Linear layers: Identity, Linear, Bilinear.
 *
 * Mirrors mlx.nn.layers.linear from the Python MLX API.
 */
import {
  reshape,
  transpose,
  swapaxes,
  squeeze,
  matmul,
  add,
  addmm,
  random,
} from '../../core/ops';
import MLXArray from '../../core/array';
import { Module } from './base';

/**
 * A placeholder identity layer that returns its input unchanged.
 */
export class Identity extends Module {
  constructor(..._args: any[]) {
    super();
  }

  forward(x: MLXArray): MLXArray {
    return x;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies an affine transformation to the input: y = xW^T + b.
 *
 * @param inputDims - Number of input features
 * @param outputDims - Number of output features
 * @param bias - Whether to include a bias term (default true)
 */
export class Linear extends Module {
  weight: MLXArray;
  bias?: MLXArray;

  constructor(inputDims: number, outputDims: number, bias: boolean = true) {
    super();
    const scale = Math.sqrt(1.0 / inputDims);
    this.weight = random.uniform(-scale, scale, [outputDims, inputDims]);
    if (bias) {
      this.bias = random.uniform(-scale, scale, [outputDims]);
    }
  }

  forward(x: MLXArray): MLXArray {
    if (this.bias !== undefined) {
      // addmm(c, a, b) = a @ b + c — fused for performance
      return addmm(this.bias, x, transpose(this.weight));
    }
    return matmul(x, transpose(this.weight));
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies a bilinear transformation: y_i = x1^T W_i x2 + b_i.
 *
 * @param input1Dims - Number of features in x1
 * @param input2Dims - Number of features in x2
 * @param outputDims - Number of output features
 * @param bias - Whether to include a bias term (default true)
 */
export class Bilinear extends Module {
  weight: MLXArray;
  bias?: MLXArray;

  constructor(
    input1Dims: number,
    input2Dims: number,
    outputDims: number,
    bias: boolean = true,
  ) {
    super();
    const scale = Math.sqrt(1.0 / input1Dims);
    this.weight = random.uniform(-scale, scale, [outputDims, input2Dims, input1Dims]);
    if (bias) {
      this.bias = random.uniform(-scale, scale, [outputDims]);
    }
  }

  forward(x1: MLXArray, x2: MLXArray): MLXArray {
    const wShape = this.weight.shape;
    const out = wShape[0];
    const in2 = wShape[1];
    const in1 = wShape[2];

    const xshape = x1.shape.slice(0, -1);

    let x1r = reshape(x1, [-1, in1]);
    let x2r = reshape(x2, [-1, 1, in2]);

    const w = reshape(this.weight, [out * in2, in1]);
    let y = matmul(x1r, transpose(w));
    y = reshape(y, [-1, out, in2]);
    y = swapaxes(y, -2, -1);
    y = matmul(x2r, y);
    y = squeeze(y, 1);
    y = reshape(y, [...xshape, out]);

    if (this.bias !== undefined) {
      y = add(y, this.bias);
    }
    return y;
  }

  __call__(x1: MLXArray, x2: MLXArray): MLXArray {
    return this.forward(x1, x2);
  }
}

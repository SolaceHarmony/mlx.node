/**
 * Normalization layers: LayerNorm, RMSNorm, GroupNorm, BatchNorm, InstanceNorm.
 *
 * Mirrors mlx.nn.layers.normalization from the Python MLX API.
 */
import {
  reshape,
  add,
  multiply,
  subtract,
  mean,
  variance,
  rsqrt,
} from '../../core/ops';
import MLXArray, { ones, zeros } from '../../core/array';
import { Module } from './base';

/**
 * Applies Layer Normalization over a mini-batch of inputs.
 *
 * Computes: y = (x - mean) / sqrt(var + eps) * weight + bias
 *
 * @param dims - Feature dimension to normalize
 * @param eps - Small constant for numerical stability (default 1e-5)
 * @param affine - Whether to include learnable weight and bias (default true)
 * @param bias - Whether to include bias (only used if affine is true, default true)
 */
export class LayerNorm extends Module {
  weight?: MLXArray;
  bias?: MLXArray;
  eps: number;
  dims: number;

  constructor(
    dims: number,
    eps: number = 1e-5,
    affine: boolean = true,
    bias: boolean = true,
  ) {
    super();
    this.eps = eps;
    this.dims = dims;
    if (affine) {
      this.weight = ones([dims]);
      if (bias) {
        this.bias = zeros([dims]);
      }
    }
  }

  forward(x: MLXArray): MLXArray {
    // Normalize over last axis
    const stats = [mean(x, -1, { keepdims: true }), variance(x, -1, { keepdims: true })];
    const mu = stats[0];
    const v = stats[1];
    
    let y = multiply(subtract(x, mu), rsqrt(add(v, this.eps)));
    if (this.weight) {
      y = multiply(y, this.weight);
    }
    if (this.bias) {
      y = add(y, this.bias);
    }
    return y;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies Root Mean Square Normalization.
 *
 * Computes: y = x / sqrt(mean(x^2) + eps) * weight
 *
 * @param dims - Feature dimension
 * @param eps - Numerical stability (default 1e-5)
 */
export class RMSNorm extends Module {
  weight: MLXArray;
  eps: number;

  constructor(dims: number, eps: number = 1e-5) {
    super();
    this.weight = ones([dims]);
    this.eps = eps;
  }

  forward(x: MLXArray): MLXArray {
    const v = mean(multiply(x, x), -1, { keepdims: true });
    return multiply(multiply(x, rsqrt(add(v, this.eps))), this.weight);
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies Group Normalization.
 *
 * @param numGroups - Number of groups to separate channels into
 * @param dims - Number of channels
 * @param eps - Stability constant (default 1e-5)
 * @param affine - Learnable per-channel scale/bias (default true)
 */
export class GroupNorm extends Module {
  weight?: MLXArray;
  bias?: MLXArray;
  numGroups: number;
  dims: number;
  eps: number;

  constructor(
    numGroups: number,
    dims: number,
    eps: number = 1e-5,
    affine: boolean = true,
  ) {
    super();
    if (dims % numGroups !== 0) {
      throw new Error(`dims (${dims}) must be divisible by numGroups (${numGroups})`);
    }
    this.numGroups = numGroups;
    this.dims = dims;
    this.eps = eps;
    if (affine) {
      this.weight = ones([dims]);
      this.bias = zeros([dims]);
    }
  }

  forward(x: MLXArray): MLXArray {
    const shape = x.shape;
    const batchDims = shape.slice(0, -1);
    const C = shape[shape.length - 1];

    // Reshape to (..., groups, dims/groups)
    const groupedShape = [...batchDims, this.numGroups, C / this.numGroups];
    let y = reshape(x, groupedShape);

    // Compute stats over last axis (dims/groups)
    const mu = mean(y, -1, { keepdims: true });
    const v = variance(y, -1, { keepdims: true });

    y = multiply(subtract(y, mu), rsqrt(add(v, this.eps)));
    y = reshape(y, shape);

    if (this.weight) y = multiply(y, this.weight);
    if (this.bias) y = add(y, this.bias);

    return y;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies Batch Normalization.
 *
 * @param numFeatures - Number of features (channels)
 * @param eps - Stability (default 1e-5)
 * @param momentum - Momentum for running stats (default 0.1)
 * @param affine - Learnable weight/bias (default true)
 * @param trackRunningStats - Whether to keep track of moving mean/var (default true)
 */
export class BatchNorm extends Module {
  weight?: MLXArray;
  bias?: MLXArray;
  runningMean?: MLXArray;
  runningVar?: MLXArray;
  eps: number;
  momentum: number;
  trackRunningStats: boolean;

  constructor(
    numFeatures: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    trackRunningStats: boolean = true,
  ) {
    super();
    this.eps = eps;
    this.momentum = momentum;
    this.trackRunningStats = trackRunningStats;

    if (affine) {
      this.weight = ones([numFeatures]);
      this.bias = zeros([numFeatures]);
    }

    if (trackRunningStats) {
      this.runningMean = zeros([numFeatures]);
      this.runningVar = ones([numFeatures]);
      // Freeze running stats: they are not trained via backprop
      this.freeze(false, ['runningMean', 'runningVar']);
    }
  }

  forward(x: MLXArray): MLXArray {
    // In MLX, reduction happens over batch and spatial dims [0, 1, 2]
    // for NHWC input.
    if (this.training || !this.trackRunningStats) {
      const axes = Array.from({ length: x.shape.length - 1 }, (_, i) => i);
      const mu = mean(x, axes, { keepdims: true });
      const v = variance(x, axes, { keepdims: true });

      if (this.training && this.trackRunningStats) {
        // Update running stats (manual update as Module doesn't support
        // stateful updates during forward pass yet in JS)
        // Note: this is a simplification.
      }

      let y = multiply(subtract(x, mu), rsqrt(add(v, this.eps)));
      if (this.weight) y = multiply(y, this.weight);
      if (this.bias) y = add(y, this.bias);
      return y;
    } else {
      // Use running stats
      let y = multiply(
        subtract(x, this.runningMean!),
        rsqrt(add(this.runningVar!, this.eps)),
      );
      if (this.weight) y = multiply(y, this.weight);
      if (this.bias) y = add(y, this.bias);
      return y;
    }
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies Instance Normalization.
 *
 * @param numFeatures - Number of channels
 * @param eps - Stability (default 1e-5)
 * @param affine - Learnable weight/bias (default false)
 */
export class InstanceNorm extends Module {
  weight?: MLXArray;
  bias?: MLXArray;
  eps: number;

  constructor(numFeatures: number, eps: number = 1e-5, affine: boolean = false) {
    super();
    this.eps = eps;
    if (affine) {
      this.weight = ones([numFeatures]);
      this.bias = zeros([numFeatures]);
    }
  }

  forward(x: MLXArray): MLXArray {
    // Spatial dims are all except first (batch) and last (channel)
    const axes = Array.from({ length: x.shape.length - 2 }, (_, i) => i + 1);
    const mu = mean(x, axes, { keepdims: true });
    const v = variance(x, axes, { keepdims: true });

    let y = multiply(subtract(x, mu), rsqrt(add(v, this.eps)));
    if (this.weight) y = multiply(y, this.weight);
    if (this.bias) y = add(y, this.bias);
    return y;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

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
export class LayerNorm {
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
    const axis = -1;
    const mu = mean(x, axis, { keepdims: true });
    const v = variance(x, axis, { keepdims: true });
    let y = multiply(subtract(x, mu), rsqrt(add(v, this.eps)));
    if (this.weight !== undefined) {
      y = multiply(this.weight, y);
      if (this.bias !== undefined) {
        y = add(y, this.bias);
      }
    }
    return y;
  }
}

/**
 * Applies Root Mean Square normalization.
 *
 * Computes: y = x / sqrt(mean(x^2) + eps) * weight
 *
 * @param dims - Feature dimension
 * @param eps - Small constant for numerical stability (default 1e-5)
 */
export class RMSNorm {
  weight: MLXArray;
  eps: number;

  constructor(dims: number, eps: number = 1e-5) {
    this.weight = ones([dims]);
    this.eps = eps;
  }

  forward(x: MLXArray): MLXArray {
    // RMS = sqrt(mean(x^2))
    // y = x * rsqrt(mean(x^2) + eps) * weight
    const x2 = multiply(x, x);
    const ms = mean(x2, -1, { keepdims: true });
    const y = multiply(x, rsqrt(add(ms, this.eps)));
    return multiply(this.weight, y);
  }
}

/**
 * Applies Group Normalization over a mini-batch of inputs.
 *
 * @param numGroups - Number of groups to separate channels into
 * @param dims - Total number of channels
 * @param eps - Small constant for numerical stability (default 1e-5)
 * @param affine - Whether to include learnable weight and bias (default true)
 */
export class GroupNorm {
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
    this.numGroups = numGroups;
    this.dims = dims;
    this.eps = eps;
    if (affine) {
      this.weight = ones([dims]);
      this.bias = zeros([dims]);
    }
  }

  forward(x: MLXArray): MLXArray {
    const batch = x.shape[0];
    const rest = x.shape.slice(1, -1);
    const dims = x.shape[x.shape.length - 1];

    // Reshape: (batch, *rest, dims) -> (batch, -1, num_groups)
    let y = reshape(x, [batch, -1, this.numGroups]);

    // Normalize along spatial axis (axis=1)
    const mu = mean(y, 1, { keepdims: true });
    const v = variance(y, 1, { keepdims: true });
    y = multiply(subtract(y, mu), rsqrt(add(v, this.eps)));

    // Reshape back: (batch, -1, num_groups) -> (batch, *rest, dims)
    y = reshape(y, [batch, ...rest, dims]);

    if (this.weight !== undefined) {
      y = add(multiply(this.weight, y), this.bias!);
    }
    return y;
  }
}

/**
 * Applies Batch Normalization over a 2D/3D/4D input.
 *
 * In training mode, uses batch statistics and updates running stats.
 * In eval mode, uses running statistics (if tracked).
 *
 * @param numFeatures - Number of features (channels)
 * @param eps - Small constant for numerical stability (default 1e-5)
 * @param momentum - EMA coefficient for running stats (default 0.1)
 * @param affine - Whether to include learnable weight and bias (default true)
 * @param trackRunningStats - Whether to track running mean/var (default true)
 */
export class BatchNorm {
  weight?: MLXArray;
  bias?: MLXArray;
  runningMean?: MLXArray;
  runningVar?: MLXArray;
  numFeatures: number;
  eps: number;
  momentum: number;
  training: boolean = true;
  trackRunningStats: boolean;

  constructor(
    numFeatures: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    trackRunningStats: boolean = true,
  ) {
    this.numFeatures = numFeatures;
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
    }
  }

  forward(x: MLXArray): MLXArray {
    if (x.shape.length < 2 || x.shape.length > 4) {
      throw new Error(
        `Expected input tensor to have 2, 3 or 4 dimensions, but got ${x.shape.length}`,
      );
    }

    // Reduce over all axes except the last (feature) axis
    const reductionAxes: number[] = [];
    for (let i = 0; i < x.shape.length - 1; i++) {
      reductionAxes.push(i);
    }

    let mu = mean(x, reductionAxes);
    let v = variance(x, reductionAxes);

    if (this.training && this.trackRunningStats) {
      const m = this.momentum;
      this.runningMean = add(
        multiply(1 - m, this.runningMean!),
        multiply(m, mu),
      );
      this.runningVar = add(
        multiply(1 - m, this.runningVar!),
        multiply(m, v),
      );
    } else if (this.trackRunningStats) {
      mu = this.runningMean!;
      v = this.runningVar!;
    }

    let y = multiply(subtract(x, mu), rsqrt(add(v, this.eps)));
    if (this.weight !== undefined) {
      y = add(multiply(this.weight, y), this.bias!);
    }
    return y;
  }
}

/**
 * Applies Instance Normalization over each channel independently.
 *
 * Normalizes over spatial dimensions (axes 1 through ndim-2).
 *
 * @param dims - Number of features (channels)
 * @param eps - Small constant for numerical stability (default 1e-5)
 * @param affine - Whether to include learnable weight and bias (default false)
 */
export class InstanceNorm {
  weight?: MLXArray;
  bias?: MLXArray;
  dims: number;
  eps: number;

  constructor(dims: number, eps: number = 1e-5, affine: boolean = false) {
    this.dims = dims;
    this.eps = eps;
    if (affine) {
      this.weight = ones([dims]);
      this.bias = zeros([dims]);
    }
  }

  forward(x: MLXArray): MLXArray {
    // Reduce over spatial axes (1 through ndim-2)
    const reductionAxes: number[] = [];
    for (let i = 1; i < x.shape.length - 1; i++) {
      reductionAxes.push(i);
    }

    const mu = mean(x, reductionAxes, { keepdims: true });
    const v = variance(x, reductionAxes, { keepdims: true });
    let y = multiply(subtract(x, mu), rsqrt(add(v, this.eps)));

    if (this.weight !== undefined) {
      y = add(multiply(this.weight, y), this.bias!);
    }
    return y;
  }
}

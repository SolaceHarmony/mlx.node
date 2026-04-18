/**
 * Convolution layers: Conv1d, Conv2d, Conv3d.
 *
 * Mirrors mlx.nn.layers.convolution from the Python MLX API.
 */
import {
  add,
  random,
  conv1d as coreConv1d,
  conv2d as coreConv2d,
  conv3d as coreConv3d,
} from '../../core/ops';
import MLXArray from '../../core/array';
import { Module } from './base';

/**
 * Applies a 1-dimensional convolution over a multi-channel input sequence.
 *
 * Input shape: NLC (batch, length, channels).
 *
 * @param inChannels - Number of input channels
 * @param outChannels - Number of output channels
 * @param kernelSize - Size of the convolution filter
 * @param stride - Stride (default 1)
 * @param padding - Padding (default 0)
 * @param dilation - Dilation (default 1)
 * @param groups - Number of groups (default 1)
 * @param bias - Whether to include a bias (default true)
 */
export class Conv1d extends Module {
  weight: MLXArray;
  bias?: MLXArray;
  padding: number;
  dilation: number;
  stride: number;
  groups: number;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    stride: number = 1,
    padding: number = 0,
    dilation: number = 1,
    groups: number = 1,
    bias: boolean = true,
  ) {
    super();
    if (inChannels % groups !== 0) {
      throw new Error(
        `The number of input channels (${inChannels}) must be divisible by the number of groups (${groups})`,
      );
    }

    const scale = Math.sqrt(1.0 / (inChannels * kernelSize));
    this.weight = random.uniform(-scale, scale, [
      outChannels,
      kernelSize,
      inChannels / groups,
    ]);
    if (bias) {
      this.bias = random.uniform(-scale, scale, [outChannels]);
    }

    this.padding = padding;
    this.stride = stride;
    this.dilation = dilation;
    this.groups = groups;
  }

  forward(x: MLXArray): MLXArray {
    let y = coreConv1d(x, this.weight, this.stride, this.padding, this.dilation, this.groups);
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
 * Applies a 2-dimensional convolution over a multi-channel input image.
 *
 * Input shape: NHWC (batch, height, width, channels).
 *
 * @param inChannels - Number of input channels
 * @param outChannels - Number of output channels
 * @param kernelSize - Size of the convolution filter ([height, width] or single number)
 * @param stride - Stride (default 1)
 * @param padding - Padding (default 0)
 * @param dilation - Dilation (default 1)
 * @param groups - Number of groups (default 1)
 * @param bias - Whether to include a bias (default true)
 */
export class Conv2d extends Module {
  weight: MLXArray;
  bias?: MLXArray;
  padding: [number, number];
  dilation: [number, number];
  stride: [number, number];
  groups: number;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    stride: number | [number, number] = 1,
    padding: number | [number, number] = 0,
    dilation: number | [number, number] = 1,
    groups: number = 1,
    bias: boolean = true,
  ) {
    super();
    if (inChannels % groups !== 0) {
      throw new Error(
        `The number of input channels (${inChannels}) must be divisible by the number of groups (${groups})`,
      );
    }

    const k = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    const s = Array.isArray(stride) ? stride : [stride, stride];
    const p = Array.isArray(padding) ? padding : [padding, padding];
    const d = Array.isArray(dilation) ? dilation : [dilation, dilation];

    const scale = Math.sqrt(1.0 / (inChannels * k[0] * k[1]));
    this.weight = random.uniform(-scale, scale, [
      outChannels,
      k[0],
      k[1],
      inChannels / groups,
    ]);
    if (bias) {
      this.bias = random.uniform(-scale, scale, [outChannels]);
    }

    this.padding = p as any;
    this.stride = s as any;
    this.dilation = d as any;
    this.groups = groups;
  }

  forward(x: MLXArray): MLXArray {
    let y = coreConv2d(x, this.weight, this.stride, this.padding, this.dilation, this.groups);
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
 * Applies a 3-dimensional convolution over a multi-channel input volume.
 *
 * Input shape: NDHWC (batch, depth, height, width, channels).
 *
 * @param inChannels - Number of input channels
 * @param outChannels - Number of output channels
 * @param kernelSize - Size of the convolution filter ([depth, height, width] or single number)
 * @param stride - Stride (default 1)
 * @param padding - Padding (default 0)
 * @param dilation - Dilation (default 1)
 * @param groups - Number of groups (default 1)
 * @param bias - Whether to include a bias (default true)
 */
export class Conv3d extends Module {
  weight: MLXArray;
  bias?: MLXArray;
  padding: [number, number, number];
  dilation: [number, number, number];
  stride: [number, number, number];
  groups: number;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number, number],
    stride: number | [number, number, number] = 1,
    padding: number | [number, number, number] = 0,
    dilation: number | [number, number, number] = 1,
    groups: number = 1,
    bias: boolean = true,
  ) {
    super();
    if (inChannels % groups !== 0) {
      throw new Error(
        `The number of input channels (${inChannels}) must be divisible by the number of groups (${groups})`,
      );
    }

    const k = Array.isArray(kernelSize)
      ? kernelSize
      : [kernelSize, kernelSize, kernelSize];
    const s = Array.isArray(stride) ? stride : [stride, stride, stride];
    const p = Array.isArray(padding) ? padding : [padding, padding, padding];
    const d = Array.isArray(dilation)
      ? dilation
      : [dilation, dilation, dilation];

    const scale = Math.sqrt(1.0 / (inChannels * k[0] * k[1] * k[2]));
    this.weight = random.uniform(-scale, scale, [
      outChannels,
      k[0],
      k[1],
      k[2],
      inChannels / groups,
    ]);
    if (bias) {
      this.bias = random.uniform(-scale, scale, [outChannels]);
    }

    this.padding = p as any;
    this.stride = s as any;
    this.dilation = d as any;
    this.groups = groups;
  }

  forward(x: MLXArray): MLXArray {
    let y = coreConv3d(x, this.weight, this.stride, this.padding, this.dilation, this.groups);
    if (this.bias) {
      y = add(y, this.bias);
    }
    return y;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

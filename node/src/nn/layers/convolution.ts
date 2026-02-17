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
import MLXArray, { zeros } from '../../core/array';

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
export class Conv1d {
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
    if (inChannels % groups !== 0) {
      throw new Error(
        `The number of input channels (${inChannels}) must be divisible by the number of groups (${groups})`,
      );
    }

    const scale = Math.sqrt(1 / (inChannels * kernelSize));
    this.weight = random.uniform(
      -scale, scale,
      [outChannels, kernelSize, Math.floor(inChannels / groups)],
    );
    if (bias) {
      this.bias = zeros([outChannels]);
    }
    this.padding = padding;
    this.dilation = dilation;
    this.stride = stride;
    this.groups = groups;
  }

  forward(x: MLXArray): MLXArray {
    let y = coreConv1d(x, this.weight, this.stride, this.padding, this.dilation, this.groups);
    if (this.bias !== undefined) {
      y = add(y, this.bias);
    }
    return y;
  }
}

/**
 * Applies a 2-dimensional convolution over a multi-channel input image.
 *
 * Input shape: NHWC (batch, height, width, channels).
 *
 * @param inChannels - Number of input channels
 * @param outChannels - Number of output channels
 * @param kernelSize - Kernel size (int or [h, w])
 * @param stride - Stride (default 1)
 * @param padding - Padding (default 0)
 * @param dilation - Dilation (default 1)
 * @param groups - Number of groups (default 1)
 * @param bias - Whether to include a bias (default true)
 */
export class Conv2d {
  weight: MLXArray;
  bias?: MLXArray;
  padding: [number, number];
  stride: [number, number];
  dilation: [number, number];
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
    if (inChannels % groups !== 0) {
      throw new Error(
        `The number of input channels (${inChannels}) must be divisible by the number of groups (${groups})`,
      );
    }

    const ks: [number, number] = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    this.stride = typeof stride === 'number' ? [stride, stride] : stride;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;
    this.dilation = typeof dilation === 'number' ? [dilation, dilation] : dilation;
    this.groups = groups;

    const scale = Math.sqrt(1 / (inChannels * ks[0] * ks[1]));
    this.weight = random.uniform(
      -scale, scale,
      [outChannels, ks[0], ks[1], Math.floor(inChannels / groups)],
    );
    if (bias) {
      this.bias = zeros([outChannels]);
    }
  }

  forward(x: MLXArray): MLXArray {
    let y = coreConv2d(x, this.weight, this.stride, this.padding, this.dilation, this.groups);
    if (this.bias !== undefined) {
      y = add(y, this.bias);
    }
    return y;
  }
}

/**
 * Applies a 3-dimensional convolution over a multi-channel input volume.
 *
 * Input shape: NDHWC (batch, depth, height, width, channels).
 *
 * @param inChannels - Number of input channels
 * @param outChannels - Number of output channels
 * @param kernelSize - Kernel size (int or [d, h, w])
 * @param stride - Stride (default 1)
 * @param padding - Padding (default 0)
 * @param dilation - Dilation (default 1)
 * @param bias - Whether to include a bias (default true)
 */
export class Conv3d {
  weight: MLXArray;
  bias?: MLXArray;
  padding: [number, number, number];
  stride: [number, number, number];
  dilation: [number, number, number];

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number, number],
    stride: number | [number, number, number] = 1,
    padding: number | [number, number, number] = 0,
    dilation: number | [number, number, number] = 1,
    bias: boolean = true,
  ) {
    const ks: [number, number, number] = typeof kernelSize === 'number'
      ? [kernelSize, kernelSize, kernelSize] : kernelSize;
    this.stride = typeof stride === 'number' ? [stride, stride, stride] : stride;
    this.padding = typeof padding === 'number' ? [padding, padding, padding] : padding;
    this.dilation = typeof dilation === 'number' ? [dilation, dilation, dilation] : dilation;

    const scale = Math.sqrt(1 / (inChannels * ks[0] * ks[1] * ks[2]));
    this.weight = random.uniform(
      -scale, scale,
      [outChannels, ks[0], ks[1], ks[2], inChannels],
    );
    if (bias) {
      this.bias = zeros([outChannels]);
    }
  }

  forward(x: MLXArray): MLXArray {
    let y = coreConv3d(x, this.weight, this.stride, this.padding, this.dilation);
    if (this.bias !== undefined) {
      y = add(y, this.bias);
    }
    return y;
  }
}

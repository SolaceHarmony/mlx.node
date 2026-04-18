/**
 * Pooling layers: MaxPool1d/2d/3d, AvgPool1d/2d/3d.
 *
 * Mirrors mlx.nn.layers.pooling from the Python MLX API.
 */
import {
  reshape,
  squeeze,
  pad,
  slice,
  mean,
  as_strided,
  maximum,
} from '../../core/ops';
import MLXArray from '../../core/array';
import { Module } from './base';

/**
 * Helper: Sliding window view using as_strided.
 * Given input of shape (..., spatial..., C), creates a windowed view for pooling.
 */
function slidingWindows(
  x: MLXArray,
  windowShape: number[],
  windowStrides: number[],
  spatialDims: number,
): MLXArray {
  const shape = x.shape;
  const ndim = shape.length;
  const batchDims = ndim - spatialDims - 1; // everything before spatial
  const C = shape[ndim - 1]; // channels last

  // Compute output spatial sizes
  const outSpatial: number[] = [];
  for (let i = 0; i < spatialDims; i++) {
    const dim = shape[batchDims + i];
    outSpatial.push(Math.floor((dim - windowShape[i]) / windowStrides[i]) + 1);
  }

  // Build strided shape: (...batch, outSpatial..., windowShape..., C)
  const outShape: number[] = [];
  for (let i = 0; i < batchDims; i++) outShape.push(shape[i]);
  for (const s of outSpatial) outShape.push(s);
  for (const w of windowShape) outShape.push(w);
  outShape.push(C);

  // Compute element strides from shape (assuming contiguous row-major)
  const elemStrides: number[] = new Array(ndim);
  elemStrides[ndim - 1] = 1;
  for (let i = ndim - 2; i >= 0; i--) {
    elemStrides[i] = elemStrides[i + 1] * shape[i + 1];
  }

  // Build output strides
  const outStrides: number[] = [];
  // Batch dims: same strides
  for (let i = 0; i < batchDims; i++) outStrides.push(elemStrides[i]);
  // Output spatial dims: strides * windowStrides
  for (let i = 0; i < spatialDims; i++) {
    outStrides.push(elemStrides[batchDims + i] * windowStrides[i]);
  }
  // Window dims: original spatial strides
  for (let i = 0; i < spatialDims; i++) {
    outStrides.push(elemStrides[batchDims + i]);
  }
  // Channel stride
  outStrides.push(1);

  // Ensure all values are numbers before passing to as_strided
  const finalShape = outShape.map(v => isNaN(v) ? 0 : v);
  const finalStrides = outStrides.map(v => isNaN(v) ? 0 : v);

  return as_strided(x, finalShape, finalStrides, 0);
}

/** Helper to select index i along the second-to-last axis */
function selectAlongPenultimate(x: MLXArray, idx: number): MLXArray {
  const ndim = x.shape.length;
  const start = new Array(ndim).fill(0);
  const stop = [...x.shape];
  start[ndim - 2] = idx;
  stop[ndim - 2] = idx + 1;
  let y = slice(x, start, stop);
  return squeeze(y, ndim - 2);
}

/** Helper to select index i along axis 3 of a 5D tensor */
function sliceAxis3(x: MLXArray, idx: number): MLXArray {
  const s = x.shape;
  const start = [0, 0, 0, idx, 0];
  const stop = [s[0], s[1], s[2], idx + 1, s[4]];
  return squeeze(slice(x, start, stop), 3);
}

/** Helper to select index i along axis 4 of a 6D tensor */
function sliceAxis4(x: MLXArray, idx: number): MLXArray {
  const s = x.shape;
  const start = [0, 0, 0, 0, idx, 0];
  const stop = [s[0], s[1], s[2], s[3], idx + 1, s[5]];
  return squeeze(slice(x, start, stop), 4);
}

/**
 * Applies 1-dimensional max pooling.
 *
 * Input shape: NLC. Kernel applied over L dimension.
 *
 * @param kernelSize - Pooling window size
 * @param stride - Pooling stride (default: kernelSize)
 * @param padding - Padding (default: 0)
 */
export class MaxPool1d extends Module {
  kernelSize: number;
  stride: number;
  padding: number;

  constructor(kernelSize: number, stride?: number, padding: number = 0) {
    super();
    this.kernelSize = kernelSize;
    this.stride = stride ?? kernelSize;
    this.padding = padding;
  }

  forward(x: MLXArray): MLXArray {
    if (this.padding > 0) {
      const p = this.padding;
      // Pad: [(0,0) for batch, (p,p) for spatial, (0,0) for channels]
      x = pad(x, [[0, 0], [p, p], [0, 0]], -Infinity);
    }
    // Create sliding windows then reduce
    const windows = slidingWindows(x, [this.kernelSize], [this.stride], 1);
    // windows shape: (N, outL, kernelSize, C) — reduce over kernelSize dim
    const shape = windows.shape;
    const flat = reshape(windows, [shape[0], shape[1], this.kernelSize, shape[shape.length - 1]]);
    // Max reduce over axis=2 via element-wise maximum
    let result = selectAlongPenultimate(flat, 0);
    for (let i = 1; i < this.kernelSize; i++) {
      result = maximum(result, selectAlongPenultimate(flat, i));
    }
    return result;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies 1-dimensional average pooling.
 *
 * Input shape: NLC. Kernel applied over L dimension.
 *
 * @param kernelSize - Pooling window size
 * @param stride - Pooling stride (default: kernelSize)
 * @param padding - Padding (default: 0)
 */
export class AvgPool1d extends Module {
  kernelSize: number;
  stride: number;
  padding: number;

  constructor(kernelSize: number, stride?: number, padding: number = 0) {
    super();
    this.kernelSize = kernelSize;
    this.stride = stride ?? kernelSize;
    this.padding = padding;
  }

  forward(x: MLXArray): MLXArray {
    if (this.padding > 0) {
      const p = this.padding;
      x = pad(x, [[0, 0], [p, p], [0, 0]], 0);
    }
    // Create sliding windows then reduce
    const windows = slidingWindows(x, [this.kernelSize], [this.stride], 1);
    // windows shape: (N, outL, kernelSize, C) — mean over kernelSize dim
    const shape = windows.shape;
    const flat = reshape(windows, [shape[0], shape[1], this.kernelSize, shape[shape.length - 1]]);
    return mean(flat, 2);
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies 2-dimensional max pooling.
 *
 * Input shape: NHWC. Kernel applied over H, W dimensions.
 *
 * @param kernelSize - Pooling window size (int or [h, w])
 * @param stride - Pooling stride (default: kernelSize)
 * @param padding - Padding (default: 0)
 */
export class MaxPool2d extends Module {
  kernelSize: [number, number];
  stride: [number, number];
  padding: [number, number];

  constructor(
    kernelSize: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0,
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    const s = stride ?? kernelSize;
    this.stride = typeof s === 'number' ? [s, s] : s;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;
  }

  forward(x: MLXArray): MLXArray {
    if (this.padding[0] > 0 || this.padding[1] > 0) {
      const [ph, pw] = this.padding;
      // Pad: [(0,0) batch, (ph,ph) H, (pw,pw) W, (0,0) channels]
      x = pad(x, [[0, 0], [ph, ph], [pw, pw], [0, 0]], -Infinity);
    }
    const windows = slidingWindows(x, [...this.kernelSize], [...this.stride], 2);
    // windows shape: (N, outH, outW, kH, kW, C)
    const s = windows.shape;
    // Flatten window dims: (N, outH, outW, kH*kW, C)
    const flat = reshape(windows, [s[0], s[1], s[2], s[3] * s[4], s[5]]);
    // Max over axis=3
    let result = sliceAxis3(flat, 0);
    for (let i = 1; i < s[3] * s[4]; i++) {
      result = maximum(result, sliceAxis3(flat, i));
    }
    return result;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies 2-dimensional average pooling.
 *
 * Input shape: NHWC.
 *
 * @param kernelSize - Pooling window size (int or [h, w])
 * @param stride - Pooling stride (default: kernelSize)
 * @param padding - Padding (default: 0)
 */
export class AvgPool2d extends Module {
  kernelSize: [number, number];
  stride: [number, number];
  padding: [number, number];

  constructor(
    kernelSize: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0,
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    const s = stride ?? kernelSize;
    this.stride = typeof s === 'number' ? [s, s] : s;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;
  }

  forward(x: MLXArray): MLXArray {
    if (this.padding[0] > 0 || this.padding[1] > 0) {
      const [ph, pw] = this.padding;
      x = pad(x, [[0, 0], [ph, ph], [pw, pw], [0, 0]], 0);
    }
    const windows = slidingWindows(x, [...this.kernelSize], [...this.stride], 2);
    const s = windows.shape;
    const flat = reshape(windows, [s[0], s[1], s[2], s[3] * s[4], s[5]]);
    return mean(flat, 3);
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies 3-dimensional max pooling.
 *
 * Input shape: NDHWC. Kernel applied over D, H, W dimensions.
 *
 * @param kernelSize - Pooling window size (int or [d, h, w])
 * @param stride - Pooling stride (default: kernelSize)
 * @param padding - Padding (default: 0)
 */
export class MaxPool3d extends Module {
  kernelSize: [number, number, number];
  stride: [number, number, number];
  padding: [number, number, number];

  constructor(
    kernelSize: number | [number, number, number],
    stride?: number | [number, number, number],
    padding: number | [number, number, number] = 0,
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize, kernelSize] : kernelSize;
    const s = stride ?? kernelSize;
    this.stride = typeof s === 'number' ? [s, s, s] : s;
    this.padding = typeof padding === 'number' ? [padding, padding, padding] : padding;
  }

  forward(x: MLXArray): MLXArray {
    if (this.padding[0] > 0 || this.padding[1] > 0 || this.padding[2] > 0) {
      const [pd, ph, pw] = this.padding;
      // Pad: [(0,0) batch, (pd,pd) D, (ph,ph) H, (pw,pw) W, (0,0) channels]
      x = pad(x, [[0, 0], [pd, pd], [ph, ph], [pw, pw], [0, 0]], -Infinity);
    }
    const windows = slidingWindows(x, [...this.kernelSize], [...this.stride], 3);
    // windows shape: (N, outD, outH, outW, kD, kH, kW, C)
    const s = windows.shape;
    const winSize = s[4] * s[5] * s[6];
    // Flatten window dims: (N, outD, outH, outW, kD*kH*kW, C)
    const flat = reshape(windows, [s[0], s[1], s[2], s[3], winSize, s[7]]);
    // Max over axis=4
    let result = sliceAxis4(flat, 0);
    for (let i = 1; i < winSize; i++) {
      result = maximum(result, sliceAxis4(flat, i));
    }
    return result;
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Applies 3-dimensional average pooling.
 *
 * Input shape: NDHWC.
 *
 * @param kernelSize - Pooling window size (int or [d, h, w])
 * @param stride - Pooling stride (default: kernelSize)
 * @param padding - Padding (default: 0)
 */
export class AvgPool3d extends Module {
  kernelSize: [number, number, number];
  stride: [number, number, number];
  padding: [number, number, number];

  constructor(
    kernelSize: number | [number, number, number],
    stride?: number | [number, number, number],
    padding: number | [number, number, number] = 0,
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize, kernelSize] : kernelSize;
    const s = stride ?? kernelSize;
    this.stride = typeof s === 'number' ? [s, s, s] : s;
    this.padding = typeof padding === 'number' ? [padding, padding, padding] : padding;
  }

  forward(x: MLXArray): MLXArray {
    if (this.padding[0] > 0 || this.padding[1] > 0 || this.padding[2] > 0) {
      const [pd, ph, pw] = this.padding;
      x = pad(x, [[0, 0], [pd, pd], [ph, ph], [pw, pw], [0, 0]], 0);
    }
    const windows = slidingWindows(x, [...this.kernelSize], [...this.stride], 3);
    const s = windows.shape;
    const winSize = s[4] * s[5] * s[6];
    const flat = reshape(windows, [s[0], s[1], s[2], s[3], winSize, s[7]]);
    return mean(flat, 4);
  }

  __call__(x: MLXArray): MLXArray {
    return this.forward(x);
  }
}

/**
 * Recurrent layers: RNN, GRU, LSTM.
 *
 * Mirrors mlx.nn.layers.recurrent from the Python MLX API.
 */
import {
  transpose,
  matmul,
  add,
  multiply,
  subtract,
  addmm,
  split,
  stack,
  slice,
  squeeze,
  random,
  sigmoid as coreSigmoid,
  tanh as coreTanh,
} from '../../core/ops';
import MLXArray from '../../core/array';

/**
 * Select a single index along the second-to-last axis.
 * Equivalent to Python `x[..., idx, :]`.
 *
 * For a tensor of shape (..., L, D) returns shape (..., D).
 */
function selectAlongPenultimate(x: MLXArray, idx: number): MLXArray {
  const ndim = x.shape.length;
  const start = new Array(ndim).fill(0);
  const stop = [...x.shape];
  start[ndim - 2] = idx;
  stop[ndim - 2] = idx + 1;
  let y = slice(x, start, stop);
  return squeeze(y, ndim - 2);
}

/**
 * Select a contiguous range along the last axis.
 * Equivalent to Python `x[..., from:to]`.
 */
function sliceLastAxis(x: MLXArray, from: number, to: number): MLXArray {
  const ndim = x.shape.length;
  const start = new Array(ndim).fill(0);
  const stop = [...x.shape];
  start[ndim - 1] = from;
  stop[ndim - 1] = to;
  return slice(x, start, stop);
}

/**
 * An Elman recurrent layer.
 *
 * Input shape: NLD or LD. Returns hidden states of shape NLH or LH.
 *
 * h_{t+1} = nonlinearity(W_ih x_t + W_hh h_t + b)
 *
 * @param inputSize - Dimension of the input (D)
 * @param hiddenSize - Dimension of the hidden state (H)
 * @param bias - Whether to use a bias (default true)
 * @param nonlinearity - Activation function (default tanh)
 */
export class RNN {
  Wxh: MLXArray;
  Whh: MLXArray;
  bias?: MLXArray;
  hiddenSize: number;
  nonlinearity: (x: MLXArray) => MLXArray;

  constructor(
    inputSize: number,
    hiddenSize: number,
    bias: boolean = true,
    nonlinearity?: (x: MLXArray) => MLXArray,
  ) {
    this.nonlinearity = nonlinearity ?? coreTanh;
    this.hiddenSize = hiddenSize;
    const scale = 1 / Math.sqrt(hiddenSize);
    this.Wxh = random.uniform(-scale, scale, [hiddenSize, inputSize]);
    this.Whh = random.uniform(-scale, scale, [hiddenSize, hiddenSize]);
    if (bias) {
      this.bias = random.uniform(-scale, scale, [hiddenSize]);
    }
  }

  forward(x: MLXArray, hidden?: MLXArray): MLXArray {
    // Project all time steps at once: x @ Wxh^T + bias
    let projected: MLXArray;
    if (this.bias !== undefined) {
      projected = addmm(this.bias, x, transpose(this.Wxh));
    } else {
      projected = matmul(x, transpose(this.Wxh));
    }

    const seqLen = x.shape[x.shape.length - 2];
    const allHidden: MLXArray[] = [];

    for (let idx = 0; idx < seqLen; idx++) {
      const xt = selectAlongPenultimate(projected, idx);
      if (hidden !== undefined) {
        hidden = addmm(xt, hidden, transpose(this.Whh));
      } else {
        hidden = xt;
      }
      hidden = this.nonlinearity(hidden);
      allHidden.push(hidden);
    }

    return stack(allHidden, -2);
  }
}

/**
 * A gated recurrent unit (GRU) RNN layer.
 *
 * Input shape: NLD or LD. Returns hidden states of shape NLH or LH.
 *
 * @param inputSize - Dimension of the input (D)
 * @param hiddenSize - Dimension of the hidden state (H)
 * @param bias - Whether to use biases (default true)
 */
export class GRU {
  Wx: MLXArray;
  Wh: MLXArray;
  b?: MLXArray;
  bhn?: MLXArray;
  hiddenSize: number;

  constructor(
    inputSize: number,
    hiddenSize: number,
    bias: boolean = true,
  ) {
    this.hiddenSize = hiddenSize;
    const scale = 1 / Math.sqrt(hiddenSize);
    this.Wx = random.uniform(-scale, scale, [3 * hiddenSize, inputSize]);
    this.Wh = random.uniform(-scale, scale, [3 * hiddenSize, hiddenSize]);
    if (bias) {
      this.b = random.uniform(-scale, scale, [3 * hiddenSize]);
      this.bhn = random.uniform(-scale, scale, [hiddenSize]);
    }
  }

  forward(x: MLXArray, hidden?: MLXArray): MLXArray {
    // Project all time steps: x @ Wx^T + b
    let projected: MLXArray;
    if (this.b !== undefined) {
      projected = addmm(this.b, x, transpose(this.Wx));
    } else {
      projected = matmul(x, transpose(this.Wx));
    }

    // Split input projections into rz and n parts
    const H = this.hiddenSize;
    const xRz = sliceLastAxis(projected, 0, 2 * H);
    const xN = sliceLastAxis(projected, 2 * H, 3 * H);

    const seqLen = x.shape[x.shape.length - 2];
    const allHidden: MLXArray[] = [];

    for (let idx = 0; idx < seqLen; idx++) {
      let rz = selectAlongPenultimate(xRz, idx);
      let hProjN: MLXArray | undefined;

      if (hidden !== undefined) {
        const hProj = matmul(hidden, transpose(this.Wh));
        const hProjRz = sliceLastAxis(hProj, 0, 2 * H);
        hProjN = sliceLastAxis(hProj, 2 * H, 3 * H);
        if (this.bhn !== undefined) {
          hProjN = add(hProjN, this.bhn);
        }
        rz = add(rz, hProjRz);
      }

      rz = coreSigmoid(rz);
      const parts = split(rz, 2, -1);
      const r = parts[0];
      const z = parts[1];

      let n = selectAlongPenultimate(xN, idx);
      if (hidden !== undefined) {
        n = add(n, multiply(r, hProjN!));
      }
      n = coreTanh(n);

      if (hidden !== undefined) {
        hidden = add(multiply(subtract(1, z), n), multiply(z, hidden));
      } else {
        hidden = multiply(subtract(1, z), n);
      }

      allHidden.push(hidden);
    }

    return stack(allHidden, -2);
  }
}

/**
 * An LSTM recurrent layer.
 *
 * Input shape: NLD or LD. Returns [hidden, cell] each of shape NLH or LH.
 *
 * @param inputSize - Dimension of the input (D)
 * @param hiddenSize - Dimension of the hidden state (H)
 * @param bias - Whether to use biases (default true)
 */
export class LSTM {
  Wx: MLXArray;
  Wh: MLXArray;
  bias?: MLXArray;
  hiddenSize: number;

  constructor(
    inputSize: number,
    hiddenSize: number,
    bias: boolean = true,
  ) {
    this.hiddenSize = hiddenSize;
    const scale = 1 / Math.sqrt(hiddenSize);
    this.Wx = random.uniform(-scale, scale, [4 * hiddenSize, inputSize]);
    this.Wh = random.uniform(-scale, scale, [4 * hiddenSize, hiddenSize]);
    if (bias) {
      this.bias = random.uniform(-scale, scale, [4 * hiddenSize]);
    }
  }

  forward(
    x: MLXArray,
    hidden?: MLXArray,
    cell?: MLXArray,
  ): [MLXArray, MLXArray] {
    // Project all time steps: x @ Wx^T + bias
    let projected: MLXArray;
    if (this.bias !== undefined) {
      projected = addmm(this.bias, x, transpose(this.Wx));
    } else {
      projected = matmul(x, transpose(this.Wx));
    }

    const seqLen = x.shape[x.shape.length - 2];
    const allHidden: MLXArray[] = [];
    const allCell: MLXArray[] = [];

    for (let idx = 0; idx < seqLen; idx++) {
      let ifgo = selectAlongPenultimate(projected, idx);
      if (hidden !== undefined) {
        ifgo = addmm(ifgo, hidden, transpose(this.Wh));
      }

      const gates = split(ifgo, 4, -1);
      const i = coreSigmoid(gates[0]);
      const f = coreSigmoid(gates[1]);
      const g = coreTanh(gates[2]);
      const o = coreSigmoid(gates[3]);

      if (cell !== undefined) {
        cell = add(multiply(f, cell), multiply(i, g));
      } else {
        cell = multiply(i, g);
      }
      hidden = multiply(o, coreTanh(cell));

      allHidden.push(hidden);
      allCell.push(cell);
    }

    return [stack(allHidden, -2), stack(allCell, -2)];
  }
}

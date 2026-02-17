/**
 * Neural network activation functions.
 *
 * Functional activations compose core MLX ops (GPU-backed).
 * Module classes wrap functionals for use in nn.Module graphs.
 *
 * Mirrors mlx.nn.layers.activations from the Python MLX API.
 */

import MLXArray from '../../core/array';
import { full } from '../../core/array';
import {
  sigmoid as mx_sigmoid,
  tanh as mx_tanh,
  erf as mx_erf,
  split as mx_split,
  softmax as mx_softmax,
  logsumexp,
  logaddexp,
  exp,
  maximum,
  minimum,
  multiply,
  divide,
  add,
  subtract,
  where,
  negative,
  power,
  abs,
  sign,
  greater,
} from '../../core/ops';

// ───────────────────────── functional activations ─────────────────────────

export function sigmoid(x: MLXArray): MLXArray {
  return mx_sigmoid(x);
}

export function relu(x: MLXArray): MLXArray {
  return maximum(x, 0);
}

export function leaky_relu(x: MLXArray, negative_slope = 0.01): MLXArray {
  return maximum(multiply(negative_slope, x), x);
}

export function log_softmax(x: MLXArray, axis = -1): MLXArray {
  return subtract(x, logsumexp(x, axis, { keepdims: true }));
}

export function elu(x: MLXArray, alpha = 1.0): MLXArray {
  return where(greater(x, 0), x, multiply(alpha, subtract(exp(x), 1)));
}

export function relu6(x: MLXArray): MLXArray {
  return minimum(maximum(x, 0), 6.0);
}

export function softmax(x: MLXArray, axis = -1): MLXArray {
  return mx_softmax(x, axis);
}

export function softplus(x: MLXArray): MLXArray {
  return logaddexp(x, 0);
}

export function softsign(x: MLXArray): MLXArray {
  return divide(x, add(1, abs(x)));
}

export function softshrink(x: MLXArray, lambd = 0.5): MLXArray {
  return where(
    greater(abs(x), lambd),
    subtract(x, multiply(sign(x), lambd)),
    0,
  );
}

export function celu(x: MLXArray, alpha = 1.0): MLXArray {
  return add(
    maximum(x, 0.0),
    multiply(alpha, subtract(exp(divide(minimum(x, 0.0), alpha)), 1)),
  );
}

export function silu(x: MLXArray): MLXArray {
  return multiply(x, mx_sigmoid(x));
}

export function log_sigmoid(x: MLXArray): MLXArray {
  return negative(softplus(negative(x)));
}

export function gelu(x: MLXArray): MLXArray {
  const sqrt2 = Math.sqrt(2);
  return divide(multiply(x, add(1, mx_erf(divide(x, sqrt2)))), 2);
}

export function gelu_approx(x: MLXArray): MLXArray {
  const c = Math.sqrt(2 / Math.PI);
  return multiply(
    0.5,
    multiply(
      x,
      add(1, mx_tanh(multiply(c, add(x, multiply(0.044715, power(x, 3)))))),
    ),
  );
}

export function gelu_fast_approx(x: MLXArray): MLXArray {
  return multiply(x, mx_sigmoid(multiply(1.702, x)));
}

export function glu(x: MLXArray, axis = -1): MLXArray {
  const [a, b] = mx_split(x, 2, axis);
  return multiply(a, mx_sigmoid(b));
}

export function step(x: MLXArray, threshold = 0.0): MLXArray {
  return where(greater(x, threshold), 1, 0);
}

export function selu(x: MLXArray): MLXArray {
  return multiply(1.0507, elu(x, 1.67326));
}

export function prelu(x: MLXArray, alpha: MLXArray): MLXArray {
  return add(maximum(0, x), multiply(alpha, minimum(0, x)));
}

export function mish(x: MLXArray): MLXArray {
  return multiply(x, mx_tanh(softplus(x)));
}

export function hardswish(x: MLXArray): MLXArray {
  return divide(multiply(x, minimum(maximum(add(x, 3), 0), 6)), 6);
}

export function hard_tanh(x: MLXArray, min_val = -1.0, max_val = 1.0): MLXArray {
  return minimum(maximum(x, min_val), max_val);
}

export function hard_shrink(x: MLXArray, lambd = 0.5): MLXArray {
  return where(greater(abs(x), lambd), x, 0);
}

export function softmin(x: MLXArray, axis = -1): MLXArray {
  return mx_softmax(negative(x), axis);
}

export function tanh(x: MLXArray): MLXArray {
  return mx_tanh(x);
}

// ───────────────────────── Module classes ──────────────────────────────────

export class Sigmoid {
  __call__(x: MLXArray): MLXArray { return sigmoid(x); }
}

export class Mish {
  __call__(x: MLXArray): MLXArray { return mish(x); }
}

export class ReLU {
  __call__(x: MLXArray): MLXArray { return relu(x); }
}

export class LeakyReLU {
  private _negative_slope: number;
  constructor(negative_slope = 0.01) { this._negative_slope = negative_slope; }
  __call__(x: MLXArray): MLXArray { return leaky_relu(x, this._negative_slope); }
}

export class ELU {
  private _alpha: number;
  constructor(alpha = 1.0) { this._alpha = alpha; }
  __call__(x: MLXArray): MLXArray { return elu(x, this._alpha); }
}

export class ReLU6 {
  __call__(x: MLXArray): MLXArray { return relu6(x); }
}

export class Softmax {
  __call__(x: MLXArray): MLXArray { return softmax(x); }
}

export class Softplus {
  __call__(x: MLXArray): MLXArray { return softplus(x); }
}

export class Softsign {
  __call__(x: MLXArray): MLXArray { return softsign(x); }
}

export class Softshrink {
  private lambd: number;
  constructor(lambd = 0.5) { this.lambd = lambd; }
  __call__(x: MLXArray): MLXArray { return softshrink(x, this.lambd); }
}

export class CELU {
  private _alpha: number;
  constructor(alpha = 1.0) { this._alpha = alpha; }
  __call__(x: MLXArray): MLXArray { return celu(x, this._alpha); }
}

export class SiLU {
  __call__(x: MLXArray): MLXArray { return silu(x); }
}

export class LogSoftmax {
  __call__(x: MLXArray): MLXArray { return log_softmax(x); }
}

export class LogSigmoid {
  __call__(x: MLXArray): MLXArray { return log_sigmoid(x); }
}

export class PReLU {
  weight: MLXArray;
  constructor(num_parameters = 1, init = 0.25) {
    this.weight = full([num_parameters], init);
  }
  __call__(x: MLXArray): MLXArray { return prelu(x, this.weight); }
}

export class GELU {
  private _approx: string;
  constructor(approx = 'none') {
    const allowed = ['none', 'precise', 'tanh', 'fast'];
    if (!allowed.includes(approx)) {
      throw new Error(
        `The approximation should be in ${JSON.stringify(allowed)} but '${approx}' was given`,
      );
    }
    this._approx = approx;
  }
  __call__(x: MLXArray): MLXArray {
    if (this._approx === 'none') return gelu(x);
    if (this._approx === 'precise' || this._approx === 'tanh') return gelu_approx(x);
    return gelu_fast_approx(x);
  }
}

export class Tanh {
  __call__(x: MLXArray): MLXArray { return tanh(x); }
}

export class Hardswish {
  __call__(x: MLXArray): MLXArray { return hardswish(x); }
}

export class Step {
  private threshold: number;
  constructor(threshold = 0.0) { this.threshold = threshold; }
  __call__(x: MLXArray): MLXArray { return step(x, this.threshold); }
}

export class SELU {
  __call__(x: MLXArray): MLXArray { return selu(x); }
}

export class HardTanh {
  __call__(x: MLXArray): MLXArray { return hard_tanh(x); }
}

export class HardShrink {
  __call__(x: MLXArray): MLXArray { return hard_shrink(x); }
}

export class Softmin {
  __call__(x: MLXArray): MLXArray { return softmin(x); }
}

export class GLU {
  private axis: number;
  constructor(axis = -1) { this.axis = axis; }
  __call__(x: MLXArray): MLXArray { return glu(x, this.axis); }
}

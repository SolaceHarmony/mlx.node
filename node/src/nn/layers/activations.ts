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
  abs,
  sign,
  greater,
  split,
} from '../../core/ops';
import { Module } from './base';

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
  // tanh approximation
  const inner = multiply(
    Math.sqrt(2 / Math.PI),
    add(x, multiply(0.044715, multiply(x, multiply(x, x)))),
  );
  return multiply(0.5, multiply(x, add(1, mx_tanh(inner))));
}

export function gelu_fast_approx(x: MLXArray): MLXArray {
  return multiply(x, sigmoid(multiply(1.702, x)));
}

export function glu(x: MLXArray, axis = -1): MLXArray {
  const parts = x.shape[axis < 0 ? x.shape.length + axis : axis];
  if (parts % 2 !== 0) {
    throw new Error('GLU: input size along axis must be even');
  }
  const [a, b] = split(x, 2, axis);
  return multiply(a, sigmoid(b));
}

export function step(x: MLXArray, threshold = 0.0): MLXArray {
  return where(greater(x, threshold), 1.0, 0.0);
}

export function selu(x: MLXArray): MLXArray {
  const alpha = 1.6732632423543772848170429916717;
  const lambda = 1.0507009873554804934193349852946;
  return multiply(
    lambda,
    where(greater(x, 0), x, multiply(alpha, subtract(exp(x), 1))),
  );
}

export function prelu(x: MLXArray, weight: MLXArray): MLXArray {
  return add(maximum(x, 0), multiply(weight, minimum(x, 0)));
}

export function mish(x: MLXArray): MLXArray {
  return multiply(x, mx_tanh(softplus(x)));
}

export function hardswish(x: MLXArray): MLXArray {
  return multiply(x, divide(relu6(add(x, 3)), 6));
}

export function hard_tanh(x: MLXArray, min_val = -1.0, max_val = 1.0): MLXArray {
  return maximum(minimum(x, max_val), min_val);
}

export function hard_shrink(x: MLXArray, lambd = 0.5): MLXArray {
  return where(greater(abs(x), lambd), x, 0.0);
}

export function softmin(x: MLXArray, axis = -1): MLXArray {
  return mx_softmax(negative(x), axis);
}

export function tanh(x: MLXArray): MLXArray {
  return mx_tanh(x);
}

// ───────────────────────── Module classes ──────────────────────────────────

export class Sigmoid extends Module {
  __call__(x: MLXArray): MLXArray { return sigmoid(x); }
}

export class Mish extends Module {
  __call__(x: MLXArray): MLXArray { return mish(x); }
}

export class ReLU extends Module {
  __call__(x: MLXArray): MLXArray { return relu(x); }
}

export class LeakyReLU extends Module {
  private _negative_slope: number;
  constructor(negative_slope = 0.01) {
    super();
    this._negative_slope = negative_slope;
  }
  __call__(x: MLXArray): MLXArray { return leaky_relu(x, this._negative_slope); }
}

export class ELU extends Module {
  private _alpha: number;
  constructor(alpha = 1.0) {
    super();
    this._alpha = alpha;
  }
  __call__(x: MLXArray): MLXArray { return elu(x, this._alpha); }
}

export class ReLU6 extends Module {
  __call__(x: MLXArray): MLXArray { return relu6(x); }
}

export class Softmax extends Module {
  __call__(x: MLXArray): MLXArray { return softmax(x); }
}

export class Softplus extends Module {
  __call__(x: MLXArray): MLXArray { return softplus(x); }
}

export class Softsign extends Module {
  __call__(x: MLXArray): MLXArray { return softsign(x); }
}

export class Softshrink extends Module {
  private lambd: number;
  constructor(lambd = 0.5) {
    super();
    this.lambd = lambd;
  }
  __call__(x: MLXArray): MLXArray { return softshrink(x, this.lambd); }
}

export class CELU extends Module {
  private _alpha: number;
  constructor(alpha = 1.0) {
    super();
    this._alpha = alpha;
  }
  __call__(x: MLXArray): MLXArray { return celu(x, this._alpha); }
}

export class SiLU extends Module {
  __call__(x: MLXArray): MLXArray { return silu(x); }
}

export class LogSoftmax extends Module {
  __call__(x: MLXArray): MLXArray { return log_softmax(x); }
}

export class LogSigmoid extends Module {
  __call__(x: MLXArray): MLXArray { return log_sigmoid(x); }
}

export class PReLU extends Module {
  weight: MLXArray;
  constructor(num_parameters = 1, init = 0.25) {
    super();
    this.weight = full([num_parameters], init);
  }
  __call__(x: MLXArray): MLXArray { return prelu(x, this.weight); }
}

export class GELU extends Module {
  private _approx: string;
  constructor(approx = 'none') {
    super();
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

export class Tanh extends Module {
  __call__(x: MLXArray): MLXArray { return tanh(x); }
}

export class Hardswish extends Module {
  __call__(x: MLXArray): MLXArray { return hardswish(x); }
}

export class Step extends Module {
  private threshold: number;
  constructor(threshold = 0.0) {
    super();
    this.threshold = threshold;
  }
  __call__(x: MLXArray): MLXArray { return step(x, this.threshold); }
}

export class SELU extends Module {
  __call__(x: MLXArray): MLXArray { return selu(x); }
}

export class HardTanh extends Module {
  __call__(x: MLXArray): MLXArray { return hard_tanh(x); }
}

export class HardShrink extends Module {
  __call__(x: MLXArray): MLXArray { return hard_shrink(x); }
}

export class Softmin extends Module {
  __call__(x: MLXArray): MLXArray { return softmin(x); }
}

export class GLU extends Module {
  private axis: number;
  constructor(axis = -1) {
    super();
    this.axis = axis;
  }
  __call__(x: MLXArray): MLXArray { return glu(x, this.axis); }
}

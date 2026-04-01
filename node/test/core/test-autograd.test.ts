import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../../src/core';

/**
 * Autograd / transforms tests ported from python/tests/test_autograd.py.
 *
 * batch6-transforms.test.ts already covers the basics (grad of x^2, basic
 * vjp/jvp, simple compile/checkpoint, multi-arg grad).  This file adds
 * deeper coverage: higher-order derivatives, power grad edge cases,
 * captured-variable grads, stop_gradient, multi-input/output vjp/jvp,
 * vmap, compiled-grad equivalence, checkpointed-grad equivalence, and more.
 */

const approx = (a: number, b: number, tol = 1e-5) => Math.abs(a - b) < tol;

/** Extract scalar float from a 0-d or 1-element array */
const scalar = (arr: any): number => {
  const v = arr.toArray();
  return Array.isArray(v) ? (v as number[])[0] : v as number;
};

describe('autograd: grad fundamentals', () => {
  it('grad of x^2 at x=0.5 gives 1.0', () => {
    const f = (x: any) => core.sum(core.multiply(x, x));
    const df = core.grad(f);
    const g = df(core.array(new Float32Array([0.5]))) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 1.0), `Expected 1.0, got ${scalar(g)}`);
  });

  it('grad of x^3 gives 3x^2', () => {
    // f(x) = x^3 = x * x * x, f'(x) = 3x^2
    const f = (x: any) => core.sum(core.multiply(core.multiply(x, x), x));
    const df = core.grad(f);
    const x = core.array(new Float32Array([2.0]));
    const g = df(x) as any;
    core.eval_op(g);
    // f'(2) = 3 * 4 = 12
    assert.ok(approx(scalar(g), 12.0), `Expected 12.0, got ${scalar(g)}`);
  });

  it('grad of x^3 at x=3 gives 27', () => {
    const f = (x: any) => core.sum(core.multiply(core.multiply(x, x), x));
    const df = core.grad(f);
    const x = core.array(new Float32Array([3.0]));
    const g = df(x) as any;
    core.eval_op(g);
    // f'(3) = 3 * 9 = 27
    assert.ok(approx(scalar(g), 27.0), `Expected 27.0, got ${scalar(g)}`);
  });

  it('second derivative of x^2 is 2', () => {
    // f(x) = sum(x^2), f'(x) = 2x (as array), need sum for next grad
    // For higher-order: each grad level must return scalar
    const f = (x: any) => core.sum(core.multiply(x, x));
    const df = core.grad(f);
    // df returns an MLXArray (non-scalar for 1-element input), wrap in sum
    const d2f = core.grad((x: any) => core.sum(df(x) as any));
    const x = core.array(new Float32Array([0.5]));
    const g = d2f(x) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 2.0), `Expected 2.0, got ${scalar(g)}`);
  });

  it('third derivative of x^2 is 0', () => {
    const f = (x: any) => core.sum(core.multiply(x, x));
    const df = core.grad(f);
    const d2f = core.grad((x: any) => core.sum(df(x) as any));
    const d3f = core.grad((x: any) => core.sum(d2f(x) as any));
    const x = core.array(new Float32Array([0.5]));
    const g = d3f(x) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 0.0), `Expected 0.0, got ${scalar(g)}`);
  });

  it('second derivative of x^3 is 6x', () => {
    // f(x) = sum(x*x*x), f'(x)=3x^2, f''(x)=6x
    const f = (x: any) => core.sum(core.multiply(core.multiply(x, x), x));
    const df = core.grad(f);
    const d2f = core.grad((x: any) => core.sum(df(x) as any));
    const x = core.array(new Float32Array([2.0]));
    const g = d2f(x) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 12.0), `Expected 12.0 (6*2), got ${scalar(g)}`);
  });
});

describe('autograd: value_and_grad', () => {
  it('returns both value and gradient for x^2', () => {
    const f = (x: any) => core.sum(core.multiply(x, x));
    const vg = core.value_and_grad(f);
    const x = core.array(new Float32Array([0.5]));
    const [value, grads] = vg(x);
    core.eval_op(value, ...grads);
    assert.ok(approx(scalar(value), 0.25), `value: expected 0.25, got ${scalar(value)}`);
    assert.ok(approx(scalar(grads[0]), 1.0), `grad: expected 1.0, got ${scalar(grads[0])}`);
  });

  it('value_and_grad with multi-element array', () => {
    const f = (x: any) => core.sum(core.multiply(x, x));
    const vg = core.value_and_grad(f);
    const x = core.array(new Float32Array([1, 2, 3]));
    const [value, grads] = vg(x);
    core.eval_op(value, ...grads);
    // sum(1+4+9) = 14
    assert.ok(approx(scalar(value), 14.0));
    const gv = grads[0].toArray() as number[];
    assert.ok(approx(gv[0], 2.0));
    assert.ok(approx(gv[1], 4.0));
    assert.ok(approx(gv[2], 6.0));
  });

  it('value_and_grad for f(x,y)=x*y, grad w.r.t. first arg', () => {
    const f = (x: any, y: any) => core.sum(core.multiply(x, y));
    const vg = core.value_and_grad(f, 0);
    const x = core.array(new Float32Array([2.0]));
    const y = core.array(new Float32Array([3.0]));
    const [value, grads] = vg(x, y);
    core.eval_op(value, ...grads);
    assert.ok(approx(scalar(value), 6.0));
    // df/dx = y = 3
    assert.ok(approx(scalar(grads[0]), 3.0));
  });
});

describe('autograd: grad with argnums', () => {
  it('grad w.r.t. second argument (argnums=1)', () => {
    const f = (x: any, y: any) => core.sum(core.multiply(x, y));
    const dfdy = core.grad(f as any, 1);
    const x = core.array(new Float32Array([2.0]));
    const y = core.array(new Float32Array([3.0]));
    const g = dfdy(x, y) as any;
    core.eval_op(g);
    // df/dy = x = 2
    assert.ok(approx(scalar(g), 2.0), `Expected 2.0, got ${scalar(g)}`);
  });

  it('grad w.r.t. first arg of three-arg function', () => {
    // f(x,y,z) = sum(x*y*z), df/dx = y*z
    const f = (x: any, y: any, z: any) => core.sum(core.multiply(core.multiply(x, y), z));
    const dfdx = core.grad(f as any, 0);
    const x = core.array(new Float32Array([2.0]));
    const y = core.array(new Float32Array([3.0]));
    const z = core.array(new Float32Array([4.0]));
    const g = dfdx(x, y, z) as any;
    core.eval_op(g);
    // df/dx = y*z = 12
    assert.ok(approx(scalar(g), 12.0));
  });

  it('grad w.r.t. multiple argnums [0, 1]', () => {
    const f = (x: any, y: any) => core.sum(core.multiply(x, y));
    const df = core.grad(f as any, [0, 1]);
    const x = core.array(new Float32Array([2.0]));
    const y = core.array(new Float32Array([3.0]));
    const grads = df(x, y) as any[];
    core.eval_op(...grads);
    // df/dx = y = 3, df/dy = x = 2
    assert.ok(approx(scalar(grads[0]), 3.0));
    assert.ok(approx(scalar(grads[1]), 2.0));
  });
});

describe('autograd: captured variables', () => {
  it('grad of f(x) = a + x gives 1 (a is captured)', () => {
    const a = core.array(new Float32Array([5.0]));
    const f = (x: any) => core.sum(core.add(a, x));
    const df = core.grad(f);
    const g = df(a) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 1.0));
  });

  it('grad of g(x) = a + a gives 0 (x unused)', () => {
    const a = core.array(new Float32Array([5.0]));
    const g = (x: any) => core.sum(core.add(a, a));
    const dg = core.grad(g);
    const result = dg(a) as any;
    core.eval_op(result);
    assert.ok(approx(scalar(result), 0.0));
  });

  it('grad of h(x) = x + x gives 2', () => {
    const a = core.array(new Float32Array([5.0]));
    const h = (x: any) => core.sum(core.add(x, x));
    const dh = core.grad(h);
    const result = dh(a) as any;
    core.eval_op(result);
    assert.ok(approx(scalar(result), 2.0));
  });

  it('second derivative of captured f(x) = a + x is 0', () => {
    const a = core.array(new Float32Array([5.0]));
    const f = (x: any) => core.sum(core.add(a, x));
    const df = core.grad(f);
    const d2f = core.grad((x: any) => core.sum(df(x) as any));
    const result = d2f(a) as any;
    core.eval_op(result);
    assert.ok(approx(scalar(result), 0.0));
  });
});

describe('autograd: stop_gradient', () => {
  it('gradient through stop_gradient is zero', () => {
    const f = (x: any) => core.sum(core.add(x, core.stop_gradient(x)));
    const df = core.grad(f);
    const x = core.array(new Float32Array([3.0]));
    const g = df(x) as any;
    core.eval_op(g);
    // d/dx(x + stop_grad(x)) = 1 + 0 = 1
    assert.ok(approx(scalar(g), 1.0), `Expected 1.0, got ${scalar(g)}`);
  });

  it('stop_gradient blocks all gradient flow when used alone', () => {
    const f = (x: any) => core.sum(core.multiply(core.stop_gradient(x), core.stop_gradient(x)));
    const df = core.grad(f);
    const x = core.array(new Float32Array([3.0]));
    const g = df(x) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 0.0), `Expected 0.0, got ${scalar(g)}`);
  });
});

describe('autograd: power grad', () => {
  it('grad of x^2 at x=0 is 0', () => {
    const f = (x: any) => core.sum(core.power(x, core.array(new Float32Array([2.0]))));
    const df = core.grad(f);
    const x = core.array(new Float32Array([0.0]));
    const g = df(x) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 0.0));
  });

  it('grad of x^2 at x=2 is 4', () => {
    const f = (x: any) => core.sum(core.power(x, core.array(new Float32Array([2.0]))));
    const df = core.grad(f);
    const x = core.array(new Float32Array([2.0]));
    const g = df(x) as any;
    core.eval_op(g);
    assert.ok(approx(scalar(g), 4.0, 1e-4), `Expected 4.0, got ${scalar(g)}`);
  });
});

describe('autograd: vjp (vector-Jacobian product)', () => {
  it('vjp of 2*x with cotangent 2', () => {
    const f = (x: any) => core.multiply(core.array(new Float32Array([2.0])), x);
    const x = core.array(new Float32Array([1.0]));
    const ct = core.array(new Float32Array([2.0]));
    const [outputs, vjps] = core.vjp(f as any, [x], [ct]);
    core.eval_op(...outputs, ...vjps);
    // f(1) = 2
    assert.ok(approx(scalar(outputs[0]), 2.0));
    // vjp = 2 * ct = 4
    assert.ok(approx(scalar(vjps[0]), 4.0));
  });

  it('vjp with two inputs: f(x,y) = x*y', () => {
    const f = (x: any, y: any) => core.multiply(x, y);
    const x = core.array(new Float32Array([4.0]));
    const y = core.array(new Float32Array([2.0]));
    const ct = core.array(new Float32Array([3.0]));
    const [_, vjps] = core.vjp(f as any, [x, y], [ct]);
    core.eval_op(...vjps);
    // df/dx = y = 2, vjp_x = y * ct = 6
    assert.ok(approx(scalar(vjps[0]), 6.0), `Expected 6.0, got ${scalar(vjps[0])}`);
    // df/dy = x = 4, vjp_y = x * ct = 12
    assert.ok(approx(scalar(vjps[1]), 12.0), `Expected 12.0, got ${scalar(vjps[1])}`);
  });

  it('vjp with two outputs: f(x,y,z) = (x*y, y*z)', () => {
    const f = (x: any, y: any, z: any) => [core.multiply(x, y), core.multiply(y, z)];
    const x = core.array(new Float32Array([2.0]));
    const y = core.array(new Float32Array([4.0]));
    const z = core.array(new Float32Array([6.0]));
    const ct0 = core.array(new Float32Array([1.0]));
    const ct1 = core.array(new Float32Array([3.0]));
    const [outputs, vjps] = core.vjp(f as any, [x, y, z], [ct0, ct1]);
    core.eval_op(...outputs, ...vjps);

    assert.equal(vjps.length, 3);
    // d(x*y)/dx * ct0 = y * 1 = 4
    assert.ok(approx(scalar(vjps[0]), 4.0), `vjp[0]: expected 4.0, got ${scalar(vjps[0])}`);
    // d(x*y)/dy * ct0 + d(y*z)/dy * ct1 = x*1 + z*3 = 2 + 18 = 20
    assert.ok(approx(scalar(vjps[1]), 20.0), `vjp[1]: expected 20.0, got ${scalar(vjps[1])}`);
    // d(y*z)/dz * ct1 = y * 3 = 12
    assert.ok(approx(scalar(vjps[2]), 12.0), `vjp[2]: expected 12.0, got ${scalar(vjps[2])}`);
  });
});

describe('autograd: jvp (Jacobian-vector product)', () => {
  it('jvp of 2*x with tangent 2', () => {
    const f = (x: any) => core.multiply(core.array(new Float32Array([2.0])), x);
    const x = core.array(new Float32Array([1.0]));
    const t = core.array(new Float32Array([2.0]));
    const [outputs, jvps] = core.jvp(f as any, [x], [t]);
    core.eval_op(...outputs, ...jvps);
    assert.ok(approx(scalar(outputs[0]), 2.0));
    // jvp = 2 * tangent = 4
    assert.ok(approx(scalar(jvps[0]), 4.0));
  });

  it('jvp with two inputs: f(x,y) = x*y', () => {
    const f = (x: any, y: any) => core.multiply(x, y);
    const x = core.array(new Float32Array([4.0]));
    const y = core.array(new Float32Array([2.0]));
    const tx = core.array(new Float32Array([3.0]));
    const ty = core.array(new Float32Array([2.0]));
    const [_, jvps] = core.jvp(f as any, [x, y], [tx, ty]);
    core.eval_op(...jvps);
    // d(x*y) = y*dx + x*dy = 2*3 + 4*2 = 14
    assert.ok(approx(scalar(jvps[0]), 14.0), `Expected 14.0, got ${scalar(jvps[0])}`);
  });

  it('jvp with two outputs: f(x,y,z) = (x*y, y*z)', () => {
    const f = (x: any, y: any, z: any) => [core.multiply(x, y), core.multiply(y, z)];
    const x = core.array(new Float32Array([2.0]));
    const y = core.array(new Float32Array([4.0]));
    const z = core.array(new Float32Array([6.0]));
    const tx = core.array(new Float32Array([1.0]));
    const ty = core.array(new Float32Array([3.0]));
    const tz = core.array(new Float32Array([1.0]));
    const [outputs, jvps] = core.jvp(f as any, [x, y, z], [tx, ty, tz]);
    core.eval_op(...outputs, ...jvps);

    assert.equal(jvps.length, 2);
    // d(x*y) = y*tx + x*ty = 4*1 + 2*3 = 10
    assert.ok(approx(scalar(jvps[0]), 10.0), `jvp[0]: expected 10.0, got ${scalar(jvps[0])}`);
    // d(y*z) = z*ty + y*tz = 6*3 + 4*1 = 22
    assert.ok(approx(scalar(jvps[1]), 22.0), `jvp[1]: expected 22.0, got ${scalar(jvps[1])}`);
  });
});

describe('autograd: vmap', () => {
  it('vmap of element-wise add', () => {
    const f = (x: any) => core.add(x, x);
    const vmapped = core.vmap(f as any);
    const x = core.array(new Float32Array([1, 2, 3]));
    const result = vmapped(x) as any;
    core.eval_op(result);
    const vals = result.toArray() as number[];
    assert.ok(approx(vals[0], 2.0));
    assert.ok(approx(vals[1], 4.0));
    assert.ok(approx(vals[2], 6.0));
  });

  it('vmap of element-wise multiply', () => {
    const f = (x: any, y: any) => core.multiply(x, y);
    const vmapped = core.vmap(f as any);
    const x = core.array(new Float32Array([1, 2, 3]));
    const y = core.array(new Float32Array([4, 5, 6]));
    const result = vmapped(x, y) as any;
    core.eval_op(result);
    const vals = result.toArray() as number[];
    assert.ok(approx(vals[0], 4.0));
    assert.ok(approx(vals[1], 10.0));
    assert.ok(approx(vals[2], 18.0));
  });

  it('vmap of square', () => {
    const f = (x: any) => core.multiply(x, x);
    const vmapped = core.vmap(f as any);
    const x = core.array(new Float32Array([2, 3, 4]));
    const result = vmapped(x) as any;
    core.eval_op(result);
    const vals = result.toArray() as number[];
    assert.ok(approx(vals[0], 4.0));
    assert.ok(approx(vals[1], 9.0));
    assert.ok(approx(vals[2], 16.0));
  });
});

describe('autograd: compile', () => {
  it('compiled function produces same results as uncompiled', () => {
    const f = (x: any) => core.add(core.multiply(x, x), x);
    const compiled = core.compile_fn(f as any);
    const x = core.array(new Float32Array([3.0]));

    const expected = f(x);
    const actual = compiled(x) as any;
    core.eval_op(expected, actual);

    assert.ok(approx(scalar(expected), scalar(actual)),
      `expected ${scalar(expected)}, got ${scalar(actual)}`);
  });

  it('compiled multi-element function', () => {
    const f = (x: any) => core.add(x, core.multiply(x, x));
    const compiled = core.compile_fn(f as any);
    const x = core.array(new Float32Array([1, 2, 3]));

    const expected = f(x);
    const actual = compiled(x) as any;
    core.eval_op(expected, actual);

    const ev = expected.toArray() as number[];
    const av = (actual as any).toArray() as number[];
    for (let i = 0; i < 3; i++) {
      assert.ok(approx(ev[i], av[i]));
    }
  });

  it('compiled function called multiple times', () => {
    const f = (x: any) => core.multiply(x, x);
    const compiled = core.compile_fn(f as any);

    for (const val of [1.0, 2.0, 3.0, 4.0]) {
      const x = core.array(new Float32Array([val]));
      const result = compiled(x) as any;
      core.eval_op(result);
      assert.ok(approx(scalar(result), val * val));
    }
  });
});

describe('autograd: checkpoint', () => {
  it('checkpointed grad matches regular grad for x^2', () => {
    const f = (x: any) => core.sum(core.multiply(x, x));
    const cpf = core.checkpoint(f as any);

    const regularGrad = core.grad(f);
    const checkpointGrad = core.grad(cpf as any);

    const x = core.array(new Float32Array([3.0]));
    const g1 = regularGrad(x) as any;
    const g2 = checkpointGrad(x) as any;
    core.eval_op(g1, g2);

    assert.ok(approx(scalar(g1), scalar(g2)),
      `regular: ${scalar(g1)}, checkpoint: ${scalar(g2)}`);
  });

  it('checkpointed grad matches for multi-element array', () => {
    const f = (x: any) => core.sum(core.multiply(x, x));
    const cpf = core.checkpoint(f as any);

    const regularGrad = core.grad(f);
    const checkpointGrad = core.grad(cpf as any);

    const x = core.array(new Float32Array([1, 2, 3, 4]));
    const g1 = regularGrad(x) as any;
    const g2 = checkpointGrad(x) as any;
    core.eval_op(g1, g2);

    const v1 = g1.toArray() as number[];
    const v2 = g2.toArray() as number[];
    for (let i = 0; i < 4; i++) {
      assert.ok(approx(v1[i], v2[i]), `index ${i}: regular=${v1[i]}, checkpoint=${v2[i]}`);
    }
  });

  it('checkpointed function forward pass unchanged', () => {
    const f = (x: any) => core.add(core.multiply(x, x), x);
    const cpf = core.checkpoint(f as any);

    const x = core.array(new Float32Array([5.0]));
    const expected = f(x);
    const actual = cpf(x) as any;
    core.eval_op(expected, actual);

    assert.ok(approx(scalar(expected), scalar(actual)));
  });
});

describe('autograd: combined transforms', () => {
  it('grad of compiled function matches grad of original', () => {
    const f = (x: any) => core.sum(core.multiply(x, x));
    const compiled = core.compile_fn(f as any);

    const gOrig = core.grad(f);
    const gCompiled = core.grad(compiled as any);

    const x = core.array(new Float32Array([4.0]));
    const r1 = gOrig(x) as any;
    const r2 = gCompiled(x) as any;
    core.eval_op(r1, r2);

    assert.ok(approx(scalar(r1), scalar(r2)),
      `original grad: ${scalar(r1)}, compiled grad: ${scalar(r2)}`);
  });

  it('vjp and grad agree for simple function', () => {
    // Both should give f'(x) = 2x for f(x) = x^2
    const f = (x: any) => core.multiply(x, x);
    const x = core.array(new Float32Array([3.0]));
    const ct = core.array(new Float32Array([1.0]));

    const gradResult = core.grad((x: any) => core.sum(core.multiply(x, x)))(x) as any;
    const [_, vjpResult] = core.vjp(f as any, [x], [ct]);

    core.eval_op(gradResult, ...vjpResult);
    assert.ok(approx(scalar(gradResult), scalar(vjpResult[0])),
      `grad: ${scalar(gradResult)}, vjp: ${scalar(vjpResult[0])}`);
  });

  it('jvp and grad agree for simple function', () => {
    // f(x) = x^2, f'(3) = 6
    // jvp with tangent=1 should give the same derivative
    const f = (x: any) => core.multiply(x, x);
    const x = core.array(new Float32Array([3.0]));
    const t = core.array(new Float32Array([1.0]));

    const gradResult = core.grad((x: any) => core.sum(core.multiply(x, x)))(x) as any;
    const [_, jvpResult] = core.jvp(f as any, [x], [t]);

    core.eval_op(gradResult, ...jvpResult);
    assert.ok(approx(scalar(gradResult), scalar(jvpResult[0])),
      `grad: ${scalar(gradResult)}, jvp: ${scalar(jvpResult[0])}`);
  });
});

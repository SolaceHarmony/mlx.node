import { describe, it } from 'mocha';
import * as assert from 'assert';
import * as core from '../../src/core';

describe('batch 6: transforms', () => {
  it('grad computes derivative of x^2', () => {
    // f(x) = sum(x^2), f'(x) = 2x
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.sum(core.multiply(x, x));
    const gradF = core.grad(f);
    const x = core.array(new Float32Array([3]), [1]);
    const g = gradF(x) as InstanceType<typeof core.MLXArray>;
    core.eval_op(g);
    const val = g.toArray() as number[];
    assert.ok(Math.abs(val[0] - 6) < 1e-4, `Expected ~6, got ${val[0]}`);
  });

  it('grad with multi-element array', () => {
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.sum(core.multiply(x, x));
    const gradF = core.grad(f);
    const x = core.array(new Float32Array([1, 2, 3]), [3]);
    const g = gradF(x) as InstanceType<typeof core.MLXArray>;
    core.eval_op(g);
    const val = g.toArray() as number[];
    assert.ok(Math.abs(val[0] - 2) < 1e-4);
    assert.ok(Math.abs(val[1] - 4) < 1e-4);
    assert.ok(Math.abs(val[2] - 6) < 1e-4);
  });

  it('value_and_grad returns both value and gradient', () => {
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.sum(core.multiply(x, x));
    const vgF = core.value_and_grad(f);
    const x = core.array(new Float32Array([2, 3]), [2]);
    const [value, grads] = vgF(x);
    core.eval_op(value, ...grads);
    const v = value.toArray() as number[];
    // sum(x^2) = 4 + 9 = 13
    assert.ok(Math.abs(v[0] - 13) < 1e-4, `Expected ~13, got ${v[0]}`);
    const gv = grads[0].toArray() as number[];
    assert.ok(Math.abs(gv[0] - 4) < 1e-4);
    assert.ok(Math.abs(gv[1] - 6) < 1e-4);
  });

  it('vjp computes vector-Jacobian product', () => {
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.multiply(x, x); // f(x) = x^2
    const x = core.array(new Float32Array([3]), [1]);
    const ct = core.array(new Float32Array([1]), [1]);
    const [outputs, vjps] = core.vjp(f as any, [x], [ct]);
    core.eval_op(...outputs, ...vjps);
    // f(3) = 9
    const outVal = outputs[0].toArray() as number[];
    assert.ok(Math.abs(outVal[0] - 9) < 1e-4);
    // vjp = 2*x * ct = 6
    const vjpVal = vjps[0].toArray() as number[];
    assert.ok(Math.abs(vjpVal[0] - 6) < 1e-4);
  });

  it('jvp computes Jacobian-vector product', () => {
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.multiply(x, x); // f(x) = x^2
    const x = core.array(new Float32Array([3]), [1]);
    const t = core.array(new Float32Array([1]), [1]);
    const [outputs, jvps] = core.jvp(f as any, [x], [t]);
    core.eval_op(...outputs, ...jvps);
    // f(3) = 9
    const outVal = outputs[0].toArray() as number[];
    assert.ok(Math.abs(outVal[0] - 9) < 1e-4);
    // jvp = 2*x * t = 6
    const jvpVal = jvps[0].toArray() as number[];
    assert.ok(Math.abs(jvpVal[0] - 6) < 1e-4);
  });

  it('enable_compile and disable_compile do not throw', () => {
    core.enable_compile();
    core.disable_compile();
    core.enable_compile(); // restore
  });

  it('compile wraps a function', () => {
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.add(x, x);
    const compiled = core.compile_fn(f as any);
    const x = core.array(new Float32Array([1, 2, 3]), [3]);
    const result = compiled(x) as InstanceType<typeof core.MLXArray>;
    core.eval_op(result);
    assert.deepStrictEqual(result.toArray(), [2, 4, 6]);
  });

  it('checkpoint wraps a function', () => {
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.multiply(x, x);
    const cpF = core.checkpoint(f as any);
    const x = core.array(new Float32Array([2, 3]), [2]);
    const result = cpF(x) as InstanceType<typeof core.MLXArray>;
    core.eval_op(result);
    assert.deepStrictEqual(result.toArray(), [4, 9]);
  });

  it('grad of multi-arg function', () => {
    // f(x, y) = sum(x * y), df/dx = y
    const f = (x: InstanceType<typeof core.MLXArray>, y: InstanceType<typeof core.MLXArray>) =>
      core.sum(core.multiply(x, y));
    const gradF = core.grad(f as any, 0); // gradient w.r.t. first arg
    const x = core.array(new Float32Array([1, 2]), [2]);
    const y = core.array(new Float32Array([3, 4]), [2]);
    const g = gradF(x, y) as InstanceType<typeof core.MLXArray>;
    core.eval_op(g);
    const val = g.toArray() as number[];
    // df/dx = y = [3, 4]
    assert.ok(Math.abs(val[0] - 3) < 1e-4);
    assert.ok(Math.abs(val[1] - 4) < 1e-4);
  });
});

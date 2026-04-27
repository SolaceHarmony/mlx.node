import { strict as assert } from 'node:assert';
import * as mx from '../src';

describe('ops parity', () => {
  it('test_full_ones_zeros', () => {
    let x = mx.full([2], 3.0);
    assert.deepEqual(x.shape, [2]);
    assert.deepEqual(x.toArray(), [3.0, 3.0]);

    x = mx.full([2, 3], 2.0);
    assert.equal(x.dtype, 'int32');
    assert.deepEqual(x.shape, [2, 3]);
    assert.deepEqual(x.toArray(), [[2, 2, 2], [2, 2, 2]]);

    x = mx.full([3, 2], mx.array([false, true]));
    assert.equal(x.dtype, mx.bool.key);
    assert.deepEqual(x.toArray(), [[false, true], [false, true], [false, true]]);

    x = mx.full([3, 2], mx.array([2.0, 3.0]));
    assert.deepEqual(x.toArray(), [[2, 3], [2, 3], [2, 3]]);

    x = mx.zeros([2]);
    assert.deepEqual(x.shape, [2]);
    assert.deepEqual(x.toArray(), [0.0, 0.0]);

    x = mx.ones([2]);
    assert.deepEqual(x.shape, [2]);
    assert.deepEqual(x.toArray(), [1.0, 1.0]);

    const testTypes = [mx.bool, mx.int32, mx.float32];
    for (const t of testTypes) {
      x = mx.zeros([2, 2], t);
      assert.equal(x.dtype, t.key);
      assert.ok(mx.array_equal(x, mx.array(new Int32Array([0, 0, 0, 0]), [2, 2])));
      
      const y = mx.zeros_like(x);
      assert.equal(y.dtype, t.key);
      assert.ok(mx.array_equal(y, x));

      x = mx.ones([2, 2], t);
      assert.equal(x.dtype, t.key);
      assert.ok(mx.array_equal(x, mx.array(new Int32Array([1, 1, 1, 1]), [2, 2])));
      
      const z = mx.ones_like(x);
      assert.equal(z.dtype, t.key);
      assert.ok(mx.array_equal(z, x));
    }
  });

  it('test_scalar_inputs', () => {
    const a = mx.add(false, true);
    assert.equal(a.dtype, mx.bool.key);
    assert.equal(a.toArray()[0], true);

    const b = mx.add(1, 2); // 2.0 is 2 in JS
    assert.equal(b.dtype, 'int32');
    assert.equal(b.toArray()[0], 3);

    const c = mx.multiply(2, 3);
    assert.equal(c.dtype, 'int32');
    assert.equal(c.toArray()[0], 6);
  });
});

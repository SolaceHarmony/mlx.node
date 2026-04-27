import { strict as assert } from 'node:assert';
import * as mx from '../src';

describe('mx.stream targeting', () => {
  it('should execute on GPU when requested', () => {
    console.log('mx.gpu type:', typeof mx.gpu);
    const a = mx.array([1, 2, 3]);
    const b = mx.array([4, 5, 6]);
    // Target GPU explicitly
    const c = mx.add(a, b, { stream: mx.gpu() });
    assert.deepEqual(c.toArray(), [5, 7, 9]);
  });

  it('should execute on CPU when requested', () => {
    const a = mx.array([1, 2, 3]);
    const b = mx.array([4, 5, 6]);
    // Target CPU explicitly
    const d = mx.add(a, b, { stream: mx.cpu });
    assert.deepEqual(d.toArray(), [5, 7, 9]);
  });

  it('should work with new_stream', () => {
    const a = mx.array([1, 2, 3]);
    const s = mx.new_stream(mx.gpu());
    const c = mx.add(a, a, { stream: s });
    assert.deepEqual(c.toArray(), [2, 4, 6]);
  });

  it('should work with with_stream', async () => {
    const a = mx.array([1, 2, 3]);
    const s = mx.new_stream(mx.cpu);
    const result = await mx.with_stream(s, () => {
        return mx.add(a, a);
    });
    assert.deepEqual(result.toArray(), [2, 4, 6]);
  });
});

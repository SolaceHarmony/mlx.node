/**
 * @fileoverview Node.js parity tests for python/tests/test_nn.py
 *
 * Covers: TestLayers activations, layer shapes, and sequential.
 * Weight-load / save / quantize / grad-of-module tests are omitted —
 * they require mx.grad / module.parameters() JS parity not yet fully ported.
 */

import { strict as assert } from 'node:assert';
import { nn, zeros, ones, array, from_js_array } from '../src';
import * as mx from '../src';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Alias for the MLXArray return type (mx.array returns this)
type MLXArray = ReturnType<typeof mx.array>;

/** Assert two MLXArrays are element-wise close. */
function assertAllClose(a: MLXArray, b: MLXArray, atol = 1e-4): void {
  const diff = mx.max(mx.abs(mx.subtract(a, b)));
  const val = (diff.toArray() as number[])[0];
  assert.ok(
    val <= atol,
    `Arrays not close: max diff ${val} > atol ${atol}`,
  );
}

/** Assert all elements of a flat float array are close to expected values. */
function assertClose(
  a: mx.Array,
  expected: number[],
  atol = 1e-4,
): void {
  const vals = a.toArray() as number[];
  assert.equal(vals.length, expected.length, 'length mismatch');
  for (let i = 0; i < vals.length; i++) {
    assert.ok(
      Math.abs(vals[i] - expected[i]) <= atol,
      `element[${i}]: got ${vals[i]}, expected ${expected[i]}, diff=${Math.abs(vals[i] - expected[i])}`,
    );
  }
}

// ---------------------------------------------------------------------------
// TestLayers — Identity, Linear, Bilinear
// ---------------------------------------------------------------------------

describe('TestLayers', () => {
  describe('identity', () => {
    it('output shape matches input shape', () => {
      // Python: layer = nn.Identity(); outputs = layer(mx.zeros((10, 4)))
      const inputs = zeros([10, 4], 'float32');
      const layer = new nn.Identity();
      const outputs = layer.forward(inputs);
      assert.deepEqual(outputs.shape, [10, 4]);
    });
  });

  describe('linear', () => {
    it('output shape (10, 8) for input (10, 4)', () => {
      // Python: layer = nn.Linear(input_dims=4, output_dims=8); outputs = layer(zeros((10,4)))
      const inputs = zeros([10, 4], 'float32');
      const layer = new nn.Linear(4, 8);
      const outputs = layer.forward(inputs);
      assert.deepEqual(outputs.shape, [10, 8]);
    });
  });

  describe('bilinear', () => {
    it('output shape (10, 6) for two inputs', () => {
      // Python: layer = nn.Bilinear(2, 4, 6); outputs = layer(zeros((10,2)), zeros((10,4)))
      const inputs1 = zeros([10, 2], 'float32');
      const inputs2 = zeros([10, 4], 'float32');
      const layer = new nn.Bilinear(2, 4, 6);
      const outputs = layer.forward(inputs1, inputs2);
      assert.deepEqual(outputs.shape, [10, 6]);
    });
  });

  // ---------------------------------------------------------------------------
  // Activation functions — functional API
  // ---------------------------------------------------------------------------

  describe('relu', () => {
    it('relu clamps negatives to zero', () => {
      // Python: y = nn.relu(mx.array([1.0, -1.0, 0.0]))
      //         self.assertTrue(mx.array_equal(y, mx.array([1.0, 0.0, 0.0])))
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.relu(x);
      assert.deepEqual(y.shape, [3]);
      assert.equal(y.dtype, 'float32');
      assert.deepEqual(y.toArray(), [1.0, 0.0, 0.0]);
    });
  });

  describe('leaky_relu', () => {
    it('default negative_slope=0.01', () => {
      // Python: y = nn.leaky_relu(mx.array([1.0, -1.0, 0.0]))
      //         self.assertTrue(mx.array_equal(y, mx.array([1.0, -0.01, 0.0])))
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.leaky_relu(x);
      assertClose(y, [1.0, -0.01, 0.0]);
    });

    it('LeakyReLU class with negative_slope=0.1', () => {
      // Python: y = nn.LeakyReLU(negative_slope=0.1)(x)
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      // LeakyReLU uses __call__ per the activations.ts implementation
      const y = new nn.LeakyReLU(0.1).__call__(x);
      assertClose(y, [1.0, -0.1, 0.0]);
    });
  });

  describe('elu', () => {
    it('functional elu default alpha=1.0', () => {
      // Python: y = nn.elu(mx.array([1.0, -1.0, 0.0]))
      //         expected_y = mx.array([1.0, -0.6321, 0.0])
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.elu(x);
      assertClose(y, [1.0, -0.6321, 0.0]);
    });

    it('ELU class with alpha=1.1', () => {
      // Python: y = nn.ELU(alpha=1.1)(x); expected_y = [1.0, -0.6953, 0.0]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = new nn.ELU(1.1).__call__(x);
      assertClose(y, [1.0, -0.6953, 0.0]);
    });
  });

  describe('relu6', () => {
    it('clamps at 6', () => {
      // Python: y = nn.relu6(mx.array([1.0, -1.0, 0.0, 7.0, -7.0]))
      //         self.assertTrue(mx.array_equal(y, mx.array([1.0, 0.0, 0.0, 6.0, 0.0])))
      const x = array(new Float32Array([1.0, -1.0, 0.0, 7.0, -7.0]));
      const y = nn.relu6(x);
      assert.deepEqual(y.shape, [5]);
      assertClose(y, [1.0, 0.0, 0.0, 6.0, 0.0], 1e-6);
    });
  });

  describe('softmax', () => {
    it('sums to ~1 and matches expected', () => {
      // Python: y = nn.softmax(mx.array([1.0, -1.0, 0.0]))
      //         expected_y = [0.6652, 0.0900, 0.2447]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.softmax(x);
      assert.deepEqual(y.shape, [3]);
      assertClose(y, [0.6652, 0.0900, 0.2447]);
    });
  });

  describe('softmin', () => {
    it('matches expected', () => {
      // Python: y = nn.softmin(mx.array([1.0, 2.0, 3.0]))
      //         expected_y = [0.6652, 0.2447, 0.0900]
      const x = array(new Float32Array([1.0, 2.0, 3.0]));
      const y = nn.softmin(x);
      assertClose(y, [0.6652, 0.2447, 0.0900]);
    });
  });

  describe('softplus', () => {
    it('matches expected values', () => {
      // Python: expected_y = [1.3133, 0.3133, 0.6931]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.softplus(x);
      assertClose(y, [1.3133, 0.3133, 0.6931]);
    });
  });

  describe('softsign', () => {
    it('x / (1 + |x|)', () => {
      // Python: expected_y = [0.5, -0.5, 0.0]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.softsign(x);
      assertClose(y, [0.5, -0.5, 0.0]);
    });
  });

  describe('softshrink', () => {
    it('default lambd=0.5', () => {
      // Python: expected_y = [0.5, -0.5, 0.0]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.softshrink(x);
      assertClose(y, [0.5, -0.5, 0.0]);
    });

    it('Softshrink class with lambd=0.7', () => {
      // Python: expected_y = [0.3, -0.3, 0.0]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = new nn.Softshrink(0.7).__call__(x);
      assertClose(y, [0.3, -0.3, 0.0]);
    });
  });

  describe('celu', () => {
    it('functional celu default', () => {
      // Python: expected_y = [1.0, -0.6321, 0.0]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.celu(x);
      assertClose(y, [1.0, -0.6321, 0.0]);
    });

    it('CELU class with alpha=1.1', () => {
      // Python: expected_y = [1.0, -0.6568, 0.0]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = new nn.CELU(1.1).__call__(x);
      assertClose(y, [1.0, -0.6568, 0.0]);
    });
  });

  describe('log_softmax', () => {
    it('matches expected', () => {
      // Python: expected_y = [-2.4076, -1.4076, -0.4076]
      const x = array(new Float32Array([1.0, 2.0, 3.0]));
      const y = nn.log_softmax(x);
      assertClose(y, [-2.4076, -1.4076, -0.4076]);
    });
  });

  describe('log_sigmoid', () => {
    it('matches expected', () => {
      // Python: expected_y = [-0.3133, -1.3133, -0.6931]
      const x = array(new Float32Array([1.0, -1.0, 0.0]));
      const y = nn.log_sigmoid(x);
      assertClose(y, [-0.3133, -1.3133, -0.6931]);
    });
  });

  describe('prelu', () => {
    it('PReLU default weight=0.25', () => {
      // Python: nn.PReLU()(mx.array([1.0, -1.0, 0.0, 0.5])) → [1.0, -0.25, 0.0, 0.5]
      const x = array(new Float32Array([1.0, -1.0, 0.0, 0.5]));
      const y = new nn.PReLU().__call__(x);
      assertClose(y, [1.0, -0.25, 0.0, 0.5]);
    });
  });

  describe('mish', () => {
    it('Mish activation values', () => {
      // Python: nn.Mish()(mx.array([1.0, -1.0, 0.0, 0.5])) → [0.8651, -0.3034, 0.0, 0.3752]
      const x = array(new Float32Array([1.0, -1.0, 0.0, 0.5]));
      const y = new nn.Mish().__call__(x);
      assertClose(y, [0.8651, -0.3034, 0.0000, 0.3752]);
    });
  });

  describe('hardswish', () => {
    it('matches expected values', () => {
      // Python: expected_y = [0.0, -0.375, 0.0, 1.125, 3.0]
      const x = array(new Float32Array([-3.0, -1.5, 0.0, 1.5, 3.0]));
      const y = nn.hardswish(x);
      assertClose(y, [0.0, -0.375, 0.0, 1.125, 3.0]);
    });
  });

  describe('glu', () => {
    it('gated linear unit', () => {
      // Python: x = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.float32)
      //         y = mx.array([[[0.952574, 1.96403]]], dtype=mx.float32)
      const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      const x = array(data, [1, 1, 4], 'float32');
      const y = nn.glu(x);
      assert.deepEqual(y.shape, [1, 1, 2]);
      const vals = y.toArray() as number[];
      assert.ok(Math.abs(vals[0] - 0.952574) < 1e-4);
      assert.ok(Math.abs(vals[1] - 1.96403) < 1e-4);
    });
  });

  describe('hard_tanh', () => {
    it('clamps to [-1, 1]', () => {
      // Python: expected_y = [1.0, -1.0, 0.0, 0.5, 1.0]
      const x = array(new Float32Array([1.0, -2.0, 0.0, 0.5, 2.0]));
      const y = nn.hard_tanh(x);
      assert.deepEqual(y.toArray(), [1.0, -1.0, 0.0, 0.5, 1.0]);
    });
  });

  describe('hard_shrink', () => {
    it('default lambd=0.5', () => {
      // Python: expected_y = [1.0, 0.0, 0.0, 0.0, -1.5]
      const x = array(new Float32Array([1.0, -0.5, 0.0, 0.5, -1.5]));
      const y = nn.hard_shrink(x);
      assert.deepEqual(y.toArray(), [1.0, 0.0, 0.0, 0.0, -1.5]);
    });

    it('hard_shrink with lambd=0.1', () => {
      // Python: expected_y = [1.0, -0.5, 0.0, 0.5, -1.5]
      const x = array(new Float32Array([1.0, -0.5, 0.0, 0.5, -1.5]));
      const y = nn.hard_shrink(x, 0.1);
      assert.deepEqual(y.toArray(), [1.0, -0.5, 0.0, 0.5, -1.5]);
    });
  });

  describe('gelu', () => {
    it('gelu matches JAX reference', () => {
      // Python: inputs = [1.15286231, -0.81037411, 0.35816911, 0.77484438, 0.66276414]
      //         expected = [1.0093501, -0.16925684, 0.22918941, 0.60498625, 0.49459383]
      const inputs = new Float32Array([
        1.15286231, -0.81037411, 0.35816911, 0.77484438, 0.66276414,
      ]);
      const x = array(inputs);
      const y = nn.gelu(x);
      assertClose(y, [1.0093501, -0.16925684, 0.22918941, 0.60498625, 0.49459383], 1e-3);
    });

    it('gelu_approx is within 0.0005 of exact gelu', () => {
      // Python: self.assertLess(mx.abs(y - y_hat1).max(), 0.0005)
      const x = mx.arange(-6.0, 6.0, 0.12, { dtype: 'float32' });
      const y = nn.gelu(x);
      const yHat = nn.gelu_approx(x);
      const diff = mx.max(mx.abs(mx.subtract(y, yHat)));
      assert.ok((diff.toArray() as number[])[0] < 0.0005);
    });

    it('gelu_fast_approx is within 0.025 of exact gelu', () => {
      // Python: self.assertLess(mx.abs(y - y_hat2).max(), 0.025)
      const x = mx.arange(-6.0, 6.0, 0.12, { dtype: 'float32' });
      const y = nn.gelu(x);
      const yHat = nn.gelu_fast_approx(x);
      const diff = mx.max(mx.abs(mx.subtract(y, yHat)));
      assert.ok((diff.toArray() as number[])[0] < 0.025);
    });

    it('GELU class forward', () => {
      const inputs = new Float32Array([
        1.15286231, -0.81037411, 0.35816911, 0.77484438, 0.66276414,
      ]);
      const x = array(inputs);
      const y = new nn.GELU().__call__(x);
      assertClose(y, [1.0093501, -0.16925684, 0.22918941, 0.60498625, 0.49459383], 1e-3);
    });
  });

  describe('sigmoid', () => {
    it('functional sigmoid matches class', () => {
      // Python: y1 = mx.sigmoid(x); y2 = nn.activations.sigmoid(x); y3 = nn.Sigmoid()(x)
      const x = array(new Float32Array([1.0, 0.0, -1.0]));
      const y1 = mx.sigmoid(x);
      const y2 = new nn.Sigmoid().__call__(x);

      // Both should match
      assertAllClose(y1, y2, 1e-6);
    });
  });

  describe('tanh', () => {
    it('Tanh class matches mx.tanh', () => {
      const x = array(new Float32Array([0.0, 1.0, -1.0]));
      const y1 = mx.tanh(x);
      const y2 = new nn.Tanh().__call__(x);
      assertAllClose(y1, y2, 1e-6);
    });
  });

  describe('step', () => {
    it('step activation', () => {
      const x = array(new Float32Array([-1.0, 0.0, 1.0]));
      const y = nn.step(x);
      assert.deepEqual(y.toArray(), [0.0, 0.0, 1.0]);
    });
  });

  // ---------------------------------------------------------------------------
  // Layer shapes — Conv1d, Conv2d, Dropout, Embedding
  // ---------------------------------------------------------------------------

  describe('conv1d', () => {
    it('output shape without bias', () => {
      // Python: N=5, L=12, ks=3, C_in=2, C_out=4
      //         y.shape == (N, L - ks + 1, C_out) == (5, 10, 4)
      const N = 5, L = 12, ks = 3, Cin = 2, Cout = 4;
      const x = ones([N, L, Cin], 'float32');
      const c = new nn.Conv1d(Cin, Cout, ks);
      const y = c.forward(x);
      assert.deepEqual(y.shape, [N, L - ks + 1, Cout]);
    });

    it('output shape with stride=2', () => {
      // Python: c = nn.Conv1d(C_in, C_out, ks, stride=2); y.shape == (N, (L-ks+1)//2, C_out)
      const N = 5, L = 12, ks = 3, Cin = 2, Cout = 4;
      const x = ones([N, L, Cin], 'float32');
      const c = new nn.Conv1d(Cin, Cout, ks, 2 /* stride */);
      const y = c.forward(x);
      assert.deepEqual(y.shape, [N, Math.floor((L - ks + 1) / 2), Cout]);
    });
  });

  describe('conv2d', () => {
    it('output shape for 8x8 kernel', () => {
      // Python: x = mx.ones((4, 8, 8, 3)); c = nn.Conv2d(3, 1, 8); y.shape == (4, 1, 1, 1)
      const x = ones([4, 8, 8, 3], 'float32');
      const c = new nn.Conv2d(3, 1, 8);
      const y = c.forward(x);
      assert.deepEqual(y.shape, [4, 1, 1, 1]);
    });

    it('output shape for 3x3 kernel', () => {
      // Python: c = nn.Conv2d(3, 8, 3); y.shape == (4, 6, 6, 8)
      const x = ones([4, 8, 8, 3], 'float32');
      const c = new nn.Conv2d(3, 8, 3);
      const y = c.forward(x);
      assert.deepEqual(y.shape, [4, 6, 6, 8]);
    });

    it('output shape for 3x3 kernel with padding=1', () => {
      // Python: c = nn.Conv2d(3, 8, 3, padding=1); y.shape == (4, 8, 8, 8)
      const x = ones([4, 8, 8, 3], 'float32');
      const c = new nn.Conv2d(3, 8, 3, 1 /* stride */, 1 /* padding */);
      const y = c.forward(x);
      assert.deepEqual(y.shape, [4, 8, 8, 8]);
    });
  });

  describe('dropout', () => {
    it('dropout output shape and dtype preserved', () => {
      // Python: x = mx.ones((2, 4)); y = nn.Dropout(0.5)(x)
      //         y.shape == x.shape; y.dtype == float32
      const x = ones([2, 4], 'float32');
      const y = new nn.Dropout(0.5).forward(x);
      assert.deepEqual(y.shape, [2, 4]);
      assert.equal(y.dtype, 'float32');
    });
  });

  describe('dropout2d', () => {
    it('dropout2d output shape and dtype preserved', () => {
      // Python: x = mx.ones((2, 4, 4, 4)); y = nn.Dropout2d(0.5)(x)
      const x = ones([2, 4, 4, 4], 'float32');
      const y = new nn.Dropout2d(0.5).forward(x);
      assert.deepEqual(y.shape, [2, 4, 4, 4]);
      assert.equal(y.dtype, 'float32');
    });
  });

  describe('dropout3d', () => {
    it('dropout3d output shape and dtype preserved', () => {
      // Python: x = mx.ones((2, 4, 4, 4, 4)); y = nn.Dropout3d(0.5)(x)
      const x = ones([2, 4, 4, 4, 4], 'float32');
      const y = new nn.Dropout3d(0.5).forward(x);
      assert.deepEqual(y.shape, [2, 4, 4, 4, 4]);
      assert.equal(y.dtype, 'float32');
    });
  });

  describe('embedding', () => {
    it('output shape for embedding lookup', () => {
      // Python: e = nn.Embedding(num_embeddings=10, dims=8)
      //         x = mx.array([0, 1, 2]); y = e(x); y.shape == (3, 8)
      const e = new nn.Embedding(10, 8);
      const x = from_js_array([0, 1, 2], 'int32');
      const y = e.forward(x);
      assert.deepEqual(y.shape, [3, 8]);
    });
  });

  describe('layernorm', () => {
    it('output shape preserved', () => {
      const x = array(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const ln = new nn.LayerNorm(2);
      const y = ln.forward(x);
      assert.deepEqual(y.shape, [2, 2]);
    });
  });

  describe('rmsnorm', () => {
    it('output shape preserved', () => {
      const x = array(new Float32Array([1, 2, 3, 4]), [2, 2], 'float32');
      const rms = new nn.RMSNorm(2);
      const y = rms.forward(x);
      assert.deepEqual(y.shape, [2, 2]);
    });
  });

  // ---------------------------------------------------------------------------
  // Pooling layers
  // ---------------------------------------------------------------------------

  describe('MaxPool1d', () => {
    it('kernel=2 stride=1 no_padding output is (N, L-1, C)', () => {
      // Python: x shape (2, 4, 3) → MaxPool1d(2, stride=1) → (2, 3, 3)
      const x = from_js_array(
        [0,1,2, 3,4,5, 6,7,8, 9,10,11,
         12,13,14, 15,16,17, 18,19,20, 21,22,23],
        'float32',
        [2, 4, 3],
      );
      const pool = new nn.MaxPool1d(2, 1, 0);
      const y = pool.forward(x);
      assert.deepEqual(y.shape, [2, 3, 3]);
      // Expected: [[[3,4,5],[6,7,8],[9,10,11]],[[15,16,17],[18,19,20],[21,22,23]]]
      const expected = [3,4,5, 6,7,8, 9,10,11, 15,16,17, 18,19,20, 21,22,23];
      const vals = y.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        assert.equal(vals[i], expected[i]);
      }
    });
  });

  describe('AvgPool1d', () => {
    it('kernel=2 stride=1 no_padding output is correct', () => {
      // Python: expected_avg  [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]] for batch 0
      const x = from_js_array(
        [0,1,2, 3,4,5, 6,7,8, 9,10,11,
         12,13,14, 15,16,17, 18,19,20, 21,22,23],
        'float32',
        [2, 4, 3],
      );
      const pool = new nn.AvgPool1d(2, 1, 0);
      const y = pool.forward(x);
      assert.deepEqual(y.shape, [2, 3, 3]);
      const expected = [1.5,2.5,3.5, 4.5,5.5,6.5, 7.5,8.5,9.5,
                        13.5,14.5,15.5, 16.5,17.5,18.5, 19.5,20.5,21.5];
      const vals = y.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        assert.ok(Math.abs(vals[i] - expected[i]) < 1e-4, `element ${i}: ${vals[i]} vs ${expected[i]}`);
      }
    });
  });

  describe('MaxPool2d', () => {
    it('kernel=2 stride=1 no_padding output shape', () => {
      // Python: x shape (1, 4, 4, 2) → MaxPool2d(2,1,0) → (1, 3, 3, 2)
      const xData = new Float32Array(Array.from({length: 32}, (_, i) => i));
      const x = array(xData, [1, 4, 4, 2], 'float32');
      const pool = new nn.MaxPool2d(2, 1, 0);
      const y = pool.forward(x);
      assert.deepEqual(y.shape, [1, 3, 3, 2]);
    });

    it('kernel=2 stride=2 no_padding output shape', () => {
      // Python: y.shape == [1, 2, 2, 2]
      const xData = new Float32Array(Array.from({length: 32}, (_, i) => i));
      const x = array(xData, [1, 4, 4, 2], 'float32');
      const pool = new nn.MaxPool2d(2, 2, 0);
      const y = pool.forward(x);
      assert.deepEqual(y.shape, [1, 2, 2, 2]);
    });
  });

  describe('AvgPool2d', () => {
    it('kernel=2 stride=1 no_padding output shape', () => {
      const xData = new Float32Array(Array.from({length: 32}, (_, i) => i));
      const x = array(xData, [1, 4, 4, 2], 'float32');
      const pool = new nn.AvgPool2d(2, 1, 0);
      const y = pool.forward(x);
      assert.deepEqual(y.shape, [1, 3, 3, 2]);
    });
  });

  // ---------------------------------------------------------------------------
  // Sequential
  // ---------------------------------------------------------------------------

  describe('Sequential', () => {
    // NOTE: Sequential is not yet exported; once it is, these tests activate.
    // Python: m = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
    //         y = m(mx.ones((10, 2))); y.shape == (10, 1)
    it('Sequential forward pass shape', () => {
      const model = new nn.Sequential(
        new nn.Linear(2, 10),
        new nn.ReLU(),
        new nn.Linear(10, 1)
      );
      const x = ones([5, 2]);
      const y = model.forward(x);
      assert.deepEqual(y.shape, [5, 1]);
    });
  });

  // ---------------------------------------------------------------------------
  // BatchNorm
  // ---------------------------------------------------------------------------

  describe('BatchNorm', () => {
    it('output shape matches input', () => {
      // Python: bn = nn.BatchNorm(num_features=4, affine=True)
      //         y = bn(x); x.shape == y.shape
      const bn = new nn.BatchNorm(4); // numFeatures=4, defaults: eps=1e-5, momentum=0.1, affine=true
      const xData = new Float32Array([
        -0.207, 1.069, 0.637, 1.285,
        -1.217, -0.937, -0.244, -0.414,
         0.645, -0.294, -0.233, -0.068,
         0.156,  0.069, -0.270, -0.227,
         0.631, -0.057,  1.108, -0.367,
      ]);
      const x = array(xData, [5, 4], 'float32');
      const y = bn.forward(x);
      assert.deepEqual(y.shape, [5, 4]);
    });
  });

  // ---------------------------------------------------------------------------
  // nn init (kept here to consolidate; also covered in nn_init.test.ts)
  // ---------------------------------------------------------------------------

  describe('nn.init', () => {
    it('he_uniform correct shape', () => {
      const fn = nn.init.he_uniform();
      const w = fn(zeros([4, 3]));
      assert.deepEqual(w.shape, [4, 3]);
    });

    it('glorot_uniform correct shape', () => {
      const fn = nn.init.glorot_uniform();
      const w = fn(zeros([4, 3]));
      assert.deepEqual(w.shape, [4, 3]);
    });

    it('he_normal correct shape', () => {
      const fn = nn.init.he_normal();
      const w = fn(zeros([4, 3]));
      assert.deepEqual(w.shape, [4, 3]);
    });

    it('glorot_normal correct shape', () => {
      const fn = nn.init.glorot_normal();
      const w = fn(zeros([4, 3]));
      assert.deepEqual(w.shape, [4, 3]);
    });

    it('constant initializer', () => {
      const fn = nn.init.constant(7.0);
      const w = fn(zeros([3, 3]));
      const vals = w.toArray() as number[];
      assert.ok(vals.every((v) => Math.abs(v - 7.0) < 1e-6));
    });
  });
});

import { strict as assert } from 'assert';
import { core, nn } from '../src';

const mx = core;

describe('mlx.nn.losses', () => {
  describe('cross_entropy', () => {
    it('computes cross entropy with class indices', () => {
      const logits = mx.array(new Float32Array([2.0, -1.0, -1.0, 2.0]), [2, 2]);
      const targets = mx.array(new Int32Array([0, 1]), [2], 'int32');
      const loss = nn.losses.cross_entropy(logits, targets);
      const result = loss.toFloat32Array();
      assert.equal(result.length, 2);
      assert.ok(Math.abs(result[0] - 0.0486) < 0.01);
      assert.ok(Math.abs(result[1] - 0.0486) < 0.01);
    });

    it('computes cross entropy with probabilities', () => {
      const logits = mx.array(new Float32Array([2.0, -1.0, -1.0, 2.0]), [2, 2]);
      const targets = mx.array(new Float32Array([0.9, 0.1, 0.1, 0.9]), [2, 2]);
      const loss = nn.losses.cross_entropy(logits, targets);
      const result = loss.toFloat32Array();
      assert.equal(result.length, 2);
      assert.ok(Math.abs(result[0] - 0.3486) < 0.01);
    });

    it('supports mean reduction', () => {
      const logits = mx.array(new Float32Array([2.0, -1.0, -1.0, 2.0]), [2, 2]);
      const targets = mx.array(new Int32Array([0, 1]), [2], 'int32');
      const loss = nn.losses.cross_entropy(logits, targets, { reduction: 'mean' });
      const result = loss.toFloat32Array();
      assert.equal(result.length, 1);
      assert.ok(Math.abs(result[0] - 0.0486) < 0.01);
    });
  });

  describe('binary_cross_entropy', () => {
    it('computes BCE with logits', () => {
      const logits = mx.array(new Float32Array([0.105361, 0.223144, 1.20397, 0.916291]), [4]);
      const targets = mx.array(new Float32Array([0, 0, 1, 1]), [4]);
      const loss = nn.binary_cross_entropy(logits, targets, { reduction: 'mean' });
      const result = loss.toFloat32Array();
      assert.equal(result.length, 1);
      assert.ok(Math.abs(result[0] - 0.5392) < 0.01);
    });
  });

  describe('l1_loss', () => {
    it('computes L1 loss with mean reduction', () => {
      const predictions = mx.array(new Float32Array([1.0, 2.0, 3.0]), [3]);
      const targets = mx.array(new Float32Array([1.5, 2.5, 3.0]), [3]);
      const loss = nn.l1_loss(predictions, targets);
      const result = loss.toFloat32Array();
      assert.equal(result.length, 1);
      assert.ok(Math.abs(result[0] - 1.0 / 3) < 0.01);
    });
  });

  describe('mse_loss', () => {
    it('computes MSE loss', () => {
      const predictions = mx.array(new Float32Array([1.0, 2.0, 3.0]), [3]);
      const targets = mx.array(new Float32Array([1.5, 2.5, 3.0]), [3]);
      const loss = nn.mse_loss(predictions, targets);
      const result = loss.toFloat32Array();
      assert.equal(result.length, 1);
      assert.ok(Math.abs(result[0] - 1.0 / 6) < 0.01);
    });
  });

  describe('nll_loss', () => {
    it('computes NLL loss', () => {
      const logProbs = mx.array(new Float32Array([-0.5, -1.5, -1.5, -0.5]), [2, 2]);
      const targets = mx.array(new Int32Array([0, 1]), [2], 'int32');
      const loss = nn.nll_loss(logProbs, targets);
      const result = loss.toFloat32Array();
      assert.equal(result.length, 2);
      assert.ok(Math.abs(result[0] - 0.5) < 0.01);
      assert.ok(Math.abs(result[1] - 0.5) < 0.01);
    });
  });

  describe('hinge_loss', () => {
    it('computes hinge loss', () => {
      const inputs = mx.array(new Float32Array([1.0, -1.0, 0.5]), [3]);
      const targets = mx.array(new Float32Array([1.0, -1.0, -1.0]), [3]);
      const loss = nn.hinge_loss(inputs, targets);
      const result = loss.toFloat32Array();
      assert.ok(Math.abs(result[0] - 0) < 0.01);
      assert.ok(Math.abs(result[1] - 0) < 0.01);
      assert.ok(Math.abs(result[2] - 1.5) < 0.01);
    });
  });

  describe('huber_loss', () => {
    it('computes Huber loss', () => {
      const inputs = mx.array(new Float32Array([1.0, 5.0]), [2]);
      const targets = mx.array(new Float32Array([1.5, 2.0]), [2]);
      const loss = nn.huber_loss(inputs, targets);
      const result = loss.toFloat32Array();
      assert.ok(Math.abs(result[0] - 0.125) < 0.01);
      assert.ok(Math.abs(result[1] - 2.5) < 0.01);
    });
  });

  describe('log_cosh_loss', () => {
    it('computes log cosh loss', () => {
      const inputs = mx.array(new Float32Array([0.0, 1.0]), [2]);
      const targets = mx.array(new Float32Array([0.0, 0.0]), [2]);
      const loss = nn.log_cosh_loss(inputs, targets);
      const result = loss.toFloat32Array();
      assert.ok(Math.abs(result[0]) < 0.01);
      assert.ok(Math.abs(result[1] - 0.4337) < 0.01);
    });
  });

  describe('margin_ranking_loss', () => {
    it('computes margin ranking loss', () => {
      const inputs1 = mx.array(new Float32Array([1.0, 2.0]), [2]);
      const inputs2 = mx.array(new Float32Array([2.0, 1.0]), [2]);
      const targets = mx.array(new Float32Array([1.0, 1.0]), [2]);
      const loss = nn.margin_ranking_loss(inputs1, inputs2, targets);
      const result = loss.toFloat32Array();
      assert.ok(Math.abs(result[0] - 1.0) < 0.01);
      assert.ok(Math.abs(result[1]) < 0.01);
    });
  });

  describe('cosine_similarity_loss', () => {
    it('computes cosine similarity', () => {
      // Two identical vectors → similarity should be 1.0
      const x1 = mx.array(new Float32Array([1.0, 0.0, 0.0, 1.0]), [2, 2]);
      const x2 = mx.array(new Float32Array([1.0, 0.0, 0.0, 1.0]), [2, 2]);
      const loss = nn.cosine_similarity_loss(x1, x2);
      const result = loss.toFloat32Array();
      assert.equal(result.length, 2);
      assert.ok(Math.abs(result[0] - 1.0) < 0.01);
      assert.ok(Math.abs(result[1] - 1.0) < 0.01);
    });

    it('computes cosine similarity for orthogonal vectors', () => {
      // Orthogonal vectors → similarity should be 0.0
      const x1 = mx.array(new Float32Array([1.0, 0.0, 0.0, 1.0]), [2, 2]);
      const x2 = mx.array(new Float32Array([0.0, 1.0, 1.0, 0.0]), [2, 2]);
      const loss = nn.cosine_similarity_loss(x1, x2);
      const result = loss.toFloat32Array();
      assert.equal(result.length, 2);
      assert.ok(Math.abs(result[0]) < 0.01);
      assert.ok(Math.abs(result[1]) < 0.01);
    });
  });
});

// Copyright © 2023 Apple Inc.
// Ported from python/tests/test_losses.py — line‑for‑line transliteration.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as mx from '../src';

// ---------------------------------------------------------------------------
// Helper: check two MLXArrays are elementwise close within `atol`.
// Mirrors Python's mx.allclose(a, b) default (atol=1e-5, rtol=1e-5).
// ---------------------------------------------------------------------------
const allclose = (
  a: ReturnType<typeof mx.array>,
  b: ReturnType<typeof mx.array>,
  atol = 1e-4,
): boolean => {
  const av = (a.toArray() as any[]).flat(Infinity) as number[];
  const bv = (b.toArray() as any[]).flat(Infinity) as number[];
  if (av.length !== bv.length) return false;
  return av.every((x, i) => Math.abs(x - bv[i]) <= atol + 1e-5 * Math.abs(bv[i]));
};

/** Scalar value of a 0-d array or single element array. */
const item = (a: ReturnType<typeof mx.array>): number =>
  (a.toArray() as any[]).flat(Infinity)[0] as number;

describe('TestLosses', () => {
  // ---------------------------------------------------------------------------
  // test_cross_entropy
  // ---------------------------------------------------------------------------
  describe('cross_entropy', () => {
    it('computes correct loss for perfect one-hot predictions (class indices)', () => {
      // logits = [[0, -inf], [-inf, 0]], targets = [0, 1] -> loss = [0, 0]
      const logits = mx.array(
        new Float32Array([0.0, -Infinity, -Infinity, 0.0]),
        [2, 2],
      );
      const indices = mx.array(new Int32Array([0, 1]));
      const expected = mx.array(new Float32Array([0.0, 0.0]));
      const loss = mx.nn.losses.cross_entropy(logits, indices, {
        reduction: 'none',
      });
      assert.ok(allclose(loss, expected), 'cross_entropy perfect logits (indices)');
    });

    it('produces NaN for prob targets matched against -inf logits', () => {
      // probs targets produce NaNs like PyTorch
      const logits = mx.array(
        new Float32Array([0.0, -Infinity, -Infinity, 0.0]),
        [2, 2],
      );
      const probs = mx.array(new Float32Array([1.0, 0.0, 0.0, 1.0]), [2, 2]);
      const loss = mx.nn.losses.cross_entropy(logits, probs, {
        reduction: 'none',
      });
      const vals = (loss.toArray() as any[]).flat(Infinity) as number[];
      assert.ok(
        vals.every((v) => isNaN(v)),
        'Expected all NaN for -inf logit rows paired with prob targets',
      );
    });

    it('computes weighted cross_entropy with class indices', () => {
      const logits = mx.array(
        new Float32Array([2.0, -1.0, -1.0, 2.0]),
        [2, 2],
      );
      const indices = mx.array(new Int32Array([0, 1]));
      const weights = mx.array(new Float32Array([1.0, 2.0]));
      const expected = mx.array(new Float32Array([0.04858735, 0.0971747]));
      const loss = mx.nn.losses.cross_entropy(logits, indices, {
        weights,
        reduction: 'none',
      });
      assert.ok(allclose(loss, expected), 'weighted cross_entropy (indices)');
    });

    it('computes weighted cross_entropy with prob targets', () => {
      const logits = mx.array(
        new Float32Array([2.0, -1.0, -1.0, 2.0]),
        [2, 2],
      );
      const probs = mx.array(new Float32Array([1.0, 0.0, 0.0, 1.0]), [2, 2]);
      const weights = mx.array(new Float32Array([1.0, 2.0]));
      const expected = mx.array(new Float32Array([0.04858735, 0.0971747]));
      const loss = mx.nn.losses.cross_entropy(logits, probs, {
        weights,
        reduction: 'none',
      });
      assert.ok(allclose(loss, expected), 'weighted cross_entropy (probs)');
    });

    it('computes label-smoothed cross_entropy with class indices', () => {
      const logits = mx.array(
        new Float32Array([2.0, -1.0, -1.0, 2.0]),
        [2, 2],
      );
      const indices = mx.array(new Int32Array([0, 1]));
      const expected = mx.array(new Float32Array([0.498587, 0.498587]));
      const loss = mx.nn.losses.cross_entropy(logits, indices, {
        labelSmoothing: 0.3,
        reduction: 'none',
      });
      assert.ok(
        allclose(loss, expected, 1e-4),
        'label-smoothed cross_entropy (indices)',
      );
    });

    it('computes label-smoothed cross_entropy with prob targets', () => {
      const logits = mx.array(
        new Float32Array([2.0, -1.0, -1.0, 2.0]),
        [2, 2],
      );
      const probs = mx.array(new Float32Array([1.0, 0.0, 0.0, 1.0]), [2, 2]);
      const expected = mx.array(new Float32Array([0.498587, 0.498587]));
      const loss = mx.nn.losses.cross_entropy(logits, probs, {
        labelSmoothing: 0.3,
        reduction: 'none',
      });
      assert.ok(
        allclose(loss, expected, 1e-4),
        'label-smoothed cross_entropy (probs)',
      );
    });

    it('computes weighted + label-smoothed cross_entropy with class indices', () => {
      const logits = mx.array(
        new Float32Array([2.0, -1.0, -1.0, 2.0]),
        [2, 2],
      );
      const indices = mx.array(new Int32Array([0, 1]));
      const weights = mx.array(new Float32Array([1.0, 2.0]));
      const expected = mx.array(new Float32Array([0.49858734, 0.9971747]));
      const loss = mx.nn.losses.cross_entropy(logits, indices, {
        weights,
        labelSmoothing: 0.3,
        reduction: 'none',
      });
      assert.ok(
        allclose(loss, expected, 1e-4),
        'weighted + label-smoothed cross_entropy (indices)',
      );
    });

    it('computes weighted + label-smoothed cross_entropy with prob targets', () => {
      const logits = mx.array(
        new Float32Array([2.0, -1.0, -1.0, 2.0]),
        [2, 2],
      );
      const probs = mx.array(new Float32Array([1.0, 0.0, 0.0, 1.0]), [2, 2]);
      const weights = mx.array(new Float32Array([1.0, 2.0]));
      const expected = mx.array(new Float32Array([0.49858734, 0.9971747]));
      const loss = mx.nn.losses.cross_entropy(logits, probs, {
        weights,
        labelSmoothing: 0.3,
        reduction: 'none',
      });
      assert.ok(
        allclose(loss, expected, 1e-4),
        'weighted + label-smoothed cross_entropy (probs)',
      );
    });
  });

  // ---------------------------------------------------------------------------
  // test_binary_cross_entropy
  // ---------------------------------------------------------------------------
  describe('binary_cross_entropy', () => {
    describe('logits as inputs', () => {
      const logits = mx.array(
        new Float32Array([0.105361, 0.223144, 1.20397, 0.916291]),
      );
      const targets = mx.array(new Float32Array([0, 0, 1, 1]));
      const expectedNone = mx.array(
        new Float32Array([0.747215, 0.810930, 0.262365, 0.336472]),
      );

      it('reduction none', () => {
        const loss = mx.nn.losses.binary_cross_entropy(logits, targets, {
          reduction: 'none',
        });
        assert.ok(allclose(loss, expectedNone), 'BCE logits reduction=none');
      });

      it('reduction mean', () => {
        const loss = mx.nn.losses.binary_cross_entropy(logits, targets, {
          reduction: 'mean',
        });
        const expectedMean = mx.mean(expectedNone);
        assert.ok(allclose(loss, expectedMean), 'BCE logits reduction=mean');
      });

      it('reduction sum', () => {
        const loss = mx.nn.losses.binary_cross_entropy(logits, targets, {
          reduction: 'sum',
        });
        const expectedSum = mx.sum(expectedNone);
        assert.ok(allclose(loss, expectedSum), 'BCE logits reduction=sum');
      });

      it('with weights, no label smoothing', () => {
        const weights = mx.array(new Float32Array([1.0, 2.0, 1.0, 2.0]));
        const expected = mx.array(
          new Float32Array([0.747215, 1.62186, 0.262365, 0.672944]),
        );
        const loss = mx.nn.losses.binary_cross_entropy(logits, targets, {
          weights,
          reduction: 'none',
        });
        assert.ok(allclose(loss, expected), 'BCE logits with weights');
      });
    });

    describe('probs as inputs (with_logits=false)', () => {
      const probs = mx.array(new Float32Array([0.5, 0.6, 0.7, 0.8]));
      const targets = mx.array(new Float32Array([0, 0, 1, 1]));
      const expectedNone = mx.array(
        new Float32Array([0.693147, 0.916291, 0.356675, 0.223144]),
      );

      it('reduction none', () => {
        const loss = mx.nn.losses.binary_cross_entropy(probs, targets, {
          withLogits: false,
          reduction: 'none',
        });
        assert.ok(allclose(loss, expectedNone), 'BCE probs reduction=none');
      });

      it('reduction mean', () => {
        const loss = mx.nn.losses.binary_cross_entropy(probs, targets, {
          withLogits: false,
          reduction: 'mean',
        });
        const expectedMean = mx.mean(expectedNone);
        assert.ok(allclose(loss, expectedMean), 'BCE probs reduction=mean');
      });

      it('reduction sum', () => {
        const loss = mx.nn.losses.binary_cross_entropy(probs, targets, {
          withLogits: false,
          reduction: 'sum',
        });
        const expectedSum = mx.sum(expectedNone);
        assert.ok(allclose(loss, expectedSum), 'BCE probs reduction=sum');
      });
    });

    describe('tiny probs as inputs', () => {
      const TINY_PROB = 1e-38; // Use float32-safe tiny value (1e-59 underflows to 0 in f32)
      const probs = mx.array(
        new Float32Array([0, TINY_PROB, 1 - TINY_PROB, 1]),
      );
      const targets = mx.array(new Float32Array([0, 0, 1, 1]));
      const expectedNone = mx.array(
        new Float32Array([0.0, TINY_PROB, TINY_PROB, 0.0]),
      );

      it('reduction none', () => {
        const loss = mx.nn.losses.binary_cross_entropy(probs, targets, {
          withLogits: false,
          reduction: 'none',
        });
        assert.ok(allclose(loss, expectedNone, 1e-6), 'BCE tiny probs reduction=none');
      });

      it('reduction mean', () => {
        const loss = mx.nn.losses.binary_cross_entropy(probs, targets, {
          withLogits: false,
          reduction: 'mean',
        });
        const expectedMean = mx.mean(expectedNone);
        assert.ok(allclose(loss, expectedMean, 1e-6), 'BCE tiny probs reduction=mean');
      });

      it('reduction sum', () => {
        const loss = mx.nn.losses.binary_cross_entropy(probs, targets, {
          withLogits: false,
          reduction: 'sum',
        });
        const expectedSum = mx.sum(expectedNone);
        assert.ok(allclose(loss, expectedSum, 1e-6), 'BCE tiny probs reduction=sum');
      });
    });
  });

  // ---------------------------------------------------------------------------
  // test_l1_loss
  // ---------------------------------------------------------------------------
  describe('l1_loss', () => {
    const predictions = mx.array(new Float32Array([0.5, 0.2, 0.9, 0.0]));
    const targets = mx.array(new Float32Array([0.5, 0.2, 0.9, 0.0]));
    const expectedNone = mx.zeros([4]); // all zeros because predictions == targets
    const expectedSum = mx.sum(expectedNone);
    const expectedMean = mx.mean(expectedNone);

    it('reduction none', () => {
      const loss = mx.nn.losses.l1_loss(predictions, targets, {
        reduction: 'none',
      });
      assert.ok(mx.array_equal(loss, expectedNone).toArray()[0], 'l1_loss none');
    });

    it('reduction sum', () => {
      const loss = mx.nn.losses.l1_loss(predictions, targets, {
        reduction: 'sum',
      });
      assert.ok(mx.array_equal(loss, expectedSum).toArray()[0], 'l1_loss sum');
    });

    it('reduction mean', () => {
      const loss = mx.nn.losses.l1_loss(predictions, targets, {
        reduction: 'mean',
      });
      assert.ok(mx.array_equal(loss, expectedMean).toArray()[0], 'l1_loss mean');
    });
  });

  // ---------------------------------------------------------------------------
  // test_mse_loss
  // ---------------------------------------------------------------------------
  describe('mse_loss', () => {
    const predictions = mx.array(new Float32Array([0.5, 0.2, 0.9, 0.0]));
    const targets = mx.array(new Float32Array([0.7, 0.1, 0.8, 0.2]));
    const expectedNone = mx.array(new Float32Array([0.04, 0.01, 0.01, 0.04]));
    const expectedMean = mx.mean(expectedNone);
    const expectedSum = mx.sum(expectedNone);

    it('reduction none', () => {
      const loss = mx.nn.losses.mse_loss(predictions, targets, {
        reduction: 'none',
      });
      assert.ok(allclose(loss, expectedNone), 'mse_loss none');
    });

    it('reduction mean', () => {
      const loss = mx.nn.losses.mse_loss(predictions, targets, {
        reduction: 'mean',
      });
      assert.ok(allclose(loss, expectedMean), 'mse_loss mean');
    });

    it('reduction sum', () => {
      const loss = mx.nn.losses.mse_loss(predictions, targets, {
        reduction: 'sum',
      });
      assert.ok(allclose(loss, expectedSum), 'mse_loss sum');
    });
  });

  // ---------------------------------------------------------------------------
  // test_smooth_l1_loss
  // ---------------------------------------------------------------------------
  describe('smooth_l1_loss', () => {
    const predictions = mx.array(new Float32Array([1.5, 2.5, 0.5, 3.5]));
    const targets = mx.array(new Float32Array([1.0, 2.0, 0.5, 2.5]));
    const beta = 1.0;
    const expectedNone = mx.array(new Float32Array([0.125, 0.125, 0.0, 0.5]));
    const expectedSum = mx.sum(expectedNone);
    const expectedMean = mx.mean(expectedNone);

    it('reduction none', () => {
      const loss = mx.nn.losses.smooth_l1_loss(predictions, targets, {
        beta,
        reduction: 'none',
      });
      assert.ok(mx.array_equal(loss, expectedNone).toArray()[0], 'smooth_l1_loss none');
    });

    it('reduction sum', () => {
      const loss = mx.nn.losses.smooth_l1_loss(predictions, targets, {
        beta,
        reduction: 'sum',
      });
      assert.ok(
        Math.abs(item(loss) - item(expectedSum)) < 1e-5,
        'smooth_l1_loss sum',
      );
    });

    it('reduction mean', () => {
      const loss = mx.nn.losses.smooth_l1_loss(predictions, targets, {
        beta,
        reduction: 'mean',
      });
      assert.ok(
        Math.abs(item(loss) - item(expectedMean)) < 1e-5,
        'smooth_l1_loss mean',
      );
    });
  });

  // ---------------------------------------------------------------------------
  // test_nll_loss
  // ---------------------------------------------------------------------------
  describe('nll_loss', () => {
    const logits = mx.array(
      new Float32Array([0.0, -Infinity, -Infinity, 0.0]),
      [2, 2],
    );
    const targets = mx.array(new Int32Array([0, 1]));
    const expectedNone = mx.array(new Float32Array([0.0, 0.0]));
    const expectedMean = mx.mean(expectedNone);
    const expectedSum = mx.sum(expectedNone);

    it('reduction none', () => {
      const loss = mx.nn.losses.nll_loss(logits, targets, {
        reduction: 'none',
      });
      assert.ok(mx.array_equal(loss, expectedNone).toArray()[0], 'nll_loss none');
    });

    it('reduction mean', () => {
      const loss = mx.nn.losses.nll_loss(logits, targets, {
        reduction: 'mean',
      });
      assert.ok(
        Math.abs(item(loss) - item(expectedMean)) < 1e-5,
        'nll_loss mean',
      );
    });

    it('reduction sum', () => {
      const loss = mx.nn.losses.nll_loss(logits, targets, {
        reduction: 'sum',
      });
      assert.ok(
        Math.abs(item(loss) - item(expectedSum)) < 1e-5,
        'nll_loss sum',
      );
    });
  });

  // ---------------------------------------------------------------------------
  // test_gaussian_nll_loss
  // ---------------------------------------------------------------------------
  describe('gaussian_nll_loss', () => {
    const inputs = mx.array(
      new Float32Array([0.1, 0.2, 0.3, 0.4]),
      [2, 2],
    );
    const tgts = mx.array(
      new Float32Array([0.2, 0.1, 0.1, 0.2]),
      [2, 2],
    );
    const vars_ = mx.array(
      new Float32Array([0.1, 0.2, 0.3, 0.4]),
      [2, 2],
    );
    const expectedNone = mx.array(
      new Float32Array([-1.101293, -0.779719, -0.535320, -0.408145]),
      [2, 2],
    );
    const expectedNoneFull = mx.array(
      new Float32Array([-0.182354, 0.139220, 0.383619, 0.510793]),
      [2, 2],
    );

    it('reduction none, full=false', () => {
      const loss = mx.nn.losses.gaussian_nll_loss(inputs, tgts, vars_, {
        reduction: 'none',
      });
      assert.ok(allclose(loss, expectedNone), 'gaussian_nll_loss none full=false');
    });

    it('reduction mean, full=false', () => {
      const loss = mx.nn.losses.gaussian_nll_loss(inputs, tgts, vars_, {
        reduction: 'mean',
      });
      assert.ok(allclose(loss, mx.mean(expectedNone)), 'gaussian_nll_loss mean full=false');
    });

    it('reduction sum, full=false', () => {
      const loss = mx.nn.losses.gaussian_nll_loss(inputs, tgts, vars_, {
        reduction: 'sum',
      });
      assert.ok(allclose(loss, mx.sum(expectedNone)), 'gaussian_nll_loss sum full=false');
    });

    it('reduction none, full=true', () => {
      const loss = mx.nn.losses.gaussian_nll_loss(inputs, tgts, vars_, {
        full: true,
        reduction: 'none',
      });
      assert.ok(allclose(loss, expectedNoneFull), 'gaussian_nll_loss none full=true');
    });

    it('reduction mean, full=true', () => {
      const loss = mx.nn.losses.gaussian_nll_loss(inputs, tgts, vars_, {
        full: true,
        reduction: 'mean',
      });
      assert.ok(allclose(loss, mx.mean(expectedNoneFull)), 'gaussian_nll_loss mean full=true');
    });

    it('reduction sum, full=true', () => {
      const loss = mx.nn.losses.gaussian_nll_loss(inputs, tgts, vars_, {
        full: true,
        reduction: 'sum',
      });
      assert.ok(allclose(loss, mx.sum(expectedNoneFull)), 'gaussian_nll_loss sum full=true');
    });
  });

  // ---------------------------------------------------------------------------
  // test_kl_div_loss
  // ---------------------------------------------------------------------------
  describe('kl_div_loss', () => {
    const pLogits = mx.log(mx.array(new Float32Array([0.5, 0.5, 0.8, 0.2]), [2, 2]));
    const qLogits = mx.log(mx.array(new Float32Array([0.5, 0.5, 0.2, 0.8]), [2, 2]));
    const expectedNone = mx.array(new Float32Array([0.0, 0.831777]));

    it('reduction none', () => {
      const loss = mx.nn.losses.kl_div_loss(pLogits, qLogits, {
        reduction: 'none',
      });
      assert.ok(allclose(loss, expectedNone), 'kl_div_loss none');
    });

    it('reduction mean', () => {
      const loss = mx.nn.losses.kl_div_loss(pLogits, qLogits, {
        reduction: 'mean',
      });
      assert.ok(allclose(loss, mx.mean(expectedNone)), 'kl_div_loss mean');
    });

    it('reduction sum', () => {
      const loss = mx.nn.losses.kl_div_loss(pLogits, qLogits, {
        reduction: 'sum',
      });
      assert.ok(allclose(loss, mx.sum(expectedNone)), 'kl_div_loss sum');
    });
  });

  // ---------------------------------------------------------------------------
  // test_triplet_loss
  // ---------------------------------------------------------------------------
  describe('triplet_loss', () => {
    const anchors = mx.array(
      new Float32Array([1, 2, 3, 1, 2, 3]),
      [2, 3],
    );
    const positives = mx.array(
      new Float32Array([4, 5, 6, 0, -1, 2]),
      [2, 3],
    );
    const negatives = mx.array(
      new Float32Array([7, 8, 9, 3, 2, 3]),
      [2, 3],
    );
    const expectedNone = mx.array(new Float32Array([0, 2.31662]));

    it('reduction none', () => {
      const loss = mx.nn.losses.triplet_loss(anchors, positives, negatives, {
        reduction: 'none',
      });
      assert.ok(allclose(loss, expectedNone, 1e-4), 'triplet_loss none');
    });

    it('reduction mean', () => {
      const loss = mx.nn.losses.triplet_loss(anchors, positives, negatives, {
        reduction: 'mean',
      });
      assert.ok(allclose(loss, mx.mean(expectedNone), 1e-4), 'triplet_loss mean');
    });

    it('reduction sum', () => {
      const loss = mx.nn.losses.triplet_loss(anchors, positives, negatives, {
        reduction: 'sum',
      });
      assert.ok(allclose(loss, mx.sum(expectedNone), 1e-4), 'triplet_loss sum');
    });
  });

  // ---------------------------------------------------------------------------
  // test_hinge_loss
  // ---------------------------------------------------------------------------
  describe('hinge_loss', () => {
    it('computes hinge loss for ones vs zeros (mean reduction = 1.0)', () => {
      // inputs = ones(2,4), targets = zeros(2,4) => hinge = max(0, 1 - 1*0) = 1 each element
      // mean = 1.0
      const onesData = new Float32Array(8).fill(1.0);
      const zerosData = new Float32Array(8).fill(0.0);
      const inputs = mx.array(onesData, [2, 4]);
      const targets = mx.array(zerosData, [2, 4]);
      const loss = mx.nn.losses.hinge_loss(inputs, targets, {
        reduction: 'mean',
      });
      assert.ok(Math.abs(item(loss) - 1.0) < 1e-5, `hinge_loss mean expected 1.0, got ${item(loss)}`);
    });
  });

  // ---------------------------------------------------------------------------
  // test_huber_loss
  // ---------------------------------------------------------------------------
  describe('huber_loss', () => {
    it('computes huber loss for ones vs zeros (mean reduction = 0.5)', () => {
      // inputs = ones(2,4), targets = zeros(2,4)
      // diff = 1.0 which equals delta(1.0) exactly: huber = 0.5 * 1^2 = 0.5
      const onesData = new Float32Array(8).fill(1.0);
      const zerosData = new Float32Array(8).fill(0.0);
      const inputs = mx.array(onesData, [2, 4]);
      const targets = mx.array(zerosData, [2, 4]);
      const loss = mx.nn.losses.huber_loss(inputs, targets, {
        reduction: 'mean',
      });
      assert.ok(Math.abs(item(loss) - 0.5) < 1e-5, `huber_loss mean expected 0.5, got ${item(loss)}`);
    });
  });

  // ---------------------------------------------------------------------------
  // test_log_cosh_loss
  // ---------------------------------------------------------------------------
  describe('log_cosh_loss', () => {
    it('computes log cosh loss for ones vs zeros (mean ≈ 0.433781)', () => {
      const onesData = new Float32Array(8).fill(1.0);
      const zerosData = new Float32Array(8).fill(0.0);
      const inputs = mx.array(onesData, [2, 4]);
      const targets = mx.array(zerosData, [2, 4]);
      const loss = mx.nn.losses.log_cosh_loss(inputs, targets, {
        reduction: 'mean',
      });
      assert.ok(
        Math.abs(item(loss) - 0.433781) < 1e-5,
        `log_cosh_loss mean expected ≈0.433781, got ${item(loss)}`,
      );
    });
  });

  // ---------------------------------------------------------------------------
  // test_cosine_similarity_loss
  // ---------------------------------------------------------------------------
  describe('cosine_similarity_loss', () => {
    const embeddings1 = mx.array(
      new Float32Array([0.5, 0.5, 0.2, 0.9, 0.1, 0.3, 0.5, 0.5]),
      [2, 4],
    );
    const embeddings2 = mx.array(
      new Float32Array([0.6, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.4]),
      [2, 4],
    );
    const expectedNone = mx.array(new Float32Array([0.985344, 0.961074]));

    it('reduction none', () => {
      const loss = mx.nn.losses.cosine_similarity_loss(
        embeddings1,
        embeddings2,
        { reduction: 'none' },
      );
      assert.ok(allclose(loss, expectedNone, 1e-4), 'cosine_similarity_loss none');
    });

    it('reduction mean', () => {
      const loss = mx.nn.losses.cosine_similarity_loss(
        embeddings1,
        embeddings2,
        { reduction: 'mean' },
      );
      assert.ok(
        allclose(loss, mx.mean(expectedNone), 1e-4),
        'cosine_similarity_loss mean',
      );
    });

    it('reduction sum', () => {
      const loss = mx.nn.losses.cosine_similarity_loss(
        embeddings1,
        embeddings2,
        { reduction: 'sum' },
      );
      assert.ok(
        allclose(loss, mx.sum(expectedNone), 1e-4),
        'cosine_similarity_loss sum',
      );
    });
  });

  // ---------------------------------------------------------------------------
  // test_margin_ranking_loss
  // ---------------------------------------------------------------------------
  describe('margin_ranking_loss', () => {
    const inputs1 = mx.array(new Float32Array([-0.573409, -0.765166, -0.0638]));
    const inputs2 = mx.array(new Float32Array([0.75596, 0.225763, 0.256995]));
    const targets = mx.array(new Float32Array([1, 1, -1]));

    it('no margin (default margin=0)', () => {
      const loss = mx.nn.losses.margin_ranking_loss(inputs1, inputs2, targets, {
        reduction: 'none',
      });
      const expected = mx.array(new Float32Array([1.329369, 0.990929, 0.0]));
      assert.ok(allclose(loss, expected, 1e-4), 'margin_ranking_loss no margin');
    });

    it('with margin=0.5', () => {
      const loss = mx.nn.losses.margin_ranking_loss(inputs1, inputs2, targets, {
        margin: 0.5,
        reduction: 'none',
      });
      const expected = mx.array(new Float32Array([1.829369, 1.490929, 0.179205]));
      assert.ok(allclose(loss, expected, 1e-4), 'margin_ranking_loss margin=0.5');
    });
  });
});

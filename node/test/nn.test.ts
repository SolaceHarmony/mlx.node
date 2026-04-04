import { strict as assert } from 'node:assert';
import { nn, zeros, array } from '../src';

const toArray = (tensor: ReturnType<typeof array>): number[] => tensor.toArray() as number[];

describe('nn.init', () => {
  describe('he_uniform', () => {
    it('initializes weights with correct shape', () => {
      const initFn = nn.init.he_uniform();
      const weights = initFn(zeros([4, 3]));
      assert.deepEqual(weights.shape, [4, 3]);
    });

    it('uses fan_in mode by default', () => {
      const initFn = nn.init.he_uniform();
      const weights = initFn(zeros([4, 3]));
      const values = toArray(weights);

      // For fan_in mode with shape [4, 3]: fan_in = 3
      // limit = gain * sqrt(3.0 / fan) = 1.0 * sqrt(3.0 / 3) = 1.0
      // So values should be in [-1, 1)
      values.forEach(v => {
        assert.ok(v >= -1.0 && v < 1.0, `Value ${v} should be in range [-1, 1)`);
      });
    });

    it('respects fan_out mode', () => {
      const initFn = nn.init.he_uniform();
      const weights = initFn(zeros([4, 3]), 'fan_out');
      const values = toArray(weights);

      // For fan_out mode with shape [4, 3]: fan_out = 4
      // limit = gain * sqrt(3.0 / fan) = 1.0 * sqrt(3.0 / 4) = sqrt(0.75) ≈ 0.866
      // So values should be in [-0.866, 0.866)
      const expectedLimit = Math.sqrt(3.0 / 4);
      values.forEach(v => {
        assert.ok(
          v >= -expectedLimit && v < expectedLimit,
          `Value ${v} should be in range [${-expectedLimit}, ${expectedLimit})`
        );
      });
    });

    it('respects custom gain parameter', () => {
      const initFn = nn.init.he_uniform();
      const gain = 2.0;
      const weights = initFn(zeros([4, 3]), 'fan_in', gain);
      const values = toArray(weights);

      // For fan_in mode with shape [4, 3] and gain=2: fan_in = 3
      // limit = gain * sqrt(3.0 / fan) = 2.0 * sqrt(3.0 / 3) = 2.0
      // So values should be in [-2, 2)
      values.forEach(v => {
        assert.ok(v >= -2.0 && v < 2.0, `Value ${v} should be in range [-2, 2)`);
      });
    });

    it('handles 3D tensors correctly', () => {
      const initFn = nn.init.he_uniform();
      const weights = initFn(zeros([4, 3, 2]));
      const values = toArray(weights);

      // For 3D tensor [4, 3, 2]: fan_in = 2 * 3 = 6, fan_out = 4 * 3 = 12
      // Using fan_in mode: limit = sqrt(3.0 / 6) = sqrt(0.5) ≈ 0.707
      const expectedLimit = Math.sqrt(3.0 / 6);
      values.forEach(v => {
        assert.ok(
          v >= -expectedLimit && v < expectedLimit,
          `Value ${v} should be in range [${-expectedLimit}, ${expectedLimit})`
        );
      });
    });

    it('throws error for 1D tensors', () => {
      const initFn = nn.init.he_uniform();
      assert.throws(
        () => initFn(zeros([10])),
        /requires at least 2 dimensional input/
      );
    });

    it('throws error for invalid mode', () => {
      const initFn = nn.init.he_uniform();
      assert.throws(
        () => initFn(zeros([4, 3]), 'invalid_mode' as any),
        /Invalid mode/
      );
    });

    it('generates different values for each call', () => {
      const initFn = nn.init.he_uniform();
      const weights1 = initFn(zeros([3, 3]));
      const weights2 = initFn(zeros([3, 3]));
      const values1 = toArray(weights1);
      const values2 = toArray(weights2);

      // With high probability, at least one value should differ
      const allSame = values1.every((v, i) => v === values2[i]);
      assert.ok(!allSame, 'Random initializations should differ between calls');
    });
  });

  describe('glorot_uniform', () => {
    it('initializes weights with correct shape', () => {
      const initFn = nn.init.glorot_uniform();
      const weights = initFn(zeros([4, 3]));
      assert.deepEqual(weights.shape, [4, 3]);
    });

    it('uses correct distribution', () => {
      const initFn = nn.init.glorot_uniform();
      const weights = initFn(zeros([4, 3]));
      const values = toArray(weights);

      // For shape [4, 3]: fan_in = 3, fan_out = 4
      // limit = gain * sqrt(6.0 / (fan_in + fan_out)) = 1.0 * sqrt(6.0 / 7) ≈ 0.926
      const expectedLimit = Math.sqrt(6.0 / 7);
      values.forEach(v => {
        assert.ok(
          v >= -expectedLimit && v < expectedLimit,
          `Value ${v} should be in range [${-expectedLimit}, ${expectedLimit})`
        );
      });
    });

    it('respects custom gain parameter', () => {
      const initFn = nn.init.glorot_uniform();
      const gain = 3.0;
      const weights = initFn(zeros([4, 3]), gain);
      const values = toArray(weights);

      // limit = 3.0 * sqrt(6.0 / 7) ≈ 2.778
      const expectedLimit = 3.0 * Math.sqrt(6.0 / 7);
      values.forEach(v => {
        assert.ok(
          v >= -expectedLimit && v < expectedLimit,
          `Value ${v} should be in range [${-expectedLimit}, ${expectedLimit})`
        );
      });
    });
  });
});

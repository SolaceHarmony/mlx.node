import { describe, it } from 'mocha';
import * as assert from 'assert';
import * as core from '../../src/core';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

describe('batch 7: export ops', () => {
  const tmpDir = os.tmpdir();

  it('export_function and import_function roundtrip', () => {
    const f = (x: InstanceType<typeof core.MLXArray>) =>
      core.multiply(x, x);
    const x = core.array(new Float32Array([2, 3, 4]), [3]);
    const file = path.join(tmpDir, `mlx_export_${Date.now()}.mlxfn`);
    try {
      core.export_function(file, f as any, [x]);
      assert.ok(fs.existsSync(file));
      const imported = core.import_function(file);
      const results = imported(x);
      assert.ok(Array.isArray(results));
      const result = results[0];
      core.eval_op(result);
      assert.deepStrictEqual(result.toArray(), [4, 9, 16]);
    } finally {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    }
  });

  it('export_to_dot returns DOT string', () => {
    const a = core.array(new Float32Array([1, 2]), [2]);
    const b = core.add(a, a);
    const dot = core.export_to_dot(b);
    assert.ok(typeof dot === 'string');
    assert.ok(dot.includes('digraph'));
  });
});

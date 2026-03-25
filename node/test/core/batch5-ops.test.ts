import { describe, it } from 'mocha';
import * as assert from 'assert';
import * as core from '../../src/core';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

describe('batch 5: eval ops', () => {
  it('eval forces computation', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [4]);
    const b = core.add(a, a);
    // eval should not throw and should force computation
    core.eval_op(b);
    assert.deepStrictEqual(b.toArray(), [2, 4, 6, 8]);
  });

  it('eval accepts multiple arrays', () => {
    const a = core.array(new Float32Array([1, 2]), [2]);
    const b = core.array(new Float32Array([3, 4]), [2]);
    const c = core.add(a, b);
    const d = core.multiply(a, b);
    core.eval_op(c, d);
    assert.deepStrictEqual(c.toArray(), [4, 6]);
    assert.deepStrictEqual(d.toArray(), [3, 8]);
  });

  it('eval accepts array of arrays', () => {
    const a = core.array(new Float32Array([1, 2]), [2]);
    const b = core.add(a, a);
    core.eval_op([b]);
    assert.deepStrictEqual(b.toArray(), [2, 4]);
  });

  it('async_eval does not throw', () => {
    const a = core.array(new Float32Array([1, 2, 3]), [3]);
    const b = core.multiply(a, a);
    // async_eval should not throw
    core.async_eval(b);
    // Force sync after async to read result
    core.eval_op(b);
    assert.deepStrictEqual(b.toArray(), [1, 4, 9]);
  });
});

describe('batch 5: IO ops', () => {
  const tmpDir = os.tmpdir();

  it('save and load .npy roundtrip', () => {
    const a = core.array(new Float32Array([1.5, 2.5, 3.5, 4.5]), [2, 2]);
    const file = path.join(tmpDir, `mlx_test_${Date.now()}.npy`);
    try {
      core.save(file, a);
      assert.ok(fs.existsSync(file));
      const loaded = core.load(file) as InstanceType<typeof core.MLXArray>;
      assert.deepStrictEqual(loaded.shape, [2, 2]);
      const vals = loaded.toArray() as number[];
      assert.ok(Math.abs(vals[0] - 1.5) < 1e-5);
      assert.ok(Math.abs(vals[3] - 4.5) < 1e-5);
    } finally {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    }
  });

  it('save_safetensors and load .safetensors roundtrip', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const b = core.array(new Float32Array([5, 6, 7]), [3]);
    const file = path.join(tmpDir, `mlx_test_${Date.now()}.safetensors`);
    try {
      core.save_safetensors(file, { weight: a, bias: b });
      assert.ok(fs.existsSync(file));
      const result = core.load(file) as core.LoadResult;
      assert.ok(result.arrays);
      assert.ok(result.arrays.weight);
      assert.ok(result.arrays.bias);
      assert.deepStrictEqual(result.arrays.weight.shape, [2, 2]);
      assert.deepStrictEqual(result.arrays.bias.shape, [3]);
    } finally {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    }
  });

  it('save_safetensors with metadata', () => {
    const a = core.array(new Float32Array([1, 2]), [2]);
    const file = path.join(tmpDir, `mlx_test_${Date.now()}.safetensors`);
    try {
      core.save_safetensors(file, { data: a }, { metadata: { format: 'test', version: '1' } });
      assert.ok(fs.existsSync(file));
      const result = core.load(file) as core.LoadResult;
      assert.ok(result.metadata);
      // safetensors metadata should be preserved
      assert.strictEqual(result.metadata.format, 'test');
      assert.strictEqual(result.metadata.version, '1');
    } finally {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    }
  });

  it('save_gguf and load .gguf roundtrip', () => {
    const a = core.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    core.eval_op(a); // force eval before save
    const file = path.join(tmpDir, `mlx_test_${Date.now()}.gguf`);
    try {
      core.save_gguf(file, { tensor: a });
      assert.ok(fs.existsSync(file));
      const result = core.load(file) as core.LoadResult;
      assert.ok(result.arrays);
      assert.ok(result.arrays.tensor);
      assert.deepStrictEqual(result.arrays.tensor.shape, [2, 3]);
    } finally {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    }
  });

  it('load throws for non-existent file', () => {
    assert.throws(() => core.load('/nonexistent/path/file.npy'));
  });

  it('save throws for invalid path', () => {
    const a = core.array(new Float32Array([1]), [1]);
    assert.throws(() => core.save('/nonexistent/dir/file.npy', a));
  });
});

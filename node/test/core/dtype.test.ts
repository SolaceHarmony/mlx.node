import { strict as assert } from 'assert';
import mlx from '../../src';

describe('mlx.core.dtype', () => {
  it('exposes dtype constants', () => {
    const f32 = mlx.core.float32;
    assert.equal(f32.key, 'float32');
    assert.equal(f32.size, 4);
    assert.equal(f32.category, 'floating');
    assert.equal(f32.toString(), 'mlx.core.float32');
  });

  it('supports dtype.fromString lookup', () => {
    const fromStr = mlx.core.dtype.fromString('int8');
    assert.equal(fromStr.key, 'int8');
    assert.ok(fromStr.equals(mlx.core.int8));
  });

  it('provides dtype keys', () => {
    const keys = mlx.core.dtype.keys();
    assert.ok(Array.isArray(keys));
    assert.ok(keys.includes('float16'));
    const values = mlx.core.dtype.values();
    assert.equal(values.length, keys.length);
    const items = mlx.core.dtype.items();
    assert.equal(items.length, keys.length);
    assert.deepEqual(items[0][0], keys[0]);
    assert.ok(mlx.core.dtype.dir().includes('keys'));
    assert.ok(mlx.core.dtype.has('float32'));
    assert.equal(mlx.core.dtype.get('float32').key, 'float32');
  });

  it('exposes dtype categories', () => {
    const floating = mlx.core.floating;
    assert.equal(floating.name, 'floating');
    assert.equal(floating.toString(), 'DtypeCategory.floating');
    assert.ok(floating.equals(mlx.core.floating));
    const categoryKeys = mlx.core.dtype.categoryKeys();
    assert.ok(categoryKeys.includes('generic'));
    const categoryValues = mlx.core.dtype.categoryValues();
    assert.equal(categoryValues.length, categoryKeys.length);
    const categoryItems = mlx.core.dtype.categoryItems();
    assert.equal(categoryItems.length, categoryKeys.length);
  });

  it('supports issubdtype checks', () => {
    assert.ok(mlx.core.issubdtype(mlx.core.float32, mlx.core.float32));
    assert.ok(mlx.core.issubdtype(mlx.core.float32, mlx.core.floating));
    assert.ok(mlx.core.issubdtype(mlx.core.floating, mlx.core.number));
    assert.ok(!mlx.core.issubdtype(mlx.core.float32, mlx.core.integer));
  });
});

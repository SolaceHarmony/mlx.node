import { strict as assert } from 'assert';
import { core } from '../../src';

describe('core.dtype', () => {
  it('exposes dtype constants', () => {
    const f32 = core.float32;
    assert.equal(f32.key, 'float32');
    assert.equal(f32.size, 4);
    assert.equal(f32.category, 'floating');
    assert.equal(f32.toString(), 'mlx.core.float32');
  });

  it('supports dtype.fromString lookup', () => {
    const fromStr = core.dtype.fromString('int8');
    assert.equal(fromStr.key, 'int8');
    assert.ok(fromStr.equals(core.int8));
  });

  it('provides dtype keys', () => {
    const keys = core.dtype.keys();
    assert.ok(Array.isArray(keys));
    assert.ok(keys.includes('float16'));
    const values = core.dtype.values();
    assert.equal(values.length, keys.length);
    const items = core.dtype.items();
    assert.equal(items.length, keys.length);
    assert.deepEqual(items[0][0], keys[0]);
    assert.ok(core.dtype.dir().includes('keys'));
    assert.ok(core.dtype.has('float32'));
    assert.equal(core.dtype.get('float32').key, 'float32');
  });

  it('exposes dtype categories', () => {
    const floating = core.floating;
    assert.equal(floating.name, 'floating');
    assert.equal(floating.toString(), 'DtypeCategory.floating');
    assert.ok(floating.equals(core.floating));
    const categoryKeys = core.dtype.categoryKeys();
    assert.ok(categoryKeys.includes('generic'));
    const categoryValues = core.dtype.categoryValues();
    assert.equal(categoryValues.length, categoryKeys.length);
    const categoryItems = core.dtype.categoryItems();
    assert.equal(categoryItems.length, categoryKeys.length);
  });

  it('supports issubdtype checks', () => {
    assert.ok(core.issubdtype(core.float32, core.float32));
    assert.ok(core.issubdtype(core.float32, core.floating));
    assert.ok(core.issubdtype(core.floating, core.number));
    assert.ok(!core.issubdtype(core.float32, core.integer));
  });
});

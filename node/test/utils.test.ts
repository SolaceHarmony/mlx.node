import { strict as assert } from 'assert';
import { utils } from '../src';

const {
  tree_map,
  tree_map_with_path,
  tree_flatten,
  tree_unflatten,
  tree_reduce,
  tree_merge,
} = utils;

describe('mlx.utils tree helpers', () => {
  it('tree_map applies function to leaves', () => {
    const input = { a: 0, b: 1, c: 2 };
    const result = tree_map((x: number) => x + 1, input) as Record<string, number>;
    assert.deepStrictEqual(result, { a: 1, b: 2, c: 3 });
  });

  it('tree_map_with_path tracks traversal paths', () => {
    const structure = { model: [{ w: 0, b: 1 }, { w: 2, b: 3 }] };
    const seen: string[] = [];
    const output = tree_map_with_path((path: string, value: number) => {
      seen.push(path);
      return value;
    }, structure) as typeof structure;

    assert.deepStrictEqual(output, structure);
    assert.deepStrictEqual(seen, ['model.0.w', 'model.0.b', 'model.1.w', 'model.1.b']);
  });

  it('tree_map validates tree prefixes', () => {
    const base = { a: 1 };
    const mismatched = {};
    assert.throws(() => {
      tree_map((a: number, b: number) => a + b, base, mismatched);
    }, /Tree is not a valid prefix tree/);
  });

  it('tree_flatten and tree_unflatten round-trip', () => {
    const tree = [{ a: 1, b: 2 }, 'c'];
    const flattened = tree_flatten(tree) as Array<[string, unknown]>;
    assert.deepStrictEqual(flattened.map(([, value]) => value), [1, 2, 'c']);
    const restored = tree_unflatten(flattened);
    assert.deepStrictEqual(restored, tree);
  });

  it('tree_reduce aggregates leaves', () => {
    const tree = { a: [1, 2, 3], b: [4, 5] };
    const sum = tree_reduce((acc: number, value: number) => acc + value, tree, 0) as number;
    assert.equal(sum, 15);
  });

  it('tree_merge combines complementary trees', () => {
    const left = { a: 0 };
    const right = { b: 1 };
    const merged = tree_merge(left, right) as Record<string, number>;
    assert.deepStrictEqual(merged, { a: 0, b: 1 });

    assert.throws(() => {
      tree_merge(left, left);
    }, /no merge function was provided/);

    const stacked = tree_merge(
      { layers: [{ w: 1, b: 2 }] },
      { layers: [undefined, { w: 3, b: 4 }] },
    ) as { layers: Array<Record<string, number> | undefined> };

    assert.deepStrictEqual(stacked.layers?.length, 2);
    assert.deepStrictEqual(stacked.layers?.[0], { w: 1, b: 2 });
    assert.deepStrictEqual(stacked.layers?.[1], { w: 3, b: 4 });
  });
});

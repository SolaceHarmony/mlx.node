import { strict as assert } from 'assert';
import { utils } from '../src';

const {
  treeMap,
  treeMapWithPath,
  treeFlatten,
  treeUnflatten,
  treeReduce,
  treeMerge,
} = utils;

describe('mlx.utils tree helpers', () => {
  it('treeMap applies function to leaves', () => {
    const input = { a: 0, b: 1, c: 2 };
    const result = treeMap((x: number) => x + 1, input) as Record<string, number>;
    assert.deepStrictEqual(result, { a: 1, b: 2, c: 3 });
  });

  it('treeMapWithPath tracks traversal paths', () => {
    const structure = { model: [{ w: 0, b: 1 }, { w: 2, b: 3 }] };
    const seen: string[] = [];
    const output = treeMapWithPath((path: string, value: number) => {
      seen.push(path);
      return value;
    }, structure) as typeof structure;

    assert.deepStrictEqual(output, structure);
    assert.deepStrictEqual(seen, ['model.0.w', 'model.0.b', 'model.1.w', 'model.1.b']);
  });

  it('treeMap validates tree prefixes', () => {
    const base = { a: 1 };
    const mismatched = {};
    assert.throws(() => {
      treeMap((a: number, b: number) => a + b, base, mismatched);
    }, /Tree is not a valid prefix tree/);
  });

  it('treeFlatten and treeUnflatten round-trip', () => {
    const tree = [{ a: 1, b: 2 }, 'c'];
    const flattened = treeFlatten(tree) as Array<[string, unknown]>;
    assert.deepStrictEqual(flattened.map(([, value]) => value), [1, 2, 'c']);
    const restored = treeUnflatten(flattened);
    assert.deepStrictEqual(restored, tree);
  });

  it('treeReduce aggregates leaves', () => {
    const tree = { a: [1, 2, 3], b: [4, 5] };
    const sum = treeReduce((acc: number, value: number) => acc + value, tree, 0) as number;
    assert.equal(sum, 15);
  });

  it('treeMerge combines complementary trees', () => {
    const left = { a: 0 };
    const right = { b: 1 };
    const merged = treeMerge(left, right) as Record<string, number>;
    assert.deepStrictEqual(merged, { a: 0, b: 1 });

    assert.throws(() => {
      treeMerge(left, left);
    }, /no merge function was provided/);

    const stacked = treeMerge(
      { layers: [{ w: 1, b: 2 }] },
      { layers: [undefined, { w: 3, b: 4 }] },
    ) as { layers: Array<Record<string, number> | undefined> };

    assert.deepStrictEqual(stacked.layers?.length, 2);
    assert.deepStrictEqual(stacked.layers?.[0], { w: 1, b: 2 });
    assert.deepStrictEqual(stacked.layers?.[1], { w: 3, b: 4 });
  });
});

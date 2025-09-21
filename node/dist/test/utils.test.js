"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const assert_1 = require("assert");
const src_1 = require("../src");
const { treeMap, treeMapWithPath, treeFlatten, treeUnflatten, treeReduce, treeMerge, } = src_1.utils;
describe('mlx.utils tree helpers', () => {
    it('treeMap applies function to leaves', () => {
        const input = { a: 0, b: 1, c: 2 };
        const result = treeMap((x) => x + 1, input);
        assert_1.strict.deepStrictEqual(result, { a: 1, b: 2, c: 3 });
    });
    it('treeMapWithPath tracks traversal paths', () => {
        const structure = { model: [{ w: 0, b: 1 }, { w: 2, b: 3 }] };
        const seen = [];
        const output = treeMapWithPath((path, value) => {
            seen.push(path);
            return value;
        }, structure);
        assert_1.strict.deepStrictEqual(output, structure);
        assert_1.strict.deepStrictEqual(seen, ['model.0.w', 'model.0.b', 'model.1.w', 'model.1.b']);
    });
    it('treeMap validates tree prefixes', () => {
        const base = { a: 1 };
        const mismatched = {};
        assert_1.strict.throws(() => {
            treeMap((a, b) => a + b, base, mismatched);
        }, /Tree is not a valid prefix tree/);
    });
    it('treeFlatten and treeUnflatten round-trip', () => {
        const tree = [{ a: 1, b: 2 }, 'c'];
        const flattened = treeFlatten(tree);
        assert_1.strict.deepStrictEqual(flattened.map(([, value]) => value), [1, 2, 'c']);
        const restored = treeUnflatten(flattened);
        assert_1.strict.deepStrictEqual(restored, tree);
    });
    it('treeReduce aggregates leaves', () => {
        const tree = { a: [1, 2, 3], b: [4, 5] };
        const sum = treeReduce((acc, value) => acc + value, tree, 0);
        assert_1.strict.equal(sum, 15);
    });
    it('treeMerge combines complementary trees', () => {
        const left = { a: 0 };
        const right = { b: 1 };
        const merged = treeMerge(left, right);
        assert_1.strict.deepStrictEqual(merged, { a: 0, b: 1 });
        assert_1.strict.throws(() => {
            treeMerge(left, left);
        }, /no merge function was provided/);
        const stacked = treeMerge({ layers: [{ w: 1, b: 2 }] }, { layers: [undefined, { w: 3, b: 4 }] });
        assert_1.strict.deepStrictEqual(stacked.layers?.length, 2);
        assert_1.strict.deepStrictEqual(stacked.layers?.[0], { w: 1, b: 2 });
        assert_1.strict.deepStrictEqual(stacked.layers?.[1], { w: 3, b: 4 });
    });
});

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const node_assert_1 = require("node:assert");
const src_1 = require("../../src");
const toArray = (tensor) => tensor.toArray();
describe('core ops', () => {
    it('reshape matches element order', () => {
        const original = (0, src_1.array)([1, 2, 3, 4], [2, 2]);
        const reshaped = (0, src_1.reshape)(original, [4, 1]);
        node_assert_1.strict.deepEqual(reshaped.shape, [4, 1]);
        node_assert_1.strict.deepEqual(toArray(reshaped), [1, 2, 3, 4]);
    });
    it('transpose without axes reverses dims', () => {
        const original = (0, src_1.array)([1, 2, 3, 4], [2, 2]);
        const transposed = (0, src_1.transpose)(original);
        node_assert_1.strict.deepEqual(transposed.shape, [2, 2]);
        node_assert_1.strict.deepEqual(toArray(transposed), [1, 3, 2, 4]);
    });
    it('transpose with axes reorders dims explicitly', () => {
        const original = (0, src_1.array)([1, 2, 3, 4, 5, 6], [1, 2, 3]);
        const transposed = (0, src_1.transpose)(original, [2, 0, 1]);
        node_assert_1.strict.deepEqual(transposed.shape, [3, 1, 2]);
    });
    it('moveaxis shifts axes correctly', () => {
        const original = (0, src_1.array)([1, 2, 3, 4], [2, 2]);
        const moved = (0, src_1.moveaxis)(original, 0, 1);
        node_assert_1.strict.deepEqual(moved.shape, [2, 2]);
        node_assert_1.strict.deepEqual(toArray(moved), [1, 3, 2, 4]);
    });
    it('swapaxes exchanges two axes', () => {
        const original = (0, src_1.array)([1, 2, 3, 4, 5, 6], [2, 3]);
        const swapped = (0, src_1.swapaxes)(original, 0, 1);
        node_assert_1.strict.deepEqual(swapped.shape, [3, 2]);
        node_assert_1.strict.deepEqual(toArray(swapped), [1, 3, 5, 2, 4, 6]);
    });
    it('add performs elementwise addition', () => {
        const a = (0, src_1.array)([1, 2, 3], [3, 1]);
        const b = (0, src_1.array)([4, 5, 6], [3, 1]);
        const result = (0, src_1.add)(a, b);
        node_assert_1.strict.deepEqual(result.shape, [3, 1]);
        node_assert_1.strict.deepEqual(toArray(result), [5, 7, 9]);
    });
    it('multiply performs elementwise product', () => {
        const a = (0, src_1.array)([1, 2, 3], [3, 1]);
        const b = (0, src_1.array)([4, 5, 6], [3, 1]);
        const result = (0, src_1.multiply)(a, b);
        node_assert_1.strict.deepEqual(result.shape, [3, 1]);
        node_assert_1.strict.deepEqual(toArray(result), [4, 10, 18]);
    });
    it('where selects values elementwise', () => {
        const condition = (0, src_1.array)([1, 0, 1, 0], [4, 1]);
        const x = (0, src_1.array)([10, 20, 30, 40], [4, 1]);
        const y = (0, src_1.array)([100, 200, 300, 400], [4, 1]);
        const result = (0, src_1.where)(condition, x, y);
        node_assert_1.strict.deepEqual(result.shape, [4, 1]);
        node_assert_1.strict.deepEqual(toArray(result), [10, 200, 30, 400]);
    });
    it('operations respect explicit streams', async () => {
        const stream = (0, src_1.newStream)();
        await (0, src_1.withStream)(stream, () => {
            const a = (0, src_1.array)([1, 2, 3, 4], [2, 2]);
            const reshaped = (0, src_1.reshape)(a, [4, 1]);
            node_assert_1.strict.deepEqual(reshaped.shape, [4, 1]);
            const transposed = (0, src_1.transpose)(reshaped);
            node_assert_1.strict.deepEqual(transposed.shape, [1, 4]);
        });
    });
});

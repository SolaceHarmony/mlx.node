"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const assert_1 = require("assert");
const src_1 = __importDefault(require("../../src"));
describe('mlx.core.dtype', () => {
    it('exposes dtype constants', () => {
        const f32 = src_1.default.core.float32;
        assert_1.strict.equal(f32.key, 'float32');
        assert_1.strict.equal(f32.size, 4);
        assert_1.strict.equal(f32.category, 'floating');
        assert_1.strict.equal(f32.toString(), 'mlx.core.float32');
    });
    it('supports dtype.fromString lookup', () => {
        const fromStr = src_1.default.core.dtype.fromString('int8');
        assert_1.strict.equal(fromStr.key, 'int8');
        assert_1.strict.ok(fromStr.equals(src_1.default.core.int8));
    });
    it('provides dtype keys', () => {
        const keys = src_1.default.core.dtype.keys();
        assert_1.strict.ok(Array.isArray(keys));
        assert_1.strict.ok(keys.includes('float16'));
        const values = src_1.default.core.dtype.values();
        assert_1.strict.equal(values.length, keys.length);
        const items = src_1.default.core.dtype.items();
        assert_1.strict.equal(items.length, keys.length);
        assert_1.strict.deepEqual(items[0][0], keys[0]);
        assert_1.strict.ok(src_1.default.core.dtype.dir().includes('keys'));
        assert_1.strict.ok(src_1.default.core.dtype.has('float32'));
        assert_1.strict.equal(src_1.default.core.dtype.get('float32').key, 'float32');
    });
    it('exposes dtype categories', () => {
        const floating = src_1.default.core.floating;
        assert_1.strict.equal(floating.name, 'floating');
        assert_1.strict.equal(floating.toString(), 'DtypeCategory.floating');
        assert_1.strict.ok(floating.equals(src_1.default.core.floating));
        const categoryKeys = src_1.default.core.dtype.categoryKeys();
        assert_1.strict.ok(categoryKeys.includes('generic'));
        const categoryValues = src_1.default.core.dtype.categoryValues();
        assert_1.strict.equal(categoryValues.length, categoryKeys.length);
        const categoryItems = src_1.default.core.dtype.categoryItems();
        assert_1.strict.equal(categoryItems.length, categoryKeys.length);
    });
    it('supports issubdtype checks', () => {
        assert_1.strict.ok(src_1.default.core.issubdtype(src_1.default.core.float32, src_1.default.core.float32));
        assert_1.strict.ok(src_1.default.core.issubdtype(src_1.default.core.float32, src_1.default.core.floating));
        assert_1.strict.ok(src_1.default.core.issubdtype(src_1.default.core.floating, src_1.default.core.number));
        assert_1.strict.ok(!src_1.default.core.issubdtype(src_1.default.core.float32, src_1.default.core.integer));
    });
});

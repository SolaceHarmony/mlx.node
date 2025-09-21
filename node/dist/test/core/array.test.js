"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const assert_1 = require("assert");
const src_1 = __importStar(require("../../src"));
describe('mlx.core.array', () => {
    it('creates float32 array from typed array', () => {
        const data = new Float32Array([1, 2, 3, 4, 5, 6]);
        const arr = src_1.default.core.array(data, [2, 3], 'float32');
        assert_1.strict.ok(arr instanceof src_1.Array);
        assert_1.strict.deepEqual(arr.shape, [2, 3]);
        assert_1.strict.equal(arr.dtype, 'float32');
        assert_1.strict.deepEqual(Array.from(arr.toFloat32Array()), Array.from(data));
        assert_1.strict.deepEqual(arr.toArray(), Array.from(data));
    });
    it('throws when shape does not match data length', () => {
        const data = new Float32Array([1, 2, 3, 4]);
        assert_1.strict.throws(() => {
            src_1.default.core.array(data, [3, 3], 'float32');
        });
    });
    it('supports boolean arrays', () => {
        const arr = src_1.default.core.array([1, 0, 1, 1], [2, 2], 'bool');
        assert_1.strict.equal(arr.dtype, 'bool');
        assert_1.strict.deepEqual(arr.shape, [2, 2]);
        assert_1.strict.deepEqual(arr.toArray(), [true, false, true, true]);
        const typed = arr.toTypedArray();
        assert_1.strict.ok(typed instanceof Uint8Array);
    });
    it('supports int32 arrays from typed array', () => {
        const data = new Int32Array([1, -2, 3, -4]);
        const arr = src_1.default.core.array(data, [2, 2], 'int32');
        assert_1.strict.equal(arr.dtype, 'int32');
        assert_1.strict.deepEqual(arr.toArray(), [1, -2, 3, -4]);
        const typed = arr.toTypedArray();
        assert_1.strict.ok(typed instanceof Int32Array);
    });
    it('supports complex64 arrays', () => {
        const data = new Float32Array([1, 2, 3, 4]);
        const arr = src_1.default.core.array(data, [2], 'complex64');
        assert_1.strict.equal(arr.dtype, 'complex64');
        assert_1.strict.deepEqual(arr.shape, [2]);
        assert_1.strict.deepEqual(arr.toArray(), [
            [1, 2],
            [3, 4],
        ]);
        const typed = arr.toTypedArray();
        assert_1.strict.ok(typed instanceof Float32Array);
        assert_1.strict.deepEqual(Array.from(typed), Array.from(data));
    });
    it('exposes convenience entry points', () => {
        const data = new Float32Array([0, 1]);
        const viaNamedExport = (0, src_1.array)(data, [2], 'float32');
        const viaClass = src_1.Array.from(data, [2], 'float32');
        assert_1.strict.deepEqual(viaNamedExport.toArray(), [0, 1]);
        assert_1.strict.deepEqual(viaClass.toArray(), [0, 1]);
    });
    it('creates scalar-filled arrays', () => {
        const zeros = src_1.default.core.zeros([2, 3], 'float32');
        assert_1.strict.deepEqual(zeros.shape, [2, 3]);
        assert_1.strict.equal(zeros.dtype, 'float32');
        assert_1.strict.ok(zeros.toArray().every((value) => value === 0));
        const ones = src_1.default.core.ones([2, 2]);
        assert_1.strict.deepEqual(ones.shape, [2, 2]);
        assert_1.strict.equal(ones.dtype, 'float32');
        assert_1.strict.ok(ones.toArray().every((value) => value === 1));
        const full = src_1.default.core.full([3], 7.5, 'float64');
        assert_1.strict.deepEqual(full.shape, [3]);
        assert_1.strict.equal(full.dtype, 'float64');
        assert_1.strict.ok(full.toArray().every((value) => value === 7.5));
    });
    it('supports *_like helpers', () => {
        const base = src_1.default.core.zeros([4], 'float32');
        const zerosLike = src_1.default.core.zeros_like(base);
        assert_1.strict.deepEqual(zerosLike.shape, [4]);
        assert_1.strict.equal(zerosLike.dtype, 'float32');
        assert_1.strict.ok(zerosLike.toArray().every((value) => value === 0));
        const onesLike = src_1.default.core.ones_like(base);
        assert_1.strict.equal(onesLike.dtype, 'float32');
        assert_1.strict.ok(onesLike.toArray().every((value) => value === 1));
    });
});

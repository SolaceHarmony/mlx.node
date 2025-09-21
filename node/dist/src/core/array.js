"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.normalizeShapeInput = exports.MLXArray = void 0;
exports.array = array;
exports.zeros = zeros;
exports.ones = ones;
exports.full = full;
exports.zeros_like = zeros_like;
exports.ones_like = ones_like;
const addon_1 = __importDefault(require("../internal/addon"));
function normalizeShape(shape) {
    if (!Array.isArray(shape)) {
        throw new Error('Shape must be an array');
    }
    return shape.map((dim, index) => {
        if (!Number.isFinite(dim) || dim < 0) {
            throw new Error(`Invalid dimension at axis ${index}: ${dim}`);
        }
        return Math.trunc(dim);
    });
}
function elementCount(shape) {
    if (shape.length === 0) {
        return 1;
    }
    return shape.reduce((acc, dim) => acc * dim, 1);
}
function isTypedArray(input) {
    return ArrayBuffer.isView(input) && !(input instanceof DataView);
}
function inferDTypeFromTypedArray(array) {
    if (array instanceof Float32Array)
        return 'float32';
    if (array instanceof Float64Array)
        return 'float64';
    if (array instanceof Int8Array)
        return 'int8';
    if (array instanceof Uint8Array || array instanceof Uint8ClampedArray)
        return 'uint8';
    if (array instanceof Int16Array)
        return 'int16';
    if (array instanceof Uint16Array)
        return 'uint16';
    if (array instanceof Int32Array)
        return 'int32';
    if (array instanceof Uint32Array)
        return 'uint32';
    if (array instanceof BigInt64Array)
        return 'int64';
    if (array instanceof BigUint64Array)
        return 'uint64';
    throw new Error('Unsupported TypedArray input');
}
function createTypedArrayFromNumbers(data, dtype, elementCount) {
    if (dtype === 'complex64') {
        if (data.length !== elementCount * 2) {
            throw new Error('Complex64 data length must be elementCount * 2');
        }
        const output = new Float32Array(data.length);
        data.forEach((value, index) => {
            output[index] = Number(value);
        });
        return output;
    }
    if (data.length !== elementCount) {
        throw new Error('Data length does not match the provided shape');
    }
    switch (dtype) {
        case 'float32': {
            const output = new Float32Array(elementCount);
            data.forEach((value, index) => {
                output[index] = Number(value);
            });
            return output;
        }
        case 'float64': {
            const output = new Float64Array(elementCount);
            data.forEach((value, index) => {
                output[index] = Number(value);
            });
            return output;
        }
        case 'int8': {
            const output = new Int8Array(elementCount);
            data.forEach((value, index) => {
                output[index] = Number(value);
            });
            return output;
        }
        case 'uint8':
        case 'bool': {
            const output = new Uint8Array(elementCount);
            data.forEach((value, index) => {
                const n = Number(value);
                output[index] = dtype === 'bool' ? (n ? 1 : 0) : n;
            });
            return output;
        }
        case 'int16': {
            const output = new Int16Array(elementCount);
            data.forEach((value, index) => {
                output[index] = Number(value);
            });
            return output;
        }
        case 'uint16': {
            const output = new Uint16Array(elementCount);
            data.forEach((value, index) => {
                output[index] = Number(value);
            });
            return output;
        }
        case 'int32': {
            const output = new Int32Array(elementCount);
            data.forEach((value, index) => {
                output[index] = Number(value);
            });
            return output;
        }
        case 'uint32': {
            const output = new Uint32Array(elementCount);
            data.forEach((value, index) => {
                output[index] = Number(value);
            });
            return output;
        }
        case 'int64': {
            const output = new BigInt64Array(elementCount);
            data.forEach((value, index) => {
                output[index] = BigInt(value);
            });
            return output;
        }
        case 'uint64': {
            const output = new BigUint64Array(elementCount);
            data.forEach((value, index) => {
                const big = BigInt(value);
                if (big < 0n) {
                    throw new Error('uint64 values must be non-negative');
                }
                output[index] = big;
            });
            return output;
        }
        case 'float16':
        case 'bfloat16':
            throw new Error(`${dtype} requires a Uint16Array input. Provide encoded raw values.`);
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
}
function castTypedArrayToDType(array, dtype, elementCount) {
    const currentDType = inferDTypeFromTypedArray(array);
    if (currentDType === dtype) {
        if (dtype === 'complex64' && array.length !== elementCount * 2) {
            throw new Error('Complex64 data length must be elementCount * 2');
        }
        if (dtype !== 'complex64' && array.length !== elementCount) {
            throw new Error('Data length does not match the provided shape');
        }
        return array;
    }
    const numericArray = Array.from(array);
    return createTypedArrayFromNumbers(numericArray, dtype, elementCount);
}
function ensureTypedArray(data, dtype, elements) {
    if (isTypedArray(data)) {
        return castTypedArrayToDType(data, dtype, elements);
    }
    return createTypedArrayFromNumbers(data, dtype, elements);
}
class MLXArray {
    constructor(handle) {
        this.handle = handle;
    }
    static fromHandle(handle) {
        return new MLXArray(handle);
    }
    static from(data, shape, dtype = 'float32') {
        const normalizedShape = normalizeShape(shape);
        const elements = elementCount(normalizedShape);
        const typed = ensureTypedArray(data, dtype, elements);
        const handle = addon_1.default.Array.fromTypedArray(typed, normalizedShape, dtype);
        return MLXArray.fromHandle(handle);
    }
    static fromFloat32(data, shape) {
        return MLXArray.from(data, shape, 'float32');
    }
    get shape() {
        return globalThis.Array.from(this.handle.shape());
    }
    get dtype() {
        return this.handle.dtype();
    }
    eval() {
        this.handle.eval();
        return this;
    }
    toTypedArray() {
        return this.handle.toTypedArray();
    }
    toFloat32Array() {
        if (this.dtype !== 'float32') {
            throw new Error('Array dtype is not float32');
        }
        return this.handle.toFloat32Array();
    }
    toNative() {
        return this.handle;
    }
    toArray() {
        const data = this.toTypedArray();
        switch (this.dtype) {
            case 'bool':
                return Array.from(data, (value) => value !== 0);
            case 'complex64': {
                const values = data;
                const output = [];
                for (let i = 0; i < values.length; i += 2) {
                    output.push([values[i], values[i + 1]]);
                }
                return output;
            }
            default:
                return Array.from(data);
        }
    }
}
exports.MLXArray = MLXArray;
function array(data, shape, dtype = 'float32') {
    return MLXArray.from(data, shape, dtype);
}
const normalizeShapeInput = (shape) => Array.from(shape, (dim) => {
    if (!Number.isFinite(dim)) {
        throw new Error('Shape entries must be finite numbers');
    }
    return Math.trunc(dim);
});
exports.normalizeShapeInput = normalizeShapeInput;
function zeros(shape, dtype = 'float32') {
    return MLXArray.fromHandle(addon_1.default.zeros((0, exports.normalizeShapeInput)(shape), dtype));
}
function ones(shape, dtype = 'float32') {
    return MLXArray.fromHandle(addon_1.default.ones((0, exports.normalizeShapeInput)(shape), dtype));
}
function full(shape, value, dtype = 'float32') {
    return MLXArray.fromHandle(addon_1.default.full((0, exports.normalizeShapeInput)(shape), value, dtype));
}
function zeros_like(base) {
    return MLXArray.fromHandle(addon_1.default.zeros_like(base.toNative()));
}
function ones_like(base) {
    return MLXArray.fromHandle(addon_1.default.ones_like(base.toNative()));
}
exports.default = MLXArray;

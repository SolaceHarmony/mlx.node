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
exports.float64 = exports.float32 = exports.bfloat16 = exports.float16 = exports.uint64 = exports.uint32 = exports.uint16 = exports.uint8 = exports.int64 = exports.int32 = exports.int16 = exports.int8 = exports.bool = exports.issubdtype = exports.dtypeCategoryItems = exports.dtypeCategoryValues = exports.dtypeCategoryKeys = exports.dtypeGet = exports.dtypeHas = exports.dtypeDir = exports.dtypeItems = exports.dtypeValues = exports.dtypeKeys = exports.dtypeFromString = exports.Dtype = exports.dtype = exports.where = exports.multiply = exports.add = exports.swapaxes = exports.moveaxis = exports.transpose = exports.reshape = exports.withStream = exports.stream = exports.streamContext = exports.synchronize = exports.setDefaultStream = exports.newStream = exports.defaultStream = exports.Stream = exports.device = exports.full = exports.ones_like = exports.ones = exports.zeros_like = exports.zeros = exports.array = exports.Array = exports.MLXArray = void 0;
exports.generic = exports.number = exports.integer = exports.unsignedinteger = exports.signedinteger = exports.inexact = exports.floating = exports.complexfloating = exports.complex64 = void 0;
const array_1 = __importStar(require("./array"));
exports.MLXArray = array_1.default;
exports.Array = array_1.default;
Object.defineProperty(exports, "array", { enumerable: true, get: function () { return array_1.array; } });
Object.defineProperty(exports, "zeros", { enumerable: true, get: function () { return array_1.zeros; } });
Object.defineProperty(exports, "zeros_like", { enumerable: true, get: function () { return array_1.zeros_like; } });
Object.defineProperty(exports, "ones", { enumerable: true, get: function () { return array_1.ones; } });
Object.defineProperty(exports, "ones_like", { enumerable: true, get: function () { return array_1.ones_like; } });
Object.defineProperty(exports, "full", { enumerable: true, get: function () { return array_1.full; } });
const dtype_1 = __importStar(require("./dtype"));
exports.dtype = dtype_1.default;
Object.defineProperty(exports, "Dtype", { enumerable: true, get: function () { return dtype_1.Dtype; } });
Object.defineProperty(exports, "dtypeFromString", { enumerable: true, get: function () { return dtype_1.fromString; } });
Object.defineProperty(exports, "dtypeKeys", { enumerable: true, get: function () { return dtype_1.keys; } });
Object.defineProperty(exports, "dtypeValues", { enumerable: true, get: function () { return dtype_1.values; } });
Object.defineProperty(exports, "dtypeItems", { enumerable: true, get: function () { return dtype_1.items; } });
Object.defineProperty(exports, "dtypeDir", { enumerable: true, get: function () { return dtype_1.dir; } });
Object.defineProperty(exports, "dtypeHas", { enumerable: true, get: function () { return dtype_1.has; } });
Object.defineProperty(exports, "dtypeGet", { enumerable: true, get: function () { return dtype_1.get; } });
Object.defineProperty(exports, "issubdtype", { enumerable: true, get: function () { return dtype_1.issubdtype; } });
Object.defineProperty(exports, "dtypeCategoryKeys", { enumerable: true, get: function () { return dtype_1.categoryKeys; } });
Object.defineProperty(exports, "dtypeCategoryValues", { enumerable: true, get: function () { return dtype_1.categoryValues; } });
Object.defineProperty(exports, "dtypeCategoryItems", { enumerable: true, get: function () { return dtype_1.categoryItems; } });
Object.defineProperty(exports, "bool", { enumerable: true, get: function () { return dtype_1.bool; } });
Object.defineProperty(exports, "int8", { enumerable: true, get: function () { return dtype_1.int8; } });
Object.defineProperty(exports, "int16", { enumerable: true, get: function () { return dtype_1.int16; } });
Object.defineProperty(exports, "int32", { enumerable: true, get: function () { return dtype_1.int32; } });
Object.defineProperty(exports, "int64", { enumerable: true, get: function () { return dtype_1.int64; } });
Object.defineProperty(exports, "uint8", { enumerable: true, get: function () { return dtype_1.uint8; } });
Object.defineProperty(exports, "uint16", { enumerable: true, get: function () { return dtype_1.uint16; } });
Object.defineProperty(exports, "uint32", { enumerable: true, get: function () { return dtype_1.uint32; } });
Object.defineProperty(exports, "uint64", { enumerable: true, get: function () { return dtype_1.uint64; } });
Object.defineProperty(exports, "float16", { enumerable: true, get: function () { return dtype_1.float16; } });
Object.defineProperty(exports, "bfloat16", { enumerable: true, get: function () { return dtype_1.bfloat16; } });
Object.defineProperty(exports, "float32", { enumerable: true, get: function () { return dtype_1.float32; } });
Object.defineProperty(exports, "float64", { enumerable: true, get: function () { return dtype_1.float64; } });
Object.defineProperty(exports, "complex64", { enumerable: true, get: function () { return dtype_1.complex64; } });
Object.defineProperty(exports, "complexfloating", { enumerable: true, get: function () { return dtype_1.complexfloating; } });
Object.defineProperty(exports, "floating", { enumerable: true, get: function () { return dtype_1.floating; } });
Object.defineProperty(exports, "inexact", { enumerable: true, get: function () { return dtype_1.inexact; } });
Object.defineProperty(exports, "signedinteger", { enumerable: true, get: function () { return dtype_1.signedinteger; } });
Object.defineProperty(exports, "unsignedinteger", { enumerable: true, get: function () { return dtype_1.unsignedinteger; } });
Object.defineProperty(exports, "integer", { enumerable: true, get: function () { return dtype_1.integer; } });
Object.defineProperty(exports, "number", { enumerable: true, get: function () { return dtype_1.number; } });
Object.defineProperty(exports, "generic", { enumerable: true, get: function () { return dtype_1.generic; } });
const deviceModule = __importStar(require("./device"));
exports.device = deviceModule;
const stream_1 = require("./stream");
Object.defineProperty(exports, "Stream", { enumerable: true, get: function () { return stream_1.MLXStream; } });
Object.defineProperty(exports, "defaultStream", { enumerable: true, get: function () { return stream_1.defaultStream; } });
Object.defineProperty(exports, "newStream", { enumerable: true, get: function () { return stream_1.newStream; } });
Object.defineProperty(exports, "setDefaultStream", { enumerable: true, get: function () { return stream_1.setDefaultStream; } });
Object.defineProperty(exports, "synchronize", { enumerable: true, get: function () { return stream_1.synchronize; } });
Object.defineProperty(exports, "streamContext", { enumerable: true, get: function () { return stream_1.streamContext; } });
Object.defineProperty(exports, "stream", { enumerable: true, get: function () { return stream_1.streamContext; } });
Object.defineProperty(exports, "withStream", { enumerable: true, get: function () { return stream_1.withStream; } });
const ops_1 = require("./ops");
Object.defineProperty(exports, "reshape", { enumerable: true, get: function () { return ops_1.reshape; } });
Object.defineProperty(exports, "transpose", { enumerable: true, get: function () { return ops_1.transpose; } });
Object.defineProperty(exports, "moveaxis", { enumerable: true, get: function () { return ops_1.moveaxis; } });
Object.defineProperty(exports, "swapaxes", { enumerable: true, get: function () { return ops_1.swapaxes; } });
Object.defineProperty(exports, "add", { enumerable: true, get: function () { return ops_1.add; } });
Object.defineProperty(exports, "multiply", { enumerable: true, get: function () { return ops_1.multiply; } });
Object.defineProperty(exports, "where", { enumerable: true, get: function () { return ops_1.where; } });
const core = {
    array: array_1.array,
    zeros: array_1.zeros,
    zeros_like: array_1.zeros_like,
    ones: array_1.ones,
    ones_like: array_1.ones_like,
    full: array_1.full,
    Array: array_1.default,
    Stream: stream_1.MLXStream,
    defaultStream: stream_1.defaultStream,
    newStream: stream_1.newStream,
    setDefaultStream: stream_1.setDefaultStream,
    synchronize: stream_1.synchronize,
    streamContext: stream_1.streamContext,
    stream: stream_1.streamContext,
    withStream: stream_1.withStream,
    reshape: ops_1.reshape,
    transpose: ops_1.transpose,
    moveaxis: ops_1.moveaxis,
    swapaxes: ops_1.swapaxes,
    add: ops_1.add,
    multiply: ops_1.multiply,
    where: ops_1.where,
    device: deviceModule,
    Dtype: dtype_1.Dtype,
    dtype: dtype_1.default,
    dtypeFromString: dtype_1.fromString,
    dtypeKeys: dtype_1.keys,
    dtypeValues: dtype_1.values,
    dtypeItems: dtype_1.items,
    dtypeDir: dtype_1.dir,
    dtypeHas: dtype_1.has,
    dtypeGet: dtype_1.get,
    dtypeCategoryKeys: dtype_1.categoryKeys,
    dtypeCategoryValues: dtype_1.categoryValues,
    dtypeCategoryItems: dtype_1.categoryItems,
    issubdtype: dtype_1.issubdtype,
    bool: dtype_1.bool,
    int8: dtype_1.int8,
    int16: dtype_1.int16,
    int32: dtype_1.int32,
    int64: dtype_1.int64,
    uint8: dtype_1.uint8,
    uint16: dtype_1.uint16,
    uint32: dtype_1.uint32,
    uint64: dtype_1.uint64,
    float16: dtype_1.float16,
    bfloat16: dtype_1.bfloat16,
    float32: dtype_1.float32,
    float64: dtype_1.float64,
    complex64: dtype_1.complex64,
    complexfloating: dtype_1.complexfloating,
    floating: dtype_1.floating,
    inexact: dtype_1.inexact,
    signedinteger: dtype_1.signedinteger,
    unsignedinteger: dtype_1.unsignedinteger,
    integer: dtype_1.integer,
    number: dtype_1.number,
    generic: dtype_1.generic,
};
exports.default = core;

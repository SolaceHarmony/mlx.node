"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.generic = exports.number = exports.integer = exports.unsignedinteger = exports.signedinteger = exports.inexact = exports.floating = exports.complexfloating = exports.complex64 = exports.float64 = exports.float32 = exports.bfloat16 = exports.float16 = exports.uint64 = exports.uint32 = exports.uint16 = exports.uint8 = exports.int64 = exports.int32 = exports.int16 = exports.int8 = exports.bool = exports.categoryItems = exports.categoryValues = exports.categoryKeys = exports.issubdtype = exports.get = exports.has = exports.dir = exports.items = exports.values = exports.keys = exports.fromString = exports.Dtype = void 0;
const addon_1 = __importDefault(require("../internal/addon"));
const nativeDtype = addon_1.default.dtype;
const Dtype = addon_1.default.Dtype;
exports.Dtype = Dtype;
const issubdtype = addon_1.default.issubdtype;
exports.issubdtype = issubdtype;
const fromString = (key) => nativeDtype.fromString(key);
exports.fromString = fromString;
const dtypeConstants = {
    bool: nativeDtype.bool,
    int8: nativeDtype.int8,
    int16: nativeDtype.int16,
    int32: nativeDtype.int32,
    int64: nativeDtype.int64,
    uint8: nativeDtype.uint8,
    uint16: nativeDtype.uint16,
    uint32: nativeDtype.uint32,
    uint64: nativeDtype.uint64,
    float16: nativeDtype.float16,
    bfloat16: nativeDtype.bfloat16,
    float32: nativeDtype.float32,
    float64: nativeDtype.float64,
    complex64: nativeDtype.complex64,
};
const dtypeKeyList = Object.freeze(Object.keys(dtypeConstants));
const categoryConstants = {
    complexfloating: nativeDtype.complexfloating,
    floating: nativeDtype.floating,
    inexact: nativeDtype.inexact,
    signedinteger: nativeDtype.signedinteger,
    unsignedinteger: nativeDtype.unsignedinteger,
    integer: nativeDtype.integer,
    number: nativeDtype.number,
    generic: nativeDtype.generic,
};
const categoryKeyList = Object.freeze(Object.keys(categoryConstants));
const keys = () => [...dtypeKeyList];
exports.keys = keys;
const values = () => dtypeKeyList.map((key) => dtypeConstants[key]);
exports.values = values;
const items = () => dtypeKeyList.map((key) => [key, dtypeConstants[key]]);
exports.items = items;
const categoryKeys = () => [...categoryKeyList];
exports.categoryKeys = categoryKeys;
const categoryValues = () => categoryKeyList.map((key) => categoryConstants[key]);
exports.categoryValues = categoryValues;
const categoryItems = () => categoryKeyList.map((key) => [key, categoryConstants[key]]);
exports.categoryItems = categoryItems;
const has = (key) => Object.prototype.hasOwnProperty.call(dtypeConstants, key);
exports.has = has;
const get = (key) => dtypeConstants[key];
exports.get = get;
const dir = () => [
    'Dtype',
    'fromString',
    'keys',
    'values',
    'items',
    'dir',
    'has',
    'get',
    'issubdtype',
    'categoryKeys',
    'categoryValues',
    'categoryItems',
    ...dtypeKeyList,
    ...categoryKeyList,
];
exports.dir = dir;
exports.bool = dtypeConstants.bool, exports.int8 = dtypeConstants.int8, exports.int16 = dtypeConstants.int16, exports.int32 = dtypeConstants.int32, exports.int64 = dtypeConstants.int64, exports.uint8 = dtypeConstants.uint8, exports.uint16 = dtypeConstants.uint16, exports.uint32 = dtypeConstants.uint32, exports.uint64 = dtypeConstants.uint64, exports.float16 = dtypeConstants.float16, exports.bfloat16 = dtypeConstants.bfloat16, exports.float32 = dtypeConstants.float32, exports.float64 = dtypeConstants.float64, exports.complex64 = dtypeConstants.complex64;
exports.complexfloating = categoryConstants.complexfloating, exports.floating = categoryConstants.floating, exports.inexact = categoryConstants.inexact, exports.signedinteger = categoryConstants.signedinteger, exports.unsignedinteger = categoryConstants.unsignedinteger, exports.integer = categoryConstants.integer, exports.number = categoryConstants.number, exports.generic = categoryConstants.generic;
exports.default = {
    Dtype,
    fromString,
    keys,
    values,
    items,
    dir,
    has,
    get,
    issubdtype,
    categoryKeys,
    categoryValues,
    categoryItems,
    ...dtypeConstants,
    ...categoryConstants,
};

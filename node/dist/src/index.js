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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.where = exports.multiply = exports.add = exports.swapaxes = exports.moveaxis = exports.transpose = exports.reshape = exports.device = exports.withStream = exports.stream = exports.streamContext = exports.synchronize = exports.setDefaultStream = exports.newStream = exports.defaultStream = exports.full = exports.ones_like = exports.ones = exports.zeros_like = exports.zeros = exports.issubdtype = exports.Stream = exports.Array = exports.array = exports.react = exports.streaming = exports.utils = exports.core = exports.native = void 0;
const addon_1 = __importDefault(require("./internal/addon"));
const utils = __importStar(require("./utils"));
exports.utils = utils;
const core = __importStar(require("./core"));
exports.core = core;
const streaming = __importStar(require("./streaming"));
exports.streaming = streaming;
const react = __importStar(require("./react"));
exports.react = react;
exports.native = {
    hello: () => addon_1.default.hello(),
};
exports.array = core.array;
exports.Array = core.Array;
exports.Stream = core.Stream;
exports.issubdtype = core.issubdtype;
exports.zeros = core.zeros;
exports.zeros_like = core.zeros_like;
exports.ones = core.ones;
exports.ones_like = core.ones_like;
exports.full = core.full;
exports.defaultStream = core.defaultStream;
exports.newStream = core.newStream;
exports.setDefaultStream = core.setDefaultStream;
exports.synchronize = core.synchronize;
exports.streamContext = core.streamContext;
exports.stream = core.stream;
exports.withStream = core.withStream;
exports.device = core.device;
exports.reshape = core.reshape;
exports.transpose = core.transpose;
exports.moveaxis = core.moveaxis;
exports.swapaxes = core.swapaxes;
exports.add = core.add;
exports.multiply = core.multiply;
exports.where = core.where;
exports.default = {
    native: exports.native,
    core,
    utils,
    react,
    streaming,
    device: exports.device,
    array: exports.array,
    Array: exports.Array,
    issubdtype: exports.issubdtype,
    zeros: exports.zeros,
    zeros_like: exports.zeros_like,
    ones: exports.ones,
    ones_like: exports.ones_like,
    full: exports.full,
    Stream: exports.Stream,
    defaultStream: exports.defaultStream,
    newStream: exports.newStream,
    setDefaultStream: exports.setDefaultStream,
    synchronize: exports.synchronize,
    streamContext: exports.streamContext,
    stream: exports.stream,
    withStream: exports.withStream,
    reshape: exports.reshape,
    transpose: exports.transpose,
    moveaxis: exports.moveaxis,
    swapaxes: exports.swapaxes,
    add: exports.add,
    multiply: exports.multiply,
    where: exports.where,
};

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
exports.reshape = reshape;
exports.transpose = transpose;
exports.moveaxis = moveaxis;
exports.swapaxes = swapaxes;
exports.add = add;
exports.multiply = multiply;
exports.where = where;
const addon_1 = __importDefault(require("../internal/addon"));
const array_1 = __importStar(require("./array"));
const stream_1 = require("./stream");
function toNativeHandle(tensor) {
    return tensor.toNative();
}
function normalizeStream(stream) {
    if (stream == null) {
        return undefined;
    }
    return (0, stream_1.toNativeStreamArgument)(stream);
}
function normalizeAxes(axes) {
    return Array.from(axes, (axis) => {
        if (!Number.isInteger(axis)) {
            throw new Error('Axis indices must be integers');
        }
        return Number(axis);
    });
}
function normalizeAxisSpec(value, name) {
    if (Array.isArray(value)) {
        return normalizeAxes(value);
    }
    if (!Number.isInteger(value)) {
        throw new Error(`${name} must be an integer or array of integers`);
    }
    return [Number(value)];
}
function appendStreamArg(args, stream) {
    const native = normalizeStream(stream);
    if (native !== undefined) {
        args.push(native);
    }
}
function reshape(tensor, shape, options) {
    const normalizedShape = (0, array_1.normalizeShapeInput)(shape);
    const args = [toNativeHandle(tensor), normalizedShape];
    appendStreamArg(args, options?.stream);
    const handle = addon_1.default.reshape(...args);
    return array_1.default.fromHandle(handle);
}
function transpose(tensor, axesOrOptions, maybeOptions) {
    let axes;
    let options;
    if (Array.isArray(axesOrOptions)) {
        axes = normalizeAxes(axesOrOptions);
        options = maybeOptions ?? undefined;
    }
    else if (axesOrOptions && 'stream' in axesOrOptions) {
        options = axesOrOptions;
    }
    else if (axesOrOptions != null) {
        throw new Error('transpose axes must be an array of integers');
    }
    else {
        options = maybeOptions;
    }
    const args = [toNativeHandle(tensor)];
    if (axes) {
        args.push(axes);
    }
    appendStreamArg(args, options?.stream);
    const handle = addon_1.default.transpose(...args);
    return array_1.default.fromHandle(handle);
}
function moveaxis(tensor, source, destination, options) {
    const src = normalizeAxisSpec(source, 'source axes');
    const dst = normalizeAxisSpec(destination, 'destination axes');
    const args = [toNativeHandle(tensor), src, dst];
    appendStreamArg(args, options?.stream);
    const handle = addon_1.default.moveaxis(...args);
    return array_1.default.fromHandle(handle);
}
function swapaxes(tensor, axis1, axis2, options) {
    if (!Number.isInteger(axis1) || !Number.isInteger(axis2)) {
        throw new Error('swapaxes expects integer axis indices');
    }
    const args = [
        toNativeHandle(tensor),
        Number(axis1),
        Number(axis2),
    ];
    appendStreamArg(args, options?.stream);
    const handle = addon_1.default.swapaxes(...args);
    return array_1.default.fromHandle(handle);
}
function binaryOp(name, a, b, options) {
    const args = [toNativeHandle(a), toNativeHandle(b)];
    appendStreamArg(args, options?.stream);
    const handle = addon_1.default[name](...args);
    return array_1.default.fromHandle(handle);
}
function add(a, b, options) {
    return binaryOp('add', a, b, options);
}
function multiply(a, b, options) {
    return binaryOp('multiply', a, b, options);
}
function where(condition, onTrue, onFalse, options) {
    const args = [
        toNativeHandle(condition),
        toNativeHandle(onTrue),
        toNativeHandle(onFalse),
    ];
    appendStreamArg(args, options?.stream);
    const handle = addon_1.default.where(...args);
    return array_1.default.fromHandle(handle);
}

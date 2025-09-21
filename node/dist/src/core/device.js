"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.gpu = exports.cpu = void 0;
exports.normalizeDevice = normalizeDevice;
exports.device = device;
exports.cpu = Object.freeze({ type: 'cpu', index: 0 });
const gpu = (index = 0) => ({ type: 'gpu', index });
exports.gpu = gpu;
function normalizeDevice(device) {
    if (!device) {
        return { ...exports.cpu };
    }
    if (typeof device === 'string') {
        return { type: device, index: 0 };
    }
    if ('type' in device) {
        return {
            type: device.type,
            index: device.index ?? 0,
        };
    }
    throw new TypeError('Unsupported device specification');
}
function device(type, index = 0) {
    return { type, index };
}

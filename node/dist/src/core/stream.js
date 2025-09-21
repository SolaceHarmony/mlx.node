"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MLXStream = void 0;
exports.isStream = isStream;
exports.defaultStream = defaultStream;
exports.newStream = newStream;
exports.setDefaultStream = setDefaultStream;
exports.synchronize = synchronize;
exports.streamContext = streamContext;
exports.withStream = withStream;
exports.toNativeStreamArgument = toNativeStreamArgument;
const addon_1 = __importDefault(require("../internal/addon"));
const device_1 = require("./device");
class MLXStream {
    constructor(handle) {
        this.handle = handle;
    }
    static fromHandle(handle) {
        return new MLXStream(handle);
    }
    get index() {
        return this.handle.index;
    }
    get device() {
        const native = this.handle.device;
        return {
            type: native.type,
            index: native.index ?? 0,
        };
    }
    equals(other) {
        return this.handle.equals(other.toNative());
    }
    toString() {
        return this.handle.toString();
    }
    toNative() {
        return this.handle;
    }
}
exports.MLXStream = MLXStream;
function isStream(value) {
    return value instanceof MLXStream;
}
function toNativeStreamOrDevice(stream) {
    if (isStream(stream)) {
        return stream.toNative();
    }
    return (0, device_1.normalizeDevice)(stream);
}
function createContext(stream) {
    return addon_1.default.stream(toNativeStreamOrDevice(stream));
}
function defaultStream(device) {
    const arg = device == null ? undefined : (0, device_1.normalizeDevice)(device);
    const handle = arg ? addon_1.default.default_stream(arg) : addon_1.default.default_stream();
    return MLXStream.fromHandle(handle);
}
function newStream(device) {
    const arg = device == null ? undefined : (0, device_1.normalizeDevice)(device);
    const handle = arg ? addon_1.default.new_stream(arg) : addon_1.default.new_stream();
    return MLXStream.fromHandle(handle);
}
function setDefaultStream(stream) {
    addon_1.default.set_default_stream(stream.toNative());
}
function synchronize(stream) {
    if (stream == null) {
        addon_1.default.synchronize();
        return;
    }
    if (!isStream(stream)) {
        const native = defaultStream(stream);
        addon_1.default.synchronize(native.toNative());
        return;
    }
    addon_1.default.synchronize(stream.toNative());
}
function streamContext(stream) {
    return createContext(stream);
}
async function withStream(stream, fn) {
    const ctx = createContext(stream);
    ctx.enter();
    let exited = false;
    const exit = () => {
        if (!exited) {
            exited = true;
            ctx.exit();
        }
    };
    try {
        const result = fn();
        if (result instanceof Promise) {
            try {
                const awaited = await result;
                exit();
                return awaited;
            }
            catch (error) {
                exit();
                throw error;
            }
        }
        exit();
        return result;
    }
    catch (error) {
        exit();
        throw error;
    }
}
function toNativeStreamArgument(stream) {
    if (stream == null) {
        return undefined;
    }
    if (isStream(stream)) {
        return stream.toNative();
    }
    return (0, device_1.normalizeDevice)(stream);
}

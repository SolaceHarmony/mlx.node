"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.tensorToFrames = tensorToFrames;
exports.tensorsToFrameStream = tensorsToFrameStream;
const encoding_1 = require("./encoding");
function getTensorView(tensor) {
    const typed = tensor.toTypedArray();
    return new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
}
function defaultTensorId(index) {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
    }
    const stamp = Date.now().toString(36);
    const random = Math.random().toString(36).slice(2, 10);
    return `tensor-${stamp}-${index}-${random}`;
}
async function* tensorToFrames(tensor, options, index = 0) {
    const chunkBytes = options?.chunkBytes ?? 64 * 1024;
    if (chunkBytes <= 0) {
        throw new Error('chunkBytes must be positive');
    }
    const tensorId = options?.tensorId ?? defaultTensorId(index);
    tensor.eval();
    const shape = tensor.shape;
    const dtype = tensor.dtype;
    const view = getTensorView(tensor);
    yield {
        type: 'header',
        tensorId,
        shape,
        dtype,
        metadata: options?.metadata,
    };
    if (view.byteLength > 0) {
        let sequence = 0;
        for (let offset = 0; offset < view.byteLength; offset += chunkBytes) {
            const end = Math.min(offset + chunkBytes, view.byteLength);
            const chunk = view.subarray(offset, end);
            yield {
                type: 'data',
                tensorId,
                data: (0, encoding_1.toBase64)(chunk),
                encoding: 'base64',
                sequence: sequence++,
                byteLength: chunk.byteLength,
            };
        }
    }
    yield {
        type: 'end',
        tensorId,
        metadata: options?.endMetadata,
    };
}
async function* tensorsToFrameStream(tensors, options) {
    let index = 0;
    if (Symbol.asyncIterator in tensors) {
        for await (const tensor of tensors) {
            for await (const frame of tensorToFrames(tensor, options, index++)) {
                yield frame;
            }
        }
    }
    else {
        for (const tensor of tensors) {
            for await (const frame of tensorToFrames(tensor, options, index++)) {
                yield frame;
            }
        }
    }
}

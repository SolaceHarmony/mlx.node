"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createEventStream = createEventStream;
exports.eventStreamResponse = eventStreamResponse;
const array_1 = __importDefault(require("../core/array"));
const stream_1 = require("../core/stream");
const encoding_1 = require("./encoding");
const tensors_1 = require("./tensors");
const encoder = new TextEncoder();
function serializeMessage(message) {
    const retryLine = message.retry ? `retry: ${message.retry}\n` : '';
    const eventLine = message.event ? `event: ${message.event}\n` : '';
    const idLine = message.id ? `id: ${message.id}\n` : '';
    const payload = `${message.data}`.split('\n').map((line) => `data: ${line}\n`).join('');
    const text = `${retryLine}${eventLine}${idLine}${payload}\n`;
    return encoder.encode(text);
}
function toAsyncIterable(source) {
    const resolved = typeof source === 'function' ? source() : source;
    if (resolved == null) {
        throw new Error('Stream source returned null/undefined');
    }
    if (Symbol.asyncIterator in resolved) {
        return resolved;
    }
    if (Symbol.iterator in resolved) {
        return (async function* iterate() {
            for (const item of resolved) {
                yield item;
            }
        })();
    }
    throw new Error('Provided stream source is not iterable');
}
const isPromise = (value) => typeof value === 'object' && value !== null && 'then' in value;
const isAsyncIterable = (value) => typeof value === 'object' && value !== null && Symbol.asyncIterator in value;
const isIterable = (value) => typeof value === 'object' && value !== null && Symbol.iterator in value;
function isTensorFrame(value) {
    return (typeof value === 'object' &&
        value !== null &&
        'type' in value &&
        typeof value.type === 'string');
}
function isSSEMessage(value) {
    return (typeof value === 'object' &&
        value !== null &&
        'data' in value &&
        typeof value.data === 'string');
}
function resolveFactoryValue(value, tensor, index) {
    if (typeof value === 'function') {
        return value(tensor, index);
    }
    return value;
}
function getHeartbeatInterval(options) {
    const { heartbeatIntervalMs } = options ?? {};
    if (heartbeatIntervalMs === undefined) {
        return 15000;
    }
    if (heartbeatIntervalMs <= 0) {
        return undefined;
    }
    return heartbeatIntervalMs;
}
function createEventStream(source, options) {
    const iterable = toAsyncIterable(source);
    const retryDelayMs = options?.retryDelayMs;
    const heartbeatMs = getHeartbeatInterval(options);
    const tensorOptions = options?.tensor;
    const streamOption = options?.stream;
    let tensorIndex = 0;
    let streamCtx = null;
    return new ReadableStream({
        start(controller) {
            let heartbeatTimer;
            let stopped = false;
            const push = (message) => {
                if (retryDelayMs && !message.retry) {
                    message.retry = retryDelayMs;
                }
                controller.enqueue(serializeMessage(message));
            };
            const pushFrame = (frame) => {
                push((0, encoding_1.frameToMessage)(frame));
            };
            const stop = (reason) => {
                if (stopped) {
                    return;
                }
                stopped = true;
                if (heartbeatTimer) {
                    clearInterval(heartbeatTimer);
                }
                if (streamCtx) {
                    streamCtx.exit();
                    streamCtx = null;
                }
                if (reason instanceof Error && options?.quiet !== true) {
                    // eslint-disable-next-line no-console
                    console.warn('[mlx.node] SSE stream stopped:', reason.message);
                }
                controller.close();
            };
            const emitTensor = async (tensor) => {
                const currentIndex = tensorIndex++;
                const chunkBytes = tensorOptions?.chunkBytes;
                const metadata = tensorOptions?.metadata
                    ? resolveFactoryValue(tensorOptions.metadata, tensor, currentIndex)
                    : undefined;
                const endMetadata = tensorOptions?.endMetadata
                    ? resolveFactoryValue(tensorOptions.endMetadata, tensor, currentIndex)
                    : undefined;
                const tensorId = tensorOptions?.idFactory
                    ? tensorOptions.idFactory(tensor, currentIndex)
                    : undefined;
                for await (const frame of (0, tensors_1.tensorToFrames)(tensor, {
                    tensorId,
                    chunkBytes,
                    metadata,
                    endMetadata,
                }, currentIndex)) {
                    pushFrame(frame);
                }
            };
            const emitValue = async (value) => {
                if (value == null) {
                    return;
                }
                if (isPromise(value)) {
                    await emitValue(await value);
                    return;
                }
                if (value instanceof array_1.default) {
                    await emitTensor(value);
                    return;
                }
                if (isTensorFrame(value)) {
                    pushFrame(value);
                    return;
                }
                if (typeof value === 'string') {
                    push({ data: value });
                    return;
                }
                if (isSSEMessage(value)) {
                    push(value);
                    return;
                }
                if (isAsyncIterable(value)) {
                    for await (const inner of value) {
                        await emitValue(inner);
                    }
                    return;
                }
                if (isIterable(value)) {
                    for (const inner of value) {
                        // eslint-disable-next-line no-await-in-loop
                        await emitValue(inner);
                    }
                    return;
                }
                throw new Error('Unsupported value provided to event stream');
            };
            const pump = async () => {
                try {
                    if (retryDelayMs) {
                        push({ data: '', retry: retryDelayMs });
                    }
                    if (streamOption && !streamCtx) {
                        streamCtx = (0, stream_1.streamContext)(streamOption);
                        streamCtx.enter();
                    }
                    for await (const value of iterable) {
                        await emitValue(value);
                    }
                    stop();
                }
                catch (error) {
                    stop(error);
                }
            };
            pump();
            if (heartbeatMs) {
                heartbeatTimer = setInterval(() => {
                    if (stopped) {
                        return;
                    }
                    push({ event: 'heartbeat', data: JSON.stringify({ timestamp: Date.now() }) });
                }, heartbeatMs);
            }
            if (options?.signal) {
                options.signal.addEventListener('abort', () => {
                    stop(options.signal?.reason);
                }, { once: true });
            }
        },
    });
}
function eventStreamResponse(source, options) {
    const stream = createEventStream(source, options);
    const headers = new Headers({
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
    });
    if (options?.headers) {
        Object.entries(options.headers).forEach(([key, value]) => {
            headers.set(key, value);
        });
    }
    return new Response(stream, { headers });
}

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createStreamHandler = createStreamHandler;
const streaming_1 = require("../streaming");
function createStreamHandler(producer, options) {
    return async (...args) => {
        const sourceOrPromise = await producer(...args);
        const source = sourceOrPromise;
        return (0, streaming_1.eventStreamResponse)(source, options);
    };
}

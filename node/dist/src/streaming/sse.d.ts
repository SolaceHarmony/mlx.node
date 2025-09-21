import type { SSEOptions, StreamSource, StreamValue } from './types';
export declare function createEventStream(source: StreamSource<StreamValue>, options?: SSEOptions): ReadableStream<Uint8Array>;
export declare function eventStreamResponse(source: StreamSource<StreamValue>, options?: SSEOptions): Response;

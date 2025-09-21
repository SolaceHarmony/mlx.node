import type { SSEOptions, StreamSource, StreamValue } from '../streaming';
export type StreamPayload = StreamValue;
export type StreamProducer<TArgs extends unknown[]> = (...args: TArgs) => StreamSource<StreamPayload> | Promise<StreamSource<StreamPayload>>;
export type StreamHandler<TArgs extends unknown[]> = (...args: TArgs) => Promise<Response>;
export declare function createStreamHandler<TArgs extends unknown[]>(producer: StreamProducer<TArgs>, options?: SSEOptions): StreamHandler<TArgs>;

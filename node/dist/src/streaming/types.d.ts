import type MLXArray from '../core/array';
import type { StreamLike } from '../core/stream';
export type TensorDType = 'bool' | 'uint8' | 'uint16' | 'uint32' | 'uint64' | 'int8' | 'int16' | 'int32' | 'int64' | 'float16' | 'bfloat16' | 'float32' | 'float64' | 'complex64';
export interface TensorHeaderFrame {
    type: 'header';
    tensorId: string;
    shape: readonly number[];
    dtype: TensorDType;
    strides?: readonly number[];
    metadata?: Record<string, unknown>;
}
export interface TensorDataFrame {
    type: 'data';
    tensorId: string;
    data: ArrayBuffer | ArrayBufferView | string;
    encoding?: 'base64' | 'binary' | 'json';
    sequence?: number;
    byteLength?: number;
}
export interface TensorEndFrame {
    type: 'end';
    tensorId: string;
    metadata?: Record<string, unknown>;
}
export interface HeartbeatFrame {
    type: 'heartbeat';
    timestamp: number;
}
export interface ErrorFrame {
    type: 'error';
    message: string;
    tensorId?: string;
    recoverable?: boolean;
    metadata?: Record<string, unknown>;
}
export type TensorStreamFrame = TensorHeaderFrame | TensorDataFrame | TensorEndFrame | HeartbeatFrame | ErrorFrame;
export interface SSEMessage {
    data: string;
    event?: string;
    id?: string;
    retry?: number;
}
export interface SSEOptions {
    heartbeatIntervalMs?: number;
    signal?: AbortSignal;
    retryDelayMs?: number;
    quiet?: boolean;
    headers?: Record<string, string>;
    tensor?: TensorStreamConfig;
    stream?: StreamLike;
}
export interface TensorFrameOptions {
    tensorId?: string;
    chunkBytes?: number;
    metadata?: Record<string, unknown>;
    endMetadata?: Record<string, unknown>;
}
export interface TensorStreamConfig {
    chunkBytes?: number;
    idFactory?: (tensor: MLXArray, index: number) => string;
    metadata?: Record<string, unknown> | ((tensor: MLXArray, index: number) => Record<string, unknown> | undefined);
    endMetadata?: Record<string, unknown> | ((tensor: MLXArray, index: number) => Record<string, unknown> | undefined);
}
export type StreamValue = SSEMessage | TensorStreamFrame | string | MLXArray | Promise<SSEMessage | TensorStreamFrame | string | MLXArray> | AsyncIterable<StreamValue> | Iterable<StreamValue>;
export type StreamSource<T = StreamValue> = AsyncIterable<T> | Iterable<T> | (() => AsyncIterable<T> | Iterable<T>);

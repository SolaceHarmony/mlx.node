import type { TensorStreamFrame, SSEMessage } from './types';
export declare function toBase64(input: ArrayBuffer | ArrayBufferView): string;
export declare function fromBase64(input: string): Uint8Array;
export declare function frameToMessage(frame: TensorStreamFrame): SSEMessage;

import type { Device, DeviceLike } from './device';
export type StreamLike = MLXStream | DeviceLike;
interface NativeStreamContext {
    enter(): void;
    exit(): void;
}
export declare class MLXStream {
    private readonly handle;
    private constructor();
    static fromHandle(handle: any): MLXStream;
    get index(): number;
    get device(): Device;
    equals(other: MLXStream): boolean;
    toString(): string;
    toNative(): any;
}
export declare function isStream(value: unknown): value is MLXStream;
export declare function defaultStream(device?: DeviceLike | null): MLXStream;
export declare function newStream(device?: DeviceLike | null): MLXStream;
export declare function setDefaultStream(stream: MLXStream): void;
export declare function synchronize(stream?: StreamLike | null): void;
export declare function streamContext(stream: StreamLike): NativeStreamContext;
export declare function withStream<T>(stream: StreamLike, fn: () => Promise<T> | T): Promise<T>;
export declare function toNativeStreamArgument(stream?: StreamLike | null): any;
export {};

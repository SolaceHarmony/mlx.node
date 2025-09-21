import type { TensorStreamFrame } from '../streaming';
export interface UseEventStreamState<T> {
    latest: T | null;
    isConnected: boolean;
    error: Error | null;
    start: () => void;
    stop: () => void;
}
export interface UseEventStreamOptions {
    autoStart?: boolean;
    eventSourceInit?: EventSourceInit;
}
export interface UseTensorStreamOptions extends UseEventStreamOptions {
    keepHistory?: boolean;
    onFrame?: (frame: TensorStreamFrame) => void;
    decodeBinary?: boolean;
}
export interface UseTensorStreamState extends UseEventStreamState<TensorStreamFrame> {
    history: readonly TensorStreamFrame[];
}
export declare function useTensorStream(urlFactory: string | URL | (() => string | URL | null) | null, options?: UseTensorStreamOptions): UseTensorStreamState;

import MLXArray from './array';
import type { StreamLike } from './stream';
export interface StreamOptions {
    stream?: StreamLike;
}
export interface ReshapeOptions extends StreamOptions {
}
export declare function reshape(tensor: MLXArray, shape: readonly number[], options?: ReshapeOptions): MLXArray;
export interface TransposeOptions extends StreamOptions {
}
export declare function transpose(tensor: MLXArray, axesOrOptions?: readonly number[] | TransposeOptions | null, maybeOptions?: TransposeOptions): MLXArray;
export interface MoveAxisOptions extends StreamOptions {
}
export declare function moveaxis(tensor: MLXArray, source: number | readonly number[], destination: number | readonly number[], options?: MoveAxisOptions): MLXArray;
export interface SwapAxesOptions extends StreamOptions {
}
export declare function swapaxes(tensor: MLXArray, axis1: number, axis2: number, options?: SwapAxesOptions): MLXArray;
export interface BinaryOpOptions extends StreamOptions {
}
export declare function add(a: MLXArray, b: MLXArray, options?: BinaryOpOptions): MLXArray;
export declare function multiply(a: MLXArray, b: MLXArray, options?: BinaryOpOptions): MLXArray;
export interface WhereOptions extends StreamOptions {
}
export declare function where(condition: MLXArray, onTrue: MLXArray, onFalse: MLXArray, options?: WhereOptions): MLXArray;

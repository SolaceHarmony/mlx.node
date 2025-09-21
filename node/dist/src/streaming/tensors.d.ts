import MLXArray from '../core/array';
import type { TensorFrameOptions, TensorStreamFrame } from './types';
export declare function tensorToFrames(tensor: MLXArray, options?: TensorFrameOptions, index?: number): AsyncIterable<TensorStreamFrame>;
export declare function tensorsToFrameStream(tensors: AsyncIterable<MLXArray> | Iterable<MLXArray>, options?: TensorFrameOptions): AsyncIterable<TensorStreamFrame>;

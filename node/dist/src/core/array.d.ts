import type { DTypeKey } from './dtype';
export type DType = DTypeKey;
type NumericArray = readonly (number | bigint)[];
type SupportedTypedArray = Float32Array | Float64Array | Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | BigInt64Array | BigUint64Array;
export type ArrayElement = number | boolean | [number, number];
export declare class MLXArray {
    private readonly handle;
    private constructor();
    static fromHandle(handle: any): MLXArray;
    static from(data: SupportedTypedArray | NumericArray, shape: readonly number[], dtype?: DType): MLXArray;
    static fromFloat32(data: Float32Array | readonly number[], shape: readonly number[]): MLXArray;
    get shape(): number[];
    get dtype(): DType;
    eval(): this;
    toTypedArray(): SupportedTypedArray;
    toFloat32Array(): Float32Array;
    toNative(): any;
    toArray(): ArrayElement[];
}
export declare function array(data: SupportedTypedArray | NumericArray, shape: readonly number[], dtype?: DType): MLXArray;
export declare const normalizeShapeInput: (shape: readonly number[]) => number[];
export declare function zeros(shape: readonly number[], dtype?: DType): MLXArray;
export declare function ones(shape: readonly number[], dtype?: DType): MLXArray;
export declare function full(shape: readonly number[], value: number, dtype?: DType): MLXArray;
export declare function zeros_like(base: MLXArray): MLXArray;
export declare function ones_like(base: MLXArray): MLXArray;
export default MLXArray;

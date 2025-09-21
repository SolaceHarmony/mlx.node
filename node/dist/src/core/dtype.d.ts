export type DTypeKey = 'bool' | 'int8' | 'int16' | 'int32' | 'int64' | 'uint8' | 'uint16' | 'uint32' | 'uint64' | 'float16' | 'bfloat16' | 'float32' | 'float64' | 'complex64';
export type DTypeCategoryKey = 'complexfloating' | 'floating' | 'inexact' | 'signedinteger' | 'unsignedinteger' | 'integer' | 'number' | 'generic';
export interface MLXDtype {
    readonly key: string;
    readonly size: number;
    readonly category: string;
    equals(other: MLXDtype): boolean;
    toString(): string;
}
export interface DTypeCategory {
    readonly name: DTypeCategoryKey;
    readonly value: number;
    equals(other: DTypeCategory): boolean;
    toString(): string;
}
export type DTypeLike = MLXDtype | DTypeCategory;
declare const Dtype: {
    fromString(key: string): MLXDtype;
};
declare const issubdtype: (a: DTypeLike, b: DTypeLike) => boolean;
declare const fromString: (key: DTypeKey) => MLXDtype;
declare const keys: () => DTypeKey[];
declare const values: () => MLXDtype[];
declare const items: () => Array<[DTypeKey, MLXDtype]>;
declare const categoryKeys: () => DTypeCategoryKey[];
declare const categoryValues: () => DTypeCategory[];
declare const categoryItems: () => Array<[DTypeCategoryKey, DTypeCategory]>;
declare const has: (key: string) => key is DTypeKey;
declare const get: (key: DTypeKey) => MLXDtype;
declare const dir: () => string[];
export { Dtype, fromString, keys, values, items, dir, has, get, issubdtype, categoryKeys, categoryValues, categoryItems, };
export declare const bool: MLXDtype, int8: MLXDtype, int16: MLXDtype, int32: MLXDtype, int64: MLXDtype, uint8: MLXDtype, uint16: MLXDtype, uint32: MLXDtype, uint64: MLXDtype, float16: MLXDtype, bfloat16: MLXDtype, float32: MLXDtype, float64: MLXDtype, complex64: MLXDtype;
export declare const complexfloating: DTypeCategory, floating: DTypeCategory, inexact: DTypeCategory, signedinteger: DTypeCategory, unsignedinteger: DTypeCategory, integer: DTypeCategory, number: DTypeCategory, generic: DTypeCategory;
declare const _default: {
    complexfloating: DTypeCategory;
    floating: DTypeCategory;
    inexact: DTypeCategory;
    signedinteger: DTypeCategory;
    unsignedinteger: DTypeCategory;
    integer: DTypeCategory;
    number: DTypeCategory;
    generic: DTypeCategory;
    bool: MLXDtype;
    int8: MLXDtype;
    int16: MLXDtype;
    int32: MLXDtype;
    int64: MLXDtype;
    uint8: MLXDtype;
    uint16: MLXDtype;
    uint32: MLXDtype;
    uint64: MLXDtype;
    float16: MLXDtype;
    bfloat16: MLXDtype;
    float32: MLXDtype;
    float64: MLXDtype;
    complex64: MLXDtype;
    Dtype: {
        fromString(key: string): MLXDtype;
    };
    fromString: (key: DTypeKey) => MLXDtype;
    keys: () => DTypeKey[];
    values: () => MLXDtype[];
    items: () => Array<[DTypeKey, MLXDtype]>;
    dir: () => string[];
    has: (key: string) => key is DTypeKey;
    get: (key: DTypeKey) => MLXDtype;
    issubdtype: (a: DTypeLike, b: DTypeLike) => boolean;
    categoryKeys: () => DTypeCategoryKey[];
    categoryValues: () => DTypeCategory[];
    categoryItems: () => Array<[DTypeCategoryKey, DTypeCategory]>;
};
export default _default;

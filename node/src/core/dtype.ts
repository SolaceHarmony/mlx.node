import addon from '../internal/addon';

export type DTypeKey =
  | 'bool'
  | 'int8'
  | 'int16'
  | 'int32'
  | 'int64'
  | 'uint8'
  | 'uint16'
  | 'uint32'
  | 'uint64'
  | 'float16'
  | 'bfloat16'
  | 'float32'
  | 'float64'
  | 'complex64';

export type DTypeCategoryKey =
  | 'complexfloating'
  | 'floating'
  | 'inexact'
  | 'signedinteger'
  | 'unsignedinteger'
  | 'integer'
  | 'number'
  | 'generic';

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

const nativeDtype = addon.dtype as Record<string, unknown> & {
  fromString: (key: string) => MLXDtype;
};

const Dtype = addon.Dtype as { fromString(key: string): MLXDtype };
const issubdtype = addon.issubdtype as (a: DTypeLike, b: DTypeLike) => boolean;

const fromString = (key: DTypeKey): MLXDtype => nativeDtype.fromString(key);

const dtypeConstants = {
  bool: nativeDtype.bool as MLXDtype,
  int8: nativeDtype.int8 as MLXDtype,
  int16: nativeDtype.int16 as MLXDtype,
  int32: nativeDtype.int32 as MLXDtype,
  int64: nativeDtype.int64 as MLXDtype,
  uint8: nativeDtype.uint8 as MLXDtype,
  uint16: nativeDtype.uint16 as MLXDtype,
  uint32: nativeDtype.uint32 as MLXDtype,
  uint64: nativeDtype.uint64 as MLXDtype,
  float16: nativeDtype.float16 as MLXDtype,
  bfloat16: nativeDtype.bfloat16 as MLXDtype,
  float32: nativeDtype.float32 as MLXDtype,
  float64: nativeDtype.float64 as MLXDtype,
  complex64: nativeDtype.complex64 as MLXDtype,
};

const dtypeKeyList = Object.freeze(
  Object.keys(dtypeConstants) as DTypeKey[],
);

const categoryConstants = {
  complexfloating: nativeDtype.complexfloating as DTypeCategory,
  floating: nativeDtype.floating as DTypeCategory,
  inexact: nativeDtype.inexact as DTypeCategory,
  signedinteger: nativeDtype.signedinteger as DTypeCategory,
  unsignedinteger: nativeDtype.unsignedinteger as DTypeCategory,
  integer: nativeDtype.integer as DTypeCategory,
  number: nativeDtype.number as DTypeCategory,
  generic: nativeDtype.generic as DTypeCategory,
};

const categoryKeyList = Object.freeze(
  Object.keys(categoryConstants) as DTypeCategoryKey[],
);

const keys = (): DTypeKey[] => [...dtypeKeyList];
const values = (): MLXDtype[] => dtypeKeyList.map((key) => dtypeConstants[key]);
const items = (): Array<[DTypeKey, MLXDtype]> =>
  dtypeKeyList.map((key) => [key, dtypeConstants[key]]);

const categoryKeys = (): DTypeCategoryKey[] => [...categoryKeyList];
const categoryValues = (): DTypeCategory[] =>
  categoryKeyList.map((key) => categoryConstants[key]);
const categoryItems = (): Array<[DTypeCategoryKey, DTypeCategory]> =>
  categoryKeyList.map((key) => [key, categoryConstants[key]]);

const has = (key: string): key is DTypeKey =>
  Object.prototype.hasOwnProperty.call(dtypeConstants, key);

const get = (key: DTypeKey): MLXDtype => dtypeConstants[key];

const dir = (): string[] => [
  'Dtype',
  'fromString',
  'keys',
  'values',
  'items',
  'dir',
  'has',
  'get',
  'issubdtype',
  'categoryKeys',
  'categoryValues',
  'categoryItems',
  ...dtypeKeyList,
  ...categoryKeyList,
];

export {
  Dtype,
  fromString,
  keys,
  values,
  items,
  dir,
  has,
  get,
  issubdtype,
  categoryKeys,
  categoryValues,
  categoryItems,
};
export const {
  bool,
  int8,
  int16,
  int32,
  int64,
  uint8,
  uint16,
  uint32,
  uint64,
  float16,
  bfloat16,
  float32,
  float64,
  complex64,
} = dtypeConstants;

export const {
  complexfloating,
  floating,
  inexact,
  signedinteger,
  unsignedinteger,
  integer,
  number,
  generic,
} = categoryConstants;

export default {
  Dtype,
  fromString,
  keys,
  values,
  items,
  dir,
  has,
  get,
  issubdtype,
  categoryKeys,
  categoryValues,
  categoryItems,
  ...dtypeConstants,
  ...categoryConstants,
};

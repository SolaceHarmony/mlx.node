import addon from './internal/addon';
import * as utils from './utils';
import * as core from './core';
import * as streaming from './streaming';
import * as react from './react';
import * as optimizers from './optimizers';
import * as nn_init from './nn_init';
import * as nn from './nn';

// Native addon utilities
export const native = {
  hello: (): string => addon.hello(),
};

// Export namespaces for organized access
export { core, utils, streaming, react, optimizers, nn_init, nn };

// Convenience re-exports from core (TypeScript idiomatic style)
// Usage: import { array, zeros } from 'mlx';
export const {
  array,
  Array,
  Stream,
  issubdtype,
  zeros,
  zeros_like,
  ones,
  ones_like,
  full,
  default_stream,
  new_stream,
  set_default_stream,
  synchronize,
  stream_context,
  stream,
  with_stream,
  device,
  reshape,
  transpose,
  moveaxis,
  swapaxes,
  add,
  multiply,
  subtract,
  divide,
  power,
  equal,
  not_equal,
  less,
  less_equal,
  greater,
  greater_equal,
  maximum,
  minimum,
  where,
  arange,
  random,
  tan,
  sin,
  cos,
  arcsin,
  arccos,
  arctan,
  arctan2,
  rsqrt,
  square,
  sign,
  abs,
  sqrt,
  exp,
  log,
  import_function,
  // DType constants
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
} = core;

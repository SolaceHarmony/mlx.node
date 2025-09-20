import addon from './internal/addon';
import * as utils from './utils';
import * as core from './core';
import * as streaming from './streaming';
import * as react from './react';

export const native = {
  hello: (): string => addon.hello(),
};

export { core, utils, streaming, react };
export const array = core.array;
export const Array = core.Array;
export const Stream = core.Stream;
export const issubdtype = core.issubdtype;
export const zeros = core.zeros;
export const zeros_like = core.zeros_like;
export const ones = core.ones;
export const ones_like = core.ones_like;
export const full = core.full;
export const defaultStream = core.defaultStream;
export const newStream = core.newStream;
export const setDefaultStream = core.setDefaultStream;
export const synchronize = core.synchronize;
export const streamContext = core.streamContext;
export const stream = core.stream;
export const withStream = core.withStream;
export const device = core.device;
export const reshape = core.reshape;
export const transpose = core.transpose;
export const moveaxis = core.moveaxis;
export const swapaxes = core.swapaxes;
export const add = core.add;
export const multiply = core.multiply;
export const where = core.where;

export default {
  native,
  core,
  utils,
  react,
  streaming,
  device,
  array,
  Array,
  issubdtype,
  zeros,
  zeros_like,
  ones,
  ones_like,
  full,
  Stream,
  defaultStream,
  newStream,
  setDefaultStream,
  synchronize,
  streamContext,
  stream,
  withStream,
  reshape,
  transpose,
  moveaxis,
  swapaxes,
  add,
  multiply,
  where,
};

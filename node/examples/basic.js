'use strict';

const mx = require('..');

// CPU tensor
const a = mx.core.zeros([2, 2], mx.float32);
// GPU tensor on default GPU stream
const b = mx.core.ones([2, 2], mx.float16, mx.core.default_stream('gpu'));
const c = mx.core.matmul(a, b);

console.log('a.shape', a.shape(), 'a.dtype', a.dtype());
console.log('b.shape', b.shape(), 'b.dtype', b.dtype());
console.log('c.shape', c.shape(), 'c.dtype', c.dtype());


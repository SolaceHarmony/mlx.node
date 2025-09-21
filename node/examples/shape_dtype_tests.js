'use strict';

const mx = require('..');

function log(title, fn) {
  try { console.log(title, fn()); }
  catch (e) { console.log(title, 'ERR', e.message); }
}

// Shape as integer
const a = mx.core.zeros(4, mx.float32);
console.log('zeros(4) -> shape', a.shape(), 'dtype', a.dtype());

// Shape as array
const b = mx.core.ones([2, 3], mx.float32);
console.log('ones([2,3]) -> shape', b.shape(), 'dtype', b.dtype());

// dtype object only
try { mx.core.zeros([2,2], 'float32'); console.log('unexpected'); }
catch (e) { console.log('string dtype rejected:', !!e.message); }

// full values with various dtypes
const f16 = mx.core.full([1], 3.14159265, mx.float16).toTypedArray();
const bf16 = mx.core.full([1], 3.14159265, mx.bfloat16).toTypedArray();
const f32 = mx.core.full([1], 3.14159265, mx.float32).toTypedArray();
console.log('full f16 u16hex', f16[0].toString(16));
console.log('full bf16 u16hex', bf16[0].toString(16));
console.log('full f32', f32[0]);

// integers
const i8 = mx.core.full([3],  [127, -128, 130],  mx.int8);  console.log('i8',  Array.from(i8.toTypedArray()));
const u8 = mx.core.full([3],  [255, 300, -1],   mx.uint8); console.log('u8',  Array.from(u8.toTypedArray()));
const i16= mx.core.full([3],  [32767,-32768, 40000], mx.int16); console.log('i16', Array.from(i16.toTypedArray()));


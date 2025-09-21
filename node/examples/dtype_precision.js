'use strict';

const mx = require('..');

function f32ToBits(f) {
  const b = new ArrayBuffer(4);
  new DataView(b).setFloat32(0, f, false);
  return new DataView(b).getUint32(0, false);
}

// Round-to-nearest-even bfloat16 from float32
function f32ToBf16Bits(f) {
  let bits = f32ToBits(f);
  const lsb = (bits >>> 16) & 1;              // least significant bit of bf16 mantissa
  const roundBias = 0x00007FFF + lsb;         // round to nearest even
  bits += roundBias;
  return (bits >>> 16) & 0xFFFF;
}

// IEEE-754 float32 -> float16 bits (round to nearest even)
function f32ToF16Bits(f) {
  let x = f32ToBits(f);
  const sign = (x >>> 31) & 0x1;
  let exp = (x >>> 23) & 0xFF;
  let mant = x & 0x7FFFFF;
  let out;
  if (exp === 0xFF) {
    // Inf/NaN
    const isNaN = mant !== 0;
    out = (sign << 15) | (0x1F << 10) | (isNaN ? 0x200 : 0);
  } else if (exp === 0) {
    // Subnormal or zero in f32 => subnormal/zero in f16
    if (mant === 0) {
      out = sign << 15;
    } else {
      // Normalize the mantissa
      while ((mant & 0x800000) === 0) { mant <<= 1; exp -= 1; }
      mant &= 0x7FFFFF;
      exp = exp + (127 - 14) - 23; // adjust later with rounding
      let val = mant | 0x800000;
      let f16_m = val >>> (23 - 10);
      // round
      const round = (val >>> (23 - 10 - 1)) & 1;
      if (round && ((f16_m & 1) || ((val & ((1 << (23 - 10 - 1)) - 1)) !== 0))) {
        f16_m += 1;
      }
      out = (sign << 15) | (0 << 10) | (f16_m & 0x3FF);
    }
  } else {
    // Normalized
    const newExp = exp - 127 + 15;
    if (newExp >= 0x1F) {
      // Overflow => Inf
      out = (sign << 15) | (0x1F << 10);
    } else if (newExp <= 0) {
      // Subnormal
      const shift = 14 - newExp;
      let mant32 = (mant | 0x800000) >>> shift;
      // Round to nearest even
      const roundBit = (mant | 0x800000) >>> (shift - 1) & 1;
      if (roundBit && ((mant32 & 1) || (((mant | 0x800000) & ((1 << (shift - 1)) - 1)) !== 0))) {
        mant32 += 1;
      }
      out = (sign << 15) | (0 << 10) | (mant32 & 0x3FF);
    } else {
      // Normalized, round mantissa
      let mant16 = mant >>> (23 - 10);
      const round = (mant >>> (23 - 10 - 1)) & 1;
      if (round && ((mant16 & 1) || ((mant & ((1 << (23 - 10 - 1)) - 1)) !== 0))) {
        mant16 += 1;
        if (mant16 === 0x400) { // carry
          mant16 = 0;
          if (++exp === 0xFF) {
            out = (sign << 15) | (0x1F << 10);
            return out;
          }
        }
      }
      const exp16 = exp - 127 + 15;
      out = (sign << 15) | ((exp16 & 0x1F) << 10) | (mant16 & 0x3FF);
    }
  }
  return out & 0xFFFF;
}

function hex16(u16) { return '0x' + u16.toString(16).padStart(4, '0'); }

function testF16(value) {
  const expected = f32ToF16Bits(value);
  const arr = mx.core.full([1], value, mx.float16).toTypedArray(); // Uint16Array
  const got = arr[0];
  console.log(`f16 ${value} -> expected ${hex16(expected)} got ${hex16(got)} ${expected===got?'OK':'MISMATCH'}`);
}

function testBf16(value) {
  const expected = f32ToBf16Bits(value);
  const arr = mx.core.full([1], value, mx.bfloat16).toTypedArray(); // Uint16Array
  const got = arr[0];
  console.log(`bf16 ${value} -> expected ${hex16(expected)} got ${hex16(got)} ${expected===got?'OK':'MISMATCH'}`);
}

function testInt(dtype, value) {
  const arr = mx.core.full([1], value, dtype).toTypedArray();
  console.log(`int ${dtype===mx.uint8?'u8':dtype===mx.int8?'i8':dtype===mx.int16?'i16':dtype===mx.uint16?'u16':'?'} ${value} -> ${arr[0]}`);
}

console.log('--- float16 checks ---');
testF16(0.0);
testF16(-0.0);
testF16(1.0);
testF16(-2.0);
testF16(65504.0);   // max f16 normal
testF16(1e-8);      // subnormal region
testF16(Number.POSITIVE_INFINITY);
testF16(Number.NaN);

console.log('--- bfloat16 checks ---');
testBf16(1.0);
testBf16(3.14159265);
testBf16(-2.5);
testBf16(Number.POSITIVE_INFINITY);
testBf16(Number.NaN);

console.log('--- integer checks ---');
testInt(mx.int8, 127);
testInt(mx.int8, -128);
testInt(mx.int8, 130);    // observe wrap/clamp behavior
testInt(mx.uint8, 255);
testInt(mx.uint8, 300);   // observe wrap/clamp behavior


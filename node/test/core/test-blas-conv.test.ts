import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  matmul,
  addmm,
  einsum,
  tensordot,
  conv1d,
  conv2d,
  conv_transpose1d,
  array,
  transpose,
  float32,
} from '../../src';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const approx = (a: number, b: number, tol = 1e-4) => Math.abs(a - b) < tol;

const allClose = (arr: number[], expected: number[], tol = 1e-4) =>
  arr.every((v, i) => approx(v, expected[i], tol));

const flat = (a: ReturnType<typeof array>): number[] =>
  a.toArray() as number[];

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

describe('matmul', () => {
  it('2x2 @ 2x2', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const b = array([0, -1, -3, 3], [2, 2], float32);
    // [[1*0+2*-3, 1*-1+2*3], [3*0+4*-3, 3*-1+4*3]] = [[-6,5],[-12,9]]
    const c = matmul(a, b);
    assert.deepEqual(c.shape, [2, 2]);
    assert.deepEqual(flat(c), [-6, 5, -12, 9]);
  });

  it('3x2 @ 2x3', () => {
    const a = array([1, 2, 3, 4, 5, 6], [3, 2], float32);
    const b = array([1, 0, 2, 0, 1, 3], [2, 3], float32);
    // row0: [1*1+2*0, 1*0+2*1, 1*2+2*3]=[1,2,8]
    // row1: [3*1+4*0, 3*0+4*1, 3*2+4*3]=[3,4,18]
    // row2: [5*1+6*0, 5*0+6*1, 5*2+6*3]=[5,6,28]
    const c = matmul(a, b);
    assert.deepEqual(c.shape, [3, 3]);
    assert.deepEqual(flat(c), [1, 2, 8, 3, 4, 18, 5, 6, 28]);
  });

  it('identity matmul', () => {
    const a = array([1, 0, 0, 1], [2, 2], float32);
    const b = array([5, 6, 7, 8], [2, 2], float32);
    const c = matmul(a, b);
    assert.deepEqual(flat(c), [5, 6, 7, 8]);
  });

  it('matrix-vector (2D @ 1D shape via 2x1)', () => {
    // matmul of [2,3] @ [3,1] -> [2,1]
    const a = array([1, 2, 3, 4, 5, 6], [2, 3], float32);
    const b = array([1, 0, -1], [3, 1], float32);
    const c = matmul(a, b);
    assert.deepEqual(c.shape, [2, 1]);
    // [1-3, 4-6] = [-2, -2]
    assert.deepEqual(flat(c), [-2, -2]);
  });

  it('vector-matrix (1x2 @ 2x3)', () => {
    const a = array([2, 3], [1, 2], float32);
    const b = array([1, 0, 2, 0, 1, 3], [2, 3], float32);
    const c = matmul(a, b);
    assert.deepEqual(c.shape, [1, 3]);
    // [2+0, 0+3, 4+9] = [2, 3, 13]
    assert.deepEqual(flat(c), [2, 3, 13]);
  });

  it('batched 3D matmul (batch=2, 2x2 @ 2x2)', () => {
    // batch 0: [[1,0],[0,1]] @ [[2,3],[4,5]] = [[2,3],[4,5]]
    // batch 1: [[1,1],[0,1]] @ [[1,0],[0,1]] = [[1,1],[0,1]]
    const a = array([1, 0, 0, 1, 1, 1, 0, 1], [2, 2, 2], float32);
    const b = array([2, 3, 4, 5, 1, 0, 0, 1], [2, 2, 2], float32);
    const c = matmul(a, b);
    assert.deepEqual(c.shape, [2, 2, 2]);
    assert.deepEqual(flat(c), [2, 3, 4, 5, 1, 1, 0, 1]);
  });

  it('transposed a: a.T @ b', () => {
    // a is 3x2, a.T is 2x3, b is 3x2 => result is 2x2
    const a = array([1, 2, 3, 4, 5, 6], [3, 2], float32);
    const aT = transpose(a); // [2, 3]
    const b = array([1, 0, 0, 1, 1, 1], [3, 2], float32);
    const c = matmul(aT, b);
    assert.deepEqual(c.shape, [2, 2]);
    // aT = [[1,3,5],[2,4,6]]
    // row0: [1+0+5, 0+3+5]=[6,8]
    // row1: [2+0+6, 0+4+6]=[8,10]
    assert.deepEqual(flat(c), [6, 8, 8, 10]);
  });

  it('transposed b: a @ b.T', () => {
    // a is 2x3, b is 2x3, b.T is 3x2 => result is 2x2
    const a = array([1, 2, 3, 4, 5, 6], [2, 3], float32);
    const b = array([1, 0, 1, 0, 1, 0], [2, 3], float32);
    const bT = transpose(b); // [3, 2]
    const c = matmul(a, bT);
    assert.deepEqual(c.shape, [2, 2]);
    // row0: [1+0+3, 0+2+0]=[4,2]
    // row1: [4+0+6, 0+5+0]=[10,5]
    assert.deepEqual(flat(c), [4, 2, 10, 5]);
  });

  it('both transposed: a.T @ b.T', () => {
    // a is 2x3 => aT is 3x2, b is 3x2 => bT is 2x3 => result is 3x3
    const a = array([1, 0, 2, 0, 1, 0], [2, 3], float32);
    const b = array([1, 2, 3, 4, 5, 6], [3, 2], float32);
    const c = matmul(transpose(a), transpose(b));
    // aT=[[1,0],[0,1],[2,0]], bT=[[1,3,5],[2,4,6]]
    // (3x2) @ (2x3) => (3x3)
    assert.deepEqual(c.shape, [3, 3]);
    // row0: [1*1+0*2, 1*3+0*4, 1*5+0*6]=[1,3,5]
    // row1: [0*1+1*2, 0*3+1*4, 0*5+1*6]=[2,4,6]
    // row2: [2*1+0*2, 2*3+0*4, 2*5+0*6]=[2,6,10]
    assert.deepEqual(flat(c), [1, 3, 5, 2, 4, 6, 2, 6, 10]);
  });

  it('square matmul agrees with hand computation (3x3)', () => {
    const a = array([1, 2, 0, 0, 1, 1, 3, 0, 1], [3, 3], float32);
    const b = array([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], float32); // identity
    const c = matmul(a, b);
    assert.deepEqual(flat(c), [1, 2, 0, 0, 1, 1, 3, 0, 1]);
  });
});

// ---------------------------------------------------------------------------
// addmm
// ---------------------------------------------------------------------------

describe('addmm', () => {
  it('basic: alpha*(a@b) + beta*c  (2x2)', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const b = array([1, 0, 0, 1], [2, 2], float32); // identity
    const c = array([10, 20, 30, 40], [2, 2], float32);
    // alpha=0.5, beta=2 => 0.5*a + 2*c = [0.5+20, 1+40, 1.5+60, 2+80] = [20.5,41,61.5,82]
    const out = addmm(c, a, b, 0.5, 2.0);
    assert.deepEqual(out.shape, [2, 2]);
    assert.ok(allClose(flat(out), [20.5, 41, 61.5, 82]));
  });

  it('default alpha=1, beta=1', () => {
    const a = array([1, 0, 0, 1], [2, 2], float32);
    const b = array([3, 4, 5, 6], [2, 2], float32);
    const c = array([10, 10, 10, 10], [2, 2], float32);
    // a@b = b, so result = b + c = [13,14,15,16]
    const out = addmm(c, a, b);
    assert.deepEqual(flat(out), [13, 14, 15, 16]);
  });

  it('alpha=0 returns beta*c', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const b = array([5, 6, 7, 8], [2, 2], float32);
    const c = array([1, 1, 1, 1], [2, 2], float32);
    const out = addmm(c, a, b, 0.0, 3.0);
    assert.deepEqual(flat(out), [3, 3, 3, 3]);
  });

  it('broadcast c as scalar-like (1,)', () => {
    // a@b = [[19,22],[43,50]], c=[100]
    // alpha=1, beta=1 => [[119,122],[143,150]]
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const b = array([5, 6, 7, 8], [2, 2], float32);
    const c = array([100], [1], float32);
    const out = addmm(c, a, b);
    assert.deepEqual(out.shape, [2, 2]);
    assert.ok(allClose(flat(out), [119, 122, 143, 150]));
  });

  it('non-square: 2x3 @ 3x2 + 2x2', () => {
    const a = array([1, 0, 1, 0, 1, 0], [2, 3], float32);
    const b = array([1, 2, 3, 4, 5, 6], [3, 2], float32);
    const c = array([0, 0, 0, 0], [2, 2], float32);
    // a@b = [[1+0+5, 2+0+6],[0+3+0, 0+4+0]] = [[6,8],[3,4]]
    const out = addmm(c, a, b);
    assert.deepEqual(flat(out), [6, 8, 3, 4]);
  });
});

// ---------------------------------------------------------------------------
// einsum
// ---------------------------------------------------------------------------

describe('einsum', () => {
  it('matrix multiply: ij,jk->ik', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const b = array([5, 6, 7, 8], [2, 2], float32);
    const c = einsum('ij,jk->ik', [a, b]);
    assert.deepEqual(c.shape, [2, 2]);
    // [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert.deepEqual(flat(c), [19, 22, 43, 50]);
  });

  it('trace: ii->', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const tr = einsum('ii->', [a]);
    assert.deepEqual(tr.shape, []);
    assert.equal(flat(tr)[0], 5); // 1+4
  });

  it('trace of 3x3', () => {
    const a = array([1, 0, 0, 0, 2, 0, 0, 0, 3], [3, 3], float32);
    const tr = einsum('ii->', [a]);
    assert.equal(flat(tr)[0], 6);
  });

  it('outer product: i,j->ij', () => {
    const a = array([1, 2, 3], [3], float32);
    const b = array([4, 5], [2], float32);
    const c = einsum('i,j->ij', [a, b]);
    assert.deepEqual(c.shape, [3, 2]);
    assert.deepEqual(flat(c), [4, 5, 8, 10, 12, 15]);
  });

  it('batch matmul: bij,bjk->bik', () => {
    // batch=1, 2x2 @ 2x2
    const a = array([1, 0, 0, 1], [1, 2, 2], float32);
    const b = array([2, 3, 4, 5], [1, 2, 2], float32);
    const c = einsum('bij,bjk->bik', [a, b]);
    assert.deepEqual(c.shape, [1, 2, 2]);
    assert.deepEqual(flat(c), [2, 3, 4, 5]);
  });

  it('diagonal: ii->i', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const d = einsum('ii->i', [a]);
    assert.deepEqual(d.shape, [2]);
    assert.deepEqual(flat(d), [1, 4]);
  });

  it('sum all elements: ij->', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const s = einsum('ij->', [a]);
    assert.equal(flat(s)[0], 10);
  });

  it('transpose: ij->ji', () => {
    const a = array([1, 2, 3, 4, 5, 6], [2, 3], float32);
    const aT = einsum('ij->ji', [a]);
    assert.deepEqual(aT.shape, [3, 2]);
    assert.deepEqual(flat(aT), [1, 4, 2, 5, 3, 6]);
  });

  it('dot product: i,i->', () => {
    const a = array([1, 2, 3], [3], float32);
    const b = array([4, 5, 6], [3], float32);
    const c = einsum('i,i->', [a, b]);
    assert.equal(flat(c)[0], 32); // 4+10+18
  });
});

// ---------------------------------------------------------------------------
// tensordot
// ---------------------------------------------------------------------------

describe('tensordot', () => {
  it('axes=1 (standard matrix multiply)', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const b = array([5, 6, 7, 8], [2, 2], float32);
    const c = tensordot(a, b, 1);
    assert.deepEqual(c.shape, [2, 2]);
    assert.deepEqual(flat(c), [19, 22, 43, 50]);
  });

  it('axes=0 (outer product)', () => {
    const a = array([1, 2], [2], float32);
    const b = array([3, 4], [2], float32);
    const c = tensordot(a, b, 0);
    assert.deepEqual(c.shape, [2, 2]);
    assert.deepEqual(flat(c), [3, 4, 6, 8]);
  });

  it('axes as explicit index lists', () => {
    // Contract last axis of a with first axis of b
    const a = array([1, 2, 3, 4, 5, 6], [2, 3], float32);
    const b = array([1, 0, 0, 1, 1, 1], [3, 2], float32);
    const c = tensordot(a, b, [[1], [0]]);
    assert.deepEqual(c.shape, [2, 2]);
    // Same as matmul: [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
    assert.deepEqual(flat(c), [4, 5, 10, 11]);
  });

  it('full contraction axes=2 on 2x2 matrices', () => {
    const a = array([1, 2, 3, 4], [2, 2], float32);
    const b = array([1, 0, 0, 1], [2, 2], float32);
    // axes=2 contracts both dims => scalar: sum of a*b = 1+0+0+4 = 5
    const c = tensordot(a, b, 2);
    assert.deepEqual(c.shape, []);
    assert.equal(flat(c)[0], 5);
  });
});

// ---------------------------------------------------------------------------
// conv1d
// ---------------------------------------------------------------------------

describe('conv1d', () => {
  it('basic: single sample, single channel, single filter', () => {
    // input: [N=1, L=5, C_in=1], weight: [C_out=1, K=3, C_in=1]
    const input = array([1, 2, 3, 4, 5], [1, 5, 1], float32);
    const weight = array([1, 1, 1], [1, 3, 1], float32);
    const out = conv1d(input, weight);
    assert.deepEqual(out.shape, [1, 3, 1]);
    // [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    assert.ok(allClose(flat(out), [6, 9, 12]));
  });

  it('with padding=1', () => {
    const input = array([1, 2, 3, 4, 5], [1, 5, 1], float32);
    const weight = array([1, 1, 1], [1, 3, 1], float32);
    const out = conv1d(input, weight, 1, 1); // stride=1, padding=1
    assert.deepEqual(out.shape, [1, 5, 1]);
    // [0+1+2, 1+2+3, 2+3+4, 3+4+5, 4+5+0] = [3, 6, 9, 12, 9]
    assert.ok(allClose(flat(out), [3, 6, 9, 12, 9]));
  });

  it('with stride=2', () => {
    const input = array([1, 2, 3, 4, 5, 6], [1, 6, 1], float32);
    const weight = array([1, 1, 1], [1, 3, 1], float32);
    const out = conv1d(input, weight, 2); // stride=2
    assert.deepEqual(out.shape, [1, 2, 1]);
    // positions 0,2: [1+2+3, 3+4+5] = [6, 12]
    assert.ok(allClose(flat(out), [6, 12]));
  });

  it('with dilation=2', () => {
    const input = array([1, 2, 3, 4, 5], [1, 5, 1], float32);
    const weight = array([1, 1, 1], [1, 3, 1], float32);
    const out = conv1d(input, weight, 1, 0, 2); // stride=1, padding=0, dilation=2
    assert.deepEqual(out.shape, [1, 1, 1]);
    // effective kernel taps at positions 0,2,4: [1+3+5] = [9]
    assert.ok(allClose(flat(out), [9]));
  });

  it('multi-channel input (C_in=2, C_out=1)', () => {
    // input: [1, 3, 2] => 3 timesteps, 2 channels
    const input = array([
      1, 10,  // t=0
      2, 20,  // t=1
      3, 30,  // t=2
    ], [1, 3, 2], float32);
    // weight: [C_out=1, K=2, C_in=2]
    const weight = array([
      1, 0,   // k=0: ch0=1, ch1=0
      0, 1,   // k=1: ch0=0, ch1=1
    ], [1, 2, 2], float32);
    const out = conv1d(input, weight);
    assert.deepEqual(out.shape, [1, 2, 1]);
    // t=0: 1*1 + 10*0 + 2*0 + 20*1 = 21
    // t=1: 2*1 + 20*0 + 3*0 + 30*1 = 32
    assert.ok(allClose(flat(out), [21, 32]));
  });

  it('multi-output channels (C_in=1, C_out=2)', () => {
    const input = array([1, 2, 3], [1, 3, 1], float32);
    // weight: [C_out=2, K=2, C_in=1]
    const weight = array([
      1, 1,  // filter 0: sum
      1, -1, // filter 1: diff
    ], [2, 2, 1], float32);
    const out = conv1d(input, weight);
    assert.deepEqual(out.shape, [1, 2, 2]);
    // filter0: [1+2, 2+3]=[3,5], filter1: [1-2, 2-3]=[-1,-1]
    // output layout: t=0:[3,-1], t=1:[5,-1]
    assert.ok(allClose(flat(out), [3, -1, 5, -1]));
  });
});

// ---------------------------------------------------------------------------
// conv2d
// ---------------------------------------------------------------------------

describe('conv2d', () => {
  it('basic 3x3 input, 2x2 kernel, single channel (NHWC)', () => {
    // input: [N=1, H=3, W=3, C=1]
    const input = array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
    ], [1, 3, 3, 1], float32);
    // weight: [C_out=1, KH=2, KW=2, C_in=1]
    const weight = array([1, 1, 1, 1], [1, 2, 2, 1], float32);
    const out = conv2d(input, weight);
    assert.deepEqual(out.shape, [1, 2, 2, 1]);
    // top-left: 1+2+4+5=12, top-right: 2+3+5+6=16
    // bot-left: 4+5+7+8=24, bot-right: 5+6+8+9=28
    assert.ok(allClose(flat(out), [12, 16, 24, 28]));
  });

  it('with padding=1', () => {
    // 2x2 input, 2x2 kernel, padding=1 => padded 4x4, output 3x3
    const input = array([1, 2, 3, 4], [1, 2, 2, 1], float32);
    const weight = array([1, 0, 0, 0], [1, 2, 2, 1], float32); // picks top-left only
    const out = conv2d(input, weight, [1, 1], [1, 1]);
    assert.deepEqual(out.shape, [1, 3, 3, 1]);
    // padded:   0 0 0 0
    //           0 1 2 0
    //           0 3 4 0
    //           0 0 0 0
    // kernel picks top-left => output is the padded values at each position:
    // [0,0,0, 0,1,2, 0,3,4]
    assert.ok(allClose(flat(out), [0, 0, 0, 0, 1, 2, 0, 3, 4]));
  });

  it('stride=[2,2] downsamples', () => {
    // 4x4 input => with 2x2 kernel, stride 2 => 2x2 output
    const input = array([
      1, 1, 2, 2,
      1, 1, 2, 2,
      3, 3, 4, 4,
      3, 3, 4, 4,
    ], [1, 4, 4, 1], float32);
    const weight = array([0.25, 0.25, 0.25, 0.25], [1, 2, 2, 1], float32); // average
    const out = conv2d(input, weight, [2, 2]);
    assert.deepEqual(out.shape, [1, 2, 2, 1]);
    assert.ok(allClose(flat(out), [1, 2, 3, 4]));
  });

  it('multi-channel input (C_in=2, C_out=1)', () => {
    // input: [1, 2, 2, 2]
    const input = array([
      1, 10,  2, 20,
      3, 30,  4, 40,
    ], [1, 2, 2, 2], float32);
    // weight: [1, 1, 1, 2] — single 1x1 filter summing both channels
    const weight = array([1, 0.1], [1, 1, 1, 2], float32);
    const out = conv2d(input, weight);
    assert.deepEqual(out.shape, [1, 2, 2, 1]);
    // 1+1=2, 2+2=4, 3+3=6, 4+4=8
    assert.ok(allClose(flat(out), [2, 4, 6, 8]));
  });

  it('identity 1x1 conv preserves input', () => {
    const input = array([1, 2, 3, 4], [1, 2, 2, 1], float32);
    const weight = array([1], [1, 1, 1, 1], float32);
    const out = conv2d(input, weight);
    assert.deepEqual(flat(out), [1, 2, 3, 4]);
  });
});

// ---------------------------------------------------------------------------
// conv_transpose1d
// ---------------------------------------------------------------------------

describe('conv_transpose1d', () => {
  it('basic: reverses a stride-1 conv shape', () => {
    // input: [N=1, L=3, C_in=1]
    // weight: [C_out=1, K=3, C_in=1]
    // transpose conv with kernel [1,1,1] and stride=1:
    // output length = (3-1)*1 + 3 = 5
    const input = array([1, 2, 3], [1, 3, 1], float32);
    const weight = array([1, 1, 1], [1, 3, 1], float32);
    const out = conv_transpose1d(input, weight);
    assert.deepEqual(out.shape, [1, 5, 1]);
    // Full convolution (flip kernel = [1,1,1]):
    // pos0: 1, pos1: 1+2=3, pos2: 1+2+3=6, pos3: 2+3=5, pos4: 3
    assert.ok(allClose(flat(out), [1, 3, 6, 5, 3]));
  });

  it('stride=2 upsamples', () => {
    // output length = (L-1)*stride + K = (3-1)*2 + 3 = 7
    const input = array([1, 1, 1], [1, 3, 1], float32);
    const weight = array([1, 1, 1], [1, 3, 1], float32);
    const out = conv_transpose1d(input, weight, { stride: 2 });
    assert.deepEqual(out.shape, [1, 7, 1]);
    // Manual computation:
    // input [1,_,1,_,1] zero-inserted, then full conv with [1,1,1]:
    // result: [1, 1, 2, 1, 2, 1, 1]
    assert.ok(allClose(flat(out), [1, 1, 2, 1, 2, 1, 1]));
  });

  it('with padding=1', () => {
    const input = array([1, 2, 3], [1, 3, 1], float32);
    const weight = array([1, 1, 1], [1, 3, 1], float32);
    const out = conv_transpose1d(input, weight, { padding: 1 });
    assert.deepEqual(out.shape, [1, 5, 1]);
    assert.ok(allClose(flat(out), [1, 3, 6, 5, 3]));
  });

  it('identity kernel', () => {
    const input = array([1, 2, 3, 4, 5], [1, 5, 1], float32);
    const weight = array([1], [1, 1, 1], float32);
    const out = conv_transpose1d(input, weight);
    assert.deepEqual(out.shape, [1, 5, 1]);
    assert.ok(allClose(flat(out), [1, 2, 3, 4, 5]));
  });

  it('multi-channel', () => {
    // input: [1, 2, 2] — 2 timesteps, 2 channels
    const input = array([1, 0, 0, 1], [1, 2, 2], float32);
    // weight: [C_out=1, K=1, C_in=2] — 1x1 conv_transpose sums channels
    const weight = array([1, 1], [1, 1, 2], float32);
    const out = conv_transpose1d(input, weight);
    assert.deepEqual(out.shape, [1, 2, 1]);
    assert.ok(allClose(flat(out), [1, 1]));
  });
});

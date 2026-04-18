// port-lint: source include/tensor.hpp

/**
 * Simple tensor class for Tree-LSTM computations.
 * Lightweight alternative to libraries like TensorFlow.js for this specific use case.
 */
export class Tensor {
  data: Float32Array;
  rows: number = 0;
  cols: number = 0;

  constructor(r: number, c: number = 1, val: number = 0) {
    this.rows = r;
    this.cols = c;
    this.data = new Float32Array(r * c).fill(val);
  }

  static fromVector(vec: number[] | Float32Array): Tensor {
    const t = new Tensor(vec.length, 1);
    t.data.set(vec);
    return t;
  }

  size(): number {
    return this.data.length;
  }

  isVector(): boolean {
    return this.cols === 1;
  }

  get(i: number, j: number = 0): number {
    return this.data[i * this.cols + j]!;
  }

  set(i: number, j: number, val: number): void {
    this.data[i * this.cols + j] = val;
  }

  // Linear access
  at(i: number): number {
    return this.data[i]!;
  }

  setAt(i: number, val: number): void {
    this.data[i] = val;
  }

  // Initialize with zeros
  static zeros(rows: number, cols: number = 1): Tensor {
    return new Tensor(rows, cols, 0);
  }

  /**
   * Initialize with random values (Box-Muller transform for normal distribution)
   */
  static randn(r: number, c: number, scale: number = 1.0): Tensor {
    const t = new Tensor(r, c);
    for (let i = 0; i < t.data.length; i++) {
      // Box-Muller transform
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      t.data[i] = z0 * scale;
    }
    return t;
  }

  // Element-wise operations
  add(other: Tensor): Tensor {
    if (this.size() !== other.size()) {
      throw new Error("Tensor size mismatch in addition");
    }
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! + other.data[i]!;
    }
    return result;
  }

  sub(other: Tensor): Tensor {
    if (this.size() !== other.size()) {
      throw new Error("Tensor size mismatch in subtraction");
    }
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! - other.data[i]!;
    }
    return result;
  }

  // Element-wise multiplication (Hadamard product)
  hadamard(other: Tensor): Tensor {
    if (this.size() !== other.size()) {
      throw new Error("Tensor size mismatch in Hadamard product");
    }
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! * other.data[i]!;
    }
    return result;
  }

  // Scalar multiplication
  mul(scalar: number): Tensor {
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! * scalar;
    }
    return result;
  }

  // Matrix-vector multiplication
  matmul(vec: Tensor): Tensor {
    if (this.cols !== vec.rows) {
      throw new Error("Matrix-vector dimension mismatch");
    }
    const result = new Tensor(this.rows, 1);
    for (let i = 0; i < this.rows; i++) {
      let sum = 0;
      for (let j = 0; j < this.cols; j++) {
        sum += this.get(i, j) * vec.at(j);
      }
      result.setAt(i, sum);
    }
    return result;
  }

  // Dot product (for vectors)
  dot(other: Tensor): number {
    if (this.size() !== other.size()) {
      throw new Error("Vector size mismatch in dot product");
    }
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      sum += this.data[i]! * other.data[i]!;
    }
    return sum;
  }

  // L2 norm
  norm(): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      const v = this.data[i]!;
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  // Cosine similarity
  cosineSimilarity(other: Tensor): number {
    const dotProd = this.dot(other);
    const normA = this.norm();
    const normB = other.norm();
    if (normA < 1e-8 || normB < 1e-8) return 0.0;
    return dotProd / (normA * normB);
  }

  // Activation functions
  sigmoid(): Tensor {
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = 1.0 / (1.0 + Math.exp(-this.data[i]!));
    }
    return result;
  }

  tanh(): Tensor {
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = Math.tanh(this.data[i]!);
    }
    return result;
  }

  relu(): Tensor {
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = Math.max(0.0, this.data[i]!);
    }
    return result;
  }

  // Softmax
  softmax(): Tensor {
    const result = new Tensor(this.rows, this.cols);
    let maxVal = -Infinity;
    for (let i = 0; i < this.data.length; i++) {
      if (this.data[i]! > maxVal) maxVal = this.data[i]!;
    }

    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = Math.exp(this.data[i]! - maxVal);
      sum += result.data[i]!;
    }
    for (let i = 0; i < result.data.length; i++) {
      result.data[i] /= sum;
    }
    return result;
  }

  // Concatenate two vectors
  concat(other: Tensor): Tensor {
    const result = new Tensor(this.size() + other.size(), 1);
    result.data.set(this.data);
    result.data.set(other.data, this.size());
    return result;
  }

  // Absolute value
  abs(): Tensor {
    const result = new Tensor(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = Math.abs(this.data[i]!);
    }
    return result;
  }
}

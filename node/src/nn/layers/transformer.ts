/**
 * Transformer layers: MultiHeadAttention, TransformerEncoder/Decoder, Transformer.
 *
 * Mirrors mlx.nn.layers.transformer from the Python MLX API.
 */
import {
  reshape,
  transpose,
  add,
  multiply,
  less,
  arange,
  softmax,
  fast,
  maximum,
} from '../../core/ops';
import MLXArray from '../../core/array';
import { Linear } from './linear';
import { LayerNorm } from './normalization';
import { Dropout } from './dropout';

/**
 * Implements scaled dot product attention with multiple heads.
 *
 * @param dims - Model dimensions
 * @param numHeads - Number of attention heads
 * @param queryInputDims - Input dims for queries (default: dims)
 * @param keyInputDims - Input dims for keys (default: dims)
 * @param valueInputDims - Input dims for values (default: keyInputDims)
 * @param valueDims - Value dims after projection (default: dims)
 * @param valueOutputDims - Output projection dims (default: dims)
 * @param bias - Whether to use bias in projections (default: false)
 */
export class MultiHeadAttention {
  numHeads: number;
  queryProj: Linear;
  keyProj: Linear;
  valueProj: Linear;
  outProj: Linear;

  constructor(
    dims: number,
    numHeads: number,
    queryInputDims?: number,
    keyInputDims?: number,
    valueInputDims?: number,
    valueDims?: number,
    valueOutputDims?: number,
    bias: boolean = false,
  ) {
    if (dims % numHeads !== 0) {
      throw new Error(
        `The input feature dimensions should be divisible by the number of heads (${dims} % ${numHeads}) != 0`,
      );
    }

    const qid = queryInputDims ?? dims;
    const kid = keyInputDims ?? dims;
    const vid = valueInputDims ?? kid;
    const vd = valueDims ?? dims;
    const vod = valueOutputDims ?? dims;

    this.numHeads = numHeads;
    this.queryProj = new Linear(qid, dims, bias);
    this.keyProj = new Linear(kid, dims, bias);
    this.valueProj = new Linear(vid, vd, bias);
    this.outProj = new Linear(vd, vod, bias);
  }

  forward(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    mask?: MLXArray,
  ): MLXArray {
    queries = this.queryProj.forward(queries);
    keys = this.keyProj.forward(keys);
    values = this.valueProj.forward(values);

    const numHeads = this.numHeads;
    const B = queries.shape[0];
    const L = queries.shape[1];
    const headDim = Math.floor(queries.shape[2] / numHeads);

    // unflatten + transpose: (B, L, D) -> (B, L, H, Dh) -> (B, H, L, Dh)
    queries = transpose(reshape(queries, [B, L, numHeads, headDim]), [0, 2, 1, 3]);
    const S = keys.shape[1];
    keys = transpose(reshape(keys, [B, S, numHeads, headDim]), [0, 2, 1, 3]);
    const vHeadDim = Math.floor(values.shape[2] / numHeads);
    values = transpose(reshape(values, [B, S, numHeads, vHeadDim]), [0, 2, 1, 3]);

    const scale = Math.sqrt(1 / headDim);
    let output = fast.scaled_dot_product_attention(
      queries, keys, values, scale, mask,
    );

    // (B, H, L, Dh) -> (B, L, H, Dh) -> (B, L, D)
    output = reshape(transpose(output, [0, 2, 1, 3]), [B, L, -1]);
    return this.outProj.forward(output);
  }

  /**
   * Creates a causal attention mask of shape (N, N).
   * Positions that should not be attended to are -Infinity.
   */
  static createAdditiveCausalMask(N: number): MLXArray {
    const indices = arange(N);
    // indices[:, None] < indices[None] — upper triangle is true
    const rowIdx = reshape(indices, [N, 1]);
    const colIdx = reshape(indices, [1, N]);
    const mask = less(rowIdx, colIdx);
    // Convert boolean mask to float with -inf for true positions
    return multiply(mask, -1e9);
  }
}

/**
 * A single Transformer encoder layer.
 *
 * @param dims - Model dimensions
 * @param numHeads - Number of attention heads
 * @param mlpDims - MLP hidden dimension (default: dims * 4)
 * @param dropout - Dropout probability (default: 0.0)
 * @param activation - MLP activation function
 * @param normFirst - Pre-norm (true) or post-norm (false)
 */
export class TransformerEncoderLayer {
  attention: MultiHeadAttention;
  ln1: LayerNorm;
  ln2: LayerNorm;
  linear1: Linear;
  linear2: Linear;
  dropout1: Dropout;
  dropout2: Dropout;
  activation: (x: MLXArray) => MLXArray;
  normFirst: boolean;

  constructor(
    dims: number,
    numHeads: number,
    mlpDims?: number,
    dropout: number = 0.0,
    activation?: (x: MLXArray) => MLXArray,
    normFirst: boolean = true,
  ) {
    const md = mlpDims ?? dims * 4;
    this.attention = new MultiHeadAttention(dims, numHeads);
    this.ln1 = new LayerNorm(dims);
    this.ln2 = new LayerNorm(dims);
    this.linear1 = new Linear(dims, md);
    this.linear2 = new Linear(md, dims);
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.activation = activation ?? ((x: MLXArray) => maximum(x, 0)); // relu
    this.normFirst = normFirst;
  }

  forward(x: MLXArray, mask?: MLXArray): MLXArray {
    if (this.normFirst) {
      let y = this.ln1.forward(x);
      y = this.attention.forward(y, y, y, mask);
      y = this.dropout1.forward(y);
      x = add(x, y);

      y = this.ln2.forward(x);
      y = this.linear1.forward(y);
      y = this.activation(y);
      y = this.dropout2.forward(y);
      y = this.linear2.forward(y);
      return add(x, y);
    } else {
      let y = this.attention.forward(x, x, x, mask);
      y = this.dropout1.forward(y);
      x = this.ln1.forward(add(x, y));

      y = this.linear1.forward(x);
      y = this.activation(y);
      y = this.dropout2.forward(y);
      y = this.linear2.forward(y);
      return this.ln2.forward(add(x, y));
    }
  }
}

/**
 * A stack of Transformer encoder layers.
 */
export class TransformerEncoder {
  layers: TransformerEncoderLayer[];
  ln: LayerNorm;

  constructor(
    numLayers: number,
    dims: number,
    numHeads: number,
    mlpDims?: number,
    dropout: number = 0.0,
    activation?: (x: MLXArray) => MLXArray,
    normFirst: boolean = true,
  ) {
    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      this.layers.push(
        new TransformerEncoderLayer(dims, numHeads, mlpDims, dropout, activation, normFirst),
      );
    }
    this.ln = new LayerNorm(dims);
  }

  forward(x: MLXArray, mask?: MLXArray): MLXArray {
    for (const layer of this.layers) {
      x = layer.forward(x, mask);
    }
    return this.ln.forward(x);
  }
}

/**
 * A single Transformer decoder layer.
 *
 * @param dims - Model dimensions
 * @param numHeads - Number of attention heads
 * @param mlpDims - MLP hidden dimension (default: dims * 4)
 * @param dropout - Dropout probability (default: 0.0)
 * @param activation - MLP activation function
 * @param normFirst - Pre-norm (true) or post-norm (false)
 */
export class TransformerDecoderLayer {
  selfAttention: MultiHeadAttention;
  crossAttention: MultiHeadAttention;
  ln1: LayerNorm;
  ln2: LayerNorm;
  ln3: LayerNorm;
  linear1: Linear;
  linear2: Linear;
  dropout1: Dropout;
  dropout2: Dropout;
  dropout3: Dropout;
  activation: (x: MLXArray) => MLXArray;
  normFirst: boolean;

  constructor(
    dims: number,
    numHeads: number,
    mlpDims?: number,
    dropout: number = 0.0,
    activation?: (x: MLXArray) => MLXArray,
    normFirst: boolean = true,
  ) {
    const md = mlpDims ?? dims * 4;
    this.selfAttention = new MultiHeadAttention(dims, numHeads);
    this.crossAttention = new MultiHeadAttention(dims, numHeads);
    this.ln1 = new LayerNorm(dims);
    this.ln2 = new LayerNorm(dims);
    this.ln3 = new LayerNorm(dims);
    this.linear1 = new Linear(dims, md);
    this.linear2 = new Linear(md, dims);
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.dropout3 = new Dropout(dropout);
    this.activation = activation ?? ((x: MLXArray) => maximum(x, 0)); // relu
    this.normFirst = normFirst;
  }

  forward(
    x: MLXArray,
    memory: MLXArray,
    xMask?: MLXArray,
    memoryMask?: MLXArray,
  ): MLXArray {
    if (this.normFirst) {
      let y = this.ln1.forward(x);
      y = this.selfAttention.forward(y, y, y, xMask);
      y = this.dropout1.forward(y);
      x = add(x, y);

      y = this.ln2.forward(x);
      y = this.crossAttention.forward(y, memory, memory, memoryMask);
      y = this.dropout2.forward(y);
      x = add(x, y);

      y = this.ln3.forward(x);
      y = this.linear1.forward(y);
      y = this.activation(y);
      y = this.dropout3.forward(y);
      y = this.linear2.forward(y);
      return add(x, y);
    } else {
      let y = this.selfAttention.forward(x, x, x, xMask);
      y = this.dropout1.forward(y);
      x = this.ln1.forward(add(x, y));

      y = this.crossAttention.forward(x, memory, memory, memoryMask);
      y = this.dropout2.forward(y);
      x = this.ln2.forward(add(x, y));

      y = this.linear1.forward(x);
      y = this.activation(y);
      y = this.dropout3.forward(y);
      y = this.linear2.forward(y);
      return this.ln3.forward(add(x, y));
    }
  }
}

/**
 * A stack of Transformer decoder layers.
 */
export class TransformerDecoder {
  layers: TransformerDecoderLayer[];
  ln: LayerNorm;

  constructor(
    numLayers: number,
    dims: number,
    numHeads: number,
    mlpDims?: number,
    dropout: number = 0.0,
    activation?: (x: MLXArray) => MLXArray,
    normFirst: boolean = true,
  ) {
    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      this.layers.push(
        new TransformerDecoderLayer(dims, numHeads, mlpDims, dropout, activation, normFirst),
      );
    }
    this.ln = new LayerNorm(dims);
  }

  forward(
    x: MLXArray,
    memory: MLXArray,
    xMask?: MLXArray,
    memoryMask?: MLXArray,
  ): MLXArray {
    for (const layer of this.layers) {
      x = layer.forward(x, memory, xMask, memoryMask);
    }
    return this.ln.forward(x);
  }
}

/**
 * Implements a standard Transformer model.
 *
 * Based on "Attention Is All You Need".
 *
 * @param dims - Model dimensions (default: 512)
 * @param numHeads - Number of attention heads (default: 8)
 * @param numEncoderLayers - Number of encoder layers (default: 6)
 * @param numDecoderLayers - Number of decoder layers (default: 6)
 * @param mlpDims - MLP hidden dimension (default: dims * 4)
 * @param dropout - Dropout probability (default: 0.0)
 * @param activation - MLP activation function (default: relu)
 * @param normFirst - Pre-norm or post-norm (default: true)
 */
export class Transformer {
  encoder: TransformerEncoder;
  decoder: TransformerDecoder;

  constructor(
    dims: number = 512,
    numHeads: number = 8,
    numEncoderLayers: number = 6,
    numDecoderLayers: number = 6,
    mlpDims?: number,
    dropout: number = 0.0,
    activation?: (x: MLXArray) => MLXArray,
    normFirst: boolean = true,
    customEncoder?: TransformerEncoder,
    customDecoder?: TransformerDecoder,
  ) {
    this.encoder = customEncoder ?? new TransformerEncoder(
      numEncoderLayers, dims, numHeads, mlpDims, dropout, activation, normFirst,
    );
    this.decoder = customDecoder ?? new TransformerDecoder(
      numDecoderLayers, dims, numHeads, mlpDims, dropout, activation, normFirst,
    );
  }

  forward(
    src: MLXArray,
    tgt: MLXArray,
    srcMask?: MLXArray,
    tgtMask?: MLXArray,
    memoryMask?: MLXArray,
  ): MLXArray {
    const memory = this.encoder.forward(src, srcMask);
    return this.decoder.forward(tgt, memory, tgtMask, memoryMask);
  }
}

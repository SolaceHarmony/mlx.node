/**
 * Embedding layer.
 *
 * Mirrors mlx.nn.layers.embedding from the Python MLX API.
 */
import { transpose, matmul, take, random } from '../../core/ops';
import MLXArray from '../../core/array';

/**
 * Implements a simple lookup table that maps integer indices to embeddings.
 *
 * @param numEmbeddings - Number of embeddings (vocabulary size)
 * @param dims - Dimension of each embedding vector
 */
export class Embedding {
  weight: MLXArray;

  constructor(numEmbeddings: number, dims: number) {
    const scale = Math.sqrt(1 / dims);
    this.weight = random.normal([numEmbeddings, dims], { scale });
  }

  forward(x: MLXArray): MLXArray {
    return take(this.weight, x, 0);
  }

  /** Tied weight projection: x @ weight.T */
  asLinear(x: MLXArray): MLXArray {
    return matmul(x, transpose(this.weight));
  }
}

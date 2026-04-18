/**
 * Neural network layers barrel export.
 *
 * Mirrors mlx.nn.layers.__init__ from the Python MLX API.
 */

export { Module } from './base';

// Activations — functional
export {
  sigmoid,
  relu,
  leaky_relu,
  log_softmax,
  elu,
  relu6,
  softmax,
  softplus,
  softsign,
  softshrink,
  celu,
  silu,
  log_sigmoid,
  gelu,
  gelu_approx,
  gelu_fast_approx,
  glu,
  step,
  selu,
  prelu,
  mish,
  hardswish,
  hard_tanh,
  hard_shrink,
  softmin,
  tanh,
} from './activations';

// Activations — module classes
export {
  Sigmoid,
  Mish,
  ReLU,
  LeakyReLU,
  ELU,
  ReLU6,
  Softmax,
  Softplus,
  Softsign,
  Softshrink,
  CELU,
  SiLU,
  LogSoftmax,
  LogSigmoid,
  PReLU,
  GELU,
  Tanh,
  Hardswish,
  Step,
  SELU,
  HardTanh,
  HardShrink,
  Softmin,
  GLU,
} from './activations';

// Linear
export { Identity, Linear, Bilinear } from './linear';

// Convolution
export { Conv1d, Conv2d, Conv3d } from './convolution';

// Containers
export { Sequential, ModuleList, ModuleDict } from './containers';

// Dropout
export { Dropout, Dropout2d, Dropout3d } from './dropout';

// Embedding
export { Embedding } from './embedding';

// Normalization
export {
  LayerNorm,
  RMSNorm,
  GroupNorm,
  BatchNorm,
  InstanceNorm,
} from './normalization';

// Pooling
export {
  MaxPool1d,
  AvgPool1d,
  MaxPool2d,
  AvgPool2d,
  MaxPool3d,
  AvgPool3d,
} from './pooling';

// Recurrent
export { RNN, GRU, LSTM } from './recurrent';

// Transformer
export {
  MultiHeadAttention,
  TransformerEncoderLayer,
  TransformerEncoder,
  TransformerDecoderLayer,
  TransformerDecoder,
  Transformer,
} from './transformer';

# MLX Neural Network Initializers (nn)

This module provides neural network weight initializers for MLX Node.js, following the same API as Python MLX's `mlx.nn.init` module.

## Available Initializers

### `heNormal(dtype?)`

Build a He normal initializer for weight initialization.

This initializer samples from a normal distribution with a standard deviation computed from the number of input (`fan_in`) or output (`fan_out`) units according to:

```
σ = gain / sqrt(fan)
```

where `fan` is either the number of input units when the mode is `"fan_in"` or output units when the mode is `"fan_out"`.

**Reference:** [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) by Kaiming He et al.

#### Parameters

- `dtype` (MLXDtype, optional): The data type of the array. Default: `float32`

#### Returns

An initializer function with signature:
```typescript
(array: MLXArray, mode?: 'fan_in' | 'fan_out', gain?: number) => MLXArray
```

#### Initializer Parameters

- `array` (MLXArray): The array whose shape determines the output shape
- `mode` ('fan_in' | 'fan_out', optional): Determines which fan to use for computing std. Default: `'fan_in'`
- `gain` (number, optional): Scaling factor for the standard deviation. Default: `1.0`

#### Usage

```typescript
import * as mx from 'mlx';

// Create a He normal initializer
const initFn = mx.nn.heNormal();

// Initialize a fully connected layer's weights (fan_in mode by default)
const fcWeights = mx.core.zeros([512, 1024]); // 1024 input features, 512 output features
const initializedWeights = initFn(fcWeights);

// Use fan_out mode
const fcWeights2 = mx.core.zeros([512, 1024]);
const initializedWeights2 = initFn(fcWeights2, 'fan_out');

// Use custom gain (recommended for ReLU: sqrt(2) ≈ 1.414)
const reluWeights = mx.core.zeros([512, 1024]);
const initializedReluWeights = initFn(reluWeights, 'fan_in', Math.sqrt(2));

// Initialize convolutional layer weights
// Shape: [out_channels, kernel_h, kernel_w, in_channels]
const convWeights = mx.core.zeros([64, 3, 3, 32]);
const initializedConvWeights = initFn(convWeights);

// Use float16 dtype for mixed precision training
const initFnFp16 = mx.nn.heNormal(mx.core.float16);
const fp16Weights = mx.core.zeros([512, 1024]);
const initializedFp16 = initFnFp16(fp16Weights);
```

## Fan Calculation

For different tensor shapes, the fan values are calculated as follows:

- **1D tensors** (bias): `fan_in = fan_out = shape[0]`
- **2D tensors** (fully connected layers): 
  - `fan_in = shape[-1]` (number of input features)
  - `fan_out = shape[0]` (number of output features)
- **4D tensors** (convolutional layers with shape `[out_channels, kernel_h, kernel_w, in_channels]`):
  - `fan_in = in_channels * kernel_h * kernel_w` (receptive field for each output)
  - `fan_out = out_channels * kernel_h * kernel_w` (receptive field for each input)

## Recommended Gains by Activation Function

Different activation functions benefit from different gain values:

- **Linear / Sigmoid / Tanh**: `gain = 1.0` (default)
- **ReLU / Leaky ReLU**: `gain = Math.sqrt(2)` ≈ 1.414
- **SELU**: `gain = 1.0` (use `fan_in` mode)

## Implementation Notes

- This is a pure TypeScript implementation that composes existing core operations
- The underlying random number generation uses `mx.random.normal()` from the C++ bindings
- The fan calculation follows the same logic as Python MLX and PyTorch
- The initializer is stateless and can be reused for multiple weight tensors

## See Also

- `mx.random.normal()` - The underlying random number generator
- Python MLX documentation: https://ml-explore.github.io/mlx/build/html/python/nn.html#initialization

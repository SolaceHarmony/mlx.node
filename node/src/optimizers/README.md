# MLX Optimizers for Node.js

This module provides optimizer implementations for training neural networks with MLX in Node.js.

## Implementation Status

### ✅ Completed

- **Base `Optimizer` class**: Abstract base class with state management, scheduler support, and parameter tree handling
- **`SGD` optimizer class**: Stochastic Gradient Descent with momentum, weight decay, dampening, and Nesterov momentum support
- **`Lion` optimizer class**: EvoLved Sign Momentum implementation with sign-based updates and optional weight decay
- **`RMSprop` optimizer class**: Root Mean Square Propagation with adaptive learning rates based on moving averages of squared gradients
- **`Adagrad` optimizer class**: Adaptive Gradient algorithm with accumulated squared gradients for adaptive learning rates
- **`Muon` optimizer class**: MomentUm Orthogonalized by Newton-schulz, optimized for 2D+ parameters with orthogonalization
- **Core operations**: `add`, `multiply`, `subtract`, `sign`, `square`, `sqrt`, `divide`, `matmul`, `reshape`, `transpose` bindings exposed for optimizer math
- **API structure**: Matches Python MLX API for optimizer initialization and configuration
- **Type safety**: Full TypeScript types and interfaces
- **Validation**: Parameter validation (e.g., Nesterov requirements, RMSprop alpha/epsilon, Adagrad epsilon)
- **State management**: Proper state initialization and tracking
- **Tests**: Unit tests for SGD, Lion, RMSprop, Muon constructors and Adagrad constructors, validation, and state management

### ⚠️ Partially Implemented

The `SGD.applySingle()` method has a placeholder implementation because it requires additional array operations that are not yet available in the Node.js MLX bindings:

**Missing Core Operations:**
- Scalar-array arithmetic (e.g., `multiply(scalar, array)`)
- `astype()` - Dtype conversion for arrays
- Proper scalar array construction from numbers

**Current Workarounds:**
- Step counter uses `zeros([])` instead of proper scalar construction
- Learning rate setting creates empty arrays instead of proper scalar values
- `applySingle()` throws an error explaining the missing dependencies

## Usage Example

```typescript
import { SGD } from 'mlx/optimizers';
import { zeros } from 'mlx/core';

// Create optimizer
const optimizer = new SGD({
  learningRate: 0.01,
  momentum: 0.9,
  weightDecay: 0.0001,
  nesterov: false
});

// Initialize with parameters
const parameters = {
  weight: zeros([10, 5]),
  bias: zeros([5])
};
optimizer.init(parameters);

// During training (once core operations are available):
// const updatedParams = optimizer.applyGradients(gradients, parameters);
```

## API Reference

### `Optimizer` (Abstract Base Class)

**Properties:**
- `state: Record<string, any>` - The optimizer's state dictionary
- `step: MLXArray` - Current step count
- `learningRate: MLXArray` - Current learning rate

**Methods:**
- `init(parameters)` - Initialize optimizer state for a parameter tree
- `applyGradients(gradients, parameters)` - Apply gradients and return updated parameters
- `initSingle(parameter, state)` - Initialize state for a single parameter (abstract)
- `applySingle(gradient, parameter, state)` - Apply update to a single parameter (abstract)

### `SGD` (Stochastic Gradient Descent)

**Constructor Options:**
```typescript
interface SGDOptions {
  learningRate: number | Scheduler;  // Learning rate (can be scheduled)
  momentum?: number;                 // Momentum strength (default: 0)
  weightDecay?: number;              // L2 penalty (default: 0)
  dampening?: number;                // Dampening for momentum (default: 0)
  nesterov?: boolean;                // Enable Nesterov momentum (default: false)
}
```

**Update Formula:**
```
v_{t+1} = μ * v_t + (1 - τ) * g_t
w_{t+1} = w_t - λ * v_{t+1}
```

Where:
- `λ` is the learning rate
- `μ` is the momentum strength
- `τ` is the dampening
- For Nesterov momentum: use `g_t + μ * v_{t+1}` instead of `v_{t+1}`

**Validation:**
- Nesterov momentum requires `momentum > 0` and `dampening = 0`
- Throws error if validation fails

### `Lion` (EvoLved Sign Momentum)

**Constructor Options:**
```typescript
interface LionOptions {
  learningRate: number | Scheduler;
  betas?: [number, number];
  weightDecay?: number;
}
```

**Update Formula:**
```
c_{t+1} = β₁ * m_t + (1 - β₁) * g_t
m_{t+1} = β₂ * m_t + (1 - β₂) * g_t
w_{t+1} = w_t - η * (sign(c_t) + λ * w_t)
```

Where:
- `η` is the learning rate
- `β₁`, `β₂` are the momentum coefficients
- `λ` is the (optional) weight decay strength

**Notes:**
- Recommended learning rates are typically 3–10× smaller than AdamW
- Weight decay should be scaled up proportionally to maintain `lr * weightDecay`

### `RMSprop` (Root Mean Square Propagation)

**Constructor Options:**
```typescript
interface RMSpropOptions {
  learningRate: number | Scheduler;
  alpha?: number;   // Smoothing constant (default: 0.99)
  eps?: number;     // Numerical stability term (default: 1e-8)
}
```

**Update Formula:**
```
v_{t+1} = α * v_t + (1 - α) * g_t²
w_{t+1} = w_t - λ * g_t / (√v_{t+1} + ε)
```

Where:
- `λ` is the learning rate
- `α` is the smoothing constant for the moving average
- `v_t` is the moving average of squared gradients
- `ε` is added to the denominator for numerical stability

**Validation:**
- `alpha` must be >= 0
- `eps` must be > 0
- Throws error if validation fails

**Notes:**
- RMSprop adapts the learning rate for each parameter based on the magnitude of recent gradients
- Good default choice for RNNs and other recurrent architectures
- Reference: Tieleman & Hinton (2012), Lecture 6.5-rmsprop

### `Adagrad` (Adaptive Gradient Algorithm)

**Constructor Options:**
```typescript
interface AdagradOptions {
  learningRate: number | Scheduler;
  eps?: number;     // Numerical stability term (default: 1e-8)
}
```

**Update Formula:**
```
v_{t+1} = v_t + g_t²
w_{t+1} = w_t - λ * g_t / (√v_{t+1} + ε)
```

Where:
- `λ` is the learning rate
- `v_t` is the accumulated sum of squared gradients (not a moving average)
- `g_t` is the gradient
- `ε` is added to the denominator for numerical stability

**Validation:**
- `eps` must be > 0
- Throws error if validation fails

**Notes:**
- Adagrad accumulates all historical squared gradients, unlike RMSprop which uses a decaying average
- Learning rate automatically decreases over time as the accumulator grows
- Well-suited for sparse gradients and natural language processing tasks
- Reference: Duchi, Hazan & Singer (2011), Adaptive subgradient methods for online learning and stochastic optimization

## Architecture

The optimizer implementation follows the Python MLX design:

1. **Pure TypeScript/JavaScript layer**: Optimizers are not C++ primitives but use MLX array operations
2. **Tree-based parameter handling**: Uses `treeMap` utilities to handle nested parameter structures
3. **Lazy initialization**: State is initialized on first use or explicit `init()` call
4. **Scheduler support**: Learning rate and other parameters can be functions of step count
5. **Stateful updates**: Maintains velocity/momentum in state dictionary

## Testing

Run tests with:
```bash
cd node
npm test -- test/optimizers.test.ts
```

Current tests cover:
- Constructor parameter validation
- State initialization
- Property accessors
- Error cases for invalid configurations

## Next Steps

To further improve optimizer functionality:

1. ✅ **Core operations** - COMPLETED:
   - `divide`, `sqrt`, `square`, `rsqrt`, `power` are now available
   - Scalar-array arithmetic is supported

2. ⚠️ **Complete SGD.applySingle()** implementation:
   - Requires `astype()` method for dtype conversion
   - All other required operations are available

3. ⚠️ **Complete Adam.applySingle()** implementation:
   - Requires `astype()` and proper scalar construction
   - Most other required operations are available

4. **Add proper scalar construction**:
   - Support `array(5)` to create a scalar array
   - Or add `scalar(value, dtype)` helper function

5. **Add integration tests** that verify gradient application with actual arrays

6. **Implement other optimizers**:
   - AdamW
   - Adadelta
   - Adamax
   - Adafactor
   - Muon
   - etc.

## References

- Python Implementation: `python/mlx/optimizers/optimizers.py`
- MLX Documentation: [https://ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/)

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
- **Tests**: Unit tests for SGD, Lion, RMSprop, Adagrad, and Muon constructors, validation, and state management

/**
 * MLX Optimizers for Node.js
 * 
 * This module provides optimizer implementations for training neural networks.
 * Based on mlx.optimizers from the Python MLX library.
 */

import MLXArray, { zeros, zerosLike } from '../core/array';
import { add, multiply, subtract, sign, square, sqrt, divide } from '../core/ops';
import { treeMap, treeFlatten, treeUnflatten, treeMerge } from '../utils';

/**
 * Scheduler function type - maps step number to a parameter value
 */
export type Scheduler = (step: MLXArray) => MLXArray;

/**
 * Parameter value that can be either a constant or scheduled
 */
export type SchedulableParam = number | Scheduler;

/**
 * Base class for all optimizers.
 * Allows implementing an optimizer on a per-parameter basis and applying it to a parameter tree.
 */
export abstract class Optimizer {
  protected _initialized: boolean = false;
  protected _state: Record<string, any> = { step: zeros([], 'uint64') };
  protected _schedulers: Record<string, Scheduler> = {};

  constructor(schedulers?: Record<string, Scheduler>) {
    if (schedulers) {
      this._schedulers = { ...schedulers };
    }
  }

  /**
   * Initialize the optimizer's state.
   * This can be used to initialize optimizers which have state (like momentum in SGD).
   * 
   * @param parameters - A tree of parameters
   */
  init(parameters: Record<string, any>): void {
    // Initialize the optimizer state to match the parameter state
    const updateState = (params: any, state: any): any => {
      if (Array.isArray(params) || params instanceof Array) {
        const stateArray = Array.isArray(state) ? state : [];
        for (let i = 0; i < params.length; i++) {
          if (i < stateArray.length) {
            stateArray[i] = updateState(params[i], stateArray[i]);
          } else {
            stateArray.push(treeMap(() => ({}), params[i]));
          }
        }
        return stateArray;
      } else if (params !== null && typeof params === 'object' && !(params instanceof MLXArray)) {
        const stateObj = state || {};
        for (const [k, v] of Object.entries(params)) {
          if (!(k in stateObj)) {
            stateObj[k] = treeMap(() => ({}), v);
          } else {
            stateObj[k] = updateState(v, stateObj[k]);
          }
        }
        return stateObj;
      } else {
        return state;
      }
    };

    updateState(parameters, this._state);
    treeMap(
      (p: any, s: any) => {
        if (p instanceof MLXArray && (!s || Object.keys(s).length === 0)) {
          return this.initSingle(p, s || {});
        }
        return s;
      },
      parameters,
      this._state
    );
    this._initialized = true;
  }

  /**
   * Initialize the optimizer state for a single parameter.
   * To be implemented by derived classes.
   * 
   * @param parameter - A single parameter that will be optimized
   * @param state - The optimizer's state dictionary for this parameter
   */
  protected abstract initSingle(parameter: MLXArray, state: Record<string, any>): void;

  /**
   * Apply gradients to parameters and return updated parameters.
   * 
   * @param gradients - A tree of gradients
   * @param parameters - A tree of parameters
   * @returns Updated parameters
   */
  applyGradients(gradients: Record<string, any>, parameters: Record<string, any>): Record<string, any> {
    if (!this._initialized) {
      this.init(gradients);
    }

    // Update any scheduled variables
    for (const [param, scheduler] of Object.entries(this._schedulers)) {
      this._state[param] = scheduler(this.step);
    }

    // Increment the step
    const currentStep = this.step.toTypedArray()[0] as number;
    this._state.step = zeros([], 'uint64');
    // Note: We would need to set the actual value here but that requires array construction from scalar
    // For now, this is a placeholder that needs proper implementation

    // Apply the update
    return treeMap(
      (gradient: any, parameter: any, state: any) => {
        if (gradient instanceof MLXArray && parameter instanceof MLXArray) {
          return this.applySingle(gradient, parameter, state);
        }
        return parameter;
      },
      gradients,
      parameters,
      this._state
    ) as Record<string, any>;
  }

  /**
   * Apply the optimizer update to a single parameter.
   * To be implemented by derived classes.
   * 
   * @param gradient - The gradient for this parameter
   * @param parameter - The parameter to update
   * @param state - The optimizer's state for this parameter
   * @returns Updated parameter
   */
  protected abstract applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray;

  /**
   * Get the optimizer's state dictionary
   */
  get state(): Record<string, any> {
    return this._state;
  }

  /**
   * Set the optimizer's state dictionary
   */
  set state(state: Record<string, any>) {
    this._initialized = false;
    this._state = state;
  }

  /**
   * Get the current step count
   */
  get step(): MLXArray {
    return this._state.step;
  }

  /**
   * Get the learning rate
   */
  get learningRate(): MLXArray {
    return this._state.learning_rate;
  }

  /**
   * Set the learning rate
   */
  set learningRate(lr: number | MLXArray) {
    if (typeof lr === 'number') {
      // Create a scalar array - this needs proper scalar construction support
      this._state.learning_rate = zeros([]);
    } else {
      this._state.learning_rate = lr;
    }
  }

  /**
   * Helper to optionally put a parameter on a schedule
   */
  protected _maybeSchedule(name: string, param: SchedulableParam): void {
    if (typeof param === 'function') {
      this._schedulers[name] = param;
      const parameter = param(this.step);
      this._state[name] = parameter;
    } else {
      // Create scalar array from number - needs proper scalar construction
      this._state[name] = zeros([]);
    }
  }
}

/**
 * SGD Optimizer Options
 */
export interface SGDOptions {
  /** The learning rate */
  learningRate: SchedulableParam;
  /** The momentum strength (default: 0) */
  momentum?: number;
  /** The weight decay (L2 penalty) (default: 0) */
  weightDecay?: number;
  /** Dampening for momentum (default: 0) */
  dampening?: number;
  /** Enables Nesterov momentum (default: false) */
  nesterov?: boolean;
}

/**
 * The stochastic gradient descent optimizer.
 * 
 * Updates a parameter w with a gradient g as follows:
 * 
 *   v_{t+1} = μ * v_t + (1 - τ) * g_t
 *   w_{t+1} = w_t - λ * v_{t+1}
 * 
 * where λ is the learning rate, μ is the momentum strength, and τ is the dampening.
 * 
 * @example
 * ```typescript
 * const optimizer = new SGD({ learningRate: 0.01, momentum: 0.9 });
 * // ... during training:
 * // const updatedParams = optimizer.applyGradients(gradients, parameters);
 * ```
 */
export class SGD extends Optimizer {
  momentum: number;
  weightDecay: number;
  dampening: number;
  nesterov: boolean;

  constructor(options: SGDOptions) {
    super();

    const { learningRate, momentum = 0.0, weightDecay = 0.0, dampening = 0.0, nesterov = false } = options;

    if (nesterov && (momentum <= 0 || dampening !== 0)) {
      throw new Error('Nesterov momentum requires a momentum and zero dampening.');
    }

    this._maybeSchedule('learning_rate', learningRate);
    this.momentum = momentum;
    this.weightDecay = weightDecay;
    this.dampening = dampening;
    this.nesterov = nesterov;
  }

  protected initSingle(parameter: MLXArray, state: Record<string, any>): void {
    // Initialize velocity with zerosLike
    state.v = zerosLike(parameter);
  }

  protected applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray {
    // Note: This is a simplified implementation that demonstrates the structure.
    // The full implementation requires additional operations that are not yet
    // available in the Node.js bindings:
    // - Subtraction operator
    // - Scalar multiplication with arrays
    // - astype() for dtype conversion
    //
    // This code will need to be updated once those operations are available.
    
    throw new Error(
      'SGD.applySingle is not yet fully implemented. ' +
      'This requires additional core operations (subtract, scalar ops, etc.) ' +
      'that are not yet available in the Node.js MLX bindings.'
    );

    // This is what the implementation should look like once operations are available:
    /*
    let grad = gradient;
    
    // Apply weight decay if configured
    if (this.weightDecay !== 0) {
      grad = add(grad, multiply(array(this.weightDecay), parameter));
    }

    // If no momentum, do simple update
    if (this.momentum <= 0) {
      const lr = this.learningRate; // Would need .astype(gradient.dtype) 
      return subtract(parameter, multiply(lr, grad));
    }

    // Get velocity from state
    let v = state.v || zerosLike(parameter);
    
    // Update velocity
    v = multiply(array(this.momentum), v);
    if (this.dampening > 0) {
      v = add(v, multiply(array(1 - this.dampening), grad));
    } else {
      v = add(v, grad);
    }

    // Compute update
    let update: MLXArray;
    if (this.nesterov) {
      update = add(grad, multiply(array(this.momentum), v));
    } else {
      update = v;
    }

    // Store velocity
    state.v = v;

    // Apply update
    const lr = this.learningRate; // Would need .astype(gradient.dtype)
    return subtract(parameter, multiply(lr, update));
    */
  }
}

/**
 * Adam Optimizer Options
 */
export interface AdamOptions {
  /** The learning rate */
  learningRate: SchedulableParam;
  /** The coefficients (β₁, β₂) used for computing running averages of gradient and its square (default: [0.9, 0.999]) */
  betas?: [number, number];
  /** The term ε added to the denominator to improve numerical stability (default: 1e-8) */
  eps?: number;
  /** If true, bias correction is applied (default: false) */
  biasCorrection?: boolean;
}

/**
 * The Adam optimizer.
 *
 * Implements the Adam algorithm from "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015).
 *
 * The algorithm updates parameters as follows:
 *
 *   m_{t+1} = β₁ * m_t + (1 - β₁) * g_t
 *   v_{t+1} = β₂ * v_t + (1 - β₂) * g_t²
 *   w_{t+1} = w_t - λ * m_{t+1} / (√v_{t+1} + ε)
 *
 * where λ is the learning rate, m_t and v_t are the first and second moment estimates,
 * g_t is the gradient, and ε is a small constant for numerical stability.
 *
 * @example
 * ```typescript
 * const optimizer = new Adam({ learningRate: 0.001 });
 * // ... during training:
 * // const updatedParams = optimizer.applyGradients(gradients, parameters);
 * ```
 */
export class Adam extends Optimizer {
  betas: [number, number];
  eps: number;
  biasCorrection: boolean;

  constructor(options: AdamOptions) {
    super();

    const {
      learningRate,
      betas = [0.9, 0.999],
      eps = 1e-8,
      biasCorrection = false
    } = options;

    this._maybeSchedule('learning_rate', learningRate);
    this.betas = betas;
    this.eps = eps;
    this.biasCorrection = biasCorrection;
  }

  protected initSingle(parameter: MLXArray, state: Record<string, any>): void {
    // Initialize first moment (m) and second moment (v) with zerosLike
    state.m = zerosLike(parameter);
    state.v = zerosLike(parameter);
  }

  protected applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray {
    // Note: This is a simplified implementation that demonstrates the structure.
    // The full implementation requires additional operations that are not yet
    // available in the Node.js bindings:
    // - Subtraction operator
    // - Division operator
    // - square() for computing g²
    // - sqrt() for computing √v
    // - rsqrt() for bias correction
    // - Power operator for computing β^step
    // - astype() for dtype conversion
    //
    // This code will need to be updated once those operations are available.

    throw new Error(
      'Adam.applySingle is not yet fully implemented. ' +
      'This requires additional core operations (subtract, divide, square, sqrt, rsqrt, power) ' +
      'that are not yet available in the Node.js MLX bindings.'
    );

    // This is what the implementation should look like once operations are available:
    /*
    const lr = this.learningRate; // Would need .astype(gradient.dtype)
    const [b1, b2] = this.betas;
    const eps = this.eps;
    const biasCorrection = this.biasCorrection;
    const step = this.step;

    // Get moments from state
    let m = state.m;
    let v = state.v;

    // Update biased first moment estimate: m = β₁ * m + (1 - β₁) * g
    m = add(multiply(b1, m), multiply(1 - b1, gradient));

    // Update biased second moment estimate: v = β₂ * v + (1 - β₂) * g²
    v = add(multiply(b2, v), multiply(1 - b2, square(gradient)));

    // Store updated moments
    state.m = m;
    state.v = v;

    if (biasCorrection) {
      // Compute bias-corrected learning rate: lr / (1 - β₁^step)
      const c1 = divide(lr, subtract(1, power(b1, step))); // .astype(gradient.dtype)

      // Compute bias correction for second moment: 1 / √(1 - β₂^step)
      const c2 = rsqrt(subtract(1, power(b2, step))); // .astype(gradient.dtype)

      // Compute update: c1 * m / (√v * c2 + ε)
      const numerator = multiply(c1, m);
      const denominator = add(multiply(sqrt(v), c2), eps);
      const update = divide(numerator, denominator);

      return subtract(parameter, update);
    } else {
      // Compute update without bias correction: lr * m / (√v + ε)
      const update = divide(multiply(lr, m), add(sqrt(v), eps));
      return subtract(parameter, update);
    }
    */
  }
}

export interface LionOptions {
  learningRate: SchedulableParam;
  betas?: [number, number];
  weightDecay?: number;
}

export class Lion extends Optimizer {
  betas: [number, number];
  weightDecay: number;

  constructor(options: LionOptions) {
    super();
    const { learningRate, betas = [0.9, 0.99], weightDecay = 0 } = options;
    this._maybeSchedule('learning_rate', learningRate);
    this.betas = betas;
    this.weightDecay = weightDecay;
  }

  protected initSingle(parameter: MLXArray, state: Record<string, any>): void {
    state.m = zerosLike(parameter);
  }

  protected applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>,
  ): MLXArray {
    const lr = this.learningRate;
    const [beta1, beta2] = this.betas;
    const momentum = state.m;

    const c = add(multiply(beta1, momentum), multiply(1 - beta1, gradient));
    state.m = add(multiply(beta2, momentum), multiply(1 - beta2, gradient));

    let updatedParameter = parameter;
    if (this.weightDecay > 0) {
      updatedParameter = multiply(
        subtract(1, multiply(lr, this.weightDecay)),
        parameter,
      );
    }

    return subtract(updatedParameter, multiply(lr, sign(c)));
  }
}

/**
 * RMSprop Optimizer Options
 */
export interface RMSpropOptions {
  /** The learning rate */
  learningRate: SchedulableParam;
  /** The smoothing constant α (default: 0.99) */
  alpha?: number;
  /** The term ε added to the denominator to improve numerical stability (default: 1e-8) */
  eps?: number;
}

/**
 * The RMSprop optimizer.
 *
 * RMSprop is an adaptive learning rate optimization algorithm that
 * uses a moving average of squared gradients to normalize the gradient.
 *
 * Reference: Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop,
 * coursera: Neural networks for machine learning
 *
 * The algorithm updates parameters as follows:
 *
 *   v_{t+1} = α * v_t + (1 - α) * g_t²
 *   w_{t+1} = w_t - λ * g_t / (√v_{t+1} + ε)
 *
 * where λ is the learning rate, α is the smoothing constant,
 * g_t is the gradient, v_t is the moving average of squared gradients,
 * and ε is a small constant for numerical stability.
 *
 * @example
 * ```typescript
 * const optimizer = new RMSprop({ learningRate: 0.001, alpha: 0.99 });
 * // ... during training:
 * // const updatedParams = optimizer.applyGradients(gradients, parameters);
 * ```
 */
export class RMSprop extends Optimizer {
  alpha: number;
  eps: number;

  constructor(options: RMSpropOptions) {
    super();

    const { learningRate, alpha = 0.99, eps = 1e-8 } = options;

    if (alpha < 0.0) {
      throw new Error(`RMSprop alpha should be >=0, ${alpha} was provided instead`);
    }
    if (eps <= 0.0) {
      throw new Error(`RMSprop epsilon should be >0, ${eps} was provided instead`);
    }

    this._maybeSchedule('learning_rate', learningRate);
    this.alpha = alpha;
    this.eps = eps;
  }

  protected initSingle(parameter: MLXArray, state: Record<string, any>): void {
    // Initialize moving average of squared gradients with zerosLike
    state.v = zerosLike(parameter);
  }

  protected applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray {
    const lr = this.learningRate;
    const alpha = this.alpha;
    const eps = this.eps;

    // Get moving average of squared gradients from state
    let v = state.v;

    // Update moving average: v = α * v + (1 - α) * g²
    const gradSquared = square(gradient);
    v = add(multiply(alpha, v), multiply(1 - alpha, gradSquared));

    // Store updated moving average
    state.v = v;

    // Compute update: w = w - lr * g / (√v + ε)
    const sqrtV = sqrt(v);
    const denominator = add(sqrtV, eps);
    const update = divide(gradient, denominator);
    return subtract(parameter, multiply(lr, update));
  }
}

/**
 * Filter function type - takes the full path and the associated leaf value
 * (e.g., a parameter or its gradient) and returns true if it should be
 * considered for the optimizer.
 */
export type OptimizerFilter = (path: string, value: any) => boolean;

/**
 * MultiOptimizer Options
 */
export interface MultiOptimizerOptions {
  /** A list of optimizers to delegate to */
  optimizers: Optimizer[];
  /** A list of predicates/filters that should be one less than the provided optimizers */
  filters?: OptimizerFilter[];
}

/**
 * Wraps a list of optimizers with corresponding weight predicates/filters
 * to make it easy to use different optimizers for different weights.
 * 
 * The predicates take the full "path" of the weight and the weight itself and
 * return true if it should be considered for this optimizer. The last
 * optimizer in the list is a fallback optimizer and no predicate should be
 * given for it.
 * 
 * @example
 * ```typescript
 * const opt1 = new Adam({ learningRate: 0.001 });
 * const opt2 = new SGD({ learningRate: 0.01 });
 * 
 * // Use Adam for 'weight' parameters, SGD for everything else
 * const multiOpt = new MultiOptimizer({
 *   optimizers: [opt1, opt2],
 *   filters: [(path, _param) => path.includes('weight')]
 * });
 * ```
 */
export class MultiOptimizer extends Optimizer {
  optimizers: Optimizer[];
  filters: OptimizerFilter[];

  constructor(options: MultiOptimizerOptions) {
    super();
    const { optimizers, filters = [] } = options;

    this._state = {};

    if (filters.length !== optimizers.length - 1) {
      throw new Error(
        `Given ${filters.length} filters but ${optimizers.length - 1} needed.`
      );
    }

    this.optimizers = optimizers;
    // Add a catch-all filter for the last optimizer
    this.filters = [...filters, () => true];
  }

  /**
   * Split a parameter/gradient dictionary according to the filters.
   * Each optimizer gets the parameters/gradients that match its filter.
   */
  private _splitDictionary(tree: Record<string, any>): Record<string, any>[] {
    if (this.optimizers.length === 1) {
      return [tree];
    }

    // Initialize parts array with empty arrays for each optimizer
    const parts: Array<[string, any]>[] = this.optimizers.map(() => []);

    // Flatten the tree to get (path, value) pairs
    const flatTree = treeFlatten(tree) as Array<[string, any]>;

    // Assign each (path, value) pair to the first matching filter
    for (const [key, value] of flatTree) {
      for (let i = 0; i < this.filters.length; i++) {
        if (this.filters[i](key, value)) {
          parts[i].push([key, value]);
          break;
        }
      }
    }

    // Unflatten each part back into a tree
    return parts.map((part) => treeUnflatten(part) as Record<string, any>);
  }

  /**
   * Initialize the optimizer's state.
   * Delegates to each sub-optimizer with its portion of parameters.
   */
  init(parameters: Record<string, any>): void {
    const splits = this._splitDictionary(parameters);
    for (let i = 0; i < this.optimizers.length; i++) {
      this.optimizers[i].init(splits[i]);
    }
  }

  /**
   * Apply gradients to parameters and return updated parameters.
   * Delegates to each sub-optimizer and merges the results.
   */
  applyGradients(
    gradients: Record<string, any>,
    parameters: Record<string, any>
  ): Record<string, any> {
    const gradientSplits = this._splitDictionary(gradients);
    const parameterSplits = this._splitDictionary(parameters);
    let result: Record<string, any> = {};

    for (let i = 0; i < this.optimizers.length; i++) {
      const updated = this.optimizers[i].applyGradients(
        gradientSplits[i],
        parameterSplits[i]
      );
      result = treeMerge(result, updated) as Record<string, any>;
    }

    return result;
  }

  /**
   * Not needed for MultiOptimizer as it delegates to sub-optimizers.
   */
  protected initSingle(parameter: MLXArray, state: Record<string, any>): void {
    // This method is required by the abstract base class but not used by MultiOptimizer
    // since we override init() directly
  }

  /**
   * Not needed for MultiOptimizer as it delegates to sub-optimizers.
   */
  protected applySingle(
    gradient: MLXArray,
    parameter: MLXArray,
    state: Record<string, any>
  ): MLXArray {
    // This method is required by the abstract base class but not used by MultiOptimizer
    // since we override applyGradients() directly
    throw new Error('MultiOptimizer.applySingle should not be called directly');
  }

  /**
   * Get the combined state from all sub-optimizers.
   */
  get state(): Record<string, any> {
    return {
      states: this.optimizers.map((opt) => opt.state),
    };
  }

  /**
   * Set the state for all sub-optimizers.
   */
  set state(state: Record<string, any>) {
    if (!state.states || state.states.length !== this.optimizers.length) {
      throw new Error('Invalid state provided');
    }

    for (let i = 0; i < this.optimizers.length; i++) {
      this.optimizers[i].state = state.states[i];
    }
  }

  /**
   * Get the learning rate from the first optimizer.
   */
  get learningRate(): MLXArray {
    return this.optimizers[0].learningRate;
  }

  /**
   * Set the learning rate for all sub-optimizers.
   */
  set learningRate(lr: number | MLXArray) {
    for (const opt of this.optimizers) {
      opt.learningRate = lr;
    }
  }
}

export default {
  Optimizer,
  SGD,
  Adam,
  Lion,
  RMSprop,
  MultiOptimizer,
};

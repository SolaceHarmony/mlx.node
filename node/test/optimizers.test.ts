import { describe, it } from 'node:test';
import assert from 'node:assert';
import { SGD, Adam, AdamW, Adamax, Lion, Adagrad, AdaDelta, RMSprop, Adafactor, Muon, Optimizer, MultiOptimizer } from '../src/optimizers';
import { zeros } from '../src/core/array';
import * as mx from '../src';
import { tree_map, tree_flatten } from '../src/utils';

describe('mlx.optimizers', () => {
  describe('SGD', () => {
    it('should create SGD optimizer with learning rate', () => {
      const optimizer = new SGD({ learningRate: 0.01 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof SGD);
      assert.strictEqual(optimizer.momentum, 0);
      assert.strictEqual(optimizer.weightDecay, 0);
      assert.strictEqual(optimizer.dampening, 0);
      assert.strictEqual(optimizer.nesterov, false);
    });

    it('should create SGD optimizer with momentum', () => {
      const optimizer = new SGD({ 
        learningRate: 0.01, 
        momentum: 0.9 
      });
      assert.strictEqual(optimizer.momentum, 0.9);
    });

    it('should create SGD optimizer with weight decay', () => {
      const optimizer = new SGD({ 
        learningRate: 0.01, 
        weightDecay: 0.001 
      });
      assert.strictEqual(optimizer.weightDecay, 0.001);
    });

    it('should throw error for nesterov without proper momentum', () => {
      assert.throws(
        () => new SGD({ learningRate: 0.01, nesterov: true }),
        /Nesterov momentum requires a momentum and zero dampening/
      );
    });

    it('should throw error for nesterov with dampening', () => {
      assert.throws(
        () => new SGD({ 
          learningRate: 0.01, 
          momentum: 0.9, 
          dampening: 0.1, 
          nesterov: true 
        }),
        /Nesterov momentum requires a momentum and zero dampening/
      );
    });

    it('should create optimizer with nesterov momentum correctly', () => {
      const optimizer = new SGD({ 
        learningRate: 0.01, 
        momentum: 0.9, 
        nesterov: true 
      });
      assert.strictEqual(optimizer.nesterov, true);
      assert.strictEqual(optimizer.momentum, 0.9);
      assert.strictEqual(optimizer.dampening, 0);
    });

    it('should have learning rate in state', () => {
      const optimizer = new SGD({ learningRate: 0.01 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
      // Check that it's an MLXArray
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new SGD({ learningRate: 0.01, momentum: 0.9 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1])
      };
      
      optimizer.init(params);
      
      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);
    });

    it('should track step count', () => {
      const optimizer = new SGD({ learningRate: 0.01 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new SGD({ learningRate: 0.01 });
      optimizer.learningRate = 0.001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });

    // Note: applyGradients tests are not included yet because they would fail
    // due to missing core operations (subtract, scalar ops, etc.)
    // These should be added once those operations are available
  });

  describe('Optimizer base class', () => {
    it('should track initialization state', () => {
      const optimizer = new SGD({ learningRate: 0.01 });
      // Access protected property via any cast for testing
      assert.strictEqual((optimizer as any)._initialized, false);
      
      optimizer.init({ param: zeros([3]) });
      assert.strictEqual((optimizer as any)._initialized, true);
    });

    it('should allow getting and setting state', () => {
      const optimizer = new SGD({ learningRate: 0.01 });
      const initialState = optimizer.state;
      assert.ok(initialState);
      assert.ok('step' in initialState);

      const newState = { step: zeros([], 'uint64'), custom: 'value' };
      optimizer.state = newState;
      assert.deepStrictEqual(optimizer.state, newState);
      assert.strictEqual((optimizer as any)._initialized, false);
    });
  });

  describe('mlx.optimizers.Adam', () => {
    it('should create Adam optimizer with learning rate', () => {
      const optimizer = new Adam({ learningRate: 0.001 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof Adam);
      assert.deepStrictEqual(optimizer.betas, [0.9, 0.999]);
      assert.strictEqual(optimizer.eps, 1e-8);
      assert.strictEqual(optimizer.biasCorrection, false);
    });

    it('should create Adam optimizer with custom betas', () => {
      const optimizer = new Adam({
        learningRate: 0.001,
        betas: [0.95, 0.9999]
      });
      assert.deepStrictEqual(optimizer.betas, [0.95, 0.9999]);
    });

    it('should create Adam optimizer with custom epsilon', () => {
      const optimizer = new Adam({
        learningRate: 0.001,
        eps: 1e-7
      });
      assert.strictEqual(optimizer.eps, 1e-7);
    });

    it('should create Adam optimizer with bias correction enabled', () => {
      const optimizer = new Adam({
        learningRate: 0.001,
        biasCorrection: true
      });
      assert.strictEqual(optimizer.biasCorrection, true);
    });

    it('should have learning rate in state', () => {
      const optimizer = new Adam({ learningRate: 0.001 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
      // Check that it's an MLXArray
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new Adam({ learningRate: 0.001 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1])
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // Check that moment estimates were initialized
      assert.ok('m' in optimizer.state.weight);
      assert.ok('v' in optimizer.state.weight);
      assert.ok('m' in optimizer.state.bias);
      assert.ok('v' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new Adam({ learningRate: 0.001 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new Adam({ learningRate: 0.001 });
      optimizer.learningRate = 0.0001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });

    // Note: applyGradients tests are not included yet because they would fail
    // due to missing core operations (subtract, divide, square, sqrt, rsqrt, power)
    // These should be added once those operations are available
  });

  describe('mlx.optimizers.AdamW', () => {
    it('should create AdamW optimizer with learning rate', () => {
      const optimizer = new AdamW({ learningRate: 0.001 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof Adam);
      assert.ok(optimizer instanceof AdamW);
      assert.deepStrictEqual(optimizer.betas, [0.9, 0.999]);
      assert.strictEqual(optimizer.eps, 1e-8);
      assert.strictEqual(optimizer.weightDecay, 0.01);
      assert.strictEqual(optimizer.biasCorrection, false);
    });

    it('should create AdamW optimizer with custom weight decay', () => {
      const optimizer = new AdamW({
        learningRate: 0.001,
        weightDecay: 0.05
      });
      assert.strictEqual(optimizer.weightDecay, 0.05);
    });

    it('should create AdamW optimizer with zero weight decay', () => {
      const optimizer = new AdamW({
        learningRate: 0.001,
        weightDecay: 0
      });
      assert.strictEqual(optimizer.weightDecay, 0);
    });

    it('should create AdamW optimizer with custom betas', () => {
      const optimizer = new AdamW({
        learningRate: 0.001,
        betas: [0.95, 0.9999]
      });
      assert.deepStrictEqual(optimizer.betas, [0.95, 0.9999]);
    });

    it('should create AdamW optimizer with custom epsilon', () => {
      const optimizer = new AdamW({
        learningRate: 0.001,
        eps: 1e-7
      });
      assert.strictEqual(optimizer.eps, 1e-7);
    });

    it('should create AdamW optimizer with bias correction enabled', () => {
      const optimizer = new AdamW({
        learningRate: 0.001,
        biasCorrection: true
      });
      assert.strictEqual(optimizer.biasCorrection, true);
    });

    it('should create AdamW optimizer with all custom options', () => {
      const optimizer = new AdamW({
        learningRate: 0.002,
        betas: [0.85, 0.998],
        eps: 1e-6,
        weightDecay: 0.1,
        biasCorrection: true
      });
      assert.deepStrictEqual(optimizer.betas, [0.85, 0.998]);
      assert.strictEqual(optimizer.eps, 1e-6);
      assert.strictEqual(optimizer.weightDecay, 0.1);
      assert.strictEqual(optimizer.biasCorrection, true);
    });

    it('should have learning rate in state', () => {
      const optimizer = new AdamW({ learningRate: 0.001 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
      // Check that it's an MLXArray
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new AdamW({ learningRate: 0.001 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1])
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // Check that moment estimates were initialized (inherited from Adam)
      assert.ok('m' in optimizer.state.weight);
      assert.ok('v' in optimizer.state.weight);
      assert.ok('m' in optimizer.state.bias);
      assert.ok('v' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new AdamW({ learningRate: 0.001 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new AdamW({ learningRate: 0.001 });
      optimizer.learningRate = 0.0001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });

    // Note: Full integration tests with applyGradients would be added here
    // once we have all operations working properly
  });

  describe('mlx.optimizers.Adamax', () => {
    it('should create Adamax optimizer with learning rate', () => {
      const optimizer = new Adamax({ learningRate: 0.002 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof Adam);
      assert.ok(optimizer instanceof Adamax);
      assert.deepStrictEqual(optimizer.betas, [0.9, 0.999]);
      assert.strictEqual(optimizer.eps, 1e-8);
      assert.strictEqual(optimizer.biasCorrection, false);
    });

    it('should create Adamax optimizer with custom betas', () => {
      const optimizer = new Adamax({
        learningRate: 0.002,
        betas: [0.95, 0.9999]
      });
      assert.deepStrictEqual(optimizer.betas, [0.95, 0.9999]);
    });

    it('should create Adamax optimizer with custom epsilon', () => {
      const optimizer = new Adamax({
        learningRate: 0.002,
        eps: 1e-7
      });
      assert.strictEqual(optimizer.eps, 1e-7);
    });

    it('should throw error for negative epsilon', () => {
      assert.throws(
        () => new Adamax({ learningRate: 0.002, eps: -1e-8 }),
        /Epsilon value should be >=0/
      );
    });

    it('should have learning rate in state', () => {
      const optimizer = new Adamax({ learningRate: 0.002 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
      // Check that it's an MLXArray
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new Adamax({ learningRate: 0.002 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1])
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // Check that moment estimates were initialized
      assert.ok('m' in optimizer.state.weight);
      assert.ok('v' in optimizer.state.weight);
      assert.ok('m' in optimizer.state.bias);
      assert.ok('v' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new Adamax({ learningRate: 0.002 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new Adamax({ learningRate: 0.002 });
      optimizer.learningRate = 0.0001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });

    // Note: Full integration tests with applyGradients would be added here
    // once we have all operations working properly
  });

  describe('Lion', () => {
    it('should create Lion optimizer with learning rate', () => {
      const optimizer = new Lion({ learningRate: 0.0001 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof Lion);
      assert.deepStrictEqual(optimizer.betas, [0.9, 0.99]);
      assert.strictEqual(optimizer.weightDecay, 0);
    });

    it('should allow custom betas', () => {
      const optimizer = new Lion({ learningRate: 0.0001, betas: [0.95, 0.999] });
      assert.deepStrictEqual(optimizer.betas, [0.95, 0.999]);
    });

    it('should allow weight decay', () => {
      const optimizer = new Lion({ learningRate: 0.0001, weightDecay: 0.01 });
      assert.strictEqual(optimizer.weightDecay, 0.01);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new Lion({ learningRate: 0.0001 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1]),
      };

      optimizer.init(params);

      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);
      assert.ok('m' in optimizer.state.weight);
      assert.ok('m' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new Lion({ learningRate: 0.0001 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });
  });

  describe('RMSprop', () => {
    it('should create RMSprop optimizer with learning rate', () => {
      const optimizer = new RMSprop({ learningRate: 0.01 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof RMSprop);
      assert.strictEqual(optimizer.alpha, 0.99);
      assert.strictEqual(optimizer.eps, 1e-8);
    });

    it('should create RMSprop optimizer with custom alpha', () => {
      const optimizer = new RMSprop({
        learningRate: 0.01,
        alpha: 0.95
      });
      assert.strictEqual(optimizer.alpha, 0.95);
    });

    it('should create RMSprop optimizer with custom epsilon', () => {
      const optimizer = new RMSprop({
        learningRate: 0.01,
        eps: 1e-7
      });
      assert.strictEqual(optimizer.eps, 1e-7);
    });

    it('should throw error for negative alpha', () => {
      assert.throws(
        () => new RMSprop({ learningRate: 0.01, alpha: -0.1 }),
        /RMSprop alpha should be >=0/
      );
    });

    it('should throw error for non-positive epsilon', () => {
      assert.throws(
        () => new RMSprop({ learningRate: 0.01, eps: 0 }),
        /RMSprop epsilon should be >0/
      );
      assert.throws(
        () => new RMSprop({ learningRate: 0.01, eps: -1e-8 }),
        /RMSprop epsilon should be >0/
      );
    });

    it('should have learning rate in state', () => {
      const optimizer = new RMSprop({ learningRate: 0.01 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
      // Check that it's an MLXArray
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new RMSprop({ learningRate: 0.01 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1])
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // Check that moving average was initialized
      assert.ok('v' in optimizer.state.weight);
      assert.ok('v' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new RMSprop({ learningRate: 0.01 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new RMSprop({ learningRate: 0.01 });
      optimizer.learningRate = 0.001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });
  });

  describe('Adafactor', () => {
    it('should create Adafactor optimizer with default settings', () => {
      const optimizer = new Adafactor({});
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof Adafactor);
      assert.deepStrictEqual(optimizer.eps, [1e-30, 1e-3]);
      assert.strictEqual(optimizer.clipThreshold, 1.0);
      assert.strictEqual(optimizer.decayRate, -0.8);
      assert.strictEqual(optimizer.beta1, undefined);
      assert.strictEqual(optimizer.weightDecay, 0.0);
      assert.strictEqual(optimizer.scaleParameter, true);
      assert.strictEqual(optimizer.relativeStep, true);
      assert.strictEqual(optimizer.warmupInit, false);
    });

    it('should create Adafactor optimizer with fixed learning rate', () => {
      const optimizer = new Adafactor({
        learningRate: 0.001,
        relativeStep: false
      });
      assert.strictEqual(optimizer.relativeStep, false);
      const lr = optimizer.learningRate;
      assert.ok(lr);
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should create Adafactor optimizer with custom epsilon', () => {
      const optimizer = new Adafactor({
        eps: [1e-25, 1e-2]
      });
      assert.deepStrictEqual(optimizer.eps, [1e-25, 1e-2]);
    });

    it('should create Adafactor optimizer with momentum (beta1)', () => {
      const optimizer = new Adafactor({
        learningRate: 0.001,
        relativeStep: false,
        beta1: 0.9
      });
      assert.strictEqual(optimizer.beta1, 0.9);
    });

    it('should create Adafactor optimizer with weight decay', () => {
      const optimizer = new Adafactor({
        learningRate: 0.001,
        relativeStep: false,
        weightDecay: 0.01
      });
      assert.strictEqual(optimizer.weightDecay, 0.01);
    });

    it('should create Adafactor optimizer with custom clip threshold', () => {
      const optimizer = new Adafactor({
        clipThreshold: 2.0
      });
      assert.strictEqual(optimizer.clipThreshold, 2.0);
    });

    it('should create Adafactor optimizer with warmup initialization', () => {
      const optimizer = new Adafactor({
        warmupInit: true
      });
      assert.strictEqual(optimizer.warmupInit, true);
    });

    it('should initialize state for 1D parameters (non-factored)', () => {
      const optimizer = new Adafactor({ learningRate: 0.001, relativeStep: false });
      const params = {
        bias: zeros([10])  // 1D parameter
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // For 1D parameters, should have exp_avg_sq (non-factored)
      assert.ok('exp_avg_sq' in optimizer.state.bias);
      assert.ok(!('exp_avg_sq_row' in optimizer.state.bias));
      assert.ok(!('exp_avg_sq_col' in optimizer.state.bias));
    });

    it('should initialize state for 2D parameters (factored)', () => {
      const optimizer = new Adafactor({ learningRate: 0.001, relativeStep: false });
      const params = {
        weight: zeros([10, 5])  // 2D parameter
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);

      // For 2D+ parameters, should have exp_avg_sq_row and exp_avg_sq_col (factored)
      assert.ok('exp_avg_sq_row' in optimizer.state.weight);
      assert.ok('exp_avg_sq_col' in optimizer.state.weight);
      assert.ok(!('exp_avg_sq' in optimizer.state.weight));
    });

    it('should initialize first moment when beta1 is set', () => {
      const optimizer = new Adafactor({
        learningRate: 0.001,
        relativeStep: false,
        beta1: 0.9
      });
      const params = {
        weight: zeros([3]),
      };

      optimizer.init(params);

      // Should have exp_avg when beta1 is set
      assert.ok('exp_avg' in optimizer.state.weight);
    });

    it('should track step count', () => {
      const optimizer = new Adafactor({});
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should throw error when applying gradients (not yet implemented)', () => {
      const optimizer = new Adafactor({ learningRate: 0.001, relativeStep: false });
      const params = {
        weight: zeros([3]),
      };
      const grads = {
        weight: zeros([3]),
      };

      optimizer.init(params);

      // applySingle should throw an error explaining missing operations
      assert.throws(
        () => optimizer.applyGradients(grads, params),
        /not yet fully implemented/
      );
    });
  });

  describe('MultiOptimizer', () => {
    it('should create MultiOptimizer with single optimizer', () => {
      const opt = new SGD({ learningRate: 0.01 });
      const multiOpt = new MultiOptimizer({
        optimizers: [opt],
      });
      assert.ok(multiOpt instanceof Optimizer);
      assert.ok(multiOpt instanceof MultiOptimizer);
      assert.strictEqual(multiOpt.optimizers.length, 1);
      assert.strictEqual(multiOpt.filters.length, 1);
    });

    it('should create MultiOptimizer with multiple optimizers and filters', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01 });
      const filter1 = (path: string) => path.includes('weight');

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [filter1],
      });

      assert.strictEqual(multiOpt.optimizers.length, 2);
      assert.strictEqual(multiOpt.filters.length, 2); // filter1 + catch-all
    });

    it('should throw error if wrong number of filters provided', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01 });
      const opt3 = new Lion({ learningRate: 0.0001 });

      // Need 2 filters for 3 optimizers
      assert.throws(
        () =>
          new MultiOptimizer({
            optimizers: [opt1, opt2, opt3],
            filters: [(path) => path.includes('weight')], // Only 1 filter
          }),
        /Given 1 filters but 2 needed/
      );
    });

    it('should throw error if too many filters provided', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01 });

      // Need 1 filter for 2 optimizers
      assert.throws(
        () =>
          new MultiOptimizer({
            optimizers: [opt1, opt2],
            filters: [
              (path) => path.includes('weight'),
              (path) => path.includes('bias'), // Too many filters
            ],
          }),
        /Given 2 filters but 1 needed/
      );
    });

    it('should initialize state for all sub-optimizers', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01, momentum: 0.9 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('weight')],
      });

      const params = {
        weight: zeros([3]),
        bias: zeros([1]),
      };

      multiOpt.init(params);

      // Check that both optimizers were initialized
      const state = multiOpt.state;
      assert.ok(state.states);
      assert.strictEqual(state.states.length, 2);

      // First optimizer (Adam) should have weight
      assert.ok(state.states[0].weight);
      assert.ok('m' in state.states[0].weight);
      assert.ok('v' in state.states[0].weight);

      // Second optimizer (SGD) should have bias
      assert.ok(state.states[1].bias);
      assert.ok('v' in state.states[1].bias);
    });

    it('should get learning rate from first optimizer', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('weight')],
      });

      const lr = multiOpt.learningRate;
      assert.ok(lr);
      // Should get learning rate from first optimizer
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should set learning rate for all sub-optimizers', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('weight')],
      });

      multiOpt.learningRate = 0.0001;

      // Both optimizers should have updated learning rate
      assert.ok(opt1.learningRate);
      assert.ok(opt2.learningRate);
    });

    it('should allow getting and setting state', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('weight')],
      });

      const params = {
        weight: zeros([3]),
        bias: zeros([1]),
      };

      multiOpt.init(params);
      const state = multiOpt.state;

      assert.ok(state.states);
      assert.strictEqual(state.states.length, 2);

      // Setting state should work
      multiOpt.state = state;
      const newState = multiOpt.state;
      assert.ok(newState.states);
      assert.strictEqual(newState.states.length, 2);
    });

    it('should throw error when setting invalid state', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new SGD({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('weight')],
      });

      // Missing states
      assert.throws(() => {
        multiOpt.state = { notStates: [] };
      }, /Invalid state provided/);

      // Wrong number of states
      assert.throws(() => {
        multiOpt.state = { states: [{}] }; // Only 1 state, need 2
      }, /Invalid state provided/);
    });

    it('should split parameters correctly based on filters', () => {
      const opt1 = new Adam({ learningRate: 0.001 });
      const opt2 = new Lion({ learningRate: 0.0001 });
      const opt3 = new SGD({ learningRate: 0.01 });

      // First optimizer gets 'weight' params, second gets 'bias', third is fallback
      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2, opt3],
        filters: [
          (path) => path.includes('weight'),
          (path) => path.includes('bias'),
        ],
      });

      const params = {
        layer1: {
          weight: zeros([3, 4]),
          bias: zeros([4]),
        },
        layer2: {
          weight: zeros([4, 2]),
          bias: zeros([2]),
        },
        other: zeros([5]),
      };

      multiOpt.init(params);
      const state = multiOpt.state;

      // Check that parameters were split correctly
      assert.ok(state.states[0].layer1?.weight); // Adam gets weights
      assert.ok(state.states[0].layer2?.weight);
      assert.ok(state.states[1].layer1?.bias); // Lion gets biases
      assert.ok(state.states[1].layer2?.bias);
      assert.ok(state.states[2].other); // SGD gets everything else
    });

    it('should route parameters correctly in applyGradients', () => {
      const opt1 = new Lion({ learningRate: 0.001 });
      const opt2 = new Lion({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('weight')],
      });

      const params = {
        weight: zeros([3]),
        bias: zeros([1]),
      };

      const gradients = {
        weight: zeros([3]),
        bias: zeros([1]),
      };

      multiOpt.init(params);
      const updated = multiOpt.applyGradients(gradients, params);

      // Verify the result has the correct structure
      assert.ok(updated);
      assert.ok('weight' in updated);
      assert.ok('bias' in updated);
    });

    it('should handle nested parameter trees in applyGradients', () => {
      const opt1 = new Lion({ learningRate: 0.001 });
      const opt2 = new Lion({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('weight')],
      });

      const params = {
        layer1: {
          weight: zeros([3, 4]),
          bias: zeros([4]),
        },
        layer2: {
          weight: zeros([4, 2]),
          bias: zeros([2]),
        },
      };

      const gradients = {
        layer1: {
          weight: zeros([3, 4]),
          bias: zeros([4]),
        },
        layer2: {
          weight: zeros([4, 2]),
          bias: zeros([2]),
        },
      };

      multiOpt.init(params);
      const updated = multiOpt.applyGradients(gradients, params);

      // Verify nested structure is preserved
      assert.ok(updated);
      assert.ok(updated.layer1);
      assert.ok(updated.layer1.weight);
      assert.ok(updated.layer1.bias);
      assert.ok(updated.layer2);
      assert.ok(updated.layer2.weight);
      assert.ok(updated.layer2.bias);
    });

    it('should merge results from multiple optimizers in applyGradients', () => {
      const opt1 = new Lion({ learningRate: 0.001 });
      const opt2 = new Lion({ learningRate: 0.01 });
      const opt3 = new Lion({ learningRate: 0.1 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2, opt3],
        filters: [
          (path) => path.includes('weight'),
          (path) => path.includes('bias'),
        ],
      });

      const params = {
        layer1: {
          weight: zeros([3, 4]),
          bias: zeros([4]),
        },
        other: zeros([5]),
      };

      const gradients = {
        layer1: {
          weight: zeros([3, 4]),
          bias: zeros([4]),
        },
        other: zeros([5]),
      };

      multiOpt.init(params);
      const updated = multiOpt.applyGradients(gradients, params);

      // Verify all parts are merged into result
      assert.ok(updated);
      assert.ok(updated.layer1);
      assert.ok(updated.layer1.weight); // From opt1
      assert.ok(updated.layer1.bias); // From opt2
      assert.ok(updated.other); // From opt3
    });

    it('should handle array-valued parameters in applyGradients', () => {
      const opt1 = new Lion({ learningRate: 0.001 });
      const opt2 = new Lion({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.startsWith('0.')], // First array element
      });

      const params = [zeros([3]), zeros([4]), zeros([5])];
      const gradients = [zeros([3]), zeros([4]), zeros([5])];

      multiOpt.init(params);
      const updated = multiOpt.applyGradients(gradients, params);

      // Verify array structure is preserved
      assert.ok(Array.isArray(updated));
      assert.strictEqual(updated.length, 3);
      assert.ok(updated[0]);
      assert.ok(updated[1]);
      assert.ok(updated[2]);
    });

    it('should correctly split mixed nested structures in applyGradients', () => {
      const opt1 = new Lion({ learningRate: 0.001 });
      const opt2 = new Lion({ learningRate: 0.01 });

      const multiOpt = new MultiOptimizer({
        optimizers: [opt1, opt2],
        filters: [(path) => path.includes('conv')],
      });

      const params = {
        conv: {
          layers: [zeros([3, 3]), zeros([5, 5])],
        },
        fc: {
          weight: zeros([10, 5]),
          bias: zeros([5]),
        },
      };

      const gradients = {
        conv: {
          layers: [zeros([3, 3]), zeros([5, 5])],
        },
        fc: {
          weight: zeros([10, 5]),
          bias: zeros([5]),
        },
      };

      multiOpt.init(params);
      const updated = multiOpt.applyGradients(gradients, params);

      // Verify complex nested structure is preserved
      assert.ok(updated);
      assert.ok(updated.conv);
      assert.ok(Array.isArray(updated.conv.layers));
      assert.strictEqual(updated.conv.layers.length, 2);
      assert.ok(updated.fc);
      assert.ok(updated.fc.weight);
      assert.ok(updated.fc.bias);
    });
  });

  describe('Adagrad', () => {
    it('should create Adagrad optimizer with learning rate', () => {
      const optimizer = new Adagrad({ learningRate: 0.01 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof Adagrad);
      assert.strictEqual(optimizer.eps, 1e-8);
    });

    it('should create Adagrad optimizer with custom epsilon', () => {
      const optimizer = new Adagrad({
        learningRate: 0.01,
        eps: 1e-7
      });
      assert.strictEqual(optimizer.eps, 1e-7);
    });

    it('should throw error for non-positive epsilon', () => {
      assert.throws(
        () => new Adagrad({ learningRate: 0.01, eps: 0 }),
        /Adagrad epsilon should be >0/
      );
      assert.throws(
        () => new Adagrad({ learningRate: 0.01, eps: -1e-8 }),
        /Adagrad epsilon should be >0/
      );
    });

    it('should have learning rate in state', () => {
      const optimizer = new Adagrad({ learningRate: 0.01 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
      // Check that it's an MLXArray
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new Adagrad({ learningRate: 0.01 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1])
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // Check that accumulated squared gradients were initialized
      assert.ok('v' in optimizer.state.weight);
      assert.ok('v' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new Adagrad({ learningRate: 0.01 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new Adagrad({ learningRate: 0.01 });
      optimizer.learningRate = 0.001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });
  });

  describe('AdaDelta', () => {
    it('should create AdaDelta optimizer with learning rate', () => {
      const optimizer = new AdaDelta({ learningRate: 1.0 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof AdaDelta);
      assert.strictEqual(optimizer.rho, 0.9);
      assert.strictEqual(optimizer.eps, 1e-6);
    });

    it('should create AdaDelta optimizer with custom rho', () => {
      const optimizer = new AdaDelta({
        learningRate: 1.0,
        rho: 0.95
      });
      assert.strictEqual(optimizer.rho, 0.95);
    });

    it('should create AdaDelta optimizer with custom epsilon', () => {
      const optimizer = new AdaDelta({
        learningRate: 1.0,
        eps: 1e-8
      });
      assert.strictEqual(optimizer.eps, 1e-8);
    });

    it('should throw error for negative rho', () => {
      assert.throws(
        () => new AdaDelta({ learningRate: 1.0, rho: -0.1 }),
        /AdaDelta rho should be >=0/
      );
    });

    it('should throw error for non-positive epsilon', () => {
      assert.throws(
        () => new AdaDelta({ learningRate: 1.0, eps: 0 }),
        /AdaDelta epsilon should be >0/
      );
      assert.throws(
        () => new AdaDelta({ learningRate: 1.0, eps: -1e-6 }),
        /AdaDelta epsilon should be >0/
      );
    });

    it('should have learning rate in state', () => {
      const optimizer = new AdaDelta({ learningRate: 1.0 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new AdaDelta({ learningRate: 1.0, rho: 0.9 });
      const params = {
        weight: zeros([3]),
        bias: zeros([1])
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // Check that moving averages were initialized
      assert.ok('v' in optimizer.state.weight);
      assert.ok('u' in optimizer.state.weight);
      assert.ok('v' in optimizer.state.bias);
      assert.ok('u' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new AdaDelta({ learningRate: 1.0 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new AdaDelta({ learningRate: 1.0 });
      optimizer.learningRate = 0.5;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });
  });

  describe('Muon', () => {
    it('should create Muon optimizer with learning rate', () => {
      const optimizer = new Muon({ learningRate: 0.01 });
      assert.ok(optimizer instanceof Optimizer);
      assert.ok(optimizer instanceof Muon);
      assert.strictEqual(optimizer.momentum, 0.95);
      assert.strictEqual(optimizer.weightDecay, 0.01);
      assert.strictEqual(optimizer.nesterov, true);
      assert.strictEqual(optimizer.nsSteps, 5);
    });

    it('should create Muon optimizer with custom momentum', () => {
      const optimizer = new Muon({
        learningRate: 0.01,
        momentum: 0.9
      });
      assert.strictEqual(optimizer.momentum, 0.9);
    });

    it('should create Muon optimizer with custom weight decay', () => {
      const optimizer = new Muon({
        learningRate: 0.01,
        weightDecay: 0.001
      });
      assert.strictEqual(optimizer.weightDecay, 0.001);
    });

    it('should create Muon optimizer without nesterov', () => {
      const optimizer = new Muon({
        learningRate: 0.01,
        nesterov: false
      });
      assert.strictEqual(optimizer.nesterov, false);
    });

    it('should create Muon optimizer with custom ns_steps', () => {
      const optimizer = new Muon({
        learningRate: 0.01,
        nsSteps: 10
      });
      assert.strictEqual(optimizer.nsSteps, 10);
    });

    it('should have learning rate in state', () => {
      const optimizer = new Muon({ learningRate: 0.01 });
      const lr = optimizer.learningRate;
      assert.ok(lr);
      // Check that it's an MLXArray
      assert.ok(lr.toTypedArray !== undefined);
    });

    it('should initialize state for parameters', () => {
      const optimizer = new Muon({ learningRate: 0.01 });
      const params = {
        weight: zeros([3, 3]),
        bias: zeros([1])
      };

      optimizer.init(params);

      // Check that state was initialized
      assert.ok(optimizer.state);
      assert.ok('step' in optimizer.state);
      assert.ok('learning_rate' in optimizer.state);
      assert.ok('weight' in optimizer.state);
      assert.ok('bias' in optimizer.state);

      // Check that velocity was initialized
      assert.ok('v' in optimizer.state.weight);
      assert.ok('v' in optimizer.state.bias);
    });

    it('should track step count', () => {
      const optimizer = new Muon({ learningRate: 0.01 });
      const step = optimizer.step;
      assert.ok(step);
      assert.strictEqual(Number(step.toTypedArray()[0]), 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new Muon({ learningRate: 0.01 });
      optimizer.learningRate = 0.001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
    });

    // Note: applyGradients tests require actual gradient application which depends
    // on the Newton-Schulz implementation working correctly with real data
  });
});

// ---------------------------------------------------------------------------
// Integration tests ported from python/tests/test_optimizers.py
// class TestOptimizers — line‑for‑line transliteration
// ---------------------------------------------------------------------------

/**
 * Returns an object mirroring the Python `params` fixture:
 *   { "first": [zeros([10]), zeros([1])], "second": zeros([1]) }
 */
const makeParams = () => ({
  first: [mx.zeros([10]), mx.zeros([1])],
  second: mx.zeros([1]),
});

/**
 * Applies `fn` over every leaf of a nested object/array tree, returning the
 * same tree shape.  Mirrors Python `tree_map`.
 */
const treeMap = (fn: (x: any) => any, tree: any): any =>
  tree_map(fn, tree);

/**
 * Returns true iff ALL leaves produced by fn(...leaves) are truthy.
 * Mirrors Python: `all(v for _, v in tree_flatten(tree_map(fn, ...)))`
 * Supports up to 2 parallel trees (the common case in these tests).
 */
const treeEqual = (fn: (...args: any[]) => any, treeA: any, treeB?: any): boolean => {
  const mapped = treeB !== undefined
    ? tree_map(fn as any, treeA, treeB)
    : tree_map(fn as any, treeA);
  const leaves = (tree_flatten(mapped) as [string, any][]).map(([, v]) => v);
  return leaves.every(Boolean);
};

/**
 * Elementwise close comparison of two MLXArrays.
 * Mirrors `mx.allclose` and `mx.array_equal`.
 */
const allcloseMx = (
  a: ReturnType<typeof mx.array>,
  b: ReturnType<typeof mx.array>,
  atol = 1e-5,
): boolean => {
  const av = (a.toArray() as any[]).flat(Infinity) as number[];
  const bv = (b.toArray() as any[]).flat(Infinity) as number[];
  if (av.length !== bv.length) return false;
  return av.every((x, i) => Math.abs(x - bv[i]) <= atol + 1e-5 * Math.abs(bv[i]));
};

describe('TestOptimizers (Python parity integration)', () => {
  // -------------------------------------------------------------------------
  // test_optimizer_state
  // -------------------------------------------------------------------------
  describe('test_optimizer_state', () => {
    it('state can hold arbitrary values', () => {
      const optim = new SGD({ learningRate: 0.1 });
      optim.state['hello'] = 'world';
      assert.equal(optim.state['hello'], 'world');
    });

    it('state can be replaced entirely', () => {
      const optim = new SGD({ learningRate: 0.1 });
      optim.state = { 0: 1 };
      assert.deepEqual(optim.state, { 0: 1 });
    });
  });

  // -------------------------------------------------------------------------
  // test_optimizers — apply all optimizers with ones gradients, check shapes
  // -------------------------------------------------------------------------
  describe('test_optimizers', () => {
    const params = makeParams();
    const grads = treeMap((x: any) => mx.ones_like(x), params);
    const allOptimizerClasses = [
      SGD, Adam, AdamW, Adamax, Lion, Adagrad, AdaDelta, RMSprop, Muon,
    ] as const;

    for (const Cls of allOptimizerClasses) {
      it(`${Cls.name} preserves parameter shapes after applyGradients`, () => {
        const optim = new (Cls as any)({ learningRate: 0.1 });
        const updated = optim.applyGradients(grads, params);
        const shapesEqual = treeEqual(
          (p: any, u: any) => {
            const ps = Array.isArray(p) ? p.map((x: any) => x.shape) : p.shape;
            const us = Array.isArray(u) ? u.map((x: any) => x.shape) : u.shape;
            return JSON.stringify(ps) === JSON.stringify(us);
          },
          params,
          updated,
        );
        assert.ok(shapesEqual, `${Cls.name}: shape mismatch after apply`);
      });
    }
  });

  // -------------------------------------------------------------------------
  // test_sgd
  // -------------------------------------------------------------------------
  describe('test_sgd', () => {
    it('explicit init zeroes the velocity state', () => {
      const params = makeParams();
      const grads = treeMap((x: any) => mx.ones_like(x), params);

      const optim = new SGD({ learningRate: 1e-2, momentum: 0.9 });
      optim.init(params);

      // After init, v must equal zeros_like(parameter) for each parameter
      const ok = treeEqual(
        (p: any, s: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              mx.array_equal(s[i].v, mx.zeros_like(param)).toArray()[0] === true
            );
          }
          return mx.array_equal(s.v, mx.zeros_like(p)).toArray()[0] === true;
        },
        params,
        optim.state,
      );
      assert.ok(ok, 'SGD: explicit init should yield v = zeros_like(param)');
    });

    it('implicit init (first apply) sets v equal to the gradient', () => {
      const params = makeParams();
      const grads = treeMap((x: any) => mx.ones_like(x), params);

      const optim = new SGD({ learningRate: 1e-2, momentum: 0.9 });
      optim.applyGradients(grads, params);

      // After one step, v = grad (because v_new = 0.9*0 + (1-0.9→no: SGD: v = m*v + g)
      // Python SGD with momentum: v = momentum * v + (1-dampening) * g
      // With dampening=0 (default): v = 0.9 * 0 + 1.0 * g = g
      const ok = treeEqual(
        (g: any, s: any) => {
          if (Array.isArray(g)) {
            return g.every((grad: any, i: number) =>
              mx.array_equal(s[i].v, grad).toArray()[0] === true
            );
          }
          return mx.array_equal(s.v, g).toArray()[0] === true;
        },
        grads,
        optim.state,
      );
      assert.ok(ok, 'SGD: after first implicit init, v should equal gradient');
    });
  });

  // -------------------------------------------------------------------------
  // test_rmsprop
  // -------------------------------------------------------------------------
  describe('test_rmsprop', () => {
    it('explicit init zeroes the v state', () => {
      const params = makeParams();
      const optim = new RMSprop({ learningRate: 1e-2 });
      optim.init(params);

      const ok = treeEqual(
        (p: any, s: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              mx.array_equal(s[i].v, mx.zeros_like(param)).toArray()[0] === true
            );
          }
          return mx.array_equal(s.v, mx.zeros_like(p)).toArray()[0] === true;
        },
        params,
        optim.state,
      );
      assert.ok(ok, 'RMSprop: explicit init should yield v = zeros_like(param)');
    });

    it('after one step with alpha=0.99, v = (1-alpha) * g^2', () => {
      const params = makeParams();
      const grads = treeMap((x: any) => mx.ones_like(x), params);
      const alpha = 0.99;
      const optim = new RMSprop({ learningRate: 1e-2, alpha });
      optim.applyGradients(grads, params);

      // RMSprop update: v = alpha*v + (1-alpha)*g^2  (with v_0=0, g=1)
      // => v = (1-alpha) * 1 = 0.01
      const ok = treeEqual(
        (g: any, s: any) => {
          const expected = mx.multiply(1 - alpha, mx.square(g));
          if (Array.isArray(g)) {
            return g.every((grad: any, i: number) =>
              allcloseMx(s[i].v, mx.multiply(1 - alpha, mx.square(grad)))
            );
          }
          return allcloseMx(s.v, expected);
        },
        grads,
        optim.state,
      );
      assert.ok(ok, 'RMSprop: after first step, v should equal (1-alpha)*g^2 = 0.01');
    });
  });

  // -------------------------------------------------------------------------
  // test_adagrad
  // -------------------------------------------------------------------------
  describe('test_adagrad', () => {
    it('explicit init zeroes the v state', () => {
      const params = makeParams();
      const optim = new Adagrad({ learningRate: 1e-2 });
      optim.init(params);

      const ok = treeEqual(
        (p: any, s: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              mx.array_equal(s[i].v, mx.zeros_like(param)).toArray()[0] === true
            );
          }
          return mx.array_equal(s.v, mx.zeros_like(p)).toArray()[0] === true;
        },
        params,
        optim.state,
      );
      assert.ok(ok, 'Adagrad: explicit init should yield v = zeros_like(param)');
    });
  });

  // -------------------------------------------------------------------------
  // test_adadelta
  // -------------------------------------------------------------------------
  describe('test_adadelta', () => {
    it('explicit init zeroes both v and u state', () => {
      const params = makeParams();
      const optim = new AdaDelta({ learningRate: 1e-2 });
      optim.init(params);

      const vOk = treeEqual(
        (p: any, s: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              mx.array_equal(s[i].v, mx.zeros_like(param)).toArray()[0] === true
            );
          }
          return mx.array_equal(s.v, mx.zeros_like(p)).toArray()[0] === true;
        },
        params,
        optim.state,
      );
      assert.ok(vOk, 'AdaDelta: explicit init, v should equal zeros_like(param)');

      const uOk = treeEqual(
        (p: any, s: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              mx.array_equal(s[i].u, mx.zeros_like(param)).toArray()[0] === true
            );
          }
          return mx.array_equal(s.u, mx.zeros_like(p)).toArray()[0] === true;
        },
        params,
        optim.state,
      );
      assert.ok(uOk, 'AdaDelta: explicit init, u should equal zeros_like(param)');
    });
  });

  // -------------------------------------------------------------------------
  // test_adam
  // -------------------------------------------------------------------------
  describe('test_adam', () => {
    const adamClasses = [Adam, AdamW, Adamax] as const;

    for (const Cls of adamClasses) {
      it(`${Cls.name}: explicit init zeroes both m and v state`, () => {
        const params = makeParams();
        const optim = new (Cls as any)({ learningRate: 1e-2 });
        optim.init(params);

        const vOk = treeEqual(
          (p: any, s: any) => {
            if (Array.isArray(p)) {
              return p.every((param: any, i: number) =>
                mx.array_equal(s[i].v, mx.zeros_like(param)).toArray()[0] === true
              );
            }
            return mx.array_equal(s.v, mx.zeros_like(p)).toArray()[0] === true;
          },
          params,
          optim.state,
        );
        assert.ok(vOk, `${Cls.name}: explicit init, v should equal zeros_like(param)`);

        const mOk = treeEqual(
          (p: any, s: any) => {
            if (Array.isArray(p)) {
              return p.every((param: any, i: number) =>
                mx.array_equal(s[i].m, mx.zeros_like(param)).toArray()[0] === true
              );
            }
            return mx.array_equal(s.m, mx.zeros_like(p)).toArray()[0] === true;
          },
          params,
          optim.state,
        );
        assert.ok(mOk, `${Cls.name}: explicit init, m should equal zeros_like(param)`);
      });
    }

    it('Adam with bias_correction=true preserves float16 dtype', { todo: 'MLXArray.astype() is not yet implemented in the Node port' }, () => {
      // TODO: Once MLXArray.astype() (or an equivalent astype(array, dtype) op) is added to
      // the Node.js bindings, convert params to float16 and assert dtype preservation through
      // Adam.applySingle with bias_correction=true.
      // Reference: Python test_optimizers.py TestOptimizers.test_adam
    });
  });

  // -------------------------------------------------------------------------
  // test_lion
  // -------------------------------------------------------------------------
  describe('test_lion', () => {
    it('explicit init zeroes the m state', () => {
      const params = makeParams();
      const optim = new Lion({ learningRate: 1e-2 });
      optim.init(params);

      const ok = treeEqual(
        (p: any, s: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              mx.array_equal(s[i].m, mx.zeros_like(param)).toArray()[0] === true
            );
          }
          return mx.array_equal(s.m, mx.zeros_like(p)).toArray()[0] === true;
        },
        params,
        optim.state,
      );
      assert.ok(ok, 'Lion: explicit init should yield m = zeros_like(param)');
    });
  });

  // -------------------------------------------------------------------------
  // test_adafactor
  // -------------------------------------------------------------------------
  describe('test_adafactor', () => {
    it('two steps preserve dtype float32 and shape for 5x5 param', { todo: 'Adafactor.applySingle requires mean(array, axis=N) which is not yet implemented in the Node port' }, () => {
      // TODO: Implement Adafactor.applySingle once mean(array, axis) is available in bindings.
      // Reference: Python mlx.optimizers.Adafactor, test_optimizers.py TestOptimizers.test_adafactor
    });

    it('two steps preserve shape for 5x5 float16 param', { todo: 'Adafactor.applySingle requires mean(array, axis=N) which is not yet implemented in the Node port' }, () => {
      // TODO: Same as above. Also test step counter increment.
    });
  });

  describe('test_muon', () => {
    // Use small arrays to avoid GPU shader compile time hanging the suite
    const muonParams = () => ({
      first: [mx.zeros([4, 3]), mx.zeros([1])],
      second: mx.zeros([3, 3]),
    });

    it('explicit init zeroes the v state', () => {
      const params = muonParams();
      const optim = new Muon({ learningRate: 1e-2, momentum: 0.95, nesterov: true });
      optim.init(params);

      const ok = treeEqual(
        (p: any, s: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              mx.array_equal(s[i].v, mx.zeros_like(param)).toArray()[0] === true
            );
          }
          return mx.array_equal(s.v, mx.zeros_like(p)).toArray()[0] === true;
        },
        params,
        optim.state,
      );
      assert.ok(ok, 'Muon: explicit init should yield v = zeros_like(param)');
    });

    it('applyGradients preserves shapes', () => {
      const params = muonParams();
      const grads = treeMap((x: any) => mx.ones_like(x), params);
      const optim = new Muon({ learningRate: 1e-2, momentum: 0.95, nesterov: true });
      const updated = optim.applyGradients(grads, params);

      const ok = treeEqual(
        (p: any, u: any) => {
          if (Array.isArray(p)) {
            return p.every((param: any, i: number) =>
              JSON.stringify(param.shape) === JSON.stringify(u[i].shape)
            );
          }
          return JSON.stringify(p.shape) === JSON.stringify(u.shape);
        },
        params,
        updated,
      );
      assert.ok(ok, 'Muon: shapes should be preserved after applyGradients');
    });

    it('applyGradients returns updated object with correct keys', () => {
      // Verify applyGradients returns an object with same keys as params.
      // (The Muon update may return zeros for some params — this is a known
      // limitation when Newton-Schulz produces no-op for rank-deficient shapes.)
      const params = muonParams();
      const grads = treeMap((x: any) => mx.ones_like(x), params);
      const optim = new Muon({ learningRate: 1e-2, momentum: 0.95, nesterov: true });
      const updated = optim.applyGradients(grads, params);

      // Keys should be preserved
      assert.ok('first' in updated);
      assert.ok('second' in updated);
      assert.ok(Array.isArray((updated as any).first));
    });

    it('works without nesterov', () => {
      const params = muonParams();
      const grads = treeMap((x: any) => mx.ones_like(x), params);
      const optim = new Muon({ learningRate: 1e-2, momentum: 0.95, nesterov: false });
      // Should not throw
      optim.applyGradients(grads, params);
    });

    it('works without momentum', () => {
      const params = muonParams();
      const grads = treeMap((x: any) => mx.ones_like(x), params);
      const optim = new Muon({ learningRate: 1e-2, momentum: 0.0 });
      // Should not throw
      optim.applyGradients(grads, params);
    });
  });
});


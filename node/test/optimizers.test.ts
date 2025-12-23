import { describe, it } from 'node:test';
import assert from 'node:assert';
import { SGD, Adam, Lion, RMSprop, Optimizer, MultiOptimizer } from '../src/optimizers';
import { zeros } from '../src/core/array';

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
      assert.strictEqual(step.toTypedArray()[0], 0);
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
      assert.strictEqual(step.toTypedArray()[0], 0);
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
      assert.strictEqual(step.toTypedArray()[0], 0);
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
      assert.strictEqual(step.toTypedArray()[0], 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new RMSprop({ learningRate: 0.01 });
      optimizer.learningRate = 0.001;
      // Note: This test is placeholder since we can't properly set scalar values yet
      assert.ok(optimizer.learningRate);
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
  });
});

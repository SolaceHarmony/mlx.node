import { describe, it } from 'node:test';
import assert from 'node:assert';
import { SGD, Adam, Adamax, Lion, Adagrad, RMSprop, Muon, Optimizer } from '../src/optimizers';
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
      assert.strictEqual(step.toTypedArray()[0], 0);
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
      assert.strictEqual(step.toTypedArray()[0], 0);
    });

    it('should allow setting learning rate', () => {
      const optimizer = new Adagrad({ learningRate: 0.01 });
      optimizer.learningRate = 0.001;
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
      assert.strictEqual(step.toTypedArray()[0], 0);
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

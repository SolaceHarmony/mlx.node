import { strict as assert } from 'node:assert';
import * as mx from '../src';

describe('nn.Module', () => {
  class MyModule extends mx.nn.Module {
    weight: mx.Array;
    bias?: mx.Array;

    constructor() {
      super();
      this.weight = mx.random.uniform(0, 1, [2, 2]);
      this.bias = mx.zeros([2]);
    }
  }

  it('should track parameters', () => {
    const model = new MyModule();
    const params = model.parameters();
    assert.ok(params.weight instanceof mx.Array);
    assert.ok(params.bias instanceof mx.Array);
  });

  it('should support freezing parameters', () => {
    const model = new MyModule();
    model.freeze(true, ['weight']);
    const trainable = model.trainableParameters();
    assert.ok(trainable.bias instanceof mx.Array);
    assert.ok(!trainable.weight);
  });

  it('should support nested modules', () => {
    class Parent extends mx.nn.Module {
      child: MyModule;
      constructor() {
        super();
        this.child = new MyModule();
      }
    }
    const p = new Parent();
    const params = p.parameters();
    assert.ok(params.child.weight instanceof mx.Array);
  });
});

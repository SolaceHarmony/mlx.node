import { strict as assert } from 'node:assert';
import * as mx from '../src';
import { nn, utils } from '../src';
import fs from 'fs';

describe('utils.Store', () => {
  it('should save and load a nested model state with JSON metadata', () => {
    const model = new nn.Sequential(
      new nn.Linear(2, 4),
      new nn.ReLU(),
      new nn.Linear(4, 1)
    );

    const metadata = {
      'version': '1.0.0',
      'description': 'A simple sequential model',
      'config': { 'layers': 3 }
    };

    const tempFile = 'test_model.safetensors';
    try {
      // 1. Save the state_dict and metadata
      utils.Store.save(tempFile, model.state_dict(), metadata);

      // 2. Load it back
      const loaded = utils.Store.load(tempFile);

      // 3. Verify metadata
      assert.deepEqual(loaded.metadata, metadata);

      // 4. Verify state_dict keys (nested)
      const originalKeys = Object.keys(mx.tree_flatten(model.state_dict()));
      const loadedKeys = Object.keys(mx.tree_flatten(loaded.state));
      assert.deepEqual(loadedKeys.sort(), originalKeys.sort());

      // 5. Verify values
      const originalFlat = mx.tree_flatten(model.state_dict()) as [string, any][];
      const loadedFlat = loaded.state; // unflattened
      
      for (const [key, value] of originalFlat) {
          const parts = key.split('.');
          let loadedValue = loadedFlat;
          for (const part of parts) {
              loadedValue = loadedValue[part];
          }
          assert.ok(mx.array_equal(value, loadedValue));
      }

      // 6. Test updating a new model from loaded state
      const newModel = new nn.Sequential(
        new nn.Linear(2, 4),
        new nn.ReLU(),
        new nn.Linear(4, 1)
      );
      newModel.update(loaded.state);
      
      const x = mx.ones([1, 2]);
      const y1 = model.forward(x);
      const y2 = newModel.forward(x);
      assert.ok(mx.allclose(y1, y2));

    } finally {
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
    }
  });
});

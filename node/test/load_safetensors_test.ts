import * as mx from '../src';

const modelPath = '/Volumes/games/models/GLM-4-9B-0414/model-00001-of-00004.safetensors';

async function testLoad() {
    console.log(`Loading weights from: ${modelPath}`);
    try {
        const result = mx.load(modelPath);
        
        let arrays: Record<string, any>;
        // Use duck-typing to avoid circular dependency issues with instanceof
        if (result && typeof result === 'object' && 'toNative' in (result as any)) {
            console.log('Loaded a single array.');
            arrays = { 'default': result };
        } else {
            arrays = result as Record<string, any>;
            console.log(`Successfully loaded ${Object.keys(arrays).length} weights.`);
        }
        
        const keys = Object.keys(arrays);
        if (keys.length > 0) {
            const firstKey = keys[0];
            const firstWeight = arrays[firstKey];
            console.log(`- ${firstKey}: shape=[${firstWeight.shape}], dtype=${firstWeight.dtype}`);
            
            console.log(`\nVerifying ${firstKey}...`);
            const meanVal = mx.mean(firstWeight).toArray()[0];
            console.log(`Verification: Mean of ${firstKey} is ${meanVal}`);
        }

    } catch (err) {
        console.error('Failed to load weights:', err);
        process.exit(1);
    }
}

testLoad();

// Comprehensive naming convention verification
const mlx = require('./dist/src/index.js');

console.log('=== Verifying TypeScript Naming Conventions ===\n');

const snakeCasePattern = /^[a-z]+(_[a-z]+)+$/;
const issues = [];

function checkForSnakeCase(obj, path = '') {
  const props = Object.getOwnPropertyNames(obj);

  for (const prop of props) {
    // Skip private properties (start with _) and constructors
    if (prop.startsWith('_') || prop === 'constructor' || prop === 'prototype') {
      continue;
    }

    if (snakeCasePattern.test(prop)) {
      issues.push(`${path}.${prop}`);
    }
  }
}

// Check main module exports
console.log('Checking main module exports...');
checkForSnakeCase(mlx, 'mlx');

// Check MLXArray class methods
console.log('Checking MLXArray class methods...');
if (mlx.Array) {
  checkForSnakeCase(mlx.Array.prototype, 'MLXArray');
  checkForSnakeCase(mlx.Array, 'MLXArray (static)');
}

// Check core module
console.log('Checking core module...');
if (mlx.core) {
  checkForSnakeCase(mlx.core, 'mlx.core');
  if (mlx.core.Array) {
    checkForSnakeCase(mlx.core.Array.prototype, 'mlx.core.Array');
    checkForSnakeCase(mlx.core.Array, 'mlx.core.Array (static)');
  }
  if (mlx.core.Stream) {
    checkForSnakeCase(mlx.core.Stream.prototype, 'mlx.core.Stream');
    checkForSnakeCase(mlx.core.Stream, 'mlx.core.Stream (static)');
  }
}

// Check optimizers
console.log('Checking optimizers...');
if (mlx.optimizers) {
  checkForSnakeCase(mlx.optimizers, 'mlx.optimizers');
  if (mlx.optimizers.Optimizer) {
    checkForSnakeCase(mlx.optimizers.Optimizer.prototype, 'mlx.optimizers.Optimizer');
  }
  if (mlx.optimizers.SGD) {
    checkForSnakeCase(mlx.optimizers.SGD.prototype, 'mlx.optimizers.SGD');
  }
  if (mlx.optimizers.Adam) {
    checkForSnakeCase(mlx.optimizers.Adam.prototype, 'mlx.optimizers.Adam');
  }
  if (mlx.optimizers.Lion) {
    checkForSnakeCase(mlx.optimizers.Lion.prototype, 'mlx.optimizers.Lion');
  }
}

// Check utils
console.log('Checking utils...');
if (mlx.utils) {
  checkForSnakeCase(mlx.utils, 'mlx.utils');
}

console.log('\n=== Results ===');
if (issues.length === 0) {
  console.log('✅ All naming conventions are correct! No snake_case found in public API.');
  console.log('\nAll classes, methods, and exported functions use camelCase as expected.');
  process.exit(0);
} else {
  console.log('❌ Found snake_case issues:');
  issues.forEach(issue => console.log(`  - ${issue}`));
  process.exit(1);
}

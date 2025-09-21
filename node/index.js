'use strict';

const path = require('path');

function loadNative() {
  const candidates = [
    path.join(__dirname, 'build/Release/mlx.node'),
    path.join(__dirname, '../build/Release/mlx.node')
  ];
  for (const p of candidates) {
    try { return require(p); } catch (_) {}
  }
  const err = new Error(
    'Cannot find mlx.node. Build the native addon (e.g., `cd node && npx node-gyp build`).'
  );
  err.code = 'MODULE_NOT_FOUND';
  throw err;
}

const mlx = loadNative();

// Light JS surface: nn namespace scaffolding (expand incrementally)
const nn = {
  Module: class Module {
    parameters() { return []; }
    // Subclasses override forward(x)
  }
};

mlx.nn = nn;
module.exports = mlx;


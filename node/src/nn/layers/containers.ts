/**
 * @fileoverview Container layers.
 *
 * Mirrors mlx.nn.layers.containers from the Python MLX API.
 */

import { Module } from './base';

/**
 * A sequential container.
 * Modules will be added to it in the order they are passed in the constructor.
 * The forward pass will apply each module in order.
 *
 * @example
 * ```typescript
 * const model = new nn.Sequential(
 *   new nn.Linear(10, 20),
 *   new nn.ReLU(),
 *   new nn.Linear(20, 5)
 * );
 * ```
 */
export class Sequential extends Module {
  /** The list of modules in the sequential container. */
  layers: Module[];

  constructor(...layers: (Module | ((x: any) => any))[]) {
    super();
    // Wrap functions in a simple Module if needed, or just store them.
    // MLX Python Sequential supports any callable.
    this.layers = layers.map((layer, i) => {
      if (layer instanceof Module) {
        return layer;
      }
      // Simple wrapper for plain functions to ensure they are treated as modules if needed,
      // though Module.parameters() recursion handles arrays of modules.
      return new FunctionModule(layer, `layer_${i}`);
    });
  }

  /**
   * Forward pass through all layers in order.
   */
  forward(x: any): any {
    for (const layer of this.layers) {
      if (typeof (layer as any).__call__ === 'function') {
        x = (layer as any).__call__(x);
      } else if (typeof (layer as any).forward === 'function') {
        x = (layer as any).forward(x);
      } else if (typeof layer === 'function') {
        x = (layer as any)(x);
      } else {
        throw new Error('Sequential: layer is not callable (missing __call__ or forward)');
      }
    }
    return x;
  }

  __call__(x: any): any {
    return this.forward(x);
  }
}

/**
 * Internal helper to wrap plain functions in a Module for Sequential.
 */
class FunctionModule extends Module {
  constructor(private func: (x: any) => any, private name: string) {
    super();
  }
  forward(x: any): any {
    return this.func(x);
  }
  __call__(x: any): any {
    return this.forward(x);
  }
}

/**
 * A list-like container for modules.
 */
export class ModuleList extends Module {
  constructor(public modules: Module[]) {
    super();
  }

  get length(): number {
    return this.modules.length;
  }

  [Symbol.iterator]() {
    return this.modules[Symbol.iterator]();
  }

  push(module: Module): void {
    this.modules.push(module);
  }
}

/**
 * A dictionary-like container for modules.
 */
export class ModuleDict extends Module {
  constructor(public modules: { [key: string]: Module }) {
    super();
  }

  get(key: string): Module | undefined {
    return this.modules[key];
  }

  set(key: string, module: Module): void {
    this.modules[key] = module;
  }
}

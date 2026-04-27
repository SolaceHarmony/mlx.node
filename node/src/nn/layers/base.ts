/**
 * @fileoverview Base class for neural network layers.
 *
 * Mirrors mlx.nn.layers.base from the Python MLX API.
 */

import MLXArray from '../../core/array';
import { tree_flatten, tree_unflatten } from '../../utils';

/**
 * Base class for all neural network layers in MLX.
 *
 * Provides parameter tracking, training mode management, and weight loading.
 */
export abstract class Module {
  /** Map of parameter names to their values (arrays, lists, or dicts). */
  [key: string]: any;

  /** Set of keys whose parameters should not have gradients computed. */
  _no_grad: Set<string> = new Set();

  /** Whether the module is in training mode. */
  _training: boolean = true;

  constructor() {
    // We use a Proxy to intercept property assignments and store them
    // in the instance while also allowing dictionary-like behavior.
    return new Proxy(this, {
      set: (target: any, key: string, value: any) => {
        if (
          value instanceof MLXArray ||
          value instanceof Module ||
          Array.isArray(value) ||
          (value !== null && typeof value === 'object' && value.constructor === Object)
        ) {
          // If the attribute was previously set on the object but not in the
          // dict-like store, we might need to handle it. In TS we just set it.
          target[key] = value;
        } else {
          target[key] = value;
        }
        return true;
      },
    });
  }

  /** Boolean indicating if the model is in training mode. */
  get training(): boolean {
    return this._training;
  }

  /** The module's state dictionary. */
  get state(): this {
    return this;
  }

  /** Optional string representation for extra parameters. */
  _extra_repr(): string {
    return '';
  }

  /** Static check if a value is a Module. */
  static isModule(value: any): value is Module {
    return value instanceof Module;
  }

  /** Decide whether to include a value in parameters(). */
  static validParameterFilter(module: Module, key: string, value: any): boolean {
    return (
      (value instanceof MLXArray ||
        Module.isModule(value) ||
        Array.isArray(value) ||
        (value !== null && typeof value === 'object' && value.constructor === Object)) &&
      !key.startsWith('_')
    );
  }

  /** Decide whether to include a value in trainable_parameters(). */
  static trainableParameterFilter(module: Module, key: string, value: any): boolean {
    return Module.validParameterFilter(module, key, value) && !module._no_grad.has(key);
  }

  /** Decide whether to include a value in children(). */
  static validChildFilter(module: Module, key: string, value: any): boolean {
    return Array.isArray(value) || (value !== null && typeof value === 'object' && value.constructor === Object);
  }

  /** Recursively filter the contents of the module. */
  filterAndMap(
    filterFn: (m: Module, k: string, v: any) => boolean,
    mapFn: (v: any) => any = (x) => x,
    isLeafFn: (m: Module, k: string, v: any) => boolean = (m, k, v) =>
      !(v instanceof Module || Array.isArray(v) || (v !== null && typeof v === 'object' && v.constructor === Object)),
  ): any {
    const result: any = {};
    for (const [key, value] of Object.entries(this)) {
      if (filterFn(this, key, value)) {
        result[key] = this._unwrap(key, value, filterFn, mapFn, isLeafFn);
      }
    }
    return result;
  }

  private _unwrap(
    key: string,
    value: any,
    filterFn: (m: Module, k: string, v: any) => boolean,
    mapFn: (v: any) => any,
    isLeafFn: (m: Module, k: string, v: any) => boolean,
  ): any {
    if (isLeafFn(this, key, value)) {
      return mapFn(value);
    }

    if (value instanceof Module) {
      return value.filterAndMap(filterFn, mapFn, isLeafFn);
    }

    if (Array.isArray(value)) {
      return value
        .filter((v, i) => filterFn(this, `${key}.${i}`, v))
        .map((v, i) => this._unwrap(`${key}.${i}`, v, filterFn, mapFn, isLeafFn));
    }

    if (value !== null && typeof value === 'object' && value.constructor === Object) {
      const res: any = {};
      for (const [k, v] of Object.entries(value)) {
        if (filterFn(this, `${key}.${k}`, v)) {
          res[k] = this._unwrap(`${key}.${k}`, v, filterFn, mapFn, isLeafFn);
        }
      }
      return res;
    }

    return mapFn(value);
  }

  /** Recursively return all parameters. */
  parameters(): any {
    return this.filterAndMap(Module.validParameterFilter);
  }

  /** Recursively return all trainable parameters. */
  trainableParameters(): any {
    return this.filterAndMap(Module.trainableParameterFilter);
  }

  /** Return the direct descendants of this Module. */
  children(): any {
    return this.filterAndMap(Module.validChildFilter, (x) => x, (m, k, v) => v instanceof Module);
  }

  /** Replace the parameters of this Module. */
  update(parameters: any, strict: boolean = true): this {
    const apply = (dst: any, params: any) => {
      if (params !== null && typeof params === 'object' && params.constructor === Object) {
        for (const k in params) {
          if (k in dst) {
            const currentValue = dst[k];
            const newValue = params[k];
            if (currentValue instanceof MLXArray) {
              if (strict && !(newValue instanceof MLXArray)) {
                throw new Error(`Received invalid type for parameter ${k}.`);
              }
              dst[k] = newValue;
            } else {
              apply(currentValue, newValue);
            }
          } else if (strict) {
            throw new Error(`Module does not have parameter named "${k}".`);
          }
        }
      } else if (Array.isArray(params)) {
        for (let i = 0; i < params.length; i++) {
          const currentValue = dst[i];
          const newValue = params[i];
          if (currentValue instanceof MLXArray) {
            if (strict && !(newValue instanceof MLXArray)) {
              throw new Error(`Received invalid type for array index ${i}.`);
            }
            dst[i] = newValue;
          } else {
            apply(currentValue, newValue);
          }
        }
      } else if (strict) {
        throw new Error('Received invalid parameters type.');
      }
    };

    apply(this, parameters);
    return this;
  }

  /**
   * Update the parameters of the module with the given parameters.
   */
  update(parameters: any): this {
    const flattened = tree_flatten(parameters, { isLeaf: (v) => v instanceof MLXArray }) as [string, any][];
    const currentParams = this.parameters();
    for (const [key, value] of flattened) {
      const parts = key.split('.');
      let obj = currentParams as any;
      for (let i = 0; i < parts.length - 1; i++) {
        obj = obj[parts[i]];
      }
      obj[parts[parts.length - 1]] = value;
    }
    return this;
  }

  /**
   * Return the parameters of the module as a dictionary.
   */
  state_dict(): any {
    return this.parameters();
  }

  /** Set the training mode. */
  train(mode: boolean = true): this {
    this._training = mode;
    const children = tree_flatten(this.children(), { isLeaf: (v) => v instanceof Module }) as [string, any][];
    for (const [, child] of children) {
      if (child instanceof Module) {
        child.train(mode);
      }
    }
    return this;
  }

  /** Set the eval mode (training=false). */
  eval(): this {
    return this.train(false);
  }

  /** Freeze the module or specific parameters. */
  freeze(recursive: boolean = true, keys?: string[]): this {
    if (keys) {
      for (const k of keys) this._no_grad.add(k);
    } else {
      // Freeze all local array parameters
      for (const [k, v] of Object.entries(this)) {
        if (v instanceof MLXArray && !k.startsWith('_')) {
          this._no_grad.add(k);
        }
      }
    }

    if (recursive) {
      const children = tree_flatten(this.children(), { isLeaf: (v) => v instanceof Module }) as [string, any][];
      for (const [, child] of children) {
        if (child instanceof Module) {
          child.freeze(recursive);
        }
      }
    }
    return this;
  }

  /** Unfreeze the module or specific parameters. */
  unfreeze(recursive: boolean = true, keys?: string[]): this {
    if (keys) {
      for (const k of keys) this._no_grad.delete(k);
    } else {
      this._no_grad.clear();
    }

    if (recursive) {
      const children = tree_flatten(this.children(), { isLeaf: (v) => v instanceof Module }) as [string, any][];
      for (const [, child] of children) {
        if (child instanceof Module) {
          child.unfreeze(recursive);
        }
      }
    }
    return this;
  }
}

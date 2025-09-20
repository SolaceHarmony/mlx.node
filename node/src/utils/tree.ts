/* eslint-disable @typescript-eslint/no-explicit-any */
// Utility functions for working with Python-style trees.

export interface TreeVisitOptions {
  isLeaf?: (value: any) => boolean;
}

export interface TreeFlattenOptions extends TreeVisitOptions {
  prefix?: string;
  destination?: Array<[string, unknown]> | Record<string, unknown>;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || value === undefined) {
    return false;
  }
  if (Array.isArray(value)) {
    return false;
  }
  return typeof value === 'object';
}

function isVisitOptions(value: unknown): value is TreeVisitOptions {
  return (
    isPlainObject(value) &&
    Object.prototype.hasOwnProperty.call(value, 'isLeaf') &&
    (typeof (value as TreeVisitOptions).isLeaf === 'function' ||
      (value as TreeVisitOptions).isLeaf === undefined)
  );
}

function extractOptions(args: unknown[]): { restTrees: unknown[]; options: TreeVisitOptions } {
  if (args.length === 0) {
    return { restTrees: [], options: {} };
  }

  const maybeOptions = args[args.length - 1];
  if (isVisitOptions(maybeOptions)) {
    return {
      restTrees: args.slice(0, -1),
      options: maybeOptions as TreeVisitOptions,
    };
  }

  return { restTrees: args, options: {} };
}

function validateArraySubtrees(trees: unknown[], length: number): void {
  for (let i = 1; i < trees.length; i += 1) {
    const subtree = trees[i];
    if (Array.isArray(subtree)) {
      if (subtree.length !== length) {
        throw new Error('[tree_map] Additional input tree is not a valid prefix of the first tree.');
      }
    } else if (isPlainObject(subtree)) {
      throw new Error('[tree_map] Additional input tree is not a valid prefix of the first tree.');
    }
  }
}

function validateDictSubtrees(trees: unknown[]): void {
  for (let i = 1; i < trees.length; i += 1) {
    const subtree = trees[i];
    if (isPlainObject(subtree)) {
      continue;
    }
    if (Array.isArray(subtree)) {
      throw new Error('[tree_map] Additional input tree is not a valid prefix of the first tree.');
    }
  }
}

function isLeafNode(node: unknown, isLeaf?: (value: any) => boolean): boolean {
  if (isLeaf) {
    return isLeaf(node);
  }
  return !Array.isArray(node) && !isPlainObject(node);
}

type TreeMapTransform = (...args: any[]) => unknown;

function treeMapImpl(
  transform: TreeMapTransform,
  trees: unknown[],
  options: TreeVisitOptions,
  path?: string,
  includePath = false,
): unknown {
  const current = trees[0];

  if (isLeafNode(current, options.isLeaf)) {
    const rest = trees.slice(1);
    if (includePath) {
      return transform(path ?? '', current, ...rest);
    }
    return transform(current, ...rest);
  }

  if (Array.isArray(current)) {
    const length = current.length;
    validateArraySubtrees(trees, length);
    return current.map((item, index) => {
      const nextTrees = trees.map((tree) => {
        if (Array.isArray(tree)) {
          return tree[index];
        }
        return tree;
      });
      const nextPath = path !== undefined ? `${path}.${index}` : `${index}`;
      return treeMapImpl(transform, nextTrees, options, nextPath, includePath);
    });
  }

  if (isPlainObject(current)) {
    validateDictSubtrees(trees);
    const entries = Object.entries(current);
    const output: Record<string, unknown> = {};
    for (const [key, value] of entries) {
      const nextTrees = trees.map((tree) => {
        if (isPlainObject(tree)) {
          if (!Object.prototype.hasOwnProperty.call(tree, key)) {
            throw new Error('[tree_map] Tree is not a valid prefix tree of the first tree.');
          }
          return (tree as Record<string, unknown>)[key];
        }
        return tree;
      });
      const nextPath = path !== undefined ? `${path}.${key}` : key;
      output[key] = treeMapImpl(transform, nextTrees, options, nextPath, includePath);
    }
    return output;
  }

  const rest = trees.slice(1);
  if (includePath) {
    return transform(path ?? '', current, ...rest);
  }
  return transform(current, ...rest);
}

export function treeMap(fn: TreeMapTransform, tree: unknown, ...rest: unknown[]): unknown {
  const { restTrees, options } = extractOptions(rest);
  return treeMapImpl(fn, [tree, ...restTrees], options, undefined, false);
}

export function treeMapWithPath(
  fn: TreeMapTransform,
  tree: unknown,
  ...rest: unknown[]
): unknown {
  const { restTrees, options } = extractOptions(rest);
  return treeMapImpl(fn, [tree, ...restTrees], options, undefined, true);
}

export function treeFlatten(
  tree: unknown,
  options: TreeFlattenOptions = {},
): Array<[string, unknown]> | Record<string, unknown> {
  const { prefix = '', isLeaf, destination } = options;
  const dest = destination ?? [];

  let add: (key: string, value: unknown) => void;
  if (Array.isArray(dest)) {
    add = (key, value) => dest.push([key, value]);
  } else if (isPlainObject(dest)) {
    add = (key, value) => {
      (dest as Record<string, unknown>)[key] = value;
    };
  } else {
    throw new Error('Destination should be either a list or a dictionary or undefined');
  }

  function recurse(node: unknown, currentPrefix: string): void {
    if (isLeafNode(node, isLeaf)) {
      add(currentPrefix.slice(1), node);
      return;
    }

    if (Array.isArray(node)) {
      node.forEach((item, idx) => {
        recurse(item, `${currentPrefix}.${idx}`);
      });
      return;
    }

    if (isPlainObject(node)) {
      Object.entries(node).forEach(([key, value]) => {
        recurse(value, `${currentPrefix}.${key}`);
      });
      return;
    }

    add(currentPrefix.slice(1), node);
  }

  recurse(tree, prefix);
  return dest;
}

export function treeUnflatten(
  tree: Array<[string, unknown]> | Record<string, unknown>,
): unknown {
  const items = Array.isArray(tree) ? tree : Object.entries(tree);

  if (items.length === 1) {
    const [key, value] = items[0];
    if (key === '') {
      return value;
    }
  }

  const children = new Map<string, Array<[string, unknown]>>();
  for (const [pathKey, value] of items) {
    const dotIndex = pathKey.indexOf('.');
    const currentIdx = dotIndex === -1 ? pathKey : pathKey.slice(0, dotIndex);
    const nextIdx = dotIndex === -1 ? '' : pathKey.slice(dotIndex + 1);
    const bucket = children.get(currentIdx) ?? [];
    bucket.push([nextIdx, value]);
    children.set(currentIdx, bucket);
  }

  let allNumeric = true;
  const numericKeys: Array<[number, string]> = [];
  for (const key of children.keys()) {
    if (key === '') {
      allNumeric = false;
      break;
    }
    const parsed = Number.parseInt(key, 10);
    if (Number.isNaN(parsed) || `${parsed}` !== key) {
      allNumeric = false;
      break;
    }
    numericKeys.push([parsed, key]);
  }

  if (allNumeric && numericKeys.length > 0) {
    numericKeys.sort((a, b) => a[0] - b[0]);
    const result: unknown[] = [];
    for (const [index, originalKey] of numericKeys) {
      while (result.length < index) {
        result.push({});
      }
      const subtree = treeUnflatten(children.get(originalKey) ?? []);
      if (result.length === index) {
        result.push(subtree);
      } else {
        result[index] = subtree;
      }
    }
    return result;
  }

  const result: Record<string, unknown> = {};
  for (const [key, bucket] of children.entries()) {
    result[key] = treeUnflatten(bucket);
  }
  return result;
}

export function treeReduce(
  fn: (accumulator: any, value: any) => any,
  tree: unknown,
  initializer?: unknown,
  options: TreeVisitOptions = {},
): unknown {
  let actualInitializer = initializer;
  let actualOptions = options;

  if (arguments.length === 3 && isVisitOptions(initializer)) {
    actualOptions = initializer;
    actualInitializer = undefined;
  } else if (options === undefined && isVisitOptions(initializer)) {
    actualOptions = initializer;
    actualInitializer = undefined;
  }

  const { isLeaf } = actualOptions;
  let hasAccumulator = actualInitializer !== undefined;
  let accumulator = actualInitializer as unknown;

  function handleLeaf(value: unknown): void {
    if (!hasAccumulator) {
      accumulator = value;
      hasAccumulator = true;
    } else {
      accumulator = fn(accumulator, value);
    }
  }

  function recurse(node: unknown): void {
    if (isLeafNode(node, isLeaf)) {
      handleLeaf(node);
      return;
    }
    if (Array.isArray(node)) {
      node.forEach((item) => recurse(item));
      return;
    }
    if (isPlainObject(node)) {
      Object.values(node).forEach((value) => recurse(value));
      return;
    }
    handleLeaf(node);
  }

  recurse(tree);
  return accumulator;
}

export function treeMerge(
  treeA: unknown,
  treeB: unknown,
  mergeFn?: (a: unknown, b: unknown) => unknown,
): unknown {
  const normalize = (tree: unknown): unknown => {
    if (tree === undefined) {
      return null;
    }
    if (Array.isArray(tree)) {
      return tree.length === 0 ? null : tree;
    }
    if (isPlainObject(tree)) {
      return Object.keys(tree).length === 0 ? null : tree;
    }
    return tree;
  };

  const a = normalize(treeA);
  const b = normalize(treeB);

  if (a === null && b !== null) {
    return b;
  }
  if (a !== null && b === null) {
    return a;
  }
  if (a === null && b === null) {
    if (mergeFn) {
      return mergeFn(null, null);
    }
    throw new Error(
      'Trees contain elements at the same locations but no merge function was provided',
    );
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    const maxLength = Math.max(a.length, b.length);
    const result: unknown[] = [];
    for (let i = 0; i < maxLength; i += 1) {
      const left = i < a.length ? a[i] : null;
      const right = i < b.length ? b[i] : null;
      result[i] = treeMerge(left, right, mergeFn);
    }
    return result;
  }

  if (isPlainObject(a) && isPlainObject(b)) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    const result: Record<string, unknown> = {};
    keys.forEach((key) => {
      result[key] = treeMerge((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key], mergeFn);
    });
    return result;
  }

  if (mergeFn === undefined) {
    throw new Error(
      'Trees contain elements at the same locations but no merge function was provided',
    );
  }
  return mergeFn(a, b);
}

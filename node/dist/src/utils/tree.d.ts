export interface TreeVisitOptions {
    isLeaf?: (value: any) => boolean;
}
export interface TreeFlattenOptions extends TreeVisitOptions {
    prefix?: string;
    destination?: Array<[string, unknown]> | Record<string, unknown>;
}
type TreeMapTransform = (...args: any[]) => unknown;
export declare function treeMap(fn: TreeMapTransform, tree: unknown, ...rest: unknown[]): unknown;
export declare function treeMapWithPath(fn: TreeMapTransform, tree: unknown, ...rest: unknown[]): unknown;
export declare function treeFlatten(tree: unknown, options?: TreeFlattenOptions): Array<[string, unknown]> | Record<string, unknown>;
export declare function treeUnflatten(tree: Array<[string, unknown]> | Record<string, unknown>): unknown;
export declare function treeReduce(fn: (accumulator: any, value: any) => any, tree: unknown, initializer?: unknown, options?: TreeVisitOptions): unknown;
export declare function treeMerge(treeA: unknown, treeB: unknown, mergeFn?: (a: unknown, b: unknown) => unknown): unknown;
export {};

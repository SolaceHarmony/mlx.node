import * as mx from '../index';
import { tree_flatten, tree_unflatten } from './tree';

/**
 * A store for MLX models and weights, simulating the Python object store.
 * Supports saving/loading to safetensors with JSON metadata.
 */
export class Store {
  /**
   * Save a model's state and optional metadata to a safetensors file.
   */
  static save(file: string, state: any, metadata: Record<string, any> = {}): void {
    const jsonMetadata = JSON.stringify(metadata);
    mx.save_safetensors(file, state, {
      metadata: { 'mlx_node_json_metadata': jsonMetadata }
    } as any);
  }

  /**
   * Load a model's state and metadata from a safetensors file.
   */
  static load(file: string): { state: any, metadata: any } {
    const loaded = mx.load_safetensors(file);
    const state = tree_unflatten(loaded.arrays);
    
    let metadata: any = {};
    if (loaded.metadata && loaded.metadata['mlx_node_json_metadata']) {
      try {
        metadata = JSON.parse(loaded.metadata['mlx_node_json_metadata']);
      } catch (e) {
        console.warn('Failed to parse JSON metadata from safetensors.');
      }
    }
    
    return { state, metadata };
  }
}

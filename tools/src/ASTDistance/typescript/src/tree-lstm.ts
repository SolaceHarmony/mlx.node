// port-lint: source include/tree_lstm.hpp
import { Tree } from "./tree.js";
import { Tensor } from "./tensor.js";

interface NodeState {
  c: Tensor; // Cell state
  h: Tensor; // Hidden state
}

/**
 * Binary Tree-LSTM implementation.
 * Ported from Stanford TreeLSTM (Lua/Torch) to C++.
 *
 * Based on: "Improved Semantic Representations From Tree-Structured
 *           Long Short-Term Memory Networks" (Tai et al., 2015)
 */
export class BinaryTreeLSTM {
  in_dim: number; // Input embedding dimension
  mem_dim: number; // Memory/hidden state dimension
  gate_output: boolean = true;

  // Weights
  W_leaf_c!: Tensor;
  W_leaf_o!: Tensor;
  U_i_l!: Tensor;
  U_i_r!: Tensor;
  U_fl_l!: Tensor;
  U_fl_r!: Tensor;
  U_fr_l!: Tensor;
  U_fr_r!: Tensor;
  U_u_l!: Tensor;
  U_u_r!: Tensor;
  U_o_l!: Tensor;
  U_o_r!: Tensor;

  private states: Map<Tree, NodeState> = new Map();

  constructor(input_dim: number, memory_dim: number, use_output_gate: boolean = true) {
    this.in_dim = input_dim;
    this.mem_dim = memory_dim;
    this.gate_output = use_output_gate;
    this.initializeWeights();
  }

  initializeWeights(): void {
    const scale = Math.sqrt(2.0 / this.mem_dim);

    // Leaf module
    this.W_leaf_c = Tensor.randn(this.in_dim, this.mem_dim, scale);
    if (this.gate_output) {
      this.W_leaf_o = Tensor.randn(this.in_dim, this.mem_dim, scale);
    }

    // Composer weights
    this.U_i_l = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    this.U_i_r = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    this.U_fl_l = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    this.U_fl_r = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    this.U_fr_l = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    this.U_fr_r = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    this.U_u_l = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    this.U_u_r = Tensor.randn(this.mem_dim, this.mem_dim, scale);

    if (this.gate_output) {
      this.U_o_l = Tensor.randn(this.mem_dim, this.mem_dim, scale);
      this.U_o_r = Tensor.randn(this.mem_dim, this.mem_dim, scale);
    }
  }

  forward(tree: Tree, inputs: Tensor[]): Tensor {
    this.states.clear();
    this.forwardRecursive(tree, inputs);
    return this.states.get(tree)!.h;
  }

  private forwardRecursive(node: Tree, inputs: Tensor[]): void {
    if (node.isLeaf()) {
      if (node.leafIdx < 0 || node.leafIdx >= inputs.length) {
        this.states.set(node, {
          c: Tensor.zeros(this.mem_dim),
          h: Tensor.zeros(this.mem_dim),
        });
        return;
      }

      const x = inputs[node.leafIdx]!;
      const c = this.W_leaf_c.matmul(x);

      let h: Tensor;
      if (this.gate_output) {
        const o = this.W_leaf_o.matmul(x).sigmoid();
        h = o.hadamard(c.tanh());
      } else {
        h = c.tanh();
      }

      this.states.set(node, { c, h });
    } else {
      for (const child of node.children) {
        if (child) this.forwardRecursive(child, inputs);
      }

      let lc = Tensor.zeros(this.mem_dim);
      let lh = Tensor.zeros(this.mem_dim);
      let rc = Tensor.zeros(this.mem_dim);
      let rh = Tensor.zeros(this.mem_dim);

      if (node.children.length >= 1 && node.children[0]) {
        const state = this.states.get(node.children[0]);
        if (state) {
          lc = state.c;
          lh = state.h;
        }
      }
      if (node.children.length >= 2 && node.children[1]) {
        const state = this.states.get(node.children[1]);
        if (state) {
          rc = state.c;
          rh = state.h;
        }
      }

      const i = this.U_i_l.matmul(lh).add(this.U_i_r.matmul(rh)).sigmoid();
      const fl = this.U_fl_l.matmul(lh).add(this.U_fl_r.matmul(rh)).sigmoid();
      const fr = this.U_fr_l.matmul(lh).add(this.U_fr_r.matmul(rh)).sigmoid();
      const u = this.U_u_l.matmul(lh).add(this.U_u_r.matmul(rh)).tanh();

      const c = i.hadamard(u).add(fl.hadamard(lc)).add(fr.hadamard(rc));

      let h: Tensor;
      if (this.gate_output) {
        const o = this.U_o_l.matmul(lh).add(this.U_o_r.matmul(rh)).sigmoid();
        h = o.hadamard(c.tanh());
      } else {
        h = c.tanh();
      }

      this.states.set(node, { c, h });
    }
  }
}

/**
 * Siamese Tree-LSTM for computing similarity between two trees.
 */
export class TreeLSTMSimilarity {
  encoder: BinaryTreeLSTM;
  W_sim!: Tensor;
  W_out!: Tensor;
  sim_hidden_dim: number;

  constructor(input_dim: number, memory_dim: number, hidden_dim: number = 50) {
    this.encoder = new BinaryTreeLSTM(input_dim, memory_dim, false);
    this.sim_hidden_dim = hidden_dim;
    this.initializeSimWeights();
  }

  initializeSimWeights(): void {
    const scale = Math.sqrt(2.0 / this.encoder.mem_dim);
    this.W_sim = Tensor.randn(2 * this.encoder.mem_dim, this.sim_hidden_dim, scale);
    this.W_out = Tensor.randn(this.sim_hidden_dim, 2, scale);
  }

  similarity(tree1: Tree, inputs1: Tensor[], tree2: Tree, inputs2: Tensor[]): number {
    const h1 = this.encoder.forward(tree1, inputs1);
    const h2 = this.encoder.forward(tree2, inputs2);

    const diff = h1.sub(h2).abs();
    const prod = h1.hadamard(h2);
    const features = diff.concat(prod);

    const hidden = this.W_sim.matmul(features).sigmoid();
    const output = this.W_out.matmul(hidden).softmax();

    return output.at(1);
  }

  cosineSimilarity(tree1: Tree, inputs1: Tensor[], tree2: Tree, inputs2: Tensor[]): number {
    const h1 = this.encoder.forward(tree1, inputs1);
    const h2 = this.encoder.forward(tree2, inputs2);
    return h1.cosineSimilarity(h2);
  }
}

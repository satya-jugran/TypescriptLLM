/**
 * Embedding.ts — Learnable lookup tables.
 *
 * `TokenEmbedding` maps an integer token id to a dense vector; mathematically
 * it is a one-hot vector e_id multiplied by a weight matrix W of shape
 * [vocabSize, nEmbd]. Materializing the one-hot is wasteful, so we simply
 * gather rows — and scatter the gradient back into the exact same rows on
 * the backward pass (a row of W is touched once per use; multiple uses of
 * the same id in the batch accumulate, which is exactly what the chain rule
 * prescribes).
 *
 * `PositionEmbedding` is the same operation over *positions* 0..blockSize-1
 * rather than token identities. It is what injects word ORDER into the
 * model: attention itself is permutation-equivariant (shuffle the inputs
 * and the attention scores shuffle with them), so without positional
 * information "cat bit dog" and "dog bit cat" would look identical.
 */

import { Tensor } from "../autograd/Tensor";
import { Module, param, type Parameter } from "./Module";

export class TokenEmbedding extends Module {
  readonly weight: Parameter; // [vocabSize, nEmbd]

  constructor(vocabSize: number, nEmbd: number, std = 0.02) {
    super();
    this.weight = param(Tensor.randn([vocabSize, nEmbd], std)).setRequiresGrad(true) as Parameter;
    this.register(this.weight);
  }

  /** indices: int array of shape [B, T] → output [B, T, nEmbd]. */
  forward(indices: number[]): Tensor {
    const out = new Float64Array(indices.length * this.weight.shape[1]);
    const V = this.weight.shape[0];
    const E = this.weight.shape[1];
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (!Number.isInteger(idx) || idx < 0 || idx >= V) {
        throw new Error(`token id ${idx} out of range [0, ${V})`);
      }
      for (let e = 0; e < E; e++) out[i * E + e] = this.weight.data[idx * E + e];
    }
    const outShape = [...shapeOf(indices), E];
    const outT = new Tensor(out, outShape, true);
    outT.attachParent(this.weight, () => {
      // Scatter-add: row `idx` of the weight gets the incoming gradient row.
      for (let i = 0; i < indices.length; i++)
        for (let e = 0; e < E; e++)
          this.weight.grad[indices[i] * E + e] += outT.grad[i * E + e];
    });
    return outT;
  }
}

export class PositionEmbedding extends Module {
  readonly weight: Parameter; // [blockSize, nEmbd]

  constructor(blockSize: number, nEmbd: number, std = 0.02) {
    super();
    this.weight = param(Tensor.randn([blockSize, nEmbd], std)).setRequiresGrad(true) as Parameter;
    this.register(this.weight);
  }

  /** positions: int array [B, T] (typically just 0..T-1 for every row of the batch) → [B, T, nEmbd]. */
  forward(positions: number[]): Tensor {
    const E = this.weight.shape[1];
    const P = this.weight.shape[0];
    const out = new Float64Array(positions.length * E);
    for (let i = 0; i < positions.length; i++) {
      const pos = positions[i];
      if (!Number.isInteger(pos) || pos < 0 || pos >= P) {
        throw new Error(`position ${pos} out of range [0, ${P})`);
      }
      for (let e = 0; e < E; e++) out[i * E + e] = this.weight.data[pos * E + e];
    }
    const outT = new Tensor(out, [...shapeOf(positions), E], true);
    outT.attachParent(this.weight, () => {
      for (let i = 0; i < positions.length; i++)
        for (let e = 0; e < E; e++)
          this.weight.grad[positions[i] * E + e] += outT.grad[i * E + e];
    });
    return outT;
  }
}

/** Infer the [B, T] shape of a flat index array when the caller passed it nested. */
function shapeOf(indices: number[]): number[] {
  if (indices.length === 0) return [0];
  if (Array.isArray(indices[0])) {
    // Nested: [[...],[...]] → [outer length, inner length].
    const first = indices[0] as unknown as number[];
    return [indices.length, first.length];
  }
  // Flat array — caller supplies its own [B*T]; we treat it as one long row
  // and let the caller reshape. Shapes here are only used for bookkeeping.
  return [1, indices.length];
}

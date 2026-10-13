/**
 * FeedForward.ts — Position-wise feed-forward MLP inside a transformer block.
 *
 * Structure: «expand → nonlinearity → project back», applied INDEPENDENTLY to
 * each token position (hence "position-wise"):
 *
 *   FF(x) = GELU(x·W1 + b1)·W2 + b2       with W1: [C, 4C], W2: [4C, C]
 *
 * The 4× hidden expansion is the standard transformer convention. Intuition:
 * attention moves information BETWEEN tokens, and the FF layer is where each
 * token gets private "thinking time" to transform it — in effect a key-value
 * memory of learned patterns (see Geva et al., "Transformer FF Layers Are
 * Key-Value Memories"). Together attention + FF are the two alternating
 * ingredients of every transformer block.
 *
 * Why GELU instead of ReLU? Smooth at zero (ReLU's hard kink wastes gradient
 * mass on the boundary) with a small negative tail that keeps information
 * flowing for slightly-negative inputs; the tanh approximation used here is
 * what GPT-2 itself uses.
 */

import { Tensor } from "../autograd/Tensor";
import { Module } from "./Module";
import { Linear } from "./Linear";

export class FeedForward extends Module {
  readonly fc1: Linear; // C → 4C
  readonly fc2: Linear; // 4C → C

  constructor(nEmbd: number) {
    super();
    const hidden = nEmbd * 4;
    this.fc1 = new Linear(nEmbd, hidden, true);
    this.fc2 = new Linear(hidden, nEmbd, true);
    this.add(this.fc1);
    this.add(this.fc2);
  }

  /** x: [B, T, C] → [B, T, C] */
  forward(x: Tensor): Tensor {
    return this.fc2.forward(this.fc1.forward(x).gelu());
  }
}

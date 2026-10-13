/**
 * GPT.ts — The full character-level autoregressive language model.
 *
 * Architecture (top to bottom):
 *
 *   tokens [B, T]
 *     ├── TokenEmbedding      → [B, T, C]  (what each character means)
 *     ├── + PositionEmbedding → [B, T, C]  (where each character sits)
 *     ├── N × TransformerBlock             (attend + think, residually)
 *     ├── final LayerNorm                  (stable logit scale)
 *     └── lmHead (Linear C → vocab)        (score every possible next char)
 *
 * The [B, T, vocabSize] logits are read as a batch of independent
 * classification problems: at every position t the model predicts
 * tokens[t+1], the NEXT character — which works only because the attention
 * inside every block is strictly causal.
 */

import { Tensor } from "../autograd/Tensor";
import { Module } from "./Module";
import { TokenEmbedding, PositionEmbedding } from "./Embedding";
import { TransformerBlock } from "./TransformerBlock";
import { LayerNorm } from "./LayerNorm";
import { Linear } from "./Linear";

export interface GPTConfig {
  /** Number of distinct characters (derived from the training corpus). */
  vocabSize: number;
  /** Context window: how many characters the model may condition on. */
  blockSize: number;
  /** Embedding / hidden width (must be divisible by nHead). */
  nEmbd: number;
  /** Parallel attention heads per block. */
  nHead: number;
  /** Stacked transformer blocks. */
  nLayer: number;
}

export class GPT extends Module {
  readonly config: GPTConfig;
  readonly tokenEmbedding: TokenEmbedding;
  readonly positionEmbedding: PositionEmbedding;
  readonly blocks: TransformerBlock[];
  readonly lnFinal: LayerNorm;
  readonly lmHead: Linear;

  constructor(config: GPTConfig) {
    super();
    if (config.nEmbd % config.nHead !== 0) {
      throw new Error(`nEmbd (${config.nEmbd}) must be divisible by nHead (${config.nHead})`);
    }
    this.config = config;
    this.tokenEmbedding = new TokenEmbedding(config.vocabSize, config.nEmbd);
    this.positionEmbedding = new PositionEmbedding(config.blockSize, config.nEmbd);
    this.blocks = [];
    for (let i = 0; i < config.nLayer; i++) {
      const block = new TransformerBlock(config.nEmbd, config.nHead);
      this.blocks.push(block);
      this.add(block);
    }
    this.lnFinal = new LayerNorm(config.nEmbd);
    this.lmHead = new Linear(config.nEmbd, config.vocabSize, true);
    this.add(this.lnFinal);
    this.add(this.lmHead);
  }

  /**
   * Forward pass.
   * @param indices flat token ids, length B·T, row-major [B, T]
   * @param B batch size
   * @param T sequence length (≤ blockSize)
   * @returns logits [B, T, vocabSize]
   */
  forward(indices: number[], B: number, T: number): Tensor {
    if (indices.length !== B * T) {
      throw new Error(`expected ${B * T} token ids, got ${indices.length}`);
    }
    if (T > this.config.blockSize) {
      throw new Error(`sequence length ${T} exceeds blockSize ${this.config.blockSize}`);
    }

    // --- embed tokens + add positional signal -------------------------------
    let x = this.tokenEmbedding.forward(indices); // [B, T, C] (flat [B*T, C]-shaped [1, B*T, C]-ish view)
    // positions 0..T-1 repeated for every batch row, then added (broadcast add
    // over the batch dim: posEmb has shape [B, T, C] — one shared table,
    // broadcast-copied by Tensor.add's i % len rule).
    const positions: number[] = [];
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++) positions.push(t);
    x = x.add(this.positionEmbedding.forward(positions)); // [B*T, C] values, shape [B, T, C]

    // --- transformer stacks -------------------------------------------------
    for (const block of this.blocks) x = block.forward(x);

    // --- final norm + classification head -----------------------------------
    x = this.lnFinal.forward(x);
    const logits = this.lmHead.forward(x); // [B, T, V] (Batch-1 may collapse in matmul)
    // Restore the canonical [B, T, V] shape so callers always get 3-D logits.
    return logits.reshape([B, T, logits.shape[logits.shape.length - 1]]);
  }

  /**
   * Autoregressive sampling.
   *
   * Strategy: feed the model the prompt, then repeatedly (a) crop the
   * context to the last `blockSize` tokens — the model literally cannot use
   * more history — (b) read the logits at the LAST position (it has seen
   * everything before it, nothing after it), (c) adjust with temperature and
   * (d) sample.
   *
   * TEMPERATURE: dividing the logits by T>1 flattens the distribution
   * (more random / creative), T<1 sharpens it (safer, more repetitive);
   * T→0 collapses to argmax (greedy decoding).
   */
  generate(
    context: number[],
    maxNewTokens: number,
    temperature = 1.0,
  ): number[] {
    if (context.length === 0) throw new Error("generate needs a non-empty prompt");
    const out = [...context];
    for (let step = 0; step < maxNewTokens; step++) {
      // Crop to the most recent blockSize tokens (hard context limit).
      const cond = out.length > this.config.blockSize
        ? out.slice(out.length - this.config.blockSize)
        : out;
      const B = 1, T = cond.length;
      const logits = this.forward(cond, B, T); // [1, T, V]
      // The last position's logits predict token number out.length.
      const V = logits.shape[logits.shape.length - 1];
      const lastRow = logits.size - V; // flat offset of the final vocab row
      let best = -Infinity;
      let sampled: number;
      if (temperature <= 1e-9) {
        // Greedy: pick the argmax logit.
        let argmax = 0;
        for (let j = 0; j < V; j++) {
          if (logits.data[lastRow + j] > best) { best = logits.data[lastRow + j]; argmax = j; }
        }
        sampled = argmax;
      } else {
        // Temperature-scaled softmax → sample from the categorical law.
        const row = new Float64Array(V);
        let mx = -Infinity;
        for (let j = 0; j < V; j++) {
          row[j] = logits.data[lastRow + j] / temperature;
          if (row[j] > mx) mx = row[j];
        }
        let sum = 0;
        for (let j = 0; j < V; j++) { row[j] = Math.exp(row[j] - mx); sum += row[j]; }
        let r = Math.random() * sum;
        sampled = V - 1;
        for (let j = 0; j < V; j++) { r -= row[j]; if (r <= 0) { sampled = j; break; } }
      }
      out.push(sampled);
    }
    return out;
  }
}

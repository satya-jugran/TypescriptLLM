/**
 * Attention.ts — Causal multi-head self-attention, the core of the
 * transformer. Reads x of shape [B, T, C] and writes [B, T, C].
 *
 * ===================== step by step =====================
 *
 * 1. LINEAR PROJECTIONS: the input is projected three times to create
 *        Q = x·Wq   (what each position is *looking for*)
 *        K = x·Wk   (what each position *contains*, as an index)
 *        V = x·Wv   (what each position *offers*, as content to aggregate)
 *
 * 2. HEADS: the C channels are split into nHead heads of size headDim =
 *    C / nHead, each an independent attention "specialist" with its own
 *    subspace of Q/K/V. Splitting (rather than running nHead separate
 *    full-size attentions) keeps the cost identical to one head while
 *    letting different heads learn different relational patterns.
 *
 * 3. SCORES: for each pair (query position t, key position f) compute the
 *    dot product similarity   s_{t,f} = q_t · k_f / √(headDim).
 *    Dividing by √(headDim) matters: if q and k had independent components
 *    of variance σ², their dot product would have variance headDim·σ⁴, and
 *    feeding wild scores into softmax saturates it into one-hot vectors
 *    where gradients vanish. The √d factor keeps the scores at variance ~1,
 *    hence the name "scaled dot-product attention".
 *
 * 4. CAUSAL MASK: token t may only look at positions f ≤ t. We add
 *    −Infinity to every score s_{t,f} with f > t (see Operations.ts):
 *    exp(−∞) = 0 exactly, so future tokens get probability 0 and the
 *    autoregressive "next-token prediction" training signal is honest —
 *    position t never sees the letter it is being asked to predict. This is
 *    also what makes generation O(n) efficient: each new token attends to
 *    everything before it, nothing more.
 *
 * 5. SOFTMAX: converts each row of masked scores into a probability
 *    distribution over which past tokens to copy information from.
 *
 * 6. VALUE MIX: each position's output is the probability-weighted sum of
 *    value vectors:  out_t = Σ_f p_{t,f}·v_f.
 *
 * 7. OUTPUT PROJECTION: the concatenated heads pass through a final linear
 *    W_o + b, so heads can interact before feeding the residual stream.
 */

import { Tensor } from "../autograd/Tensor";
import { applyCausalMask } from "../autograd/Operations";
import { Module, param, type Parameter } from "./Module";
import { Linear } from "./Linear";

export class MultiHeadAttention extends Module {
  readonly qProj: Linear;
  readonly kProj: Linear;
  readonly vProj: Linear;
  readonly outProj: Linear;
  readonly nHead: number;
  readonly headDim: number;

  constructor(nEmbd: number, nHead: number) {
    super();
    if (nEmbd % nHead !== 0) {
      throw new Error(`nEmbd (${nEmbd}) must be divisible by nHead (${nHead})`);
    }
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    // Projection matrices (no bias here — the output projection has one, and
    // the extra parameters buy little; this mirrors nanoGPT's defaults).
    this.qProj = new Linear(nEmbd, nEmbd, false);
    this.kProj = new Linear(nEmbd, nEmbd, false);
    this.vProj = new Linear(nEmbd, nEmbd, false);
    this.outProj = new Linear(nEmbd, nEmbd, true);
    for (const child of [this.qProj, this.kProj, this.vProj, this.outProj]) {
      this.add(child);
    }
  }

  /** x: [B, T, C] → [B, T, C] */
  forward(x: Tensor): Tensor {
    const B = x.shape[0];
    const T = x.shape[1];
    const C = x.shape[2];
    const H = this.nHead;
    const Hd = this.headDim;

    // 1. Project  [B, T, C] → [B, T, C]  then re-view as heads: [B, H, T, Hd].
    const q = this.viewHeads(this.qProj.forward(x), B, H, T, Hd);
    const k = this.viewHeads(this.kProj.forward(x), B, H, T, Hd);
    const v = this.viewHeads(this.vProj.forward(x), B, H, T, Hd);

    // 2. Scaled dot-product scores:  q @ kᵀ  → [B, H, T, T], / √(headDim).
    //    transposeLast2 makes the contraction over the Hd (feature) axis.
    const scale = new Tensor(new Float64Array([1 / Math.sqrt(Hd)]), [1, 1], false);
    let scores = q.matmul(k.transposeLast2()).mul(scale);

    // 3. Causal mask, then softmax rows → attention probabilities.
    scores = applyCausalMask(scores).softmaxRows();

    // 4. Weighted sum of values:  probs @ v  → [B, H, T, Hd].
    let context = scores.matmul(v);

    // 5. Merge heads back to [B, T, C] and apply the output projection.
    context = this.mergeHeads(context, B, H, T, Hd);
    return this.outProj.forward(context);
  }

  /**
   * Reshape [B, T, C] → [B, H, T, Hd] by splitting the channel axis.
   * Data layout is preserved: row-major, the C axis is really H·Hd.
   */
  private viewHeads(t: Tensor, B: number, H: number, T: number, Hd: number): Tensor {
    return t.reshape([B, H, T, Hd]);
  }

  /** Reshape [B, H, T, Hd] → [B, T, H·Hd] (concatenate the heads). */
  private mergeHeads(t: Tensor, B: number, H: number, T: number, Hd: number): Tensor {
    return t.reshape([B, T, H * Hd]);
  }
}

/**
 * FeedForward.ts companion table is in FeedForward.ts. This file also hosts
 * the production helpers for backward-compat: `param` / `Parameter` are
 * re-exported so TransformerBlock and GPT can stay concise.
 */
export { param, type Parameter };

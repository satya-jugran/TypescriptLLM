/**
 * TransformerBlock.ts — One full pre-LN transformer layer, the periodic
 * building unit of GPT.
 *
 * Structure (pre-LN, as in GPT-2 / nanoGPT):
 *
 *   x ← x + MultiHeadAttention(LayerNorm(x))     ← communication step
 *   x ← x + FeedForward(LayerNorm(x))            ← computation step
 *
 * THE RESIDUAL CONNECTIONS ARE THE BACKBONE: adding the branch output back
 * to x gives gradients an unimpeded identity path from the loss all the way
 * down to the embeddings, which is what lets us stack many layers without
 * vanishing gradients. Each branch is normalised just before it transforms
 * x, so the residual stream itself stays un-normalized and can keep ambient
 * information (variance, rare features) that clamping to unit scale would
 * erase.
 */

import { Tensor } from "../autograd/Tensor";
import { Module } from "./Module";
import { LayerNorm } from "./LayerNorm";
import { MultiHeadAttention } from "./Attention";
import { FeedForward } from "./FeedForward";

export class TransformerBlock extends Module {
  readonly ln1: LayerNorm;
  readonly attn: MultiHeadAttention;
  readonly ln2: LayerNorm;
  readonly ff: FeedForward;

  constructor(nEmbd: number, nHead: number) {
    super();
    this.ln1 = new LayerNorm(nEmbd);
    this.attn = new MultiHeadAttention(nEmbd, nHead);
    this.ln2 = new LayerNorm(nEmbd);
    this.ff = new FeedForward(nEmbd);
    for (const child of [this.ln1, this.attn, this.ln2, this.ff]) this.add(child);
  }

  /** x: [B, T, C] → [B, T, C] */
  forward(x: Tensor): Tensor {
    // Communication: each token asks the others for relevant context.
    x = x.add(this.attn.forward(this.ln1.forward(x)));
    // Computation: each token privately transforms what it gathered.
    x = x.add(this.ff.forward(this.ln2.forward(x)));
    return x;
  }
}

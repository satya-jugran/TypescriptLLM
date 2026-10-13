/**
 * Multi-Head Attention (MHA) & Grouped-Query Attention (GQA).
 * Implements Scaled Dot-Product Attention with Rotary Position Embeddings and Decoupled KV-Cache.
 */

import { Module } from './module.js';
import { Linear } from './linear.js';
import { RotaryEmbedding } from './rope.js';
import { Tensor } from '../core/tensor.js';
import { KVCacheManager } from '../core/kv-cache.js';

export class MultiHeadAttention extends Module {
  public readonly qProj: Linear;
  public readonly kProj: Linear;
  public readonly vProj: Linear;
  public readonly oProj: Linear;
  public readonly rope: RotaryEmbedding;

  public readonly numHeads: number;
  public readonly numKvHeads: number;
  public readonly headDim: number;
  public readonly numGroups: number;
  public readonly scale: number;

  // Scratch buffer for attention scores to eliminate GC pressure
  private scoresBuffer: Float32Array;

  constructor(
    qProj: Linear,
    kProj: Linear,
    vProj: Linear,
    oProj: Linear,
    rope: RotaryEmbedding,
    numHeads: number,
    numKvHeads: number,
    headDim: number
  ) {
    super();
    this.qProj = qProj;
    this.kProj = kProj;
    this.vProj = vProj;
    this.oProj = oProj;
    this.rope = rope;

    this.numHeads = numHeads;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.numGroups = Math.floor(numHeads / numKvHeads);
    this.scale = 1.0 / Math.sqrt(headDim);

    this.scoresBuffer = new Float32Array(2048);
  }

  /**
   * Attention forward pass:
   * 1. Project Q, K, V
   * 2. Apply RoPE to Q, K
   * 3. Update KV cache
   * 4. Multi-head / Grouped-query scaled dot-product attention with causal mask
   * 5. Output projection
   */
  public forward(
    x: Tensor,
    startPos: number,
    kvCache: KVCacheManager,
    layerIdx: number
  ): Tensor {
    const seqLen = x.shape[0];

    // 1. Projections
    const q = this.qProj.forward(x); // [seqLen, numHeads * headDim]
    const k = this.kProj.forward(x); // [seqLen, numKvHeads * headDim]
    const v = this.vProj.forward(x); // [seqLen, numKvHeads * headDim]

    // 2. RoPE on Q and K
    const qRot = this.rope.forward(q, startPos, this.numHeads);
    const kRot = this.rope.forward(k, startPos, this.numKvHeads);

    // 3. Update KV Cache
    kvCache.append(layerIdx, kRot, v);
    const totalSeqLen = startPos + seqLen;

    // Retrieve active cached K and V
    const layer = kvCache.getLayer(layerIdx);
    const kData = layer.keyCache.data;
    const vData = layer.valueCache.data;

    const kvDim = kvCache.kvDim;
    const headDim = this.headDim;
    const numHeads = this.numHeads;
    const numGroups = this.numGroups;
    const qTotalDim = numHeads * headDim;

    // Ensure scores buffer capacity
    if (this.scoresBuffer.length < totalSeqLen) {
      this.scoresBuffer = new Float32Array(totalSeqLen * 2);
    }
    const scores = this.scoresBuffer;

    // Allocate attention output buffer [seqLen, numHeads * headDim]
    const attnOutData = new Float32Array(seqLen * qTotalDim);
    const qData = qRot.data;

    // 4. Scaled Dot-Product Attention for each query token
    for (let s = 0; s < seqLen; s++) {
      const qTokenOffset = s * qTotalDim;
      const outTokenOffset = s * qTotalDim;
      const currentPos = startPos + s;
      const validKeyCount = currentPos + 1; // Causal constraint: attend only to <= currentPos

      for (let hq = 0; hq < numHeads; hq++) {
        // Grouped Query Attention: map query head to KV head
        const hkv = Math.floor(hq / numGroups);
        const qHeadOffset = qTokenOffset + hq * headDim;

        // Compute dot products with all keys up to currentPos
        let maxScore = -Infinity;
        for (let t = 0; t < validKeyCount; t++) {
          const kOffset = t * kvDim + hkv * headDim;

          let dot = 0.0;
          let d = 0;
          const limit = headDim - 7;
          for (; d < limit; d += 8) {
            dot +=
              qData[qHeadOffset + d] * kData[kOffset + d] +
              qData[qHeadOffset + d + 1] * kData[kOffset + d + 1] +
              qData[qHeadOffset + d + 2] * kData[kOffset + d + 2] +
              qData[qHeadOffset + d + 3] * kData[kOffset + d + 3] +
              qData[qHeadOffset + d + 4] * kData[kOffset + d + 4] +
              qData[qHeadOffset + d + 5] * kData[kOffset + d + 5] +
              qData[qHeadOffset + d + 6] * kData[kOffset + d + 6] +
              qData[qHeadOffset + d + 7] * kData[kOffset + d + 7];
          }
          for (; d < headDim; d++) {
            dot += qData[qHeadOffset + d] * kData[kOffset + d];
          }

          const score = dot * this.scale;
          scores[t] = score;
          if (score > maxScore) {
            maxScore = score;
          }
        }

        // Softmax over valid keys
        let sumExp = 0.0;
        for (let t = 0; t < validKeyCount; t++) {
          const expVal = Math.exp(scores[t] - maxScore);
          scores[t] = expVal;
          sumExp += expVal;
        }

        const invSum = 1.0 / sumExp;
        for (let t = 0; t < validKeyCount; t++) {
          scores[t] *= invSum;
        }

        // Weighted sum of values: AttnOut = sum_t (score[t] * V[t])
        const outHeadOffset = outTokenOffset + hq * headDim;
        for (let t = 0; t < validKeyCount; t++) {
          const weight = scores[t];
          const vOffset = t * kvDim + hkv * headDim;

          for (let d = 0; d < headDim; d++) {
            attnOutData[outHeadOffset + d] += weight * vData[vOffset + d];
          }
        }
      }
    }

    const attnOut = new Tensor([seqLen, qTotalDim], attnOutData);

    // 5. Output projection: Y = AttnOut * W_o^T
    return this.oProj.forward(attnOut);
  }
}

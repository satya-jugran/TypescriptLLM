/**
 * Transformer Block combining RMSNorm, Multi-Head Attention, SwiGLU MLP, and Residual Connections.
 */

import { Module } from './module.js';
import { RMSNorm } from './rms-norm.js';
import { MultiHeadAttention } from './attention.js';
import { SwiGLUMLP } from './mlp.js';
import { Tensor } from '../core/tensor.js';
import { KVCacheManager } from '../core/kv-cache.js';

export class TransformerBlock extends Module {
  public readonly layerIdx: number;
  public readonly inputLayernorm: RMSNorm;
  public readonly selfAttn: MultiHeadAttention;
  public readonly postAttentionLayernorm: RMSNorm;
  public readonly mlp: SwiGLUMLP;

  constructor(
    layerIdx: number,
    inputLayernorm: RMSNorm,
    selfAttn: MultiHeadAttention,
    postAttentionLayernorm: RMSNorm,
    mlp: SwiGLUMLP
  ) {
    super();
    this.layerIdx = layerIdx;
    this.inputLayernorm = inputLayernorm;
    this.selfAttn = selfAttn;
    this.postAttentionLayernorm = postAttentionLayernorm;
    this.mlp = mlp;
  }

  public forward(
    x: Tensor,
    startPos: number,
    kvCache: KVCacheManager
  ): Tensor {
    // 1. Pre-attention norm & Self Attention
    const norm1 = this.inputLayernorm.forward(x);
    const attnOut = this.selfAttn.forward(norm1, startPos, kvCache, this.layerIdx);

    // 2. Residual connection 1: x + attnOut
    const h = attnOut.addInPlace(x);

    // 3. Pre-MLP norm & SwiGLU MLP
    const norm2 = this.postAttentionLayernorm.forward(h);
    const mlpOut = this.mlp.forward(norm2);

    // 4. Residual connection 2: h + mlpOut
    return mlpOut.addInPlace(h);
  }
}

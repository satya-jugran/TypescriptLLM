/**
 * Complete Transformer Model (Llama / SmolLM Architecture).
 * Encapsulates Token Embedding, Transformer Blocks, Final RMSNorm, and LM Head Projection.
 */

import { Module } from '../layers/module.js';
import { TransformerBlock } from '../layers/transformer-block.js';
import { RMSNorm } from '../layers/rms-norm.js';
import { Linear } from '../layers/linear.js';
import { Tensor } from '../core/tensor.js';
import { KVCacheManager } from '../core/kv-cache.js';
import { LlamaConfig } from '../core/types.js';

export class TransformerModel extends Module {
  public readonly config: LlamaConfig;
  public readonly embedTokens: Tensor;
  public readonly layers: readonly TransformerBlock[];
  public readonly norm: RMSNorm;
  public readonly lmHead: Linear;

  constructor(
    config: LlamaConfig,
    embedTokens: Tensor,
    layers: TransformerBlock[],
    norm: RMSNorm,
    lmHead: Linear
  ) {
    super();
    this.config = config;
    this.embedTokens = embedTokens;
    this.layers = Object.freeze([...layers]);
    this.norm = norm;
    this.lmHead = lmHead;
  }

  /**
   * Forward pass through the Transformer.
   *
   * @param inputIds Token IDs to process (array of token integers)
   * @param startPos Position in sequence (0 for prefill prompt, or current index for decode)
   * @param kvCache Key-Value cache manager
   * @param returnAllLogits If true, outputs logits for all tokens; if false, only the last token
   * @returns Logits tensor: [vocabSize] (if !returnAllLogits) or [seqLen, vocabSize]
   */
  public forward(
    inputIds: number[],
    startPos: number,
    kvCache: KVCacheManager,
    returnAllLogits = false
  ): Tensor {
    const seqLen = inputIds.length;
    if (seqLen === 0) {
      throw new Error('inputIds cannot be empty');
    }

    const hiddenSize = this.config.hidden_size;
    const tableData = this.embedTokens.data;
    const embedData = new Float32Array(seqLen * hiddenSize);

    // 1. Embedding lookup: extract rows for inputIds
    for (let i = 0; i < seqLen; i++) {
      const tokenId = inputIds[i];
      if (tokenId < 0 || tokenId >= this.config.vocab_size) {
        throw new RangeError(
          `Token ID ${tokenId} out of vocabulary range [0, ${this.config.vocab_size})`
        );
      }
      const tableOffset = tokenId * hiddenSize;
      const targetOffset = i * hiddenSize;
      embedData.set(tableData.subarray(tableOffset, tableOffset + hiddenSize), targetOffset);
    }

    let h = new Tensor([seqLen, hiddenSize], embedData);

    // 2. Pass sequentially through all transformer layers
    for (let l = 0; l < this.layers.length; l++) {
      h = this.layers[l].forward(h, startPos, kvCache);
    }

    // 3. Final layer normalization
    h = this.norm.forward(h);

    // 4. LM Head projection
    if (!returnAllLogits) {
      // Optimize generation: project ONLY the final token to logits
      const lastTokenHidden = h.slice(seqLen - 1, seqLen);
      return this.lmHead.forward(lastTokenHidden);
    } else {
      return this.lmHead.forward(h);
    }
  }
}

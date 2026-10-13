/**
 * Sampler interface following the Strategy Pattern.
 */

import { Tensor } from '../core/tensor.js';

export interface ISampler {
  /**
   * Sample the next token ID from output logits.
   *
   * @param logits Raw unnormalized logits from LM Head (shape: [vocab_size] or [1, vocab_size])
   * @param contextTokens Previously generated token IDs (used for repetition penalty)
   * @returns Chosen next token ID
   */
  sample(logits: Tensor, contextTokens: readonly number[]): number;
}

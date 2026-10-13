/**
 * Greedy Sampler (Argmax Strategy).
 * Selects the token with the highest logit score.
 */

import { ISampler } from './sampler.interface.js';
import { Tensor } from '../core/tensor.js';

export class GreedySampler implements ISampler {
  public readonly repetitionPenalty: number;

  constructor(repetitionPenalty = 1.0) {
    this.repetitionPenalty = repetitionPenalty;
  }

  public sample(logits: Tensor, contextTokens: readonly number[]): number {
    const data = logits.data;
    const vocabSize = data.length;

    if (this.repetitionPenalty === 1.0 || contextTokens.length === 0) {
      // Direct fast argmax
      return logits.argmax();
    }

    // Apply repetition penalty before finding argmax
    const penalized = new Float32Array(data);
    const seen = new Set(contextTokens);

    for (const tokenId of seen) {
      if (tokenId >= 0 && tokenId < vocabSize) {
        const val = penalized[tokenId];
        penalized[tokenId] = val > 0 ? val / this.repetitionPenalty : val * this.repetitionPenalty;
      }
    }

    let maxIdx = 0;
    let maxVal = penalized[0];
    for (let i = 1; i < vocabSize; i++) {
      if (penalized[i] > maxVal) {
        maxVal = penalized[i];
        maxIdx = i;
      }
    }

    return maxIdx;
  }
}

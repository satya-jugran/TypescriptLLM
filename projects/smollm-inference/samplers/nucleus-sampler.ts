/**
 * Nucleus Sampler (Top-P, Top-K, Temperature, and Repetition Penalty Strategy).
 */

import { ISampler } from './sampler.interface.js';
import { Tensor } from '../core/tensor.js';

interface TokenCandidate {
  index: number;
  prob: number;
}

export class NucleusSampler implements ISampler {
  public readonly temperature: number;
  public readonly topK: number;
  public readonly topP: number;
  public readonly repetitionPenalty: number;

  constructor(
    temperature = 0.7,
    topP = 0.9,
    topK = 50,
    repetitionPenalty = 1.1
  ) {
    this.temperature = Math.max(1e-5, temperature);
    this.topP = Math.max(0.0, Math.min(1.0, topP));
    this.topK = Math.max(0, topK);
    this.repetitionPenalty = Math.max(1.0, repetitionPenalty);
  }

  public sample(logits: Tensor, contextTokens: readonly number[]): number {
    const rawData = logits.data;
    const vocabSize = rawData.length;
    const scaled = new Float32Array(vocabSize);
    scaled.set(rawData);

    // 1. Repetition Penalty
    if (this.repetitionPenalty !== 1.0 && contextTokens.length > 0) {
      const seen = new Set(contextTokens);
      for (const tokenId of seen) {
        if (tokenId >= 0 && tokenId < vocabSize) {
          const val = scaled[tokenId];
          scaled[tokenId] = val > 0 ? val / this.repetitionPenalty : val * this.repetitionPenalty;
        }
      }
    }

    // 2. Temperature scaling
    const invTemp = 1.0 / this.temperature;
    let maxLogit = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const v = scaled[i] * invTemp;
      scaled[i] = v;
      if (v > maxLogit) {
        maxLogit = v;
      }
    }

    // 3. Compute Softmax Probabilities
    const probs = new Float32Array(vocabSize);
    let sumExp = 0.0;
    for (let i = 0; i < vocabSize; i++) {
      const expVal = Math.exp(scaled[i] - maxLogit);
      probs[i] = expVal;
      sumExp += expVal;
    }
    const invSum = 1.0 / sumExp;
    for (let i = 0; i < vocabSize; i++) {
      probs[i] *= invSum;
    }

    // 4. Candidate collection (with Top-K consideration)
    let candidates: TokenCandidate[];

    if (this.topK > 0 && this.topK < vocabSize) {
      // Collect top K candidates using partial selection
      candidates = this.getTopKCandidates(probs, this.topK);
    } else {
      candidates = new Array<TokenCandidate>(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        candidates[i] = { index: i, prob: probs[i] };
      }
      candidates.sort((a, b) => b.prob - a.prob);
    }

    // 5. Top-P (Nucleus) Truncation
    if (this.topP < 1.0) {
      let cumulativeProb = 0.0;
      let cutoffIndex = candidates.length - 1;

      for (let i = 0; i < candidates.length; i++) {
        cumulativeProb += candidates[i].prob;
        if (cumulativeProb >= this.topP) {
          cutoffIndex = i;
          break;
        }
      }

      candidates = candidates.slice(0, cutoffIndex + 1);
    }

    // 6. Re-normalize retained probabilities
    let filteredSum = 0.0;
    for (let i = 0; i < candidates.length; i++) {
      filteredSum += candidates[i].prob;
    }

    if (filteredSum <= 0.0) {
      return candidates[0]?.index ?? 0;
    }

    const invFilteredSum = 1.0 / filteredSum;
    for (let i = 0; i < candidates.length; i++) {
      candidates[i].prob *= invFilteredSum;
    }

    // 7. Categorical Sampling via random CDF
    const rand = Math.random();
    let cumulative = 0.0;
    for (let i = 0; i < candidates.length; i++) {
      cumulative += candidates[i].prob;
      if (rand <= cumulative) {
        return candidates[i].index;
      }
    }

    // Fallback to top candidate
    return candidates[candidates.length - 1].index;
  }

  /**
   * Helper to retrieve top-k highest probability candidates.
   */
  private getTopKCandidates(probs: Float32Array, k: number): TokenCandidate[] {
    const list: TokenCandidate[] = [];
    const n = probs.length;

    // Use a simple sorted list insertion maintaining size k
    for (let i = 0; i < n; i++) {
      const p = probs[i];
      if (list.length < k) {
        list.push({ index: i, prob: p });
        if (list.length === k) {
          list.sort((a, b) => b.prob - a.prob);
        }
      } else if (p > list[k - 1].prob) {
        // Binary search insertion
        let low = 0;
        let high = k - 1;
        while (low < high) {
          const mid = (low + high) >> 1;
          if (list[mid].prob < p) {
            high = mid;
          } else {
            low = mid + 1;
          }
        }
        list.splice(low, 0, { index: i, prob: p });
        list.pop();
      }
    }

    list.sort((a, b) => b.prob - a.prob);
    return list;
  }
}

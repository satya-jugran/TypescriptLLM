/**
 * Loss.ts — Next-token cross-entropy loss for language modeling.
 *
 * Given the model's logits [B·T, V] (each row scores every possible next
 * character at one (batch, position) slot) and the integer targets (the
 * characters actually appearing one slot later), the loss is
 *
 *   L = (1/N) Σ_i −log softmax(logits_i)[target_i]
 *
 * i.e., the negative log-likelihood the model assigns to the data. Golden
 * property for autoregressive training: minimizing it across shifting
 * windows IS maximum likelihood training of a generator.
 */

import { Tensor } from "../autograd/Tensor";
import { crossEntropyPerExample } from "../autograd/Operations";

export class CrossEntropyLoss {
  /** Stores the last losses so the Trainer can report a running average. */
  lastTerm: Float64Array = new Float64Array(0);

  /**
   * @param logits  [B*T, V] raw scores
   * @param targets length B*T array of integer token ids
   * @returns scalar Tensor loss (mean over all B*T predictions)
   */
  forward(logits: Tensor, targets: number[]): Tensor {
    if (logits.shape.length !== 2) {
      throw new Error(`CrossEntropyLoss expects 2-D [N, V] logits, got [${logits.shape}]`);
    }
    const [N, V] = logits.shape;
    if (targets.length !== N) {
      throw new Error(`expected ${N} targets, got ${targets.length}`);
    }
    const per = crossEntropyPerExample(logits, targets);
    this.lastTerm = per.data;
    return per.mean();
  }
}

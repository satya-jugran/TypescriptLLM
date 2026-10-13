/**
 * AdamW.ts — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
 *
 * Plain Adam keeps two running statistics per parameter:
 *
 *   m ← β₁·m + (1−β₁)·g    (exponential moving average of the gradient)
 *   v ← β₂·v + (1−β₂)·g²   (EMA of the *squared* gradient → rough scale)
 *
 * and then takes a step of size  α·m̂ / (√v̂ + ε),  where m̂, v̂ are the
 * bias-corrected versions (divided by (1−β^t), since both EMAs start at 0
 * and are initially biased toward 0). AdamW's twist is that weight decay is
 * applied DECOUPLED from the adaptive denominator:
 *
 *   θ ← θ − α·m̂/(√v̂+ε)  −  α·λ·θ
 *
 * which differs from L2-in-Adam where λθ gets rescaled by 1/√v̂ — large-
 * gradient coordinates would then also decay less. Decoupling makes decay
 * behave like uniform shrinkage, which generalizes better in practice.
 *
 * For this tiny educational model SGD would work too, but Adam is far more
 * forgiving about the learning rate and therefore the friendlier default.
 */

import type { Parameter } from "../nn/Module";

interface AdamWState {
  m: Float64Array;
  v: Float64Array;
}

export class AdamW {
  readonly params: Parameter[];
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  /** Number of `step()` calls so far — drives bias correction. */
  stepCount = 0;
  private state: Map<Parameter, AdamWState> = new Map();

  constructor(
    params: Parameter[],
    { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01 }: {
      lr?: number; beta1?: number; beta2?: number; eps?: number; weightDecay?: number;
    } = {},
  ) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.weightDecay = weightDecay;
  }

  /** One optimizer step: uses and then leaves current .grad values intact. */
  step(): void {
    this.stepCount++;
    const { lr, beta1, beta2, eps, weightDecay } = this;
    // Bias-correction factors (approach 1 as steps → ∞).
    const bc1 = 1 - Math.pow(beta1, this.stepCount);
    const bc2 = 1 - Math.pow(beta2, this.stepCount);

    for (const p of this.params) {
      let st = this.state.get(p);
      if (!st) {
        st = { m: new Float64Array(p.size), v: new Float64Array(p.size) };
        this.state.set(p, st);
      }
      const { m, v } = st;
      const g = p.grad, d = p.data;
      for (let i = 0; i < p.size; i++) {
        const gi = g[i];
        m[i] = beta1 * m[i] + (1 - beta1) * gi; // grad EMA
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi; // grad² EMA
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        d[i] -= lr * (mHat / (Math.sqrt(vHat) + eps) + weightDecay * d[i]);
      }
    }
  }

  /** Zero every parameter gradient (convenience for the Trainer loop). */
  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }
}

/**
 * Operations.ts — Composite (arity combination) operations built on top of
 * the Tensor primitives. These are the "compound expressions" that show up
 * everywhere in a transformer: softmax-based cross-entropy, masked attention
 * scores, and fused blend ops. Keeping them here lets the `nn/` modules read
 * as pure architecture while the math of their *backward passes* lives in
 * one place.
 */

import { Tensor, numel } from "./Tensor";

/**
 * Causal attention scores with an additive mask applied in the *forward
 * data*, not via multiplication.
 *
 * Given raw scores of shape [..., T, T] we add a lower-triangular mask that
 * is 0 on allowed positions and -Infinity on forbidden (future) ones:
 *
 *   score[t, f] += 0            if f <= t   (past or present → keep)
 *   score[t, f] += -Infinity    if f >  t   (future → exp() becomes 0)
 *
 * Since softmax(x+[0,-∞]) = softmax(x)·[1,0] (elementwise on probabilities),
 * the masked-out rows contribute exactly nothing to the denominator and the
 * attention weights stay a proper probability distribution. Multiplying by
 * an additive-mask of `true`/NaNs would NOT have this property — always
 * notice whether masking is *additive* (pre-softmax) or *multiplicative*
 * (post-softmax) in a code base.
 */
/**
 * Causal attention-scores masking, applied additively in the forward data.
 * (Full rationale in the comment block above; see also Attention.ts step 4.)
 */
export function applyCausalMask(scores: Tensor): Tensor {
  const T = scores.shape[scores.shape.length - 1];
  if (scores.shape[scores.shape.length - 2] !== T)
    throw new Error(`causal mask needs square last-dims [T,T], got [${scores.shape}]`);
  const batch = Math.max(1, numel(scores.shape.slice(0, -2)));
  const negInf = new Float64Array(batch * T * T);
  for (let t = 0; t < T; t++)
    for (let f = t + 1; f < T; f++)
      for (let b = 0; b < batch; b++) {
        negInf[(b * T + t) * T + f] = -Infinity;
      }
  const mask = new Tensor(negInf, scores.shape);
  return scores.add(mask); // add of -Infinity is ~ -Infinity (unless scores is +Inf)
}

/**
 * Cross-entropy for language modeling, fully in autograd ops.
 *
 * After the model's output head produces logits L of shape
 * [B*T, vocabSize] (one row per (sequence, position) training example), the
 * per-example cross-entropy against the integer label y is
 *
 *   CE(i) = −log softmax(L_i)[y_i] = −L_i[y_i] + log Σ_j e^{L_i[j]}
 *
 * which is the negative log-likelihood of the data under the model's
 * predicted distribution. It is THE natural training objective for
 * auto-regressive next-token prediction: minimizing it is equivalent to
 * maximum-likelihood estimation.
 *
 * Implementation trick: to subtract `log Σ e^{L_j}` per row we use the
 * identity    log Σ_j e^{L_j} = logSumExp(L) = m + log Σ_j e^{L_j − m}
 * where m is the row max (numerically stable version of log-sum-exp).
 */
export function crossEntropyPerExample(logits: Tensor, targets: number[]): Tensor {
  const V = logits.shape[logits.shape.length - 1];
  if (logits.shape.length !== 2)
    throw new Error(`crossEntropy expects 2-D [N, V] logits, got [${logits.shape}]`);
  const N = logits.shape[0];
  if (targets.length !== N)
    throw new Error(`got ${targets.length} targets for ${N} logit rows`);

  // −L[j] for all j, then fan out the correct-logit contribution.
  // Doing this with composable ops would create a huge [N, V] tape of `sub`
  // and gather; instead we hand-roll the row-wise gather/un-gather, which is
  // both clearer and dramatically cheaper. The masked gather is written as
  // an on-paper derivation:
  //   dL/dL[i,j] = softmax(L_i)[j] − 1{j == targets[i]}      (per row i)
  // so the backward scatter needs the softmax values, which we keep around.
  const out = new Float64Array(N);
  const probs = new Float64Array(logits.size);
  for (let i = 0; i < N; i++) {
    // Row max for stable log-sum-exp.
    let mx = -Infinity;
    for (let j = 0; j < V; j++) {
      const v = logits.data[i * V + j];
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp(logits.data[i * V + j] - mx);
      probs[i * V + j] = e;
      sum += e;
    }
    for (let j = 0; j < V; j++) probs[i * V + j] /= sum;
    // NLL contribution  −log(softmax[y_i])  =  m − L[y_i] + log Σ e^{L−m}.
    const y = targets[i];
    out[i] = -(Math.log(probs[i * V + y]));
  }
  const outT = new Tensor(out, [N], logits.requiresGrad);
  if (logits.requiresGrad) {
    outT.attachParent(logits, (g) => {
      // d (−log p_y) / d L_j  =  p_j − 1{j == y}, scaled by upstream g[i].
      for (let i = 0; i < N; i++) {
        const scale = g.grad[i]; // incoming ∂L/∂CE_i (mean() divides by N upstream)
        if (scale === 0) continue;
        const y = targets[i];
        for (let j = 0; j < V; j++) {
          logits.grad[i * V + j] += scale * (probs[i * V + j] - (j === y ? 1 : 0));
        }
      }
    });
  }
  return outT;
}

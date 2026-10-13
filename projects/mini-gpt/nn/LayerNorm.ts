/**
 * LayerNorm.ts — Layer Normalization over the LAST dimension of an input.
 *
 * For a vector x of size C (the embedding dim), the forward pass is
 *
 *   y = (x − μ) / sqrt(σ² + ε) · γ + b
 *
 * where μ, σ² are the mean and (biased, dividing by C) variance of x across
 * its channels, and γ (gamma) and b (beta) are learnable scale/shift
 * vectors of length C.
 *
 * Why normalize? Deep stacks of residual blocks drift: activations from
 * early layers can grow or shrink exponentially through the residual stream,
 * exploding the loss surface's curvature and making training unstable.
 * LayerNorm pins every vector back to zero mean / unit variance *per token*
 * (unlike BatchNorm, which works across the batch and is awkward for
 * variable-length autoregressive sequences). Pre-LN transformers (GPT-2,
 * nanoGPT) put the LayerNorms *inside* the residual branches:
 *
 *   x = x + Attn(LN(x));  x = x + FF(LN(x))
 *
 * which keeps a clean, un-normalized residual path flowing from embedding
 * all the way to the logits while each branch sees nicely-scaled inputs.
 */

import { Tensor } from "../autograd/Tensor";
import { Module, param, type Parameter } from "./Module";

export class LayerNorm extends Module {
  readonly gamma: Parameter; // scale, [C]
  readonly beta: Parameter; // shift,  [C]

  constructor(size: number, eps = 1e-5) {
    super();
    this.eps = eps;
    this.gamma = param(Tensor.ones([size])).setRequiresGrad(true) as Parameter;
    this.beta = param(Tensor.zeros([size])).setRequiresGrad(true) as Parameter;
    this.register(this.gamma);
    this.register(this.beta);
  }

  private eps: number;

  forward(input: Tensor): Tensor {
    const C = input.shape[input.shape.length - 1];
    if (C !== this.gamma.size) {
      throw new Error(`LayerNorm expected last-dim ${this.gamma.size}, got [${input.shape}]`);
    }
    const rows = input.size / C; // total vectors being normalized

    // ---- forward: per-row mean and (biased) variance over the C channels ----
    const mean = new Float64Array(rows);
    const rstd = new Float64Array(rows); // reciprocal std: 1/sqrt(var + eps)
    const norm = new Float64Array(input.size); // (x − μ)·rstd, cached for backward
    const out = new Float64Array(input.size);
    for (let r = 0; r < rows; r++) {
      const off = r * C;
      let m = 0;
      for (let c = 0; c < C; c++) m += input.data[off + c];
      m /= C;
      let v = 0;
      for (let c = 0; c < C; c++) {
        const d = input.data[off + c] - m;
        v += d * d;
      }
      v /= C;
      const rs = 1 / Math.sqrt(v + this.eps);
      mean[r] = m;
      rstd[r] = rs;
      for (let c = 0; c < C; c++) {
        norm[off + c] = (input.data[off + c] - m) * rs;
        out[off + c] = norm[off + c] * this.gamma.data[c] + this.beta.data[c];
      }
    }

    const outT = new Tensor(out, [...input.shape], true);
    const gamma = this.gamma, beta = this.beta;
    outT.attachParent(input, (g) => {
      /* -- backward through y = norm·γ + b ------------------------------
       * d/dx_m [ (x_m−μ)·rstd · γ_m ]
       *   = rstd·[ γ_m·g_m  −  mean_j(γ_j·g_j)  −  x̂_m · mean_j(γ_j·g_j·x̂_j) ]
       * (the two mean-terms come from ∂μ/∂x_m and the ∂σ²/∂x_m paths; the
       * eps slightly perturbs rstd, ignored — standard practice.)
       */
      for (let r = 0; r < rows; r++) {
        const off = r * C;
        let gGammaMean = 0; // mean_j(g_j·γ_j)
        let gGammaNormMean = 0; // mean_j(g_j·γ_j·x̂_j)
        for (let c = 0; c < C; c++) {
          const gg = g.grad[off + c] * gamma.data[c];
          gGammaMean += gg;
          gGammaNormMean += gg * norm[off + c];
        }
        gGammaMean /= C;
        gGammaNormMean /= C;
        for (let c = 0; c < C; c++) {
          const gd = g.grad[off + c] * gamma.data[c] - gGammaMean - norm[off + c] * gGammaNormMean;
          input.grad[off + c] += gd * rstd[r];
        }
      }
    });
    outT.attachParent(gamma, (g) => {
      // dL/dγ_c = Σ_batch g[·, c]·x̂[·, c]
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < C; c++) gamma.grad[c] += g.grad[r * C + c] * norm[r * C + c];
    });
    outT.attachParent(beta, (g) => {
      // dL/db_c = Σ_batch g[·, c]  (beta just adds, so it's a plain column sum)
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < C; c++) beta.grad[c] += g.grad[r * C + c];
    });
    return outT;
  }
}

/**
 * SwiGLU Gated Feed-Forward Network (MLP).
 * Computes: Y = down_proj( silu(gate_proj(X)) * up_proj(X) )
 */

import { Module } from './module.js';
import { Linear } from './linear.js';
import { Tensor } from '../core/tensor.js';

export class SwiGLUMLP extends Module {
  public readonly gateProj: Linear;
  public readonly upProj: Linear;
  public readonly downProj: Linear;

  constructor(gateProj: Linear, upProj: Linear, downProj: Linear) {
    super();
    this.gateProj = gateProj;
    this.upProj = upProj;
    this.downProj = downProj;
  }

  public forward(x: Tensor): Tensor {
    // 1. gate = silu(X * W_gate^T)
    const gate = this.gateProj.forward(x);
    gate.siluInPlace();

    // 2. up = X * W_up^T
    const up = this.upProj.forward(x);

    // 3. element-wise product: gate * up
    const gated = gate.mulInPlace(up);

    // 4. down = gated * W_down^T
    return this.downProj.forward(gated);
  }
}

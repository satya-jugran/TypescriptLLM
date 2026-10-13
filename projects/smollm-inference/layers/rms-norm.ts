/**
 * Root Mean Square Layer Normalization (RMSNorm).
 * Used across modern architectures including LLaMA, Mistral, and SmolLM.
 */

import { Module } from './module.js';
import { Tensor } from '../core/tensor.js';

export class RMSNorm extends Module {
  public readonly weight: Tensor;
  public readonly eps: number;
  public readonly dim: number;

  constructor(weight: Tensor, eps = 1e-5) {
    super();
    this.weight = weight;
    this.eps = eps;
    this.dim = weight.size;
  }

  public forward(input: Tensor): Tensor {
    return input.rmsNorm(this.weight, this.eps);
  }
}

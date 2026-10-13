/**
 * Linear (Dense) Projection Layer: Y = X * W^T + b
 */

import { Module } from './module.js';
import { Tensor } from '../core/tensor.js';

export class Linear extends Module {
  public readonly weight: Tensor;
  public readonly bias?: Tensor;
  public readonly inFeatures: number;
  public readonly outFeatures: number;

  constructor(weight: Tensor, bias?: Tensor) {
    super();
    this.weight = weight;
    this.bias = bias;

    if (weight.ndim !== 2) {
      throw new Error(`Linear weight must be 2D [outFeatures, inFeatures], got ${weight.ndim}D`);
    }

    this.outFeatures = weight.shape[0];
    this.inFeatures = weight.shape[1];

    if (bias && bias.size !== this.outFeatures) {
      throw new Error(
        `Bias size (${bias.size}) must match outFeatures (${this.outFeatures})`
      );
    }
  }

  public forward(input: Tensor): Tensor {
    return input.linear(this.weight, this.bias);
  }
}

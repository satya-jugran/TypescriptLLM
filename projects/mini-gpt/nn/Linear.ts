/**
 * Linear.ts — Fully-connected (dense) layer.
 *
 * Computes  y = x·Wᵀ + b  for an input of shape [..., inFeatures] and an
 * output of shape [..., outFeatures], where
 *
 *   W: [outFeatures, inFeatures]   (row-major, like PyTorch's nn.Linear)
 *   b: [outFeatures]
 *
 * All parameters are initialized with N(0, 0.02), the small-normal scheme
 * used by GPT-2/nanoGPT. Keeping initial weights tiny matters: with
 * near-random logits the initial softmax is close to uniform and the initial
 * cross-entropy lands ≈ ln(vocabSize) — the "level of surprise" of a model
 * that knows nothing — which is exactly the baseline we want to watch decay.
 */

import { Tensor } from "../autograd/Tensor";
import { Module, param, type Parameter } from "./Module";

export class Linear extends Module {
  readonly weight: Parameter; // [out, in]
  readonly bias: Parameter | null; // [out]

  /**
   * @param inFeatures  size of the input's last dim
   * @param outFeatures size of the output's last dim
   * @param bias        include a learnable additive bias (default true)
   * @param std         weight init std (default 0.02, GPT-2 style)
   */
  constructor(inFeatures: number, outFeatures: number, bias = true, std = 0.02) {
    super();
    this.weight = param(Tensor.randn([outFeatures, inFeatures], std)).setRequiresGrad(true) as Parameter;
    this.register(this.weight);
    this.bias = bias ? (param(Tensor.zeros([outFeatures])).setRequiresGrad(true) as Parameter) : null;
    if (this.bias) this.register(this.bias);
  }

  forward(input: Tensor): Tensor {
    if (input.shape[input.shape.length - 1] !== this.weight.shape[1]) {
      throw new Error(
        `Linear expected input last-dim ${this.weight.shape[1]}, got [${input.shape}]`,
      );
    }
    // Broadcasting: Wᵀ has shape [in, out]; against input [..., in] the
    // matmul broadcasts over every leading (batch/time) dim at once.
    let y = input.matmul(this.weight.transposeLast2());
    if (this.bias) y = y.add(this.bias);
    return y;
  }
}

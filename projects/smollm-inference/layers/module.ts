/**
 * PyTorch-like nn.Module Base Class for all Neural Network layers.
 */

import { Tensor } from '../core/tensor.js';

export abstract class Module {
  /**
   * Forward pass through the module.
   */
  public abstract forward(...args: unknown[]): Tensor;

  /**
   * Optional helper to collect all learnable weights/tensors inside this module.
   */
  public getParameters(): Record<string, Tensor> {
    const params: Record<string, Tensor> = {};
    for (const [key, value] of Object.entries(this)) {
      if (value instanceof Tensor) {
        params[key] = value;
      } else if (value instanceof Module) {
        const subParams = value.getParameters();
        for (const [subKey, subVal] of Object.entries(subParams)) {
          params[`${key}.${subKey}`] = subVal;
        }
      }
    }
    return params;
  }
}

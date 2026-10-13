/**
 * Rotary Position Embedding (RoPE).
 * Implements the frequency-based rotation matching Hugging Face LLaMA / SmolLM.
 */

import { Module } from './module.js';
import { Tensor } from '../core/tensor.js';

export class RotaryEmbedding extends Module {
  public readonly headDim: number;
  public readonly maxSeqLen: number;
  public readonly theta: number;

  // Precomputed tables: [maxSeqLen, halfDim]
  private readonly cosTable: Float32Array;
  private readonly sinTable: Float32Array;
  private readonly halfDim: number;

  constructor(headDim: number, maxSeqLen = 2048, theta = 10000.0) {
    super();
    this.headDim = headDim;
    this.maxSeqLen = maxSeqLen;
    this.theta = theta;
    this.halfDim = Math.floor(headDim / 2);

    this.cosTable = new Float32Array(maxSeqLen * this.halfDim);
    this.sinTable = new Float32Array(maxSeqLen * this.halfDim);

    this.precomputeFrequencies();
  }

  private precomputeFrequencies(): void {
    // theta_i = base^(-2i / d)
    const invFreq = new Float32Array(this.halfDim);
    for (let i = 0; i < this.halfDim; i++) {
      invFreq[i] = Math.pow(this.theta, (-2 * i) / this.headDim);
    }

    // Populate cos and sin for all positions up to maxSeqLen
    for (let pos = 0; pos < this.maxSeqLen; pos++) {
      const offset = pos * this.halfDim;
      for (let i = 0; i < this.halfDim; i++) {
        const angle = pos * invFreq[i];
        this.cosTable[offset + i] = Math.cos(angle);
        this.sinTable[offset + i] = Math.sin(angle);
      }
    }
  }

  /**
   * Applies RoPE to a 2D or 3D tensor of shape:
   * [seqLen, numHeads, headDim] or [seqLen, totalDim]
   *
   * @param x Input tensor
   * @param startPos Starting token position in the sequence
   * @param numHeads Number of attention or KV heads
   * @returns Rotated tensor
   */
  public forward(x: Tensor, startPos = 0, numHeads?: number): Tensor {
    const seqLen = x.shape[0];
    const totalDim = x.size / seqLen;
    const actualHeads = numHeads ?? totalDim / this.headDim;

    if (actualHeads * this.headDim !== totalDim) {
      throw new Error(
        `Dimension mismatch: totalDim ${totalDim} != numHeads ${actualHeads} * headDim ${this.headDim}`
      );
    }

    const outData = new Float32Array(x.size);
    const inData = x.data;
    const half = this.halfDim;
    const headDim = this.headDim;

    for (let s = 0; s < seqLen; s++) {
      const pos = startPos + s;
      if (pos >= this.maxSeqLen) {
        throw new RangeError(
          `Token position ${pos} exceeds maxSeqLen ${this.maxSeqLen}`
        );
      }

      const tableOffset = pos * half;
      const seqOffset = s * totalDim;

      for (let h = 0; h < actualHeads; h++) {
        const headOffset = seqOffset + h * headDim;

        for (let j = 0; j < half; j++) {
          const c = this.cosTable[tableOffset + j];
          const sVal = this.sinTable[tableOffset + j];

          const v1 = inData[headOffset + j];
          const v2 = inData[headOffset + j + half];

          // HuggingFace LLaMA rotate_half convention:
          // x_rot = [v1 * cos - v2 * sin, v2 * cos + v1 * sin]
          outData[headOffset + j] = v1 * c - v2 * sVal;
          outData[headOffset + j + half] = v2 * c + v1 * sVal;
        }
      }
    }

    return new Tensor(x.shape, outData);
  }
}

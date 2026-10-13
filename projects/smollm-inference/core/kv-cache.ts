/**
 * Key-Value Cache State Manager for Autoregressive Transformer Generation.
 * Manages pre-allocated contiguous memory per layer to avoid allocations during inference.
 */

import { Tensor } from './tensor.js';

export interface LayerKVCache {
  readonly keyCache: Tensor;
  readonly valueCache: Tensor;
  currentSeqLen: number;
}

export class KVCacheManager {
  public readonly numLayers: number;
  public readonly numKvHeads: number;
  public readonly headDim: number;
  public readonly maxSeqLen: number;
  public readonly kvDim: number;

  private readonly layers: LayerKVCache[];

  constructor(
    numLayers: number,
    numKvHeads: number,
    headDim: number,
    maxSeqLen = 2048
  ) {
    this.numLayers = numLayers;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.maxSeqLen = maxSeqLen;
    this.kvDim = numKvHeads * headDim;

    this.layers = new Array<LayerKVCache>(numLayers);

    for (let l = 0; l < numLayers; l++) {
      // Shape: [maxSeqLen, numKvHeads, headDim]
      const keyCache = Tensor.zeros([maxSeqLen, numKvHeads, headDim]);
      const valueCache = Tensor.zeros([maxSeqLen, numKvHeads, headDim]);
      this.layers[l] = {
        keyCache,
        valueCache,
        currentSeqLen: 0,
      };
    }
  }

  /**
   * Resets the active sequence length for all layers to 0 between generation runs.
   * Keeps the pre-allocated buffers intact for zero-allocation reuse.
   */
  public reset(): void {
    for (let l = 0; l < this.numLayers; l++) {
      this.layers[l].currentSeqLen = 0;
    }
  }

  /**
   * Get the current sequence length for a given layer.
   */
  public getSeqLen(layerIdx: number): number {
    return this.layers[layerIdx].currentSeqLen;
  }

  /**
   * Returns the underlying layer cache object.
   */
  public getLayer(layerIdx: number): LayerKVCache {
    return this.layers[layerIdx];
  }

  /**
   * Appends key and value tensors into the cache at currentSeqLen.
   * Keys and values can represent 1 token (decode) or S tokens (prefill).
   *
   * @param layerIdx Layer index
   * @param keys Tensor of shape [S, kvDim] or [S, numKvHeads, headDim]
   * @param values Tensor of shape [S, kvDim] or [S, numKvHeads, headDim]
   * @returns The updated active sequence length
   */
  public append(layerIdx: number, keys: Tensor, values: Tensor): number {
    const layer = this.layers[layerIdx];
    const incomingTokens = keys.size / this.kvDim;

    if (layer.currentSeqLen + incomingTokens > this.maxSeqLen) {
      throw new RangeError(
        `KV cache overflow: requested capacity ${layer.currentSeqLen + incomingTokens} exceeds maximum sequence length ${this.maxSeqLen}`
      );
    }

    const startOffset = layer.currentSeqLen * this.kvDim;
    const numElements = incomingTokens * this.kvDim;

    // Fast copy into contiguous cache buffer
    layer.keyCache.data.set(keys.data.subarray(0, numElements), startOffset);
    layer.valueCache.data.set(values.data.subarray(0, numElements), startOffset);

    layer.currentSeqLen += incomingTokens;
    return layer.currentSeqLen;
  }

  /**
   * Retrieves Key tensor slice containing all cached tokens up to currentSeqLen.
   */
  public getActiveKeys(layerIdx: number): Tensor {
    const layer = this.layers[layerIdx];
    const len = layer.currentSeqLen;
    return new Tensor(
      [len, this.numKvHeads, this.headDim],
      layer.keyCache.data.subarray(0, len * this.kvDim)
    );
  }

  /**
   * Retrieves Value tensor slice containing all cached tokens up to currentSeqLen.
   */
  public getActiveValues(layerIdx: number): Tensor {
    const layer = this.layers[layerIdx];
    const len = layer.currentSeqLen;
    return new Tensor(
      [len, this.numKvHeads, this.headDim],
      layer.valueCache.data.subarray(0, len * this.kvDim)
    );
  }
}

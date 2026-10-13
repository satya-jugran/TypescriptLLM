/**
 * Pure TypeScript Contiguous Tensor Abstraction.
 * Backed by Float32Array with cache-friendly layouts and optimized CPU math primitives.
 */

export class Tensor {
  public readonly shape: readonly number[];
  public readonly stride: readonly number[];
  public readonly data: Float32Array;
  public readonly size: number;
  public readonly ndim: number;

  constructor(shape: readonly number[], data?: Float32Array | number[]) {
    this.shape = Object.freeze([...shape]);
    this.ndim = this.shape.length;
    this.size = this.shape.reduce((acc, dim) => acc * dim, 1);
    this.stride = Object.freeze(Tensor.computeStrides(this.shape));

    if (data) {
      if (data instanceof Float32Array) {
        if (data.length !== this.size) {
          throw new Error(
            `Data length mismatch: expected ${this.size} elements, got ${data.length}`
          );
        }
        this.data = data;
      } else {
        if (data.length !== this.size) {
          throw new Error(
            `Array length mismatch: expected ${this.size} elements, got ${data.length}`
          );
        }
        this.data = new Float32Array(data);
      }
    } else {
      this.data = new Float32Array(this.size);
    }
  }

  /**
   * Helper to compute row-major strides for a given shape.
   */
  public static computeStrides(shape: readonly number[]): number[] {
    const ndim = shape.length;
    if (ndim === 0) return [];
    const strides = new Array<number>(ndim);
    strides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  // --- Factory Methods ---

  public static zeros(shape: readonly number[]): Tensor {
    return new Tensor(shape);
  }

  public static ones(shape: readonly number[]): Tensor {
    const t = new Tensor(shape);
    t.data.fill(1.0);
    return t;
  }

  public static fromArray(shape: readonly number[], data: Float32Array | number[]): Tensor {
    return new Tensor(shape, data);
  }

  // --- Memory & Access Utilities ---

  public clone(): Tensor {
    const copy = new Float32Array(this.size);
    copy.set(this.data);
    return new Tensor(this.shape, copy);
  }

  public fill(value: number): this {
    this.data.fill(value);
    return this;
  }

  public copyFrom(src: Tensor, srcOffset = 0, dstOffset = 0, count?: number): this {
    const elementsToCopy = count ?? Math.min(src.size - srcOffset, this.size - dstOffset);
    if (elementsToCopy <= 0) return this;
    const srcSlice = src.data.subarray(srcOffset, srcOffset + elementsToCopy);
    this.data.set(srcSlice, dstOffset);
    return this;
  }

  public get(...indices: number[]): number {
    if (indices.length !== this.ndim) {
      throw new Error(`Expected ${this.ndim} indices, got ${indices.length}`);
    }
    let offset = 0;
    for (let i = 0; i < this.ndim; i++) {
      const idx = indices[i];
      if (idx < 0 || idx >= this.shape[i]) {
        throw new RangeError(`Index ${idx} out of bounds for axis ${i} with size ${this.shape[i]}`);
      }
      offset += idx * this.stride[i];
    }
    return this.data[offset];
  }

  public set(value: number, ...indices: number[]): void {
    if (indices.length !== this.ndim) {
      throw new Error(`Expected ${this.ndim} indices, got ${indices.length}`);
    }
    let offset = 0;
    for (let i = 0; i < this.ndim; i++) {
      const idx = indices[i];
      if (idx < 0 || idx >= this.shape[i]) {
        throw new RangeError(`Index ${idx} out of bounds for axis ${i} with size ${this.shape[i]}`);
      }
      offset += idx * this.stride[i];
    }
    this.data[offset] = value;
  }

  public reshape(newShape: readonly number[]): Tensor {
    const newSize = newShape.reduce((acc, dim) => acc * dim, 1);
    if (newSize !== this.size) {
      throw new Error(
        `Cannot reshape tensor of size ${this.size} into shape [${newShape.join(',')}] of size ${newSize}`
      );
    }
    return new Tensor(newShape, this.data);
  }

  /**
   * Slice along the first dimension [start, end).
   */
  public slice(start: number, end: number): Tensor {
    if (this.ndim === 0) throw new Error('Cannot slice 0-dim tensor');
    const dim0 = this.shape[0];
    const actualStart = Math.max(0, Math.min(start, dim0));
    const actualEnd = Math.max(actualStart, Math.min(end, dim0));
    const slicedDim0 = actualEnd - actualStart;

    const rowSize = this.stride[0];
    const offset = actualStart * rowSize;
    const length = slicedDim0 * rowSize;
    const subData = this.data.subarray(offset, offset + length);

    const newShape = [slicedDim0, ...this.shape.slice(1)];
    return new Tensor(newShape, new Float32Array(subData));
  }

  // --- Mathematical Operations ---

  /**
   * Element-wise addition: this + other (supports broadcasting when other has 1D or same shape).
   */
  public add(other: Tensor): Tensor {
    const result = this.clone();
    result.addInPlace(other);
    return result;
  }

  public addInPlace(other: Tensor): this {
    const dst = this.data;
    const src = other.data;

    if (this.size === other.size) {
      for (let i = 0; i < this.size; i++) {
        dst[i] += src[i];
      }
    } else if (other.ndim === 1 && other.size === this.shape[this.ndim - 1]) {
      // Broadcast over the last dimension
      const lastDim = other.size;
      const numRows = this.size / lastDim;
      for (let r = 0; r < numRows; r++) {
        const offset = r * lastDim;
        for (let c = 0; c < lastDim; c++) {
          dst[offset + c] += src[c];
        }
      }
    } else {
      throw new Error(
        `Broadcasting not supported for shapes [${this.shape.join(',')}] and [${other.shape.join(',')}]`
      );
    }
    return this;
  }

  /**
   * Element-wise multiplication (Hadamard product) or scalar scaling.
   */
  public mul(other: Tensor | number): Tensor {
    const result = this.clone();
    result.mulInPlace(other);
    return result;
  }

  public mulInPlace(other: Tensor | number): this {
    const dst = this.data;
    if (typeof other === 'number') {
      const val = other;
      for (let i = 0; i < this.size; i++) {
        dst[i] *= val;
      }
      return this;
    }

    const src = other.data;
    if (this.size === other.size) {
      for (let i = 0; i < this.size; i++) {
        dst[i] *= src[i];
      }
    } else if (other.ndim === 1 && other.size === this.shape[this.ndim - 1]) {
      // Broadcast over last dimension
      const lastDim = other.size;
      const numRows = this.size / lastDim;
      for (let r = 0; r < numRows; r++) {
        const offset = r * lastDim;
        for (let c = 0; c < lastDim; c++) {
          dst[offset + c] *= src[c];
        }
      }
    } else {
      throw new Error(
        `Shape mismatch for multiplication: [${this.shape.join(',')}] and [${other.shape.join(',')}]`
      );
    }
    return this;
  }

  /**
   * Linear projection: Y = X * W^T (+ bias)
   * Where X is this [M, K], and W is [N, K] (standard Hugging Face weights format).
   * Result is [M, N].
   * Highly optimized CPU loop using continuous memory streams and loop unrolling.
   */
  public linear(weight: Tensor, bias?: Tensor): Tensor {
    const lastDimIdx = this.ndim - 1;
    const K = this.shape[lastDimIdx];
    const N = weight.shape[0];
    const wK = weight.shape[1];

    if (K !== wK) {
      throw new Error(
        `Linear dimension mismatch: input last dim ${K} != weight in_features ${wK}`
      );
    }

    const M = this.size / K;
    const outShape = this.ndim === 1 ? [N] : [...this.shape.slice(0, lastDimIdx), N];
    const outData = new Float32Array(M * N);

    const xData = this.data;
    const wData = weight.data;
    const bData = bias?.data;

    for (let m = 0; m < M; m++) {
      const xRowOffset = m * K;
      const yRowOffset = m * N;

      for (let n = 0; n < N; n++) {
        const wRowOffset = n * K;
        let sum = bData ? bData[n] : 0.0;

        let k = 0;
        const unrollLimit = K - 7;
        for (; k < unrollLimit; k += 8) {
          sum +=
            xData[xRowOffset + k] * wData[wRowOffset + k] +
            xData[xRowOffset + k + 1] * wData[wRowOffset + k + 1] +
            xData[xRowOffset + k + 2] * wData[wRowOffset + k + 2] +
            xData[xRowOffset + k + 3] * wData[wRowOffset + k + 3] +
            xData[xRowOffset + k + 4] * wData[wRowOffset + k + 4] +
            xData[xRowOffset + k + 5] * wData[wRowOffset + k + 5] +
            xData[xRowOffset + k + 6] * wData[wRowOffset + k + 6] +
            xData[xRowOffset + k + 7] * wData[wRowOffset + k + 7];
        }
        for (; k < K; k++) {
          sum += xData[xRowOffset + k] * wData[wRowOffset + k];
        }

        outData[yRowOffset + n] = sum;
      }
    }

    return new Tensor(outShape, outData);
  }

  /**
   * Standard 2D matrix multiplication: C = A * B
   * Where A is [M, K] and B is [K, N] -> [M, N]
   */
  public matmul(other: Tensor): Tensor {
    if (this.ndim < 2 || other.ndim < 2) {
      throw new Error(`Matmul requires tensors with >= 2 dimensions`);
    }

    const M = this.shape[this.ndim - 2];
    const K1 = this.shape[this.ndim - 1];
    const K2 = other.shape[other.ndim - 2];
    const N = other.shape[other.ndim - 1];

    if (K1 !== K2) {
      throw new Error(`Matmul inner dimension mismatch: ${K1} != ${K2}`);
    }

    const outShape = [...this.shape.slice(0, this.ndim - 2), M, N];
    const outData = new Float32Array(M * N);
    const aData = this.data;
    const bData = other.data;

    for (let i = 0; i < M; i++) {
      const aOffset = i * K1;
      const cOffset = i * N;
      for (let k = 0; k < K1; k++) {
        const aVal = aData[aOffset + k];
        const bOffset = k * N;
        for (let j = 0; j < N; j++) {
          outData[cOffset + j] += aVal * bData[bOffset + j];
        }
      }
    }

    return new Tensor(outShape, outData);
  }

  /**
   * SiLU (Swish-1) activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
   */
  public silu(): Tensor {
    const result = this.clone();
    result.siluInPlace();
    return result;
  }

  public siluInPlace(): this {
    const data = this.data;
    const len = this.size;
    for (let i = 0; i < len; i++) {
      const x = data[i];
      // Numerically stable sigmoid
      data[i] = x / (1.0 + Math.exp(-x));
    }
    return this;
  }

  /**
   * RMS Normalization: y = (x / sqrt(mean(x^2) + eps)) * weight
   * Evaluated across the last dimension.
   */
  public rmsNorm(weight: Tensor, eps: number): Tensor {
    const lastDim = this.shape[this.ndim - 1];
    if (weight.size !== lastDim) {
      throw new Error(`RMSNorm weight size ${weight.size} != tensor last dim ${lastDim}`);
    }

    const numRows = this.size / lastDim;
    const outData = new Float32Array(this.size);
    const inData = this.data;
    const wData = weight.data;

    for (let r = 0; r < numRows; r++) {
      const offset = r * lastDim;

      // 1. Calculate sum of squares
      let sumSq = 0.0;
      let i = 0;
      const limit = lastDim - 7;
      for (; i < limit; i += 8) {
        const v0 = inData[offset + i];
        const v1 = inData[offset + i + 1];
        const v2 = inData[offset + i + 2];
        const v3 = inData[offset + i + 3];
        const v4 = inData[offset + i + 4];
        const v5 = inData[offset + i + 5];
        const v6 = inData[offset + i + 6];
        const v7 = inData[offset + i + 7];
        sumSq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
      }
      for (; i < lastDim; i++) {
        const v = inData[offset + i];
        sumSq += v * v;
      }

      // 2. Normalization factor: 1 / sqrt(mean(x^2) + eps)
      const rms = 1.0 / Math.sqrt(sumSq / lastDim + eps);

      // 3. Scale and apply learned weight
      for (let j = 0; j < lastDim; j++) {
        outData[offset + j] = inData[offset + j] * rms * wData[j];
      }
    }

    return new Tensor(this.shape, outData);
  }

  /**
   * Numerically stable Softmax along the last dimension.
   */
  public softmax(): Tensor {
    const lastDim = this.shape[this.ndim - 1];
    const numRows = this.size / lastDim;
    const outData = new Float32Array(this.size);
    const inData = this.data;

    for (let r = 0; r < numRows; r++) {
      const offset = r * lastDim;

      // Find max
      let maxVal = -Infinity;
      for (let c = 0; c < lastDim; c++) {
        const val = inData[offset + c];
        if (val > maxVal) maxVal = val;
      }

      // Compute exp and sum
      let sumExp = 0.0;
      for (let c = 0; c < lastDim; c++) {
        const expVal = Math.exp(inData[offset + c] - maxVal);
        outData[offset + c] = expVal;
        sumExp += expVal;
      }

      // Normalize
      const invSum = 1.0 / sumExp;
      for (let c = 0; c < lastDim; c++) {
        outData[offset + c] *= invSum;
      }
    }

    return new Tensor(this.shape, outData);
  }

  /**
   * Returns index of the maximum value in the tensor (argmax).
   */
  public argmax(): number {
    let maxIdx = 0;
    let maxVal = this.data[0];
    const len = this.size;
    for (let i = 1; i < len; i++) {
      if (this.data[i] > maxVal) {
        maxVal = this.data[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }
}

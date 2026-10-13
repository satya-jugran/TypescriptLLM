/**
 * Tensor.ts — A tiny reverse-mode autograd Tensor.
 *
 * This is the heart of the engine. A `Tensor` wraps:
 *
 *   1. `data`  — a flat Float64Array plus a shape, holding the forward values.
 *   2. `grad`  — a flat Float64Array of the same size, holding accumulated
 *                ∂L/∂this (filled by `backward()`).
 *   3. `_parents` — edges to the tensors this node was computed from, each
 *      carrying a `gradFn` closure that knows how to push this node's
 *      output-gradient into that parent's gradient buffer.
 *
 * How the graph works: every operation creates a NEW Tensor and records
 * its parents. Calling `backward()` on the final scalar loss walks the
 * graph *topologically* (parents before children — i.e. the chain rule
 * applied in reverse order of construction) and routes gradients through
 * each edge exactly once. Same idea as micrograd, lifted to n-dimensional
 * tensors.
 *
 * Convention: we never mutate `data` in-place during a forward pass (each
 * op returns a fresh node) because stale references in the graph would
 * corrupt already-computed values. Learning happens by mutating `data` in
 * the optimizer step, which happens after the graph is consumed.
 */

/** Shape of an n-d array, row-major (C order). `[]` denotes a scalar. */
export type Shape = number[];

export function numel(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** Broadcast a pair of shapes following NumPy rules (dims align from the right). */
function broadcastShapes(a: Shape, b: Shape): Shape {
  const out: Shape = [];
  for (let k = 0; k < Math.max(a.length, b.length); k++) {
    const da = a.length - 1 - k >= 0 ? a[a.length - 1 - k] : 1;
    const db = b.length - 1 - k >= 0 ? b[b.length - 1 - k] : 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
    }
    out.unshift(da === 1 && db !== 1 ? db : da);
  }
  return out;
}

export class Tensor {
  readonly data: Float64Array;
  readonly shape: Shape;
  /** Gradient w.r.t. `data` (only meaningful between zeroGrad and backward). */
  readonly grad: Float64Array;
  private _requiresGrad: boolean;
  private _parents: Array<{ tensor: Tensor; gradFn: (gOut: Tensor) => void }> = [];

  constructor(data: Float64Array | number[], shape?: Shape, requiresGrad = false) {
    const arr = data instanceof Float64Array ? data : new Float64Array(data);
    this.data = arr;
    this.shape = shape ?? [arr.length];
    if (this.shape.length === 0 && arr.length !== 1) {
      throw new Error("a scalar shape [] must hold exactly one element");
    }
    if (numel(this.shape) !== arr.length) {
      throw new Error(`shape [${this.shape}] does not match ${arr.length} elements`);
    }
    this.grad = new Float64Array(arr.length);
    this._requiresGrad = requiresGrad;
  }

  // ---------------------------------------------------------------- factories

  static from(values: number[] | Float64Array, shape?: Shape, requiresGrad = false): Tensor {
    return new Tensor(values, shape, requiresGrad);
  }

  static scalar(v: number): Tensor {
    return new Tensor(new Float64Array([v]), [], false);
  }

  static zeros(shape: Shape): Tensor {
    return new Tensor(new Float64Array(numel(shape)), shape);
  }

  static ones(shape: Shape): Tensor {
    const d = new Float64Array(numel(shape)).fill(1);
    return new Tensor(d, shape);
  }

  /** Uniform randoms in [-a, a]. */
  static rand(shape: Shape, a = 0.1): Tensor {
    const d = new Float64Array(numel(shape));
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * a;
    return new Tensor(d, shape);
  }

  /** Normal(0, std) via Box-Muller. Used for weight init (std ≈ 0.02, GPT style). */
  static randn(shape: Shape, std = 0.02): Tensor {
    const d = new Float64Array(numel(shape));
    for (let i = 0; i < d.length; i++) {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      d[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
    }
    return new Tensor(d, shape);
  }

  // ---------------------------------------------------------------- plumbing

  get size(): number { return this.data.length; }
  get requiresGrad(): boolean { return this._requiresGrad; }
  get parents(): ReadonlyArray<{ tensor: Tensor; gradFn: (gOut: Tensor) => void }> {
    return this._parents;
  }

  /**
   * Register a gradient edge: when `this` node backward-propagates, run
   * `gradFn` with this node as the argument so it can scatter ∂L/∂this into
   * `parent.grad`. Used by custom ops (embeddings, cross-entropy) that build
   * their own backward rules outside the standard op library.
   */
  attachParent(parent: Tensor, gradFn: (gOut: Tensor) => void): void {
    this._parents.push({ tensor: parent, gradFn });
  }

  /** Mark this tensor as gradient-tracking (leaves / parameters). */
  setRequiresGrad(value = true): this {
    this._requiresGrad = value;
    return this;
  }

  /** The single number inside a scalar-shaped tensor. */
  item(): number {
    if (this.data.length !== 1) throw new Error(`item() on a ${this.data.length}-element tensor`);
    return this.data[0];
  }

  /** Fill `this.grad` with zeros (call before each backward pass). */
  zeroGrad(): void { this.grad.fill(0); }

  /** Copy of the data, detached from any graph (for inference / logging). */
  detach(): Tensor {
    return new Tensor(new Float64Array(this.data), [...this.shape], false);
  }

  // ---------------------------------------------------------------- unary ops

  neg(): Tensor { return this.mul(Tensor.scalar(-1)); }
  sub(other: Tensor): Tensor { return this.add(other.neg()); }

  add(other: Tensor): Tensor {
    const outShape = broadcastShapes(this.shape, other.shape);
    const a = this.data, b = other.data;
    const out = new Float64Array(numel(outShape));
    for (let i = 0; i < out.length; i++) out[i] = a[i % a.length] + b[i % b.length];
    const outT = new Tensor(out, outShape, this._requiresGrad || other._requiresGrad);
    if (this._requiresGrad) {
      // Broadcasting Ω means source index ≡ out index mod source size; the
      // reverse op "sums the gradient over the broadcast copies" for free.
      outT._parents.push({ tensor: this, gradFn: (g) => {
        for (let i = 0; i < g.size; i++) this.grad[i % a.length] += g.grad[i];
      }});
    }
    if (other._requiresGrad) {
      outT._parents.push({ tensor: other, gradFn: (g) => {
        for (let i = 0; i < g.size; i++) other.grad[i % b.length] += g.grad[i];
      }});
    }
    return outT;
  }

  mul(other: Tensor): Tensor {
    const outShape = broadcastShapes(this.shape, other.shape);
    const a = this.data, b = other.data;
    const out = new Float64Array(numel(outShape));
    for (let i = 0; i < out.length; i++) out[i] = a[i % a.length] * b[i % b.length];
    const outT = new Tensor(out, outShape, this._requiresGrad || other._requiresGrad);
    if (this._requiresGrad) {
      outT._parents.push({ tensor: this, gradFn: (g) => {
        for (let i = 0; i < g.size; i++)
          this.grad[i % a.length] += g.grad[i] * b[i % b.length];
      }});
    }
    if (other._requiresGrad) {
      outT._parents.push({ tensor: other, gradFn: (g) => {
        for (let i = 0; i < g.size; i++)
          other.grad[i % b.length] += g.grad[i] * a[i % a.length];
      }});
    }
    return outT;
  }

  div(other: Tensor): Tensor {
    const outShape = broadcastShapes(this.shape, other.shape);
    const a = this.data, b = other.data;
    const out = new Float64Array(numel(outShape));
    for (let i = 0; i < out.length; i++) out[i] = a[i % a.length] / b[i % b.length];
    const outT = new Tensor(out, outShape, this._requiresGrad || other._requiresGrad);
    if (this._requiresGrad) {
      outT._parents.push({ tensor: this, gradFn: (g) => {
        for (let i = 0; i < g.size; i++)
          this.grad[i % a.length] += g.grad[i] / b[i % b.length];
      }});
    }
    if (other._requiresGrad) {
      // d(a/b)/db = -a / b²  (chain rule on the denominator).
      outT._parents.push({ tensor: other, gradFn: (g) => {
        for (let i = 0; i < g.size; i++) {
          const bi = i % b.length;
          const ai = i % a.length;
          other.grad[bi] += -g.grad[i] * a[ai] / (b[bi] * b[bi]);
        }
      }});
    }
    return outT;
  }

  /** Elementwise ReLU: max(0, x). d/dx = 1{x > 0}. */
  relu(): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < out.length; i++) out[i] = Math.max(0, this.data[i]);
    const outT = new Tensor(out, [...this.shape], this._requiresGrad);
    if (this._requiresGrad) {
      outT._parents.push({ tensor: this, gradFn: (g) => {
        for (let i = 0; i < g.size; i++)
          if (this.data[i] > 0) this.grad[i] += g.grad[i];
      }});
    }
    return outT;
  }

  /**
   * GELU, tanh approximation (as in GPT-2 / BERT):
   *
   *   gelu(x) = 0.5·x·(1 + tanh(u)),  u = √(2/π)·(x + 0.044715·x³)
   *
   * Why GELU over ReLU? It is smooth at 0 and its small negative tail lets
   * gradients keep flowing for slightly-negative inputs, which empirically
   * helps small transformers. We cache `u` so the backward pass can reuse it.
   */
  gelu(): Tensor {
    const C = Math.sqrt(2 / Math.PI);
    const out = new Float64Array(this.size);
    const u = new Float64Array(this.size); // tanh argument, cached for backward
    for (let i = 0; i < out.length; i++) {
      const x = this.data[i];
      u[i] = C * (x + 0.044715 * x * x * x);
      out[i] = 0.5 * x * (1 + Math.tanh(u[i]));
    }
    const outT = new Tensor(out, [...this.shape], this._requiresGrad);
    if (this._requiresGrad) {
      // Product rule on 0.5·x·(1 + tanh(u)):
      //   d/dx = 0.5·(1 + t) + 0.5·x·(1 − t²)·du/dx,  du/dx = C·(1 + 3·0.044715·x²)
      outT._parents.push({ tensor: this, gradFn: (g) => {
        for (let i = 0; i < g.size; i++) {
          const x = this.data[i];
          const t = Math.tanh(u[i]);
          const dTanh = (1 - t * t) * C * (1 + 3 * 0.044715 * x * x);
          this.grad[i] += g.grad[i] * (0.5 * (1 + t) + 0.5 * x * dTanh);
        }
      }});
    }
    return outT;
  }

  /** Elementwise exponential. d/dx e^x = e^x (so gradient uses the *output*). */
  exp(): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < out.length; i++) out[i] = Math.exp(this.data[i]);
    const outT = new Tensor(out, [...this.shape], this._requiresGrad);
    if (this._requiresGrad) {
      outT._parents.push({ tensor: this, gradFn: (g) => {
        for (let i = 0; i < g.data.length; i++) this.grad[i] += g.data[i] * out[i];
      }});
    }
    return outT;
  }

  /** Elementwise natural log. d/dx ln x = 1/x. */
  log(): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < out.length; i++) out[i] = Math.log(this.data[i]);
    const outT = new Tensor(out, [...this.shape], this._requiresGrad);
    if (this._requiresGrad) {
      outT._parents.push({ tensor: this, gradFn: (g) => {
        for (let i = 0; i < g.size; i++) this.grad[i] += g.grad[i] / this.data[i];
      }});
    }
    return outT;
  }

  /** Sum over ALL dims → scalar Tensor. d(sum)/dx_i = 1 for every i. */
  sum(): Tensor {
    let s = 0;
    for (let i = 0; i < this.size; i++) s += this.data[i];
    const out = new Tensor(new Float64Array([s]), [], this._requiresGrad);
    if (this._requiresGrad) {
      out._parents.push({ tensor: this, gradFn: (g) => {
        const gd = g.grad[0]; // incoming ∂L/∂(this sum), seeded to 1 by backward()
        for (let i = 0; i < this.size; i++) this.grad[i] += gd;
      }});
    }
    return out;
  }

  /** Mean of all elements → scalar Tensor. */
  mean(): Tensor { return this.sum().div(Tensor.scalar(this.size)); }

  // ---------------------------------------------------------------- shape ops

  /** Reshape (row-major); layout is preserved so the backward is a plain add. */
  reshape(shape: Shape): Tensor {
    if (numel(shape) !== this.size) throw new Error(`reshape ${this.size} -> [${shape}]`);
    const out = new Tensor(this.data, [...shape], this._requiresGrad);
    out._parents.push({ tensor: this, gradFn: (g) => {
      for (let i = 0; i < g.size; i++) this.grad[i] += g.grad[i];
    }});
    return out;
  }

  /** Transpose the last two dims of a ≥2-d tensor (its own inverse). */
  transposeLast2(): Tensor {
    const shape = this.shape;
    if (shape.length < 2) throw new Error("transposeLast2 needs ndim >= 2");
    const rows = shape[shape.length - 2];
    const cols = shape[shape.length - 1];
    const batch = numel(shape.slice(0, -2));
    const outData = new Float64Array(this.size);
    for (let b = 0; b < batch; b++) {
      const off = b * rows * cols;
      for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
          outData[off + j * rows + i] = this.data[off + i * cols + j];
    }
    const out = new Tensor(outData, [...shape.slice(0, -2), cols, rows], this._requiresGrad);
    out._parents.push({ tensor: this, gradFn: (g) => {
      // Transpose is an involution: transposing the incoming *gradient*
      // (not the values!) recovers ∂L/∂this.
      const rows2 = g.shape[g.shape.length - 2];
      const cols2 = g.shape[g.shape.length - 1];
      const batch2 = numel(g.shape.slice(0, -2));
      for (let b = 0; b < batch2; b++) {
        const off = b * rows2 * cols2;
        for (let i = 0; i < rows2; i++)
          for (let j = 0; j < cols2; j++)
            this.grad[off + i * cols2 + j] += g.grad[off + j * rows2 + i];
      }
    }});
    return out;
  }

  // ---------------------------------------------------------------- matmul

  /**
   * Batched matrix multiplication over the last two dims.
   * A: [BA…, m, k] @ B: [BB…, k, n] → [BC…, m, n], where BC is the NumPy
   * broadcast of BA and BB.
   *
   * Note the gradient formulas (chain rule on C = A·B):
   *   ∂L/∂A[i,k] = Σ_j ∂L/∂C[i,j] · B[k,j]   (gradient @ B.transpose)
   *   ∂L/∂B[k,j] = Σ_i A[i,k]      · ∂L/∂C[i,j]   (A.transpose @ gradient)
   */
  matmul(other: Tensor): Tensor {
    const a = this, b = other;
    if (a.shape.length < 2 || b.shape.length < 2)
      throw new Error("matmul needs tensors with ndim >= 2");
    const k = a.shape[a.shape.length - 1];
    if (k !== b.shape[b.shape.length - 2])
      throw new Error(`matmul inner dims mismatch: [${a.shape}] @ [${b.shape}]`);
    const m = a.shape[a.shape.length - 2];
    const n = b.shape[b.shape.length - 1];

    const batchShape = broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2));
    const batch = numel(batchShape);
    const aBatch = numel(a.shape.slice(0, -2)) || 1;
    const bBatch = numel(b.shape.slice(0, -2)) || 1;
    // A batch of 1 broadcasts across the whole batch; otherwise dims are equal.
    if (aBatch !== batch && aBatch !== 1) throw new Error(`matmul batch dims [${a.shape}] incompatible with [${b.shape}]`);
    if (bBatch !== batch && bBatch !== 1) throw new Error(`matmul batch dims [${a.shape}] incompatible with [${b.shape}]`);

    const outData = new Float64Array(batch * m * n);
    const ai = a.data, bi = b.data;
    const aMat = m * k, bMat = k * n;
    const aStep = aBatch === batch ? aMat : 0;
    const bStep = bBatch === batch ? bMat : 0;
    for (let bt = 0; bt < batch; bt++) {
      const aOff = bt * aStep, bOff = bt * bStep, oOff = bt * m * n;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let acc = 0;
          for (let t = 0; t < k; t++) acc += ai[aOff + i * k + t] * bi[bOff + t * n + j];
          outData[oOff + i * n + j] = acc;
        }
      }
    }
    const outShape: Shape = [...batchShape, m, n];
    // Degenerate case: no batch dims at all (plain [m,k] @ [k,n]).
    if (a.shape.length === 2 && b.shape.length === 2) outShape.length = 0;
    if (batch === 1 && aBatch === 1 && bBatch === 1) {
      outShape.length = 0;
      outShape.push(m, n);
    }
    const outT = new Tensor(outData, outShape, a._requiresGrad || b._requiresGrad);

    if (a._requiresGrad) {
      // ∂L/∂A = gradient · Bᵀ  (per batch):
      //   dA[bt,i,t] += g[bt,i,j] · B[bt,t,j]
      outT._parents.push({ tensor: a, gradFn: (g) => {
        for (let bt = 0; bt < batch; bt++) {
          const aOff = bt * aStep, bOff = bt * bStep, oOff = bt * m * n;
          for (let i = 0; i < m; i++)
            for (let j = 0; j < n; j++) {
              const gv = g.grad[oOff + i * n + j];
              if (gv === 0) continue; // skip pulling in a whole zero row
              for (let t = 0; t < k; t++) a.grad[aOff + i * k + t] += gv * bi[bOff + t * n + j];
            }
        }
      }});
    }
    if (b._requiresGrad) {
      // ∂L/∂B = Aᵀ · gradient  (per batch):
      //   dB[bt,t,j] += A[bt,i,t] · g[bt,i,j]
      outT._parents.push({ tensor: b, gradFn: (g) => {
        for (let bt = 0; bt < batch; bt++) {
          const aOff = bt * aStep, bOff = bt * bStep, oOff = bt * m * n;
          for (let i = 0; i < m; i++)
            for (let j = 0; j < n; j++) {
              const gv = g.grad[oOff + i * n + j];
              if (gv === 0) continue;
              for (let t = 0; t < k; t++) b.grad[bOff + t * n + j] += gv * ai[aOff + i * k + t];
            }
        }
      }});
    }
    return outT;
  }

  /**
   * Row-wise softmax, softmax(x)_i = e^{x_i} / Σ_j e^{x_j}, over the LAST dim.
   *
   * Numerical stability: we subtract the row max before exponentiating.
   * Shifting x by any constant leaves softmax unchanged (the shift cancels),
   * but prevents `exp` from overflowing on large logits.
   *
   * The backward uses the classic softmax Jacobian-vector product:
   *   ∂L/∂x_i = p_i·(g_i − Σ_j p_j·g_j)
   * which is exactly what you get by multiplying the full Jacobian
   * diag(p) − p·pᵀ by the incoming gradient g.
   */
  softmaxRows(): Tensor {
    const cols = this.shape[this.shape.length - 1];
    const rows = this.size / cols;
    const outData = new Float64Array(this.size);
    for (let r = 0; r < rows; r++) {
      let mx = -Infinity;
      for (let c = 0; c < cols; c++) mx = Math.max(mx, this.data[r * cols + c]);
      let s = 0;
      for (let c = 0; c < cols; c++) {
        const e = Math.exp(this.data[r * cols + c] - mx);
        outData[r * cols + c] = e;
        s += e;
      }
      for (let c = 0; c < cols; c++) outData[r * cols + c] /= s;
    }
    const outT = new Tensor(outData, [...this.shape], this._requiresGrad);
    if (this._requiresGrad) {
      outT._parents.push({ tensor: this, gradFn: (g) => {
        const p = outData;
        for (let r = 0; r < rows; r++) {
          let dot = 0;
          for (let c = 0; c < cols; c++) dot += p[r * cols + c] * g.grad[r * cols + c];
          for (let c = 0; c < cols; c++)
            this.grad[r * cols + c] += p[r * cols + c] * (g.grad[r * cols + c] - dot);
        }
      }});
    }
    return outT;
  }

  // ---------------------------------------------------------------- backward

  /**
   * Reverse-mode backward pass. `this` must be the scalar loss (or any scalar
   * seed). Implementation: topologically sort the graph so every parent is
   * processed before its children, seed ∂L/∂L = 1, then release the gradient
   * along each recorded edge — i.e., the chain rule, batched over the tape.
   */
  backward(): void {
    if (this.size !== 1) throw new Error("backward() must be called on a scalar tensor");
    this.grad.fill(0);
    this.grad[0] = 1;

    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const visit = (t: Tensor) => {
      if (visited.has(t)) return;
      visited.add(t);
      for (const p of t._parents) visit(p.tensor);
      topo.push(t);
    };
    visit(this);

    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      // gradFn accumulates ∂L/t → ∂L/parent directly into parent.grad.
      for (const { gradFn } of t._parents) gradFn(t);
      // Drop consumed edges so graphs reused across steps don't grow forever.
      // (Each parent's own outgoing edges are untouched until IT is processed.)
      t._parents.length = 0;
    }
  }
}

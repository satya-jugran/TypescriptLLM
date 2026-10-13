/**
 * Module.ts — Abstract base class for every neural-network building block
 * (Keras/PyTorch-style `Module`, minus the framework).
 *
 * A Module owns zero or more `Parameter`s (learnable Tensors) and possibly
 * child Modules. The two contracts every subclass must honor:
 *
 *   1. `parameters()` — return the learnable tensors, in a STABLE order
 *      (the trainer iterates them together with their gradients; a
 *      reordering would silently pair one gradient with the wrong tensor).
 *   2. `forward(...)` — pure graph construction: consume input Tensors,
 *      return an output Tensor. On top of that, `zeroGrad()` walks the
 *      parameter list and clears each `.grad` buffer, which is what lets
 *      the Trainer reuse parameter tensors across optimizer steps.
 *
 * Composition over inheritance: a Module that contains other Modules
 * assembles them in its constructor (exactly like PyTorch `nn.Sequential`),
 * and its `parameters()` is the concatenation of the children's in
 * declaration order.
 */

import type { Tensor } from "../autograd/Tensor";

/** A learnable tensor. Same as Tensor but semantically marked as a weight. */
export type Parameter = Tensor & { readonly isParameter: true };

/** Wrap a Tensor into a Parameter (adds `isParameter: true` for tooling). */
export function param(tensor: Tensor): Parameter {
  return Object.assign(tensor, { isParameter: true as const });
}

export abstract class Module {
  /** Learnable tensors owned directly by this module (not by children). */
  protected ownParameters: Parameter[] = [];
  /** Child modules whose parameters are delegated to. */
  protected children: Module[] = [];

  /**
   * All learnable parameters of this module, in a stable order:
   * own parameters first, then each child's parameters in turn.
   */
  parameters(): Parameter[] {
    const out: Parameter[] = [...this.ownParameters];
    for (const child of this.children) out.push(...child.parameters());
    return out;
  }

  /** Register a learnable tensor so `parameters()` picks it up. */
  protected register(p: Parameter): Parameter {
    this.ownParameters.push(p);
    return p;
  }

  /** Register a child module so its parameters are inherited. */
  protected add(child: Module): void {
    this.children.push(child);
  }

  /** Zero the gradient buffers of every parameter (call before backprop). */
  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }

  /** Number of learnable scalars tracked by this module (for logging). */
  numParams(): number {
    return this.parameters().reduce((acc, p) => acc + p.size, 0);
  }

  /** Print a tree of modules/parameters, PyTorch-style. Handy in demos. */
  describe(indent = ""): string {
    const lines: string[] = [];
    const name = this.constructor.name;
    let n = 0;
    for (const p of this.ownParameters) n += p.size;
    lines.push(`${indent}${name}${n ? ` (${n.toLocaleString()} own params)` : ""}`);
    for (const c of this.children) lines.push(...c.describe(indent + "  ").split("\n"));
    return lines.join("\n");
  }

  /**
   * Forward pass. Typed loosely (variadic Tensor) so embeddings may accept
   * index arrays; concrete subclasses narrow the signature.
   */
  abstract forward(...inputs: any[]): Tensor;
}

/** Put a tensor into training mode (identity here — no dropout — kept for API parity). */
export function train(): boolean { return true; }

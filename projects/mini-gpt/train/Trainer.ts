/**
 * Trainer.ts — Orchestrates the training loop: random patching of the text
 * into (context, target) pairs, forward/backward/step, loss tracking,
 * periodic validation and sampling.
 *
 * BATCHING STRATEGY (nanoGPT-style): pick a random offset in the corpus,
 * then read B consecutive stretches of T+1 characters. For each stretch,
 * the input is chars[t .. t+T) and the target is chars[t+1 .. t+T+1) —
 * the SAME text shifted by one. That single shift is the whole supervised
 * task: "given these characters so far, predict the next one."
 *
 * Every training step therefore:
 *   1. samples a fresh batch          (fresh gradients, no repeats)
 *   2. forward  → logits & loss      (builds the autograd graph)
 *   3. backward → ∂loss/∂θ           (releases the graph, fills .grad)
 *   4. optimizer.step()               (updates θ in place)
 *   5. zeroGrad                       (prepares the next accumulation)
 */

import { Tensor } from "../autograd/Tensor";
import type { GPT } from "../nn/GPT";
import type { CharTokenizer } from "../data/CharTokenizer";
import { CrossEntropyLoss } from "./Loss";
import type { AdamW } from "../optim/AdamW";

export interface TrainerConfig {
  /** Sequences per batch. */
  batchSize: number;
  /** Context length fed to the model (≤ blockSize). */
  seqLen: number;
  /** Eval every `evalEvery` steps (0 to disable). */
  evalEvery?: number;
  /** Validation split fraction (held-out tail of the corpus). */
  valFraction?: number;
  /** Progress print every N steps (0 to disable). */
  logEvery?: number;
  /** Optional callback invoked with (step, loss) at each logged step. */
  onLog?: (step: number, trainLoss: number, valLoss: number | null) => void;
}

export class Trainer {
  readonly model: GPT;
  readonly tokenizer: CharTokenizer;
  readonly optimizer: AdamW;
  readonly lossFn: CrossEntropyLoss;
  readonly config: TrainerConfig;

  private tokens: number[]; // full corpus ids
  private trainEnd: number; // index where validation data begins
  private losses: number[] = [];
  private step = 0;

  constructor(
    model: GPT,
    tokenizer: CharTokenizer,
    optimizer: AdamW,
    config: TrainerConfig,
  ) {
    if (config.seqLen > (model.config.blockSize)) {
      throw new Error(`seqLen (${config.seqLen}) must be ≤ blockSize (${model.config.blockSize})`);
    }
    this.model = model;
    this.tokenizer = tokenizer;
    this.optimizer = optimizer;
    this.lossFn = new CrossEntropyLoss();
    this.config = config;
    const corpus = tokenizer.corpus;
    this.tokens = tokenizer.encode(corpus);
    const valFrac = config.valFraction ?? 0.1;
    this.trainEnd = Math.max(1, Math.floor(this.tokens.length * (1 - valFrac)));
  }

  /** Loss history, one entry per training step. */
  get history(): readonly number[] { return this.losses; }

  /** Sample one random batch: inputs [B,T] and next-char targets [B,T]. */
  private getBatch(): { inputs: number[]; targets: number[] } {
    const B = this.config.batchSize;
    const T = this.config.seqLen;
    const inputs: number[] = [];
    const targets: number[] = [];
    for (let b = 0; b < B; b++) {
      // Pick the start of a random T+1-long window from the train split.
      const maxStart = this.trainEnd - (T + 1);
      const start = Math.floor(Math.random() * (maxStart + 1));
      for (let t = 0; t < T; t++) {
        inputs.push(this.tokens[start + t]);
        targets.push(this.tokens[start + t + 1]); // shifted by one = the label
      }
    }
    return { inputs, targets };
  }

  /** Compute loss on a fixed (deterministic) validation batch. */
  evaluate(iters = 4): number {
    const savedTrainEnd = this.trainEnd;
    // Evaluate strictly on the held-out tail of the corpus.
    const total = this.tokens.length;
    this.trainEnd = total - Math.max(1, Math.floor(total * (this.config.valFraction ?? 0.1)));
    let sum = 0;
    for (let i = 0; i < iters; i++) {
      const { inputs, targets } = this.sampleValBatch();
      const logits = this.model.forward(inputs, this.config.batchSize, this.config.seqLen);
      // Flatten [B, T, V] → [B*T, V] for the loss (row-major reshape).
      const [V] = [logits.shape[logits.shape.length - 1]];
      const flat = logits.reshape([logits.size / V, V]);
      const loss = this.lossFn.forward(flat, targets);
      sum += loss.item();
    }
    this.trainEnd = savedTrainEnd;
    return sum / iters;
  }

  private sampleValBatch(): { inputs: number[]; targets: number[] } {
    const B = this.config.batchSize;
    const T = this.config.seqLen;
    const valStart = this.trainEnd;
    const valLen = this.tokens.length - valStart;
    const maxStart = valLen - (T + 1);
    if (maxStart < 0) throw new Error("validation split too small for seqLen");
    const inputs: number[] = [];
    const targets: number[] = [];
    for (let b = 0; b < B; b++) {
      const start = valStart + Math.floor(Math.random() * (maxStart + 1));
      for (let t = 0; t < T; t++) {
        inputs.push(this.tokens[start + t]);
        targets.push(this.tokens[start + t + 1]);
      }
    }
    return { inputs, targets };
  }

  /** One training step: sample batch → forward → backward → update → zero. */
  trainStep(): number {
    const { inputs, targets } = this.getBatch();
    const B = this.config.batchSize, T = this.config.seqLen;
    const logits = this.model.forward(inputs, B, T); // [B, T, V]
    const [V] = [logits.shape[logits.shape.length - 1]];
    const flat = logits.reshape([B * T, V]); // group the vocab axis for the loss
    const loss = this.lossFn.forward(flat, targets);

    this.optimizer.zeroGrad();
    loss.backward();
    this.optimizer.step();

    const l = loss.item();
    this.losses.push(l);
    this.step++;
    return l;
  }

  /** Full training run with periodic logging/validation. */
  train(iters: number): number[] {
    const logEvery = this.config.logEvery ?? 50;
    const evalEvery = this.config.evalEvery ?? 100;
    // Warm-up: never report a loss before at least one step happened.
    for (let i = 1; i <= iters; i++) {
      const loss = this.trainStep();
      this.config.onLog?.(this.step, loss, null);
      if (logEvery > 0 && i % logEvery === 0) {
        console.log(`step ${String(this.step).padStart(4)} | loss ${loss.toFixed(4)}`);
      }
      if (evalEvery > 0 && i % evalEvery === 0 && i < iters) {
        const val = this.evaluate();
        console.log(`step ${String(this.step).padStart(4)} | val loss ${val.toFixed(4)}`);
      }
    }
    return this.losses;
  }

  /** Sample characters from the current model given a prompt string. */
  generate(prompt: string, maxNewTokens: number, temperature = 1.0): string {
    const ids = this.tokenizer.encode(prompt);
    const out = this.model.generate(ids, maxNewTokens, temperature);
    return this.tokenizer.decode(out);
  }
}

/** Re-export so callers don't need to import Tensor for reshape plumbing. */
export { Tensor };

/**
 * test.ts — Numerical gradient checks + end-to-end training sanity.
 *
 * The gold standard for validating an autograd engine: compare the
 * analytic gradient (computed by our backward pass) against the *finite
 * difference* approximation of the derivative, which needs no graph at all:
 *
 *   f'(x) ≈ (f(x+h) - f(x-h)) / (2h)      (central difference, O(h²) error)
 *
 * Run:   npm test   (tsx src/autograd/test.ts)
 */

import { Tensor } from "./Tensor";
import { applyCausalMask, crossEntropyPerExample } from "./Operations";
import { GPT } from "../nn/GPT";
import { CharTokenizer } from "../data/CharTokenizer";
import { AdamW } from "../optim/AdamW";

let failures = 0;
let checks = 0;

function checkClose(name: string, a: number, b: number, tol: number, detail: string): boolean {
  const err = Math.abs(a - b);
  const ok = err <= tol;
  checks++;
  if (!ok) {
    failures++;
    console.error(`FAIL ${name}: got ${a.toExponential(3)}, expected ${b.toExponential(3)} (|Δ|=${err.toExponential(3)} > ${tol.toExponential(3)}) — ${detail}`);
  }
  return ok;
}

function gradCheck(
  name: string,
  build: (requiresGrad: boolean) => { loss: Tensor | number; params: { t: Tensor; values: number[] }[] },
): void {
  // Build a fresh graph with gradients enabled.
  const withGrad = build(true);
  const lossT = withGrad.loss instanceof Tensor ? withGrad.loss : Tensor.scalar(withGrad.loss);
  if (lossT.size !== 1) throw new Error(`gradCheck ${name}: loss must be scalar`);
  for (const p of withGrad.params) p.t.zeroGrad();
  lossT.backward();

  const H = 1e-6;
  for (const p of withGrad.params) {
    for (let i = 0; i < p.values.length; i++) {
      const analytic = p.t.grad[i];
      const orig = p.values[i];
      // f(x + h)
      p.values[i] = orig + H;
      const rebuilt1 = build(false);
      const fPlus = rebuilt1.loss instanceof Tensor ? rebuilt1.loss.item() : rebuilt1.loss;
      // f(x - h)
      p.values[i] = orig - H;
      const rebuilt2 = build(false);
      const fMinus = rebuilt2.loss instanceof Tensor ? rebuilt2.loss.item() : rebuilt2.loss;
      p.values[i] = orig;
      const numeric = (fPlus - fMinus) / (2 * H);
      checkClose(`${name}[${i}]`, analytic, numeric, 1e-4, "analytic vs central difference");
    }
  }
  console.log(`ok  ${name}`);
}

// ---------------------------------------------------------------------------
// 1. Elementwise ops: f = Σ (a+b)·c / d with ReLU and GELU sprinkled in.
// ---------------------------------------------------------------------------
{
  const aVals = [0.5, -1.2, 2.0];
  const bVals = [0.3, 0.8, -0.4];
  const cVals = [1.1, -2.2, 0.7];
  const dVals = [2.0, 3.0, 4.0];
  gradCheck("add/mul/div", (rg) => {
    const a = new Tensor(aVals, [3], rg);
    const b = new Tensor(bVals, [3], rg);
    const c = new Tensor(cVals, [3], rg);
    const d = new Tensor(dVals, [3], rg);
    const loss = a.add(b).mul(c).div(d).sum();
    return { loss, params: [
      { t: a, values: aVals },
      { t: b, values: bVals },
      { t: c, values: cVals },
      { t: d, values: dVals },
    ]};
  });
}
{
  const aVals = [0.7, -0.3, 1.5];
  const bVals = [2.0, 1.0, 0.5];
  gradCheck("gelu + relu", (rg) => {
    const a = new Tensor(aVals, [3], rg);
    const b = new Tensor(bVals, [3], rg);
    const loss = a.gelu().mul(b).relu().sum();
    return { loss, params: [{ t: a, values: aVals }] };
  });
}
// ---------------------------------------------------------------------------
// 2. Softmax, log, mean.
// ---------------------------------------------------------------------------
{
  const xVals = [1.0, 2.0, 3.0];
  gradCheck("softmax → log → weighted sum", (rg) => {
    const x = new Tensor(xVals, [1, 3], rg);
    const l = x.softmaxRows().log();
    const w = new Tensor([0.2, 0.5, 0.3], [1, 3]);
    return { loss: l.mul(w).sum(), params: [{ t: x, values: xVals }] };
  });
}
// ---------------------------------------------------------------------------
// 3. Matmul chain (2-D).
// ---------------------------------------------------------------------------
{
  const AVals = [0.5, -0.3, 0.8, 1.1]; // [2, 2]
  const BVals = [0.2, 1.0, -0.4, 0.6]; // [2, 2]
  gradCheck("matmul 2x2 chain", (rg) => {
    const A = new Tensor(AVals, [2, 2], rg);
    const B = new Tensor(BVals, [2, 2], rg);
    const loss = A.matmul(B).relu().sum();
    return { loss, params: [
      { t: A, values: AVals },
      { t: B, values: BVals },
    ]};
  });
}
// ---------------------------------------------------------------------------
// 4. Causal mask + softmax path (attention core).
// ---------------------------------------------------------------------------
{
  const SVals = [
    0.1, 0.8, 0.3,
    0.4, 0.2, 0.9,
    0.5, 0.7, 0.6,
  ]; // [3, 3]
  gradCheck("causal mask + softmax", (rg) => {
    const S = new Tensor(SVals, [3, 3], rg);
    const out = applyCausalMask(S).softmaxRows();
    return { loss: out.sum(), params: [{ t: S, values: SVals }] };
  });
}
// ---------------------------------------------------------------------------
// 5. Cross-entropy loss.
// ---------------------------------------------------------------------------
{
  const LVals = [0.1, 2.0, 0.5, 1.0, 0.3, 0.2]; // [2, 3]
  const targets = [1, 0];
  gradCheck("cross entropy", (rg) => {
    const L = new Tensor(LVals, [2, 3], rg);
    return { loss: crossEntropyPerExample(L, targets).mean(), params: [{ t: L, values: LVals }] };
  });
}
// ---------------------------------------------------------------------------
// 6. A tiny end-to-end GPT forward/backward with a scalar loss.
// ---------------------------------------------------------------------------
{
  const tokenizer = new CharTokenizer("abc");
  const cfg = { vocabSize: 3, blockSize: 4, nEmbd: 6, nHead: 2, nLayer: 1 };
  function buildGPT() {
    // Build a model, then extract copies of params so finite differences can
    // nudge them in-place (parameters are plain Tensors).
    const model = new GPT(cfg);
    const tokens = [0, 1, 2, 0];
    const targets = [1, 2, 0, 1];
    return () => {
      const logits = model.forward(tokens, 1, 4);
      const V = logits.shape[logits.shape.length - 1];
      const flat = logits.reshape([logits.size / V, V]);
      return crossEntropyPerExample(flat, targets).mean();
    };
  }
  const model = new GPT(cfg);
  const tokens = [0, 1, 2, 0];
  const targets = [1, 2, 0, 1];
  const params = model.parameters();

  // Analytic pass.
  for (const p of params) p.zeroGrad();
  const logits = model.forward(tokens, 1, 4);
  const V = logits.shape[logits.shape.length - 1];
  const loss = crossEntropyPerExample(logits.reshape([logits.size / V, V]), targets).mean();
  loss.backward();

  // Numeric pass, sample a handful of coordinates across all parameters.
  const H = 1e-6;
  let checked = 0;
  for (let pi = 0; pi < params.length && checked < 40; pi++) {
    const p = params[pi];
    for (let i = 0; i < p.size && checked < 40; i += Math.max(1, Math.floor(p.size / 4))) {
      const orig = p.data[i];
      const analytic = p.grad[i];
      p.data[i] = orig + H;
      const fPlus = crossEntropyPerExample(
        model.forward(tokens, 1, 4).reshape([model.forward(tokens, 1, 4).size / V, V]),
        targets,
      ).mean().item();
      p.data[i] = orig - H;
      const fMinus = crossEntropyPerExample(
        model.forward(tokens, 1, 4).reshape([model.forward(tokens, 1, 4).size / V, V]),
        targets,
      ).mean().item();
      p.data[i] = orig;
      const numeric = (fPlus - fMinus) / (2 * H);
      checkClose(`GPT param ${pi}[${i}]`, analytic, numeric, 1e-4, "analytic vs finite diff");
      checked++;
    }
  }
  console.log(`ok  GPT end-to-end (${checked} sampled coordinates)`);
}
// ---------------------------------------------------------------------------
// 7. Training sanity: loss must start near ln(V) and decrease.
// ---------------------------------------------------------------------------
{
  const corpus = "the quick brown fox jumps over the lazy dog. ".repeat(20);
  const tokenizer = new CharTokenizer(corpus);
  const model = new GPT({ vocabSize: tokenizer.vocabSize, blockSize: 16, nEmbd: 32, nHead: 2, nLayer: 2 });
  const opt = new AdamW(model.parameters(), { lr: 1e-3 });
  const targetsOf = (inputs: number[]) => inputs.slice(1).concat(inputs.slice(0, 1));

  const first = (() => {
    const inputs = tokenizer.encode(corpus.slice(0, 16));
    const targets = targetsOf(inputs);
    const logits = model.forward(inputs, 1, 16);
    const V = logits.shape[2];
    return crossEntropyPerExample(logits.reshape([16, V]), targets).mean().item();
  })();
  checkClose("initial loss ≈ ln(vocab)", first, Math.log(tokenizer.vocabSize), 0.75, "GPT-2-style init");

  // A few training steps must drive the loss down.
  const seqLen = 16;
  let last = first;
  for (let step = 0; step < 30; step++) {
    const start = Math.floor(Math.random() * (corpus.length - seqLen - 1));
    const inputs = tokenizer.encode(corpus.slice(start, start + seqLen));
    const targets = tokenizer.encode(corpus.slice(start + 1, start + seqLen + 1));
    const logits = model.forward(inputs, 1, seqLen);
    const V = logits.shape[2];
    const loss = crossEntropyPerExample(logits.reshape([seqLen, V]), targets).mean();
    opt.zeroGrad();
    loss.backward();
    opt.step();
    last = loss.item();
  }
  checks++;
  if (last >= first) {
    failures++;
    console.error(`FAIL training: loss did not decrease (${first.toFixed(3)} → ${last.toFixed(3)})`);
  } else {
    console.log(`ok  training drives loss down (${first.toFixed(3)} → ${last.toFixed(3)})`);
  }
}

console.log(`\n${checks - failures}/${checks} checks passed${failures ? ` — ${failures} FAILURES` : ""}`);
if (failures > 0) process.exit(1);

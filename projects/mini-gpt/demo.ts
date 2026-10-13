/**
 * demo.ts — End-to-end educational demo: train a character-level GPT on a
 * small text corpus in pure TypeScript, entirely on the CPU.
 *
 * Run:   npm run demo    (or:   bun run src/demo.ts)
 */

import { CharTokenizer } from "./data/CharTokenizer";
import { GPT } from "./nn/GPT";
import { AdamW } from "./optim/AdamW";
import { Trainer } from "./train/Trainer";

/** A compact Shakespeare-flavored excerpt (all lowercase-ready, ~1.1k chars). */
const CORPUS = `
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep—
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry,
And lose the name of action.—Soft you now!
The fair Ophelia! Nymph, in thy orisons
Be all my sins remembered.
`;
// Repeat the excerpt so the tiny model can actually memorize its style
// within the 100-step demo budget (an un-repeated 1.5k-string is too little
// signal per step for a character model to overfit fast).
const HF_CORPUS = CORPUS.repeat(4);

interface DemoConfig {
  blockSize: number;
  nEmbd: number;
  nHead: number;
  nLayer: number;
  batchSize: number;
  lr: number;
  steps: number;
  evalEvery: number;
  logEvery: number;
}

/**
 * Deliberately tiny: trains visibly within seconds on CPU.
 * blockSize 32, nEmbd 32, nHead 4, nLayer 2, batch 4 — all at the small end
 * of the suggested ranges, because on a pure-TypeScript interpreter every op
 * costs real time; the architecture is unchanged and scales up smoothly.
 */
const CONFIG: DemoConfig = {
  blockSize: 32,
  nEmbd: 32,
  nHead: 4,
  nLayer: 2,
  batchSize: 4,
  lr: 1e-2,
  steps: 100,
  evalEvery: 50,
  logEvery: 10,
};

function main(): void {
  console.log("=== mini-GPT (character-level, pure TypeScript autograd) ===\n");

  // 1. Vocabulary: fit the character set of the corpus and tokenize it.
  const tokenizer = new CharTokenizer(HF_CORPUS);
  console.log(`corpus: ${HF_CORPUS.length} chars | vocab size: ${tokenizer.vocabSize}`);
  console.log(`vocab: ${JSON.stringify(tokenizer.vocab.join(""))}\n`);

  // 2. Model.
  const model = new GPT({
    vocabSize: tokenizer.vocabSize,
    blockSize: CONFIG.blockSize,
    nEmbd: CONFIG.nEmbd,
    nHead: CONFIG.nHead,
    nLayer: CONFIG.nLayer,
  });
  console.log(`model: ${CONFIG.nLayer} blocks × ${CONFIG.nHead} heads, nEmbd=${CONFIG.nEmbd}`);
  console.log(`parameters: ${model.numParams().toLocaleString()} learnable scalars\n`);

  // 3. Optimizer + trainer.
  const optimizer = new AdamW(model.parameters(), {
    lr: CONFIG.lr,
    weightDecay: 0.01,
  });
  const trainer = new Trainer(model, tokenizer, optimizer, {
    batchSize: CONFIG.batchSize,
    seqLen: CONFIG.blockSize,
    evalEvery: CONFIG.evalEvery,
    logEvery: CONFIG.logEvery,
  });

  // Sanity check: the initial loss of an untrained model should be ≈ ln(V)
  // (uniform guessing). If it is wildly off, something is wrong with init.
  const expected = Math.log(tokenizer.vocabSize);
  const firstLoss = trainer.trainStep();
  console.log(
    `sanity: first loss ${firstLoss.toFixed(3)} vs ln(vocabSize)=${expected.toFixed(3)} ` +
    `(untrained model ≈ uniform guessing)\n`,
  );

  // 4. Training loop (starts at step 2 since we took one warm-up step).
  console.log("--- training ---");
  const t0 = performance.now();
  trainer.train(CONFIG.steps - 1);
  const dt = (performance.now() - t0) / 1000;
  console.log(`training done in ${dt.toFixed(1)}s\n`);

  // 5. Validation + sample generation.
  const valLoss = trainer.evaluate();
  console.log(`final val loss: ${valLoss.toFixed(3)}\n`);

  const prompt = "The ";
  console.log(`greedy (temperature=0) from ${JSON.stringify(prompt)}:`);
  console.log("  " + trainer.generate(prompt, 50, 0));
  console.log();

  for (const temp of [0.5, 0.8, 1.0]) {
    console.log(`sampling (temperature=${temp}) from ${JSON.stringify(prompt)}:`);
    console.log("  " + trainer.generate(prompt, 50, temp));
    console.log();
  }
}

main();

/**
 * Comprehensive Test Suite for TypescriptLLM.
 * Tests Tensor Math, RoPE, Attention, MLP, KV-Cache, Samplers, SafeTensors, and Transformer forward pass.
 */

import { Tensor } from '../projects/smollm-inference/core/tensor.js';
import { KVCacheManager } from '../projects/smollm-inference/core/kv-cache.js';
import { Linear } from '../projects/smollm-inference/layers/linear.js';
import { RMSNorm } from '../projects/smollm-inference/layers/rms-norm.js';
import { RotaryEmbedding } from '../projects/smollm-inference/layers/rope.js';
import { SwiGLUMLP } from '../projects/smollm-inference/layers/mlp.js';
import { MultiHeadAttention } from '../projects/smollm-inference/layers/attention.js';
import { TransformerBlock } from '../projects/smollm-inference/layers/transformer-block.js';
import { GreedySampler } from '../projects/smollm-inference/samplers/greedy-sampler.js';
import { NucleusSampler } from '../projects/smollm-inference/samplers/nucleus-sampler.js';
import { SamplerFactory } from '../projects/smollm-inference/samplers/sampler-factory.js';
import { SafeTensorsReader } from '../projects/smollm-inference/io/safetensors.js';

let passed = 0;
let failed = 0;

function assert(condition: boolean, msg: string): void {
  if (condition) {
    console.log(`  ✓ ${msg}`);
    passed++;
  } else {
    console.error(`  ✗ FAIL: ${msg}`);
    failed++;
  }
}

function assertClose(a: number, b: number, eps = 1e-4, msg?: string): void {
  const diff = Math.abs(a - b);
  assert(diff <= eps, `${msg ?? 'Value comparison'}: expected ${b}, got ${a} (diff ${diff.toExponential(2)})`);
}

console.log('='.repeat(60));
console.log('           RUNNING TypescriptLLM TEST SUITE');
console.log('='.repeat(60));

// --- 1. Tensor Tests ---
console.log('\n[1] Testing Tensor Abstraction & Math Primitives');
{
  const t = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
  assert(t.shape[0] === 2 && t.shape[1] === 3, 'Tensor shape [2, 3]');
  assert(t.stride[0] === 3 && t.stride[1] === 1, 'Tensor strides [3, 1]');
  assert(t.get(1, 2) === 6, 'Tensor indexing get(1, 2) == 6');

  // Linear projection: Y = X * W^T
  // X: [1, 2] = [2, 3]
  // W: [2, 2] = [[1, 2], [3, 4]]
  // Y = [2*1 + 3*2, 2*3 + 3*4] = [8, 18]
  const x = new Tensor([1, 2], [2, 3]);
  const w = new Tensor([2, 2], [1, 2, 3, 4]);
  const y = x.linear(w);
  assertClose(y.get(0, 0), 8, 1e-5, 'Linear row 0');
  assertClose(y.get(0, 1), 18, 1e-5, 'Linear row 1');

  // SiLU
  // silu(2.0) = 2.0 / (1 + exp(-2.0)) = 2.0 / (1 + 0.135335) = 1.761594
  const s = new Tensor([1], [2.0]).silu();
  assertClose(s.data[0], 1.761594, 1e-4, 'SiLU activation at x=2.0');

  // RMSNorm
  // x = [3.0, 4.0], weight = [1.0, 1.0], eps = 0
  // rms = sqrt((9 + 16)/2) = sqrt(12.5) = 3.53553
  // y = [3/3.53553, 4/3.53553] = [0.848528, 1.13137]
  const normWeight = new Tensor([2], [1.0, 1.0]);
  const normIn = new Tensor([1, 2], [3.0, 4.0]);
  const normOut = normIn.rmsNorm(normWeight, 1e-5);
  assertClose(normOut.data[0], 0.848528, 1e-4, 'RMSNorm elem 0');
  assertClose(normOut.data[1], 1.13137, 1e-4, 'RMSNorm elem 1');

  // Softmax
  const smIn = new Tensor([1, 3], [1.0, 2.0, 3.0]);
  const smOut = smIn.softmax();
  assertClose(smOut.data[0] + smOut.data[1] + smOut.data[2], 1.0, 1e-5, 'Softmax sums to 1.0');
  assert(smOut.data[2] > smOut.data[1] && smOut.data[1] > smOut.data[0], 'Softmax monotonicity');
}

// --- 2. RoPE Tests ---
console.log('\n[2] Testing Rotary Position Embedding (RoPE)');
{
  const headDim = 4;
  const rope = new RotaryEmbedding(headDim, 100, 10000.0);
  // At pos = 0, cos = 1, sin = 0, so RoPE is identity!
  const q0 = new Tensor([1, headDim], [1.0, 2.0, 3.0, 4.0]);
  const rot0 = rope.forward(q0, 0, 1);
  assertClose(rot0.data[0], 1.0, 1e-5, 'RoPE pos 0 is identity (elem 0)');
  assertClose(rot0.data[1], 2.0, 1e-5, 'RoPE pos 0 is identity (elem 1)');
  assertClose(rot0.data[2], 3.0, 1e-5, 'RoPE pos 0 is identity (elem 2)');
  assertClose(rot0.data[3], 4.0, 1e-5, 'RoPE pos 0 is identity (elem 3)');

  // At pos > 0, L2 norm of the vector must be preserved!
  const rot1 = rope.forward(q0, 5, 1);
  const normBefore = Math.hypot(...q0.data);
  const normAfter = Math.hypot(...rot1.data);
  assertClose(normBefore, normAfter, 1e-5, 'RoPE preserves L2 norm (isometry)');
}

// --- 3. KV-Cache Manager Tests ---
console.log('\n[3] Testing KVCacheManager');
{
  const kvCache = new KVCacheManager(2, 1, 4, 16);
  assert(kvCache.getSeqLen(0) === 0, 'Initial sequence length is 0');

  const k1 = new Tensor([1, 4], [1, 2, 3, 4]);
  const v1 = new Tensor([1, 4], [5, 6, 7, 8]);
  kvCache.append(0, k1, v1);
  assert(kvCache.getSeqLen(0) === 1, 'Sequence length is 1 after append');

  const activeK = kvCache.getActiveKeys(0);
  assert(activeK.shape[0] === 1, 'Active keys shape [1, 1, 4]');
  assertClose(activeK.data[0], 1, 1e-5, 'Active key matches inserted data');

  kvCache.reset();
  assert(kvCache.getSeqLen(0) === 0, 'Sequence length reset to 0');
}

// --- 4. SwiGLU MLP Tests ---
console.log('\n[4] Testing SwiGLU MLP');
{
  const hiddenSize = 4;
  const interSize = 8;
  const gateProj = new Linear(Tensor.ones([interSize, hiddenSize]));
  const upProj = new Linear(Tensor.ones([interSize, hiddenSize]));
  const downProj = new Linear(Tensor.ones([hiddenSize, interSize]));

  const mlp = new SwiGLUMLP(gateProj, upProj, downProj);
  const x = Tensor.ones([1, hiddenSize]);
  const out = mlp.forward(x);
  assert(out.shape[0] === 1 && out.shape[1] === hiddenSize, 'MLP output shape matches hiddenSize');
}

// --- 5. Attention & Transformer Block Tests ---
console.log('\n[5] Testing Attention & Transformer Block');
{
  const hiddenSize = 8;
  const numHeads = 4;
  const numKvHeads = 2; // GQA (2 groups)
  const headDim = 2;

  const qProj = new Linear(Tensor.ones([hiddenSize, hiddenSize]));
  const kProj = new Linear(Tensor.ones([numKvHeads * headDim, hiddenSize]));
  const vProj = new Linear(Tensor.ones([numKvHeads * headDim, hiddenSize]));
  const oProj = new Linear(Tensor.ones([hiddenSize, hiddenSize]));
  const rope = new RotaryEmbedding(headDim, 16);

  const attn = new MultiHeadAttention(qProj, kProj, vProj, oProj, rope, numHeads, numKvHeads, headDim);
  const kvCache = new KVCacheManager(1, numKvHeads, headDim, 16);

  const x = Tensor.ones([1, hiddenSize]);
  const attnOut = attn.forward(x, 0, kvCache, 0);
  assert(attnOut.shape[0] === 1 && attnOut.shape[1] === hiddenSize, 'Attention output shape [1, 8]');

  const inputNorm = new RMSNorm(Tensor.ones([hiddenSize]));
  const postNorm = new RMSNorm(Tensor.ones([hiddenSize]));
  const mlp = new SwiGLUMLP(
    new Linear(Tensor.ones([16, hiddenSize])),
    new Linear(Tensor.ones([16, hiddenSize])),
    new Linear(Tensor.ones([hiddenSize, 16]))
  );

  const block = new TransformerBlock(0, inputNorm, attn, postNorm, mlp);
  kvCache.reset();
  const blockOut = block.forward(x, 0, kvCache);
  assert(blockOut.shape[0] === 1 && blockOut.shape[1] === hiddenSize, 'TransformerBlock forward pass succeeds');
}

// --- 6. Samplers Tests ---
console.log('\n[6] Testing Samplers (Greedy & Nucleus)');
{
  const logits = new Tensor([4], [1.0, 5.0, 2.0, 3.0]);

  const greedy = new GreedySampler();
  assert(greedy.sample(logits, []) === 1, 'Greedy selects argmax index 1');

  // Repetition penalty
  const greedyPenalized = new GreedySampler(10.0);
  assert(greedyPenalized.sample(logits, [1]) === 3, 'Greedy with repetition penalty selects next best index 3');

  // Nucleus Sampler
  const nucleus = new NucleusSampler(0.5, 0.9, 10, 1.0);
  const sampled = nucleus.sample(logits, []);
  assert(sampled >= 0 && sampled < 4, `Nucleus sampling returns valid token index: ${sampled}`);

  // SamplerFactory
  const fGreedy = SamplerFactory.create({ type: 'greedy' });
  assert(fGreedy instanceof GreedySampler, 'SamplerFactory creates GreedySampler');
  const fNucleus = SamplerFactory.create({ temperature: 0.8 });
  assert(fNucleus instanceof NucleusSampler, 'SamplerFactory creates NucleusSampler');
}

// --- 7. SafeTensors Binary Serialization & BF16/F16 Decoding Tests ---
console.log('\n[7] Testing SafeTensors & BF16 Conversion');
{
  // Synthesize a valid SafeTensors file in-memory
  // Header JSON:
  // { "test.weight": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] } }
  // 1.0 in BF16 = 0x3F80 (uint16 LE: 0x80, 0x3F)
  // 2.0 in BF16 = 0x4000 (uint16 LE: 0x00, 0x40)
  const headerObj = {
    'test.weight': {
      dtype: 'BF16',
      shape: [2],
      data_offsets: [0, 4],
    },
  };
  const headerJson = Buffer.from(JSON.stringify(headerObj), 'utf-8');
  const headerLenBuf = Buffer.alloc(8);
  headerLenBuf.writeBigUInt64LE(BigInt(headerJson.length), 0);

  const payload = Buffer.alloc(4);
  payload.writeUInt16LE(0x3f80, 0); // 1.0 in BF16
  payload.writeUInt16LE(0x4000, 2); // 2.0 in BF16

  const fullBuffer = Buffer.concat([headerLenBuf, headerJson, payload]);
  const reader = new SafeTensorsReader(fullBuffer);

  assert(reader.hasTensor('test.weight'), 'SafeTensorsReader finds test.weight');
  const tensor = reader.readTensor('test.weight');
  assertClose(tensor.data[0], 1.0, 1e-4, 'BF16 on-the-fly decoded 1.0');
  assertClose(tensor.data[1], 2.0, 1e-4, 'BF16 on-the-fly decoded 2.0');
}

console.log('\n' + '='.repeat(60));
console.log(`TEST RESULTS: ${passed} Passed, ${failed} Failed`);
console.log('='.repeat(60));

if (failed > 0) {
  process.exit(1);
}

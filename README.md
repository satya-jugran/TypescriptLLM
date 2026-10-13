# TypescriptLLM: Educational Pure TypeScript Neural Networks on CPU

An educational, dependency-free (zero-GPU, 100% CPU-only) machine learning suite built entirely from scratch in strict TypeScript for Node.js.

The repository contains two complete, standalone neural network projects demonstrating modern Transformer architectures and ML systems engineering from fundamental principles:

1. **[SmolLM-135M Inference Engine](./projects/smollm-inference)**: A production-style, high-performance CPU inference engine executing pre-trained Hugging Face models (`SmolLM-135M` / LLaMA architecture) directly from binary SafeTensors with zero-allocation KV-caching, GQA, RoPE, and real-time streaming token generation.
2. **[miniGPT Autograd & Training](./projects/mini-gpt)**: A reverse-mode automatic differentiation engine (`autograd`) and character-level GPT transformer trained end-to-end on CPU with custom backward graphs, AdamW optimizer, and gradient verification against finite differences.

---

## Repository Architecture

```text
typescript-llm/
├── guide/                               # Interactive visual browser guide (HTML/CSS/JS)
│   ├── css/guide.css                    # Styling for dark-mode interactive widgets
│   ├── js/guide-app.js                  # Application routing & chapter state
│   ├── js/widgets.js                    # Interactive sandboxes (Tensors, RoPE, Attention, etc.)
│   └── index.html                       # Standalone single-page guide app
├── models/                              # Downloaded model weights (gitignored)
│   └── SmolLM-135M/                     # safetensors, tokenizer, and config files
├── projects/
│   ├── smollm-inference/                # 1. SmolLM-135M CPU Inference Engine
│   │   ├── core/
│   │   │   ├── kv-cache.ts              # Zero-allocation stateful Key-Value cache manager
│   │   │   ├── tensor.ts                # Contiguous Float32Array wrapper & vectorized math
│   │   │   └── types.ts                 # Architecture configs, DTOs & interfaces
│   │   ├── engine/
│   │   │   └── inference-engine.ts      # High-level Facade orchestrating Prefill & Decode
│   │   ├── io/
│   │   │   ├── model-loader.ts          # Hugging Face config parser & ModelBuilder
│   │   │   └── safetensors.ts           # Binary SafeTensors reader with BF16/F16 decoding
│   │   ├── layers/
│   │   │   ├── attention.ts             # Scaled Dot-Product & Grouped-Query Attention (GQA)
│   │   │   ├── linear.ts                # Dense linear projection (Y = XW^T)
│   │   │   ├── mlp.ts                   # SwiGLU Gated Feed-Forward Network
│   │   │   ├── module.ts                # Abstract base Module (nn.Module pattern)
│   │   │   ├── rms-norm.ts              # Root Mean Square layer normalization
│   │   │   ├── rope.ts                  # Rotary Position Embeddings (RoPE)
│   │   │   └── transformer-block.ts     # Pre-norm residual Transformer layer
│   │   ├── model/
│   │   │   └── transformer.ts           # Complete SmolLM/LLaMA Transformer (30 Blocks)
│   │   ├── samplers/
│   │   │   ├── greedy-sampler.ts        # Argmax with frequency repetition penalty
│   │   │   ├── nucleus-sampler.ts       # Top-P, Top-K, temperature & repetition penalty
│   │   │   ├── sampler-factory.ts       # Factory Pattern for sampler instantiation
│   │   │   └── sampler.interface.ts     # Strategy Pattern sampler interface
│   │   ├── scripts/
│   │   │   ├── download-model.ts        # Resumable streaming downloader for SmolLM-135M
│   │   │   └── serve-guide.ts           # Zero-dependency HTTP static server for guide
│   │   ├── index.ts                     # CLI runner with real-time streaming & telemetry
│   │   └── tsconfig.json                # Composite TypeScript project configuration
│   └── mini-gpt/                        # 2. miniGPT Training & Autograd Engine
│       ├── autograd/
│       │   ├── Operations.ts            # Differentiable ops (causal mask, cross-entropy)
│       │   ├── Tensor.ts                # Dynamic computation graph & backward pass
│       │   └── test.ts                  # Numerical gradient checks (central finite diff)
│       ├── data/
│       │   └── CharTokenizer.ts         # Character-level vocabulary encoder/decoder
│       ├── nn/
│       │   ├── Attention.ts             # Differentiable causal multi-head attention
│       │   ├── Embedding.ts             # Learnable Token and Position embeddings
│       │   ├── FeedForward.ts           # 2-layer MLP with GELU activation
│       │   ├── GPT.ts                   # Full GPT architecture with parameter tracking
│       │   ├── LayerNorm.ts             # Standard LayerNorm with learnable scale & shift
│       │   ├── Linear.ts                # Dense linear layer with weight & bias gradients
│       │   ├── Module.ts                # PyTorch-like base module & parameter tree
│       │   └── TransformerBlock.ts      # Residual Pre-Norm Transformer block
│       ├── optim/
│       │   └── AdamW.ts                 # AdamW optimizer with decoupled weight decay
│       ├── train/
│       │   ├── Loss.ts                  # Cross-Entropy loss with Log-Sum-Exp stability
│       │   └── Trainer.ts               # Batched training loop, validation & sampling
│       ├── demo.ts                      # End-to-end Shakespeare training demo
│       ├── index.ts                     # Public API exports
│       └── tsconfig.json                # Project TypeScript configuration
├── test/
│   └── run-tests.ts                     # Unit test suite for SmolLM inference engine
├── package.json                         # Root workspace scripts & package metadata
├── tsconfig.json                        # Root project references
└── .gitignore                           # Git ignore rules for node_modules, dist, models
```

---

## Project Highlights

### Project 1: SmolLM-135M Inference Engine (`projects/smollm-inference`)
- **Contiguous Vectorized Tensors**: Row-major flat `Float32Array` buffers with 8-way unrolled inner products to avoid garbage-collection overhead.
- **RMSNorm (Root Mean Square Normalization)**: Mean-centering-free normalization minimizing memory passes.
- **Rotary Position Embeddings (RoPE)**: Precomputed frequency tables with Hugging Face `rotate_half` formulation.
- **Grouped-Query Attention (GQA)**: 9 Query heads mapped to 3 KV heads (3:1 ratio) to minimize memory bandwidth.
- **Stateful Zero-Allocation KV-Cache**: Pre-allocated contiguous buffers managing prefill and autoregressive decode steps without per-token allocations.
- **SwiGLU Feed-Forward Network**: Gated SiLU activation matching standard LLaMA architecture.
- **Binary SafeTensors & BF16 Conversion**: Direct binary header parsing and on-the-fly Bfloat16/FP16 to FP32 conversion.
- **Streaming Samplers**: Real-time token generator supporting Greedy and Nucleus sampling (Top-P, Top-K, temperature, repetition penalty).

### Project 2: miniGPT Training & Autograd Engine (`projects/mini-gpt`)
- **Reverse-Mode Autograd Engine**: Dynamic directed acyclic computation graph (DAG) with topological sort backpropagation and multivariable chain rule gradient accumulation.
- **Numerical Gradient Checks**: Analytic backprop gradients validated against central finite difference approximations ($O(h^2)$ error).
- **Modular Neural Network Layers**: PyTorch-style `Module` hierarchy with parameter discovery (`Linear`, `LayerNorm`, `TokenEmbedding`, `PositionEmbedding`, `TransformerBlock`).
- **Causal Multi-Head Self-Attention**: Multi-head attention with causal upper-triangular masking to enforce autoregressive conditioning.
- **Numerically Stable Cross-Entropy Loss**: Log-sum-exp trick to prevent floating-point overflow during exponentiation.
- **AdamW Optimizer**: First and second moment tracking with bias correction and decoupled weight decay.
- **End-to-End CPU Trainer**: Batched training loop with validation split, loss tracking, and sample generation.

---

## Getting Started

### Prerequisites
- **Node.js**: v18.0.0 or later (tested on Node v20+ and v24+)
- **npm**: v9.0.0 or later

### Installation
Clone the repository and install developer dependencies:
```bash
git clone https://github.com/satya-jugran/typescript-llm.git
cd typescript-llm
npm install
```

---

## Commands & Scripts Reference

All primary workflows are configured as npm scripts in `package.json`:

| Command | Script / Target | Description |
|---|---|---|
| `npm run smollm:start` | `tsx projects/smollm-inference/index.ts` | Run SmolLM-135M inference engine with streaming output |
| `npm run smollm:download` | `tsx projects/smollm-inference/scripts/download-model.ts` | Download SmolLM-135M weights and config (~538 MB) |
| `npm run smollm:build` *(or `npm run build`)* | `tsc -b projects/smollm-inference/tsconfig.json` | Compile TypeScript inference engine into `dist/` |
| `npm run smollm:test` *(or `npm test`)* | `tsx test/run-tests.ts` | Run SmolLM mathematical layer & component test suite |
| `npm run minigpt:demo` | `tsx projects/mini-gpt/demo.ts` | Train character-level miniGPT with autograd on CPU |
| `npm run minigpt:test` | `tsx projects/mini-gpt/autograd/test.ts` | Run autograd numerical gradient checks (finite difference) |
| `npm run minigpt:typecheck` | `tsc -p projects/mini-gpt/tsconfig.json` | Type-check the mini-GPT codebase with strict TypeScript |
| `npm run guide` | `tsx projects/smollm-inference/scripts/serve-guide.ts` | Serve the interactive visual browser guide at `localhost:3000` |

---

## Usage Guide

### 1. SmolLM-135M Inference Engine

#### Step A: Download Model Weights
Download the pre-trained weights from Hugging Face Hub (stored under `models/SmolLM-135M/`):
```bash
npm run smollm:download
```
*(If weights are not found when starting inference, `smollm:start` will automatically download them.)*

#### Step B: Run Streaming Generation
Run generation with custom prompt, token budget, and temperature:
```bash
npm run smollm:start -- --prompt "The key principles of computer systems architecture are" --max-tokens 48 --temperature 0.7
```

Deterministic greedy generation:
```bash
npm run smollm:start -- --prompt "The boiling point of water is" --max-tokens 15 --temperature 0
```

#### Step C: Build and Run Compiled JavaScript
```bash
npm run build
node dist/index.js --prompt "Artificial intelligence is" --max-tokens 30 --temperature 0.7
```

#### Inference CLI Options

| Option | Default | Description |
|---|---|---|
| `--model-dir` | `./models/SmolLM-135M` | Path to directory with `model.safetensors` and `config.json` |
| `--prompt` | `"The key principles..."` | Input prompt text |
| `--max-tokens` | `48` | Maximum number of autoregressive tokens to generate |
| `--temperature` | `0.7` | Temperature scaling (`0` for deterministic greedy sampling) |
| `--top-p` | `0.9` | Nucleus sampling cumulative probability threshold |
| `--top-k` | `50` | Top-K candidate truncation |
| `--rep-penalty` | `1.1` | Repetition penalty applied to previous tokens |

---

### 2. miniGPT Training Demo

Train a 2-layer, 4-head character-level transformer on CPU in pure TypeScript using backpropagation:
```bash
npm run minigpt:demo
```

Sample output:
```text
=== mini-GPT (character-level, pure TypeScript autograd) ===

corpus: 5956 chars | vocab size: 46
model: 2 blocks × 4 heads, nEmbd=32
parameters: 26,798 learnable scalars

sanity: first loss 3.844 vs ln(vocabSize)=3.829 (untrained model ≈ uniform guessing)

--- training ---
step   11 | loss 3.5124
step   21 | loss 3.2996
step   31 | loss 3.2044
step   41 | loss 3.2683
step   51 | loss 3.0819
step   51 | val loss 3.1124
step   61 | loss 2.9255
step   71 | loss 3.0412
step   81 | loss 2.9254
step   91 | loss 2.8023
training done in 5.2s

final val loss: 2.790

greedy (temperature=0) from "The ":
  The t th tathe t the t to the the the the the the the
```

---

### 3. Interactive Visual Guide

You can access the live guide hosted on GitHub Pages:
👉 **[https://satya-jugran.github.io/typescript-llm/](https://satya-jugran.github.io/typescript-llm/)**

Or launch it locally:
```bash
npm run guide
```
Open [http://localhost:3000](http://localhost:3000) in your browser (or open `guide/index.html` directly). The interactive guide features live sandboxes explaining:
- **Tensors & Strides**: Memory layout, row-major indexing, contiguous buffer slicing.
- **Embeddings**: Token ID to dense vector lookup.
- **Rotary Position Embeddings (RoPE)**: 2D vector rotation visualization and isometry.
- **Attention & KV-Cache**: Scaled dot-product attention heatmap, causal masking, memory savings.
- **SafeTensors & BF16**: Binary float conversion breakdown.
- **Sampling Strategies**: Interactive temperature, Top-K, and Top-P probability distribution explorer.

---

## Performance Benchmark on CPU

Execution on standard modern consumer CPU (SmolLM-135M, 30 Layers, 576 Hidden Size, 49,152 Vocab):
- **Model Load Time**: ~0.8s to 1.6s (SafeTensors parsing + BF16 FP32 conversion)
- **Prefill Latency**: ~140 – 170 ms / prompt token
- **Autoregressive Decode Throughput**: **~6 to 7 tokens / second** in pure TypeScript
- **Memory Footprint**: ~600 MB (fits comfortably within default Node.js memory limit)

---

## Testing

Run the SmolLM inference mathematical unit test suite:
```bash
npm test
```
*Validates Tensor indexing, linear projections, SiLU, RMSNorm, Softmax, RoPE isometry, KV-Cache sliding window, SwiGLU MLP, Multi-Head Attention, Greedy/Nucleus samplers, and SafeTensors BF16 decoding (31/31 assertions).*

Run the autograd finite-difference gradient checks:
```bash
npm run minigpt:test
```
*Validates analytic backpropagation gradients against central finite difference estimates across operations, loss, and full transformer layers.*

Type-check both projects:
```bash
npm run build
npm run minigpt:typecheck
```

# TypescriptLLM: Educational Pure TypeScript Transformer Inference Engine

An educational, dependency-free (zero-GPU, CPU-only) Large Language Model (Transformer) inference engine built entirely from scratch in strict TypeScript for Node.js.

Designed following strict Object-Oriented Programming (OOP) and ML Systems Architecture principles, **TypescriptLLM** executes standard modern Transformer architectures (LLaMA, SmolLM) with pure JavaScript/TypeScript math routines, custom contiguous tensor buffers, and real-time streaming token generation.

---

## Architecture & Design Patterns

TypescriptLLM implements a modular, PyTorch-like `nn.Module` hierarchy designed for clarity, extensibility, and CPU cache-friendliness:

```text
src/
├── core/
│   ├── tensor.ts          # Contiguous Float32Array wrapper & vectorized math
│   ├── kv-cache.ts        # Zero-allocation stateful Key-Value cache manager
│   └── types.ts           # Architectural configs, DTOs & interfaces
├── layers/
│   ├── module.ts          # Abstract base Module (nn.Module pattern)
│   ├── linear.ts          # Dense linear projection (Y = XW^T)
│   ├── rms-norm.ts        # Root Mean Square layer normalization
│   ├── rope.ts            # Rotary Position Embeddings (RoPE)
│   ├── attention.ts       # Scaled Dot-Product & Grouped-Query Attention (GQA)
│   ├── mlp.ts             # SwiGLU Gated Feed-Forward Network
│   └── transformer-block.ts # Pre-norm residual Transformer layer
├── model/
│   └── transformer.ts     # Complete Transformer (Embed -> 30 Blocks -> Norm -> LM Head)
├── samplers/
│   ├── sampler.interface.ts # Strategy Pattern interface
│   ├── greedy-sampler.ts    # Argmax with repetition penalty
│   ├── nucleus-sampler.ts   # Top-P, Top-K, temperature & repetition penalty
│   └── sampler-factory.ts   # Factory Pattern for sampler instantiation
├── io/
│   ├── safetensors.ts     # Binary SafeTensors reader with on-the-fly BF16/F16 conversion
│   └── model-loader.ts    # Hugging Face config parser & ModelBuilder
├── tokenizer/
│   ├── tokenizer.interface.ts # ITokenizer abstraction
│   └── hf-tokenizer.ts       # Adapter pattern for Hugging Face tokenizers
├── engine/
│   └── inference-engine.ts   # High-level Facade orchestrating Prefill & Decode
├── scripts/
│   └── download-model.ts     # Resumable streaming downloader for SmolLM-135M
└── index.ts                  # Interactive CLI demo with streaming & telemetry
```

---

## Mathematical Formulations

### 1. Contiguous Vectorized Tensor Operations (`core/tensor.ts`)
Instead of nested `number[][]` arrays that create millions of GC objects and memory fragmentation, TypescriptLLM stores elements in flat, row-major `Float32Array` buffers:

$$\text{offset}(i_0, i_1, \dots, i_{k}) = \sum_{j=0}^{k} i_j \cdot \text{stride}[j]$$

The dense linear projection computes $Y = X W^T$ directly from contiguous memory rows using 8-way unrolled inner products:

$$Y_{m, n} = \sum_{k=0}^{K-1} X_{m, k} \cdot W_{n, k}$$

### 2. Root Mean Square Layer Normalization (`layers/rms-norm.ts`)
RMSNorm replaces standard LayerNorm by dispensing with mean centering, improving CPU efficiency while preserving representation quality:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

$$y_i = \frac{x_i}{\text{RMS}(x)} \cdot w_i$$

### 3. Rotary Position Embedding (`layers/rope.ts`)
Frequencies are precomputed into static $\cos$ and $\sin$ lookup tables:

$$\theta_i = \text{rope\_theta}^{-2i / d}, \quad i \in \left[0, \frac{d}{2}-1\right]$$

Applying Hugging Face's `rotate_half` formulation:

$$\begin{aligned}
v_1 &= x[0 : d/2], \quad v_2 = x[d/2 : d] \\
x_{\text{rope}}[0 : d/2] &= v_1 \cos(m\theta) - v_2 \sin(m\theta) \\
x_{\text{rope}}[d/2 : d] &= v_2 \cos(m\theta) + v_1 \sin(m\theta)
\end{aligned}$$

### 4. Grouped-Query Attention with KV-Cache (`layers/attention.ts`)
SmolLM-135M uses 9 Query heads and 3 Key/Value heads ($3\times$ Grouped-Query Attention). Each KV head is mapped to:

$$h_{kv} = \left\lfloor \frac{h_q}{3} \right\rfloor$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$

### 5. SwiGLU Gated Feed-Forward Network (`layers/mlp.ts`)
Modern LLaMA architectures utilize SwiGLU instead of standard GELU/ReLU activations:

$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}^T) \odot (x W_{\text{up}}^T) \right) W_{\text{down}}^T$$

where $\text{SiLU}(z) = \frac{z}{1 + e^{-z}}$.

---

## Getting Started

### Prerequisites
- Node.js 18+ (tested on Node v24.15.0)
- npm

### 1. Installation
```bash
git clone <repo-url>
cd TypescriptLLM
npm install
```

### 2. Run the Interactive Visual Guide
Launch the built-in visual guide to walk through every concept (Tensors, Embeddings, RoPE, Attention, KV-Cache, BF16, and Sampling) with interactive sandboxes:
```bash
npm run guide
```
Then open `http://localhost:3000` in your browser (or double-click `guide/index.html`).

### 3. Run Test Suite
Verify all mathematical layers, tensor math, RoPE, KV cache, and samplers:
```bash
npm test
```

### 4. Download Model Weights
Download `HuggingFaceTB/SmolLM-135M` (or pass custom arguments):
```bash
npm run download
```

### 5. Interactive CLI Generation
Run the inference engine in TypeScript:
```bash
npm start -- --prompt "The capital of France is" --max-tokens 15 --temperature 0
```

Or run creative nucleus sampling:
```bash
npx tsx src/index.ts --prompt "Once upon a time in a small village," --max-tokens 32 --temperature 0.7 --top-p 0.9
```

Or run the compiled distribution:
```bash
npm run build
node dist/index.js --prompt "The boiling point of water is" --max-tokens 10 --temperature 0
```

---

## Supported Options

| Option | Default | Description |
|---|---|---|
| `--model-dir` | `./models/SmolLM-135M` | Path to model folder containing `model.safetensors` and `config.json` |
| `--prompt` | `"The key principles..."` | Input prompt text |
| `--max-tokens` | `48` | Maximum number of autoregressive tokens to generate |
| `--temperature` | `0.7` | Temperature scaling (set `0` for deterministic greedy sampling) |
| `--top-p` | `0.9` | Nucleus sampling cumulative probability threshold |
| `--top-k` | `50` | Top-K candidate truncation |
| `--rep-penalty`| `1.1` | Frequency repetition penalty applied to context tokens |

---

## Performance Benchmark on CPU

Execution on standard multi-core consumer CPU (SmolLM-135M, 30 Layers, 576 Hidden Size, 49,152 Vocab):
- **Model Load Time**: ~0.7s to 1.0s
- **Prefill Latency**: ~140 ms / prompt token
- **Autoregressive Decode Throughput**: **~5 to 7 tokens / second** in pure TypeScript
- **Memory Footprint**: ~600 MB (fits comfortably in default Node.js heap)

/**
 * InferenceEngine Facade & Generation Orchestrator.
 * Manages the generation lifecycle: Tokenization -> Prefill Phase -> Autoregressive Decode Phase -> Streaming.
 */

import { performance } from 'perf_hooks';
import { TransformerModel } from '../model/transformer.js';
import { ITokenizer } from '../tokenizer/tokenizer.interface.js';
import { KVCacheManager } from '../core/kv-cache.js';
import { SamplerFactory } from '../samplers/sampler-factory.js';
import { LlamaConfig, GenerationRequest, GenerationResult } from '../core/types.js';
import { ModelLoader } from '../io/model-loader.js';
import { HFTokenizer } from '../tokenizer/hf-tokenizer.js';

export class InferenceEngine {
  public readonly model: TransformerModel;
  public readonly tokenizer: ITokenizer;
  public readonly config: LlamaConfig;
  public readonly kvCache: KVCacheManager;

  constructor(model: TransformerModel, tokenizer: ITokenizer, config: LlamaConfig) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.config = config;

    const headDim = config.head_dim ?? Math.floor(config.hidden_size / config.num_attention_heads);
    const maxSeqLen = config.max_position_embeddings ?? 2048;

    this.kvCache = new KVCacheManager(
      config.num_hidden_layers,
      config.num_key_value_heads,
      headDim,
      maxSeqLen
    );
  }

  /**
   * High-level factory to load and initialize engine from a model directory.
   */
  public static async fromDirectory(modelDir: string): Promise<InferenceEngine> {
    const { model, config } = ModelLoader.loadFromDirectory(modelDir);
    const tokenizer = await HFTokenizer.fromPretrained(modelDir, {
      eosTokenId: config.eos_token_id ?? 0,
      bosTokenId: config.bos_token_id ?? 0,
    });
    return new InferenceEngine(model, tokenizer, config);
  }

  /**
   * Generates text given a prompt and sampling hyperparameters.
   */
  public async generate(request: GenerationRequest): Promise<GenerationResult> {
    const {
      prompt,
      maxTokens = 64,
      temperature = 0.7,
      topK = 50,
      topP = 0.9,
      repetitionPenalty = 1.1,
      stopTokens = [],
      onToken,
    } = request;

    // Reset KV cache state for fresh generation
    this.kvCache.reset();

    // 1. Encode prompt
    const promptTokens = await this.tokenizer.encode(prompt);
    if (promptTokens.length === 0) {
      throw new Error('Prompt cannot be empty or tokenized into 0 tokens');
    }

    const sampler = SamplerFactory.create({
      type: temperature <= 0 ? 'greedy' : 'nucleus',
      temperature,
      topK,
      topP,
      repetitionPenalty,
    });

    const contextTokens = [...promptTokens];
    const generatedTokens: number[] = [];

    // Stop token IDs and stop words
    const numericStopTokens = new Set<number>();
    const stringStopTokens: string[] = [];

    if (this.tokenizer.eosTokenId !== undefined) {
      numericStopTokens.add(this.tokenizer.eosTokenId);
    }
    if (this.config.eos_token_id !== undefined) {
      numericStopTokens.add(this.config.eos_token_id);
    }

    for (const stop of stopTokens) {
      if (typeof stop === 'number') {
        numericStopTokens.add(stop);
      } else {
        stringStopTokens.push(stop);
      }
    }

    // 2. Prefill Phase
    const prefillStart = performance.now();
    // Feed full prompt at startPos = 0
    const prefillLogits = this.model.forward(promptTokens, 0, this.kvCache, false);
    const prefillEnd = performance.now();
    const prefillTimeMs = prefillEnd - prefillStart;

    // Sample first token
    let currentToken = sampler.sample(prefillLogits, contextTokens);
    let currentPos = promptTokens.length;

    // 3. Autoregressive Decode Phase
    const decodeStart = performance.now();
    let accumulatedText = '';

    while (generatedTokens.length < maxTokens) {
      // Check EOS or stop token IDs
      if (numericStopTokens.has(currentToken)) {
        break;
      }

      generatedTokens.push(currentToken);
      contextTokens.push(currentToken);

      // Stream token if callback provided
      if (onToken) {
        const tokenStr = await this.tokenizer.decode([currentToken]);
        accumulatedText += tokenStr;
        onToken(tokenStr, currentToken);
      }

      // Check string stop patterns
      if (stringStopTokens.length > 0) {
        if (!onToken) {
          accumulatedText = await this.tokenizer.decode(generatedTokens);
        }
        const hasStopWord = stringStopTokens.some((word) => accumulatedText.includes(word));
        if (hasStopWord) {
          break;
        }
      }

      if (generatedTokens.length >= maxTokens) {
        break;
      }

      // Single-token forward step
      const stepLogits = this.model.forward([currentToken], currentPos, this.kvCache, false);
      currentToken = sampler.sample(stepLogits, contextTokens);
      currentPos++;
    }

    const decodeEnd = performance.now();
    const decodeTimeMs = decodeEnd - decodeStart;

    const fullText = await this.tokenizer.decode(generatedTokens);
    const totalTimeMs = prefillTimeMs + decodeTimeMs;
    const tokensGenerated = generatedTokens.length;
    const tokensPerSecond =
      decodeTimeMs > 0 ? Number(((tokensGenerated / decodeTimeMs) * 1000).toFixed(2)) : 0;

    return {
      text: fullText,
      tokens: generatedTokens,
      tokensGenerated,
      promptTokensCount: promptTokens.length,
      prefillTimeMs: Number(prefillTimeMs.toFixed(2)),
      decodeTimeMs: Number(decodeTimeMs.toFixed(2)),
      totalTimeMs: Number(totalTimeMs.toFixed(2)),
      tokensPerSecond,
    };
  }
}

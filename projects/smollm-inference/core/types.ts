/**
 * Core type definitions, configuration models, and DTOs for the TypescriptLLM engine.
 */

export interface LlamaConfig {
  readonly hidden_size: number;
  readonly intermediate_size: number;
  readonly num_attention_heads: number;
  readonly num_hidden_layers: number;
  readonly num_key_value_heads: number;
  readonly vocab_size: number;
  readonly rms_norm_eps: number;
  readonly rope_theta: number;
  readonly max_position_embeddings: number;
  readonly tie_word_embeddings?: boolean;
  readonly bos_token_id?: number;
  readonly eos_token_id?: number;
  readonly architectures?: string[];
  readonly model_type?: string;
  readonly head_dim?: number;
}

export type SamplerType = 'greedy' | 'nucleus';

export interface SamplerConfig {
  readonly type?: SamplerType;
  readonly temperature?: number;
  readonly topK?: number;
  readonly topP?: number;
  readonly repetitionPenalty?: number;
}

export interface GenerationRequest {
  readonly prompt: string;
  readonly maxTokens?: number;
  readonly temperature?: number;
  readonly topK?: number;
  readonly topP?: number;
  readonly repetitionPenalty?: number;
  readonly stopTokens?: (string | number)[];
  readonly onToken?: (token: string, tokenId: number) => void;
}

export interface GenerationResult {
  readonly text: string;
  readonly tokens: number[];
  readonly tokensGenerated: number;
  readonly promptTokensCount: number;
  readonly prefillTimeMs: number;
  readonly decodeTimeMs: number;
  readonly totalTimeMs: number;
  readonly tokensPerSecond: number;
}

export type SafeTensorDType = 'F32' | 'F16' | 'BF16' | 'I32' | 'I64' | 'U8';

export interface SafeTensorMetadata {
  readonly dtype: string;
  readonly shape: number[];
  readonly data_offsets: [number, number];
}

export interface SafeTensorsHeader {
  readonly __metadata__?: Record<string, string>;
  readonly [tensorName: string]: SafeTensorMetadata | Record<string, string> | undefined;
}

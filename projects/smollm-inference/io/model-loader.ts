/**
 * ModelBuilder and ModelLoader implementing the Builder & Factory Patterns.
 * Maps Hugging Face config.json and SafeTensors weights into a structured TransformerModel.
 */

import * as fs from 'fs';
import * as path from 'path';
import { LlamaConfig } from '../core/types.js';
import { TransformerModel } from '../model/transformer.js';
import { TransformerBlock } from '../layers/transformer-block.js';
import { MultiHeadAttention } from '../layers/attention.js';
import { SwiGLUMLP } from '../layers/mlp.js';
import { RMSNorm } from '../layers/rms-norm.js';
import { Linear } from '../layers/linear.js';
import { RotaryEmbedding } from '../layers/rope.js';
import { IWeightReader, SafeTensorsReader } from './safetensors.js';

export class ModelBuilder {
  private config?: LlamaConfig;
  private reader?: IWeightReader;

  public setConfig(config: LlamaConfig): this {
    this.config = config;
    return this;
  }

  public setWeightReader(reader: IWeightReader): this {
    this.reader = reader;
    return this;
  }

  public build(): TransformerModel {
    if (!this.config) {
      throw new Error('ModelBuilder: Config must be set before building');
    }
    if (!this.reader) {
      throw new Error('ModelBuilder: WeightReader must be set before building');
    }

    const cfg = this.config;
    const reader = this.reader;

    const headDim = cfg.head_dim ?? Math.floor(cfg.hidden_size / cfg.num_attention_heads);
    const ropeTheta = cfg.rope_theta ?? 10000.0;
    const maxSeqLen = cfg.max_position_embeddings ?? 2048;
    const rmsNormEps = cfg.rms_norm_eps ?? 1e-5;

    // 1. Token Embeddings
    const embedWeight = reader.readTensor('model.embed_tokens.weight');

    // 2. Transformer Blocks
    const layers: TransformerBlock[] = [];

    // Shared RoPE instance for all attention heads
    const rope = new RotaryEmbedding(headDim, maxSeqLen, ropeTheta);

    for (let i = 0; i < cfg.num_hidden_layers; i++) {
      const prefix = `model.layers.${i}`;

      // Input RMSNorm
      const inputNormWeight = reader.readTensor(`${prefix}.input_layernorm.weight`);
      const inputLayernorm = new RMSNorm(inputNormWeight, rmsNormEps);

      // Attention projections
      const qWeight = reader.readTensor(`${prefix}.self_attn.q_proj.weight`);
      const kWeight = reader.readTensor(`${prefix}.self_attn.k_proj.weight`);
      const vWeight = reader.readTensor(`${prefix}.self_attn.v_proj.weight`);
      const oWeight = reader.readTensor(`${prefix}.self_attn.o_proj.weight`);

      const qBias = reader.hasTensor(`${prefix}.self_attn.q_proj.bias`)
        ? reader.readTensor(`${prefix}.self_attn.q_proj.bias`)
        : undefined;
      const kBias = reader.hasTensor(`${prefix}.self_attn.k_proj.bias`)
        ? reader.readTensor(`${prefix}.self_attn.k_proj.bias`)
        : undefined;
      const vBias = reader.hasTensor(`${prefix}.self_attn.v_proj.bias`)
        ? reader.readTensor(`${prefix}.self_attn.v_proj.bias`)
        : undefined;
      const oBias = reader.hasTensor(`${prefix}.self_attn.o_proj.bias`)
        ? reader.readTensor(`${prefix}.self_attn.o_proj.bias`)
        : undefined;

      const qProj = new Linear(qWeight, qBias);
      const kProj = new Linear(kWeight, kBias);
      const vProj = new Linear(vWeight, vBias);
      const oProj = new Linear(oWeight, oBias);

      const selfAttn = new MultiHeadAttention(
        qProj,
        kProj,
        vProj,
        oProj,
        rope,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        headDim
      );

      // Post-attention RMSNorm
      const postNormWeight = reader.readTensor(`${prefix}.post_attention_layernorm.weight`);
      const postAttentionLayernorm = new RMSNorm(postNormWeight, rmsNormEps);

      // SwiGLU MLP
      const gateWeight = reader.readTensor(`${prefix}.mlp.gate_proj.weight`);
      const upWeight = reader.readTensor(`${prefix}.mlp.up_proj.weight`);
      const downWeight = reader.readTensor(`${prefix}.mlp.down_proj.weight`);

      const gateBias = reader.hasTensor(`${prefix}.mlp.gate_proj.bias`)
        ? reader.readTensor(`${prefix}.mlp.gate_proj.bias`)
        : undefined;
      const upBias = reader.hasTensor(`${prefix}.mlp.up_proj.bias`)
        ? reader.readTensor(`${prefix}.mlp.up_proj.bias`)
        : undefined;
      const downBias = reader.hasTensor(`${prefix}.mlp.down_proj.bias`)
        ? reader.readTensor(`${prefix}.mlp.down_proj.bias`)
        : undefined;

      const mlp = new SwiGLUMLP(
        new Linear(gateWeight, gateBias),
        new Linear(upWeight, upBias),
        new Linear(downWeight, downBias)
      );

      layers.push(
        new TransformerBlock(i, inputLayernorm, selfAttn, postAttentionLayernorm, mlp)
      );
    }

    // 3. Final RMSNorm
    const finalNormWeight = reader.readTensor('model.norm.weight');
    const norm = new RMSNorm(finalNormWeight, rmsNormEps);

    // 4. LM Head (Check if tied or dedicated)
    let lmHead: Linear;
    if (reader.hasTensor('lm_head.weight')) {
      const lmHeadWeight = reader.readTensor('lm_head.weight');
      lmHead = new Linear(lmHeadWeight);
    } else {
      // Tied word embeddings: reuse embed_tokens weight
      lmHead = new Linear(embedWeight);
    }

    return new TransformerModel(cfg, embedWeight, layers, norm, lmHead);
  }
}

export class ModelLoader {
  /**
   * Loads config.json and model.safetensors from a model directory.
   */
  public static loadFromDirectory(modelDir: string): {
    model: TransformerModel;
    config: LlamaConfig;
  } {
    const configPath = path.join(modelDir, 'config.json');
    const weightsPath = path.join(modelDir, 'model.safetensors');

    if (!fs.existsSync(configPath)) {
      throw new Error(`config.json not found in model directory: ${modelDir}`);
    }
    if (!fs.existsSync(weightsPath)) {
      throw new Error(`model.safetensors not found in model directory: ${modelDir}`);
    }

    const configRaw = fs.readFileSync(configPath, 'utf-8');
    const config = JSON.parse(configRaw) as LlamaConfig;

    const reader = new SafeTensorsReader(weightsPath);
    const builder = new ModelBuilder();
    const model = builder.setConfig(config).setWeightReader(reader).build();

    return { model, config };
  }
}

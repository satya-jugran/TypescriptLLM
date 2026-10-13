/**
 * CLI Demo Entrypoint for TypescriptLLM Inference Engine.
 * Loads the model, configures hyperparameters, runs inference with streaming, and outputs telemetry.
 */

import * as fs from 'fs';
import * as path from 'path';
import { InferenceEngine } from './engine/inference-engine.js';
import { downloadModel } from './scripts/download-model.js';

function parseArgs(): {
  modelDir: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
} {
  const args = process.argv.slice(2);
  const getArg = (flag: string, fallback: string): string => {
    const idx = args.indexOf(flag);
    return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : fallback;
  };

  return {
    modelDir: getArg('--model-dir', path.resolve(process.cwd(), 'models', 'SmolLM-135M')),
    prompt: getArg(
      '--prompt',
      'The key principles of computer systems architecture are'
    ),
    maxTokens: parseInt(getArg('--max-tokens', '48'), 10),
    temperature: parseFloat(getArg('--temperature', '0.7')),
    topP: parseFloat(getArg('--top-p', '0.9')),
    topK: parseInt(getArg('--top-k', '50'), 10),
    repetitionPenalty: parseFloat(getArg('--rep-penalty', '1.1')),
  };
}

async function main(): Promise<void> {
  const options = parseArgs();

  console.log('='.repeat(70));
  console.log('       TypescriptLLM - Pure TypeScript Transformer Inference Engine');
  console.log('                 (Zero-GPU / CPU Optimized)');
  console.log('='.repeat(70));

  // Verify model existence, download automatically if missing
  const weightsFile = path.join(options.modelDir, 'model.safetensors');
  const configFile = path.join(options.modelDir, 'config.json');

  if (!fs.existsSync(weightsFile) || !fs.existsSync(configFile)) {
    console.log(`\nModel not found at: ${options.modelDir}`);
    console.log('Automatically downloading HuggingFaceTB/SmolLM-135M (~538 MB)...');
    await downloadModel('HuggingFaceTB/SmolLM-135M', options.modelDir);
  }

  console.log(`\n[1/3] Loading weights and initializing architecture from: ${options.modelDir}`);
  const loadStart = performance.now();
  const engine = await InferenceEngine.fromDirectory(options.modelDir);
  const loadEnd = performance.now();
  console.log(`✓ Model loaded successfully in ${((loadEnd - loadStart) / 1000).toFixed(2)}s`);

  const cfg = engine.config;
  console.log('\n--- Model Architecture ---');
  console.log(`  Layers:          ${cfg.num_hidden_layers}`);
  console.log(`  Hidden Size:     ${cfg.hidden_size}`);
  console.log(`  Intermediate:    ${cfg.intermediate_size}`);
  console.log(`  Attention Heads: ${cfg.num_attention_heads} (Query) / ${cfg.num_key_value_heads} (KV Head GQA)`);
  console.log(`  Head Dimension:  ${cfg.head_dim ?? cfg.hidden_size / cfg.num_attention_heads}`);
  console.log(`  Vocab Size:      ${cfg.vocab_size}`);
  console.log(`  Context Limit:   ${cfg.max_position_embeddings}`);

  console.log('\n--- Generation Configuration ---');
  console.log(`  Prompt:          "${options.prompt}"`);
  console.log(`  Max Tokens:      ${options.maxTokens}`);
  console.log(`  Temperature:     ${options.temperature}`);
  console.log(`  Top-P:           ${options.topP}`);
  console.log(`  Top-K:           ${options.topK}`);
  console.log(`  Rep. Penalty:    ${options.repetitionPenalty}`);

  console.log('\n[2/3] Generating text (streaming tokens in real-time):\n');
  process.stdout.write(`\x1b[36m${options.prompt}\x1b[0m`);

  const result = await engine.generate({
    prompt: options.prompt,
    maxTokens: options.maxTokens,
    temperature: options.temperature,
    topP: options.topP,
    topK: options.topK,
    repetitionPenalty: options.repetitionPenalty,
    onToken: (tokenStr: string) => {
      process.stdout.write(tokenStr);
    },
  });

  console.log('\n\n' + '='.repeat(70));
  console.log('[3/3] Inference Telemetry & Performance Metrics');
  console.log('='.repeat(70));
  console.log(`  Prompt Tokens:      ${result.promptTokensCount}`);
  console.log(`  Tokens Generated:   ${result.tokensGenerated}`);
  console.log(`  Prefill Latency:    ${result.prefillTimeMs} ms (${(result.prefillTimeMs / result.promptTokensCount).toFixed(2)} ms/token)`);
  console.log(`  Decode Latency:     ${result.decodeTimeMs} ms`);
  console.log(`  Decode Throughput:  \x1b[32m${result.tokensPerSecond} tokens/sec\x1b[0m`);
  console.log(`  Total Engine Time:  ${result.totalTimeMs} ms`);
  console.log('='.repeat(70));
}

main().catch((err) => {
  console.error('\nFatal engine error:', err);
  process.exit(1);
});

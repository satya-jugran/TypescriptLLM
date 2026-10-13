/**
 * Hugging Face Tokenizer Adapter.
 * Adapts @xenova/transformers AutoTokenizer to the ITokenizer interface.
 */

import path from 'path';
import fs from 'fs';
import { ITokenizer } from './tokenizer.interface.js';

// Interface matching @xenova/transformers tokenizer instance
interface XenovaTokenizerInstance {
  (text: string): { input_ids: { data: ArrayLike<number | bigint> } };
  decode(tokens: number[], options?: Record<string, unknown>): string;
  eos_token_id?: number;
  bos_token_id?: number;
}

export class HFTokenizer implements ITokenizer {
  private readonly tokenizer: XenovaTokenizerInstance;
  public readonly eosTokenId: number;
  public readonly bosTokenId: number;

  constructor(
    tokenizer: XenovaTokenizerInstance,
    eosTokenId = 0,
    bosTokenId = 0
  ) {
    this.tokenizer = tokenizer;
    this.eosTokenId = tokenizer.eos_token_id ?? eosTokenId;
    this.bosTokenId = tokenizer.bos_token_id ?? bosTokenId;
  }

  /**
   * Factory method to load AutoTokenizer from a local directory or Hugging Face Hub.
   */
  public static async fromPretrained(
    modelPathOrId: string,
    options?: { eosTokenId?: number; bosTokenId?: number }
  ): Promise<HFTokenizer> {
    const { AutoTokenizer, env } = await import('@xenova/transformers');

    const isLocalDir = fs.existsSync(modelPathOrId) && fs.statSync(modelPathOrId).isDirectory();

    if (isLocalDir) {
      env.allowLocalModels = true;
      const absoluteDir = path.resolve(modelPathOrId);
      // In @xenova/transformers, local paths are resolved relative to env.localModelPath
      env.localModelPath = path.dirname(absoluteDir);
      const subDir = path.basename(absoluteDir);

      const rawTokenizer = (await AutoTokenizer.from_pretrained(subDir, {
        local_files_only: true,
      })) as unknown as XenovaTokenizerInstance;

      return new HFTokenizer(rawTokenizer, options?.eosTokenId, options?.bosTokenId);
    } else {
      env.allowRemoteModels = true;
      const rawTokenizer = (await AutoTokenizer.from_pretrained(
        modelPathOrId
      )) as unknown as XenovaTokenizerInstance;

      return new HFTokenizer(rawTokenizer, options?.eosTokenId, options?.bosTokenId);
    }
  }

  /**
   * Encodes a prompt string into token IDs.
   */
  public async encode(text: string): Promise<number[]> {
    const output = this.tokenizer(text);
    const rawData = output.input_ids.data;
    const tokens = new Array<number>(rawData.length);
    for (let i = 0; i < rawData.length; i++) {
      tokens[i] = Number(rawData[i]);
    }
    return tokens;
  }

  /**
   * Decodes token IDs into readable text.
   */
  public async decode(tokens: number[] | readonly number[]): Promise<string> {
    if (tokens.length === 0) return '';
    const numericTokens = Array.from(tokens).map(Number);
    return this.tokenizer.decode(numericTokens, {
      skip_special_tokens: false,
    });
  }
}

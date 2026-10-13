/**
 * Tokenizer Interface.
 * Standardizes token encoding, decoding, and special token properties.
 */

export interface ITokenizer {
  /**
   * Encodes a raw text string into token IDs.
   */
  encode(text: string): Promise<number[]>;

  /**
   * Decodes an array of token IDs back into human-readable text.
   */
  decode(tokens: number[] | readonly number[]): Promise<string>;

  /**
   * End of Sequence (EOS) token ID.
   */
  readonly eosTokenId: number;

  /**
   * Beginning of Sequence (BOS) token ID.
   */
  readonly bosTokenId: number;
}

/**
 * CharTokenizer.ts — Character-level encoder/decoder.
 *
 * The simplest possible tokenization: every distinct character in the
 * training corpus becomes one vocabulary entry (ids sorted by code point,
 * so the mapping is deterministic given the same corpus).
 *
 * Why characters instead of words/subwords? For pedagogy: the vocabulary is
 * tiny (Shakespeare ≈ 65 symbols), every downstream tensor is small, and the
 * model must learn spelling, grammar and style purely from composition.
 * Trade-off, of course, is that a character model needs more compute to
 * express the same text length — the whole reason real GPTs use BPE.
 */

export class CharTokenizer {
  /** All vocabulary characters, sorted (id == index). */
  readonly vocab: string[];
  private charToId: Map<string, number>;
  /** Length of the longest text this tokenizer can 'encode' — informational. */
  readonly corpus: string;

  constructor(corpus: string) {
    this.corpus = corpus;
    this.vocab = [...new Set([...corpus])].sort();
    this.charToId = new Map(this.vocab.map((ch, i) => [ch, i]));
  }

  get vocabSize(): number {
    return this.vocab.length;
  }

  /** String → array of token ids. Throws on characters outside the corpus. */
  encode(text: string): number[] {
    const out: number[] = [];
    for (const ch of text) {
      const id = this.charToId.get(ch);
      if (id === undefined) {
        throw new Error(
          `character ${JSON.stringify(ch)} not in vocabulary (corpus does not contain it)`,
        );
      }
      out.push(id);
    }
    return out;
  }

  /** Array of token ids → string. */
  decode(ids: number[]): string {
    const out: string[] = [];
    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= this.vocab.length) {
        throw new Error(`token id ${id} out of range [0, ${this.vocab.length})`);
      }
      out.push(this.vocab[id]);
    }
    return out.join("");
  }
}

/**
 * index.ts — Public API of the mini-GPT library.
 */

export { Tensor, numel, type Shape } from "./autograd/Tensor";
export {
  applyCausalMask,
  crossEntropyPerExample,
} from "./autograd/Operations";

export { Module, param, type Parameter } from "./nn/Module";
export { Linear } from "./nn/Linear";
export { LayerNorm } from "./nn/LayerNorm";
export { TokenEmbedding, PositionEmbedding } from "./nn/Embedding";
export { MultiHeadAttention } from "./nn/Attention";
export { FeedForward } from "./nn/FeedForward";
export { TransformerBlock } from "./nn/TransformerBlock";
export { GPT, type GPTConfig } from "./nn/GPT";

export { AdamW } from "./optim/AdamW";
export { CharTokenizer } from "./data/CharTokenizer";
export { CrossEntropyLoss } from "./train/Loss";
export { Trainer, type TrainerConfig } from "./train/Trainer";

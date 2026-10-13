/**
 * Sampler Factory implementing the Factory Pattern.
 */

import { ISampler } from './sampler.interface.js';
import { GreedySampler } from './greedy-sampler.js';
import { NucleusSampler } from './nucleus-sampler.js';
import { SamplerConfig } from '../core/types.js';

export class SamplerFactory {
  /**
   * Instantiates the appropriate ISampler implementation based on configuration.
   */
  public static create(config?: SamplerConfig): ISampler {
    if (!config) {
      return new GreedySampler();
    }

    const { type, temperature = 0.7, topK = 50, topP = 0.9, repetitionPenalty = 1.1 } = config;

    // Zero temperature or explicit greedy type selects argmax
    if (type === 'greedy' || temperature <= 0.0) {
      return new GreedySampler(repetitionPenalty);
    }

    return new NucleusSampler(temperature, topP, topK, repetitionPenalty);
  }
}

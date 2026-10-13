/**
 * SafeTensors Binary Reader with on-the-fly BF16, F16, and F32 conversion.
 * Conforms to the SafeTensors specification: https://github.com/huggingface/safetensors
 */

import * as fs from 'fs';
import { Tensor } from '../core/tensor.js';
import { SafeTensorMetadata, SafeTensorsHeader } from '../core/types.js';

export interface IWeightReader {
  readTensor(name: string): Tensor;
  hasTensor(name: string): boolean;
  getTensorNames(): string[];
}

export class SafeTensorsReader implements IWeightReader {
  private readonly fileBuffer: Buffer;
  private readonly header: SafeTensorsHeader;
  private readonly dataStartOffset: number;
  private readonly tensorNames: string[];

  constructor(filePathOrBuffer: string | Buffer) {
    if (typeof filePathOrBuffer === 'string') {
      if (!fs.existsSync(filePathOrBuffer)) {
        throw new Error(`SafeTensors file not found: ${filePathOrBuffer}`);
      }
      this.fileBuffer = fs.readFileSync(filePathOrBuffer);
    } else {
      this.fileBuffer = filePathOrBuffer;
    }

    // 1. Read 8-byte little-endian header size
    if (this.fileBuffer.length < 8) {
      throw new Error('Invalid SafeTensors file: too small to contain 8-byte header size');
    }

    const headerLength = Number(this.fileBuffer.readBigUInt64LE(0));
    this.dataStartOffset = 8 + headerLength;

    if (this.fileBuffer.length < this.dataStartOffset) {
      throw new Error(
        `SafeTensors corrupted: header claims length ${headerLength}, but file length is ${this.fileBuffer.length}`
      );
    }

    // 2. Parse UTF-8 JSON header
    const headerJsonStr = this.fileBuffer.subarray(8, this.dataStartOffset).toString('utf-8');
    this.header = JSON.parse(headerJsonStr) as SafeTensorsHeader;

    this.tensorNames = Object.keys(this.header).filter((key) => key !== '__metadata__');
  }

  public getTensorNames(): string[] {
    return [...this.tensorNames];
  }

  public hasTensor(name: string): boolean {
    return this.tensorNames.includes(name);
  }

  public getMetadata(name: string): SafeTensorMetadata {
    const meta = this.header[name];
    if (!meta || 'data_offsets' in meta === false) {
      throw new Error(`Tensor "${name}" not found in SafeTensors header`);
    }
    return meta as SafeTensorMetadata;
  }

  /**
   * Reads a tensor by name and converts it into a Float32 Tensor instance.
   */
  public readTensor(name: string): Tensor {
    const meta = this.getMetadata(name);
    const [startRel, endRel] = meta.data_offsets;
    const start = this.dataStartOffset + startRel;
    const end = this.dataStartOffset + endRel;
    const byteLength = end - start;

    const slice = this.fileBuffer.subarray(start, end);
    const shape = meta.shape;
    const dtype = meta.dtype.toUpperCase();

    let f32Data: Float32Array;

    switch (dtype) {
      case 'F32': {
        const numFloats = byteLength / 4;
        f32Data = new Float32Array(numFloats);
        // Fast buffer copy ensuring 4-byte memory alignment
        const byteOffset = slice.byteOffset;
        const parentBuffer = slice.buffer;
        if (byteOffset % 4 === 0) {
          f32Data.set(new Float32Array(parentBuffer, byteOffset, numFloats));
        } else {
          // Unaligned fallback
          const tempBuffer = new Uint8Array(slice);
          f32Data.set(new Float32Array(tempBuffer.buffer, tempBuffer.byteOffset, numFloats));
        }
        break;
      }

      case 'BF16': {
        // BFloat16 to Float32 conversion on-the-fly:
        // In IEEE 754, BF16 simply occupies the top 16 bits of Float32.
        const numFloats = byteLength / 2;
        f32Data = new Float32Array(numFloats);
        const u32View = new Uint32Array(f32Data.buffer);

        // Convert 16-bit chunks
        for (let i = 0; i < numFloats; i++) {
          const bf16Val = slice.readUInt16LE(i * 2);
          u32View[i] = bf16Val << 16;
        }
        break;
      }

      case 'F16': {
        // IEEE 754 Float16 to Float32 conversion
        const numFloats = byteLength / 2;
        f32Data = new Float32Array(numFloats);
        for (let i = 0; i < numFloats; i++) {
          const f16Val = slice.readUInt16LE(i * 2);
          f32Data[i] = SafeTensorsReader.float16ToFloat32(f16Val);
        }
        break;
      }

      default:
        throw new Error(`Unsupported tensor dtype: ${dtype} for tensor ${name}`);
    }

    return new Tensor(shape, f32Data);
  }

  /**
   * Converts a 16-bit IEEE 754 half-precision float to 32-bit single-precision float.
   */
  public static float16ToFloat32(h: number): number {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;

    if (e === 0) {
      return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 0x1f) {
      return f ? NaN : (s ? -1 : 1) * Infinity;
    }
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
}

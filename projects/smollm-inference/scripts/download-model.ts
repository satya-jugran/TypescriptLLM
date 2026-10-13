/**
 * Model Downloader Script.
 * Downloads SmolLM-135M files (config.json, model.safetensors, tokenizer.json, tokenizer_config.json)
 * from Hugging Face Hub with streaming progress bar.
 */

import * as fs from 'fs';
import * as path from 'path';
import { Readable } from 'stream';

const DEFAULT_REPO = 'HuggingFaceTB/SmolLM-135M';
const DEFAULT_DIR = path.resolve(process.cwd(), 'models', 'SmolLM-135M');

const FILES_TO_DOWNLOAD = [
  'config.json',
  'tokenizer_config.json',
  'tokenizer.json',
  'model.safetensors',
];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function downloadFile(
  repoId: string,
  filename: string,
  targetDir: string
): Promise<void> {
  const targetPath = path.join(targetDir, filename);
  const fileUrl = `https://huggingface.co/${repoId}/resolve/main/${filename}`;

  console.log(`\nChecking: ${filename}`);

  const headResponse = await fetch(fileUrl, { method: 'HEAD', redirect: 'follow' });
  if (!headResponse.ok) {
    throw new Error(
      `Failed to retrieve headers for ${filename}: HTTP ${headResponse.status} ${headResponse.statusText}`
    );
  }

  const contentLength = Number(headResponse.headers.get('content-length') || 0);

  // Skip if already downloaded and size matches
  if (fs.existsSync(targetPath)) {
    const stats = fs.statSync(targetPath);
    if (contentLength > 0 && stats.size === contentLength) {
      console.log(`✓ Already exists (${formatBytes(stats.size)}), skipping.`);
      return;
    }
  }

  console.log(`Downloading ${filename} (${formatBytes(contentLength)})...`);

  const response = await fetch(fileUrl, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to download ${filename}: HTTP ${response.status} ${response.statusText}`
    );
  }

  const fileStream = fs.createWriteStream(targetPath);
  // Convert web ReadableStream to Node.js stream
  const nodeReadable = Readable.fromWeb(response.body as unknown as Parameters<typeof Readable.fromWeb>[0]);

  let downloadedBytes = 0;
  const startTime = Date.now();
  let lastReportTime = startTime;

  await new Promise<void>((resolve, reject) => {
    nodeReadable.on('data', (chunk: Buffer) => {
      downloadedBytes += chunk.length;
      const now = Date.now();
      if (now - lastReportTime > 500 || downloadedBytes === contentLength) {
        lastReportTime = now;
        const percent = contentLength > 0 ? ((downloadedBytes / contentLength) * 100).toFixed(1) : '?';
        const elapsedSec = (now - startTime) / 1000;
        const speed = elapsedSec > 0 ? downloadedBytes / elapsedSec : 0;
        process.stdout.write(
          `\r[${percent}%] ${formatBytes(downloadedBytes)} / ${formatBytes(contentLength)} (${formatBytes(speed)}/s)   `
        );
      }
    });

    nodeReadable.pipe(fileStream);

    fileStream.on('finish', () => {
      process.stdout.write('\n');
      console.log(`✓ Finished ${filename}`);
      resolve();
    });

    nodeReadable.on('error', (err: Error) => {
      fileStream.close();
      reject(err);
    });

    fileStream.on('error', (err: Error) => {
      reject(err);
    });
  });
}

export async function downloadModel(
  repoId = DEFAULT_REPO,
  outputDir = DEFAULT_DIR
): Promise<void> {
  console.log('='.repeat(60));
  console.log(`Model Downloader: ${repoId}`);
  console.log(`Destination: ${outputDir}`);
  console.log('='.repeat(60));

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  for (const file of FILES_TO_DOWNLOAD) {
    await downloadFile(repoId, file, outputDir);
  }

  console.log('\n' + '='.repeat(60));
  console.log(`Model files are ready at: ${outputDir}`);
  console.log('='.repeat(60));
}

// Direct execution from CLI
const isDirectRun =
  process.argv[1] &&
  (process.argv[1].endsWith('download-model.ts') || process.argv[1].endsWith('download-model.js'));

if (isDirectRun) {
  const repo = process.argv[2] || DEFAULT_REPO;
  const target = process.argv[3] || DEFAULT_DIR;

  downloadModel(repo, target).catch((err: unknown) => {
    console.error('\nDownload failed:', err);
    process.exit(1);
  });
}

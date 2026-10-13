/**
 * Lightweight Zero-Dependency Static HTTP Server for the Interactive Guide.
 */

import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';

const PORT = 3000;
const GUIDE_DIR = path.resolve(process.cwd(), 'guide');

const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

const server = http.createServer((req, res) => {
  const reqUrl = req.url?.split('?')[0] || '/';
  const safePath = reqUrl === '/' ? '/index.html' : reqUrl;
  const filePath = path.join(GUIDE_DIR, safePath);

  if (!filePath.startsWith(GUIDE_DIR) || !fs.existsSync(filePath)) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found');
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  res.writeHead(200, { 'Content-Type': contentType });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log(`🧠 TypescriptLLM Interactive Guide is live!`);
  console.log(`👉 Open in your browser: http://localhost:${PORT}`);
  console.log('='.repeat(60));
});

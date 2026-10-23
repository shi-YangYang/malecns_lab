import path from 'node:path';
import { open, realpath } from 'node:fs/promises';

export const SIMULATION_URL = 'http://127.0.0.1:6081/vnc.html?autoconnect=true&resize=scale';
export const MAX_JSON_BYTES = 256 * 1024;
export const MAX_MANIFEST_BYTES = 1024 * 1024;
export const MAX_ARCHIVE_BYTES = 16 * 1024 * 1024;
export const TERMINAL_STATES = new Set(['completed', 'cancelled', 'failed']);
export const STATES = new Set(['preparing', 'awaiting_simulation', 'running', 'stopping', ...TERMINAL_STATES]);
export function simulationRequestAllowed(value) {
  try {
    const url = new URL(value);
    return ['http:', 'ws:'].includes(url.protocol) && url.hostname === '127.0.0.1' && url.port === '6081' && !url.username && !url.password;
  } catch { return false; }
}
export function trustedFrame(event, contents) { return Boolean(contents && event.sender === contents && event.senderFrame === contents.mainFrame && event.senderFrame?.url === 'workbench://app/index.html'); }
export function recordIdAllowed(id) { return typeof id === 'string' && /^desktop-\d{8}-\d{6}-[a-f0-9]{8}$/.test(id); }
export function boundedRectangle(value, width, height) {
  if (!value || !['x', 'y', 'width', 'height'].every(key => typeof value[key] === 'number' && Number.isFinite(value[key]))) throw new Error('仿真视图位置无效');
  const x = Math.max(0, Math.min(width, Math.round(value.x)));
  const y = Math.max(62, Math.min(height, Math.round(value.y)));
  return { x, y, width: Math.max(0, Math.min(width - x, Math.round(value.width))), height: Math.max(0, Math.min(height - y, Math.round(value.height))) };
}
export async function readBoundedJson(root, relative, limit = MAX_JSON_BYTES) {
  if (typeof relative !== 'string' || relative.startsWith('/') || relative.split('/').some(part => !part || part === '.' || part === '..')) throw new Error('记录路径无效');
  const canonicalRoot = await realpath(root);
  const canonicalFile = await realpath(path.join(root, relative));
  if (!canonicalFile.startsWith(`${canonicalRoot}${path.sep}`)) throw new Error('记录越出允许目录');
  const handle = await open(canonicalFile, 'r');
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size > limit) throw new Error('记录超过读取上限');
    // Fixed-size reads also bound a file that grows after stat().
    const buffer = Buffer.alloc(limit + 1);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    if (bytesRead > limit) throw new Error('记录超过读取上限');
    return JSON.parse(buffer.subarray(0, bytesRead).toString('utf8'));
  } finally { await handle.close(); }
}
export class JsonLineReader {
  buffer = '';
  constructor(onLine, onError, limit = MAX_JSON_BYTES) { this.onLine = onLine; this.onError = onError; this.limit = limit; }
  push(chunk) {
    this.buffer += chunk;
    if (Buffer.byteLength(this.buffer) > this.limit && !this.buffer.includes('\n')) { this.buffer = ''; this.onError(new Error('实验输出超过上限')); return; }
    let end;
    while ((end = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, end); this.buffer = this.buffer.slice(end + 1);
      if (!line.trim()) continue;
      if (Buffer.byteLength(line) > this.limit) { this.onError(new Error('实验输出超过上限')); continue; }
      try { this.onLine(JSON.parse(line)); } catch (error) { this.onError(error); }
    }
    if (Buffer.byteLength(this.buffer) > this.limit) { this.buffer = ''; this.onError(new Error('实验输出超过上限')); }
  }
}

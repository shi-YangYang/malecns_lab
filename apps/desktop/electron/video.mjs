import { createHash, randomBytes } from 'node:crypto';
import { constants, unlinkSync } from 'node:fs';
import { mkdir, open, lstat, realpath, statfs, link, unlink } from 'node:fs/promises';
import path from 'node:path';
import { captureIdentity, safeFile, MEDIA_LIMITS_PROFILE } from './recording.mjs';
import { SIMULATION_URL, recordIdAllowed, MAX_MANIFEST_BYTES } from './boundaries.mjs';

const LEGACY_VIDEO_LIMITS = Object.freeze({ fps: 15, bitrate: 2500000, mime: 'video/webm;codecs=vp8', slice_ms: 1000, duration_ms: 240000, chunk_bytes: 8 * 1024 * 1024, pending_bytes: 16 * 1024 * 1024, file_bytes: 256 * 1024 * 1024, index_bytes: 4 * 1024 * 1024, free_bytes: 512 * 1024 * 1024, ack_ms: 1500 });
export const VIDEO_LIMITS = Object.freeze({ ...LEGACY_VIDEO_LIMITS, duration_ms: 480000 });
export const NOVNC_HASHES = Object.freeze({ 'vnc.html': '720ee6e3090aa60863e8133d5ccc8f3aa3ca2a26513855a0fc3ff42f5da754ae', 'app/ui.js': '49dd68cd630ba75cca74602b8950ed84d14da39458819387c0ef76880d4c9172', 'core/rfb.js': '708f8abd828b5b45c0cfdf101c51c39506a4210e2c5af5dd605493e67ac7c9cd', 'core/display.js': '768bb889ffbfb8c3db327be0c7b9b6039f80b59bd192f4424aeaa358905678de' });
const PREFIX = 'malecns_lab.x2_simulation_video.';
const SOURCES = ['video.mjs', 'simulation-preload.cjs', 'main.mjs', 'controller.mjs', 'recording.mjs', 'boundaries.mjs'];
const requireValue = (ok, message) => { if (!ok) throw new Error(message); };
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const encode = value => Buffer.from(JSON.stringify(value));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const time = v => Number.isFinite(v) && v >= 0;
const ns = v => typeof v === 'string' && /^\d{1,24}$/.test(v);
async function json(root, name) { return JSON.parse((await safeFile(root, name, VIDEO_LIMITS.index_bytes)).toString()); }
async function exists(root, name) { try { await lstat(path.join(root, name)); return true; } catch (e) { if (e.code === 'ENOENT') return false; throw e; } }
async function publish(root, name, bytes) {
  const temporary = path.join(root, `${name}.${randomBytes(8).toString('hex')}.partial`);
  const handle = await open(temporary, 'wx');
  try { await handle.writeFile(bytes); await handle.sync(); } finally { await handle.close(); }
  await link(temporary, path.join(root, name)); await unlink(temporary);
}
export async function verifyNoVNC(fetcher = fetch) {
  const deadline = Date.now() + 5000, result = {};
  for (const [name, expected] of Object.entries(NOVNC_HASHES)) {
    requireValue(Date.now() < deadline, 'noVNC 来源校验超时');
    const abort = new AbortController(), timer = setTimeout(() => abort.abort(), Math.min(2000, deadline - Date.now()));
    try {
      const response = await fetcher(`http://127.0.0.1:6081/${name}`, { redirect: 'error', signal: abort.signal, cache: 'no-store' });
      requireValue(response.ok && response.body, 'noVNC 来源无法读取');
      const reader = response.body.getReader(), hash = createHash('sha256'); let bytes = 0;
      try { for (;;) { const next = await reader.read(); if (next.done) break; bytes += next.value.byteLength; requireValue(bytes <= MAX_MANIFEST_BYTES && Date.now() < deadline, 'noVNC 来源超界或超时'); hash.update(next.value); } }
      finally { await reader.cancel(); }
      const actual = hash.digest('hex'); requireValue(actual === expected, `noVNC 来源不匹配：${name}`); result[name] = actual;
    } finally { clearTimeout(timer); }
  }
  return result;
}
export function trustedVideo(event, contents, envelope, authorization) {
  return Boolean(authorization && contents && event.sender === contents && event.senderFrame === contents.mainFrame && event.senderFrame?.url === SIMULATION_URL && envelope?.generation === authorization.generation && envelope?.token === authorization.token);
}

export class SimulationVideo {
  constructor({ root, recordId, sourceRoot, send, context, onFailure, generation, initialWindow = false, onLimit = null, clock = () => process.hrtime.bigint(), freeSpace = statfs, verifySource = verifyNoVNC, versions = process.versions }) {
    requireValue(!initialWindow || typeof onLimit === 'function', '首段视频需要统一封存回调');
    Object.assign(this, { root, recordId, sourceRoot, send, context, onFailure, generation, initialWindow, onLimit, clock, freeSpace, verifySource, versions });
    this.limitReason = null;
    this.folder = path.join(root, 'simulation_video'); this.token = randomBytes(32).toString('hex');
    this.state = 'initializing'; this.reason = null; this.finalized = false; this.closed = false; this.stopping = false; this.started = false; this.stopped = false; this.finalChunk = false; this.sequence = 0; this.bytes = 0; this.chunks = []; this.events = []; this.clocks = []; this.probes = new Map(); this.inFlight = null; this.hash = createHash('sha256');
  }
  get authorization() { return this.closed ? null : { generation: this.generation, token: this.token }; }
  status() { return { state: this.state, complete: this.state === 'sealed', reason: this.reason, bytes: this.bytes, ...(this.initialWindow ? { archive_scope: 'initial_window', truncation_reason: this.limitReason } : {}) }; }
  command(kind, values = {}) { this.send({ ...this.authorization, kind, ...values }); }
  checkContext() { const c = this.context(); requireValue(c.root === this.root && c.status.run_id === this.recordId && c.status.state === 'running' && !c.status.resources_stopped && c.visible && c.loaded && !c.loading && !c.error && !this.closed && !this.stopping, '视频采集条件已改变'); }
  async initialize() {
    const binding = await captureIdentity(this.root, this.recordId);
    const free = await this.freeSpace(this.root); requireValue(Number(free.bavail) * Number(free.bsize) >= VIDEO_LIMITS.free_bytes, '视频磁盘预留不足 512 MiB');
    const sources = Object.fromEntries(await Promise.all(SOURCES.map(async name => [name, sha(await safeFile(this.sourceRoot, name, MAX_MANIFEST_BYTES))])));
    // Create the durable incomplete marker even if source verification subsequently fails.
    await mkdir(this.folder); this.binding = binding;
    await publish(this.folder, 'seal_pending.json', encode({ identity: binding.identity, generation: this.generation }));
    const novnc = await this.verifySource(); this.checkContext();
    requireValue(same(binding, await captureIdentity(this.root, this.recordId)), '视频身份在准备期间改变');
    this.manifest = { schema: PREFIX + 'manifest.v1', ...binding, generation: this.generation, limits_profile: MEDIA_LIMITS_PROFILE, limits: VIDEO_LIMITS, source: 'novnc_visible_canvas', source_freshness: 'unknown', source_sha256: sources, novnc_sha256: novnc, versions: { electron: this.versions.electron, chrome: this.versions.chrome }, clock_domain: 'preload_performance_with_host_roundtrip', media_anchor: null, media_anchor_reason: 'first_encoded_frame_to_host_anchor_not_verified' };
    if (this.initialWindow) Object.assign(this.manifest, { archive_scope: 'initial_window', truncation_reason: 'bounded_initial_window_policy', complete_means: 'window_sealed_not_full_session' });
    await publish(this.folder, 'manifest.json', encode(this.manifest));
    this.file = await open(path.join(this.folder, 'recording.webm.partial'), 'wx');
    this.journal = await open(path.join(this.folder, 'events.jsonl.partial'), 'wx');
    this.checkContext(); this.state = 'recording';
    this.command('start', { limits_profile: MEDIA_LIMITS_PROFILE, limits: VIDEO_LIMITS, ...(this.initialWindow ? { archive_scope: 'initial_window' } : {}) });
    this.probe(); this.clockTimer = setInterval(() => this.probe(), 5000);
    this.startTimer = setTimeout(() => { if (!this.started) this.fail('视频编码器未确认启动'); }, 2000);
    this.durationTimer = setTimeout(() => this.limit('视频达到 480 秒上限'), VIDEO_LIMITS.duration_ms - (this.initialWindow ? 2000 : 0));
  }
  probe() { if (this.closed || this.probes.size >= 3 || this.clocks.length >= 100) return; const nonce = randomBytes(16).toString('hex'); this.probes.set(nonce, this.clock().toString()); this.command('clock', { nonce }); }
  receiveClock(value) {
    const sent = this.probes.get(value.nonce); requireValue(sent && time(value.performance_ms) && time(value.time_origin_ms) && typeof value.clock_id === 'string' && value.clock_id.length <= 80, '视频时钟应答无效'); this.probes.delete(value.nonce);
    if (this.clockId) requireValue(this.clockId === value.clock_id, '视频时钟身份改变'); this.clockId = value.clock_id;
    this.clocks.push({ nonce: value.nonce, host_send_ns: sent, host_receive_ns: this.clock().toString(), performance_ms: value.performance_ms, time_origin_ms: value.time_origin_ms, clock_id: value.clock_id }); return { accepted: true };
  }
  receive(value) {
    requireValue(!this.closed && this.file && !this.stopped, '视频已经终止');
    requireValue(!this.inFlight, '视频写入背压超限');
    requireValue(Number.isInteger(value.sequence) && value.sequence === this.sequence + 1 && value.sequence <= 1000, '视频块序号不连续');
    requireValue(['started', 'chunk', 'stopped'].includes(value.kind) && time(value.performance_ms) && time(value.event_timestamp_ms), '视频事件无效');
    requireValue(typeof value.clock_id === 'string' && value.clock_id.length <= 80 && (!this.clockId || this.clockId === value.clock_id), '视频时钟身份不匹配'); this.clockId = value.clock_id;
    // Snapshot accepted bytes before any asynchronous IO; never trust a mutable IPC view later.
    const data = value.kind === 'chunk' ? value.data : null;
    if (value.kind === 'chunk') requireValue(data instanceof Uint8Array && data.byteLength <= VIDEO_LIMITS.chunk_bytes && (data.byteLength > 0 || value.final === true), '视频块大小无效');
    const copy = data && Buffer.from(data);
    const received = this.clock().toString();
    const work = this.write(value, copy, received).catch(error => { this.fail(error); throw error; });
    this.inFlight = work.finally(() => { this.inFlight = null; if (this.closed) { for (const field of ['file', 'journal']) if (this[field]) { void this[field].close().catch(() => {}); this[field] = null; } } }); return this.inFlight;
  }
  async write(value, data, received) {
    const metadata = { sequence: value.sequence, kind: value.kind, performance_ms: value.performance_ms, event_timestamp_ms: value.event_timestamp_ms, clock_id: value.clock_id, host_receive_ns: received };
    if (value.kind === 'started') {
      requireValue(!this.started && value.sequence === 1 && !this.stopping && time(value.start_call_ms) && value.start_call_ms <= value.performance_ms && value.mime === VIDEO_LIMITS.mime && Number.isInteger(value.width) && value.width > 0 && value.width <= 1280 && Number.isInteger(value.height) && value.height > 0 && value.height <= 800, '视频启动信息无效');
      Object.assign(metadata, { start_call_ms: value.start_call_ms, width: value.width, height: value.height, mime: value.mime }); this.started = true; clearTimeout(this.startTimer);
    } else if (value.kind === 'chunk') {
      requireValue(this.started && !this.finalChunk && typeof value.final === 'boolean' && (value.timecode_ms === null || time(value.timecode_ms)), '视频末块或时间无效');
      requireValue(this.bytes + data.length <= VIDEO_LIMITS.file_bytes, '视频文件超过 256 MiB');
      const free = await this.freeSpace(this.root); requireValue(Number(free.bavail) * Number(free.bsize) >= VIDEO_LIMITS.free_bytes, '视频磁盘预留不足 512 MiB');
      requireValue(!this.closed, '视频已过封存期限');
      let written = 0; while (written < data.length) { const part = await this.file.write(data, written, data.length - written, this.bytes + written); requireValue(part.bytesWritten > 0, '视频写入无进展'); written += part.bytesWritten; }
      requireValue(!this.closed, '视频写入超过封存期限');
      Object.assign(metadata, { offset: this.bytes, bytes: data.length, sha256: sha(data), timecode_ms: value.timecode_ms, final: value.final, host_written_ns: this.clock().toString() }); this.bytes += data.length; this.hash.update(data); this.chunks.push(metadata); this.finalChunk = value.final;
    } else {
      requireValue(this.started && this.stopping && this.finalChunk && value.final_sequence === this.sequence && time(value.stop_call_ms) && value.stop_call_ms <= value.performance_ms, '视频缺少末块或停止确认');
      metadata.stop_call_ms = value.stop_call_ms; this.stopped = true;
    }
    this.events.push(metadata); requireValue(encode({ events: this.events, clocks: this.clocks }).length <= VIDEO_LIMITS.index_bytes, '视频索引超过 4 MiB');
    requireValue(!this.closed, '视频事件超过封存期限');
    await this.journal.writeFile(JSON.stringify(metadata) + '\n');
    requireValue(!this.closed, '视频事件超过封存期限'); this.sequence = value.sequence;
    // Stop early enough for the in-flight/queued/final chunks and stop metadata.
    // The callback only requests a stop; sealing must run outside this inFlight.
    if (this.initialWindow && !this.stopping) {
      if (this.bytes >= VIDEO_LIMITS.file_bytes - 3 * VIDEO_LIMITS.chunk_bytes) this.limit('视频首段达到文件容量预留边界');
      else if (this.events.length >= 996 || encode({ events: this.events, clocks: this.clocks }).length >= VIDEO_LIMITS.index_bytes - 64 * 1024) this.limit('视频首段达到索引容量预留边界');
    }
    return { accepted: true, sequence: value.sequence };
  }
  limit(reason) {
    if (!this.initialWindow) { this.fail(reason); return; }
    if (this.limitReason || this.stopping || this.closed) return;
    this.limitReason = reason; this.onLimit(reason);
  }
  fail(error) { if (this.reason || this.closed) return; this.reason = String(error?.message ?? error).slice(0, 1000); this.state = 'incomplete'; try { this.onFailure(this.reason); } finally { this.requestStop('video_failed'); } }
  requestStop(reason = 'control_ended') { if (this.stopping || this.closed) return; this.stopping = true; this.stopReason = reason; clearTimeout(this.durationTimer); clearTimeout(this.startTimer); clearInterval(this.clockTimer); this.probe(); this.command('stop', { reason }); }
  viewChanged(context) { if (this.state === 'recording' && !this.stopping && (!context.visible || !context.loaded || context.loading || context.error)) this.fail(context.error ?? (!context.visible ? '视频视图已隐藏' : '视频页面已断开或重载')); }
  revoke(reason) { if (this.closed) return; this.fail(reason); this.closed = true; }
  seal(deadline = Date.now() + 2000) {
    if (this.sealing) return this.sealing;
    this.requestStop();
    this.sealing = this.finalize(deadline).finally(() => { this.finalized = true; }); return this.sealing;
  }
  async finalize(deadline) {
    let timer;
    const expire = () => { this.reason ??= '视频封存超过共享 2 秒期限'; this.state = 'incomplete'; this.closed = true; };
    const work = (async () => {
      while (!this.stopped && !this.reason && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
      if (this.inFlight) await this.inFlight;
      requireValue(!this.closed && Date.now() < deadline && this.manifest && this.started && this.stopped && this.finalChunk && this.bytes > 0 && !this.reason, this.reason ?? '视频缺少完整停止确认');
      requireValue(same(this.binding, await captureIdentity(this.root, this.recordId)), '视频封存身份已改变');
      await this.file.sync(); await this.journal.sync(); await this.file.close(); await this.journal.close(); this.file = null; this.journal = null;
      const index = encode({ schema: PREFIX + 'index.v1', identity: this.binding.identity, events: this.events, clocks: this.clocks });
      requireValue(index.length <= VIDEO_LIMITS.index_bytes, '视频索引超界');
      await publish(this.folder, 'index.json', index);
      await link(path.join(this.folder, 'recording.webm.partial'), path.join(this.folder, 'recording.webm'));
      await publish(this.folder, 'summary.json', encode({ schema: PREFIX + 'summary.v1', identity: this.binding.identity, generation: this.generation, complete: true, state: 'sealed', bytes: this.bytes, sha256: this.hash.digest('hex'), index_sha256: sha(index), reason: this.stopReason, ended_host_ns: this.clock().toString(), ...(this.initialWindow ? { archive_scope: 'initial_window', truncation_reason: this.limitReason ?? this.stopReason, complete_means: 'window_sealed_not_full_session' } : {}) }));
      requireValue(!this.closed && Date.now() < deadline, '视频封存超过共享 2 秒期限');
      // Pending marker is authoritative, including when a late link publishes a summary.
      unlinkSync(path.join(this.folder, 'seal_pending.json')); this.state = 'sealed'; this.closed = true;
    })().catch(error => { this.reason ??= error.message; this.state = 'incomplete'; this.closed = true; });
    await Promise.race([work, new Promise(resolve => { timer = setTimeout(() => { expire(); resolve(); }, Math.max(0, deadline - Date.now())); })]); clearTimeout(timer);
    if (!this.inFlight) for (const field of ['file', 'journal']) if (this[field]) { void this[field].close().catch(() => {}); this[field] = null; }
    // Incomplete metadata is best-effort only; the durable pending marker always wins.
    if (this.state !== 'sealed' && this.binding) this.incompleteWrite = publish(this.folder, 'incomplete.json', encode({ identity: this.binding.identity, reason: this.reason, stop_reason: this.stopReason, bytes: this.bytes, events: this.events, clocks: this.clocks })).catch(() => {});
    return this.status();
  }
}

export async function readVideoArchive(root, id, session, telemetry) {
  if (!await exists(root, 'simulation_video')) return { state: 'absent', complete: false, manifest: null, media_url: null };
  const folder = path.join(root, 'simulation_video');
  if (await exists(folder, 'seal_pending.json')) return { state: 'incomplete', complete: false, manifest: null, media_url: null, reason: '视频未完成封存' };
  const manifest = await json(folder, 'manifest.json'), summary = await json(folder, 'summary.json');
  requireValue(recordIdAllowed(id) && session.run_id === id && session.resources_stopped && ['completed', 'failed', 'cancelled'].includes(session.state), '视频运行尚未结束');
  const binding = await captureIdentity(root, id);
  requireValue(manifest.schema === PREFIX + 'manifest.v1' && same(binding.identity, manifest.identity) && same(binding.inputs_sha256, manifest.inputs_sha256) && same(binding.official_assets_sha256, manifest.official_assets_sha256) && ['run_id','session_id','execution_id','checkpoint_id','policy_id','input_map_id'].every(k => telemetry[k] === manifest.identity[k]), '视频身份不匹配');
  const limits = manifest.limits_profile === MEDIA_LIMITS_PROFILE ? VIDEO_LIMITS
    : manifest.limits_profile === undefined ? LEGACY_VIDEO_LIMITS : null;
  requireValue(limits && same(manifest.limits, limits) && same(manifest.novnc_sha256, NOVNC_HASHES) && manifest.source === 'novnc_visible_canvas' && SOURCES.every(n => /^[a-f0-9]{64}$/.test(manifest.source_sha256?.[n])) && typeof manifest.versions?.electron === 'string' && typeof manifest.versions.chrome === 'string' && manifest.media_anchor === null, '视频来源或时间锚点无效');
  requireValue(summary.schema === PREFIX + 'summary.v1' && summary.complete === true && summary.state === 'sealed' && same(summary.identity, binding.identity) && summary.generation === manifest.generation && Number.isInteger(summary.bytes) && summary.bytes > 0 && summary.bytes <= limits.file_bytes && /^[a-f0-9]{64}$/.test(summary.sha256), '视频摘要无效');
  requireValue(manifest.archive_scope === undefined || (manifest.archive_scope === 'initial_window' && summary.archive_scope === 'initial_window' && manifest.complete_means === 'window_sealed_not_full_session' && summary.complete_means === manifest.complete_means && typeof summary.truncation_reason === 'string' && summary.truncation_reason.length > 0), '视频首段保存范围无效');
  const raw = await safeFile(folder, 'index.json', limits.index_bytes); requireValue(sha(raw) === summary.index_sha256, '视频索引完整性失败');
  const index = JSON.parse(raw.toString()); requireValue(index.schema === PREFIX + 'index.v1' && same(index.identity, binding.identity) && Array.isArray(index.events) && index.events.length >= 3 && index.events.length <= 1000 && Array.isArray(index.clocks) && index.clocks.length <= 100, '视频事件索引无效');
  let offset = 0, final = false, previous = -1, clockId = null;
  for (const [i, e] of index.events.entries()) {
    requireValue(e.sequence === i + 1 && time(e.performance_ms) && e.performance_ms >= previous && time(e.event_timestamp_ms) && ns(e.host_receive_ns) && typeof e.clock_id === 'string' && (!clockId || e.clock_id === clockId), '视频事件时钟或序号无效'); previous = e.performance_ms; clockId = e.clock_id;
    if (i === 0) requireValue(e.kind === 'started' && time(e.start_call_ms) && e.start_call_ms <= e.performance_ms, '视频开始事件缺失');
    else if (i === index.events.length - 1) requireValue(e.kind === 'stopped' && final && time(e.stop_call_ms), '视频停止事件缺失');
    else { requireValue(e.kind === 'chunk' && !final && e.offset === offset && Number.isInteger(e.bytes) && e.bytes >= 0 && e.bytes <= limits.chunk_bytes && /^[a-f0-9]{64}$/.test(e.sha256) && typeof e.final === 'boolean' && (e.timecode_ms === null || time(e.timecode_ms)) && ns(e.host_written_ns), '视频块索引无效'); offset += e.bytes; final = e.final; }
  }
  requireValue(offset === summary.bytes, '视频字节总量不匹配');
  for (const c of index.clocks) requireValue(ns(c.host_send_ns) && ns(c.host_receive_ns) && BigInt(c.host_receive_ns) >= BigInt(c.host_send_ns) && time(c.performance_ms) && time(c.time_origin_ms) && c.clock_id === clockId && /^[a-f0-9]{32}$/.test(c.nonce), '视频时钟往返证据无效');
  return { state: 'sealed', complete: true, manifest, summary, index, media_url: `workbench://app/media/${id}/recording.webm`, alignment: null };
}
export function videoRange(header, size) {
  if (header === null) return { start: 0, end: size - 1, partial: false };
  const match = /^bytes=(\d*)-(\d*)$/.exec(header); requireValue(match && (match[1] || match[2]), 'Range 无效');
  const suffix = !match[1]; const first = Number(suffix ? match[2] : match[1]); const last = match[2] ? Number(match[2]) : size - 1;
  requireValue(Number.isSafeInteger(first) && Number.isSafeInteger(last) && (!suffix || first > 0), 'Range 无效');
  const start = suffix ? Math.max(0, size - first) : first, end = suffix ? size - 1 : Math.min(size - 1, last);
  requireValue(start >= 0 && start < size && end >= start, 'Range 无效'); return { start, end, partial: true };
}
export async function videoResponse(request, root, archive) {
  requireValue(['GET','HEAD'].includes(request.method) && archive.complete && archive.state === 'sealed', '视频请求未授权');
  const folder = path.join(root, 'simulation_video'), file = path.join(folder, 'recording.webm');
  for (const p of [root, folder, file]) requireValue(await realpath(p) === p && !(await lstat(p)).isSymbolicLink(), '视频路径使用符号链接');
  const handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW); let transferred = false;
  try {
    const stats = await handle.stat(); requireValue(stats.isFile() && stats.size === archive.summary.bytes, '视频大小不匹配');
    // Stream-hash the same descriptor used below; memory is bounded to one encoded chunk.
    const whole = createHash('sha256');
    for (const chunk of archive.index.events.filter(e => e.kind === 'chunk')) {
      const buffer = Buffer.alloc(chunk.bytes); let read = 0;
      while (read < buffer.length) { const part = await handle.read(buffer, read, buffer.length - read, chunk.offset + read); requireValue(part.bytesRead > 0, '视频读取提前结束'); read += part.bytesRead; }
      requireValue(sha(buffer) === chunk.sha256, '视频块完整性失败'); whole.update(buffer);
    }
    requireValue(whole.digest('hex') === archive.summary.sha256, '视频完整性失败');
    let range; try { range = videoRange(request.headers.get('range'), stats.size); } catch { return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${stats.size}` } }); }
    const headers = { 'Content-Type': 'video/webm', 'Content-Length': String(range.end - range.start + 1), 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
    if (range.partial) headers['Content-Range'] = `bytes ${range.start}-${range.end}/${stats.size}`;
    if (request.method === 'HEAD') return new Response(null, { status: range.partial ? 206 : 200, headers });
    let position = range.start; transferred = true;
    // Recheck each encoded chunk before exposing bytes, closing the hash-to-stream mutation gap.
    let cachedChunk = null, cachedBytes = null;
    const body = new ReadableStream({
      async pull(controller) {
        try {
          if (position > range.end) { controller.close(); await handle.close(); return; }
          const chunk = archive.index.events.find(e => e.kind === 'chunk' && position >= e.offset && position < e.offset + e.bytes);
          requireValue(chunk, '视频范围缺块');
          if (cachedChunk !== chunk) { cachedBytes = Buffer.alloc(chunk.bytes); let count = 0; while (count < cachedBytes.length) { const r = await handle.read(cachedBytes, count, cachedBytes.length - count, chunk.offset + count); requireValue(r.bytesRead > 0, '视频块截断'); count += r.bytesRead; } requireValue(sha(cachedBytes) === chunk.sha256, '视频播放内容已改变'); cachedChunk = chunk; }
          const end = Math.min(position + 65536, range.end + 1, chunk.offset + chunk.bytes); controller.enqueue(Uint8Array.from(cachedBytes.subarray(position - chunk.offset, end - chunk.offset))); position = end;
        } catch (e) { controller.error(e); await handle.close(); }
      }, async cancel() { await handle.close(); },
    });
    return new Response(body, { status: range.partial ? 206 : 200, headers });
  } finally { if (!transferred) await handle.close(); }
}

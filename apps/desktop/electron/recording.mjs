import { createHash, randomBytes } from 'node:crypto';
import { constants, unlinkSync } from 'node:fs';
import { lstat, open, realpath, mkdir, link, unlink, statfs, readdir } from 'node:fs/promises';
import path from 'node:path';
import { MAX_MANIFEST_BYTES, recordIdAllowed } from './boundaries.mjs';

const PREFIX = 'malecns_lab.x2_simulation_capture.';
const SEAL_PROTOCOL = 'pending_marker_v1';
export const CAPTURE_LIMITS = Object.freeze({ interval_ms: 500, duration_ms: 240000, frames: 480, frame_bytes: 512 * 1024, image_bytes: 256 * 1024 * 1024, index_bytes: 4 * 1024 * 1024, free_bytes: 512 * 1024 * 1024, width: 1280, height: 800, jpeg_quality: 70, seal_ms: 2000, max_frame_age_ns: '1000000000' });
const ID_KEYS = ['record_id', 'run_id', 'session_id', 'execution_id', 'checkpoint_id', 'policy_id', 'input_map_id'];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const encoded = value => Buffer.from(JSON.stringify(value));
const ns = value => typeof value === 'string' && /^\d{1,24}$/.test(value);
const nonempty = value => typeof value === 'string' && value.length > 0 && value.length <= 256;
const requireValue = (test, message) => { if (!test) throw new Error(message); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sourceNames = ['recording.mjs', 'main.mjs', 'controller.mjs', 'boundaries.mjs', 'preload.cjs'];

// Resolve every component, including directories. Unlike realpath containment alone,
// this also refuses an in-root symlink. O_NOFOLLOW covers a last-component swap.
export async function safeFile(root, relative, limit) {
  requireValue(typeof relative === 'string' && relative.split('/').every(p => p && p !== '.' && p !== '..') && !path.isAbsolute(relative), '画面记录路径无效');
  const canonical = await realpath(root);
  requireValue(canonical === root && !(await lstat(root)).isSymbolicLink(), '画面目录使用符号链接');
  let target = root;
  for (const part of relative.split('/')) {
    target = path.join(target, part);
    requireValue(!(await lstat(target)).isSymbolicLink(), '画面记录不能使用符号链接');
  }
  requireValue(await realpath(target) === target, '画面记录路径越界');
  const handle = await open(target, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stats = await handle.stat();
    requireValue(stats.isFile() && stats.size <= limit, '画面记录超过读取上限');
    const buffer = Buffer.alloc(limit + 1);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    requireValue(bytesRead <= limit, '画面记录超过读取上限');
    return buffer.subarray(0, bytesRead);
  } finally { await handle.close(); }
}
async function json(root, name, limit = MAX_MANIFEST_BYTES) { return JSON.parse((await safeFile(root, name, limit)).toString('utf8')); }
async function optionalJson(root, name, limit) { try { return await json(root, name, limit); } catch (error) { if (error.code === 'ENOENT') return null; throw error; } }
async function exclusiveFile(root, relative, content, stillCurrent = () => true) {
  const target = path.join(root, relative), temporary = `${target}.${randomBytes(5).toString('hex')}.partial`;
  const handle = await open(temporary, 'wx');
  try { await handle.writeFile(content); await handle.sync(); } finally { await handle.close(); }
  // link is an atomic, no-overwrite publication, including on macOS.
  if (!stillCurrent()) return; // preserve a late partial without publishing it
  await link(temporary, target); await unlink(temporary);
}
function checkIdentity(value) {
  requireValue(value && ID_KEYS.every(key => nonempty(value[key])) && recordIdAllowed(value.record_id) && value.run_id === value.record_id && value.execution_id === value.session_id && value.backend === 'sim', '采样档案身份无效');
}
export async function captureIdentity(root, id) {
  requireValue(recordIdAllowed(id) && path.basename(root) === id, '采样运行标识无效');
  const registration = await json(root, 'behavior_registration.json');
  const ready = await json(root, 'brain_ready.json');
  const telemetry = await json(root, 'telemetry/manifest.json');
  const official = await json(root, 'official_assets.json');
  requireValue(registration.output === root && registration.run_id === id && registration.simulation_only === true && registration.qualification_only === true, '采样不是当前自有仿真验证');
  requireValue(telemetry.schema === 'malecns_lab.x2_behavior_telemetry_manifest.v1' && telemetry.clock_domain === 'host_worker_monotonic', '脑遥测不能联合对齐');
  for (const key of ['run_id', 'session_id']) requireValue(registration[key] === ready[key] && ready[key] === telemetry[key], '运行或神经会话身份不匹配');
  for (const key of ['checkpoint_id', 'policy_id', 'input_map_id']) requireValue(nonempty(ready[key]) && ready[key] === telemetry[key], '脑策略身份不匹配');
  requireValue(registration.checkpoint_metadata?.checkpoint_id === ready.checkpoint_id && telemetry.execution_id === ready.session_id, '神经执行身份不匹配');
  requireValue(official.sha256 && Object.keys(official.sha256).length > 0 && Object.values(official.sha256).every(v => /^[a-f0-9]{64}$/.test(v)), '官方资产身份缺失');
  const identity = Object.fromEntries(ID_KEYS.map(key => [key, key === 'record_id' ? id : key === 'execution_id' ? telemetry.execution_id : ready[key]]));
  identity.backend = 'sim'; checkIdentity(identity);
  return { identity, official_assets_sha256: official.sha256, inputs_sha256: Object.fromEntries(await Promise.all(['behavior_registration.json', 'brain_ready.json', 'telemetry/manifest.json', 'official_assets.json'].map(async name => [name, sha(await safeFile(root, name, MAX_MANIFEST_BYTES))]))) };
}
function contextOf(status, latest) {
  return { state: status.state, phase: status.phase ?? null, task: status.task ?? null, request_id: status.request_id ?? null, decision_sequence: status.decision_seq ?? null, skill_execution_id: latest?.skill_execution_id ?? null, skill_context_source: latest?.skill_execution_id ? 'last_available_neural_batch_input_feedback' : null };
}
function hostTime(value, unit) {
  requireValue(typeof value === 'number' && Number.isFinite(value) && value >= 0, '档案缺少可比较的主机时间');
  // Preserve original Number fields beside this explicitly approximate cursor.
  return BigInt(Math.round(unit === 's' ? value * 1e9 : value)).toString();
}
function sameRun(value, identity) {
  for (const key of ['run_id', 'session_id']) if (value?.[key] !== undefined) requireValue(value[key] === identity[key], '回放事件跨运行或会话');
}
function validateBatch(value, identity) {
  requireValue(value.schema === 'malecns_lab.x2_behavior_telemetry_batch.v1' && ['run_id','session_id','execution_id','checkpoint_id','input_map_id'].every(key => value[key] === identity[key]) && Number.isInteger(value.batch_sequence) && value.batch_sequence > 0, '回放批次身份不匹配');
}

export class SimulationCapture {
  constructor({ root, recordId, sourceRoot, capture, context, onFailure, clock = () => process.hrtime.bigint(), platform = process.platform, freeSpace = statfs }) {
    Object.assign(this, { root, recordId, sourceRoot, capture, context, onFailure, clock, platform, freeSpace });
    this.folder = path.join(root, 'simulation_capture'); this.generation = 1; this.accepting = false; this.inFlight = null; this.timer = null; this.frames = []; this.gaps = []; this.events = []; this.bytes = 0; this.indexBytes = 0; this.error = null; this.reason = null; this.gap = null; this.skipped = 0; this.state = 'initializing'; this.sealing = null; this.finalized = false;
  }
  async initialize() {
    requireValue(this.platform === 'darwin', '联合采样当前仅支持同机 macOS 单调时钟');
    const verified = await captureIdentity(this.root, this.recordId);
    const available = await this.freeSpace(this.root);
    requireValue(Number(available.bavail) * Number(available.bsize) >= CAPTURE_LIMITS.free_bytes, '采样磁盘预留不足 512 MiB');
    const sources = Object.fromEntries(await Promise.all(sourceNames.map(async name => [name, sha(await safeFile(this.sourceRoot, name, MAX_MANIFEST_BYTES))])));
    this.started = this.clock(); this.binding = verified;
    this.manifest = { schema: PREFIX + 'manifest.v1', ...verified, seal_protocol: SEAL_PROTOCOL, seal_generation: this.generation, media_kind: 'sampled_frames', source: 'novnc_webcontents_snapshot', limits: CAPTURE_LIMITS, source_sha256: sources, started_host_monotonic_ns: this.started.toString(), clock: { domain: 'macos_same_host_monotonic', node_source: 'process.hrtime.bigint', python_source: 'time.monotonic/time.monotonic_ns', comparability: 'same_host_macos_route; actual_run_comparison_pending', existing_number_precision: 'original JSON Number retained; cursor conversion approximate beyond safe integer', pixel_source_timestamp: null, claim: 'host_time_association_only' } };
    await mkdir(this.folder); await mkdir(path.join(this.folder, 'frames'));
    // Durable before any publishable manifest: a late summary alone is never a
    // completed seal. Timeout/crash leaves this marker for every future reader.
    await exclusiveFile(this.folder, 'seal_pending.json', encoded({ schema: PREFIX + 'pending.v1', identity: verified.identity, generation: this.generation }));
    await exclusiveFile(this.folder, 'manifest.json', encoded(this.manifest));
    await exclusiveFile(this.folder, 'frame_index.jsonl', Buffer.alloc(0));
    this.accepting = true; this.state = 'recording';
  }
  start() { this.deadlineTimer = setTimeout(() => this.fail('画面采样达到时长上限'), CAPTURE_LIMITS.duration_ms); const loop = async () => { await this.tick(); if (this.accepting) this.timer = setTimeout(loop, CAPTURE_LIMITS.interval_ms); }; void loop(); }
  status() { return { state: this.state, frames: this.frames.length, reason: this.error ?? this.reason, complete: this.state === 'sealed' && !this.error }; }
  freeze(reason) { this.accepting = false; this.reason ??= reason; clearTimeout(this.timer); clearTimeout(this.deadlineTimer); }
  fail(error) {
    if (this.error) return;
    this.error = error instanceof Error ? error.message : String(error); this.state = 'failed';
    // Stop the controller before any sealing, IO wait, or capture cancellation.
    try { this.onFailure(this.error); } finally { this.freeze('capture_failed'); }
  }
  viewChanged(state) {
    if (!this.accepting) return;
    if (!state.visible || !state.loaded || state.loading || state.error) {
      this.gap ??= { start_host_monotonic_ns: this.clock().toString(), end_host_monotonic_ns: null, reason: state.error ? 'load_error' : !state.visible ? 'view_hidden' : 'page_loading' };
    }
  }
  noteStatus(status) {
    if (!this.accepting) return;
    const event = { kind: 'desktop_status', host_monotonic_ns: this.clock().toString(), state: status.state, phase: status.phase ?? null, request_id: status.request_id ?? null, behavior: status.behavior ?? null, message: String(status.message ?? '').slice(0, 2000) };
    this.events.push(event);
    if (encoded(this.events).length > CAPTURE_LIMITS.index_bytes / 4) this.fail('采样状态索引达到上限');
  }
  async tick() {
    if (!this.accepting || this.inFlight) { if (this.inFlight) this.skipped++; return; }
    const token = this.generation;
    this.inFlight = this.sample(token).catch(error => { if (token === this.generation) this.fail(error); }).finally(() => { this.inFlight = null; });
    await this.inFlight;
  }
  async sample(token) {
    const now = this.clock();
    if (now - this.started >= BigInt(CAPTURE_LIMITS.duration_ms) * 1000000n || this.frames.length >= CAPTURE_LIMITS.frames) { this.fail('画面采样达到时长或帧数上限'); return; }
    const state = this.context();
    requireValue(state.root === this.root && state.status.run_id === this.binding.identity.run_id, '采样运行身份改变');
    if (state.status.state !== 'running' || state.status.resources_stopped) { this.freeze('control_ended'); return; }
    const verified = await captureIdentity(this.root, this.recordId);
    requireValue(same(verified, this.binding), '采样来源身份改变');
    if (!this.accepting || token !== this.generation) return;
    if (!state.visible || !state.loaded || state.loading || state.error) {
      const reason = state.error ? 'load_error' : !state.visible ? 'view_hidden' : 'page_loading';
      if (!this.gap) this.gap = { start_host_monotonic_ns: now.toString(), end_host_monotonic_ns: null, reason };
      this.skipped++; return;
    }
    const latest = await optionalJson(this.root, 'telemetry/latest.json');
    if (latest) validateBatch(latest, this.binding.identity);
    const start = this.clock();
    const image = await this.capture();
    if (!this.accepting || token !== this.generation) return;
    const finish = this.clock(), after = this.context();
    requireValue(after.root === this.root && after.status.run_id === this.binding.identity.run_id, '采样期间运行身份改变');
    if (!after.visible || !after.loaded || after.loading || after.error || after.status.state !== 'running') { this.gap ??= { start_host_monotonic_ns: start.toString(), end_host_monotonic_ns: null, reason: 'view_changed_during_capture' }; this.skipped++; return; }
    requireValue(image && !image.isEmpty(), '仿真采样得到空图');
    const size = image.getSize();
    requireValue(size.width > 0 && size.height > 0, '仿真采样尺寸无效');
    const scale = Math.min(1, CAPTURE_LIMITS.width / size.width, CAPTURE_LIMITS.height / size.height);
    const resized = scale < 1 ? image.resize({ width: Math.max(1, Math.floor(size.width * scale)), height: Math.max(1, Math.floor(size.height * scale)), quality: 'good' }) : image;
    const dimensions = resized.getSize(), jpeg = resized.toJPEG(CAPTURE_LIMITS.jpeg_quality);
    requireValue(Buffer.isBuffer(jpeg) && jpeg.length > 0 && jpeg.length <= CAPTURE_LIMITS.frame_bytes && dimensions.width <= CAPTURE_LIMITS.width && dimensions.height <= CAPTURE_LIMITS.height, '画面采样超过单帧上限');
    requireValue(this.bytes + jpeg.length <= CAPTURE_LIMITS.image_bytes, '画面采样达到图片总上限');
    const sequence = this.frames.length + 1;
    const frame = { sequence, capture_start_host_monotonic_ns: start.toString(), capture_end_host_monotonic_ns: finish.toString(), width: dimensions.width, height: dimensions.height, bytes: jpeg.length, sha256: sha(jpeg), context: contextOf(state.status, latest), page: { visible: state.visible, loaded: state.loaded, source_freshness: 'unknown_no_pixel_source_timestamp' } };
    const line = Buffer.from(JSON.stringify(frame) + '\n');
    requireValue(this.indexBytes + line.length <= CAPTURE_LIMITS.index_bytes / 2, '画面帧索引达到上限');
    if (!this.accepting || token !== this.generation) return;
    await exclusiveFile(this.folder, `frames/${String(sequence).padStart(6, '0')}.jpg`, jpeg);
    if (!this.accepting || token !== this.generation) return; // an orphan stays explicit, never enters a later run
    const handle = await open(path.join(this.folder, 'frame_index.jsonl'), constants.O_WRONLY | constants.O_APPEND | constants.O_NOFOLLOW);
    try { await handle.writeFile(line); } finally { await handle.close(); }
    if (token !== this.generation) return;
    this.frames.push(frame); this.bytes += jpeg.length; this.indexBytes += line.length;
    if (this.gap) { this.gap.end_host_monotonic_ns = finish.toString(); this.gaps.push(this.gap); this.gap = null; }
  }
  seal(reason = 'control_ended', deadline = Date.now() + CAPTURE_LIMITS.seal_ms) {
    if (this.sealing) return this.sealing;
    this.freeze(reason);
    this.sealing = this.finalize(deadline).finally(() => { this.finalized = true; }); return this.sealing;
  }
  async finalize(deadline) {
    const token = this.generation;
    let timedOut = false, timeout;
    const expire = () => { if (!timedOut) { timedOut = true; this.generation++; this.state = 'incomplete'; this.error ??= '画面封存超过 2 秒，保留未完成档案'; } };
    const work = (async () => {
      if (this.inFlight) await this.inFlight;
      if (this.gap) { this.gap.end_host_monotonic_ns = this.clock().toString(); this.gaps.push(this.gap); this.gap = null; }
      if (!this.manifest || timedOut) return;
      const replay = await buildReplay(this.root, this.binding.identity, this.frames, this.gaps, this.events);
      if (timedOut) return;
      requireValue(this.indexBytes + encoded(replay).length <= CAPTURE_LIMITS.index_bytes, '画面回放索引超过 4 MiB');
      await exclusiveFile(this.folder, 'replay_index.json', encoded(replay), () => !timedOut);
      if (timedOut) return;
      const complete = !this.error;
      const result = { schema: PREFIX + 'summary.v1', identity: this.binding.identity, seal_generation: token, state: complete ? 'sealed' : 'failed', complete, reason: this.error ?? this.reason, frame_count: this.frames.length, image_bytes: this.bytes, skipped: this.skipped, ended_host_monotonic_ns: this.clock().toString(), replay_sha256: sha(encoded(replay)), index_sha256: sha(await safeFile(this.folder, 'frame_index.jsonl', CAPTURE_LIMITS.index_bytes)) };
      if (timedOut) return;
      await exclusiveFile(this.folder, 'summary.json', encoded(result), () => !timedOut);
      if (timedOut || token !== this.generation) return;
      if (Date.now() >= deadline) { expire(); return; }
      // No asynchronous commit after this decision: otherwise the timer could
      // invalidate the generation while removal of the pending marker is in IO.
      // Only this single metadata unlink is synchronous; image/index IO is not.
      unlinkSync(path.join(this.folder, 'seal_pending.json'));
      this.state = result.state;
    })().catch(error => { if (!timedOut) { this.error ??= error.message; this.state = 'incomplete'; } });
    await Promise.race([work, new Promise(resolve => { timeout = setTimeout(() => { expire(); resolve(); }, Math.max(0, deadline - Date.now())); })]);
    clearTimeout(timeout);
    return this.status();
  }
}

async function buildReplay(root, identity, frames, gaps, desktopEvents) {
  const entries = [...desktopEvents];
  for (const frame of frames) entries.push({ kind: 'frame', host_monotonic_ns: frame.capture_end_host_monotonic_ns, sequence: frame.sequence });
  for (const gap of gaps) {
    entries.push({ kind: 'gap_start', host_monotonic_ns: gap.start_host_monotonic_ns });
    if (gap.end_host_monotonic_ns) entries.push({ kind: 'gap_end', host_monotonic_ns: gap.end_host_monotonic_ns });
  }
  const names = await readdir(path.join(root, 'telemetry/batches'));
  requireValue(names.length <= 4096 && names.every(name => /^\d{6}\.json$/.test(name)), '回放批次目录超界');
  for (const name of names.sort()) {
    const bytes = await safeFile(root, `telemetry/batches/${name}`, 256 * 1024);
    const batch = JSON.parse(bytes.toString('utf8')); validateBatch(batch, identity);
    requireValue(batch.batch_sequence === Number(name.slice(0, 6)), '回放批次文件序号不匹配');
    entries.push({ kind: 'batch', host_monotonic_ns: hostTime(batch.host_monotonic_ns, 'ns'), sequence: batch.batch_sequence, original_host_monotonic_ns: batch.host_monotonic_ns, bytes: bytes.length, sha256: sha(bytes), time_precision: Number.isSafeInteger(batch.host_monotonic_ns) ? 'safe_integer' : 'number_rounded' });
  }
  // These two files carry host timestamps. bridge_events uses remote Linux time
  // and is deliberately not mixed into this axis; host 'execution' events carry
  // the original bridge event and its separately named host receipt.
  for (const file of ['task_requests.jsonl', 'behavior_events.jsonl']) {
    let buffer;
    try { buffer = await safeFile(root, file, 16 * 1024 * 1024); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    for (const line of buffer.toString('utf8').split('\n')) {
      if (!line) continue;
      requireValue(Buffer.byteLength(line) <= 256 * 1024, '主机事件超过读取上限');
      const event = JSON.parse(line); sameRun(event, identity);
      for (const field of ['request', 'observation', 'decision', 'event']) sameRun(event[field], identity);
      const at = file === 'task_requests.jsonl' ? event.host_received_monotonic : event.host_monotonic;
      const source = event.event ?? event.decision ?? event.request ?? event.observation ?? event;
      entries.push({ kind: file === 'task_requests.jsonl' ? 'task_request' : String(event.kind).slice(0, 80), host_monotonic_ns: hostTime(at, 's'), original_host_monotonic_seconds: at, time_precision: 'floating_seconds_to_ns_approximate', source_file: file, request_id: source.request_id ?? null, decision_sequence: source.decision_seq ?? event.decision_seq ?? null, skill_execution_id: event.event?.execution_id ?? null, task: source.task ?? null, behavior: source.behavior ?? null, state: source.status ?? null });
      requireValue(entries.length <= 10000, '回放事件数超过上限');
    }
  }
  entries.sort((a, b) => BigInt(a.host_monotonic_ns) < BigInt(b.host_monotonic_ns) ? -1 : BigInt(a.host_monotonic_ns) > BigInt(b.host_monotonic_ns) ? 1 : 0);
  return { schema: PREFIX + 'replay.v1', identity, clock_domain: 'macos_same_host_monotonic', max_frame_age_ns: CAPTURE_LIMITS.max_frame_age_ns, entries, frames, gaps };
}

export async function readCaptureArchive(root, id, session, telemetry) {
  let directory;
  try { directory = await lstat(path.join(root, 'simulation_capture')); } catch (error) { if (error.code === 'ENOENT') return { state: 'absent', complete: false, manifest: null, replay: null, summary: null }; throw error; }
  requireValue(directory.isDirectory() && !directory.isSymbolicLink(), '采样目录不能使用符号链接');
  const folder = path.join(root, 'simulation_capture');
  const manifest = await json(folder, 'manifest.json');
  requireValue(manifest.schema === PREFIX + 'manifest.v1' && manifest.media_kind === 'sampled_frames' && manifest.source === 'novnc_webcontents_snapshot' && same(manifest.limits, CAPTURE_LIMITS) && manifest.clock?.domain === 'macos_same_host_monotonic', '采样清单或时间域无效');
  requireValue(manifest.source_sha256 && sourceNames.every(name => /^[a-f0-9]{64}$/.test(manifest.source_sha256[name])), '采样源码身份缺失');
  checkIdentity(manifest.identity);
  requireValue(manifest.identity.record_id === id && session.run_id === id && ['run_id','session_id','execution_id','checkpoint_id','policy_id','input_map_id'].every(key => telemetry[key] === manifest.identity[key]), '采样与脑回放身份不匹配');
  const verified = await captureIdentity(root, id);
  requireValue(same(verified.identity, manifest.identity) && same(verified.inputs_sha256, manifest.inputs_sha256) && same(verified.official_assets_sha256, manifest.official_assets_sha256), '采样来源已改变');
  if (manifest.seal_protocol === undefined) return { state: 'incomplete', complete: false, reason: '旧画面档案缺少封存及批次校验信息，仅可回放脑记录', manifest, replay: null, summary: null };
  requireValue(manifest.seal_protocol === SEAL_PROTOCOL && Number.isInteger(manifest.seal_generation) && manifest.seal_generation > 0, '采样封存协议无效');
  const pending = await optionalJson(folder, 'seal_pending.json');
  if (pending) {
    requireValue(pending.schema === PREFIX + 'pending.v1' && same(pending.identity, manifest.identity) && pending.generation === manifest.seal_generation, '采样未完成标记无效');
    return { state: 'incomplete', complete: false, reason: '画面封存尚未完成或已超时', manifest, replay: null, summary: null };
  }
  const summary = await optionalJson(folder, 'summary.json');
  if (!summary) return { state: 'incomplete', complete: false, manifest, replay: null, summary: null };
  requireValue(summary.schema === PREFIX + 'summary.v1' && same(summary.identity, manifest.identity) && summary.seal_generation === manifest.seal_generation && ['sealed','failed'].includes(summary.state) && typeof summary.complete === 'boolean' && summary.complete === (summary.state === 'sealed'), '采样封存状态无效');
  const index = await safeFile(folder, 'frame_index.jsonl', CAPTURE_LIMITS.index_bytes);
  const bytes = await safeFile(folder, 'replay_index.json', CAPTURE_LIMITS.index_bytes - index.length);
  requireValue(sha(index) === summary.index_sha256 && sha(bytes) === summary.replay_sha256, '采样索引完整性校验失败');
  const replay = JSON.parse(bytes.toString('utf8'));
  requireValue(replay.schema === PREFIX + 'replay.v1' && same(replay.identity, manifest.identity) && replay.clock_domain === manifest.clock.domain && replay.max_frame_age_ns === CAPTURE_LIMITS.max_frame_age_ns && Array.isArray(replay.frames) && Array.isArray(replay.entries) && Array.isArray(replay.gaps) && replay.frames.length <= CAPTURE_LIMITS.frames && replay.entries.length <= 10000, '回放索引无效');
  const indexed = index.toString('utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  requireValue(same(indexed, replay.frames) && indexed.length === summary.frame_count, '帧索引不匹配');
  let total = 0;
  replay.frames.forEach((frame, i) => {
    requireValue(frame.sequence === i + 1 && ns(frame.capture_start_host_monotonic_ns) && ns(frame.capture_end_host_monotonic_ns) && BigInt(frame.capture_end_host_monotonic_ns) >= BigInt(frame.capture_start_host_monotonic_ns) && Number.isInteger(frame.width) && frame.width > 0 && frame.width <= 1280 && Number.isInteger(frame.height) && frame.height > 0 && frame.height <= 800 && Number.isInteger(frame.bytes) && frame.bytes > 0 && frame.bytes <= CAPTURE_LIMITS.frame_bytes && /^[a-f0-9]{64}$/.test(frame.sha256), '采样帧元数据超界'); total += frame.bytes;
  });
  requireValue(total === summary.image_bytes && total <= CAPTURE_LIMITS.image_bytes, '采样图片总量不匹配');
  let previous = 0n;
  const batchSequences = new Set();
  for (const entry of replay.entries) {
    requireValue(ns(entry.host_monotonic_ns) && BigInt(entry.host_monotonic_ns) >= previous && typeof entry.kind === 'string', '回放主机时间无效'); previous = BigInt(entry.host_monotonic_ns);
    if (['frame','batch'].includes(entry.kind)) requireValue(Number.isInteger(entry.sequence) && entry.sequence > 0 && entry.sequence <= 999999, '回放序号无效');
    if (entry.kind === 'batch') {
      requireValue(!batchSequences.has(entry.sequence), '回放批次序号重复'); batchSequences.add(entry.sequence);
      await readBoundBatch(root, manifest.identity, entry);
    }
  }
  for (const gap of replay.gaps) requireValue(ns(gap.start_host_monotonic_ns) && (gap.end_host_monotonic_ns === null || (ns(gap.end_host_monotonic_ns) && BigInt(gap.end_host_monotonic_ns) >= BigInt(gap.start_host_monotonic_ns))), '采样空档无效');
  return { state: summary.state, complete: summary.complete, manifest, replay, summary };
}
async function readBoundBatch(root, identity, entry) {
  requireValue(Number.isInteger(entry.bytes) && entry.bytes > 0 && entry.bytes <= 256 * 1024 && /^[a-f0-9]{64}$/.test(entry.sha256) && ns(entry.host_monotonic_ns) && hostTime(entry.original_host_monotonic_ns, 'ns') === entry.host_monotonic_ns, '回放批次内容或时间绑定无效');
  const bytes = await safeFile(root, `telemetry/batches/${String(entry.sequence).padStart(6, '0')}.json`, 256 * 1024);
  requireValue(bytes.length === entry.bytes && sha(bytes) === entry.sha256, '回放批次完整性校验失败');
  const batch = JSON.parse(bytes.toString('utf8')); validateBatch(batch, identity);
  requireValue(batch.batch_sequence === entry.sequence && batch.host_monotonic_ns === entry.original_host_monotonic_ns && hostTime(batch.host_monotonic_ns, 'ns') === entry.host_monotonic_ns, '回放批次序号或时间不匹配');
  return batch;
}
export async function readCapturedBatch(root, archive, sequence, cursorIndex) {
  requireValue(Number.isInteger(sequence) && sequence >= 1 && sequence <= 999999 && Number.isInteger(cursorIndex) && cursorIndex >= 0 && cursorIndex < 10000, '联合回放批次或游标无效');
  requireValue(archive.replay && ['sealed','failed'].includes(archive.state), '画面记录尚未封存或校验失败');
  const cursor = archive.replay.entries[cursorIndex]; requireValue(cursor, '联合回放游标不在索引中');
  let selected = null;
  for (const entry of archive.replay.entries) {
    if (BigInt(entry.host_monotonic_ns) > BigInt(cursor.host_monotonic_ns)) break;
    if (entry.kind === 'batch') selected = entry;
  }
  requireValue(selected?.sequence === sequence, '回放批次不属于当前游标或晚于游标');
  // Re-read the exact bytes returned over IPC, even if openRecord just verified
  // the archive. A file can change between opening and requesting the batch.
  return readBoundBatch(root, archive.manifest.identity, selected);
}
export async function readCapturedFrame(root, archive, sequence) {
  requireValue(Number.isInteger(sequence) && sequence >= 1 && sequence <= CAPTURE_LIMITS.frames, '画面帧序号无效');
  requireValue(archive.replay && ['sealed','failed'].includes(archive.state), '画面记录尚未封存');
  const frame = archive.replay.frames.find(item => item.sequence === sequence);
  requireValue(frame, '画面帧不在索引中');
  const bytes = await safeFile(path.join(root, 'simulation_capture'), `frames/${String(sequence).padStart(6, '0')}.jpg`, CAPTURE_LIMITS.frame_bytes);
  requireValue(bytes.length === frame.bytes && sha(bytes) === frame.sha256 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes.at(-2) === 0xff && bytes.at(-1) === 0xd9, '画面图片完整性校验失败');
  return { sequence, identity: archive.manifest.identity, frame, jpeg_base64: bytes.toString('base64') };
}

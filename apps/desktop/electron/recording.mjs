import { createHash, randomBytes } from 'node:crypto';
import { constants, unlinkSync } from 'node:fs';
import { lstat, open, realpath, mkdir, link, unlink, statfs, readdir } from 'node:fs/promises';
import path from 'node:path';
import { MAX_MANIFEST_BYTES, recordIdAllowed } from './boundaries.mjs';

const PREFIX = 'malecns_lab.x2_simulation_capture.';
const SEAL_PROTOCOL = 'pending_marker_v1';
export const CLOCK_DOMAIN = 'python_host_monotonic';
export const CLOCK_SCHEMA = 'malecns_lab.desktop_clock.v1';
export const CLOCK_TIMEOUT_MS = 250;
const CLOCK_PROTOCOL = 'python_capture_outer_bounds_v1';
const MAX_CAPTURE_INTERVAL_NS = 1000000000n;
const LEGACY_CAPTURE_LIMITS = Object.freeze({ interval_ms: 500, duration_ms: 240000, frames: 480, frame_bytes: 512 * 1024, image_bytes: 256 * 1024 * 1024, index_bytes: 4 * 1024 * 1024, free_bytes: 512 * 1024 * 1024, width: 1280, height: 800, jpeg_quality: 70, seal_ms: 2000, max_frame_age_ns: '1000000000' });
export const MEDIA_LIMITS_PROFILE = 'vm480_v2';
export const CAPTURE_LIMITS = Object.freeze({ ...LEGACY_CAPTURE_LIMITS, duration_ms: 480000, frames: 960, index_bytes: 8 * 1024 * 1024 });
// Separate from the immutable historical limits so old archives remain readable.
export const CAPTURE_RECOVERY = Object.freeze({ policy: 'bounded_snapshot_gap_v1', capture_timeout_ms: 1000, max_consecutive_failures: 3, max_gap_ms: 2000 });
const ID_KEYS = ['record_id', 'run_id', 'session_id', 'execution_id', 'checkpoint_id', 'policy_id', 'input_map_id'];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const encoded = value => Buffer.from(JSON.stringify(value));
const ns = value => typeof value === 'string' && /^\d{1,24}$/.test(value);
const nonempty = value => typeof value === 'string' && value.length > 0 && value.length <= 256;
const requireValue = (test, message) => { if (!test) throw new Error(message); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sourceNames = ['recording.mjs', 'main.mjs', 'controller.mjs', 'boundaries.mjs', 'preload.cjs'];

function captureLimits(manifest) {
  const limits = manifest.limits_profile === MEDIA_LIMITS_PROFILE ? CAPTURE_LIMITS
    : manifest.limits_profile === undefined ? LEGACY_CAPTURE_LIMITS : null;
  requireValue(limits && same(manifest.limits, limits), '采样上限版本无效');
  return limits;
}

export function validateClockSample(value, runId, clockId = null) {
  requireValue(value?.schema === CLOCK_SCHEMA && value.type === 'clock' && value.run_id === runId && /^[a-f0-9]{32}$/.test(value.nonce) && /^[a-f0-9]{32}$/.test(value.clock_id) && (!clockId || value.clock_id === clockId), '主机时钟应答身份无效');
  requireValue(value.clock_source?.api === 'time.monotonic_ns' && value.clock_source.monotonic === true && value.clock_source.adjustable === false && nonempty(value.clock_source.implementation), '主机时钟来源无效');
  requireValue(ns(value.python_monotonic_ns) && ns(value.node_send_ns) && ns(value.node_receive_ns) && BigInt(value.node_receive_ns) >= BigInt(value.node_send_ns) && BigInt(value.node_receive_ns) - BigInt(value.node_send_ns) <= BigInt(CLOCK_TIMEOUT_MS) * 1000000n, '主机时钟往返超时或时间无效');
  return value;
}

function validateFrameClock(frame, identity, clockId) {
  const before = validateClockSample(frame.clock_before, identity.run_id, clockId);
  const after = validateClockSample(frame.clock_after, identity.run_id, clockId);
  requireValue(before.nonce !== after.nonce && same(before.clock_source, after.clock_source) && ns(frame.capture_start_node_ns) && ns(frame.capture_end_node_ns)
    && BigInt(before.node_receive_ns) <= BigInt(frame.capture_start_node_ns) && BigInt(frame.capture_start_node_ns) <= BigInt(frame.capture_end_node_ns)
    && BigInt(frame.capture_end_node_ns) <= BigInt(after.node_send_ns) && BigInt(after.node_receive_ns) - BigInt(before.node_send_ns) <= MAX_CAPTURE_INTERVAL_NS
    && frame.capture_start_host_monotonic_ns === before.python_monotonic_ns && frame.capture_end_host_monotonic_ns === after.python_monotonic_ns
    && BigInt(after.python_monotonic_ns) >= BigInt(before.python_monotonic_ns) && BigInt(after.python_monotonic_ns) - BigInt(before.python_monotonic_ns) <= MAX_CAPTURE_INTERVAL_NS, '截图缺少有效的 Python 外包时间区间');
}

// Resolve every component, including directories. Unlike realpath containment alone,
// this also refuses an in-root symlink. O_NOFOLLOW covers a last-component swap.
export async function safeFile(root, relative, limit, prefixOnly = false) {
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
    requireValue(stats.isFile() && (prefixOnly || stats.size <= limit), '画面记录超过读取上限');
    const buffer = Buffer.alloc(limit + 1);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    requireValue(prefixOnly || bytesRead <= limit, '画面记录超过读取上限');
    return buffer.subarray(0, Math.min(bytesRead, limit));
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
  constructor({ root, recordId, sourceRoot, capture, context, onFailure, hostClock, initialWindow = false, onLimit = null, clock = () => process.hrtime.bigint(), platform = process.platform, freeSpace = statfs }) {
    requireValue(!initialWindow || typeof onLimit === 'function', '首段采样需要统一封存回调');
    Object.assign(this, { root, recordId, sourceRoot, capture, context, onFailure, hostClock, initialWindow, onLimit, clock, platform, freeSpace });
    this.limitReason = null;
    this.folder = path.join(root, 'simulation_capture'); this.generation = 1; this.accepting = false; this.inFlight = null; this.timer = null; this.frames = []; this.gaps = []; this.events = []; this.localEvents = []; this.bytes = 0; this.indexBytes = 0; this.error = null; this.reason = null; this.gap = null; this.skipped = 0; this.state = 'initializing'; this.sealing = null; this.finalized = false;
    this.sampleFailures = []; this.consecutiveFailures = 0; this.recoveryStarted = null; this.recoveryTimer = null;
  }
  async initialize() {
    requireValue(this.platform === 'darwin' && typeof this.hostClock === 'function', '联合采样需要同机 Python 主机时钟');
    const verified = await captureIdentity(this.root, this.recordId);
    const available = await this.freeSpace(this.root);
    requireValue(Number(available.bavail) * Number(available.bsize) >= CAPTURE_LIMITS.free_bytes, '采样磁盘预留不足 512 MiB');
    const sources = Object.fromEntries(await Promise.all(sourceNames.map(async name => [name, sha(await safeFile(this.sourceRoot, name, MAX_MANIFEST_BYTES))])));
    const initial = validateClockSample(await this.hostClock(), this.recordId);
    this.started = this.clock(); this.binding = verified;
    this.manifest = { schema: PREFIX + 'manifest.v2', ...verified, seal_protocol: SEAL_PROTOCOL, seal_generation: this.generation, media_kind: 'sampled_frames', source: 'novnc_webcontents_snapshot', limits_profile: MEDIA_LIMITS_PROFILE, limits: CAPTURE_LIMITS, source_sha256: sources, started_node_ns: this.started.toString(), clock: { domain: CLOCK_DOMAIN, protocol: CLOCK_PROTOCOL, clock_id: initial.clock_id, initial_sample: initial, node_source: 'process.hrtime.bigint', python_source: initial.clock_source, interval: 'Python samples before and after capture; outer bounds, not pixel source time', existing_number_precision: 'original Python JSON Number retained', pixel_source_timestamp: null, claim: 'bounded_host_capture_association_only' } };
    this.manifest.snapshot_recovery = CAPTURE_RECOVERY;
    if (this.initialWindow) Object.assign(this.manifest, { archive_scope: 'initial_window', truncation_reason: 'bounded_initial_window_policy', complete_means: 'window_sealed_not_full_session' });
    await mkdir(this.folder); await mkdir(path.join(this.folder, 'frames'));
    // Durable before any publishable manifest: a late summary alone is never a
    // completed seal. Timeout/crash leaves this marker for every future reader.
    await exclusiveFile(this.folder, 'seal_pending.json', encoded({ schema: PREFIX + 'pending.v1', identity: verified.identity, generation: this.generation }));
    await exclusiveFile(this.folder, 'manifest.json', encoded(this.manifest));
    await exclusiveFile(this.folder, 'frame_index.jsonl', Buffer.alloc(0));
    await exclusiveFile(this.folder, 'sample_failures.jsonl', Buffer.alloc(0));
    this.accepting = true; this.state = 'recording';
  }
  start() { this.deadlineTimer = setTimeout(() => this.limit('画面采样达到时长上限'), CAPTURE_LIMITS.duration_ms - (this.initialWindow ? CAPTURE_LIMITS.seal_ms : 0)); const loop = async () => { await this.tick(); if (this.accepting) this.timer = setTimeout(loop, CAPTURE_LIMITS.interval_ms); }; void loop(); }
  status() { return { state: this.state, frames: this.frames.length, reason: this.error ?? this.gap?.reason ?? this.reason, complete: this.state === 'sealed' && !this.error, sample_failures: this.sampleFailures.length, sampling_gap: this.gap !== null, ...(this.initialWindow ? { archive_scope: 'initial_window', truncation_reason: this.limitReason } : {}) }; }
  limit(reason) {
    if (!this.initialWindow) { this.fail(reason); return; }
    if (this.limitReason || !this.accepting) return;
    this.limitReason = reason; this.freeze(reason); this.onLimit(reason); // Never await seal from sample/inFlight.
  }
  freeze(reason) { this.accepting = false; this.reason ??= reason; clearTimeout(this.timer); clearTimeout(this.deadlineTimer); clearTimeout(this.recoveryTimer); }
  fail(error) {
    if (this.error) return;
    this.error = error instanceof Error ? error.message : String(error); this.state = 'failed';
    // Stop the controller before any sealing, IO wait, or capture cancellation.
    try { this.onFailure(this.error); } finally { this.freeze('capture_failed'); }
  }
  viewChanged(state) {
    if (!this.accepting) return;
    if (state.status.state !== 'running' || state.status.resources_stopped) { this.freeze('control_ended'); return; }
    if (!state.visible || !state.loaded || state.loading || state.error) {
      this.openGap(state.error ? 'load_error' : !state.visible ? 'view_hidden' : 'page_loading');
      this.fail(state.error ?? '仿真视图已隐藏、断开或重载');
    }
  }
  async snapshotFailure(error, before, token) {
    if (!this.accepting || token !== this.generation) return;
    const state = this.context();
    requireValue(state.root === this.root && state.status.run_id === this.recordId, '采样运行身份改变');
    this.viewChanged(state);
    if (!this.accepting) return;
    const now = this.clock();
    if (this.recoveryStarted === null) {
      this.recoveryStarted = now;
      this.recoveryTimer = setTimeout(() => this.fail('画面采样缺口超过 2 秒恢复上限'), CAPTURE_RECOVERY.max_gap_ms);
    }
    this.consecutiveFailures++; this.skipped++; this.openGap('snapshot_unavailable');
    const failure = { sequence: this.sampleFailures.length + 1, reason: String(error?.message ?? error).slice(0, 1000),
      noticed_node_ns: now.toString(), clock_before: before, consecutive: this.consecutiveFailures };
    this.sampleFailures.push(failure);
    requireValue(encoded(this.sampleFailures).length <= CAPTURE_LIMITS.index_bytes / 4, '采样错误索引达到上限');
    const handle = await open(path.join(this.folder, 'sample_failures.jsonl'), constants.O_WRONLY | constants.O_APPEND | constants.O_NOFOLLOW);
    try { await handle.writeFile(JSON.stringify(failure) + '\n'); await handle.sync(); } finally { await handle.close(); }
    if (!this.accepting || token !== this.generation) return;
    if (this.consecutiveFailures >= CAPTURE_RECOVERY.max_consecutive_failures || now - this.recoveryStarted >= BigInt(CAPTURE_RECOVERY.max_gap_ms) * 1000000n) this.fail(`画面采样恢复达到上限：${failure.reason}`);
    // The ordinary 500 ms loop may take the next sample. Never retry in-place,
    // copy an old image, or claim that a failed snapshot was a disconnection.
  }
  openGap(reason) {
    // Without a fresh Python sample, invalidate conservatively from the last
    // verified frame; never rename or subtract a Node clock value.
    this.gap ??= { start_host_monotonic_ns: this.frames.at(-1)?.capture_end_host_monotonic_ns ?? this.manifest.clock.initial_sample.python_monotonic_ns, end_host_monotonic_ns: null, noticed_node_ns: this.clock().toString(), boundary: 'last_verified_python_sample_conservative', reason };
  }
  noteStatus(status) {
    if (!this.accepting) return;
    const event = { kind: 'desktop_status', received_node_ns: this.clock().toString(), state: status.state, phase: status.phase ?? null, request_id: status.request_id ?? null, behavior: status.behavior ?? null, message: String(status.message ?? '').slice(0, 2000) };
    if (status.clock_source === 'python_runtime' && status.host_clock_id === this.manifest.clock.clock_id) {
      this.events.push({ ...event, host_monotonic_ns: hostTime(status.host_monotonic_ns, 'ns'), original_host_monotonic_ns: status.host_monotonic_ns, host_clock_id: status.host_clock_id });
    } else this.localEvents.push(event);
    if (encoded([this.events, this.localEvents]).length > CAPTURE_LIMITS.index_bytes / (this.initialWindow ? 8 : 4)) this.limit('采样状态索引达到上限');
  }
  async tick() {
    if (!this.accepting || this.inFlight) { if (this.inFlight) this.skipped++; return; }
    const token = this.generation;
    this.inFlight = this.sample(token).catch(error => { if (token === this.generation) this.fail(error); }).finally(() => { this.inFlight = null; });
    await this.inFlight;
  }
  async sample(token) {
    const now = this.clock();
    if (now - this.started >= BigInt(CAPTURE_LIMITS.duration_ms - (this.initialWindow ? CAPTURE_LIMITS.seal_ms : 0)) * 1000000n || this.frames.length >= CAPTURE_LIMITS.frames) { this.limit('画面采样达到时长或帧数上限'); return; }
    if (this.recoveryStarted !== null && now - this.recoveryStarted >= BigInt(CAPTURE_RECOVERY.max_gap_ms) * 1000000n) { this.fail('画面采样缺口超过 2 秒恢复上限'); return; }
    const state = this.context();
    requireValue(state.root === this.root && state.status.run_id === this.binding.identity.run_id, '采样运行身份改变');
    if (state.status.state !== 'running' || state.status.resources_stopped) { this.freeze('control_ended'); return; }
    const verified = await captureIdentity(this.root, this.recordId);
    requireValue(same(verified, this.binding), '采样来源身份改变');
    if (!this.accepting || token !== this.generation) return;
    if (!state.visible || !state.loaded || state.loading || state.error) {
      this.viewChanged(state); return;
    }
    const latest = await optionalJson(this.root, 'telemetry/latest.json');
    if (latest) validateBatch(latest, this.binding.identity);
    const before = validateClockSample(await this.hostClock(), this.recordId, this.manifest.clock.clock_id);
    if (!this.accepting || token !== this.generation) return;
    const start = this.clock();
    let image, captureTimer;
    try {
      image = await Promise.race([this.capture(), new Promise((_, reject) => {
        captureTimer = setTimeout(() => reject(new Error('画面捕获超过 1 秒上限')), CAPTURE_RECOVERY.capture_timeout_ms);
      })]);
    } catch (error) {
      if (error?.message === 'UnknownVizError') { await this.snapshotFailure(error, before, token); return; }
      throw error; // Unknown errors and a hung capturer are not retryable.
    } finally { clearTimeout(captureTimer); }
    if (!this.accepting || token !== this.generation) return;
    if (image && image.isEmpty()) { await this.snapshotFailure(new Error('仿真采样得到空图'), before, token); return; }
    const finish = this.clock();
    if (this.recoveryStarted !== null && finish - this.recoveryStarted >= BigInt(CAPTURE_RECOVERY.max_gap_ms) * 1000000n) { this.fail('画面采样缺口超过 2 秒恢复上限'); return; }
    const afterClock = validateClockSample(await this.hostClock(), this.recordId, this.manifest.clock.clock_id);
    if (!this.accepting || token !== this.generation) return;
    const after = this.context();
    requireValue(after.root === this.root && after.status.run_id === this.binding.identity.run_id, '采样期间运行身份改变');
    if (!after.visible || !after.loaded || after.loading || after.error || after.status.state !== 'running' || after.status.resources_stopped) { this.viewChanged(after); return; }
    requireValue(image && !image.isEmpty(), '仿真采样得到空图');
    const size = image.getSize();
    requireValue(size.width > 0 && size.height > 0, '仿真采样尺寸无效');
    const scale = Math.min(1, CAPTURE_LIMITS.width / size.width, CAPTURE_LIMITS.height / size.height);
    const resized = scale < 1 ? image.resize({ width: Math.max(1, Math.floor(size.width * scale)), height: Math.max(1, Math.floor(size.height * scale)), quality: 'good' }) : image;
    const dimensions = resized.getSize(), jpeg = resized.toJPEG(CAPTURE_LIMITS.jpeg_quality);
    requireValue(Buffer.isBuffer(jpeg) && jpeg.length > 0 && jpeg.length <= CAPTURE_LIMITS.frame_bytes && dimensions.width <= CAPTURE_LIMITS.width && dimensions.height <= CAPTURE_LIMITS.height, '画面采样超过单帧上限');
    if (this.bytes + jpeg.length > CAPTURE_LIMITS.image_bytes) { this.limit('画面采样达到图片总上限'); return; }
    const sequence = this.frames.length + 1;
    const frame = { sequence, capture_start_host_monotonic_ns: before.python_monotonic_ns, capture_end_host_monotonic_ns: afterClock.python_monotonic_ns, capture_start_node_ns: start.toString(), capture_end_node_ns: finish.toString(), clock_before: before, clock_after: afterClock, width: dimensions.width, height: dimensions.height, bytes: jpeg.length, sha256: sha(jpeg), context: contextOf(state.status, latest), page: { visible: state.visible, loaded: state.loaded, source_freshness: 'unknown_no_pixel_source_timestamp' } };
    validateFrameClock(frame, this.binding.identity, this.manifest.clock.clock_id);
    const line = Buffer.from(JSON.stringify(frame) + '\n');
    if (this.indexBytes + line.length > CAPTURE_LIMITS.index_bytes / (this.initialWindow ? 4 : 2)) { this.limit('画面帧索引达到上限'); return; }
    if (!this.accepting || token !== this.generation) return;
    await exclusiveFile(this.folder, `frames/${String(sequence).padStart(6, '0')}.jpg`, jpeg);
    if (!this.accepting || token !== this.generation) return; // an orphan stays explicit, never enters a later run
    const handle = await open(path.join(this.folder, 'frame_index.jsonl'), constants.O_WRONLY | constants.O_APPEND | constants.O_NOFOLLOW);
    try { await handle.writeFile(line); } finally { await handle.close(); }
    if (token !== this.generation) return;
    this.frames.push(frame); this.bytes += jpeg.length; this.indexBytes += line.length;
    if (this.gap) { this.gap.end_host_monotonic_ns = afterClock.python_monotonic_ns; this.gaps.push(this.gap); this.gap = null; }
    clearTimeout(this.recoveryTimer); this.recoveryTimer = null; this.recoveryStarted = null; this.consecutiveFailures = 0;
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
      if (this.gap) { this.gaps.push(this.gap); this.gap = null; }
      if (!this.manifest || timedOut) return;
      const replay = await buildReplay(this.root, this.binding.identity, this.frames, this.gaps, this.events, this.localEvents, this.manifest.clock, this.initialWindow);
      replay.sample_failures = this.sampleFailures;
      if (timedOut) return;
      requireValue(this.indexBytes + encoded(replay).length <= CAPTURE_LIMITS.index_bytes, '画面回放索引超过 8 MiB');
      await exclusiveFile(this.folder, 'replay_index.json', encoded(replay), () => !timedOut);
      if (timedOut) return;
      const complete = !this.error;
      const result = { schema: PREFIX + 'summary.v2', identity: this.binding.identity, seal_generation: token, state: complete ? 'sealed' : 'failed', complete, reason: this.error ?? this.reason, frame_count: this.frames.length, image_bytes: this.bytes, skipped: this.skipped, ended_node_ns: this.clock().toString(), replay_sha256: sha(encoded(replay)), index_sha256: sha(await safeFile(this.folder, 'frame_index.jsonl', CAPTURE_LIMITS.index_bytes)) };
      if (this.initialWindow) Object.assign(result, { archive_scope: 'initial_window', truncation_reason: this.limitReason ?? this.reason, complete_means: 'window_sealed_not_full_session', event_index_omitted: replay.event_index_omitted });
      result.sample_failures_sha256 = sha(await safeFile(this.folder, 'sample_failures.jsonl', CAPTURE_LIMITS.index_bytes / 4));
      result.sample_failures = this.sampleFailures.length;
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

async function buildReplay(root, identity, frames, gaps, desktopEvents, localEvents, clock, initialWindow = false) {
  const entries = [...desktopEvents];
  const cutoff = frames.at(-1)?.capture_end_host_monotonic_ns ?? clock.initial_sample.python_monotonic_ns;
  const omitted = new Set(); let extraBytes = 0;
  const append = (entry, source) => {
    if (initialWindow && (BigInt(entry.host_monotonic_ns) > BigInt(cutoff) || entries.length >= 9000 || extraBytes + encoded(entry).length > 1024 * 1024)) { omitted.add(source); return; }
    entries.push(entry); extraBytes += encoded(entry).length;
  };
  for (const frame of frames) entries.push({ kind: 'frame', host_monotonic_ns: frame.capture_end_host_monotonic_ns, sequence: frame.sequence });
  for (const gap of gaps) {
    entries.push({ kind: 'gap_start', host_monotonic_ns: gap.start_host_monotonic_ns });
    if (gap.end_host_monotonic_ns) entries.push({ kind: 'gap_end', host_monotonic_ns: gap.end_host_monotonic_ns });
  }
  const listing = await readdir(path.join(root, 'telemetry/batches'));
  // A continuous worker is still publishing batches during first-window seal.
  // Only its exact atomic-write temporary name is excluded; unknown files fail.
  const names = initialWindow ? listing.filter(name => !/^\d{6}\.json\.tmp$/.test(name)) : listing;
  requireValue(names.length <= 4096 && names.every(name => /^\d{6}\.json$/.test(name)), '回放批次目录超界');
  for (const name of names.sort()) {
    const bytes = await safeFile(root, `telemetry/batches/${name}`, 256 * 1024);
    const batch = JSON.parse(bytes.toString('utf8')); validateBatch(batch, identity);
    requireValue(batch.batch_sequence === Number(name.slice(0, 6)), '回放批次文件序号不匹配');
    append({ kind: 'batch', host_monotonic_ns: hostTime(batch.host_monotonic_ns, 'ns'), sequence: batch.batch_sequence, original_host_monotonic_ns: batch.host_monotonic_ns, bytes: bytes.length, sha256: sha(bytes), time_precision: Number.isSafeInteger(batch.host_monotonic_ns) ? 'safe_integer' : 'number_rounded' }, 'telemetry');
  }
  // These two files carry host timestamps. bridge_events uses remote Linux time
  // and is deliberately not mixed into this axis; host 'execution' events carry
  // the original bridge event and its separately named host receipt.
  for (const file of ['task_requests.jsonl', 'behavior_events.jsonl']) {
    let buffer;
    try { buffer = await safeFile(root, file, 16 * 1024 * 1024, initialWindow); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    if (initialWindow) { const boundary = buffer.lastIndexOf(10); if (boundary !== buffer.length - 1 || buffer.length === 16 * 1024 * 1024) omitted.add(file); buffer = buffer.subarray(0, boundary + 1); }
    for (const line of buffer.toString('utf8').split('\n')) {
      if (!line) continue;
      requireValue(Buffer.byteLength(line) <= 256 * 1024, '主机事件超过读取上限');
      const event = JSON.parse(line); sameRun(event, identity);
      for (const field of ['request', 'observation', 'decision', 'event']) sameRun(event[field], identity);
      const at = file === 'task_requests.jsonl' ? event.host_received_monotonic : event.host_monotonic;
      const source = event.event ?? event.decision ?? event.request ?? event.observation ?? event;
      append({ kind: file === 'task_requests.jsonl' ? 'task_request' : String(event.kind).slice(0, 80), host_monotonic_ns: hostTime(at, 's'), original_host_monotonic_seconds: at, time_precision: 'floating_seconds_to_ns_approximate', source_file: file, request_id: source.request_id ?? null, decision_sequence: source.decision_seq ?? event.decision_seq ?? null, skill_execution_id: event.event?.execution_id ?? null, task: source.task ?? null, behavior: source.behavior ?? null, state: source.status ?? null }, file);
      requireValue(entries.length <= 10000, '回放事件数超过上限');
    }
  }
  entries.sort((a, b) => BigInt(a.host_monotonic_ns) < BigInt(b.host_monotonic_ns) ? -1 : BigInt(a.host_monotonic_ns) > BigInt(b.host_monotonic_ns) ? 1 : 0);
  return { schema: PREFIX + 'replay.v2', identity, clock_domain: CLOCK_DOMAIN, clock_id: clock.clock_id, clock_protocol: CLOCK_PROTOCOL, max_frame_age_ns: CAPTURE_LIMITS.max_frame_age_ns, entries, frames, gaps, node_local_events: localEvents, ...(initialWindow ? { archive_scope: 'initial_window', cutoff_host_monotonic_ns: cutoff, event_index_omitted: [...omitted] } : {}) };
}

export async function readCaptureArchive(root, id, session, telemetry) {
  let directory;
  try { directory = await lstat(path.join(root, 'simulation_capture')); } catch (error) { if (error.code === 'ENOENT') return { state: 'absent', complete: false, manifest: null, replay: null, summary: null }; throw error; }
  requireValue(directory.isDirectory() && !directory.isSymbolicLink(), '采样目录不能使用符号链接');
  const folder = path.join(root, 'simulation_capture');
  const manifest = await json(folder, 'manifest.json');
  requireValue([PREFIX + 'manifest.v1', PREFIX + 'manifest.v2'].includes(manifest.schema) && manifest.media_kind === 'sampled_frames' && manifest.source === 'novnc_webcontents_snapshot', '采样清单无效');
  const limits = captureLimits(manifest);
  requireValue(manifest.source_sha256 && sourceNames.every(name => /^[a-f0-9]{64}$/.test(manifest.source_sha256[name])), '采样源码身份缺失');
  if (manifest.snapshot_recovery !== undefined) requireValue(same(manifest.snapshot_recovery, CAPTURE_RECOVERY), '采样恢复规则不匹配');
  checkIdentity(manifest.identity);
  requireValue(manifest.identity.record_id === id && session.run_id === id && ['run_id','session_id','execution_id','checkpoint_id','policy_id','input_map_id'].every(key => telemetry[key] === manifest.identity[key]), '采样与脑回放身份不匹配');
  const verified = await captureIdentity(root, id);
  requireValue(same(verified.identity, manifest.identity) && same(verified.inputs_sha256, manifest.inputs_sha256) && same(verified.official_assets_sha256, manifest.official_assets_sha256), '采样来源已改变');
  if (manifest.schema === PREFIX + 'manifest.v1') return { state: 'incomplete', complete: false, reason: '旧画面记录缺少跨进程时钟证据，仅回放脑记录', manifest, replay: null, summary: null };
  requireValue(manifest.clock?.domain === CLOCK_DOMAIN && manifest.clock.protocol === CLOCK_PROTOCOL, '采样时间域或协议无效');
  validateClockSample(manifest.clock.initial_sample, id, manifest.clock.clock_id);
  requireValue(same(manifest.clock.python_source, manifest.clock.initial_sample.clock_source), '采样主机时钟来源不匹配');
  if (manifest.seal_protocol === undefined) return { state: 'incomplete', complete: false, reason: '旧画面档案缺少封存及批次校验信息，仅可回放脑记录', manifest, replay: null, summary: null };
  requireValue(manifest.seal_protocol === SEAL_PROTOCOL && Number.isInteger(manifest.seal_generation) && manifest.seal_generation > 0, '采样封存协议无效');
  const pending = await optionalJson(folder, 'seal_pending.json');
  if (pending) {
    requireValue(pending.schema === PREFIX + 'pending.v1' && same(pending.identity, manifest.identity) && pending.generation === manifest.seal_generation, '采样未完成标记无效');
    return { state: 'incomplete', complete: false, reason: '画面封存尚未完成或已超时', manifest, replay: null, summary: null };
  }
  const summary = await optionalJson(folder, 'summary.json');
  if (!summary) return { state: 'incomplete', complete: false, manifest, replay: null, summary: null };
  requireValue(summary.schema === PREFIX + 'summary.v2' && same(summary.identity, manifest.identity) && summary.seal_generation === manifest.seal_generation && ['sealed','failed'].includes(summary.state) && typeof summary.complete === 'boolean' && summary.complete === (summary.state === 'sealed'), '采样封存状态无效');
  requireValue(manifest.archive_scope === undefined || (manifest.archive_scope === 'initial_window' && summary.archive_scope === 'initial_window' && manifest.complete_means === 'window_sealed_not_full_session' && summary.complete_means === manifest.complete_means && nonempty(summary.truncation_reason)), '采样首段保存范围无效');
  const index = await safeFile(folder, 'frame_index.jsonl', limits.index_bytes);
  const bytes = await safeFile(folder, 'replay_index.json', limits.index_bytes - index.length);
  requireValue(sha(index) === summary.index_sha256 && sha(bytes) === summary.replay_sha256, '采样索引完整性校验失败');
  const replay = JSON.parse(bytes.toString('utf8'));
  if (manifest.snapshot_recovery !== undefined) {
    const failures = await safeFile(folder, 'sample_failures.jsonl', limits.index_bytes / 4);
    const rows = failures.toString('utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
    requireValue(sha(failures) === summary.sample_failures_sha256 && same(rows, replay.sample_failures) && rows.length === summary.sample_failures, '采样错误记录完整性校验失败');
    for (const [index, row] of rows.entries()) {
      requireValue(row.sequence === index + 1 && ns(row.noticed_node_ns) && typeof row.reason === 'string' && row.reason.length <= 1000 && Number.isInteger(row.consecutive) && row.consecutive >= 1 && row.consecutive <= CAPTURE_RECOVERY.max_consecutive_failures, '采样错误记录无效');
      validateClockSample(row.clock_before, id, manifest.clock.clock_id);
    }
  }
  requireValue(replay.schema === PREFIX + 'replay.v2' && same(replay.identity, manifest.identity) && replay.clock_domain === manifest.clock.domain && replay.clock_id === manifest.clock.clock_id && replay.clock_protocol === CLOCK_PROTOCOL && replay.max_frame_age_ns === limits.max_frame_age_ns && Array.isArray(replay.frames) && Array.isArray(replay.entries) && Array.isArray(replay.gaps) && Array.isArray(replay.node_local_events) && replay.frames.length <= limits.frames && replay.entries.length <= 10000, '回放索引无效');
  const indexed = index.toString('utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  requireValue(same(indexed, replay.frames) && indexed.length === summary.frame_count, '帧索引不匹配');
  let total = 0;
  const nonces = new Set([manifest.clock.initial_sample.nonce]);
  let lastClock = manifest.clock.initial_sample;
  replay.frames.forEach((frame, i) => {
    validateFrameClock(frame, manifest.identity, manifest.clock.clock_id);
    requireValue(!nonces.has(frame.clock_before.nonce) && !nonces.has(frame.clock_after.nonce)
      && BigInt(frame.clock_before.node_send_ns) >= BigInt(lastClock.node_receive_ns)
      && BigInt(frame.clock_before.python_monotonic_ns) >= BigInt(lastClock.python_monotonic_ns)
      && same(frame.clock_before.clock_source, manifest.clock.python_source), '采样时钟重复、倒退或来源改变');
    nonces.add(frame.clock_before.nonce); nonces.add(frame.clock_after.nonce); lastClock = frame.clock_after;
    requireValue(frame.sequence === i + 1 && ns(frame.capture_start_host_monotonic_ns) && ns(frame.capture_end_host_monotonic_ns) && BigInt(frame.capture_end_host_monotonic_ns) >= BigInt(frame.capture_start_host_monotonic_ns) && Number.isInteger(frame.width) && frame.width > 0 && frame.width <= limits.width && Number.isInteger(frame.height) && frame.height > 0 && frame.height <= limits.height && Number.isInteger(frame.bytes) && frame.bytes > 0 && frame.bytes <= limits.frame_bytes && /^[a-f0-9]{64}$/.test(frame.sha256), '采样帧元数据超界'); total += frame.bytes;
  });
  requireValue(total === summary.image_bytes && total <= limits.image_bytes, '采样图片总量不匹配');
  let previous = 0n;
  const batchSequences = new Set();
  for (const entry of replay.entries) {
    requireValue(ns(entry.host_monotonic_ns) && BigInt(entry.host_monotonic_ns) >= previous && typeof entry.kind === 'string', '回放主机时间无效'); previous = BigInt(entry.host_monotonic_ns);
    if (['frame','batch'].includes(entry.kind)) requireValue(Number.isInteger(entry.sequence) && entry.sequence > 0 && entry.sequence <= 999999, '回放序号无效');
    if (entry.kind === 'batch') {
      requireValue(!batchSequences.has(entry.sequence), '回放批次序号重复'); batchSequences.add(entry.sequence);
      await readBoundBatch(root, manifest.identity, entry);
    }
    if (entry.kind === 'frame') requireValue(replay.frames[entry.sequence - 1]?.capture_end_host_monotonic_ns === entry.host_monotonic_ns, '回放帧游标缺少同域时间绑定');
    if (entry.kind === 'desktop_status') requireValue(entry.host_clock_id === manifest.clock.clock_id && hostTime(entry.original_host_monotonic_ns, 'ns') === entry.host_monotonic_ns && ns(entry.received_node_ns), '桌面状态时间来源无效');
  }
  for (const event of replay.node_local_events) requireValue(ns(event.received_node_ns) && event.host_monotonic_ns === undefined, '本地状态不能进入 Python 回放时间轴');
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
  requireValue(archive.replay && ['sealed','failed'].includes(archive.state), '画面记录尚未封存');
  const limits = captureLimits(archive.manifest);
  requireValue(Number.isInteger(sequence) && sequence >= 1 && sequence <= limits.frames, '画面帧序号无效');
  const frame = archive.replay.frames.find(item => item.sequence === sequence);
  requireValue(frame, '画面帧不在索引中');
  const bytes = await safeFile(path.join(root, 'simulation_capture'), `frames/${String(sequence).padStart(6, '0')}.jpg`, limits.frame_bytes);
  requireValue(bytes.length === frame.bytes && sha(bytes) === frame.sha256 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes.at(-2) === 0xff && bytes.at(-1) === 0xd9, '画面图片完整性校验失败');
  return { sequence, identity: archive.manifest.identity, frame, jpeg_base64: bytes.toString('base64') };
}

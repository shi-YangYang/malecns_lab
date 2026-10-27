import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { readdir, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { readVideoArchive } from './video.mjs';
import { NeuralStateReader } from './neural-state.mjs';
import { readCaptureArchive, readCapturedFrame, readCapturedBatch, validateClockSample, CLOCK_SCHEMA, CLOCK_TIMEOUT_MS, CAPTURE_LIMITS } from './recording.mjs';
import { JsonLineReader, MAX_ARCHIVE_BYTES, MAX_MANIFEST_BYTES, STATES, TERMINAL_STATES, readBoundedJson, recordIdAllowed } from './boundaries.mjs';

const EVENT_SCHEMA = 'malecns_lab.desktop_event.v1';
const TASKS = new Set(['GREET', 'FACE_LEFT', 'FACE_RIGHT', 'SHOW_FORWARD', 'END_INTERACTION']);
export const TASK_ACK_TIMEOUT_MS = 2000;
export class ExperimentController {
  constructor(repo, onChange = () => {}, launch = spawn, clock = () => process.hrtime.bigint()) {
    this.repo = repo;
    this.recordsRoot = path.join(repo, 'data/x2_sim/agibot/data/var');
    this.onChange = onChange;
    this.launch = launch;
    this.clock = clock; this.clockRequests = new Map(); this.hostClockId = null;
    this.child = null; this.taskAwaitingAck = null; this.runtimeState = null; this.taskAckFailure = null;
    this.output = null;
    this.state = { state: 'idle', mode: 'brain_control', brain_controlled: false, message: '未运行', simulator_ready: false, motion_started: false, stop_confirmed: null, resources_stopped: true };
    this.cached = null; this.lastRead = 0; this.reading = null; this.stderr = '';
    this.neuralStateReader = new NeuralStateReader();
  }
  notify() { this.cached = null; this.onChange(this.state); }
  prepare() {
    if (this.child || !this.state.resources_stopped) throw new Error('上一实验尚未完成资源清理');
    this.stopRequested = false;
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
    this.output = path.join(this.recordsRoot, `desktop-${stamp}-${randomBytes(4).toString('hex')}`);
    this.cancelClockRequests('实验会话已改变'); this.hostClockId = null;
    this.state = { schema: EVENT_SCHEMA, type: 'status', state: 'preparing', mode: 'brain_control', brain_controlled: false, message: '正在准备脑控验证', output: this.output, run_id: null, simulator_ready: false, motion_started: false, stop_confirmed: null, resources_stopped: false };
    this.stderr = ''; this.protocolError = null; this.clearTaskAck(); this.runtimeState = null; this.taskAckFailure = null; this.notify();
    const child = this.launch(path.join(this.repo, '.venv-train/bin/python'), ['-u', '-m', 'malecns_lab.runtime.x2.behavior', '--execute', '--sim-only', '--qualification', '--automatic-start', '--session-profile', 'continuous', '--output', this.output], {
      cwd: this.repo, shell: false, stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONPATH: path.join(this.repo, 'src'), PYTHONUNBUFFERED: '1' },
    });
    this.child = child;
    const lines = new JsonLineReader(event => {
      if (event.type === 'clock' || event.schema === CLOCK_SCHEMA) { this.receiveClock(event); return; }
      if (event.schema !== EVENT_SCHEMA || event.type !== 'status' || !STATES.has(event.state) || event.output !== this.output || typeof event.message !== 'string') throw new Error('实验状态协议无效');
      if (typeof event.simulator_ready !== 'boolean' || typeof event.motion_started !== 'boolean' || typeof event.resources_stopped !== 'boolean' || ![true, false, null].includes(event.stop_confirmed)) throw new Error('实验安全状态无效');
      const rejected = event.notice === 'request_rejected' || event.phase === 'request_rejected';
      // Older runtimes encoded a notification in phase. Restore only their
      // authoritative request state, never the optimistic local queued state.
      if (event.phase === 'request_rejected') {
        const previous = this.runtimeState;
        const idle = event.action_busy === false && event.request_pending !== true && !event.active_request_id;
        event = { ...event, phase: idle ? 'waiting_task' : previous?.active_request_id === event.active_request_id ? previous?.phase ?? 'queued' : 'queued',
          task: event.active_task ?? previous?.task, notice: 'request_rejected' };
      }
      // Keep the local lock until the runtime acknowledges this particular task.
      // An earlier waiting_task event may already be in the stdout pipe.
      const awaiting = this.taskAwaitingAck;
      if (awaiting && (event.state !== 'running' ||
          (event.phase === 'queued' && event.task === awaiting.task && event.active_request_id && event.active_request_id !== awaiting.previousId) ||
          (rejected && event.rejected_task === awaiting.task))) this.clearTaskAck();
      this.runtimeState = event;
      this.state = { ...event, brain_controlled: event.brain_controlled === true, clock_source: /^[a-f0-9]{32}$/.test(event.host_clock_id) && Number.isFinite(event.host_monotonic_ns) && event.host_monotonic_ns >= 0 ? 'python_runtime' : 'desktop_local', message: event.message.slice(0, 2000) };
      if (this.taskAwaitingAck) this.state = { ...this.state, action_busy: true, request_pending: true, task: this.taskAwaitingAck.task, phase: 'queued', message: '任务已提交，等待脑计算' };
      if (this.taskAckFailure) this.state = { ...this.state, state: TERMINAL_STATES.has(event.state) ? 'failed' : 'stopping',
        brain_controlled: false, reason: 'task_ack_timeout', message: this.taskAckFailure };
      this.notify();
    }, error => { this.protocolError = error.message; this.stop(); });
    child.stdout.setEncoding('utf8'); child.stdout.on('data', chunk => lines.push(chunk));
    child.stderr.setEncoding('utf8'); child.stderr.on('data', chunk => { this.stderr = `${this.stderr}${chunk}`.slice(-4096); });
    child.stdin.on('error', () => { /* Exit handler records an unexpected broken control pipe. */ });
    child.once('error', error => {
      this.clearTaskAck();
      this.cancelClockRequests('实验进程无法启动');
      this.state = { ...this.state, brain_controlled: false, state: 'failed', message: `无法启动实验：${error.message}`, resources_stopped: true, simulator_ready: false };
      this.child = null; this.notify();
    });
    child.once('exit', (code, signal) => {
      this.clearTaskAck();
      this.cancelClockRequests('实验进程已退出');
      this.child = null;
      this.state = { ...this.state, brain_controlled: false };
      if (!TERMINAL_STATES.has(this.state.state)) this.state = { ...this.state, clock_source: 'desktop_local', state: 'failed', message: `实验进程意外退出（${signal ?? code}），未收到停车与资源清理确认`, simulator_ready: false, resources_stopped: false, stop_confirmed: this.state.motion_started ? false : null };
      this.notify();
    });
    return this.state;
  }
  continue() {
    if (this.stopRequested || !this.child || this.state.state !== 'awaiting_simulation' || this.state.request_pending) throw new Error('实验尚未准备完成');
    this.child.stdin.write(`${JSON.stringify({ command: 'continue' })}\n`);
    // Runtime owns the authoritative transition; reject repeated clicks meanwhile.
    this.state = { ...this.state, clock_source: 'desktop_local', request_pending: true, message: '等待实验启动确认' }; this.notify();
    return this.state;
  }
  stop() {
    this.stopRequested = true;
    this.clearTaskAck();
    if (this.child && !TERMINAL_STATES.has(this.state.state) && this.state.state !== 'stopping') {
      this.child.stdin.write(`${JSON.stringify({ command: 'stop' })}\n`);
      this.cancelClockRequests('实验正在停止');
      this.state = { ...this.state, clock_source: 'desktop_local', state: 'stopping', message: '正在停车并关闭实验资源' }; this.notify();
    }
    return this.state;
  }
  clearTaskAck() {
    clearTimeout(this.taskAwaitingAck?.timer);
    this.taskAwaitingAck = null;
  }
  sampleHostClock() {
    if (!this.child || this.state.state !== 'running' || this.clockRequests.size >= 4) return Promise.reject(new Error('主机时钟暂不可用'));
    const nonce = randomBytes(16).toString('hex'), runId = path.basename(this.output), sent = this.clock().toString();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.clockRequests.delete(nonce); reject(new Error('主机时钟应答超时')); }, CLOCK_TIMEOUT_MS);
      this.clockRequests.set(nonce, { runId, sent, resolve, reject, timer });
      try { this.child.stdin.write(`${JSON.stringify({ command: 'clock', run_id: runId, nonce })}\n`); }
      catch (error) { clearTimeout(timer); this.clockRequests.delete(nonce); reject(error); }
    });
  }
  cancelClockRequests(reason) {
    for (const request of this.clockRequests.values()) { clearTimeout(request.timer); request.reject(new Error(reason)); }
    this.clockRequests.clear();
  }
  receiveClock(event) {
    if (!/^[a-f0-9]{32}$/.test(event.nonce)) { this.cancelClockRequests('主机时钟应答缺少 nonce'); return; }
    const request = this.clockRequests.get(event.nonce);
    if (!request) return; // Expired/unsolicited replies never establish a clock.
    clearTimeout(request.timer); this.clockRequests.delete(event.nonce);
    try {
      const value = validateClockSample({ ...event, node_send_ns: request.sent, node_receive_ns: this.clock().toString() }, request.runId, this.hostClockId);
      this.hostClockId = value.clock_id; request.resolve(value);
    } catch (error) { request.reject(error); }
  }
  task(task) {
    if (typeof task !== 'string' || !TASKS.has(task)) throw new Error('互动任务无效');
    if (!this.child || this.state.state !== 'running' || this.state.mode !== 'brain_control' || this.state.brain_controlled !== true || this.state.phase === 'initializing') throw new Error('脑模型尚未准备完成');
    const busy = this.state.action_busy || this.state.request_pending || this.state.phase !== 'waiting_task';
    if (task !== 'END_INTERACTION' && busy) throw new Error('当前动作尚未结束，请等待机器人停稳');
    if (task === 'END_INTERACTION' && this.state.task === task && busy) throw new Error('正在结束互动，请等待机器人停稳');
    // Every ordinary task, including END, enters the model. Emergency stop is separate.
    const previous = this.state, previousAwaiting = this.taskAwaitingAck;
    const awaiting = { task, previousId: this.state.active_request_id ?? null, timer: null };
    this.taskAwaitingAck = awaiting;
    this.state = { ...this.state, action_busy: true, request_pending: true, task, phase: 'queued', message: '任务已提交，等待脑计算', clock_source: 'desktop_local' };
    awaiting.timer = setTimeout(() => {
      if (this.taskAwaitingAck !== awaiting || !this.child) return;
      this.taskAckFailure = '任务接收确认超时，正在独立停车；请等待停车与资源清理确认';
      this.protocolError = this.taskAckFailure;
      this.stop();
      this.state = { ...this.state, state: 'stopping', brain_controlled: false,
        reason: 'task_ack_timeout', message: this.taskAckFailure };
      this.notify();
    }, TASK_ACK_TIMEOUT_MS);
    awaiting.timer.unref?.();
    try { this.child.stdin.write(`${JSON.stringify({ command: 'task', task })}\n`); }
    catch (error) { this.clearTaskAck(); this.taskAwaitingAck = previousAwaiting; this.state = previous; throw error; }
    clearTimeout(previousAwaiting?.timer);
    this.notify();
    return this.state;
  }
  get safeToClose() { return !this.child && this.state.resources_stopped === true; }
  async recordRoot(id) {
    if (!recordIdAllowed(id)) throw new Error('记录标识无效');
    const canonical = await realpath(this.recordsRoot);
    const record = await realpath(path.join(canonical, id));
    if (record !== path.join(canonical, id)) throw new Error('记录路径越界或使用符号链接');
    return record;
  }
  async listRecords() {
    const results = [];
    let entries;
    try { entries = await readdir(this.recordsRoot, { withFileTypes: true }); } catch (error) { if (error.code === 'ENOENT') return []; throw error; }
    for (const entry of entries.filter(item => item.isDirectory() && recordIdAllowed(item.name)).sort((a, b) => b.name.localeCompare(a.name)).slice(0, 100)) {
      try {
        const root = await this.recordRoot(entry.name);
        const session = await readBoundedJson(root, 'desktop_session.json');
        if (session.schema !== EVENT_SCHEMA || !TERMINAL_STATES.has(session.state) || session.resources_stopped !== true || session.mode === 'manual_demo') continue;
        results.push({ id: entry.name, state: session.state, run_id: session.run_id, message: session.message });
      } catch { /* Incomplete, oversized, or unsafe archives are not offered for replay. */ }
    }
    return results;
  }
  async openRecord(id) {
    const root = await this.recordRoot(id);
    const session = await readBoundedJson(root, 'desktop_session.json');
    if (session.schema !== EVENT_SCHEMA || !TERMINAL_STATES.has(session.state) || session.resources_stopped !== true) throw new Error('记录尚未封存');
    const telemetry = path.join(root, 'telemetry');
    if (await realpath(telemetry) !== telemetry || await realpath(path.join(telemetry, 'batches')) !== path.join(telemetry, 'batches')) throw new Error('遥测目录不能使用符号链接');
    let bytes = 0;
    for (const folder of ['', 'batches']) {
      for (const entry of await readdir(path.join(telemetry, folder), { withFileTypes: true })) {
        if (!folder && entry.isDirectory() && entry.name === 'batches') continue;
        if (!entry.isFile() || !entry.name.endsWith('.json')) throw new Error('遥测档案包含不支持的文件');
        bytes += (await stat(path.join(telemetry, folder, entry.name))).size;
        if (bytes > MAX_ARCHIVE_BYTES) throw new Error('遥测档案超过 16 MiB 上限');
      }
    }
    const manifest = await readBoundedJson(root, 'telemetry/manifest.json', MAX_MANIFEST_BYTES);
    const summary = await this.optionalJson(root, 'telemetry/summary.json');
    const names = await readdir(path.join(telemetry, 'batches'));
    const sequences = names.filter(name => /^\d{6}\.json$/.test(name)).map(name => Number(name.slice(0, 6))).sort((a, b) => a - b);
    let capture;
    try { capture = await readCaptureArchive(root, id, session, manifest); }
    catch (error) { capture = { state: 'invalid', complete: false, reason: error.message, manifest: null, replay: null, summary: null }; }
    let video;
    try { video = await readVideoArchive(root, id, session, manifest); }
    catch (error) { video = { state: 'invalid', complete: false, reason: error.message, manifest: null, media_url: null }; }
    return { id, session, manifest, summary, sequences, capture, video };
  }
  async readRecordedFrame(id, sequence) {
    if (!Number.isInteger(sequence) || sequence < 1 || sequence > CAPTURE_LIMITS.frames) throw new Error('画面帧序号无效');
    const record = await this.openRecord(id);
    return readCapturedFrame(await this.recordRoot(id), record.capture, sequence);
  }
  async readBatch(id, sequence, cursorIndex) {
    if (!Number.isInteger(sequence) || sequence < 1 || sequence > 999999) throw new Error('批次序号无效');
    const root = await this.recordRoot(id);
    if (cursorIndex !== undefined) {
      if (!Number.isInteger(cursorIndex) || cursorIndex < 0 || cursorIndex >= 10000) throw new Error('联合回放游标无效');
      const record = await this.openRecord(id);
      return readCapturedBatch(root, record.capture, sequence, cursorIndex);
    }
    const session = await readBoundedJson(root, 'desktop_session.json');
    if (session.schema !== EVENT_SCHEMA || !TERMINAL_STATES.has(session.state) || !session.resources_stopped) throw new Error('记录尚未封存');
    return readBoundedJson(root, `telemetry/batches/${String(sequence).padStart(6, '0')}.json`);
  }
  async optionalJson(root, relative, limit) {
    try { return await readBoundedJson(root, relative, limit); } catch (error) { if (error.code === 'ENOENT') return null; throw error; }
  }
  async snapshot() {
    if (this.reading) return this.reading;
    if (this.cached && Date.now() - this.lastRead < 250) return this.cached;
    this.reading = (async () => {
      let manifest = null, batch = null, summary = null, readError = null;
      let neuralState = null, neuralStateError = null;
      if (this.output) {
        try {
          manifest = await this.optionalJson(this.output, 'telemetry/manifest.json', MAX_MANIFEST_BYTES);
          batch = await this.optionalJson(this.output, 'telemetry/latest.json');
          summary = await this.optionalJson(this.output, 'telemetry/summary.json');
        } catch (error) { readError = error.message; }
        if (manifest && !readError) {
          try { neuralState = await this.neuralStateReader.read(this.output, manifest, batch); }
          catch (error) { neuralStateError = error.message; }
        }
      }
      let delayMs = null, clockSample = null;
      if (batch && Number.isSafeInteger(batch.host_monotonic_ns) && batch.host_monotonic_ns >= 0) {
        try {
          clockSample = await this.sampleHostClock();
          const age = BigInt(clockSample.python_monotonic_ns) - BigInt(batch.host_monotonic_ns);
          if (age >= 0n) delayMs = Number(age) / 1e6;
        } catch { /* Unknown clock means unknown age, never zero or a guessed offset. */ }
      }
      this.cached = { status: this.state, manifest, batch, summary, neural_state: neuralState, neural_state_error: neuralStateError, read_error: readError ?? this.protocolError ?? null, transport_delay_ms: delayMs, transport_clock: clockSample, received_at_ms: Date.now() };
      this.lastRead = Date.now();
      return this.cached;
    })();
    try { return await this.reading; } finally { this.reading = null; }
  }
}

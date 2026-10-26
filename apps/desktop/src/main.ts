import type { Atlas } from '../../atlas/src/types';
import { BrainView } from './brain';
import { LiveStatus, activeStates, canEmergencyStop, canSubmitTask } from './live-status';
import { loadAtlas, validBatch, validManifest, validCapture, replayAt, validReplayBatch } from './data';
import type { ExperimentStatus, OpenRecord, Snapshot, TelemetryBatch, TelemetryManifest, TelemetrySummary, Task } from './types';
import './style.css';

const icons = {
  brain: '<path d="M9 4a3 3 0 0 0-5 2 4 4 0 0 0-1 7 4 4 0 0 0 6 5m6-14a3 3 0 0 1 5 2 4 4 0 0 1 1 7 4 4 0 0 1-6 5M9 3v18m6-18v18M5 10h4m6 4h4"/>',
  robot: '<rect x="5" y="5" width="14" height="14" rx="4"/><path d="M12 2v3M8 10h1m6 0h1M9 15h6M2 10v5m20-5v5"/>',
  reset: '<path d="M4 5v5h5M4 10a8 8 0 1 1 1 8"/>',
  arrow: '<path d="m5 9 7 7 7-7"/>',
};
function icon(name: keyof typeof icons): string { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]}</svg>`; }
function element<T extends HTMLElement = HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function option(value: string, label: string): HTMLOptionElement { const item = document.createElement('option'); item.value = value; item.textContent = label; return item; }
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="titlebar"><div class="brand">${icon('brain')}<span>MaleCNS <b>Lab</b></span></div><span class="titlebar-divider"></span><span class="workspace-label">脑控验证</span><div class="titlebar-right"><span class="experiment-state" id="experiment-state">未运行</span><button class="tool-button primary" id="prepare" disabled>准备验证</button><button class="tool-button" id="start" disabled>开始</button><button class="tool-button stop" id="stop" disabled>紧急停止</button><label class="select-shell record-shell"><span class="sr-only">实验记录</span><select id="record-select" aria-label="实验记录"><option value="">实时</option></select>${icon('arrow')}</label><button class="icon-button" id="refresh-records" aria-label="刷新实验记录" title="刷新记录">${icon('reset')}</button></div></header>
  <main>
    <div class="notice" id="notice" role="alert" hidden></div>
    <div class="taskbar" id="taskbar" aria-label="互动任务"><div class="task-buttons"><button class="tool-button" data-task="GREET" disabled>打招呼</button><button class="tool-button" data-task="FACE_LEFT" disabled>左转 90°</button><button class="tool-button" data-task="FACE_RIGHT" disabled>右转 90°</button><button class="tool-button" data-task="SHOW_FORWARD" disabled>前进 1 米</button><button class="tool-button" data-task="END_INTERACTION" disabled>结束互动</button></div><span id="behavior-state" role="status">未运行</span></div>
    <div class="workspace" id="workspace">
      <section class="panel simulation-panel" aria-labelledby="simulation-title">
        <div class="panel-heading"><div class="panel-title">${icon('robot')}<h2 id="simulation-title">X2 仿真</h2></div><div class="panel-actions"><label class="select-shell" id="media-selector" hidden><select id="media-kind" aria-label="回放画面类型"><option value="sample">采样</option><option value="video">视频</option></select></label><button class="tool-button" id="video-play" hidden>播放</button><span class="badge" id="simulation-state">未运行</span></div></div>
        <div class="simulation-notice" id="simulation-notice" hidden>先 Pause、Reset，确认直立暂停后 Run，再点击“开始”。</div>
        <div class="simulation-stage" id="simulation-stage"><video id="recorded-video" class="recorded-frame" preload="metadata" muted playsinline disablepictureinpicture hidden></video><img id="recorded-frame" class="recorded-frame" alt="本次仿真采样画面" hidden/><div class="empty-simulation" id="empty-simulation">${icon('robot')}<strong id="simulation-message">未运行</strong><span id="simulation-detail">点击“准备验证”连接</span></div></div>
      </section>
      <section class="panel brain-panel" aria-labelledby="brain-title">
        <div class="panel-heading"><div class="panel-title">${icon('brain')}<h2 id="brain-title">MaleCNS 脑模型</h2></div><div class="panel-actions"><span class="badge" id="brain-state">未运行</span><button class="icon-button" id="previous-batch" aria-label="上一批" title="上一批" hidden>‹</button><span class="batch-number" id="batch-number" hidden></span><button class="icon-button" id="next-batch" aria-label="下一批" title="下一批" hidden>›</button><button id="reset-view" class="icon-button" aria-label="复位脑图视角" title="复位视角">${icon('reset')}</button></div></div>
        <div class="brain-stage"><div id="brain"></div><div class="region-legend" id="region-legend" aria-label="神经群组颜色"></div><div class="brain-hint">拖动旋转 · 滚轮缩放</div><div class="activity-legend"><span>群组平均发放率</span><div class="legend-gradient"></div><div class="legend-range"><span>0</span><span id="max-rate">— Hz</span></div></div></div>
        <div class="activity-panel"><div class="activity-heading"><div class="activity-title"><span class="small-dot"></span>群组均值<label class="select-shell group-shell"><span class="sr-only">选择神经群组</span><select id="group-select" aria-label="选择神经群组" disabled></select>${icon('arrow')}</label></div><div class="rate-value"><strong id="current-rate">—</strong><span>Hz</span></div></div><svg id="rate-chart" class="rate-chart" viewBox="0 0 560 100" preserveAspectRatio="none" role="img" aria-label="所选群组已记录的平均发放率曲线"></svg><div class="spike-heading"><span id="spike-label">神经元脉冲 · 未记录</span><span id="neural-time">—</span></div><svg id="spike-chart" class="spike-chart" viewBox="0 0 560 75" preserveAspectRatio="none" role="img" aria-label="选定神经元真实脉冲栅格"></svg></div>
      </section>
    </div>
  </main>`;

const api = window.workbench;
const groupSelect = element<HTMLSelectElement>('group-select');
const recordSelect = element<HTMLSelectElement>('record-select');
let atlas: Atlas, brain: BrainView;
let manifest: TelemetryManifest | null = null, current: TelemetryBatch | null = null;
let summary: TelemetrySummary | null = null, lastStatusNotice = '';
let history: TelemetryBatch[] = [], replay: OpenRecord | null = null, replayIndex = 0;
let replayFrameSequence: number | null = null, replayPreviousBatch = false, replayCursor: string | null = null;
let snapshot: Snapshot | null = null, displaySkipped = 0, lastSequence = 0;
let disposed = false, pending = false, modeVersion = 0;
const liveStatus = new LiveStatus();
let lifecyclePending = false, stopPending = false;
let pollTimer: ReturnType<typeof setTimeout> | null = null, polling = false, pollRequested = false;
let lastVisible: boolean | null = null;
let videoMode = false;
const recordedVideo = element<HTMLVideoElement>('recorded-video');
function clearVideo(): void { recordedVideo.pause(); recordedVideo.removeAttribute('src'); recordedVideo.load(); videoMode = false; element<HTMLSelectElement>('media-kind').value = 'sample'; }
const stateLabels: Record<ExperimentStatus['state'], string> = { idle: '未运行', preparing: '正在准备', awaiting_simulation: '等待仿真启动', running: '实验进行中', stopping: '正在停止', completed: '已完成', cancelled: '已停止', failed: '实验失败' };

let noticeTimer: ReturnType<typeof setTimeout> | null = null, shownMessage = '';
function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  if (!message || message === shownMessage) return;
  shownMessage = message;
  if (noticeTimer !== null) clearTimeout(noticeTimer);
  element('notice').textContent = message; element('notice').hidden = false;
  noticeTimer = setTimeout(() => { element('notice').hidden = true; noticeTimer = null; }, 5000);
}
function clearError(): void {
  if (noticeTimer !== null) clearTimeout(noticeTimer);
  noticeTimer = null; shownMessage = '';
  element('notice').hidden = true; element('notice').textContent = '';
}
function setBounds(): void {
  if (!api || document.hidden) return;
  const { x, y, width, height } = element('simulation-stage').getBoundingClientRect();
  void api.setSimulationBounds({ x, y, width, height }).catch(showError);
}
new ResizeObserver(setBounds).observe(element('simulation-stage'));
window.addEventListener('resize', setBounds);

function useManifest(value: TelemetryManifest): void {
  if (!validManifest(value)) throw new Error('控制脑遥测身份或群组格式不匹配');
  if (manifest?.execution_id === value.execution_id && manifest.run_id === value.run_id) return;
  manifest = value; summary = null; current = null; history = []; lastSequence = 0; displaySkipped = 0;
  brain.setGroups(value.groups);
  groupSelect.replaceChildren(...value.groups.map(group => option(group.id, group.name)));
  groupSelect.disabled = false;
  const preferred = value.groups.find(group => group.name === 'DNg02_L');
  if (preferred) groupSelect.value = preferred.id;
  element('spike-label').textContent = `神经元脉冲 · ${value.selected_neurons.length}/${value.reference_output_neuron_count ?? value.selected_neurons.length}`;
  element('spike-label').title = value.selected_neurons.map((neuron, index) => `第 ${index + 1} 行：${neuron.body_id}`).join('\n');
}
function useSummary(value: TelemetrySummary | null, runId?: string | null): void {
  summary = null;
  if (!value) return;
  if (!manifest || value.schema !== (manifest.schema === 'malecns_lab.x2_behavior_telemetry_manifest.v1' ? 'malecns_lab.x2_behavior_telemetry_summary.v1' : 'malecns_lab.x2_telemetry_summary.v1') || value.run_id !== manifest.run_id || value.execution_id !== manifest.execution_id || (runId && value.run_id !== runId) || !['running', 'completed', 'failed'].includes(value.status) || typeof value.complete !== 'boolean' || !Number.isInteger(value.dropped_batches) || value.dropped_batches < 0 || !Array.isArray(value.errors) || !value.errors.every(error => typeof error === 'string') || (value.error !== null && typeof value.error !== 'string')) throw new Error('遥测汇总与当前控制脑记录不匹配');
  summary = value;
}
function telemetryIssue(status: ExperimentStatus): string {
  const errors = [...new Set([...(summary?.errors ?? []), summary?.error].filter((error): error is string => Boolean(error)))];
  if (errors.length) return `脑活动采集错误：${errors.join('；')}`;
  const ended = replay || (summary && summary.status !== 'running') || ['completed', 'cancelled', 'failed'].includes(status.state);
  if (manifest && ended && !summary?.complete) return `遥测档案不完整${summary?.dropped_batches ? `：丢失 ${summary.dropped_batches} 批` : ''}。仅显示已保存数据。`;
  return '';
}
function dataQualitySummary(): string {
  const dropped = summary?.dropped_batches ?? current?.dropped_batches ?? 0;
  return manifest ? `已归档但未逐批显示 ${displaySkipped} 批；采集丢批 ${dropped}；图谱缺坐标 ${atlas.missing_count} 个。` : '';
}
function useBatch(value: TelemetryBatch, isReplay = false): void {
  if (!manifest || !validBatch(value, manifest)) throw new Error('遥测批次与当前控制脑记录不匹配');
  if (!isReplay && value.batch_sequence === lastSequence) return;
  if (!isReplay && value.batch_sequence < lastSequence) throw new Error('遥测批次倒退，已暂停显示');
  if (!isReplay) {
    displaySkipped += Math.max(0, value.batch_sequence - lastSequence - 1);
    history.push(value); if (history.length > 120) history.shift();
  } else history = [value];
  lastSequence = value.batch_sequence; current = value;
  drawActivity();
}
function drawActivity(): void {
  const groupIndex = manifest?.groups.findIndex(group => group.id === groupSelect.value) ?? -1;
  const rates = current?.group_rates_hz ?? null;
  const maximum = Math.max(1, ...history.flatMap(batch => batch.group_rates_hz));
  brain.update(rates, groupSelect.value, maximum, current);
  element('max-rate').textContent = current ? `${maximum.toFixed(0)} Hz` : '— Hz';
  const rate = rates?.[groupIndex]; element('current-rate').textContent = rate === undefined ? '—' : rate.toFixed(1);
  const chart = document.getElementById('rate-chart')!;
  const first = history[0]?.step_start ?? 0, last = history.at(-1)?.step_end ?? 1;
  const x = (step: number) => 32 + (step - first) / (last - first || 1) * 514;
  const y = (value: number) => 73 - value / maximum * 58;
  let line = '', previous: TelemetryBatch | null = null;
  for (const batch of history) {
    const value = batch.group_rates_hz[groupIndex]; if (value === undefined) continue;
    const connected = previous && previous.batch_sequence + 1 === batch.batch_sequence && previous.step_end === batch.step_start;
    line += `${connected ? 'L' : 'M'}${x(batch.step_end).toFixed(2)},${y(value).toFixed(2)} `;
    previous = batch;
  }
  const point = current && rate !== undefined ? `<circle cx="${x(current.step_end)}" cy="${y(rate)}" r="2.8" class="chart-dot"/>` : '';
  chart.innerHTML = `<path d="M32 15H546M32 44H546M32 73H546" class="chart-grid"/><text x="0" y="19">${maximum.toFixed(0)}</text><text x="14" y="77">0</text><path d="${line}" class="chart-line"/>${point}<text x="32" y="97">${(first * (manifest?.dt_ms ?? .1)).toFixed(1)} ms</text><text x="546" y="97" text-anchor="end">${(last * (manifest?.dt_ms ?? .1)).toFixed(1)} ms</text>`;
  // Replace the whole raster each batch. No persistence/interpolation invents individual spikes.
  const raster = document.getElementById('spike-chart')!;
  let marks = '';
  if (manifest && current) {
    const rows = new Map(manifest.selected_neurons.map((item, index) => [item.body_id, index]));
    for (const event of current.spike_events) {
      const row = rows.get(event.body_id)!;
      const px = 32 + (event.step - current.step_start) / (current.step_end - current.step_start) * 514;
      const py = 5 + row / Math.max(1, rows.size - 1) * 60;
      marks += `M${px.toFixed(2)} ${py.toFixed(2)}v1.8 `;
    }
    element('neural-time').textContent = `${(current.step_start * current.dt_ms).toFixed(1)}–${(current.step_end * current.dt_ms).toFixed(1)} ms · #${current.batch_sequence}`;
    element('neural-time').title = `神经模拟时间；${current.phase === 'precondition' ? '预置输入' : `观测 ${current.observation_sequence}`}。此批采集时尚未形成动作决策。`;
  } else element('neural-time').textContent = '—';
  raster.innerHTML = `<path d="M32 3V69H546" class="chart-grid"/><path d="${marks}" class="spike-mark"/>`;
}
function updateStatus(value: ExperimentStatus): void {
  const controlsPending = pending || stopPending;
  element('media-selector').hidden = !replay?.video.complete;
  element('video-play').hidden = !videoMode;
  element('recorded-video').hidden = !videoMode;
  element('video-play').textContent = recordedVideo.paused ? '播放' : '暂停';
  const behaviorNames: Record<string, string> = { HOLD: '待命', ADVANCE: '前进', TURN_LEFT: '左转', TURN_RIGHT: '右转', WAVE_RIGHT: '挥手' };
  element('taskbar').hidden = Boolean(replay);
  const brainActive = value.mode === 'brain_control' && value.brain_controlled === true;
  app.querySelectorAll<HTMLButtonElement>('[data-task]').forEach(button => {
    button.disabled = controlsPending || Boolean(replay) || !canSubmitTask(value, button.dataset.task as Task);
    button.classList.toggle('is-active', value.action_busy === true && value.task === button.dataset.task);
    button.title = value.action_busy && button.dataset.task !== 'END_INTERACTION' ? '当前动作结束并停稳后可用' : '向脑模型提交互动任务';
  });
  element('behavior-state').textContent = value.state === 'running' ? [value.message, value.phase === 'executing' ? behaviorNames[value.behavior ?? ''] : ''].filter(Boolean).join(' · ') : stateLabels[value.state];
  element('behavior-state').title = value.reason ?? '动作由当前候选脑模型选择；此会话用于受保护验证';
  const busy = activeStates.has(value.state);
  const issue = telemetryIssue(value);
  element('experiment-state').textContent = stateLabels[value.state];
  element('experiment-state').title = value.message;
  element('experiment-state').classList.toggle('failed', value.state === 'failed');
  element<HTMLButtonElement>('prepare').disabled = controlsPending || busy || !value.resources_stopped || Boolean(replay) || !api;
  element<HTMLButtonElement>('start').disabled = controlsPending || value.state !== 'awaiting_simulation' || Boolean(value.request_pending) || Boolean(replay);
  element<HTMLButtonElement>('stop').disabled = !canEmergencyStop(value, lifecyclePending, stopPending);
  recordSelect.disabled = busy || controlsPending;
  element('simulation-notice').hidden = value.state !== 'awaiting_simulation' || Boolean(replay);
  const visible = !replay && !document.hidden;
  if (api && visible !== lastVisible) { lastVisible = visible; void api.setSimulationVisible(visible).catch(showError); }
  element('simulation-state').textContent = videoMode ? '原生视频' : replay ? replay.capture.replay ? '采样回放' : '回放无画面' : value.simulator_ready && !value.resources_stopped ? '实时画面' : stateLabels[value.state];
  element('simulation-message').textContent = replay ? replay.capture.replay ? '此处未采集' : '脑活动回放' : stateLabels[value.state];
  element('simulation-detail').textContent = replay ? replay.capture.state === 'absent' ? '此记录未归档画面' : replay.capture.replay ? '' : '画面档案不完整' : value.state === 'idle' ? '点击“准备验证”连接' : value.message;
  const stale = !replay && snapshot?.transport_delay_ms !== null && snapshot?.transport_delay_ms !== undefined && snapshot.transport_delay_ms > 2000;
  const timeUnknown = !replay && current && brainActive && snapshot?.transport_delay_ms == null;
  element('brain-state').textContent = videoMode ? '视频未对齐' : issue ? summary?.errors.length || summary?.error ? '采集异常' : '记录不完整' : replay ? replayPreviousBatch ? '上一批' : current ? '回放' : '尚无批次' : current ? brainActive ? timeUnknown ? '时间未知' : stale ? '等待新批次' : '实时' : '最后一批' : brainActive ? '等待遥测' : busy ? '等待脑就绪' : '未运行';
  element('brain-state').classList.toggle('mint', !issue && !timeUnknown && !stale && !replay && brainActive && Boolean(current));
  element('brain-state').title = videoMode ? '视频未取得同步时间，脑活动不与视频配对。' : issue || (timeUnknown ? '主机时钟暂不可用，无法确认当前同步状态。' : !replay && snapshot?.transport_delay_ms !== null && snapshot?.transport_delay_ms !== undefined ? `主机取时点的批次年龄 ${(snapshot.transport_delay_ms / 1000).toFixed(2)} 秒；收到时间 ${new Date(snapshot.received_at_ms).toLocaleTimeString('zh-CN')}。显示更新频率不等于脑计算频率。` : replay ? replay.capture.replay ? '截图记录于主机时间区间；没有仿真像素源时间。' : '已封存真实遥测；未归档画面' : '尚无本次控制脑遥测');
  element('brain-state').title += ` ${dataQualitySummary()}`;
  for (const id of ['previous-batch', 'next-batch', 'batch-number']) element(id).hidden = !replay || videoMode;
  element<HTMLButtonElement>('previous-batch').disabled = controlsPending || !replay || replayIndex <= 0;
  element<HTMLButtonElement>('next-batch').disabled = controlsPending || !replay || replayIndex >= (replay.capture.replay?.entries.length ?? replay.sequences.length) - 1;
  if (replay) {
    const count = replay.capture.replay?.entries.length ?? replay.sequences.length;
    element('batch-number').textContent = `${replayIndex + 1}/${count}`;
    element('batch-number').title = replayCursor ? `主机单调时间 ${replayCursor} ns；采样回放，不是连续视频` : '神经批次';
    for (const [id, label] of [['previous-batch', '上一步'], ['next-batch', '下一步']]) { element(id).title = label; element(id).setAttribute('aria-label', label); }
  }
  element('recorded-frame').hidden = videoMode || !replay || replayFrameSequence === null;
  element('empty-simulation').hidden = videoMode || Boolean(replay && replayFrameSequence !== null);
  const captureIssue = replay && !['absent', 'sealed'].includes(replay.capture.state) ? `画面档案不完整：${replay.capture.reason ?? replay.capture.summary?.reason ?? '未完成封存'}` : !replay && snapshot?.capture && ['failed', 'incomplete'].includes(snapshot.capture.state) ? `画面采集失败：${snapshot.capture.reason}` : '';
  const videoIssue = replay && !['absent', 'sealed'].includes(replay.video.state) ? `视频档案不完整：${replay.video.reason ?? '未完成封存'}` : !replay && snapshot?.video && snapshot.video.state === 'incomplete' ? `视频采集失败：${snapshot.video.reason}` : '';
  const statusNotice = [value.state === 'failed' ? value.message : '', issue, captureIssue, videoIssue].filter(Boolean).join('；');
  if (statusNotice && statusNotice !== lastStatusNotice) showError(statusNotice);
  else if (!statusNotice && element('notice').textContent === lastStatusNotice) clearError();
  lastStatusNotice = statusNotice;
}
function renderStatus(): void { if (liveStatus.current) updateStatus(liveStatus.current); }
async function refreshRecords(): Promise<void> {
  if (!api) return;
  const records = await api.listRecords();
  const selected = recordSelect.value;
  const labels: Record<string, string> = { completed: '完成', cancelled: '停止', failed: '失败' };
  recordSelect.replaceChildren(option('', '实时'), ...records.map(record => option(record.id, `${record.id.slice(8, 16)} ${record.id.slice(17, 23)} · ${labels[record.state] ?? record.state}`)));
  recordSelect.value = selected;
}
async function action(kind: 'prepare' | 'start' | 'stop'): Promise<void> {
  if (!api || !liveStatus.current || (kind === 'stop' ? !canEmergencyStop(liveStatus.current, lifecyclePending, stopPending) : pending || stopPending)) return;
  const revision = liveStatus.beginChange();
  clearError();
  if (kind === 'stop') stopPending = true;
  else { pending = true; lifecyclePending = true; }
  renderStatus();
  try {
    if (kind === 'prepare') { manifest = null; summary = null; current = null; history = []; lastSequence = 0; displaySkipped = 0; brain.setGroups([]); groupSelect.replaceChildren(); groupSelect.disabled = true; element('spike-label').textContent = '神经元脉冲 · 未记录'; drawActivity(); }
    const status = await api[kind]();
    if (liveStatus.accept(status, revision)) renderStatus();
  } catch (error) { showError(error); }
  finally {
    if (kind === 'stop') stopPending = false;
    else { pending = false; lifecyclePending = false; }
    renderStatus(); requestPoll();
  }
}
async function showReplay(index: number): Promise<void> {
  if (!api || !replay || videoMode) return;
  const version = modeVersion;
  pending = true;
  try {
    const captured = replay.capture.replay;
    if (captured) {
      const match = replayAt(captured, index);
      const [batch, frame] = await Promise.all([
        match.batch === null ? null : api.readBatch(replay.id, match.batch, index),
        match.frame === null ? null : api.readRecordedFrame(replay.id, match.frame.sequence),
      ]);
      if (version !== modeVersion) return;
      if (batch && !validReplayBatch(batch, replay.manifest, match)) throw new Error('脑批次与当前回放游标不匹配');
      if (frame && (frame.identity.run_id !== replay.id || frame.identity.execution_id !== replay.manifest.execution_id || frame.sequence !== match.frame?.sequence || frame.jpeg_base64.length > 700000)) throw new Error('画面与当前回放不匹配');
      replayIndex = index; replayPreviousBatch = match.previousBatch; replayCursor = match.cursor;
      replayFrameSequence = frame?.sequence ?? null;
      const image = element<HTMLImageElement>('recorded-frame');
      if (frame) image.src = `data:image/jpeg;base64,${frame.jpeg_base64}`; else image.removeAttribute('src');
      if (batch) useBatch(batch, true); else { current = null; history = []; drawActivity(); }
    } else {
      const batch = await api.readBatch(replay.id, replay.sequences[index]);
      if (version !== modeVersion) return;
      replayIndex = index; useBatch(batch, true);
      const sequence = replay.sequences[index], previous = replay.sequences[index - 1] ?? 0;
      if (sequence !== previous + 1) showError(`此处缺少 ${sequence - previous - 1} 批记录，未补造数据。`);
    }
  } finally { pending = false; renderStatus(); }
}
function requestPoll(): void {
  if (disposed) return;
  if (pollTimer !== null) { clearTimeout(pollTimer); pollTimer = null; }
  if (polling) { pollRequested = true; return; }
  void poll();
}
async function poll(): Promise<void> {
  if (disposed) return;
  polling = true;
  if (api) {
    const revision = liveStatus.revision;
    try {
      const next = await api.getSnapshot();
      const previousState = liveStatus.current?.state;
      if (!liveStatus.accept(next.status, revision)) return;
      snapshot = next;
      // Status and emergency controls must update even while the window is hidden.
      renderStatus();
      if (!replay && !document.hidden) {
        if (next.manifest) useManifest(next.manifest);
        useSummary(next.summary, next.status.run_id);
        if (next.batch) useBatch(next.batch);
      }
      renderStatus();
      if (next.read_error || next.simulation_error) showError([next.read_error, next.simulation_error, next.status.state === 'failed' ? next.status.message : '', telemetryIssue(next.status)].filter(Boolean).join('；'));
      if (previousState !== next.status.state && !activeStates.has(next.status.state)) await refreshRecords();
    } catch (error) { showError(error); }
    finally { schedulePoll(); }
  } else schedulePoll();
}
function schedulePoll(): void {
  polling = false;
  if (!disposed) {
    pollTimer = setTimeout(() => { pollTimer = null; void poll(); }, pollRequested ? 0 : document.hidden ? 1000 : 250);
  }
  pollRequested = false;
}
app.querySelectorAll<HTMLButtonElement>('[data-task]').forEach(button => button.addEventListener('click', async () => {
  if (!api || pending || stopPending || replay || !liveStatus.current || !canSubmitTask(liveStatus.current, button.dataset.task as Task)) return;
  const revision = liveStatus.beginChange();
  clearError(); pending = true; renderStatus();
  try { const status = await api.task(button.dataset.task as Task); if (liveStatus.accept(status, revision)) renderStatus(); }
  catch (error) { showError(error); }
  finally { pending = false; renderStatus(); requestPoll(); }
}));
for (const kind of ['prepare', 'start', 'stop'] as const) element(kind).addEventListener('click', () => { void action(kind); });
element('refresh-records').addEventListener('click', () => { void refreshRecords().catch(showError); });
element('previous-batch').addEventListener('click', () => { void showReplay(replayIndex - 1).catch(showError); });
element('next-batch').addEventListener('click', () => { void showReplay(replayIndex + 1).catch(showError); });
recordSelect.addEventListener('change', async () => {
  if (!api) return;
  const version = ++modeVersion; clearVideo(); clearError(); pending = true; replayFrameSequence = null; replayPreviousBatch = false; replayCursor = null; element<HTMLImageElement>('recorded-frame').removeAttribute('src');
  try {
    if (recordSelect.value) { lastVisible = false; await api.setSimulationVisible(false); }
    const id = recordSelect.value;
    replay = null; manifest = null; summary = null; current = null; history = []; lastSequence = 0; displaySkipped = 0;
    if (id) {
      const opened = await api.openRecord(id); if (version !== modeVersion) return;
      if (opened.video.complete && (!opened.video.manifest || opened.video.manifest.identity.run_id !== opened.id || opened.video.manifest.identity.execution_id !== opened.manifest.execution_id || opened.video.media_url !== `workbench://app/media/${opened.id}/recording.webm` || opened.video.manifest.media_anchor !== null)) throw new Error('视频身份或时间锚点无效');
      if (!validCapture(opened.capture, opened)) throw new Error('画面回放身份或时间域无效');
      replay = opened; replayIndex = 0; useManifest(opened.manifest);
      useSummary(opened.summary, opened.session.run_id);
      if (!(opened.capture.replay?.entries.length ?? opened.sequences.length)) throw new Error('记录没有可回放的数据');
      await showReplay(0);
    } else if (snapshot?.manifest) { useManifest(snapshot.manifest); useSummary(snapshot.summary, snapshot.status.run_id); if (snapshot.batch) useBatch(snapshot.batch); }
    else { brain.setGroups([]); groupSelect.replaceChildren(); groupSelect.disabled = true; drawActivity(); }
  } catch (error) { showError(error); }
  finally { pending = false; renderStatus(); setBounds(); }
});
element('media-kind').addEventListener('change', async () => {
  if (!replay || pending || stopPending || (liveStatus.current && activeStates.has(liveStatus.current.state))) return;
  ++modeVersion; recordedVideo.pause();
  videoMode = element<HTMLSelectElement>('media-kind').value === 'video';
  if (videoMode) {
    if (!replay.video.complete || !replay.video.media_url) { videoMode = false; return; }
    recordedVideo.src = replay.video.media_url;
    current = null; history = []; drawActivity();
  } else { recordedVideo.removeAttribute('src'); recordedVideo.load(); await showReplay(replayIndex).catch(showError); }
  renderStatus();
});
element('video-play').addEventListener('click', () => {
  if (!videoMode || !replay || (liveStatus.current && activeStates.has(liveStatus.current.state))) return;
  if (recordedVideo.paused) void recordedVideo.play().catch(showError); else recordedVideo.pause();
});
for (const event of ['play', 'pause', 'ended']) recordedVideo.addEventListener(event, () => { element('video-play').textContent = recordedVideo.paused ? '播放' : '暂停'; });
recordedVideo.addEventListener('error', () => { if (videoMode) { recordedVideo.pause(); showError('视频无法解码或档案校验失败'); } });
document.addEventListener('visibilitychange', () => { if (document.hidden) recordedVideo.pause(); renderStatus(); if (!document.hidden) { setBounds(); requestPoll(); } });
async function start(): Promise<void> {
  atlas = await loadAtlas(); brain = new BrainView(element('brain'), atlas);
  element('region-legend').replaceChildren(...brain.getLegend().map(item => {
    const label = document.createElement('span'), dot = document.createElement('i');
    dot.style.backgroundColor = item.color;
    label.append(dot, item.label); label.title = `${item.count.toLocaleString('zh-CN')} 个有坐标神经元`;
    return label;
  }));
  element('reset-view').addEventListener('click', () => brain.reset()); groupSelect.addEventListener('change', drawActivity);
  window.addEventListener('beforeunload', () => { recordedVideo.pause(); disposed = true; if (noticeTimer !== null) clearTimeout(noticeTimer); if (pollTimer !== null) clearTimeout(pollTimer); brain.dispose(); });
  drawActivity();
  if (!api) { element<HTMLButtonElement>('prepare').disabled = true; showError('请在 Electron 桌面应用中使用实验操作。'); return; }
  await refreshRecords(); setBounds(); requestPoll();
}
start().catch(error => { showError(error); app.querySelectorAll<HTMLButtonElement | HTMLSelectElement>('button, select').forEach(control => { control.disabled = true; }); });

import type { Atlas } from '../../atlas/src/types';
import { BrainView } from './brain';
import { loadBrainDisplay } from './brain-data';
import { LiveStatus, activeStates, canEmergencyStop, canSubmitTask, startAction } from './live-status';
import { loadAtlas, validBatch, validManifest } from './data';
import type { ExperimentStatus, Snapshot, TelemetryBatch, TelemetryManifest, TelemetrySummary, Task } from './types';
import './style.css';

const icons = {
  fly: '<path d="m10.5 4-1-2m4 2 1-2M10 9C6 3 1 5 3 10c1 3 4 4 7 1m4-2c4-6 9-4 7 1-1 3-4 4-7 1M10 10 6 8m8 2 4-2M10 13l-5 2-2 3m11-5 5 2 2 3M10 17l-3 3v2m7-5 3 3v2"/><ellipse cx="10.5" cy="5.5" rx="1.5" ry="1.8"/><ellipse cx="13.5" cy="5.5" rx="1.5" ry="1.8"/><ellipse cx="12" cy="9.5" rx="2" ry="2.5"/><ellipse cx="12" cy="16" rx="2.4" ry="4.5"/><path d="M10 15h4m-4 3h4"/>',
  robot: '<rect x="5" y="5" width="14" height="14" rx="4"/><path d="M12 2v3M8 10h1m6 0h1M9 15h6M2 10v5m20-5v5"/>',
  reset: '<path d="M4 5v5h5M4 10a8 8 0 1 1 1 8"/>',
  forward: '<path d="m3 5 8 7-8 7zm10 0 8 7-8 7z" fill="currentColor" stroke="none"/>',
  left: '<path d="M8 4 4 8l4 4M4 8h9a7 7 0 0 1 0 14h-3"/>',
  right: '<path d="m16 4 4 4-4 4m4-4h-9a7 7 0 0 0 0 14h3"/>',
  wave: '<path d="M8 12V5a1.5 1.5 0 0 1 3 0v6-8a1.5 1.5 0 0 1 3 0v8-6a1.5 1.5 0 0 1 3 0v7-4a1.5 1.5 0 0 1 3 0v7c0 4-2.5 7-6.5 7-3.2 0-4.8-1.4-6.5-4L4 13c-1-1.6 1.2-3.2 2.4-1.8L8 13"/>',
  finish: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
  arrow: '<path d="m5 9 7 7 7-7"/>',
};
function icon(name: keyof typeof icons): string { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]}</svg>`; }
function element<T extends HTMLElement = HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function option(value: string, label: string): HTMLOptionElement { const item = document.createElement('option'); item.value = value; item.textContent = label; return item; }
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="titlebar">
    <div class="brand">${icon('fly')}<span>MaleCNS</span></div>
    <div class="titlebar-right">
      <span class="experiment-state" id="experiment-state">未运行</span>
      <button class="tool-button primary" id="start" disabled>开始</button>
      <button class="tool-button stop" id="stop" disabled>停止</button>
    </div>
  </header>
  <main>
    <div class="notice" id="notice" role="alert" hidden></div>
    <div class="workspace" id="workspace">
      <section class="panel simulation-panel" aria-labelledby="simulation-title">
        <h2 id="simulation-title" class="sr-only">X2 仿真</h2>
        <div class="simulation-stage" id="simulation-stage">
          <div class="empty-simulation" id="empty-simulation">${icon('robot')}<strong id="simulation-message">未运行</strong><span id="simulation-detail">点击“开始”连接</span></div>
        </div>
        <div class="simulation-footer">
          <nav class="taskbar" id="taskbar" aria-label="互动任务">
            <div class="task-buttons">
              <button class="action-button" data-task="SHOW_FORWARD" aria-label="前进 1 米" disabled>${icon('forward')}<span>前进</span></button>
              <button class="action-button" data-task="FACE_LEFT" aria-label="左转 90°" disabled>${icon('left')}<span>左转</span></button>
              <button class="action-button" data-task="FACE_RIGHT" aria-label="右转 90°" disabled>${icon('right')}<span>右转</span></button>
              <button class="action-button" data-task="GREET" aria-label="挥手" disabled>${icon('wave')}<span>挥手</span></button>
              <button class="action-button" data-task="END_INTERACTION" disabled>${icon('finish')}<span>结束互动</span></button>
            </div>
          </nav>
        </div>
      </section>
      <section class="panel brain-panel" aria-labelledby="brain-title">
        <h2 id="brain-title" class="sr-only">MaleCNS 脑模型</h2>
        <div class="brain-stage">
          <div id="brain"></div>
          <button id="reset-view" class="icon-button view-reset" aria-label="复位脑图视角" title="复位视角">${icon('reset')}</button>
        </div>
        <div class="activity-panel">
          <div class="activity-heading">
            <div class="activity-title"><span class="small-dot"></span><span>群组活动</span><label class="select-shell group-shell"><span class="sr-only">选择神经群组</span><select id="group-select" aria-label="选择神经群组" disabled></select>${icon('arrow')}</label></div>
            <div class="activity-tools"><span class="brain-status" id="brain-state">未运行</span><div class="rate-value"><strong id="current-rate">—</strong><span>Hz</span></div></div>
          </div>
          <svg id="rate-chart" class="rate-chart" viewBox="0 0 560 100" preserveAspectRatio="none" role="img" aria-label="所选群组已记录的平均发放率曲线"></svg>
        </div>
      </section>
    </div>
  </main>`;

const api = window.workbench;
const groupSelect = element<HTMLSelectElement>('group-select');
let atlas: Atlas, brain: BrainView;
let manifest: TelemetryManifest | null = null, current: TelemetryBatch | null = null;
let summary: TelemetrySummary | null = null, lastStatusNotice = '';
let history: TelemetryBatch[] = [];
let snapshot: Snapshot | null = null, displaySkipped = 0, lastSequence = 0;
let disposed = false, pending = false;
const liveStatus = new LiveStatus();
let lifecyclePending = false, stopPending = false;
let pollTimer: ReturnType<typeof setTimeout> | null = null, polling = false, pollRequested = false;
let lastVisible: boolean | null = null;
const stateLabels: Record<ExperimentStatus['state'], string> = { idle: '未运行', preparing: '正在准备', awaiting_simulation: '正在连接画面', running: '实验进行中', stopping: '正在停止', completed: '已完成', cancelled: '已停止', failed: '实验失败' };

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
  const ended = (summary && summary.status !== 'running') || ['completed', 'cancelled', 'failed'].includes(status.state);
  if (manifest && ended && !summary?.complete) return `遥测档案不完整${summary?.dropped_batches ? `：丢失 ${summary.dropped_batches} 批` : ''}。仅显示已保存数据。`;
  return '';
}
function dataQualitySummary(): string {
  const dropped = summary?.dropped_batches ?? current?.dropped_batches ?? 0;
  return manifest ? `已归档但未逐批显示 ${displaySkipped} 批；采集丢批 ${dropped}；图谱缺坐标 ${atlas.missing_count} 个。` : '';
}
function useBatch(value: TelemetryBatch): void {
  if (!manifest || !validBatch(value, manifest)) throw new Error('遥测批次与当前控制脑记录不匹配');
  if (value.batch_sequence === lastSequence) return;
  if (value.batch_sequence < lastSequence) throw new Error('遥测批次倒退，已暂停显示');
  displaySkipped += Math.max(0, value.batch_sequence - lastSequence - 1);
  history.push(value); if (history.length > 120) history.shift();
  lastSequence = value.batch_sequence; current = value;
  drawActivity();
}
function drawActivity(): void {
  const groupIndex = manifest?.groups.findIndex(group => group.id === groupSelect.value) ?? -1;
  const rates = current?.group_rates_hz ?? null;
  const maximum = Math.max(1, ...history.flatMap(batch => batch.group_rates_hz));
  brain.update(rates, groupSelect.value, maximum, current);
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
}
function updateStatus(value: ExperimentStatus): void {
  const controlsPending = pending || stopPending;
  const brainActive = value.mode === 'brain_control' && value.brain_controlled === true;
  app.querySelectorAll<HTMLButtonElement>('[data-task]').forEach(button => {
    button.disabled = controlsPending || !canSubmitTask(value, button.dataset.task as Task);
    button.classList.toggle('is-active', value.action_busy === true && value.task === button.dataset.task);
    button.title = value.action_busy && button.dataset.task !== 'END_INTERACTION' ? '当前动作结束并停稳后可用' : '向脑模型提交互动任务';
  });
  const busy = activeStates.has(value.state);
  const issue = telemetryIssue(value);
  element('experiment-state').textContent = value.state === 'preparing' ? value.message : stateLabels[value.state];
  element('experiment-state').title = value.message;
  element('experiment-state').classList.toggle('failed', value.state === 'failed');
  element<HTMLButtonElement>('start').disabled = !api || startAction(value, controlsPending) === null;
  element<HTMLButtonElement>('stop').disabled = !api || !canEmergencyStop(value, lifecyclePending, stopPending);
  const visible = !document.hidden;
  if (api && visible !== lastVisible) { lastVisible = visible; void api.setSimulationVisible(visible).catch(showError); }
  element('simulation-message').textContent = stateLabels[value.state];
  element('simulation-detail').textContent = value.state === 'idle' ? '点击“开始”连接' : value.message;
  const stale = snapshot?.transport_delay_ms !== null && snapshot?.transport_delay_ms !== undefined && snapshot.transport_delay_ms > 2000;
  const timeUnknown = current && brainActive && snapshot?.transport_delay_ms == null;
  element('brain-state').textContent = issue ? summary?.errors.length || summary?.error ? '采集异常' : '记录不完整' : current ? brainActive ? timeUnknown ? '时间未知' : stale ? '等待新批次' : '正在更新' : '最后一批' : brainActive ? '等待遥测' : busy ? '等待脑就绪' : '未运行';
  element('brain-state').classList.toggle('mint', !issue && !timeUnknown && !stale && brainActive && Boolean(current));
  element('brain-state').title = issue || (timeUnknown ? '主机时钟暂不可用，无法确认当前同步状态。' : snapshot?.transport_delay_ms !== null && snapshot?.transport_delay_ms !== undefined ? `主机取时点的批次年龄 ${(snapshot.transport_delay_ms / 1000).toFixed(2)} 秒；收到时间 ${new Date(snapshot.received_at_ms).toLocaleTimeString('zh-CN')}。显示更新频率不等于脑计算频率。` : '尚无本次控制脑遥测');
  element('brain-state').title += ` ${dataQualitySummary()}`;
  const captureIssue = snapshot?.capture && ['failed', 'incomplete'].includes(snapshot.capture.state) ? `画面采集失败：${snapshot.capture.reason}` : '';
  const videoIssue = snapshot?.video?.state === 'incomplete' ? `视频采集失败：${snapshot.video.reason}` : '';
  const statusNotice = [value.state === 'failed' ? value.message : '', issue, captureIssue, videoIssue].filter(Boolean).join('；');
  if (statusNotice && statusNotice !== lastStatusNotice) showError(statusNotice);
  else if (!statusNotice && element('notice').textContent === lastStatusNotice) clearError();
  lastStatusNotice = statusNotice;
}
function renderStatus(): void { if (liveStatus.current) updateStatus(liveStatus.current); }
async function action(kind: 'prepare' | 'stop'): Promise<void> {
  if (!api || !liveStatus.current || (kind === 'stop' ? !canEmergencyStop(liveStatus.current, lifecyclePending, stopPending) : pending || stopPending)) return;
  const revision = liveStatus.beginChange();
  clearError();
  if (kind === 'stop') stopPending = true;
  else { pending = true; lifecyclePending = true; }
  renderStatus();
  try {
    if (kind === 'prepare') { manifest = null; summary = null; current = null; history = []; lastSequence = 0; displaySkipped = 0; brain.setGroups([]); groupSelect.replaceChildren(); groupSelect.disabled = true; drawActivity(); }
    const status = await api[kind]();
    if (liveStatus.accept(status, revision)) renderStatus();
  } catch (error) { showError(error); }
  finally {
    if (kind === 'stop') stopPending = false;
    else { pending = false; lifecyclePending = false; }
    renderStatus(); requestPoll();
  }
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
      if (!liveStatus.accept(next.status, revision)) return;
      snapshot = next;
      // Status and emergency controls must update even while the window is hidden.
      renderStatus();
      if (!document.hidden) {
        if (next.manifest) useManifest(next.manifest);
        useSummary(next.summary, next.status.run_id);
        if (next.batch) useBatch(next.batch);
      }
      renderStatus();
      if (next.read_error || next.simulation_error) showError([next.read_error, next.simulation_error, next.status.state === 'failed' ? next.status.message : '', telemetryIssue(next.status)].filter(Boolean).join('；'));
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
  if (!api || pending || stopPending || !liveStatus.current || !canSubmitTask(liveStatus.current, button.dataset.task as Task)) return;
  const revision = liveStatus.beginChange();
  clearError(); pending = true; renderStatus();
  try { const status = await api.task(button.dataset.task as Task); if (liveStatus.accept(status, revision)) renderStatus(); }
  catch (error) { showError(error); }
  finally { pending = false; renderStatus(); requestPoll(); }
}));
element('start').addEventListener('click', () => {
  if (!api || !liveStatus.current) return;
  const kind = startAction(liveStatus.current, pending || stopPending);
  if (kind) void action(kind);
});
element('stop').addEventListener('click', () => { void action('stop'); });
document.addEventListener('visibilitychange', () => { renderStatus(); if (!document.hidden) { setBounds(); requestPoll(); } });
async function start(): Promise<void> {
  const [loadedAtlas, display] = await Promise.all([loadAtlas(), loadBrainDisplay()]);
  atlas = loadedAtlas; brain = new BrainView(element('brain'), atlas, display);
  element('reset-view').addEventListener('click', () => brain.reset()); groupSelect.addEventListener('change', drawActivity);
  window.addEventListener('beforeunload', () => { disposed = true; if (noticeTimer !== null) clearTimeout(noticeTimer); if (pollTimer !== null) clearTimeout(pollTimer); brain.dispose(); });
  drawActivity();
  if (!api) { element<HTMLButtonElement>('start').disabled = true; showError('请在 Electron 桌面应用中使用实验操作。'); return; }
  setBounds(); requestPoll();
}
start().catch(error => { showError(error); app.querySelectorAll<HTMLButtonElement | HTMLSelectElement>('button, select').forEach(control => { control.disabled = true; }); });

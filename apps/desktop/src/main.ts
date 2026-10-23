import type { Atlas } from '../../atlas/src/types';
import { BrainView } from './brain';
import { loadAtlas, validBatch, validManifest } from './data';
import type { ExperimentStatus, OpenRecord, Snapshot, TelemetryBatch, TelemetryManifest, TelemetrySummary } from './types';
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
  <header class="titlebar"><div class="brand">${icon('brain')}<span>MaleCNS <b>Lab</b></span></div><span class="titlebar-divider"></span><span class="workspace-label">实验工作台</span><div class="titlebar-right"><span class="experiment-state" id="experiment-state">未运行</span><button class="tool-button primary" id="prepare">准备实验</button><button class="tool-button" id="start" disabled>开始</button><button class="tool-button stop" id="stop" disabled>停止</button><label class="select-shell record-shell"><span class="sr-only">实验记录</span><select id="record-select" aria-label="实验记录"><option value="">实时</option></select>${icon('arrow')}</label><button class="icon-button" id="refresh-records" aria-label="刷新实验记录" title="刷新记录">${icon('reset')}</button></div></header>
  <main>
    <div class="notice" id="notice" role="alert" hidden></div>
    <div class="workspace" id="workspace">
      <section class="panel simulation-panel" aria-labelledby="simulation-title">
        <div class="panel-heading"><div class="panel-title">${icon('robot')}<h2 id="simulation-title">X2 仿真</h2></div><span class="badge" id="simulation-state">未运行</span></div>
        <div class="simulation-notice" id="simulation-notice" hidden>请在仿真中点击 Reset、Run，再点击顶部“开始”。</div>
        <div class="simulation-stage" id="simulation-stage"><div class="empty-simulation">${icon('robot')}<strong id="simulation-message">未运行</strong><span id="simulation-detail">准备实验后连接仿真</span></div></div>
      </section>
      <div class="splitter" id="splitter" role="separator" aria-label="调整仿真与脑图宽度" aria-orientation="vertical" aria-valuemin="35" aria-valuemax="65" aria-valuenow="48" tabindex="0"><span></span></div>
      <section class="panel brain-panel" aria-labelledby="brain-title">
        <div class="panel-heading"><div class="panel-title">${icon('brain')}<h2 id="brain-title">MaleCNS 脑模型</h2></div><div class="panel-actions"><span class="badge" id="brain-state">未运行</span><button class="icon-button" id="previous-batch" aria-label="上一批" title="上一批" hidden>‹</button><span class="batch-number" id="batch-number" hidden></span><button class="icon-button" id="next-batch" aria-label="下一批" title="下一批" hidden>›</button><button id="reset-view" class="icon-button" aria-label="复位脑图视角" title="复位视角">${icon('reset')}</button></div></div>
        <div class="brain-stage"><div id="brain"></div><div class="brain-overlay"><span>真实胞体位置</span><strong id="soma-count">—</strong><span id="data-quality"></span></div><div class="brain-hint">拖动旋转 · 滚轮缩放</div><div class="activity-legend"><span>群组平均发放率</span><div class="legend-gradient"></div><div class="legend-range"><span>0</span><span id="max-rate">— Hz</span></div></div></div>
        <div class="activity-panel"><div class="activity-heading"><div class="activity-title"><span class="small-dot"></span>群组均值<label class="select-shell group-shell"><span class="sr-only">选择神经群组</span><select id="group-select" aria-label="选择神经群组" disabled></select>${icon('arrow')}</label></div><div class="rate-value"><strong id="current-rate">—</strong><span>Hz</span></div></div><svg id="rate-chart" class="rate-chart" viewBox="0 0 560 100" preserveAspectRatio="none" role="img" aria-label="所选群组已记录的平均发放率曲线"></svg><div class="spike-heading"><span id="spike-label">神经元脉冲 · 未记录</span><span id="neural-time">—</span></div><svg id="spike-chart" class="spike-chart" viewBox="0 0 560 75" preserveAspectRatio="none" role="img" aria-label="选定神经元真实脉冲栅格"></svg></div>
      </section>
    </div>
  </main>`;

const api = window.workbench;
const groupSelect = element<HTMLSelectElement>('group-select');
const recordSelect = element<HTMLSelectElement>('record-select');
const workspace = element('workspace'), splitter = element('splitter');
let atlas: Atlas, brain: BrainView;
let manifest: TelemetryManifest | null = null, current: TelemetryBatch | null = null;
let summary: TelemetrySummary | null = null, lastStatusNotice = '';
let history: TelemetryBatch[] = [], replay: OpenRecord | null = null, replayIndex = 0;
let snapshot: Snapshot | null = null, displaySkipped = 0, lastSequence = 0;
let ratio = 48, disposed = false, pending = false, modeVersion = 0;
let lastVisible: boolean | null = null;
const activeStates = new Set(['preparing', 'awaiting_simulation', 'running', 'stopping']);
const stateLabels: Record<ExperimentStatus['state'], string> = { idle: '未运行', preparing: '正在准备', awaiting_simulation: '等待仿真启动', running: '实验进行中', stopping: '正在停止', completed: '已完成', cancelled: '已停止', failed: '实验失败' };

function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  element('notice').textContent = message; element('notice').hidden = false;
}
function clearError(): void { element('notice').hidden = true; element('notice').textContent = ''; }
function setBounds(): void {
  if (!api || document.hidden) return;
  const { x, y, width, height } = element('simulation-stage').getBoundingClientRect();
  void api.setSimulationBounds({ x, y, width, height }).catch(showError);
}
function setRatio(value: number): void {
  ratio = Math.min(65, Math.max(35, value));
  workspace.style.setProperty('--split', `${ratio}%`); splitter.setAttribute('aria-valuenow', String(Math.round(ratio)));
}
splitter.addEventListener('pointerdown', event => { splitter.setPointerCapture(event.pointerId); splitter.classList.add('dragging'); });
splitter.addEventListener('pointermove', event => { if (splitter.hasPointerCapture(event.pointerId)) { const bounds = workspace.getBoundingClientRect(); setRatio((event.clientX - bounds.left) / bounds.width * 100); } });
splitter.addEventListener('pointerup', event => { splitter.releasePointerCapture(event.pointerId); splitter.classList.remove('dragging'); });
splitter.addEventListener('lostpointercapture', () => splitter.classList.remove('dragging'));
splitter.addEventListener('keydown', event => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); setRatio(ratio + (event.key === 'ArrowLeft' ? -2 : 2)); } });
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
  if (!manifest || value.schema !== 'malecns_lab.x2_telemetry_summary.v1' || value.run_id !== manifest.run_id || value.execution_id !== manifest.execution_id || (runId && value.run_id !== runId) || !['running', 'completed', 'failed'].includes(value.status) || typeof value.complete !== 'boolean' || !Number.isInteger(value.dropped_batches) || value.dropped_batches < 0 || !Array.isArray(value.errors) || !value.errors.every(error => typeof error === 'string') || (value.error !== null && typeof value.error !== 'string')) throw new Error('遥测汇总与当前控制脑记录不匹配');
  summary = value;
}
function telemetryIssue(status: ExperimentStatus): string {
  const errors = [...new Set([...(summary?.errors ?? []), summary?.error].filter((error): error is string => Boolean(error)))];
  if (errors.length) return `脑活动采集错误：${errors.join('；')}`;
  const ended = replay || (summary && summary.status !== 'running') || ['completed', 'cancelled', 'failed'].includes(status.state);
  if (manifest && ended && !summary?.complete) return `遥测档案不完整${summary?.dropped_batches ? `：丢失 ${summary.dropped_batches} 批` : ''}。仅显示已保存数据。`;
  return '';
}
function updateDataQuality(): void {
  const points = new Set(atlas.body_ids);
  const selectedGroup = manifest?.groups.find(group => group.id === groupSelect.value);
  const missing = selectedGroup?.body_ids.filter(id => !points.has(id)).length ?? 0;
  const skipped = replay ? '回放' : `跳显 ${displaySkipped}`;
  const dropped = summary?.status !== 'running' && summary ? summary.dropped_batches : Math.max(summary?.dropped_batches ?? 0, current?.dropped_batches ?? 0);
  element('data-quality').textContent = manifest ? `${skipped} · 丢批 ${dropped} · 无坐标 ${missing}` : '';
  element('data-quality').title = '跳显：显示端未逐批展示；丢批：采集端未归档；无坐标：当前群组中没有胞体坐标的成员，仍计入均值。';
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
  brain.update(rates, groupSelect.value, maximum);
  element('max-rate').textContent = current ? `${maximum.toFixed(0)} Hz` : '— Hz';
  const rate = rates?.[groupIndex]; element('current-rate').textContent = rate === undefined ? '—' : rate.toFixed(1);
  updateDataQuality();
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
  const busy = activeStates.has(value.state);
  const issue = telemetryIssue(value);
  element('experiment-state').textContent = stateLabels[value.state];
  element('experiment-state').title = value.message;
  element('experiment-state').classList.toggle('failed', value.state === 'failed');
  element<HTMLButtonElement>('prepare').disabled = pending || busy || !value.resources_stopped || Boolean(replay) || !api;
  element<HTMLButtonElement>('start').disabled = pending || value.state !== 'awaiting_simulation' || Boolean(value.request_pending) || Boolean(replay);
  element<HTMLButtonElement>('stop').disabled = pending || !busy || value.state === 'stopping';
  recordSelect.disabled = busy || pending;
  element('simulation-notice').hidden = value.state !== 'awaiting_simulation' || Boolean(replay);
  const visible = !replay && !document.hidden;
  if (api && visible !== lastVisible) { lastVisible = visible; void api.setSimulationVisible(visible).catch(showError); }
  element('simulation-state').textContent = replay ? '回放无画面' : value.simulator_ready && !value.resources_stopped ? '实时画面' : stateLabels[value.state];
  element('simulation-message').textContent = replay ? '脑活动回放' : stateLabels[value.state];
  element('simulation-detail').textContent = replay ? '此记录未归档仿真视频' : value.state === 'idle' ? '准备实验后连接仿真' : value.message;
  element('brain-state').textContent = issue ? summary?.errors.length || summary?.error ? '采集异常' : '记录不完整' : replay ? '回放' : current ? busy ? '实时' : '最后一批' : busy ? '等待遥测' : '未运行';
  element('brain-state').classList.toggle('mint', !issue && !replay && busy && Boolean(current));
  element('brain-state').title = issue || (!replay && snapshot?.transport_delay_ms !== null && snapshot?.transport_delay_ms !== undefined ? `最后一批距今 ${(snapshot.transport_delay_ms / 1000).toFixed(2)} 秒；收到时间 ${new Date(snapshot.received_at_ms).toLocaleTimeString('zh-CN')}。显示更新频率不等于脑计算频率。` : replay ? '已封存真实遥测；未播放仿真视频' : '尚无本次控制脑遥测');
  for (const id of ['previous-batch', 'next-batch', 'batch-number']) element(id).hidden = !replay;
  element<HTMLButtonElement>('previous-batch').disabled = pending || !replay || replayIndex <= 0;
  element<HTMLButtonElement>('next-batch').disabled = pending || !replay || replayIndex >= replay.sequences.length - 1;
  if (replay) element('batch-number').textContent = `${replayIndex + 1}/${replay.sequences.length}`;
  const statusNotice = [value.state === 'failed' ? value.message : '', issue].filter(Boolean).join('；');
  if (statusNotice) showError(statusNotice);
  else if (element('notice').textContent === lastStatusNotice) clearError();
  lastStatusNotice = statusNotice;
}
async function refreshRecords(): Promise<void> {
  if (!api) return;
  const records = await api.listRecords();
  const selected = recordSelect.value;
  const labels: Record<string, string> = { completed: '完成', cancelled: '停止', failed: '失败' };
  recordSelect.replaceChildren(option('', '实时'), ...records.map(record => option(record.id, `${record.id.slice(8, 16)} ${record.id.slice(17, 23)} · ${labels[record.state] ?? record.state}`)));
  recordSelect.value = selected;
}
async function action(kind: 'prepare' | 'start' | 'stop'): Promise<void> {
  if (!api || pending) return;
  clearError(); pending = true;
  if (snapshot) updateStatus(snapshot.status);
  try {
    if (kind === 'prepare') { manifest = null; summary = null; current = null; history = []; lastSequence = 0; displaySkipped = 0; brain.setGroups([]); groupSelect.replaceChildren(); groupSelect.disabled = true; element('spike-label').textContent = '神经元脉冲 · 未记录'; drawActivity(); }
    const status = await api[kind](); updateStatus(status);
  } catch (error) { showError(error); }
  finally { pending = false; if (snapshot) updateStatus(snapshot.status); }
}
async function showReplay(index: number): Promise<void> {
  if (!api || !replay) return;
  const version = modeVersion;
  pending = true;
  try {
    const batch = await api.readBatch(replay.id, replay.sequences[index]);
    if (version !== modeVersion) return;
    replayIndex = index; useBatch(batch, true);
    const sequence = replay.sequences[index], previous = replay.sequences[index - 1] ?? 0;
    if (sequence !== previous + 1) showError(`此处缺少 ${sequence - previous - 1} 批记录，未补造数据。`);
  } finally { pending = false; if (snapshot) updateStatus(snapshot.status); }
}
async function poll(): Promise<void> {
  if (disposed) return;
  if (api && !document.hidden) {
    try {
      const next = await api.getSnapshot();
      const previousState = snapshot?.status.state; snapshot = next;
      if (!replay) {
        if (next.manifest) useManifest(next.manifest);
        const previousSummary = summary;
        useSummary(next.summary, next.status.run_id);
        if (next.batch) useBatch(next.batch);
        if (previousSummary?.dropped_batches !== summary?.dropped_batches || previousSummary?.status !== summary?.status) updateDataQuality();
      }
      updateStatus(next.status);
      if (next.read_error || next.simulation_error) showError([next.read_error, next.simulation_error, next.status.state === 'failed' ? next.status.message : '', telemetryIssue(next.status)].filter(Boolean).join('；'));
      if (previousState !== next.status.state && !activeStates.has(next.status.state)) await refreshRecords();
    } catch (error) { showError(error); }
  }
  if (!disposed) setTimeout(() => { void poll(); }, 250);
}
for (const kind of ['prepare', 'start', 'stop'] as const) element(kind).addEventListener('click', () => { void action(kind); });
element('refresh-records').addEventListener('click', () => { void refreshRecords().catch(showError); });
element('previous-batch').addEventListener('click', () => { void showReplay(replayIndex - 1).catch(showError); });
element('next-batch').addEventListener('click', () => { void showReplay(replayIndex + 1).catch(showError); });
recordSelect.addEventListener('change', async () => {
  if (!api) return;
  const version = ++modeVersion; clearError(); pending = true;
  try {
    const id = recordSelect.value;
    replay = null; manifest = null; summary = null; current = null; history = []; lastSequence = 0; displaySkipped = 0;
    if (id) {
      const opened = await api.openRecord(id); if (version !== modeVersion) return;
      replay = opened; replayIndex = 0; useManifest(opened.manifest);
      useSummary(opened.summary, opened.session.run_id);
      if (!opened.sequences.length) throw new Error('记录没有可回放的遥测批次');
      await showReplay(0);
    } else if (snapshot?.manifest) { useManifest(snapshot.manifest); useSummary(snapshot.summary, snapshot.status.run_id); if (snapshot.batch) useBatch(snapshot.batch); updateDataQuality(); }
    else { brain.setGroups([]); groupSelect.replaceChildren(); groupSelect.disabled = true; drawActivity(); }
  } catch (error) { showError(error); }
  finally { pending = false; if (snapshot) updateStatus(snapshot.status); setBounds(); }
});
document.addEventListener('visibilitychange', () => { if (snapshot) updateStatus(snapshot.status); if (!document.hidden) setBounds(); });
async function start(): Promise<void> {
  atlas = await loadAtlas(); brain = new BrainView(element('brain'), atlas);
  element('soma-count').textContent = `${atlas.positioned_count.toLocaleString('en-US')} 个`;
  element('reset-view').addEventListener('click', () => brain.reset()); groupSelect.addEventListener('change', drawActivity);
  window.addEventListener('beforeunload', () => { disposed = true; brain.dispose(); });
  drawActivity();
  if (!api) { element<HTMLButtonElement>('prepare').disabled = true; showError('请在 Electron 桌面应用中使用实验操作。'); return; }
  await refreshRecords(); setBounds(); void poll();
}
start().catch(error => { showError(error); app.querySelectorAll<HTMLButtonElement | HTMLSelectElement>('button, select').forEach(control => { control.disabled = true; }); });

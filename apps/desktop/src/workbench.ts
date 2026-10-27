import type { Atlas } from '../../atlas/src/types';
import { BrainView } from './brain';
import { ActivityTimeline, type ActivityReading } from './activity-timeline';
import { loadBrainDisplay, loadBrainGraphInfo, type BrainDisplayManifest } from './brain-data';
import { createNeuralMapping, mapNeuralSignals, VOLTAGE_DELTA_HALF_MV, SPIKE_RATE_HALF_HZ, type NeuralMapping, type NeuralSignals } from './neural-state';
import { ActivityDisplay, LiveStatus, activeStates, canEmergencyStop, canSubmitTask, startAction } from './live-status';
import { groupActivityRate, loadAtlas, validBatch, validManifest } from './data';
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
          <aside class="brain-info" aria-label="脑模型信息">
            <dl><div><dt>神经元</dt><dd id="brain-neuron-count">—</dd></div><div><dt>神经连接</dt><dd id="brain-edge-count">—</dd></div></dl>
          </aside>
          <div id="fly-orientation" class="fly-orientation" role="img" aria-label="果蝇方向示意，随脑图视角同步旋转"></div>
          <aside class="brain-legend" aria-label="活动高亮图例">
            <span id="neuron-legend"><i class="legend-neuron" aria-hidden="true"></i><span id="neuron-legend-label">膜电位变化</span></span>
            <span id="connection-legend"><i class="legend-connection" aria-hidden="true"></i><span id="connection-legend-label">突触前放电</span></span>
            <span id="brain-signal-status" hidden></span>
          </aside>
          <button id="reset-view" class="icon-button view-reset" aria-label="复位脑图视角" title="复位视角">${icon('reset')}</button>
        </div>
        <div class="activity-panel">
          <div class="activity-heading">
            <div class="activity-title"><span class="small-dot"></span><span class="activity-label">神经活动</span><span class="activity-scope" id="activity-scope">已记录群组均值</span></div>
            <div class="activity-tools">
              <details class="activity-details">
                <summary>详情${icon('arrow')}</summary>
                <div class="activity-details-content">
                  <label for="group-select">统计范围</label>
                  <div class="select-shell group-shell"><select id="group-select" aria-label="选择神经群组" aria-describedby="activity-definition" disabled><option value="">已记录群组均值</option></select>${icon('arrow')}</div>
                  <p id="activity-definition">各已记录群组发放率的算术均值；群组可重叠，不代表全脑平均。</p>
                </div>
              </details>
              <span class="brain-status" id="brain-state">未运行</span><div class="rate-value"><strong id="current-rate">0.0</strong><span>Hz</span></div>
            </div>
          </div>
          <svg id="rate-chart" class="rate-chart" viewBox="0 0 560 100" preserveAspectRatio="none" role="img" aria-label="神经活动：暂无实时数据"></svg>
        </div>
      </section>
    </div>
  </main>`;

const api = window.workbench;
const groupSelect = element<HTMLSelectElement>('group-select');
let atlas: Atlas, brain: BrainView;
let displayManifest: BrainDisplayManifest;
let fullNeural = false, neuralMapping: NeuralMapping | null = null, neuralSignals: NeuralSignals | null = null;
let neuralManifestError = '', neuralFrameError = '';
let checkedNeuralSnapshot: Snapshot | null = null, checkedNeuralBatch: TelemetryBatch | null = null;
let renderedNeuralSignals: NeuralSignals | null = null;
let manifest: TelemetryManifest | null = null, current: TelemetryBatch | null = null;
let summary: TelemetrySummary | null = null, lastStatusNotice = '';
let snapshot: Snapshot | null = null, displaySkipped = 0, lastSequence = 0;
let snapshotReceivedAt = 0, activityReadError = '';
let renderedBrainBatch: TelemetryBatch | null = null;
let disposed = false, pending = false;
const liveStatus = new LiveStatus();
const activityDisplay = new ActivityDisplay();
const activityTimeline = new ActivityTimeline();
let activityTimer: ReturnType<typeof setTimeout> | null = null;
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

async function useManifest(value: TelemetryManifest): Promise<void> {
  if (!validManifest(value)) throw new Error('控制脑遥测身份或群组格式不匹配');
  if (manifest?.execution_id === value.execution_id && manifest.run_id === value.run_id) return;
  manifest = value; summary = null; current = null; lastSequence = 0; displaySkipped = 0;
  renderedBrainBatch = null; activityTimeline.reset();
  brain.setGroups(value.groups);
  fullNeural = Object.prototype.hasOwnProperty.call(value, 'neural_state');
  neuralMapping = null; neuralSignals = null; renderedNeuralSignals = null;
  neuralManifestError = ''; neuralFrameError = ''; checkedNeuralSnapshot = null; checkedNeuralBatch = null;
  groupSelect.replaceChildren(option('', '已记录群组均值'), ...value.groups.map(group => option(group.id, group.name)));
  groupSelect.disabled = false;
  groupSelect.value = '';
  if (fullNeural) {
    try {
      const mapping = await createNeuralMapping(value.neural_state!, displayManifest);
      if (manifest === value) neuralMapping = mapping;
    } catch (error) {
      if (manifest === value) neuralManifestError = error instanceof Error ? error.message : String(error);
    }
  }
}
function clearActivity(): void { current = null; drawActivity(); }
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
  if (!activityDisplay.accepts(value)) return;
  if (!manifest || !validBatch(value, manifest)) throw new Error('遥测批次与当前控制脑记录不匹配');
  if (value.batch_sequence === lastSequence) return;
  if (value.batch_sequence < lastSequence) throw new Error('遥测批次倒退，已暂停显示');
  const breakBefore = current !== null && value.batch_sequence > lastSequence + 1;
  displaySkipped += Math.max(0, value.batch_sequence - lastSequence - 1);
  lastSequence = value.batch_sequence; current = value;
  drawActivity(breakBefore);
}
function activityReading(now = performance.now()): ActivityReading {
  const status = liveStatus.current;
  if (activityReadError || (status && (telemetryIssue(status) || status.state === 'failed'))) return { kind: 'gap', rates: null };
  if (activityDisplay.mode !== 'active') return { kind: 'baseline', rates: null };
  if (!current || current.phase !== 'forward' || current.run_id !== activityDisplay.runId || !activityDisplay.requestId || current.request_id !== activityDisplay.requestId) return { kind: 'gap', rates: null };
  // Snapshot age belongs to this actual batch. Local elapsed time prevents a
  // stopped/unresponsive poll from keeping the last measurement live forever.
  const delay = snapshot?.batch?.batch_sequence === current.batch_sequence ? snapshot.transport_delay_ms : null;
  if (delay === null || delay === undefined || !Number.isFinite(delay) || delay < 0 || delay + now - snapshotReceivedAt > 2000) return { kind: 'gap', rates: null };
  return { kind: 'measured', rates: current.group_rates_hz };
}
function currentNeuralSignals(batch: TelemetryBatch | null): NeuralSignals | null {
  if (!batch || !fullNeural) return null;
  if (checkedNeuralSnapshot === snapshot && checkedNeuralBatch === batch) return neuralSignals;
  checkedNeuralSnapshot = snapshot; checkedNeuralBatch = batch;
  neuralSignals = null; neuralFrameError = snapshot?.neural_state_error ?? '';
  if (neuralManifestError || neuralFrameError || !snapshot?.neural_state || !neuralMapping || !manifest) return null;
  try { neuralSignals = mapNeuralSignals(snapshot.neural_state, batch, manifest, neuralMapping); }
  catch (error) { neuralFrameError = error instanceof Error ? error.message : String(error); }
  return neuralSignals;
}
function updateNeuralLegend(liveBatch: TelemetryBatch | null, signals: NeuralSignals | null): string {
  const full = !manifest || fullNeural;
  element('neuron-legend-label').textContent = full ? '膜电位变化' : '群组活动';
  element('connection-legend-label').textContent = full ? '突触前放电' : '神经连接';
  element('neuron-legend').title = full ? `紫色：每个神经元相对本次请求初始状态的膜电位变化绝对值，固定 ${VOLTAGE_DELTA_HALF_MV} mV 半亮度；未放电也可以变化。` : '旧记录：已记录群组平均放电率，同组细胞使用同一亮度。';
  element('connection-legend').title = full ? `金色：真实抽样连接的突触前神经元实际放电率，固定 ${SPIKE_RATE_HALF_HZ} Hz 半亮度；不表示突触电流或轴突传播。` : '旧记录：真实抽样连接的突触前群组活动；不是逐细胞放电测量。';
  const status = element('brain-signal-status');
  const error = neuralManifestError || neuralFrameError;
  status.textContent = manifest && !fullNeural ? '旧记录' : liveBatch && !signals ? error ? '全脑数据异常' : '等待全脑数据' : '';
  status.hidden = !status.textContent;
  status.title = error || (full ? '尚未收到与当前动作批次匹配的全脑状态；不以群组均值替代。' : '此记录未采集全脑膜电位，只能展示已有群组放电率。');
  if (!full) return '脑图为旧记录群组放电活动，未采集全脑膜电位；连接使用突触前群组放电率，不是逐条突触电流或传播测量。';
  return `脑图紫色表示逐细胞膜电位相对本次请求初态的变化绝对值（50 步窗口均值）；金色连接按突触前细胞的真实放电率调亮。全脑采集 ${displayManifest.node_count.toLocaleString('zh-CN')} 个神经元，其中 ${brain.positionedNeuronCount.toLocaleString('zh-CN')} 个有当前真实展示坐标；缺坐标不补造。灰暗不表示膜电位不变，低于显示尺度的变化可能不可见。连线为真实连接的抽样示意，不是轴突形态或传播测量。`;
}
function drawActivity(breakBefore = false): void {
  if (!brain || document.hidden) return;
  const now = performance.now();
  const reading = activityReading(now);
  const elapsed = activityTimeline.record(reading, now, breakBefore);
  const liveBatch = reading.kind === 'measured' ? current : null;
  const signals = currentNeuralSignals(liveBatch);
  // A chart clock tick only redraws SVG. Upload neural textures on data or
  // visibility-state changes, not four times a second while idle.
  if (liveBatch !== renderedBrainBatch || signals !== renderedNeuralSignals) {
    if (fullNeural) brain.updateNeural(signals, liveBatch);
    else brain.update(liveBatch?.group_rates_hz ?? null, liveBatch);
    renderedBrainBatch = liveBatch;
    renderedNeuralSignals = signals;
  }
  const neuralDefinition = updateNeuralLegend(liveBatch, signals);
  const groupIndex = groupSelect.value === '' ? null : manifest?.groups.findIndex(group => group.id === groupSelect.value) ?? -1;
  const valueFor = (item: ActivityReading): number | undefined => item.kind === 'baseline' ? 0 : item.kind === 'measured' ? groupActivityRate(item.rates, groupIndex) : undefined;
  const rate = valueFor(reading);
  const definition = groupIndex === null ? '已记录群组平均发放率的算术均值；群组可重叠，不代表全脑或去重神经元平均' : '所选群组已记录的平均发放率';
  element('activity-scope').textContent = groupSelect.selectedOptions[0]?.label ?? '已记录群组均值';
  element('activity-scope').title = definition;
  element('activity-definition').textContent = `${definition}。横轴为页面可见时的显示经过时间；空闲、结束或没有当前有效数据时，曲线回到显示零基线，不表示全脑静默。实测值最多保持 2 秒；缺失或过期仍由状态提示，基线切换不表示实测放电率的升降过程。${neuralDefinition}`;
  element('current-rate').textContent = rate === undefined ? '—' : rate.toFixed(1);
  element('current-rate').title = reading.kind === 'measured' ? definition : reading.kind === 'baseline' ? '0 Hz 显示基线；当前没有动作活动在展示。' : '当前活动数据缺失、过期或无法确认时间。';
  groupSelect.title = definition;
  const chart = document.getElementById('rate-chart')!;
  const grid = '<path d="M32 15H546M32 44H546M32 73H546" class="chart-grid"/><text x="14" y="77">0</text>';
  chart.setAttribute('aria-label', `${definition}曲线；${reading.kind === 'baseline' ? '当前为 0 Hz 显示基线' : reading.kind === 'gap' ? '当前数据缺口' : '当前为已记录活动'}；横轴为显示经过秒数`);
  const first = Math.max(0, elapsed - activityTimeline.windowMs);
  const points = activityTimeline.points;
  const peak = Math.max(0, ...points.map(point => valueFor(point.reading) ?? 0));
  const maximum = Math.max(1, Math.ceil(peak * 1.1 * 10) / 10);
  const x = (time: number) => 32 + Math.max(0, time - first) / Math.max(1, elapsed - first) * 514;
  const y = (value: number) => 73 - value / maximum * 58;
  let line = '', previousKind: ActivityReading['kind'] | null = null;
  for (const point of points) {
    // Missing telemetry stays missing. Only the display trace returns to its
    // baseline; never interpolate an activity ramp through unavailable time.
    const px = x(point.atMs).toFixed(2), py = y(valueFor(point.reading) ?? 0).toFixed(2);
    if (previousKind === null) line = `M${px},${py} `;
    else if (previousKind === 'measured' && point.reading.kind === 'measured') line += `L${px},${py} `;
    else line += `H${px}V${py} `;
    previousKind = point.reading.kind;
  }
  // Initial/idle zero is a presentation baseline, not a fabricated telemetry batch.
  if (points.length < 2 && reading.kind === 'baseline') line = 'M32,73L546,73';
  else if (points.length) line += 'H546';
  const point = rate === undefined ? '' : `<circle cx="546" cy="${y(rate)}" r="2.8" class="chart-dot"/>`;
  chart.innerHTML = `${grid}<text x="0" y="19">${maximum.toFixed(1)}</text><path d="${line}" class="chart-line"/>${point}`;
}
function scheduleActivity(): void {
  if (activityTimer !== null) clearTimeout(activityTimer);
  activityTimer = null;
  if (disposed || document.hidden) return;
  activityTimer = setTimeout(() => { activityTimer = null; drawActivity(); if (liveStatus.current) updateStatus(liveStatus.current); scheduleActivity(); }, 250);
}
function updateStatus(value: ExperimentStatus): void {
  const controlsPending = pending || stopPending;
  const brainActive = value.mode === 'brain_control' && value.brain_controlled === true;
  app.querySelectorAll<HTMLButtonElement>('[data-task]').forEach(button => {
    const task = button.dataset.task as Task;
    const selectedNext = value.next_task === task;
    const labels: Record<Task, string> = { SHOW_FORWARD: '前进', FACE_LEFT: '左转', FACE_RIGHT: '右转', GREET: '挥手', END_INTERACTION: '结束互动' };
    button.disabled = controlsPending || !canSubmitTask(value, task);
    button.classList.toggle('is-active', selectedNext || (!value.closing && value.action_busy === true && value.task === task));
    button.querySelector('span')!.textContent = labels[task] + (selectedNext ? ' · 待执行' : '');
    button.title = selectedNext ? '已选择，准备完成后自动执行；结束互动可取消' : value.can_queue_task && task !== 'END_INTERACTION' ? '选择下一动作，准备完成后自动执行' : value.action_busy && task !== 'END_INTERACTION' ? '当前动作结束后可选' : '向脑模型提交互动任务';
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
  const stale = snapshot?.transport_delay_ms !== null && snapshot?.transport_delay_ms !== undefined && snapshot.transport_delay_ms + performance.now() - snapshotReceivedAt > 2000;
  const live = activityDisplay.mode === 'active';
  const timeUnknown = live && current && brainActive && snapshot?.transport_delay_ms == null;
  const activityStatus = activityReadError ? '采集异常' : issue ? summary?.errors.length || summary?.error ? '采集异常' : '记录不完整' : activityDisplay.mode === 'closing' ? '已结束' : activityDisplay.mode === 'waiting' ? value.phase === 'waiting_task' ? '待命' : '等待任务' : !live ? stateLabels[value.state] : current ? brainActive ? timeUnknown ? '时间未知' : stale ? '等待新批次' : '正在更新' : '最后一批' : brainActive ? '等待遥测' : busy ? '等待脑就绪' : '未运行';
  element('brain-state').textContent = activityStatus;
  element('brain-state').classList.toggle('mint', live && !activityReadError && !issue && !timeUnknown && !stale && brainActive && Boolean(current));
  element('brain-state').title = activityReadError || issue || (!live ? activityDisplay.mode === 'closing' ? '动作已结束，可选择下一动作；准备完成后自动执行。' : '当前展示 0 Hz 基线，不代表全脑发放率。' : timeUnknown ? '主机时钟暂不可用，无法确认当前同步状态。' : snapshot?.transport_delay_ms !== null && snapshot?.transport_delay_ms !== undefined ? `主机取时点的批次年龄 ${(snapshot.transport_delay_ms / 1000).toFixed(2)} 秒；收到时间 ${new Date(snapshot.received_at_ms).toLocaleTimeString('zh-CN')}。显示更新频率不等于脑计算频率。` : '尚无本次控制脑遥测');
  element('brain-state').title += ` ${dataQualitySummary()}`;
  const captureIssue = snapshot?.capture && ['failed', 'incomplete'].includes(snapshot.capture.state) ? `画面采集失败：${snapshot.capture.reason}` : '';
  const videoIssue = snapshot?.video?.state === 'incomplete' ? `视频采集失败：${snapshot.video.reason}` : '';
  const statusNotice = [value.state === 'failed' ? value.message : '', issue, captureIssue, videoIssue].filter(Boolean).join('；');
  if (statusNotice && statusNotice !== lastStatusNotice) showError(statusNotice);
  else if (!statusNotice && element('notice').textContent === lastStatusNotice) clearError();
  lastStatusNotice = statusNotice;
}
function renderStatus(): void {
  if (!liveStatus.current) return;
  const previousMode = activityDisplay.mode;
  if (activityDisplay.update(liveStatus.current)) clearActivity();
  else if (previousMode !== activityDisplay.mode) drawActivity();
  updateStatus(liveStatus.current);
}
async function action(kind: 'prepare' | 'stop'): Promise<void> {
  if (!api || !liveStatus.current || (kind === 'stop' ? !canEmergencyStop(liveStatus.current, lifecyclePending, stopPending) : pending || stopPending)) return;
  const revision = liveStatus.beginChange();
  clearError();
  if (kind === 'stop') stopPending = true;
  else { pending = true; lifecyclePending = true; }
  renderStatus();
  try {
    if (kind === 'prepare') { manifest = null; summary = null; current = null; lastSequence = 0; displaySkipped = 0; activityReadError = ''; fullNeural = false; neuralMapping = null; neuralSignals = null; renderedNeuralSignals = null; neuralManifestError = ''; neuralFrameError = ''; checkedNeuralSnapshot = null; checkedNeuralBatch = null; activityDisplay.reset(); activityTimeline.reset(); brain.setGroups([]); renderedBrainBatch = null; groupSelect.replaceChildren(option('', '已记录群组均值')); groupSelect.disabled = true; drawActivity(); }
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
      const acceptedRevision = liveStatus.revision;
      snapshot = next; snapshotReceivedAt = performance.now(); activityReadError = next.read_error ?? '';
      // Status and emergency controls must update even while the window is hidden.
      renderStatus();
      if (!document.hidden) {
        if (next.manifest && next.manifest.run_id === next.status.run_id) await useManifest(next.manifest);
        if (liveStatus.revision !== acceptedRevision) return;
        useSummary(next.summary, next.status.run_id);
        if (next.batch) useBatch(next.batch);
        drawActivity();
      }
      renderStatus();
      if (next.read_error || next.simulation_error) showError([next.read_error, next.simulation_error, next.status.state === 'failed' ? next.status.message : '', telemetryIssue(next.status)].filter(Boolean).join('；'));
    } catch (error) { activityReadError = error instanceof Error ? error.message : String(error); drawActivity(); if (liveStatus.current) updateStatus(liveStatus.current); showError(error); }
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
  clearError(); pending = true;
  // Selecting a next action does not submit a new brain request yet.
  if (!liveStatus.current.can_queue_task || button.dataset.task === 'END_INTERACTION') {
    activityDisplay.beginRequest(); clearActivity();
  }
  renderStatus();
  try { const status = await api.task(button.dataset.task as Task); if (liveStatus.accept(status, revision)) renderStatus(); }
  catch (error) { activityDisplay.cancelRequest(); showError(error); }
  finally { pending = false; renderStatus(); requestPoll(); }
}));
element('start').addEventListener('click', () => {
  if (!api || !liveStatus.current) return;
  const kind = startAction(liveStatus.current, pending || stopPending);
  if (kind) void action(kind);
});
element('stop').addEventListener('click', () => { void action('stop'); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) activityTimeline.pause(); else activityTimeline.resume();
  scheduleActivity(); renderStatus(); drawActivity();
  if (!document.hidden) { setBounds(); requestPoll(); }
});
async function start(): Promise<void> {
  const [loadedAtlas, display] = await Promise.all([loadAtlas(), loadBrainDisplay()]);
  displayManifest = display.manifest;
  atlas = loadedAtlas; brain = new BrainView(element('brain'), atlas, display, element('fly-orientation'));
  const graph = await loadBrainGraphInfo(atlas.node_count, display.manifest);
  element('brain-neuron-count').textContent = graph.node_count.toLocaleString('en-US');
  element('brain-edge-count').textContent = graph.edge_count.toLocaleString('en-US');
  element('reset-view').addEventListener('click', () => brain.reset()); groupSelect.addEventListener('change', () => drawActivity());
  window.addEventListener('beforeunload', () => { disposed = true; if (noticeTimer !== null) clearTimeout(noticeTimer); if (pollTimer !== null) clearTimeout(pollTimer); if (activityTimer !== null) clearTimeout(activityTimer); brain.dispose(); });
  if (document.hidden) activityTimeline.pause();
  drawActivity(); scheduleActivity();
  if (!api) { element<HTMLButtonElement>('start').disabled = true; showError('请在 Electron 桌面应用中使用实验操作。'); return; }
  setBounds(); requestPoll();
}
start().catch(error => { showError(error); app.querySelectorAll<HTMLButtonElement | HTMLSelectElement>('button, select').forEach(control => { control.disabled = true; }); });

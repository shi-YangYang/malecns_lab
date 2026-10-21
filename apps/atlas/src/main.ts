import './style.css';
import { loadDataset } from './data';
import { BrainView } from './brain';
import { FlyView } from './fly';
import { drawRateChart } from './chart';
import { ExperimentPanel, stimulusLabels, experimentLabel, parameterDetail } from './experiments';
import type { Job } from './experiments';
import type { Dataset, Run } from './types';

const icon = (name: string, size = 18): string => {
  const paths: Record<string, string> = {
    play: '<path d="m8 5 11 7-11 7V5Z"/>', pause: '<path d="M8 5v14M16 5v14"/>',
    reset: '<path d="M4 10a8 8 0 1 1 2 8M4 4v6h6"/>',
    back: '<path d="M5 5v14M18 5 7 12l11 7V5Z"/>', next: '<path d="M19 5v14M6 5l11 7-11 7V5Z"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.1"/>',
    expand: '<path d="M5 9V5h4M15 5h4v4M19 15v4h-4M9 19H5v-4"/>',
    static: '<path d="M5 5h5v5H5zM14 5h5v5h-5zM5 14h5v5H5zM14 14h5v5h-5z"/>',
    left: '<path d="m10 5-7 7 7 7M3 12h18"/>', right: '<path d="m14 5 7 7-7 7M3 12h18"/>',
    up: '<path d="m5 10 7-7 7 7M12 3v18"/>', down: '<path d="m5 14 7 7 7-7M12 3v18"/>',
    expansion: '<circle cx="12" cy="12" r="3"/><path d="m3 8 0-5 5 0M3 3l5 5M16 3h5v5m0-5-5 5M3 16v5h5m-5 0 5-5m8 5h5v-5m0 5-5-5"/>',
    brain: '<path d="M12 4C7 0 2 5 4 9c-5 4-1 10 4 9 0 3 4 3 4 0V4Zm0 0c5-4 10 1 8 5 5 4 1 10-4 9 0 3-4 3-4 0M7 8l3 2M17 8l-3 2M6 14l4-1m8 1-4-1"/>',
  };
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] ?? paths.info}</svg>`;
};
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="topbar">
    <a class="brand" href="/" aria-label="MaleCNS Lab 首页"><span class="brand-mark">${icon('brain', 24)}</span><span>MaleCNS <b>Lab</b></span></a>
    <button class="text-button" id="sources-button">${icon('info', 16)} 数据来源</button>
  </header>
  <main>
    <section class="page-heading">
      <h1>神经活动图谱</h1>
      <div class="mode-badge"><span class="status-dot"></span><span id="view-mode">已有记录 · 仿真</span></div>
    </section>
    <section id="experiments" class="panel experiments" aria-label="完整脑实验"></section>
    <div class="view-heading"><p id="view-description">已有刺激记录</p><span id="view-loading" role="status" hidden>正在载入结果…</span></div>
    <p id="view-error" class="view-error" role="alert" hidden></p>
    <div id="load-state" class="load-state" role="status"><div class="loading-symbol"></div><h2>正在载入图谱…</h2></div>
    <div id="dashboard" hidden>
      <section class="workspace" aria-label="神经图谱研究工作区">
        <aside class="left-panel">
          <section class="panel stimulus-panel">
            <div class="panel-heading"><h2 id="stimulus-title">视觉刺激</h2></div>
            <div id="stimulus-list" class="stimulus-list" role="group" aria-label="选择刺激"></div>
            <div class="preview-heading"><span id="preview-title">刺激画面</span><span class="small-tag" id="preview-kind">重建</span></div>
            <div class="stimulus-preview" id="stimulus-preview"><canvas id="stimulus-canvas" aria-label="当前记录帧对应的重建刺激"></canvas><span id="preview-message">正在载入重建画面</span><div class="preview-corners" aria-hidden="true"></div></div>
          </section>
          <section class="panel reference-panel">
            <div class="panel-heading"><h2>图谱概况</h2></div>
            <dl class="reference-stats"><div><dt>神经元</dt><dd id="node-count">—</dd></div><div><dt>有向连接</dt><dd id="edge-count">—</dd></div></dl>
            <div class="coverage-line"><span>位置覆盖</span><strong id="coverage-percent">—</strong></div><div class="coverage-bar"><span id="coverage-bar"></span></div>
            <div class="coverage-caption"><span id="positioned-count">—</span><span id="missing-count">—</span></div>
          </section>
        </aside>
        <section class="brain-panel panel" aria-labelledby="brain-heading">
          <div class="brain-top"><h2 id="brain-heading">MaleCNS 脑图</h2><button class="icon-button" id="reset-view" aria-label="复位脑图视角" title="复位脑图视角">${icon('expand')}</button></div>
          <div class="brain-canvas-container" id="brain-view"></div>
          <div class="brain-empty" id="brain-empty" hidden></div>
          <div class="brain-bottom"><div class="legend-heading"><span>群组平均放电率</span><span class="unit">Hz</span></div><div class="color-scale"></div><div class="legend-ticks"><span>0</span><span id="legend-mid">—</span><span id="legend-max">—</span></div><p>同组共用均值，非逐神经元实测</p></div>
          <div class="view-axis" aria-hidden="true"><span class="axis-x">X</span><span class="axis-y">Y</span><span class="axis-z">Z</span><i></i></div>
        </section>
        <aside class="right-panel">
          <section class="panel activity-panel">
            <div class="panel-heading"><h2>群组活动</h2></div>
            <label class="sr-only" for="group-select">选择神经群组</label><select id="group-select" aria-label="选择神经群组"></select>
            <div class="rate-line"><div><strong id="current-rate" data-testid="current-rate">—</strong><span>Hz</span></div></div>
            <div id="group-coverage" class="group-coverage">—</div>
            <svg id="rate-chart" class="rate-chart" role="img"></svg>
            <div class="chart-caption"><label class="switch-label"><input type="checkbox" id="highlight-group"> 突出所选群组</label></div>
            <div class="network-stats"><div><span>本帧脉冲</span><strong id="network-spikes">—</strong></div><div><span>活跃神经元</span><strong id="active-neurons">—</strong></div></div>
          </section>
          <section class="panel body-panel">
            <div class="panel-heading"><h2>果蝇</h2><span class="small-tag" id="body-kind">静态示意</span></div>
            <div id="fly-view" class="fly-view"></div>
          </section>
        </aside>
      </section>
      <section class="timeline panel" aria-label="记录回放控制">
        <div class="timeline-top"><div class="timeline-title"><h2>回放</h2><span id="phase-badge" class="phase-badge">预热</span><span id="frame-validity" class="frame-validity"></span></div><div class="time-readout"><strong id="time-value" data-testid="time-value">0.00</strong><span>秒</span><span class="time-total" id="time-total">/ 2.50 秒</span><span class="frame-readout" id="frame-readout">第 — / — 帧</span></div></div>
        <div class="timeline-main"><div class="playback-buttons"><button class="icon-button" id="previous-frame" aria-label="上一记录帧">${icon('back')}</button><button class="play-button" id="play-button" aria-label="播放回放">${icon('play')}</button><button class="icon-button" id="next-frame" aria-label="下一记录帧">${icon('next')}</button></div><div class="scrubber"><label class="sr-only" for="timeline-slider">记录帧时间轴</label><div class="timeline-track"><div id="warmup-band" class="warmup-band"></div><input type="range" id="timeline-slider" aria-label="记录帧时间轴" min="0" max="49" step="1" value="0"></div><div class="timeline-labels"><span id="timeline-start">0.00 秒</span><span id="warmup-label">预热 0–2.00 秒</span><span id="timeline-end">2.50 秒</span></div></div><label class="speed-control"><span>速度</span><select id="playback-speed" aria-label="回放速度"><option value="0.25" selected>0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option></select></label><button class="icon-button restart-button" id="restart-playback" aria-label="从头回放" title="从头回放">${icon('reset')}</button></div>
        <p id="playback-note" class="playback-note" role="status"></p>
      </section>
    </div>
    <footer><span>MaleCNS v1.0 · <a href="https://male-cns.janelia.org/" target="_blank" rel="noreferrer">CC BY 4.0 ↗</a></span></footer>
  </main>
  <dialog id="sources-dialog" aria-labelledby="sources-title"><div class="dialog-heading"><h2 id="sources-title">数据来源</h2><button class="icon-button" id="close-dialog" aria-label="关闭数据来源">×</button></div><div id="source-content"></div></dialog>`;

const el = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const text = (id: string, value: string): void => { el(id).textContent = value; };
const number = (value: number | null | undefined): string => value === null || value === undefined ? '—' : value.toLocaleString('en-US');
const sourcesDialog = el<HTMLDialogElement>('sources-dialog');
el('sources-button').addEventListener('click', () => sourcesDialog.showModal());
el('close-dialog').addEventListener('click', () => sourcesDialog.close());
sourcesDialog.addEventListener('click', event => { if (event.target === sourcesDialog) sourcesDialog.close(); });
el('source-content').textContent = '正在读取本地数据来源。';

class AtlasApp {
  private run: Run;
  private frame = 0;
  private playing = false;
  private speed = 0.25;
  private elapsed = 0;
  private previousTimestamp = 0;
  private animation = 0;
  private selectedGroup = '';
  private brain: BrainView | null = null;
  private fly: FlyView | null = null;
  private previewImages = new Map<string, HTMLImageElement>();
  private previewFailures = new Set<string>();
  private buttons = new Map<string, HTMLButtonElement>();
  private previewGeneration = 0;
  private readonly events = new AbortController();
  constructor(private data: Dataset) {
    this.run = data.runs.find(run => run.stimulus === 'expansion' && run.time_s.length > 0) ?? data.runs[0];
    this.populate();
    el('load-state').hidden = true; el('dashboard').hidden = false;
    this.createBrain();
    try { this.fly = new FlyView(el('fly-view')); } catch { el('fly-view').textContent = 'WebGL 不可用 · 身体示意未加载'; }
    this.bind(); this.changeRun(this.run.id, true);
    this.animation = requestAnimationFrame(this.tick);
    window.addEventListener('pagehide', () => this.dispose(), { once: true, signal: this.events.signal });
  }
  setDataset(data: Dataset): void {
    this.setPlaying(false); this.previewGeneration++;
    this.data = data; this.run = data.runs.find(run => run.stimulus === 'expansion') ?? data.runs[0];
    this.previewImages.clear(); this.previewFailures.clear(); this.buttons.clear();
    el('stimulus-list').replaceChildren(); el('group-select').replaceChildren();
    this.brain?.dispose(); this.brain = null; el('brain-view').replaceChildren(); el('brain-empty').hidden = true;
    this.populate(); this.createBrain(); this.changeRun(this.run.id, true);
  }
  private createBrain(): void {
    try {
      if (this.data.atlas.positioned_count) this.brain = new BrainView(el('brain-view'), this.data.atlas);
      else this.brainError('没有可显示的胞体位置。参考节点仍保留在统计中。');
    } catch (error) { this.brainError(`WebGL 暂不可用：${error instanceof Error ? error.message : String(error)}`); }
  }
  dispose(): void {
    cancelAnimationFrame(this.animation); this.events.abort(); this.previewGeneration++;
    this.brain?.dispose(); this.fly?.dispose(); this.previewImages.clear();
  }
  private brainError(message: string): void { text('brain-empty', message); el('brain-empty').hidden = false; }
  private populate(): void {
    const { catalog, atlas, maxRate } = this.data;
    text('node-count', number(catalog.graph.node_count)); text('edge-count', number(catalog.graph.edge_count));
    const fraction = atlas.node_count > 0 ? atlas.positioned_count / atlas.node_count : 0;
    text('coverage-percent', `${(fraction * 100).toFixed(1)}%`); el('coverage-bar').style.width = `${fraction * 100}%`;
    text('positioned-count', `已显示 ${number(atlas.positioned_count)}`); text('missing-count', `缺坐标 ${number(atlas.missing_count)}`);
    text('legend-max', maxRate.toFixed(1)); text('legend-mid', (maxRate / 2).toFixed(1));
    const labels = stimulusLabels;
    for (const run of this.data.runs) {
      const button = document.createElement('button'); button.className = 'stimulus-button';
      const label = labels[run.stimulus] ?? run.label;
      button.innerHTML = `${icon(run.stimulus, 19)}<span></span>`;
      button.querySelector('span')!.textContent = label;
      button.dataset.runId = run.id; button.setAttribute('aria-label', `选择${label}刺激`);
      button.addEventListener('click', () => this.changeRun(run.id, true));
      this.buttons.set(run.id, button); el('stimulus-list').append(button);
    }
    const select = el<HTMLSelectElement>('group-select');
    for (const group of atlas.groups) { const option = document.createElement('option'); option.value = group.name; option.textContent = `${group.name}${group.total_neurons === 0 ? ' · 空群组' : ''}`; select.append(option); }
    this.selectedGroup = atlas.groups.find(group => group.name === this.selectedGroup)?.name ?? atlas.groups.find(group => group.name === 'R16_L')?.name ?? atlas.groups[0]?.name ?? '';
    select.value = this.selectedGroup; select.disabled = atlas.groups.length === 0;
    this.populateSource();
  }
  private populateSource(): void {
    const content = el('source-content'); content.replaceChildren();
    for (const [title, description] of [
      ['模型', '固定 LIF 完整参考模型的诊断仿真，不代表活体测量或已训练策略。'],
      ['脑图', '来自 MaleCNS v1.0 官方胞体位置，未包含神经形态与突触连线。缺坐标的神经元计入总数，不在图中显示。'],
      ['活动与颜色', '颜色表示群组平均放电率，已查看的记录共用色标。载入更高响应时，色标上限随之扩大；具体比较请以 Hz 数值为准。无记录的节点保持背景色，重叠群组取最大均值。'],
      ['刺激与回放', this.run.stimulus_preview?.kind !== 'reconstructed' && this.run.stimulus_preview ? '画面是本次计算使用的实际输入。按仿真时间逐帧回放，灰色区间为预热；计算耗时与仿真时长分别记录。' : '画面按原记录参数重建，非摄像头录像。按原始仿真时间逐帧回放，灰色区间为预热。'],
      ['果蝇示意', '仅展示静态形态，未由神经活动驱动，也未接入实体设备。'],
    ]) { const section = document.createElement('section'); const h3 = document.createElement('h3'); h3.textContent = title; const p = document.createElement('p'); p.textContent = description; section.append(h3, p); content.append(section); }
    const attribution = document.createElement('p'); attribution.className = 'source-attribution';
    const link = document.createElement('a'); link.href = 'https://male-cns.janelia.org/'; link.target = '_blank'; link.rel = 'noreferrer'; link.textContent = 'MaleCNS v1.0 · CC BY 4.0 ↗';
    attribution.append(link); content.append(attribution);
  }
  private bind(): void {
    const options = { signal: this.events.signal };
    el('reset-view').addEventListener('click', () => this.brain?.reset(), options);
    el<HTMLSelectElement>('group-select').addEventListener('change', event => { this.selectedGroup = (event.target as HTMLSelectElement).value; this.update(); }, options);
    el('highlight-group').addEventListener('change', () => this.update(), options);
    el('play-button').addEventListener('click', () => {
      if (this.frame === this.run.time_s.length - 1 && !this.playing) this.frame = 0;
      this.setPlaying(!this.playing); this.update();
    }, options);
    el('previous-frame').addEventListener('click', () => { this.setPlaying(false); this.setFrame(this.frame - 1); }, options);
    el('next-frame').addEventListener('click', () => { this.setPlaying(false); this.setFrame(this.frame + 1); }, options);
    el('restart-playback').addEventListener('click', () => { this.frame = 0; this.setPlaying(true); this.update(); }, options);
    el<HTMLInputElement>('timeline-slider').addEventListener('input', event => { this.setPlaying(false); this.setFrame(Number((event.target as HTMLInputElement).value)); }, options);
    el<HTMLSelectElement>('playback-speed').addEventListener('change', event => { this.speed = Number((event.target as HTMLSelectElement).value); this.elapsed = 0; }, options);
    document.addEventListener('visibilitychange', () => { this.previousTimestamp = 0; }, options);
  }
  private changeRun(id: string, observation: boolean): void {
    this.setPlaying(false); this.run = this.data.runs.find(run => run.id === id)!;
    this.frame = observation ? Math.max(0, this.run.time_s.findIndex((time, index) => time >= this.run.warmup_s && this.run.valid[index])) : 0;
    for (const [buttonId, button] of this.buttons) button.setAttribute('aria-pressed', String(buttonId === id));
    el<HTMLInputElement>('timeline-slider').max = String(Math.max(0, this.run.time_s.length - 1));
    for (const id of ['play-button', 'previous-frame', 'next-frame', 'timeline-slider', 'restart-playback']) (el(id) as HTMLButtonElement | HTMLInputElement).disabled = this.run.time_s.length === 0;
    if (this.data.atlas.positioned_count) { el('brain-empty').hidden = this.brain !== null; }
    if (this.run.time_s.length === 0) { text('playback-note', '此刺激没有记录帧，无法播放'); }
    else text('playback-note', '');
    this.loadPreview(); this.populateSource(); this.update();
  }
  private loadPreview(): void {
    const id = this.run.id, url = this.data.previewUrls.get(id), generation = this.previewGeneration;
    if (!url || this.previewImages.has(id) || this.previewFailures.has(id)) return;
    const image = new Image();
    image.onload = () => {
      if (generation !== this.previewGeneration) return;
      const run = this.data.runs.find(item => item.id === id)!;
      if (image.naturalWidth !== run.stimulus_preview?.width || image.naturalHeight !== run.stimulus_preview.height * run.stimulus_preview.frames) this.previewFailures.add(id);
      else this.previewImages.set(id, image);
      if (this.run.id === id) this.drawPreview();
    };
    image.onerror = () => { if (generation !== this.previewGeneration) return; this.previewFailures.add(id); if (this.run.id === id) this.drawPreview(); }; image.src = url;
  }
  private drawPreview(): void {
    const canvas = el<HTMLCanvasElement>('stimulus-canvas'); const context = canvas.getContext('2d');
    const image = this.previewImages.get(this.run.id), preview = this.run.stimulus_preview;
    const generated = preview?.kind === 'generated';
    text('preview-kind', generated ? '实际输入' : '重建');
    canvas.setAttribute('aria-label', generated ? '当前帧实际输入画面' : '当前帧重建刺激画面');
    const available = Boolean(image && preview && this.run.time_s.length > 0);
    canvas.hidden = !available; el('preview-message').hidden = available;
    if (available && image && preview && context) {
      canvas.width = preview.width; canvas.height = preview.height;
      context.drawImage(image, 0, this.frame * preview.height, preview.width, preview.height, 0, 0, preview.width, preview.height);
    } else text('preview-message', this.previewFailures.has(this.run.id) ? '刺激预览无法显示' : !preview ? '此记录没有刺激预览' : this.run.time_s.length === 0 ? '此刺激没有记录帧' : '正在读取刺激画面…');
  }
  private setPlaying(value: boolean): void {
    this.playing = value && this.run.time_s.length > 0; this.elapsed = 0; this.previousTimestamp = 0;
    el('play-button').innerHTML = icon(this.playing ? 'pause' : 'play');
    el('play-button').setAttribute('aria-label', this.playing ? '暂停回放' : '播放回放');
    el('play-button').setAttribute('aria-pressed', String(this.playing));
  }
  private setFrame(value: number): void { this.frame = Math.max(0, Math.min(this.run.time_s.length - 1, value)); this.update(); }
  private tick = (timestamp: number): void => {
    if (this.playing && this.previousTimestamp && !document.hidden) {
      this.elapsed += Math.min(timestamp - this.previousTimestamp, 250) / 1000 * this.speed;
      let changed = false;
      while (this.frame < this.run.time_s.length - 1) {
        const interval = this.run.time_s[this.frame + 1] - this.run.time_s[this.frame];
        if (this.elapsed < interval) break;
        this.elapsed -= interval; this.frame++; changed = true;
      }
      if (changed) this.update();
      if (this.frame === this.run.time_s.length - 1) this.setPlaying(false);
    }
    this.previousTimestamp = timestamp; this.animation = requestAnimationFrame(this.tick);
  };
  private update(): void {
    const run = this.run, hasFrame = run.time_s.length > 0;
    const time = run.time_s[this.frame] ?? 0, end = run.time_s.at(-1) ?? 0;
    const valid = hasFrame && run.valid[this.frame];
    const group = this.data.atlas.groups.find(item => item.name === this.selectedGroup);
    const column = run.groups.indexOf(this.selectedGroup);
    const rate = valid && column >= 0 ? run.group_rates_hz[this.frame]?.[column] : null;
    text('current-rate', rate === null || rate === undefined ? '—' : rate.toFixed(1));
    text('group-coverage', !group ? '没有可选择的群组' : group.total_neurons === 0 ? '空群组 · 没有神经元成员' : `${number(group.total_neurons)} 个神经元 · ${number(group.positioned_neurons)} 个有位置${column < 0 ? ' · 本刺激无记录' : ''}`);
    text('network-spikes', valid ? number(run.network_spikes[this.frame]) : '—'); text('active-neurons', valid ? number(run.network_active_neurons[this.frame]) : '—');
    text('time-value', hasFrame ? time.toFixed(2) : '—'); text('time-total', `/ ${end.toFixed(2)} 秒`);
    text('frame-readout', hasFrame ? `第 ${this.frame + 1} / ${run.time_s.length} 帧` : '无记录帧');
    const warmup = time < run.warmup_s;
    text('phase-badge', !hasFrame ? '无记录' : warmup ? '预热' : '观察'); el('phase-badge').classList.toggle('observation', !warmup && hasFrame);
    text('frame-validity', !hasFrame || valid ? '' : `此帧数据无效${run.invalid_reason?.[this.frame] ? ` · ${run.invalid_reason[this.frame]}` : ''}`);
    el('frame-validity').classList.toggle('invalid', hasFrame && !valid);
    const slider = el<HTMLInputElement>('timeline-slider'); slider.value = String(this.frame);
    slider.setAttribute('aria-valuetext', hasFrame ? `记录第 ${this.frame + 1} 帧，仿真 ${time.toFixed(2)} 秒，${warmup ? '预热' : '观察'}阶段` : '无记录帧');
    const start = run.time_s[0] ?? 0;
    el('warmup-band').style.width = `${Math.max(0, Math.min(100, (run.warmup_s - start) / Math.max(end - start, 1 / run.frame_hz) * 100))}%`;
    text('timeline-start', `${start.toFixed(2)} 秒`); text('timeline-end', `${end.toFixed(2)} 秒`); text('warmup-label', `预热至 ${run.warmup_s.toFixed(2)} 秒`);
    this.brain?.update(run, this.frame, this.selectedGroup, this.data.maxRate, el<HTMLInputElement>('highlight-group').checked);
    drawRateChart(document.getElementById('rate-chart') as unknown as SVGSVGElement, run, this.selectedGroup, this.frame, this.data.maxRate);
    this.drawPreview();
  }
}

let viewer: AtlasApp | null = null;
let viewRequest = 0;
let sharedMaxRate = 1;

async function openDataset(job: Job | null): Promise<boolean> {
  const requestId = ++viewRequest;
  el('view-loading').hidden = false; el('view-error').hidden = true;
  try {
    const data = await loadDataset(job?.catalog_url ?? '/data/catalog.json');
    if (requestId !== viewRequest) return false;
    sharedMaxRate = Math.max(sharedMaxRate, data.maxRate); data.maxRate = sharedMaxRate;
    if (viewer) viewer.setDataset(data); else viewer = new AtlasApp(data);
    text('view-mode', job ? '实验结果 · 仿真' : '已有记录 · 仿真');
    text('view-description', job
      ? `${experimentLabel(job.parameters)} · ${parameterDetail(job.parameters)} · 观察 ${job.parameters.duration_s.toFixed(2)} 秒`
      : '已有刺激记录');
    return true;
  } catch (error) {
    if (requestId !== viewRequest) return false;
    const message = error instanceof Error ? error.message : String(error);
    text('view-error', `结果载入失败：${message}`); el('view-error').hidden = false;
    if (!viewer) {
      const state = el('load-state'); state.setAttribute('role', 'status'); state.replaceChildren();
      const heading = document.createElement('h2'); heading.textContent = '暂未载入图谱';
      const instruction = document.createElement('p'); instruction.textContent = '可运行新实验，或重试读取已有记录。';
      const reload = document.createElement('button'); reload.className = 'retry-button'; reload.textContent = '重试载入';
      reload.addEventListener('click', () => { void openDataset(job); });
      state.append(heading, instruction, reload);
      el('source-content').textContent = '数据尚未载入。';
    }
    return false;
  } finally { if (requestId === viewRequest) el('view-loading').hidden = true; }
}

new ExperimentPanel(openDataset);
void openDataset(null);

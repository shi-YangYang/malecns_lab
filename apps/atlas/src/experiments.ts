export const stimulusLabels: Record<string, string> = {
  static: '静止', left: '向左', right: '向右', up: '向上', down: '向下', expansion: '扩张',
};
type JobStatus = 'queued' | 'loading' | 'running' | 'exporting' | 'completed' | 'cancelled' | 'failed';
export interface Parameters { stimulus: string; speed: number; contrast: number; duration_s: number }
export const experimentLabel = (parameters: Parameters): string => stimulusLabels[parameters.stimulus];
export const parameterDetail = (parameters: Parameters): string => `速度 ${parameters.speed}× · 对比度 ${Math.round(parameters.contrast * 100)}%`;
export interface Job {
  id: string; status: JobStatus; parameters: Parameters; created_at: string;
  frames_completed: number; total_frames: number; simulation_s: number; elapsed_s: number;
  error: string | null; catalog_url: string | null;
}
const statusLabels: Record<JobStatus, string> = {
  queued: '准备中', loading: '载入模型', running: '计算中', exporting: '整理结果',
  completed: '已完成', cancelled: '已取消', failed: '未完成',
};
const terminal = (job: Job): boolean => ['completed', 'cancelled', 'failed'].includes(job.status);
const node = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const errorText = (error: unknown): string => error instanceof Error ? error.message : '请求失败，请重试。';
const seconds = (value: number): string => value < 60 ? `${value.toFixed(1)} 秒` : `${Math.floor(value / 60)} 分 ${Math.floor(value % 60)} 秒`;

function parametersData(raw: unknown): Parameters {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('实验参数不完整，请重试。');
  const value = raw as Record<string, unknown>;
  const within = (value: unknown, min: number, max: number): value is number => typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
  const keys = ['stimulus', 'speed', 'contrast', 'duration_s'];
  if (Object.keys(value).length !== keys.length || !keys.every(key => Object.hasOwn(value, key))
    || !within(value.duration_s, 0.1, 3)
    || Math.abs(value.duration_s * 20 - Math.round(value.duration_s * 20)) > 1e-8
    || typeof value.stimulus !== 'string' || !Object.hasOwn(stimulusLabels, value.stimulus)
    || !within(value.speed, 0.25, 2) || !within(value.contrast, 0.25, 1)) {
    throw new Error('实验参数格式或范围无效，请重试。');
  }
  return value as unknown as Parameters;
}

function jobData(raw: unknown): Job {
  if (!raw || typeof raw !== 'object') throw new Error('实验状态无法读取，请重试。');
  const job = raw as Job;
  const parameters = parametersData(job.parameters);
  const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
  if (typeof job.id !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(job.id) || !Object.hasOwn(statusLabels, job.status)
    || ![job.frames_completed, job.total_frames, job.simulation_s, job.elapsed_s].every(finite)
    || !Number.isInteger(job.frames_completed) || !Number.isInteger(job.total_frames)
    || job.frames_completed > job.total_frames || typeof job.created_at !== 'string'
    || !Number.isFinite(Date.parse(job.created_at))
    || !(job.error === null || typeof job.error === 'string')
    || !(job.catalog_url === null || (typeof job.catalog_url === 'string' && job.catalog_url.length > 0))
    || job.total_frames !== 40 + Math.round(parameters.duration_s * 20)
    || (job.status === 'completed' && typeof job.catalog_url !== 'string')) {
    throw new Error('实验状态不完整，请重试。');
  }
  return job;
}

async function request(path: string, body?: unknown): Promise<unknown> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json', 'X-MaleCNS-Request': '1' },
      body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
    });
    let value: unknown;
    try { value = await response.json(); } catch { throw new Error('实验服务未连接，请确认本地服务已启动后重试。'); }
    if (!response.ok) {
      const message = value && typeof value === 'object' && 'error' in value ? value.error : null;
      throw new Error(typeof message === 'string' ? message : `请求未成功（${response.status}），请重试。`);
    }
    return value;
  } catch (error) {
    if (error instanceof TypeError || (error instanceof DOMException && error.name === 'AbortError')) {
      throw new Error('实验服务暂时无法连接，请重试。');
    }
    throw error;
  } finally { window.clearTimeout(timer); }
}

export class ExperimentPanel {
  private jobs: Job[] = [];
  private current: Job | null = null;
  private busy = false;
  private polling = false;
  private timer = 0;
  private disposed = false;
  private selectedId: string | null = null;
  private historySignature = '';
  private readonly events = new AbortController();

  constructor(private open: (job: Job | null) => Promise<boolean>) {
    node('experiments').innerHTML = `
      <div class="experiment-heading"><h2>新实验</h2><button id="open-recordings" class="text-button" type="button">已有记录</button></div>
      <form id="experiment-form" class="experiment-form">
        <label>刺激<select id="experiment-stimulus" name="stimulus" aria-label="实验刺激">${Object.entries(stimulusLabels).map(([value, label]) => `<option value="${value}"${value === 'expansion' ? ' selected' : ''}>${label}</option>`).join('')}</select></label>
        <label>速度倍率<input id="experiment-speed" name="speed" type="number" min="0.25" max="2" step="0.25" value="1" required></label>
        <label>对比度<input id="experiment-contrast" name="contrast" type="number" min="0.25" max="1" step="0.05" value="1" required></label>
        <label>观察时长（秒）<input id="experiment-duration" name="duration_s" type="number" min="0.1" max="3" step="0.05" value="0.5" required></label>
        <div class="experiment-actions"><button id="run-experiment" class="primary-button" type="submit">运行实验</button><button id="cancel-experiment" class="secondary-button" type="button" hidden>取消</button></div>
      </form>
      <p class="experiment-limit" id="experiment-limit">完整脑仿真 · 含 2 秒预热，计算耗时另计</p>
      <div id="experiment-progress" class="experiment-progress" hidden>
        <div class="progress-heading"><strong id="experiment-status" role="status"></strong><span id="experiment-elapsed"></span></div>
        <progress id="experiment-progress-bar" max="1" value="0" aria-label="实验计算进度"></progress>
        <div class="progress-caption"><span id="experiment-frames"></span><span id="experiment-simulation"></span></div>
      </div>
      <div id="experiment-error" class="experiment-error" role="alert" hidden><span id="experiment-error-text"></span><button type="button" id="retry-connection" class="text-button">重试连接</button></div>
      <details id="experiment-history" class="experiment-history"><summary>最近实验 <span id="experiment-count"></span></summary><ul id="experiment-list"></ul><p id="experiment-empty">暂无实验</p></details>`;
    const options = { signal: this.events.signal };
    node('experiment-form').addEventListener('submit', event => { event.preventDefault(); void this.start(); }, options);
    node('cancel-experiment').addEventListener('click', () => { void this.cancel(); }, options);
    node('retry-connection').addEventListener('click', () => { void this.refresh(); }, options);
    node('open-recordings').addEventListener('click', () => { void this.show(null); }, options);
    window.addEventListener('pagehide', () => this.dispose(), { once: true, signal: this.events.signal });
    void this.refresh();
  }

  private showError(message: string): void {
    node('experiment-error-text').textContent = message;
    node('experiment-error').hidden = !message;
  }

  private controls(): void {
    const active = this.current !== null && !terminal(this.current);
    node<HTMLButtonElement>('run-experiment').disabled = this.busy || active;
    node('cancel-experiment').hidden = !active;
    node<HTMLButtonElement>('cancel-experiment').disabled = this.busy;
    for (const field of node<HTMLFormElement>('experiment-form').querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')) field.disabled = this.busy || active || Boolean(field.closest('[hidden]'));
  }

  private update(job: Job): void {
    if (this.current?.id === job.id && terminal(this.current) && this.current.status !== job.status) return;
    this.current = job;
    this.jobs = [job, ...this.jobs.filter(item => item.id !== job.id)]
      .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
    node('experiment-progress').hidden = false;
    node('experiment-status').textContent = `${experimentLabel(job.parameters)} · ${statusLabels[job.status]}`;
    node('experiment-elapsed').textContent = `实际耗时 ${seconds(job.elapsed_s)}`;
    node('experiment-frames').textContent = `${job.frames_completed} / ${job.total_frames} 帧`;
    node('experiment-simulation').textContent = `已模拟 ${job.simulation_s.toFixed(2)} 秒`;
    const progress = node<HTMLProgressElement>('experiment-progress-bar');
    progress.max = Math.max(job.total_frames, 1); progress.value = job.frames_completed;
    if (job.error) this.showError(job.error);
    this.controls(); this.history();
  }

  private history(): void {
    const signature = JSON.stringify([this.selectedId, this.jobs.map(job => [job.id, job.status, job.parameters, job.created_at, job.error])]);
    if (signature === this.historySignature) return;
    this.historySignature = signature;
    const list = node('experiment-list'); list.replaceChildren();
    node('experiment-count').textContent = this.jobs.length ? String(this.jobs.length) : '';
    node('experiment-empty').hidden = this.jobs.length > 0;
    for (const job of this.jobs.slice(0, 12)) {
      const row = document.createElement('li');
      const description = document.createElement('div');
      const title = document.createElement('strong');
      title.textContent = `${experimentLabel(job.parameters)} · ${job.parameters.duration_s.toFixed(2)} 秒`;
      const detail = document.createElement('span');
      detail.textContent = `${new Date(job.created_at).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })} · ${parameterDetail(job.parameters)}`;
      description.append(title, detail);
      const action = document.createElement('button'); action.type = 'button'; action.className = 'history-action';
      action.dataset.experimentId = job.id;
      action.textContent = job.id === this.selectedId ? '正在查看' : job.status === 'completed' ? '查看结果' : statusLabels[job.status];
      action.disabled = job.status !== 'completed' || job.id === this.selectedId;
      action.addEventListener('click', () => { void this.show(job); });
      if (job.error) { const failure = document.createElement('span'); failure.className = 'history-error'; failure.textContent = job.error; description.append(failure); }
      row.append(description, action); list.append(row);
    }
  }

  private async show(job: Job | null): Promise<void> {
    if (await this.open(job)) { this.selectedId = job?.id ?? null; this.history(); }
  }

  private async refresh(): Promise<void> {
    if (this.disposed || this.busy) return;
    this.busy = true; this.controls(); this.showError('');
    try {
      const raw = await request('/api/experiments');
      if (!raw || typeof raw !== 'object' || !('experiments' in raw) || !Array.isArray(raw.experiments)) throw new Error('实验列表无法读取，请重试。');
      if (this.disposed) return;
      this.jobs = raw.experiments.map(jobData).sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
      const active = this.jobs.find(job => !terminal(job));
      if (active) { this.update(active); this.schedule(); }
      else if (this.current) {
        const latest = this.jobs.find(job => job.id === this.current?.id);
        if (latest) this.update(latest);
      }
      this.history();
    } catch (error) { if (!this.disposed) this.showError(errorText(error)); }
    finally { this.busy = false; if (!this.disposed) this.controls(); }
  }

  private async start(): Promise<void> {
    if (this.busy || (this.current && !terminal(this.current))) return;
    const form = node<HTMLFormElement>('experiment-form'); if (!form.reportValidity()) return;
    const parameters: Parameters = {
      stimulus: node<HTMLSelectElement>('experiment-stimulus').value,
      speed: node<HTMLInputElement>('experiment-speed').valueAsNumber,
      contrast: node<HTMLInputElement>('experiment-contrast').valueAsNumber,
      duration_s: node<HTMLInputElement>('experiment-duration').valueAsNumber,
    };
    this.busy = true; this.controls(); this.showError('');
    try {
      const job = jobData(await request('/api/experiments', parameters));
      if (this.disposed) return;
      this.update(job); node<HTMLDetailsElement>('experiment-history').open = true; this.schedule();
    } catch (error) { if (!this.disposed) this.showError(errorText(error)); }
    finally { this.busy = false; if (!this.disposed) this.controls(); }
  }

  private schedule(delay = 900): void {
    window.clearTimeout(this.timer);
    if (!this.disposed && this.current && !terminal(this.current)) this.timer = window.setTimeout(() => { void this.poll(); }, delay);
  }

  private async poll(): Promise<void> {
    if (this.disposed || !this.current || terminal(this.current) || this.polling) return;
    this.polling = true; const id = this.current.id;
    try {
      const job = jobData(await request(`/api/experiments/${encodeURIComponent(id)}`));
      if (this.disposed || this.current?.id !== id) return;
      this.showError(''); this.update(job);
      if (job.status === 'completed' && this.current.status === 'completed') await this.show(job);
      this.schedule();
    } catch (error) { if (!this.disposed) { this.showError(errorText(error)); this.schedule(3000); } }
    finally { this.polling = false; }
  }

  private async cancel(): Promise<void> {
    if (this.busy || !this.current || terminal(this.current)) return;
    this.busy = true; this.controls(); this.showError('');
    try {
      const job = jobData(await request(`/api/experiments/${encodeURIComponent(this.current.id)}/cancel`, {}));
      if (this.disposed) return;
      this.update(job); this.schedule();
    } catch (error) { if (!this.disposed) this.showError(errorText(error)); }
    finally { this.busy = false; if (!this.disposed) this.controls(); }
  }

  dispose(): void { this.disposed = true; window.clearTimeout(this.timer); this.events.abort(); }
}

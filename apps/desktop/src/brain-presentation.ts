import { BrainView } from './brain';
import { loadBrainDisplay } from './brain-data';
import './presentation.css';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = '<main class="brain-presentation"><div id="brain-scene"></div><p class="presentation-message" role="status">正在读取脑结构…</p></main>';
const stage = app.querySelector<HTMLElement>('.brain-presentation')!;
const message = app.querySelector<HTMLElement>('.presentation-message')!;

async function start(): Promise<void> {
  const data = await loadBrainDisplay();
  const brain = new BrainView(document.getElementById('brain-scene')!, null, data);
  message.remove();
  const controls = document.createElement('div');
  controls.className = 'presentation-controls';
  controls.setAttribute('aria-label', '历史神经活动回放');
  const caption = document.createElement('p');
  caption.className = 'replay-caption';
  // This is essential interpretation, not a live-inference status or debug tag.
  caption.textContent = '历史片段 · 相对动作均值的电流差异';
  caption.title = '高亮为 L1/L2 电流相对四个起始动作同时间窗均值的差幅值；待机来自较晚的结束片段。连线是连接示意，不是轴突路径。';
  const actions = document.createElement('div');
  actions.className = 'presentation-actions';
  const buttons: HTMLButtonElement[] = [];
  let selected = 0, half = 0, playing = false, timer = 0, hideTimer = 0;
  let disposed = false;
  const clips = data.manifest.clips;
  const playButton = document.createElement('button');
  playButton.className = 'play-button';
  playButton.textContent = '慢放';
  playButton.title = '将两个 25 ms 的已记录均值窗口慢放；空格播放/暂停';
  const staticButton = document.createElement('button');
  staticButton.textContent = '结构'; staticButton.title = '仅看结构 · 0';
  const showControls = (): void => {
    stage.classList.add('controls-visible');
    clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => { if (!controls.contains(document.activeElement)) stage.classList.remove('controls-visible'); }, 3200);
  };
  function draw(): void {
    brain.setRecordedFrame(selected < 0 ? null : clips[selected], half);
    buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(selected === i)));
    staticButton.setAttribute('aria-pressed', String(selected < 0));
    playButton.disabled = selected < 0;
    stage.dataset.action = selected < 0 ? 'structure' : clips[selected].id;
    stage.dataset.recordedWindow = String(half);
  }
  function schedule(): void {
    clearTimeout(timer);
    if (!playing || document.hidden || disposed) return;
    timer = window.setTimeout(() => {
      // Show each of the two recorded means once, then stop on the last one.
      // Do not repeat/reverse them to fabricate an ongoing neural sequence.
      half = 1; draw(); playing = false; playButton.textContent = '重播';
    }, 2200);
  }
  function select(index: number): void {
    clearTimeout(timer); selected = index; half = 0; playing = false;
    playButton.textContent = '慢放'; draw(); showControls();
  }
  function play(): void {
    if (selected < 0) return;
    playing = !playing;
    if (playing) { half = 0; draw(); }
    playButton.textContent = playing ? '暂停' : '慢放';
    schedule(); showControls();
  }
  clips.forEach((clip, i) => {
    const button = document.createElement('button');
    button.textContent = clip.label; button.title = `${clip.label} · 历史决策 ${clip.decision.decision_seq} · 快捷键 ${i+1}`;
    button.addEventListener('click', () => { select(i); button.blur(); });
    buttons.push(button); actions.append(button);
  });
  staticButton.addEventListener('click', () => { select(-1); staticButton.blur(); });
  playButton.addEventListener('click', () => { play(); playButton.blur(); });
  const separator = document.createElement('i'); separator.setAttribute('aria-hidden', 'true');
  actions.append(separator, staticButton, playButton);
  const hint = document.createElement('p'); hint.className = 'interaction-hint';
  hint.textContent = '拖动旋转 · 滚轮缩放 · 双击复位';
  controls.append(caption, actions, hint); stage.append(controls);
  const onKey = (event: KeyboardEvent): void => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (/^[1-5]$/.test(event.key)) select(Number(event.key)-1);
    else if (event.key === '0') select(-1);
    else if (event.code === 'Space' && !(event.target instanceof HTMLButtonElement)) { event.preventDefault(); play(); }
    else if (event.key === 'Escape') { stage.classList.remove('controls-visible'); return; }
    else return;
    showControls();
  };
  const onVisibility = (): void => { if (document.hidden) clearTimeout(timer); else schedule(); };
  stage.addEventListener('pointermove', showControls);
  controls.addEventListener('focusin', showControls);
  controls.addEventListener('focusout', showControls);
  document.getElementById('brain-scene')!.addEventListener('dblclick', () => brain.reset());
  document.addEventListener('keydown', onKey);
  document.addEventListener('visibilitychange', onVisibility);
  window.addEventListener('beforeunload', () => {
    disposed = true; clearTimeout(timer); clearTimeout(hideTimer); brain.dispose();
    document.removeEventListener('keydown', onKey); document.removeEventListener('visibilitychange', onVisibility);
  }, { once: true });
  select(0);
}
start().catch(error => { message.textContent = error instanceof Error ? error.message : String(error); message.setAttribute('role', 'alert'); });

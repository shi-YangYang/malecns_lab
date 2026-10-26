import { app, BrowserWindow, dialog, ipcMain, protocol, session, WebContentsView } from 'electron';
import { readFile, readdir, realpath } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { randomBytes } from 'node:crypto';
import { SimulationVideo, trustedVideo, videoResponse } from './video.mjs';
import { SimulationCapture } from './recording.mjs';
import { ExperimentController } from './controller.mjs';
import { SIMULATION_URL, boundedRectangle, simulationRequestAllowed, trustedFrame } from './boundaries.mjs';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
const repo = fileURLToPath(new URL('../../../', import.meta.url));
const origin = 'workbench://app';
let window, simulation, controller, capture, video;
let documentGeneration = 0, videoInitializing = null, archiveDeadline = null;
let closePromise = null;
let captureRoot = null, captureInitializing = null, captureClosing = null, simulationVisible = false, simulationLoading = false;
let requestedVisible = true;
let sourceCheck = null;
let windowLimitReason = null, liveSourceRoot = null, liveSourceTimer = null;
let rectangle = { x: 0, y: 62, width: 0, height: 0 };
let closing = false, quitApproved = false, simulationLoaded = false, simulationError = null;
protocol.registerSchemesAsPrivileged([{ scheme: 'workbench', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } }]);
app.setName('MaleCNS Lab');
if (!app.requestSingleInstanceLock()) app.quit();
app.on('second-instance', () => { window?.show(); window?.focus(); });

async function assetAllowlist() {
  const manifest = JSON.parse(await readFile(path.join(root, 'prototype/manifest.json'), 'utf8'));
  if (manifest.schema !== 'malecns_lab.desktop.assets.v1' || !Array.isArray(manifest.assets)) throw new Error('图谱素材清单无效');
  const allowed = new Set(['/index.html']);
  for (const name of await readdir(path.join(root, 'assets'))) if (/^[a-zA-Z0-9_.-]+\.(js|css)$/.test(name)) allowed.add(`/assets/${name}`);
  for (const name of manifest.assets) {
    if (typeof name !== 'string' || !/^[a-zA-Z0-9_./-]+$/.test(name) || name.split('/').some(part => !part || part === '.' || part === '..')) throw new Error('素材清单包含非法路径');
    allowed.add(`/prototype/${name}`);
  }
  return allowed;
}
function lockSession(target) {
  target.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  target.setPermissionCheckHandler(() => false);
  target.on('will-download', event => event.preventDefault());
}
function lockContents(contents) {
  contents.setWindowOpenHandler(() => ({ action: 'deny' }));
  contents.on('will-navigate', event => event.preventDefault());
  contents.on('will-frame-navigate', event => event.preventDefault());
  contents.on('will-redirect', event => event.preventDefault());
  contents.on('will-attach-webview', event => event.preventDefault());
}
function updateSimulation() {
  if (!simulation || !window || window.isDestroyed()) return;
  const ready = controller.state.simulator_ready && !controller.state.resources_stopped;
  const visible = ready && requestedVisible && window.isVisible() && !window.isMinimized() && rectangle.width > 0 && rectangle.height > 0;
  simulationVisible = visible; simulation.setVisible(visible);
  if (visible && !simulationLoaded) {
    simulationLoaded = true; simulationLoading = true; simulationError = null;
    simulation.webContents.loadURL(SIMULATION_URL).catch(error => { simulationError = `仿真画面连接失败：${error.message}`; simulationLoaded = false; simulationLoading = false; simulationVisible = false; simulation.setVisible(false); capture?.viewChanged(captureContext()); video?.viewChanged(captureContext()); enforceLiveSource(); });
  }
  if (!ready && simulationLoaded) { simulationLoaded = false; simulation.webContents.stop(); }
  capture?.viewChanged(captureContext()); video?.viewChanged(captureContext());
  enforceLiveSource();
}
function trusted(event) {
  if (!trustedFrame(event, window?.webContents)) throw new Error('不允许的桌面请求来源');
}
function bind(name, fn) { ipcMain.handle(name, (event, ...args) => { trusted(event); return fn(...args); }); }
function captureContext() {
  return { root: controller.output, status: controller.state, visible: simulationVisible, loaded: simulationLoaded, loading: simulationLoading, error: simulationError };
}
async function confirmCaptureSource() {
  updateSimulation();
  const context = captureContext();
  if (!context.visible || !context.loaded || context.loading || context.error) throw new Error(context.error ?? '请等待仿真画面可见并连接后再开始');
  if (sourceCheck) throw new Error('正在检查仿真画面');
  // Page load is not an RFB/canvas readiness signal. Ask the isolated preload
  // before starting control; real recording faults still stop an active run.
  const nonce = randomBytes(16).toString('hex'), generation = documentGeneration;
  const result = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { sourceCheck = null; reject(new Error('仿真画面检查超时，请确认连接后再开始')); }, 2000);
    sourceCheck = { nonce, generation, resolve: value => { clearTimeout(timer); sourceCheck = null; resolve(value); } };
    simulation.webContents.send('simulation:source-check', { nonce, generation });
  });
  if (result.ready !== true) throw new Error(`仿真画面尚未就绪：${result.reason}`);
  updateSimulation();
  const current = captureContext();
  if (documentGeneration !== generation || !current.visible || !current.loaded || current.loading || current.error) throw new Error(current.error ?? '仿真画面在检查期间改变，请确认可见并连接后再开始');
}
function stopExperiment() {
  const state = controller.stop(); // writes the independent stop request first
  capture?.freeze('independent_stop'); video?.requestStop('independent_stop');
  return state;
}
function liveSourceRequired() { return liveSourceRoot !== null && liveSourceRoot === controller.output && controller.state.state === 'running' && !controller.state.resources_stopped; }
function enforceLiveSource() {
  if (!liveSourceRequired()) return false;
  const current = captureContext();
  if (!current.visible || !current.loaded || current.loading || current.error) {
    simulationError = current.error ?? '实时仿真画面已隐藏、断开或重载'; stopExperiment(); return false;
  }
  return true;
}
function watchLiveSource() {
  if (!liveSourceRequired()) { clearInterval(liveSourceTimer); liveSourceTimer = null; return; }
  if (!enforceLiveSource() || sourceCheck) return; // Share the start preflight; never overlap probes.
  const activeRoot = liveSourceRoot;
  void confirmCaptureSource().catch(error => {
    if (liveSourceRoot === activeRoot && liveSourceRequired()) { simulationError = error.message; stopExperiment(); }
  });
}
function limitMediaWindow(reason) {
  if (windowLimitReason || controller.state.session_profile !== 'continuous') return;
  windowLimitReason = reason;
  // Synchronous quiescence first. Defer seal so sample/write can finish and ACK.
  capture.limitReason ??= reason; video.limitReason ??= reason;
  capture.freeze(reason); video.requestStop(reason);
  liveSourceRoot = controller.output;
  liveSourceTimer = setInterval(watchLiveSource, 5000);
  enforceLiveSource();
  captureClosing = new Promise(resolve => setImmediate(resolve)).then(() => sealArchives(reason, true)).finally(() => { if (closing) closeWhenSafe(); });
}
function maintainCapture() {
  // Action demos have no neural telemetry identity; do not fabricate a paired archive.
  if (controller.state.mode === 'manual_demo') return;
  if (controller.state.state !== 'running') { capture?.freeze('control_ended'); video?.requestStop('control_ended'); clearInterval(liveSourceTimer); liveSourceTimer = null; }
  if ((capture || video) && controller.safeToClose && !captureClosing) captureClosing = sealArchives().finally(() => { if (closing) closeWhenSafe(); });
  if (!simulation || captureInitializing || controller.state.state !== 'running' || controller.state.resources_stopped || !simulationVisible || !simulationLoaded || simulationLoading || simulationError || captureRoot === controller.output) return;
  captureRoot = controller.output;
  const initialWindow = controller.state.session_profile === 'continuous';
  capture = new SimulationCapture({ root: captureRoot, recordId: path.basename(captureRoot), sourceRoot: fileURLToPath(new URL('./', import.meta.url)).replace(/\/$/, ''),
    capture: () => simulation.webContents.capturePage(undefined, { stayHidden: false, stayAwake: true }), hostClock: () => controller.sampleHostClock(), context: captureContext, onFailure: () => stopExperiment(), initialWindow, onLimit: limitMediaWindow });
  const active = capture;
  captureClosing = null; archiveDeadline = null;
  video = new SimulationVideo({ root: captureRoot, recordId: path.basename(captureRoot), sourceRoot: fileURLToPath(new URL('./', import.meta.url)).replace(/\/$/, ''), generation: documentGeneration, context: captureContext, send: message => { if (!simulation.webContents.isDestroyed()) simulation.webContents.send('video:command', message); }, onFailure: () => stopExperiment(), initialWindow, onLimit: limitMediaWindow });
  const activeVideo = video;
  videoInitializing = activeVideo.initialize().catch(error => activeVideo.fail(error)).finally(() => { videoInitializing = null; });
  captureInitializing = active.initialize().then(() => {
    if (active !== capture || controller.output !== active.root || controller.state.state !== 'running' || closing) active.freeze('control_ended_during_initialization');
    else active.start();
  }).catch(error => active.fail(error)).finally(() => { captureInitializing = null; if (controller.safeToClose) closeWhenSafe(); });
}
function sealArchives(reason = 'control_ended', initialWindow = false) {
  archiveDeadline ??= Date.now() + 2000;
  const deadline = archiveDeadline;
  let timer;
  const work = Promise.allSettled([
    Promise.resolve(captureInitializing).then(() => capture?.seal(reason, deadline)),
    Promise.resolve(videoInitializing).then(() => video?.seal(deadline)),
  ]);
  return Promise.race([work, new Promise(resolve => { timer = setTimeout(() => { video?.requestStop('archive_deadline'); resolve(); }, Math.max(0, deadline - Date.now())); })]).then(() => {
    if (initialWindow && (capture?.status().complete !== true || video?.status().complete !== true)) {
      simulationError = '首段媒体封存失败：' + (capture?.error ?? video?.reason ?? '封存未在期限内完成'); stopExperiment();
    }
  }).finally(() => clearTimeout(timer));
}
function closeWhenSafe() {
  if (controller.state.state !== 'running') video?.requestStop('control_ended');
  updateSimulation(); capture?.noteStatus(controller.state); maintainCapture();
  if (closing && controller.safeToClose && !closePromise) {
    closePromise = (captureClosing ?? sealArchives()).finally(finishClose);
  } else if (closing && !controller.child && !controller.safeToClose) {
    closing = false;
    dialog.showMessageBox(window, { type: 'error', title: '实验清理未确认', message: controller.state.message, detail: '窗口保留以查看失败状态。实验进程退出本身不代表停车或资源清理成功。', buttons: ['保留窗口'] });
  }
}
function finishClose() { if (closing && controller.safeToClose) { capture?.freeze('window_close'); quitApproved = true; app.quit(); } }
function requestClose(event) {
  if (quitApproved) return;
  event.preventDefault();
  if (!closing) { closing = true; stopExperiment(); closeWhenSafe(); }
}
function createWindow() {
  window = new BrowserWindow({
    width: 1400, height: 880, minWidth: 1100, minHeight: 720,
    backgroundColor: '#0c1119', title: 'MaleCNS Lab', titleBarStyle: 'hiddenInset', trafficLightPosition: { x: 20, y: 20 },
    webPreferences: { preload: fileURLToPath(new URL('./preload.cjs', import.meta.url)), nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true, backgroundThrottling: false },
  });
  // Control state must keep updating while the operator uses another window.
  // Expensive brain rendering still pauses through its visibility handler.
  lockContents(window.webContents);
  window.webContents.on('render-process-gone', () => stopExperiment());
  window.removeMenu();
  const isolated = session.fromPartition('malecns-simulation');
  lockSession(isolated);
  isolated.webRequest.onBeforeRequest((details, callback) => callback({ cancel: !simulationRequestAllowed(details.url) }));
  simulation = new WebContentsView({ webPreferences: { preload: fileURLToPath(new URL('./simulation-preload.cjs', import.meta.url)), session: isolated, nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true, backgroundThrottling: false } });
  lockContents(simulation.webContents);
  simulation.webContents.on('render-process-gone', () => { simulationLoaded = false; simulationError = '仿真画面进程已退出'; simulationLoading = false; stopExperiment(); });
  simulation.webContents.on('did-fail-load', (_event, code, description, _url, isMainFrame) => { if (isMainFrame && code !== -3) { simulationError = `仿真画面无法读取：${description}`; simulationLoading = false; simulationVisible = false; simulation.setVisible(false); capture?.viewChanged(captureContext()); video?.viewChanged(captureContext()); enforceLiveSource(); } });
  simulation.webContents.on('did-start-loading', () => { documentGeneration++; video?.revoke('视频页面已重载'); simulationLoading = true; capture?.viewChanged(captureContext()); video?.viewChanged(captureContext()); enforceLiveSource(); });
  simulation.webContents.on('did-finish-load', () => { simulationLoading = false; simulationError = null; maintainCapture(); });
  window.contentView.addChildView(simulation); simulation.setVisible(false);
  window.on('close', requestClose);
  window.on('closed', () => { if (!simulation.webContents.isDestroyed()) simulation.webContents.close(); });
  for (const event of ['show', 'hide', 'minimize', 'restore']) window.on(event, updateSimulation);
  window.on('resize', () => { const [width, height] = window.getContentSize(); rectangle = boundedRectangle(rectangle, width, height); simulation.setBounds(rectangle); updateSimulation(); });
  window.loadURL(`${origin}/index.html`);
}

app.whenReady().then(async () => {
  const allowed = await assetAllowlist();
  const canonicalRoot = await realpath(root);
  lockSession(session.defaultSession);
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => callback({ cancel: !details.url.startsWith(`${origin}/`) }));
  protocol.handle('workbench', async request => {
    const url = new URL(request.url);
    const media = /^\/media\/(desktop-\d{8}-\d{6}-[a-f0-9]{8})\/recording\.webm$/.exec(url.pathname);
    if (media && url.host === 'app' && !url.search && !url.hash && ['GET', 'HEAD'].includes(request.method)) {
      try { if (!controller.safeToClose) return new Response('Experiment active', { status: 409 }); const record = await controller.openRecord(media[1]); return await videoResponse(request, await controller.recordRoot(media[1]), record.video); }
      catch { return new Response('Video unavailable', { status: 404 }); }
    }
    if (request.method !== 'GET' || url.host !== 'app' || url.search || !allowed.has(url.pathname)) return new Response('Not found', { status: 404 });
    try {
      const file = await realpath(path.join(root, url.pathname));
      if (!file.startsWith(`${canonicalRoot}${path.sep}`)) return new Response('Forbidden', { status: 403 });
      const contentType = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json' }[path.extname(file)] ?? 'application/octet-stream';
      return new Response(await readFile(file), { headers: { 'Content-Type': contentType, 'X-Content-Type-Options': 'nosniff' } });
    } catch { return new Response('Missing atlas asset', { status: 404 }); }
  });
  controller = new ExperimentController(repo, closeWhenSafe);
  bind('experiment:prepare', () => { if (closing || captureInitializing || videoInitializing || !controller.safeToClose || (capture && !capture.finalized) || (video && !video.finalized)) throw new Error('正在关闭或封存实验'); clearInterval(liveSourceTimer); liveSourceTimer = null; liveSourceRoot = null; windowLimitReason = null; capture = null; video = null; captureRoot = null; captureClosing = null; archiveDeadline = null; simulationLoaded = false; simulationLoading = false; return controller.prepare(); });
  bind('experiment:start', async () => { if (controller.state.mode !== 'manual_demo') await confirmCaptureSource(); return controller.continue(); });
  bind('experiment:stop', () => stopExperiment());
  bind('experiment:task', task => controller.task(task));
  bind('experiment:snapshot', async () => { maintainCapture(); return { ...await controller.snapshot(), simulation_error: simulationError, capture: capture?.status() ?? null, video: video?.status() ?? null }; });
  bind('records:list', () => controller.listRecords());
  bind('records:open', id => { if (!controller.safeToClose) throw new Error('请先结束当前实验'); return controller.openRecord(id); });
  bind('records:frame', (id, sequence) => controller.readRecordedFrame(id, sequence));
  bind('records:batch', (id, sequence, cursorIndex) => controller.readBatch(id, sequence, cursorIndex));
  bind('simulation:bounds', value => { const [width, height] = window.getContentSize(); rectangle = boundedRectangle(value, width, height); simulation.setBounds(rectangle); updateSimulation(); });
  bind('simulation:visible', value => { if (typeof value !== 'boolean') throw new Error('显示状态无效'); requestedVisible = value; updateSimulation(); });
  ipcMain.on('simulation:source-status', (event, message) => {
    const pending = sourceCheck;
    if (!pending || event.sender !== simulation?.webContents || event.senderFrame !== simulation.webContents.mainFrame || event.senderFrame?.url !== SIMULATION_URL || message?.generation !== documentGeneration || message.generation !== pending.generation || message.nonce !== pending.nonce) return;
    if (typeof message.ready !== 'boolean' || (message.ready === false && (typeof message.reason !== 'string' || message.reason.length > 1000))) return;
    pending.resolve(message);
  });
  for (const channel of ['video:event', 'video:clock', 'video:fault', 'video:limit']) ipcMain.handle(channel, async (event, message) => {
    if (!trustedVideo(event, simulation?.webContents, message, video?.authorization) || message.generation !== documentGeneration) throw new Error('不允许的视频请求来源');
    const active = video;
    try {
      if (channel === 'video:clock') return active.receiveClock(message);
      if (channel === 'video:limit') { if (!active.initialWindow || message.reason !== 'duration_limit') throw new Error('视频首段容量通知无效'); active.limit('视频达到 480 秒上限'); return { accepted: true }; }
      if (channel === 'video:fault') { if (typeof message.reason !== 'string' || message.reason.length > 1000) throw new Error('视频错误信息无效'); active.fail(message.reason); return { accepted: true }; }
      return await active.receive(message);
    } catch (error) { active.fail(error); throw error; }
  });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
}).catch(error => { dialog.showErrorBox('工作台无法启动', `请确认已构建且本地图谱齐全。\n${error.message}`); quitApproved = true; app.quit(); });
app.on('before-quit', event => { if (controller) requestClose(event); });
app.on('window-all-closed', () => app.quit());

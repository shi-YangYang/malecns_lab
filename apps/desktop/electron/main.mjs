import { app, BrowserWindow, dialog, ipcMain, protocol, session, WebContentsView } from 'electron';
import { readFile, readdir, realpath } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { ExperimentController } from './controller.mjs';
import { SIMULATION_URL, boundedRectangle, simulationRequestAllowed, trustedFrame } from './boundaries.mjs';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
const repo = fileURLToPath(new URL('../../../', import.meta.url));
const origin = 'workbench://app';
let window, simulation, controller;
let requestedVisible = true;
let rectangle = { x: 0, y: 62, width: 0, height: 0 };
let closing = false, quitApproved = false, simulationLoaded = false, simulationError = null;
protocol.registerSchemesAsPrivileged([{ scheme: 'workbench', privileges: { standard: true, secure: true, supportFetchAPI: true } }]);
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
  simulation.setVisible(visible);
  if (visible && !simulationLoaded) {
    simulationLoaded = true; simulationError = null;
    simulation.webContents.loadURL(SIMULATION_URL).catch(error => { simulationError = `仿真画面连接失败：${error.message}`; simulationLoaded = false; simulation.setVisible(false); });
  }
  if (!ready && simulationLoaded) { simulationLoaded = false; simulation.webContents.stop(); }
}
function trusted(event) {
  if (!trustedFrame(event, window?.webContents)) throw new Error('不允许的桌面请求来源');
}
function bind(name, fn) { ipcMain.handle(name, (event, ...args) => { trusted(event); return fn(...args); }); }
function closeWhenSafe() {
  updateSimulation();
  if (closing && controller.safeToClose) { quitApproved = true; app.quit(); }
  else if (closing && !controller.child && !controller.safeToClose) {
    closing = false;
    dialog.showMessageBox(window, { type: 'error', title: '实验清理未确认', message: controller.state.message, detail: '窗口保留以查看失败状态。实验进程退出本身不代表停车或资源清理成功。', buttons: ['保留窗口'] });
  }
}
function requestClose(event) {
  if (quitApproved || controller.safeToClose) return;
  event.preventDefault();
  if (!closing) { closing = true; controller.stop(); closeWhenSafe(); }
}
function createWindow() {
  window = new BrowserWindow({
    width: 1400, height: 880, minWidth: 1100, minHeight: 720,
    backgroundColor: '#0c1119', title: 'MaleCNS Lab', titleBarStyle: 'hiddenInset', trafficLightPosition: { x: 20, y: 20 },
    webPreferences: { preload: fileURLToPath(new URL('./preload.cjs', import.meta.url)), nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true, backgroundThrottling: true },
  });
  lockContents(window.webContents);
  window.webContents.on('render-process-gone', () => controller.stop());
  window.removeMenu();
  const isolated = session.fromPartition('malecns-simulation');
  lockSession(isolated);
  isolated.webRequest.onBeforeRequest((details, callback) => callback({ cancel: !simulationRequestAllowed(details.url) }));
  simulation = new WebContentsView({ webPreferences: { session: isolated, nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true, backgroundThrottling: true } });
  lockContents(simulation.webContents);
  simulation.webContents.on('render-process-gone', () => { simulationLoaded = false; simulationError = '仿真画面进程已退出'; controller.stop(); });
  simulation.webContents.on('did-fail-load', (_event, code, description, _url, isMainFrame) => { if (isMainFrame && code !== -3) { simulationError = `仿真画面无法读取：${description}`; simulation.setVisible(false); } });
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
    if (request.method !== 'GET' || url.host !== 'app' || url.search || !allowed.has(url.pathname)) return new Response('Not found', { status: 404 });
    try {
      const file = await realpath(path.join(root, url.pathname));
      if (!file.startsWith(`${canonicalRoot}${path.sep}`)) return new Response('Forbidden', { status: 403 });
      const contentType = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json' }[path.extname(file)] ?? 'application/octet-stream';
      return new Response(await readFile(file), { headers: { 'Content-Type': contentType, 'X-Content-Type-Options': 'nosniff' } });
    } catch { return new Response('Missing atlas asset', { status: 404 }); }
  });
  controller = new ExperimentController(repo, closeWhenSafe);
  bind('experiment:prepare', () => { if (closing) throw new Error('正在关闭实验'); simulationLoaded = false; return controller.prepare(); });
  bind('experiment:start', () => controller.continue());
  bind('experiment:stop', () => controller.stop());
  bind('experiment:snapshot', async () => ({ ...await controller.snapshot(), simulation_error: simulationError }));
  bind('records:list', () => controller.listRecords());
  bind('records:open', id => controller.openRecord(id));
  bind('records:batch', (id, sequence) => controller.readBatch(id, sequence));
  bind('simulation:bounds', value => { const [width, height] = window.getContentSize(); rectangle = boundedRectangle(value, width, height); simulation.setBounds(rectangle); updateSimulation(); });
  bind('simulation:visible', value => { if (typeof value !== 'boolean') throw new Error('显示状态无效'); requestedVisible = value; updateSimulation(); });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
}).catch(error => { dialog.showErrorBox('工作台无法启动', `请确认已构建且本地图谱齐全。\n${error.message}`); quitApproved = true; app.quit(); });
app.on('before-quit', event => { if (controller) requestClose(event); });
app.on('window-all-closed', () => app.quit());

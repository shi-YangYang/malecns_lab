// Presentation-only process: no experiment controller, preload, IPC or VM.
import { app, BrowserWindow, dialog, protocol, session } from 'electron';
import { readFile, readdir, realpath } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
app.setName('MaleCNS Brain');
// Do not join or reload the running experiment workbench's profile/window.
app.setPath('userData', path.join(app.getPath('appData'), 'MaleCNS Brain'));
protocol.registerSchemesAsPrivileged([{ scheme: 'brain', privileges: { standard: true, secure: true, supportFetchAPI: true } }]);
let window;
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { window?.show(); window?.focus(); });
  app.whenReady().then(async () => {
    const canonical = await realpath(root);
    const manifest = JSON.parse(await readFile(path.join(root, 'prototype/manifest.json'), 'utf8'));
    if (manifest.schema !== 'malecns_lab.desktop.assets.v1' || !Array.isArray(manifest.assets)) throw new Error('展示资产清单无效');
    const allowed = new Set(['/index.html']);
    for (const name of await readdir(path.join(root, 'assets'))) if (/^[a-zA-Z0-9_.-]+\.(js|css)$/.test(name)) allowed.add(`/assets/${name}`);
    for (const name of manifest.assets) {
      if (typeof name !== 'string' || !/^[a-zA-Z0-9_./-]+$/.test(name) || name.split('/').some(p => !p || p === '.' || p === '..')) throw new Error('展示资产路径无效');
      if (name.startsWith('brain/')) allowed.add(`/prototype/${name}`);
    }
    protocol.handle('brain', async request => {
      const url = new URL(request.url);
      if (request.method !== 'GET' || url.host !== 'app' || url.search || !allowed.has(url.pathname)) return new Response('Not found', { status: 404 });
      try {
        const file = await realpath(path.join(root, url.pathname));
        if (!file.startsWith(`${canonical}${path.sep}`)) return new Response('Forbidden', { status: 403 });
        const type = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json' }[path.extname(file)] ?? 'application/octet-stream';
        return new Response(await readFile(file), { headers: { 'Content-Type': type, 'X-Content-Type-Options': 'nosniff' } });
      } catch { return new Response('Missing display asset', { status: 404 }); }
    });
    session.defaultSession.setPermissionRequestHandler((_c, _p, callback) => callback(false));
    session.defaultSession.setPermissionCheckHandler(() => false);
    session.defaultSession.on('will-download', event => event.preventDefault());
    window = new BrowserWindow({ width: 1280, height: 1000, minWidth: 600, minHeight: 600,
      backgroundColor: '#000000', title: 'MaleCNS · 神经活动', titleBarStyle: 'hidden',
      webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true } });
    window.setWindowButtonVisibility(false);
    window.removeMenu();
    window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    for (const eventName of ['will-navigate', 'will-frame-navigate', 'will-redirect', 'will-attach-webview']) window.webContents.on(eventName, event => event.preventDefault());
    window.webContents.on('before-input-event', (event, input) => {
      if (input.type !== 'keyDown') return;
      if (input.key.toLowerCase() === 'f' && !input.control && !input.meta && !input.alt) { event.preventDefault(); window.setFullScreen(!window.isFullScreen()); }
      if (input.key === 'Escape' && window.isFullScreen()) { event.preventDefault(); window.setFullScreen(false); }
      if ((input.meta || input.control) && ['w', 'q'].includes(input.key.toLowerCase())) { event.preventDefault(); window.close(); }
    });
    await window.loadURL('brain://app/index.html');
  }).catch(error => { dialog.showErrorBox('脑展示无法启动', `请先运行 npm run build。\n${error.message}`); app.quit(); });
  app.on('window-all-closed', () => app.quit());
}

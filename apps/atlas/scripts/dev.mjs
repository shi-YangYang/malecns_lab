import { spawn } from 'node:child_process';
import { access, constants } from 'node:fs/promises';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const atlasRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const projectRoot = path.resolve(atlasRoot, '../..');
const python = path.join(projectRoot, '.venv', 'bin', 'python');
const vite = path.join(atlasRoot, 'node_modules', 'vite', 'bin', 'vite.js');
const children = [];
let stopped = false;
let shutdownPromise;
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function freePort(port) {
  await new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.once('error', () => reject(new Error(`端口 ${port} 已被占用，未启动新服务。请先退出对应服务，或使用原有页面。`)));
    probe.listen({ host: '127.0.0.1', port, exclusive: true }, () => probe.close(resolve));
  });
}

function signal(child, name) {
  if (!child.pid) return;
  try { process.kill(-child.pid, name); } catch (error) { if (error.code !== 'ESRCH') console.error(`停止服务失败：${error.message}`); }
}

function shutdown(code = 0) {
  if (shutdownPromise) return shutdownPromise;
  stopped = true; process.exitCode = code;
  shutdownPromise = (async () => {
    for (const item of children) signal(item.child, 'SIGTERM');
    let timer;
    await Promise.race([
      Promise.all(children.map(item => item.exited)),
      new Promise(resolve => { timer = setTimeout(resolve, 8000); }),
    ]);
    clearTimeout(timer);
    for (const item of children) signal(item.child, 'SIGKILL');
  })();
  return shutdownPromise;
}

function launch(command, args, cwd, label) {
  const child = spawn(command, args, { cwd, detached: true, stdio: 'inherit' });
  const exited = new Promise(resolve => {
    child.once('error', error => {
      console.error(`${label}无法启动：${error.message}`); resolve(); void shutdown(1);
    });
    child.once('exit', (code, signalName) => {
      resolve();
      if (!stopped) {
        console.error(`${label}已退出（${signalName ?? code}），正在停止本次启动的服务。`);
        void shutdown(code === 0 ? 0 : 1);
      }
    });
  });
  children.push({ child, exited });
  return child;
}

async function waitForBackend(child) {
  const deadline = Date.now() + 15000;
  while (!stopped && Date.now() < deadline) {
    let health;
    try {
      const response = await fetch('http://127.0.0.1:8766/api/health', { signal: AbortSignal.timeout(1000) });
      if (response.ok) health = await response.json();
    } catch { /* The owned server may not yet be listening. */ }
    if (health?.status === 'ok') {
      if (health.pid !== child.pid) throw new Error('端口 8766 被其他服务占用，本次启动已停止。');
      return;
    }
    await sleep(200);
  }
  if (!stopped) throw new Error('实验服务启动超时，请查看上方错误提示。');
}

process.on('SIGINT', () => { void shutdown(0); });
process.on('SIGTERM', () => { void shutdown(0); });
process.on('exit', () => { for (const item of children) signal(item.child, 'SIGKILL'); });

try {
  try { await access(python, constants.X_OK); } catch { throw new Error('项目虚拟环境未就绪：请先按安装说明配置 .venv。'); }
  try { await access(vite); } catch { throw new Error('网页依赖未就绪：请在 apps/atlas 运行 npm install。'); }
  await Promise.all([freePort(8765), freePort(8766)]);
  if (!stopped) {
    console.log('正在启动完整脑实验服务…');
    const backend = launch(python, ['-m', 'malecns_lab.visualization.server'], projectRoot, '实验服务');
    await waitForBackend(backend);
    if (!stopped) launch(process.execPath, [vite, '--host', '127.0.0.1', '--port', '8765', '--strictPort'], atlasRoot, '网页服务');
  }
} catch (error) {
  console.error(error.message); await shutdown(1);
}

// Isolated, sandboxed simulation world only. No contextBridge, page API, or page message listener.
const { ipcRenderer } = require('electron');
const clockId = crypto.randomUUID();
let active = null;
const stamp = event => ({ performance_ms: performance.now(), event_timestamp_ms: Number.isFinite(event?.timeStamp) ? event.timeStamp : performance.now(), clock_id: clockId });
function envelope(a, value) { return { generation: a.generation, token: a.token, ...value }; }
async function invoke(a, channel, value) {
  let timer;
  try { return await Promise.race([ipcRenderer.invoke(channel, envelope(a, value)), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('视频写入确认超时')), 1500); })]); }
  finally { clearTimeout(timer); }
}
function release(a) { clearInterval(a.watch); clearTimeout(a.duration); a.stream?.getTracks().forEach(track => track.stop()); }
function stop(a) {
  if (a.stopping) return;
  a.stopping = true; a.stopCall = performance.now(); clearInterval(a.watch); clearTimeout(a.duration);
  if (a.recorder && a.recorder.state !== 'inactive') a.recorder.stop();
  else release(a);
}
function fault(a, reason) {
  if (a.failed) return; a.failed = true;
  // Notify host first: its handler requests independent control stop before stopping capture.
  void invoke(a, 'video:fault', { reason: String(reason?.message ?? reason).slice(0, 1000) }).catch(() => {}).finally(() => { stop(a); release(a); });
}
function canvas() {
  const nodes = document.querySelectorAll('#noVNC_container canvas');
  if (nodes.length !== 1) throw new Error('视频画布缺失或不唯一');
  const node = nodes[0], rect = node.getBoundingClientRect(), style = getComputedStyle(node);
  if (!(node instanceof HTMLCanvasElement) || !node.isConnected || node.width < 1 || node.height < 1 || node.width > 1280 || node.height > 800 || rect.width <= 0 || rect.height <= 0 || style.visibility !== 'visible' || style.display === 'none' || document.hidden || !document.documentElement.classList.contains('noVNC_connected')) throw new Error('视频来源未连接或不可见');
  return node;
}
async function pump(a) {
  if (a.busy || a.failed || active !== a) return;
  a.busy = true;
  try {
    while (a.queue.length && !a.queue[0].held && !a.failed && active === a) {
      const item = a.queue.shift();
      let data;
      if (item.blob) data = new Uint8Array(await item.blob.arrayBuffer());
      const value = { ...item.metadata, sequence: ++a.sequence, ...(data ? { data } : {}) };
      if (value.kind === 'stopped') value.final_sequence = a.sequence - 1;
      const ack = await invoke(a, 'video:event', value);
      if (!ack?.accepted || ack.sequence !== a.sequence) throw new Error('视频确认序号不匹配');
      if (item.blob) { a.pendingBytes -= item.blob.size; a.blobs--; }
    }
  } catch (error) { fault(a, error); }
  finally { a.busy = false; }
}
function enqueue(a, metadata, blob, held = false) {
  if (a.failed || active !== a) return;
  if (blob) {
    // Count includes the in-flight Blob/ArrayBuffer until ACK and any held final candidate.
    if (blob.size > 8 * 1024 * 1024 || a.blobs >= 2 || a.pendingBytes + blob.size > 16 * 1024 * 1024) { fault(a, '视频写入背压或块大小超过上限'); return; }
    a.blobs++; a.pendingBytes += blob.size;
  }
  if (a.queue.length >= 4) { fault(a, '视频事件队列超过上限'); return; }
  const item = { metadata, blob, held };
  a.queue.push(item); void pump(a); return item;
}
function start(message) {
  if (active) return; // A document gets one segment; recovery requires a new run/document.
  const a = active = { generation: message.generation, token: message.token, queue: [], sequence: 0, busy: false, pendingBytes: 0, blobs: 0, failed: false, stopping: false };
  try {
    if (!Number.isInteger(a.generation) || !/^[a-f0-9]{64}$/.test(a.token)) throw new Error('视频授权无效');
    a.canvas = canvas(); a.width = a.canvas.width; a.height = a.canvas.height;
    const mime = 'video/webm;codecs=vp8';
    if (!MediaRecorder.isTypeSupported(mime)) throw new Error('当前环境不支持 VP8 视频录制');
    a.stream = a.canvas.captureStream(15);
    if (a.stream.getAudioTracks().length || a.stream.getVideoTracks().length !== 1) throw new Error('视频轨道无效');
    const track = a.stream.getVideoTracks()[0];
    track.addEventListener('mute', () => { if (!a.stopping) fault(a, '视频轨道已静音'); });
    track.addEventListener('ended', () => { if (!a.stopping) fault(a, '视频轨道已结束'); });
    a.recorder = new MediaRecorder(a.stream, { mimeType: mime, videoBitsPerSecond: 2500000 });
    a.recorder.addEventListener('start', event => enqueue(a, { kind: 'started', ...stamp(event), start_call_ms: a.startCall, mime: a.recorder.mimeType, width: a.width, height: a.height }));
    a.recorder.addEventListener('dataavailable', event => {
      if (!event.data.size && !a.stopping) return;
      // stop() changes state before queued timeslices arrive. Only the stop event
      // establishes which dataavailable was last; keep that candidate in the same bounded queue.
      if (a.finalCandidate) {
        if (a.finalCandidate.blob.size) a.finalCandidate.held = false;
        else { a.queue.splice(a.queue.indexOf(a.finalCandidate), 1); a.blobs--; } // Empty intermediate events carry no bytes.
      }
      a.finalCandidate = enqueue(a, { kind: 'chunk', ...stamp(event), timecode_ms: Number.isFinite(event.timecode) && event.timecode >= 0 ? event.timecode : null, final: false }, event.data, a.stopping);
      if (!a.stopping) a.finalCandidate = null;
    });
    a.recorder.addEventListener('stop', event => {
      release(a);
      if (!a.stopping || !a.finalCandidate) { fault(a, '视频缺少最终数据事件'); return; }
      a.finalCandidate.metadata.final = true; a.finalCandidate.held = false; a.finalCandidate = null;
      enqueue(a, { kind: 'stopped', ...stamp(event), stop_call_ms: a.stopCall });
    });
    a.recorder.addEventListener('error', event => fault(a, event.error?.message ?? '视频编码失败'));
    a.startCall = performance.now(); a.recorder.start(1000);
    a.watch = setInterval(() => { if (a.stopping) return; try { if (canvas() !== a.canvas || a.canvas.width !== a.width || a.canvas.height !== a.height) throw new Error('视频画布被替换或尺寸改变'); } catch (e) { fault(a, e); } }, 100);
    a.duration = setTimeout(() => fault(a, '视频达到 240 秒上限'), 240000);
  } catch (error) { fault(a, error); }
}
ipcRenderer.on('video:command', (_event, message) => {
  if (!message || typeof message !== 'object') return;
  if (message.kind === 'start') { start(message); return; }
  const a = active;
  if (!a || message.generation !== a.generation || message.token !== a.token) return;
  if (message.kind === 'stop') stop(a);
  if (message.kind === 'clock' && /^[a-f0-9]{32}$/.test(message.nonce)) void invoke(a, 'video:clock', { nonce: message.nonce, performance_ms: performance.now(), time_origin_ms: performance.timeOrigin, clock_id: clockId }).catch(error => { if (!a.stopping) fault(a, error); });
});
document.addEventListener('visibilitychange', () => { if (active && document.hidden && !active.stopping) fault(active, '视频页面已隐藏'); });
window.addEventListener('pagehide', () => { if (active && !active.stopping) fault(active, '视频页面离开'); });

import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, realpath, mkdir, writeFile, readFile, rm, symlink } from 'node:fs/promises';
import fsPromises from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import { SimulationVideo, VIDEO_LIMITS, NOVNC_HASHES, trustedVideo, verifyNoVNC, readVideoArchive, videoRange, videoResponse } from './video.mjs';
import { SIMULATION_URL } from './boundaries.mjs';
const id = 'desktop-20261023-115959-abcdef12';
const media = Buffer.from([0x1a, 0x45, 0xdf, 0xa3, 9, 8, 7, 6]); // Contract fixture, NOT decodable video evidence.
async function fixture(t, overrides = {}) {
  const temp = await mkdtemp(path.join(tmpdir(), 'malecns-video-'));
  const base = await realpath(temp), root = path.join(base, id), sourceRoot = path.join(base, 'source');
  await mkdir(path.join(root, 'telemetry'), { recursive: true }); await mkdir(sourceRoot);
  for (const name of ['video.mjs','simulation-preload.cjs','main.mjs','controller.mjs','recording.mjs','boundaries.mjs']) await writeFile(path.join(sourceRoot, name), name);
  const ready = { run_id: id, session_id: 'neural', checkpoint_id: 'checkpoint', policy_id: 'policy', input_map_id: 'input' };
  const telemetry = { ...ready, execution_id: ready.session_id, schema: 'malecns_lab.x2_behavior_telemetry_manifest.v1', clock_domain: 'host_worker_monotonic' };
  for (const [file, value] of Object.entries({ 'behavior_registration.json': { ...ready, output: root, simulation_only: true, qualification_only: true, checkpoint_metadata: { checkpoint_id: ready.checkpoint_id } }, 'brain_ready.json': ready, 'telemetry/manifest.json': telemetry, 'official_assets.json': { sha256: { mc: 'a'.repeat(64) } } })) await writeFile(path.join(root, file), JSON.stringify(value));
  const commands = [], failures = [], context = { root, status: { run_id: id, state: 'running', resources_stopped: false }, visible: true, loaded: true, loading: false, error: null };
  const video = new SimulationVideo({ root, recordId: id, sourceRoot, generation: 2, send: value => commands.push(value), context: () => context, onFailure: value => failures.push(value), freeSpace: async () => ({ bavail: 1e6, bsize: 1024 }), verifySource: async () => NOVNC_HASHES, versions: { electron: 'test', chrome: 'test' }, ...overrides });
  t.after(async () => { video.requestStop('test_end'); await video.seal(Date.now() + 5); await video.incompleteWrite; await rm(temp, { recursive: true, force: true }); });
  await video.initialize();
  return { video, root, telemetry, context, commands, failures, session: { state: 'completed', resources_stopped: true, run_id: id } };
}
function event(sequence, kind, extra = {}) { return { sequence, kind, clock_id: 'clock', performance_ms: sequence * 100, event_timestamp_ms: sequence * 100, ...extra }; }
async function complete(f) {
  await f.video.receive(event(1, 'started', { start_call_ms: 90, mime: VIDEO_LIMITS.mime, width: 1280, height: 800 }));
  const probe = f.commands.find(v => v.kind === 'clock');
  f.video.receiveClock({ nonce: probe.nonce, performance_ms: 100, time_origin_ms: 1, clock_id: 'clock' });
  await f.video.receive(event(2, 'chunk', { data: media, timecode_ms: 0, final: false }));
  f.video.requestStop();
  await f.video.receive(event(3, 'chunk', { data: new Uint8Array(), timecode_ms: 1000, final: true }));
  await f.video.receive(event(4, 'stopped', { final_sequence: 3, stop_call_ms: 250 }));
  await f.video.seal();
  return readVideoArchive(f.root, id, f.session, f.telemetry);
}
test('video exact sender, main frame, URL, generation and token are independently required', () => {
  const mainFrame = { url: SIMULATION_URL }, contents = { mainFrame }, a = { generation: 2, token: 'a'.repeat(64) }, e = { sender: contents, senderFrame: mainFrame };
  assert.equal(trustedVideo(e, contents, a, a), true);
  for (const bad of [{ ...e, sender: {} }, { ...e, senderFrame: { url: SIMULATION_URL } }]) assert.equal(trustedVideo(bad, contents, a, a), false);
  for (const url of [SIMULATION_URL + '#x', SIMULATION_URL.replace('resize=scale','resize=remote'), 'http://127.0.0.1:6081/other.html']) { mainFrame.url = url; assert.equal(trustedVideo(e, contents, a, a), false); } mainFrame.url = SIMULATION_URL;
  for (const value of [{ ...a, token: 'b'.repeat(64) }, { ...a, generation: 3 }]) assert.equal(trustedVideo(e, contents, value, a), false);
  assert.equal(trustedVideo(e, contents, a, null), false);
});
test('video source mismatch and oversized source fail without permissive fallback', async () => {
  const urls = []; await assert.rejects(verifyNoVNC(async url => { urls.push(url); return new Response('changed'); }), /来源不匹配/);
  assert.deepEqual(urls, ['http://127.0.0.1:6081/vnc.html']);
  await assert.rejects(verifyNoVNC(async () => new Response(new Uint8Array(1024 * 1024 + 1))), /超界/);
});
test('video archive binds identities, exact raw bytes, final chunk and clock evidence; media stays unaligned', async t => {
  const f = await fixture(t), archive = await complete(f);
  assert.equal(archive.state, 'sealed'); assert.equal(archive.manifest.identity.execution_id, 'neural'); assert.equal(archive.alignment, null); assert.equal(archive.manifest.media_anchor, null);
  assert.deepEqual(await readFile(path.join(f.root,'simulation_video/recording.webm')), media);
  assert.equal(archive.index.clocks.length, 1); assert.ok(BigInt(archive.index.clocks[0].host_receive_ns) >= BigInt(archive.index.clocks[0].host_send_ns));
  await assert.rejects(readVideoArchive(f.root, id, f.session, { ...f.telemetry, checkpoint_id: 'other' }), /身份/);
  assert.throws(() => f.video.receive(event(5, 'chunk', { data: media, final: true })), /终止/);
});
test('video refuses wrong order, oversize, missing final data and concurrent writes', async t => {
  const f = await fixture(t);
  assert.throws(() => f.video.receive(event(2, 'chunk', { data: media, final: false })), /序号/);
  await f.video.receive(event(1, 'started', { start_call_ms: 90, mime: VIDEO_LIMITS.mime, width: 1280, height: 800 }));
  assert.throws(() => f.video.receive(event(2, 'chunk', { data: new Uint8Array(VIDEO_LIMITS.chunk_bytes + 1), final: false })), /大小/);
  let release; const original = f.video.freeSpace; f.video.freeSpace = () => new Promise(resolve => { release = resolve; });
  const write = f.video.receive(event(2, 'chunk', { data: media, timecode_ms: 0, final: false }));
  assert.throws(() => f.video.receive(event(3, 'chunk', { data: media, timecode_ms: 1, final: false })), /背压/);
  release(await original()); await write;
  f.video.requestStop(); await assert.rejects(f.video.receive(event(3, 'stopped', { final_sequence: 2, stop_call_ms: 250 })), /末块/);
  await f.video.seal(Date.now() + 10); assert.equal((await readVideoArchive(f.root,id,f.session,f.telemetry)).state, 'incomplete');
});
test('video failure requests independent stop before recorder stop and never resumes after hidden state', async t => {
  const order = []; const f = await fixture(t, { onFailure: () => order.push('control_stop'), send: m => { if (m.kind === 'stop') order.push('recorder_stop'); } });
  f.context.visible = false; f.video.viewChanged(f.context); assert.deepEqual(order, ['control_stop', 'recorder_stop']);
  f.context.visible = true; f.video.viewChanged(f.context); assert.equal(f.video.state, 'incomplete');
  await f.video.seal(Date.now() + 10); assert.equal(f.video.authorization, null);
});
test('video late summary across shared deadline leaves durable incomplete marker authoritative', async t => {
  const f = await fixture(t);
  await f.video.receive(event(1,'started',{ start_call_ms: 90, mime: VIDEO_LIMITS.mime, width: 1, height: 1 })); f.video.requestStop();
  await f.video.receive(event(2,'chunk',{data:media,timecode_ms:0,final:true})); await f.video.receive(event(3,'stopped',{final_sequence:2,stop_call_ms:150}));
  let release, entered; const reached = new Promise(r => { entered = r; }), hold = new Promise(r => { release = r; }), original = fsPromises.link;
  t.mock.method(fsPromises,'link',async (a,b) => { if (b.endsWith('/summary.json')) { entered(); await hold; } await original(a,b); }); syncBuiltinESMExports();
  t.after(() => { release(); t.mock.restoreAll(); syncBuiltinESMExports(); });
  const sealing = f.video.seal(Date.now() + 50); await reached; await sealing; release(); await new Promise(r => setTimeout(r, 20));
  assert.equal(f.video.state,'incomplete'); assert.equal((await readVideoArchive(f.root,id,f.session,f.telemetry)).complete,false);
});
test('video media route handles GET HEAD and one Range, rejects tampering, symlinks and partial archives', async t => {
  const f = await fixture(t), archive = await complete(f), url = archive.media_url;
  let response = await videoResponse(new Request(url,{headers:{Range:'bytes=2-5'}}),f.root,archive);
  assert.equal(response.status,206); assert.equal(response.headers.get('content-range'),'bytes 2-5/8'); assert.deepEqual(Buffer.from(await response.arrayBuffer()),media.subarray(2,6));
  response = await videoResponse(new Request(url,{method:'HEAD'}),f.root,archive); assert.equal(response.headers.get('content-length'),'8'); assert.equal((await response.arrayBuffer()).byteLength,0);
  for (const range of ['bytes=1-2,4-5','bytes=9-','bytes=-0','bytes=4-2']) assert.equal((await videoResponse(new Request(url,{headers:{Range:range}}),f.root,archive)).status,416);
  assert.deepEqual(videoRange('bytes=-3',8),{start:5,end:7,partial:true});
  await assert.rejects(videoResponse(new Request(url),f.root,{...archive,complete:false}),/未授权/);
  const file = path.join(f.root,'simulation_video/recording.webm'); await writeFile(file,Buffer.alloc(8)); await assert.rejects(videoResponse(new Request(url),f.root,archive),/完整性/);
  await rm(file); await symlink(path.join(f.root,'brain_ready.json'),file); await assert.rejects(videoResponse(new Request(url),f.root,archive),/符号链接/);
});

async function preloadFixture({ hold = false, manualStop = false, receive = null, hidden = false, connected = true } = {}) {
  const source = await readFile(new URL('./simulation-preload.cjs', import.meta.url),'utf8');
  assert.ok(!source.includes('contextBridge.')); assert.ok(!source.includes('postMessage'));
  const commands = new Map(), calls = [], recorderEvents = new Map(); let recorder, fps, options, release;
  const blocked = new Promise(resolve => { release = resolve; });
  class Canvas { width = 640; height = 400; isConnected = true; getBoundingClientRect(){return {width:640,height:400};} captureStream(value){fps=value;return {getAudioTracks:()=>[],getVideoTracks:()=>[track],getTracks:()=>[track]};} }
  const track = { addEventListener(){}, stop(){} }, visible = new Canvas();
  class Recorder { static isTypeSupported(v){return v==='video/webm;codecs=vp8';} constructor(_stream, opts){options=opts;this.mimeType=opts.mimeType;recorder=this;this.state='inactive';} addEventListener(name,fn){recorderEvents.set(name,fn);} start(slice){assert.equal(slice,1000);this.state='recording';recorderEvents.get('start')({timeStamp:1});} stop(){this.state='inactive';if (!manualStop) setImmediate(() => {recorderEvents.get('dataavailable')({timeStamp:5,timecode:1000,data:new Blob([media])});recorderEvents.get('stop')({timeStamp:6});});} }
  let now = 0;
  vm.runInNewContext(source,{require:()=>({ipcRenderer:{on:(name,fn)=>commands.set(name,fn),send:(channel,value)=>calls.push({channel,...value}),invoke:async(channel,value)=>{calls.push({channel,...value}); if (receive) return receive(channel, value); if (hold && channel === 'video:event') await blocked; return {accepted:true,sequence:value.sequence};}}}),crypto:{randomUUID:()=> 'clock'},performance:{now:()=>++now,timeOrigin:1},document:{hidden,querySelectorAll:()=>[visible],documentElement:{classList:{contains:()=>connected}},addEventListener(){}},window:{addEventListener(){}},getComputedStyle:()=>({visibility:'visible',display:'block'}),HTMLCanvasElement:Canvas,MediaRecorder:Recorder,Uint8Array,Blob,Promise,setTimeout:(fn,ms)=>ms===240000?0:setTimeout(fn,ms),clearTimeout,setInterval:()=>0,clearInterval});
  return { commands, calls, recorderEvents, release, get recorder(){return recorder;}, get fps(){return fps;}, get options(){return options;} };
}
test('video source preflight reports actual hidden and disconnected canvas without starting a recorder', async () => {
  const request = { generation: 2, nonce: 'b'.repeat(32) };
  for (const [options, expected] of [[{}, true], [{ hidden: true }, false], [{ connected: false }, false]]) {
    const f = await preloadFixture(options);
    f.commands.get('simulation:source-check')(null, request);
    const reply = f.calls[0];
    assert.equal(reply.channel, 'simulation:source-status'); assert.equal(reply.ready, expected);
    assert.equal(reply.nonce, request.nonce); assert.equal(reply.generation, request.generation);
    assert.equal(f.recorder, undefined); assert.equal(f.calls.length, 1);
    if (!expected) {
      const facts = JSON.parse(reply.reason.slice(reply.reason.indexOf('：') + 1));
      assert.equal(facts.hidden, options.hidden ?? false); assert.equal(facts.connected, options.connected ?? true);
      // A real start with the same invalid source must still report a fault.
      f.commands.get('video:command')(null, { generation: 2, token: 'a'.repeat(64), kind: 'start' });
      await new Promise(resolve => setImmediate(resolve));
      assert.ok(f.calls.some(call => call.channel === 'video:fault'));
      assert.equal(f.recorder, undefined);
    }
  }
});
test('video isolated preload encodes fixed canvas and sends final data before stopped', async () => {
  const f = await preloadFixture(); const {commands, recorderEvents, calls} = f;
  const auth = {generation:2,token:'a'.repeat(64)}; commands.get('video:command')(null,{...auth,kind:'start'}); await new Promise(r=>setImmediate(r));
  recorderEvents.get('dataavailable')({timeStamp:2,timecode:0,data:new Blob([media])}); await new Promise(r=>setImmediate(r));
  commands.get('video:command')(null,{...auth,kind:'stop'}); await new Promise(r=>setImmediate(r)); await new Promise(r=>setImmediate(r));
  assert.equal(f.fps,15); assert.deepEqual(JSON.parse(JSON.stringify(f.options)),{mimeType:'video/webm;codecs=vp8',videoBitsPerSecond:2500000});
  assert.deepEqual(calls.filter(c=>c.channel==='video:event').map(c=>[c.kind,c.final??null]),[['started',null],['chunk',false],['chunk',true],['stopped',null]]);
  assert.equal(calls.at(-1).final_sequence,3); assert.equal(f.recorder.state,'inactive');
});

test('video preload permits only one pending chunk beside the in-flight data and faults on excess', async () => {
  const f = await preloadFixture({hold:true}), auth = {generation:2,token:'a'.repeat(64)};
  f.commands.get('video:command')(null,{...auth,kind:'start'});
  for (let i=0;i<3;i++) f.recorderEvents.get('dataavailable')({timeStamp:2+i,timecode:i*1000,data:new Blob([media])});
  await new Promise(r=>setImmediate(r));
  assert.equal(f.calls.filter(c=>c.channel==='video:event').length,1);
  assert.match(f.calls.find(c=>c.channel==='video:fault').reason,/背压/);
  assert.equal(f.recorder.state,'inactive'); f.release(); await new Promise(r=>setImmediate(r));
});

test('video preload stop boundary seals queued timeslice and true final bytes only after stop and asynchronous ACKs', { timeout: 3000 }, async t => {
  const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
  const f = await fixture(t), arrivals = Array.from({ length: 5 }, deferred), acks = [deferred(), deferred()];
  const p = await preloadFixture({ manualStop: true, receive: async (channel, value) => {
    if (channel === 'video:fault') { f.video.fail(value.reason); throw new Error(value.reason); }
    const ack = await f.video.receive(value);
    arrivals[value.sequence - 1].resolve();
    if (value.sequence === 3 || value.sequence === 4) await acks[value.sequence - 3].promise;
    return ack;
  } });
  const auth = f.video.authorization, command = value => p.commands.get('video:command')(null, { ...auth, ...value });
  const data = (timeStamp, timecode, bytes) => p.recorderEvents.get('dataavailable')({ timeStamp, timecode, data: new Blob([bytes]) });
  const tick = () => new Promise(r => setImmediate(r));
  const queued = Buffer.from([1, 2, 3]), last = Buffer.from([4, 5]);
  try {
    command({ kind: 'start' }); await arrivals[0].promise;
    data(2, 0, media); await arrivals[1].promise;
    f.video.requestStop(); command({ kind: 'stop' });
    assert.equal(p.recorder.state, 'inactive');
    // The already queued timeslice is dispatched after stop() made state inactive.
    await tick(); data(2.5, 500, new Uint8Array());
    await tick(); data(3, 1000, queued); await tick();
    assert.equal(p.calls.filter(c => c.channel === 'video:event').length, 2);
    assert.equal(f.video.finalChunk, false);
    assert.equal((await readVideoArchive(f.root, id, f.session, f.telemetry)).complete, false);
    // The true final data event releases the preceding candidate as non-final.
    await tick(); data(4, 2000, last); await arrivals[2].promise;
    assert.equal(p.calls.at(-1).final, false);
    assert.equal(f.video.stopped, false);
    assert.equal((await readVideoArchive(f.root, id, f.session, f.telemetry)).complete, false);
    await tick(); p.recorderEvents.get('stop')({ timeStamp: 5 });
    const sealing = f.video.seal(Date.now() + 1000);
    assert.equal(p.calls.filter(c => c.kind === 'stopped').length, 0);
    acks[0].resolve(); await arrivals[3].promise;
    assert.equal(f.video.finalChunk, true);
    assert.equal(f.video.stopped, false);
    assert.equal((await readVideoArchive(f.root, id, f.session, f.telemetry)).complete, false);
    // Even after final bytes are durable, stopped cannot overtake their ACK.
    assert.equal(p.calls.filter(c => c.kind === 'stopped').length, 0);
    acks[1].resolve(); await arrivals[4].promise; await sealing;
    const archive = await readVideoArchive(f.root, id, f.session, f.telemetry);
    assert.equal(archive.state, 'sealed'); assert.deepEqual(f.failures, []);
    assert.deepEqual(p.calls.filter(c => c.channel === 'video:event').map(c => [c.kind, c.final ?? null]), [['started', null], ['chunk', false], ['chunk', false], ['chunk', true], ['stopped', null]]);
    assert.equal(p.calls.at(-1).final_sequence, 4);
    assert.deepEqual(await readFile(path.join(f.root, 'simulation_video/recording.webm')), Buffer.concat([media, queued, last]));
  } finally { acks.forEach(ack => ack.resolve()); }
});

test('video preload stop boundary without stop event cannot publish a complete archive', async t => {
  const f = await fixture(t);
  let started; const ready = new Promise(resolve => { started = resolve; });
  const p = await preloadFixture({ manualStop: true, receive: async (_channel, value) => { const ack = await f.video.receive(value); started(); return ack; } });
  const auth = f.video.authorization, command = kind => p.commands.get('video:command')(null, { ...auth, kind });
  command('start'); await ready; f.video.requestStop(); command('stop');
  p.recorderEvents.get('dataavailable')({ timeStamp: 3, timecode: 0, data: new Blob([media]) });
  await f.video.seal(Date.now() + 25);
  assert.equal(f.video.state, 'incomplete'); assert.equal(f.video.finalChunk, false);
  assert.equal(p.calls.filter(c => c.kind === 'chunk' || c.kind === 'stopped').length, 0);
  assert.equal((await readVideoArchive(f.root, id, f.session, f.telemetry)).complete, false);
});

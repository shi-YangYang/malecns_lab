import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm, symlink } from 'node:fs/promises';
import fsPromises from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import ts from 'typescript';
import { CAPTURE_LIMITS, SimulationCapture, captureIdentity, readCaptureArchive, readCapturedFrame, readCapturedBatch } from './recording.mjs';
import { ExperimentController } from './controller.mjs';

const id = 'desktop-20261023-115959-abcdef12';
const jpeg = Buffer.from([0xff, 0xd8, 1, 2, 3, 0xff, 0xd9]);
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
function fakeImage(width = 640, height = 400, data = jpeg) { return { isEmpty: () => false, getSize: () => ({ width, height }), resize: size => fakeImage(size.width, size.height, data), toJPEG: quality => { assert.equal(quality, 70); return data; } }; }
async function fixture(t, options = {}) {
  const parent = await mkdtemp(path.join(tmpdir(), 'malecns-capture-')); t.after(() => rm(parent, { recursive: true, force: true }));
  // /var is a symlink on macOS; use the actual canonical temp root in fixtures.
  const { realpath } = await import('node:fs/promises');
  const base = await realpath(parent), root = path.join(base, id), sourceRoot = path.join(base, 'source');
  await mkdir(path.join(root, 'telemetry/batches'), { recursive: true }); await mkdir(sourceRoot);
  for (const name of ['recording.mjs','main.mjs','controller.mjs','boundaries.mjs','preload.cjs']) await writeFile(path.join(sourceRoot, name), name);
  const ready = { run_id: id, session_id: 'neural-session', policy_id: 'policy', checkpoint_id: 'checkpoint', input_map_id: 'input' };
  const manifest = { ...ready, schema: 'malecns_lab.x2_behavior_telemetry_manifest.v1', execution_id: ready.session_id, clock_domain: 'host_worker_monotonic', groups: [], selected_neurons: [], dt_ms: 0.1 };
  const registration = { run_id: id, session_id: ready.session_id, output: root, simulation_only: true, qualification_only: true, checkpoint_metadata: { checkpoint_id: ready.checkpoint_id } };
  for (const [name, data] of Object.entries({ 'behavior_registration.json': registration, 'brain_ready.json': ready, 'telemetry/manifest.json': manifest, 'official_assets.json': { sha256: { mc: 'a'.repeat(64) } } })) await writeFile(path.join(root, name), JSON.stringify(data));
  const batch = { ...ready, execution_id: ready.session_id, schema: 'malecns_lab.x2_behavior_telemetry_batch.v1', batch_sequence: 1, host_monotonic_ns: 9000000000, skill_execution_id: 'actual-skill-before-capture', phase: 'precondition', step_start: 0, step_end: 50, dt_ms: 0.1, group_rates_hz: [], spike_events: [] };
  await writeFile(path.join(root, 'telemetry/batches/000001.json'), JSON.stringify(batch)); await writeFile(path.join(root, 'telemetry/latest.json'), JSON.stringify(batch));
  await writeFile(path.join(root, 'task_requests.jsonl'), JSON.stringify({ run_id: id, session_id: ready.session_id, request_id: 'r1', task: 'GREET', host_received_monotonic: 9.5 }) + '\n');
  await writeFile(path.join(root, 'behavior_events.jsonl'), JSON.stringify({ kind: 'execution', host_monotonic: 10.1, event: { run_id: id, session_id: ready.session_id, request_id: 'r1', execution_id: 'skill-not-neural', received_monotonic: 10.1, raw_feedback: { remote_received_monotonic: 1.2 }, behavior: 'HOLD', status: 'RUNNING' } }) + '\n');
  let now = 10000000000n;
  const context = { root, status: { state: 'running', run_id: id, resources_stopped: false, request_id: 'r1' }, visible: true, loaded: true, loading: false, error: null };
  const errors = [];
  const capture = new SimulationCapture({ root, recordId: id, sourceRoot, capture: async () => fakeImage(1920, 1080), context: () => context, onFailure: message => errors.push(message), clock: () => now, platform: 'darwin', freeSpace: async () => ({ bavail: 1024 * 1024, bsize: 1024 }), ...options });
  await capture.initialize();
  const session = { schema: 'malecns_lab.desktop_event.v1', run_id: id, resources_stopped: true, state: 'completed' };
  await writeFile(path.join(root, 'desktop_session.json'), JSON.stringify(session));
  return { capture, root, manifest, batch, ready, context, errors, setTime: value => { now = value; }, session };
}

test('qualified own-run capture seals bounded real-format frames and host-only index; does not overwrite', async t => {
  const f = await fixture(t); await f.capture.tick();
  assert.equal(f.capture.frames.length, 1); assert.equal(f.capture.frames[0].width, 1280); assert.equal(f.capture.frames[0].height, 720);
  assert.equal(f.capture.frames[0].context.skill_execution_id, 'actual-skill-before-capture');
  assert.equal(f.capture.frames[0].capture_start_host_monotonic_ns, '10000000000');
  await f.capture.seal();
  const archive = await readCaptureArchive(f.root, id, f.session, f.manifest);
  assert.equal(archive.state, 'sealed'); assert.equal(archive.replay.entries.find(e => e.kind === 'execution').skill_execution_id, 'skill-not-neural');
  assert.equal(archive.replay.entries.find(e => e.kind === 'execution').host_monotonic_ns, '10100000000');
  const image = await readCapturedFrame(f.root, archive, 1); assert.equal(image.jpeg_base64, jpeg.toString('base64'));
  await assert.rejects(f.capture.initialize(), /EEXIST/);
});

test('single in-flight capture, hide gap and no catch-up; recovered sampling preserves the gap', async t => {
  const pending = deferred(); let calls = 0;
  const f = await fixture(t, { capture: () => { calls++; return pending.promise; } });
  const first = f.capture.tick();
  // sample must finish its bounded identity reads before reaching injected capture.
  while (!calls) await new Promise(resolve => setImmediate(resolve));
  await f.capture.tick(); assert.equal(calls, 1);
  pending.resolve(fakeImage()); await first;
  f.context.visible = false; f.setTime(11000000000n); await f.capture.tick();
  assert.equal(f.capture.frames.length, 1); assert.equal(f.capture.gap.reason, 'view_hidden');
  f.context.visible = true; f.setTime(12000000000n); await f.capture.tick();
  assert.equal(f.capture.gaps.length, 1); assert.equal(f.capture.frames.length, 2); assert.equal(calls, 2);
  await f.capture.seal();
});

test('capture errors send independent stop before freezing; time/frame boundaries never expand', async t => {
  let f, observed;
  f = await fixture(t, { onFailure: () => { observed = f.capture.accepting; f.context.status.state = 'stopping'; } });
  f.setTime(250000000000n); await f.capture.tick();
  assert.equal(observed, true); assert.equal(f.capture.accepting, false); assert.equal(f.capture.frames.length, 0); assert.match(f.capture.error, /上限/);
  await f.capture.seal();
  const archive = await readCaptureArchive(f.root, id, f.session, f.manifest); assert.equal(archive.state, 'failed'); assert.equal(archive.complete, false);
});

test('identity changes, empty images, oversized image and disk reserve fail without fabricated frames', async t => {
  const changed = await fixture(t); await writeFile(path.join(changed.root, 'brain_ready.json'), JSON.stringify({ ...changed.ready, checkpoint_id: 'other' })); await changed.capture.tick(); assert.equal(changed.capture.frames.length, 0); assert.match(changed.errors[0], /身份/);
  const empty = await fixture(t, { capture: async () => ({ isEmpty: () => true }) }); await empty.capture.tick(); assert.match(empty.errors[0], /空图/);
  const large = await fixture(t, { capture: async () => fakeImage(640, 400, Buffer.alloc(CAPTURE_LIMITS.frame_bytes + 1)) }); await large.capture.tick(); assert.match(large.errors[0], /单帧/);
  const low = await fixture(t); low.capture.freeSpace = async () => ({ bavail: 0, bsize: 4096 }); await assert.rejects(low.capture.initialize(), /512 MiB/);
});

test('current and missing-summary archives cannot masquerade as sealed replay', async t => {
  const f = await fixture(t); await f.capture.tick();
  const partial = await readCaptureArchive(f.root, id, f.session, f.manifest); assert.equal(partial.state, 'incomplete'); assert.equal(partial.replay, null);
  await assert.rejects(readCapturedFrame(f.root, partial, 1), /封存/);
  f.capture.freeze('test_end');
});

test('late capture after bounded seal cannot publish a frame or flow into another generation', async t => {
  const pending = deferred(); let entered = false;
  const f = await fixture(t, { capture: () => { entered = true; return pending.promise; } });
  const work = f.capture.tick(); while (!entered) await new Promise(resolve => setImmediate(resolve));
  await f.capture.seal('window_close'); assert.equal(f.capture.state, 'incomplete');
  pending.resolve(fakeImage()); await work;
  assert.equal(f.capture.frames.length, 0); assert.deepEqual(await readdir(path.join(f.root, 'simulation_capture/frames')), []);
  assert.equal((await readCaptureArchive(f.root, id, f.session, f.manifest)).state, 'incomplete');
});

test('summary publication already in IO at seal timeout cannot become a completed archive later', async t => {
  const f = await fixture(t); await f.capture.tick();
  const entered = deferred(), release = deferred(), published = deferred();
  const originalLink = fsPromises.link;
  t.mock.method(fsPromises, 'link', async (source, target) => {
    if (target === path.join(f.root, 'simulation_capture/summary.json')) {
      entered.resolve(); await release.promise;
      await originalLink(source, target); published.resolve();
    } else await originalLink(source, target);
  });
  syncBuiltinESMExports();
  t.after(() => { release.resolve(); t.mock.restoreAll(); syncBuiltinESMExports(); });
  const sealing = f.capture.seal('window_close'); await entered.promise;
  const status = await sealing;
  assert.equal(status.state, 'incomplete'); assert.equal(status.complete, false);
  assert.match(status.reason, /2 秒/);
  release.resolve(); await published.promise; await new Promise(resolve => setImmediate(resolve));
  assert.equal(JSON.parse(await readFile(path.join(f.root, 'simulation_capture/summary.json'), 'utf8')).state, 'sealed');
  const archive = await readCaptureArchive(f.root, id, f.session, f.manifest);
  assert.equal(archive.state, 'incomplete'); assert.equal(archive.complete, false); assert.equal(archive.replay, null);
  assert.equal(f.capture.state, 'incomplete'); assert.equal(f.capture.generation, 2);
});

test('joint IPC and renderer reject same-identity batch content/time changes after sealing and future selection', async t => {
  const f = await fixture(t);
  await writeFile(path.join(f.root, 'telemetry/batches/000002.json'), JSON.stringify({ ...f.batch, batch_sequence: 2, host_monotonic_ns: 20000000000 }));
  await f.capture.tick(); await f.capture.seal();
  const controller = new ExperimentController(path.dirname(f.root)); controller.recordsRoot = path.dirname(f.root);
  const opened = await controller.openRecord(id);
  const cursorIndex = opened.capture.replay.entries.findIndex(entry => entry.kind === 'frame');
  assert.deepEqual(await controller.readBatch(id, 1, cursorIndex), f.batch);
  const source = await readFile(new URL('../src/data.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
  const { replayAt, validReplayBatch } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
  const selected = replayAt(opened.capture.replay, cursorIndex);
  assert.equal(validReplayBatch(f.batch, f.manifest, selected), true);
  assert.equal(validReplayBatch({ ...f.batch, host_monotonic_ns: 20000000000 }, f.manifest, selected), false);
  assert.equal(validReplayBatch({ ...f.batch, batch_sequence: 2 }, f.manifest, selected), false);
  assert.equal(validReplayBatch(f.batch, f.manifest, { ...selected, cursor: '8000000000' }), false);
  await assert.rejects(controller.readBatch(id, 2, cursorIndex), /游标/);
  await assert.rejects(controller.readBatch(id, 1, '../'), /游标/);
  const batchPath = path.join(f.root, 'telemetry/batches/000001.json');
  for (const changed of [{ ...f.batch, host_monotonic_ns: 20000000000 }, { ...f.batch, skill_execution_id: 'changed-content-same-time' }]) {
    await writeFile(batchPath, JSON.stringify(changed));
    await assert.rejects(readCaptureArchive(f.root, id, f.session, f.manifest), /完整性/);
    assert.equal((await controller.openRecord(id)).capture.state, 'invalid');
    await assert.rejects(controller.readBatch(id, 1, cursorIndex), /校验失败/);
    // Also closes the open-to-read window using the previously valid archive.
    await assert.rejects(readCapturedBatch(f.root, opened.capture, 1, cursorIndex), /完整性/);
  }
});

test('frame and index tampering, identity mismatch and symlink paths are refused', async t => {
  const f = await fixture(t); await f.capture.tick(); await f.capture.seal();
  let archive = await readCaptureArchive(f.root, id, f.session, f.manifest);
  await assert.rejects(readCapturedFrame(f.root, archive, '../1'), /序号/);
  await assert.rejects(readCaptureArchive(f.root, id, f.session, { ...f.manifest, session_id: 'other' }), /身份/);
  const framePath = path.join(f.root, 'simulation_capture/frames/000001.jpg'); await writeFile(framePath, Buffer.from([0xff, 0xd8, 0xff, 0xd9])); await assert.rejects(readCapturedFrame(f.root, archive, 1), /完整性/);
  await rm(framePath); await symlink(path.join(f.root, 'brain_ready.json'), framePath); await assert.rejects(readCapturedFrame(f.root, archive, 1), /符号链接/);
  await writeFile(path.join(f.root, 'simulation_capture/replay_index.json'), '{}'); await assert.rejects(readCaptureArchive(f.root, id, f.session, f.manifest), /完整性/);
  const g = await fixture(t); const readyPath = path.join(g.root, 'brain_ready.json'); await rm(readyPath); await symlink(path.join(g.root, 'telemetry/manifest.json'), readyPath); await assert.rejects(captureIdentity(g.root, id), /符号链接/);
});

test('renderer cursor selects only previous data, keeps motion-only frames and exposes old brain/gaps', async () => {
  const source = await readFile(new URL('../src/data.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
  const { replayAt } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
  const replay = { max_frame_age_ns: '1000000000', frames: [{ sequence: 1, capture_end_host_monotonic_ns: '2000000000' }, { sequence: 2, capture_end_host_monotonic_ns: '4500000000' }], gaps: [{ start_host_monotonic_ns: '3000000000', end_host_monotonic_ns: '4000000000' }], entries: [{ kind: 'batch', host_monotonic_ns: '1000000000', sequence: 1 }, { kind: 'frame', host_monotonic_ns: '2000000000', sequence: 1 }, { kind: 'gap_start', host_monotonic_ns: '3000000000' }, { kind: 'frame', host_monotonic_ns: '4500000000', sequence: 2 }, { kind: 'status', host_monotonic_ns: '6000000000' }, { kind: 'batch', host_monotonic_ns: '7000000000', sequence: 2 }] };
  assert.equal(replayAt(replay, 0).frame, null); assert.equal(replayAt(replay, 1).batch, 1);
  assert.equal(replayAt(replay, 2).frame, null); assert.equal(replayAt(replay, 3).frame.sequence, 2); assert.equal(replayAt(replay, 3).batch, 1); assert.equal(replayAt(replay, 3).previousBatch, true); assert.equal(replayAt(replay, 4).frame, null); assert.equal(replayAt(replay, 5).batch, 2);
});

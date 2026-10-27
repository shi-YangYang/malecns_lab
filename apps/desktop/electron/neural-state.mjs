import path from 'node:path';
import { constants } from 'node:fs';
import { open, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const NODES = 166700;
const BYTES = NODES * 3 * 4;
const CHANNELS = ['voltage_mean_mv', 'voltage_delta_mean_mv', 'spike_rate_hz'];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
function requireValue(ok, reason) { if (!ok) throw new Error(`全脑状态：${reason}`); }

// Separate, exact binary budget; the existing 256 KiB JSON/control limit stays intact.
async function readStateFile(root, relative) {
  const expected = path.join(root, 'telemetry', relative);
  const canonicalRoot = await realpath(root);
  const canonical = await realpath(expected);
  requireValue(canonical === path.join(canonicalRoot, 'telemetry', relative), '文件路径或符号链接无效');
  const handle = await open(canonical, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const info = await handle.stat();
    requireValue(info.isFile() && info.size === BYTES, '文件长度无效');
    const buffer = Buffer.alloc(BYTES + 1);
    let length = 0;
    while (length < buffer.length) {
      const result = await handle.read(buffer, length, buffer.length - length, length);
      if (!result.bytesRead) break;
      length += result.bytesRead;
    }
    requireValue(length === BYTES, '文件读取不完整或超过上限');
    return buffer.subarray(0, BYTES);
  } finally { await handle.close(); }
}

export class NeuralStateReader {
  constructor() { this.cached = null; }
  async read(root, manifest, batch) {
    const schema = manifest?.neural_state;
    if (!schema) {
      requireValue(!batch?.neural_state, '批次缺少对应清单');
      return null; // Legacy recordings remain readable; never fabricate voltage.
    }
    requireValue(schema.schema === 'malecns_lab.neural_state_manifest.v1' && schema.node_count === NODES &&
      schema.dtype === 'float32-le' && JSON.stringify(schema.channels) === JSON.stringify(CHANNELS) &&
      schema.window_steps === 50 && schema.baseline === 'request_start_pre_step' && schema.sample === 'post_step_window_mean' &&
      Number.isFinite(schema.v_rest_mv) && Number.isFinite(schema.v_threshold_mv) && schema.v_threshold_mv > schema.v_rest_mv &&
      /^[a-f0-9]{64}$/.test(schema.graph_sha256) && schema.graph_sha256 === manifest.graph?.reference_graph_sha256,
    '清单或连接组身份无效');
    requireValue(Array.isArray(schema.body_ids) && schema.body_ids.length === NODES &&
      schema.body_ids.every(id => typeof id === 'string' && /^\d+$/.test(id)) && new Set(schema.body_ids).size === NODES &&
      sha(JSON.stringify(schema.body_ids)) === schema.body_ids_sha256, '神经元顺序校验失败');
    if (!batch) return null;
    const ref = batch.neural_state;
    requireValue(batch.run_id === manifest.run_id && batch.execution_id === manifest.execution_id &&
      batch.session_id === manifest.session_id && batch.checkpoint_id === manifest.checkpoint_id && batch.input_map_id === manifest.input_map_id &&
      Number.isInteger(batch.batch_sequence) && batch.batch_sequence > 0 && batch.batch_sequence <= 999999 &&
      Number.isInteger(batch.step_start) && Number.isInteger(batch.step_end) && batch.step_end > batch.step_start && batch.step_end - batch.step_start <= 50 &&
      batch.dt_ms === manifest.dt_ms && (batch.phase === 'precondition' || (typeof batch.request_id === 'string' && Number.isInteger(batch.decision_sequence))),
    '批次身份或时间范围无效');
    requireValue(ref?.schema === 'malecns_lab.neural_state_ref.v1' && ref.node_count === NODES && ref.bytes === BYTES &&
      ref.body_ids_sha256 === schema.body_ids_sha256 && /^[a-f0-9]{64}$/.test(ref.sha256) && typeof ref.archived === 'boolean',
    '膜电位数据缺失或描述无效');
    const archiveFile = `neural/${String(batch.batch_sequence).padStart(6, '0')}.f32`;
    requireValue(ref.archived ? ref.file === archiveFile : /^neural\/live-[0-7]\.f32$/.test(ref.file), '文件名与批次不匹配');
    const key = JSON.stringify([root, batch.run_id, batch.execution_id, batch.request_id, batch.batch_sequence, ref.sha256]);
    if (this.cached?.key === key) return this.cached.frame;
    const buffer = await readStateFile(root, ref.file);
    requireValue(sha(buffer) === ref.sha256, '数据校验失败或实时槽已更新');
    // Decode explicitly as little endian, independent of host architecture.
    const values = new Float32Array(NODES * 3);
    const maximumRate = 1000 / batch.dt_ms;
    for (let i = 0; i < values.length; i++) {
      const value = buffer.readFloatLE(i * 4);
      requireValue(Number.isFinite(value) && (i < NODES * 2 || (value >= 0 && value <= maximumRate + .01)), '非有限电位或无效放电率');
      values[i] = value;
    }
    const frame = { batch_sequence: batch.batch_sequence, run_id: batch.run_id, execution_id: batch.execution_id,
      request_id: batch.request_id ?? null, body_ids_sha256: ref.body_ids_sha256, values };
    this.cached = { key, frame };
    return frame;
  }
}

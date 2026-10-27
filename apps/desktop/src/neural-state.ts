import type { BrainDisplayManifest } from './brain-data';
import type { NeuralStateManifest, NeuralStateFrame, TelemetryManifest, TelemetryBatch } from './types';

// Display scales are fixed across batches/actions, not per-frame normalization.
export const VOLTAGE_DELTA_HALF_MV = 2;
export const SPIKE_RATE_HALF_HZ = 50;
export interface NeuralSignals { voltage: Float32Array; spikes: Float32Array }
export interface NeuralMapping { graphToDisplay: Int32Array; nodeCount: number; bodyIdsHash: string }
const isHash = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

/** The graph-order identity is verified once per run; no guessed atlas order. */
export async function createNeuralMapping(state: NeuralStateManifest, display: Pick<BrainDisplayManifest, 'node_count' | 'body_ids' | 'provenance'>): Promise<NeuralMapping> {
  if (!state || typeof state !== 'object') throw new Error('全脑状态清单缺失');
  const n = state.node_count;
  if (state.schema !== 'malecns_lab.neural_state_manifest.v1' || n !== display.node_count || !Number.isInteger(n) || n <= 0 ||
      !Array.isArray(state.body_ids) || state.body_ids.length !== n || !state.body_ids.every(id => typeof id === 'string' && /^\d+$/.test(id)) ||
      new Set(state.body_ids).size !== n || !isHash(state.body_ids_sha256) || !isHash(state.graph_sha256) || state.graph_sha256 !== display.provenance.graph_sha256 ||
      JSON.stringify(state.channels) !== '["voltage_mean_mv","voltage_delta_mean_mv","spike_rate_hz"]' || state.dtype !== 'float32-le' ||
      !Number.isFinite(state.v_rest_mv) || !Number.isFinite(state.v_threshold_mv) || state.v_threshold_mv <= state.v_rest_mv || state.window_steps !== 50 ||
      state.baseline !== 'request_start_pre_step' || state.sample !== 'post_step_window_mean') throw new Error('全脑状态身份或通道格式不匹配');
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(state.body_ids)));
  const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  if (hash !== state.body_ids_sha256) throw new Error('全脑神经元顺序校验失败');
  const indices = new Map(display.body_ids.map((id, index) => [id, index]));
  if (indices.size !== n) throw new Error('脑图神经元身份不完整');
  const graphToDisplay = new Int32Array(n);
  state.body_ids.forEach((id, index) => {
    const target = indices.get(id);
    if (target === undefined) throw new Error('全脑状态与脑图神经元不一致');
    graphToDisplay[index] = target;
  });
  return { graphToDisplay, nodeCount: n, bodyIdsHash: hash };
}

/** Accept only the state of this exact active batch/request, including 0-Hz cells. */
export function mapNeuralSignals(frame: NeuralStateFrame, batch: TelemetryBatch, manifest: TelemetryManifest, mapping: NeuralMapping): NeuralSignals {
  const ref = batch.neural_state, n = mapping.nodeCount;
  if (!ref || ref.schema !== 'malecns_lab.neural_state_ref.v1' || typeof ref.file !== 'string' || !ref.file || ref.node_count !== n || ref.bytes !== n * 3 * 4 ||
      !isHash(ref.sha256) || ref.body_ids_sha256 !== mapping.bodyIdsHash || typeof ref.archived !== 'boolean' ||
      frame.run_id !== manifest.run_id || frame.execution_id !== manifest.execution_id || frame.run_id !== batch.run_id || frame.execution_id !== batch.execution_id ||
      frame.batch_sequence !== batch.batch_sequence || frame.request_id !== (batch.request_id ?? null) || frame.body_ids_sha256 !== mapping.bodyIdsHash ||
      !(frame.values instanceof Float32Array) || frame.values.length !== n * 3) throw new Error('全脑状态与当前动作批次不匹配');
  const voltage = new Float32Array(n), spikes = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const mean = frame.values[i], delta = frame.values[n + i], hz = frame.values[2 * n + i];
    if (!Number.isFinite(mean) || !Number.isFinite(delta) || !Number.isFinite(hz) || hz < 0) throw new Error('全脑状态包含无效数值');
    const index = mapping.graphToDisplay[i], magnitude = Math.abs(delta);
    voltage[index] = magnitude / (magnitude + VOLTAGE_DELTA_HALF_MV);
    spikes[index] = hz / (hz + SPIKE_RATE_HALF_HZ);
  }
  return { voltage, spikes };
}

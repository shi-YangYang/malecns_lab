import type { Atlas } from '../../atlas/src/types';
import type { TelemetryBatch, TelemetryManifest, CaptureArchive, CaptureReplay, CaptureFrame, ReplayEntry, OpenRecord } from './types';

export async function loadAtlas(): Promise<Atlas> {
  const response = await fetch('./prototype/data/atlas.json');
  if (!response.ok) throw new Error('本地胞体图谱缺失，请恢复 atlas 导出后重新构建。');
  const atlas = await response.json() as Atlas;
  if (atlas.schema !== 'malecns_lab.atlas.v1' || !Number.isInteger(atlas.positioned_count) || atlas.positions.length !== atlas.positioned_count * 3 || atlas.body_ids.length !== atlas.positioned_count || !atlas.positions.every(Number.isFinite) || !atlas.body_ids.every(id => typeof id === 'string')) throw new Error('胞体坐标与身份不匹配');
  return atlas;
}
export function validManifest(value: TelemetryManifest): boolean {
  const behavior = value.schema === 'malecns_lab.x2_behavior_telemetry_manifest.v1';
  return (value.schema === 'malecns_lab.x2_telemetry_manifest.v1' || behavior) && (!behavior || (typeof value.session_id === 'string' && typeof value.input_map_id === 'string' && value.spike_source?.includes('X2BehaviorPolicy._step') === true)) && typeof value.run_id === 'string' && typeof value.execution_id === 'string' && Array.isArray(value.groups) && value.groups.length <= 64 && value.groups.every(group => typeof group.id === 'string' && typeof group.name === 'string' && Array.isArray(group.body_ids) && group.body_ids.every(id => typeof id === 'string') && Number.isInteger(group.neuron_count) && group.neuron_count === group.body_ids.length) && Array.isArray(value.selected_neurons) && value.selected_neurons.length <= 32 && value.selected_neurons.every(neuron => typeof neuron.body_id === 'string');
}
export function validBatch(value: TelemetryBatch, manifest: TelemetryManifest): boolean {
  const ids = new Set(manifest.selected_neurons.map(neuron => neuron.body_id));
  const behavior = manifest.schema === 'malecns_lab.x2_behavior_telemetry_manifest.v1';
  return value.schema === (behavior ? 'malecns_lab.x2_behavior_telemetry_batch.v1' : 'malecns_lab.x2_telemetry_batch.v1') && (!behavior || (value.session_id === manifest.session_id && value.checkpoint_id === manifest.checkpoint_id && value.input_map_id === manifest.input_map_id && (value.phase === 'precondition' || (typeof value.request_id === 'string' && Number.isInteger(value.decision_sequence))))) && value.run_id === manifest.run_id && value.execution_id === manifest.execution_id && Number.isInteger(value.batch_sequence) && value.batch_sequence >= 1 && Number.isInteger(value.step_start) && Number.isInteger(value.step_end) && value.step_end > value.step_start && value.step_end - value.step_start <= 50 && value.dt_ms === manifest.dt_ms && Array.isArray(value.group_rates_hz) && value.group_rates_hz.length === manifest.groups.length && value.group_rates_hz.every(rate => Number.isFinite(rate) && rate >= 0) && Array.isArray(value.spike_events) && value.spike_events.length <= 1600 && value.spike_events.every(event => ids.has(event.body_id) && Number.isInteger(event.step) && event.step >= value.step_start && event.step < value.step_end);
}

export function validCapture(capture: CaptureArchive, record: OpenRecord): boolean {
  if (!capture.replay) return ['absent', 'incomplete', 'invalid'].includes(capture.state);
  const replay = capture.replay, identity = replay.identity;
  return ['sealed', 'failed'].includes(capture.state) && replay.schema === 'malecns_lab.x2_simulation_capture.replay.v1' && replay.clock_domain === 'macos_same_host_monotonic' && identity.record_id === record.id && identity.run_id === record.manifest.run_id && identity.execution_id === record.manifest.execution_id && identity.session_id === record.manifest.session_id && identity.checkpoint_id === record.manifest.checkpoint_id && identity.policy_id === record.manifest.policy_id && identity.input_map_id === record.manifest.input_map_id && Array.isArray(replay.entries) && replay.entries.length <= 10000 && replay.entries.every(entry => typeof entry.host_monotonic_ns === 'string' && /^\d{1,24}$/.test(entry.host_monotonic_ns));
}
export function replayAt(replay: CaptureReplay, index: number): { frame: CaptureFrame | null; batch: number | null; batchEntry: ReplayEntry | null; previousBatch: boolean; cursor: string } {
  const entry = replay.entries[index];
  if (!entry) throw new Error('回放位置无效');
  const cursor = BigInt(entry.host_monotonic_ns);
  let batch: number | null = null, batchTime = 0n, batchEntry: ReplayEntry | null = null;
  for (const event of replay.entries) {
    if (BigInt(event.host_monotonic_ns) > cursor) break;
    if (event.kind === 'batch') { batch = event.sequence ?? null; batchTime = BigInt(event.host_monotonic_ns); batchEntry = event; }
  }
  let frame: CaptureFrame | null = null;
  for (const item of replay.frames) if (BigInt(item.capture_end_host_monotonic_ns) <= cursor) frame = item;
  const inGap = replay.gaps.some(gap => BigInt(gap.start_host_monotonic_ns) <= cursor && (gap.end_host_monotonic_ns === null || cursor < BigInt(gap.end_host_monotonic_ns)));
  if (inGap || (frame && cursor - BigInt(frame.capture_end_host_monotonic_ns) > BigInt(replay.max_frame_age_ns))) frame = null;
  return { frame, batch, batchEntry, previousBatch: batch !== null && cursor - batchTime > 2000000000n, cursor: cursor.toString() };
}
export function validReplayBatch(value: TelemetryBatch, manifest: TelemetryManifest, selected: ReturnType<typeof replayAt>): boolean {
  const entry = selected.batchEntry;
  return validBatch(value, manifest) && entry !== null && value.batch_sequence === entry.sequence && value.batch_sequence === selected.batch && Number.isFinite(value.host_monotonic_ns) && value.host_monotonic_ns >= 0 && value.host_monotonic_ns === entry.original_host_monotonic_ns && BigInt(Math.round(value.host_monotonic_ns)).toString() === entry.host_monotonic_ns && BigInt(entry.host_monotonic_ns) <= BigInt(selected.cursor);
}

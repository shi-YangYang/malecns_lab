import type { Atlas } from '../../atlas/src/types';
import type { TelemetryBatch, TelemetryManifest } from './types';

export async function loadAtlas(): Promise<Atlas> {
  const response = await fetch('./prototype/data/atlas.json');
  if (!response.ok) throw new Error('本地胞体图谱缺失，请恢复 atlas 导出后重新构建。');
  const atlas = await response.json() as Atlas;
  if (atlas.schema !== 'malecns_lab.atlas.v1' || !Number.isInteger(atlas.positioned_count) || atlas.positions.length !== atlas.positioned_count * 3 || atlas.body_ids.length !== atlas.positioned_count || !atlas.positions.every(Number.isFinite) || !atlas.body_ids.every(id => typeof id === 'string')) throw new Error('胞体坐标与身份不匹配');
  return atlas;
}
export function validManifest(value: TelemetryManifest): boolean {
  return value.schema === 'malecns_lab.x2_telemetry_manifest.v1' && typeof value.run_id === 'string' && typeof value.execution_id === 'string' && Array.isArray(value.groups) && value.groups.length <= 64 && value.groups.every(group => typeof group.id === 'string' && typeof group.name === 'string' && Array.isArray(group.body_ids) && group.body_ids.every(id => typeof id === 'string') && Number.isInteger(group.neuron_count) && group.neuron_count === group.body_ids.length) && Array.isArray(value.selected_neurons) && value.selected_neurons.length <= 32 && value.selected_neurons.every(neuron => typeof neuron.body_id === 'string');
}
export function validBatch(value: TelemetryBatch, manifest: TelemetryManifest): boolean {
  const ids = new Set(manifest.selected_neurons.map(neuron => neuron.body_id));
  return value.schema === 'malecns_lab.x2_telemetry_batch.v1' && value.run_id === manifest.run_id && value.execution_id === manifest.execution_id && Number.isInteger(value.batch_sequence) && value.batch_sequence >= 1 && Number.isInteger(value.step_start) && Number.isInteger(value.step_end) && value.step_end > value.step_start && value.step_end - value.step_start <= 50 && value.dt_ms === manifest.dt_ms && Array.isArray(value.group_rates_hz) && value.group_rates_hz.length === manifest.groups.length && value.group_rates_hz.every(rate => Number.isFinite(rate) && rate >= 0) && Array.isArray(value.spike_events) && value.spike_events.length <= 1600 && value.spike_events.every(event => ids.has(event.body_id) && Number.isInteger(event.step) && event.step >= value.step_start && event.step < value.step_end);
}

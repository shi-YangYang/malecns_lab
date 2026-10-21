import { SCHEMA } from './types';
import type { Atlas, Catalog, Dataset, Run } from './types';

function requireData(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`数据格式错误：${message}`);
}
function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value); }
function count(value: unknown): value is number { return finite(value) && Number.isInteger(value) && value >= 0; }
function object(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null; }
function values(value: unknown): value is unknown[] { return Array.isArray(value); }
function schema(value: unknown, label: string): asserts value is Record<string, unknown> {
  requireData(object(value) && value.schema === SCHEMA, `${label} 的 schema 应为 ${SCHEMA}`);
}
function localUrl(path: string, base: string): string {
  const url = new URL(path, base);
  requireData(url.origin === location.origin, '所有数据资产必须来自当前本地服务');
  return url.href;
}
async function readJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`无法读取 ${new URL(url).pathname}（HTTP ${response.status}）`);
  try { return await response.json(); } catch { throw new Error(`无法解析 ${new URL(url).pathname}，请重新导出数据。`); }
}
function catalogData(raw: unknown): Catalog {
  schema(raw, 'catalog');
  requireData(object(raw.graph) && count(raw.graph.node_count) && count(raw.graph.edge_count), '参考图计数缺失');
  requireData(object(raw.model) && typeof raw.model.name === 'string' && raw.model.trained === false, '必须使用已记录的固定诊断模型');
  requireData(object(raw.recording) && finite(raw.recording.frame_hz) && raw.recording.frame_hz > 0, '记录帧率缺失');
  requireData(typeof raw.atlas_url === 'string' && values(raw.runs) && raw.runs.length > 0, '未找到图谱或刺激记录');
  const ids = new Set<string>();
  for (const entry of raw.runs) {
    requireData(object(entry) && typeof entry.id === 'string' && typeof entry.url === 'string' && typeof entry.label === 'string' && typeof entry.stimulus === 'string' && count(entry.frame_count) && finite(entry.duration_s) && finite(entry.warmup_s), '刺激索引不完整');
    requireData(!ids.has(entry.id), '刺激 ID 重复'); ids.add(entry.id);
  }
  return raw as unknown as Catalog;
}
function atlasData(raw: unknown, catalog: Catalog): Atlas {
  schema(raw, 'atlas');
  requireData(raw.node_count === catalog.graph.node_count && count(raw.positioned_count) && count(raw.missing_count) && raw.positioned_count + raw.missing_count === raw.node_count, '节点及位置覆盖计数不一致');
  requireData(values(raw.positions) && raw.positions.length === raw.positioned_count * 3 && raw.positions.every(finite), '胞体坐标长度或数值不正确');
  requireData(values(raw.body_ids) && raw.body_ids.length === raw.positioned_count && raw.body_ids.every(id => typeof id === 'string'), '胞体 ID 与坐标不匹配');
  requireData(values(raw.groups), '缺少群组索引');
  const names = new Set<string>();
  for (const group of raw.groups) {
    requireData(object(group) && typeof group.name === 'string' && !names.has(group.name), '群组名称无效或重复'); names.add(group.name);
    requireData(count(group.total_neurons) && count(group.positioned_neurons) && group.positioned_neurons <= group.total_neurons, '群组计数无效');
    requireData(values(group.point_indices) && group.point_indices.length === group.positioned_neurons && group.point_indices.every(index => count(index) && index < (raw.positioned_count as number)), '群组坐标索引越界');
  }
  return raw as unknown as Atlas;
}
function runData(raw: unknown, entryId: string, atlas: Atlas): Run {
  schema(raw, entryId);
  requireData(raw.id === entryId && typeof raw.label === 'string' && typeof raw.stimulus === 'string', '回放身份不匹配');
  requireData(values(raw.groups) && raw.groups.every(name => typeof name === 'string'), '回放群组缺失');
  const groupNames = new Set(atlas.groups.map(group => group.name));
  requireData(raw.groups.every(name => groupNames.has(name as string)) && new Set(raw.groups).size === raw.groups.length, '回放群组与图谱不匹配');
  requireData(values(raw.time_s) && raw.time_s.every((time, index, times) => finite(time) && (index === 0 || time > (times[index - 1] as number))), '时间必须有限且严格递增');
  const frames = raw.time_s.length;
  requireData(finite(raw.frame_hz) && raw.frame_hz > 0 && finite(raw.warmup_s) && raw.warmup_s >= 0, '回放帧率或预热时长无效');
  requireData(values(raw.valid) && raw.valid.length === frames && raw.valid.every(value => typeof value === 'boolean'), '有效性掩码不匹配');
  requireData(values(raw.group_rates_hz) && raw.group_rates_hz.length === frames && raw.group_rates_hz.every(row => values(row) && row.length === (raw.groups as unknown[]).length && row.every(rate => rate === null || (finite(rate) && rate >= 0))), '群组 Hz 数组不匹配');
  for (const key of ['network_spikes', 'network_active_neurons']) {
    requireData(values(raw[key]) && raw[key].length === frames && raw[key].every(value => value === null || count(value)), `${key} 数组不匹配`);
  }
  if (raw.stimulus_preview !== null && raw.stimulus_preview !== undefined) {
    const preview = raw.stimulus_preview;
    requireData(object(preview) && typeof preview.url === 'string' && count(preview.width) && preview.width > 0 && count(preview.height) && preview.height > 0 && preview.frames === frames && preview.layout === 'vertical' && preview.kind === 'reconstructed', '刺激重建格式无效');
  }
  for (const key of ['simulation_start_s', 'simulation_end_s', 'frame_index']) {
    if (raw[key] !== undefined) requireData(values(raw[key]) && raw[key].length === frames && raw[key].every(finite), `${key} 数组不匹配`);
  }
  for (const key of ['prediction_valid', 'initialization']) {
    if (raw[key] !== undefined) requireData(values(raw[key]) && raw[key].length === frames && raw[key].every(value => typeof value === 'boolean'), `${key} 掩码不匹配`);
  }
  return raw as unknown as Run;
}
export async function loadDataset(): Promise<Dataset> {
  const catalogUrl = new URL('/data/catalog.json', location.href).href;
  const catalog = catalogData(await readJson(catalogUrl));
  const atlas = atlasData(await readJson(localUrl(catalog.atlas_url, catalogUrl)), catalog);
  const previewUrls = new Map<string, string>();
  const runs = await Promise.all(catalog.runs.map(async entry => {
    const url = localUrl(entry.url, catalogUrl);
    const run = runData(await readJson(url), entry.id, atlas);
    requireData(run.time_s.length === entry.frame_count, '目录帧数与回放记录不一致');
    requireData(run.model_source_id === catalog.model.source_id, '回放模型与目录模型来源不一致');
    requireData(run.stimulus === entry.stimulus && run.frame_hz === catalog.recording.frame_hz && run.warmup_s === entry.warmup_s, '回放刺激、帧率或预热时间与目录不一致');
    if (run.stimulus_preview) previewUrls.set(run.id, localUrl(run.stimulus_preview.url, url));
    return run;
  }));
  let maxRate = 0;
  for (const run of runs) run.group_rates_hz.forEach((row, frame) => {
    if (run.valid[frame]) for (const rate of row) if (rate !== null) maxRate = Math.max(maxRate, rate);
  });
  return { catalog, atlas, runs, maxRate: Math.max(maxRate, 1), previewUrls };
}

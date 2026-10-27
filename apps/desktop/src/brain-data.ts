export interface BrainClip {
  id: string; label: string; currents: number[][];
  decision: { run_id: string; decision_seq: number; request_id: string; behavior: string; neural_start_step: number; neural_end_step: number };
}
export interface BrainDisplayManifest {
  schema: string; node_count: number; point_count: number; body_ids: string[];
  readout_indices: number[]; edges: [number, number, number][]; clips: BrainClip[];
  display_scale_mv: number; window_steps: number; dt_ms: number;
  reference_currents: number[][]; comparison_decisions: number[];
  files: Record<string, { bytes: number; sha256: string }>;
  provenance: { graph_sha256: string; run_id: string; edge_geometry: string; signal_scope: string };
}
export interface BrainDisplayData {
  manifest: BrainDisplayManifest; positions: Float32Array; owners: Uint32Array; centers: Float32Array;
  activityEdges?: [number, number, number][];
}

interface NeuronCoverage {
  schema: string; node_count: number; point_count: number; positioned_count: number; base_positioned_count: number;
  graph_sha256: string; base_manifest_sha256: string; body_ids_sha256: string;
  files: Record<string, { bytes: number; sha256: string }>;
}

/** Append only previously absent identities; existing structure and edge anchors stay intact. */
export function mergeNeuronCoverage(data: BrainDisplayData, coverage: NeuronCoverage, baseHash: string, bodyHash: string,
    points: Float32Array, owners: Uint32Array): BrainDisplayData {
  const m = data.manifest, count = coverage.point_count;
  if (coverage.schema !== 'malecns_lab.neuron_coverage.v1' || coverage.node_count !== m.node_count ||
      coverage.positioned_count !== m.node_count || coverage.graph_sha256 !== m.provenance.graph_sha256 ||
      coverage.base_manifest_sha256 !== baseHash || coverage.body_ids_sha256 !== bodyHash ||
      !Number.isInteger(count) || count < 0 || count > m.node_count || owners.length !== count || points.length !== count * 3 ||
      !points.every(Number.isFinite)) throw new Error('补充细胞坐标身份或长度无效');
  const present = new Uint8Array(m.node_count);
  data.owners.forEach(owner => { present[owner] = 1; });
  if (present.reduce((sum, value) => sum + value, 0) !== coverage.base_positioned_count) throw new Error('原始细胞覆盖数不匹配');
  const centers = new Float32Array(data.centers);
  owners.forEach((owner, i) => {
    if (owner >= m.node_count || present[owner]) throw new Error('补充细胞身份重复或越界');
    present[owner] = 1;
    centers.set(points.subarray(i * 3, i * 3 + 3), owner * 3);
  });
  if (!present.every(value => value === 1)) throw new Error('仍有神经元缺少真实展示坐标');
  const positions = new Float32Array(data.positions.length + points.length);
  positions.set(data.positions); positions.set(points, data.positions.length);
  const combinedOwners = new Uint32Array(data.owners.length + owners.length);
  combinedOwners.set(data.owners); combinedOwners.set(owners, data.owners.length);
  return { ...data, positions, owners: combinedOwners, centers };
}

async function digest(buffer: ArrayBuffer): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))].map(v => v.toString(16).padStart(2, '0')).join('');
}

export interface BrainGraphInfo { node_count: number; edge_count: number; graph_sha256: string }

export function validateBrainGraphInfo(source: unknown, atlasNodeCount: number, display: BrainDisplayManifest): BrainGraphInfo {
  const value = source as { schema?: string; graph?: Partial<BrainGraphInfo> } | null;
  const graph = value?.graph;
  if (value?.schema !== 'malecns_lab.atlas.v1' || !graph || !Number.isInteger(graph.node_count) || graph.node_count !== atlasNodeCount || graph.node_count !== display.node_count ||
      !Number.isInteger(graph.edge_count) || Number(graph.edge_count) <= 0 || typeof graph.graph_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(graph.graph_sha256) || graph.graph_sha256 !== display.provenance.graph_sha256) {
    throw new Error('脑模型信息与当前图谱身份不匹配');
  }
  return graph as BrainGraphInfo;
}

export async function loadBrainGraphInfo(atlasNodeCount: number, display: BrainDisplayManifest): Promise<BrainGraphInfo> {
  const response = await fetch('./prototype/data/atlas-source.json');
  if (!response.ok) throw new Error('脑模型来源信息缺失');
  return validateBrainGraphInfo(await response.json(), atlasNodeCount, display);
}

export async function loadBrainDisplay(): Promise<BrainDisplayData> {
  const response = await fetch('./prototype/brain/manifest.json');
  if (!response.ok) throw new Error('脑结构资产缺失，请先导出本地真实数据并重新构建。');
  const manifestBytes = await response.arrayBuffer();
  const m = JSON.parse(new TextDecoder().decode(manifestBytes)) as BrainDisplayManifest;
  const validIndex = (n: number): boolean => Number.isInteger(n) && n >= 0 && n < m.node_count;
  if (m.schema !== 'malecns_lab.brain_display.v1' || m.node_count !== 166700 || !Number.isInteger(m.point_count) || m.point_count <= 0 || m.point_count > 1000000 ||
      m.body_ids?.length !== m.node_count || new Set(m.body_ids).size !== m.node_count || !m.body_ids.every(id => /^\d+$/.test(id)) ||
      m.readout_indices?.length !== 3555 || !m.readout_indices.every(validIndex) || new Set(m.readout_indices).size !== 3555 ||
      !m.edges?.length || m.edges.length > 18000 || !m.edges.every(e => e.length === 3 && validIndex(e[0]) && validIndex(e[1]) && Number.isInteger(e[2]) && e[2] !== 0) ||
      m.clips?.length !== 5 || new Set(m.clips.map(c => c.id)).size !== 5 || !m.clips.some(c => c.id === 'idle') ||
      !m.clips.every(c => c.decision.run_id === m.provenance.run_id && c.currents.length === 2 && c.currents.every(a => a.length === 3555 && a.every(Number.isFinite))) ||
      m.reference_currents?.length !== 2 || !m.reference_currents.every(a => a.length === 3555 && a.every(Number.isFinite)) ||
      JSON.stringify(m.comparison_decisions) !== '[1,9,17,25]' ||
      !Number.isFinite(m.display_scale_mv) || m.display_scale_mv <= 0 || m.window_steps !== 250 || m.dt_ms !== .1) throw new Error('脑展示资产身份、活动维度或连接无效。');
  async function binary(name: string, bytes: number): Promise<ArrayBuffer> {
    const expected = m.files[name];
    if (!expected || expected.bytes !== bytes) throw new Error(`展示资产长度不符：${name}`);
    const result = await fetch(`./prototype/brain/${name}`);
    if (!result.ok) throw new Error(`展示资产缺失：${name}`);
    const buffer = await result.arrayBuffer();
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))].map(v => v.toString(16).padStart(2, '0')).join('');
    if (buffer.byteLength !== bytes || hash !== expected.sha256) throw new Error(`展示资产校验失败：${name}`);
    return buffer;
  }
  const [p, o, c] = await Promise.all([binary('synapses.f32', m.point_count*12), binary('owners.u32', m.point_count*4), binary('centers.f32', m.node_count*12)]);
  const data = { manifest: m, positions: new Float32Array(p), owners: new Uint32Array(o), centers: new Float32Array(c) };
  if (!data.positions.every(Number.isFinite) || !data.centers.every(Number.isFinite) || !data.owners.every(validIndex)) throw new Error('突触位置或神经元身份无效。');
  const activityResponse = await fetch('./prototype/brain/activity-edges.json');
  if (!activityResponse.ok) throw new Error('真实活动连接资产缺失，请导出 activity-edges 并重新构建。');
  const activity = await activityResponse.json() as {
    schema: string; node_count: number; graph_sha256: string; body_ids_sha256: string;
    centers_sha256: string; edges: [number, number, number][];
  };
  const bodyIdsHash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(m.body_ids))))].map(v => v.toString(16).padStart(2, '0')).join('');
  const positioned = new Uint8Array(m.node_count);
  data.owners.forEach(owner => { positioned[owner] = 1; });
  if (activity.schema !== 'malecns_lab.activity_edges.v1' || activity.node_count !== m.node_count ||
      activity.graph_sha256 !== m.provenance.graph_sha256 || activity.body_ids_sha256 !== bodyIdsHash ||
      activity.centers_sha256 !== m.files['centers.f32'].sha256 || !Array.isArray(activity.edges) ||
      !activity.edges.length || activity.edges.length > 6000 ||
      !activity.edges.every(e => Array.isArray(e) && e.length === 3 && validIndex(e[0]) && validIndex(e[1]) && e[0] !== e[1] &&
        positioned[e[0]] && positioned[e[1]] && Number.isSafeInteger(e[2]) && e[2] !== 0) ||
      new Set(activity.edges.map(e => `${e[0]}:${e[1]}`)).size !== activity.edges.length) {
    throw new Error('活动连接资产的连接组、节点顺序、坐标身份或边索引无效。');
  }
  const coverageResponse = await fetch('./prototype/brain/coverage.json');
  if (!coverageResponse.ok) throw new Error('全脑细胞补充坐标缺失，请导出 coverage 并重新构建。');
  const coverage = await coverageResponse.json() as NeuronCoverage;
  if (!Number.isInteger(coverage.point_count) || coverage.point_count < 0 || coverage.point_count > m.node_count) throw new Error('补充坐标数量无效');
  async function coverageBinary(name: string, bytes: number): Promise<ArrayBuffer> {
    const expected = coverage.files?.[name];
    if (!expected || expected.bytes !== bytes) throw new Error(`补充坐标长度不符：${name}`);
    const result = await fetch(`./prototype/brain/${name}`);
    if (!result.ok) throw new Error(`补充坐标缺失：${name}`);
    const buffer = await result.arrayBuffer();
    if (buffer.byteLength !== bytes || await digest(buffer) !== expected.sha256) throw new Error(`补充坐标校验失败：${name}`);
    return buffer;
  }
  const [extraPoints, extraOwners] = await Promise.all([
    coverageBinary('coverage-points.f32', coverage.point_count * 12),
    coverageBinary('coverage-owners.u32', coverage.point_count * 4),
  ]);
  return mergeNeuronCoverage({ ...data, activityEdges: activity.edges }, coverage, await digest(manifestBytes), bodyIdsHash,
    new Float32Array(extraPoints), new Uint32Array(extraOwners));
}

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
}

export async function loadBrainDisplay(): Promise<BrainDisplayData> {
  const response = await fetch('./prototype/brain/manifest.json');
  if (!response.ok) throw new Error('脑结构资产缺失，请先导出本地真实数据并重新构建。');
  const m = await response.json() as BrainDisplayManifest;
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
  return data;
}

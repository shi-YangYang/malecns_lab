export const SCHEMA = 'malecns_lab.atlas.v1';
export interface RunEntry {
  id: string; stimulus: string; label: string; url: string;
  frame_count: number; duration_s: number; warmup_s: number;
}
export interface Catalog {
  schema: string;
  graph: { node_count: number; edge_count: number; source_id: string; graph_sha256: string };
  atlas_url: string;
  model: { name: string; source_id: string; trained: boolean; description: string };
  recording: { name: string; source_path: string; frame_hz: number };
  runs: RunEntry[];
  provenance: unknown; notes: string[];
}
export interface AtlasGroup {
  name: string; point_indices: number[]; total_neurons: number;
  positioned_neurons: number; centroid: [number, number, number] | null;
}
export interface Atlas {
  schema: string; node_count: number; positioned_count: number; missing_count: number;
  positions: number[]; body_ids: string[]; coordinate_transform: unknown; groups: AtlasGroup[];
}
export interface Run {
  schema: string; id: string; stimulus: string; label: string;
  groups: string[]; time_s: number[]; valid: boolean[];
  group_rates_hz: (number | null)[][];
  network_spikes: (number | null)[]; network_active_neurons: (number | null)[];
  warmup_s: number; frame_hz: number; source_id: string; model_source_id: string;
  stimulus_preview: {
    url: string; width: number; height: number; frames: number;
    layout: 'vertical'; kind: 'reconstructed';
  } | null;
  frame_index?: number[]; prediction_valid?: boolean[]; initialization?: boolean[];
  invalid_reason?: (string | null)[]; simulation_start_s?: number[]; simulation_end_s?: number[];
  provenance: unknown; notes: string[];
}
export interface Dataset {
  catalog: Catalog; atlas: Atlas; runs: Run[]; maxRate: number;
  previewUrls: Map<string, string>;
}

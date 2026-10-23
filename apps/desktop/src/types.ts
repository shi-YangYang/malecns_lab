export interface TelemetryGroup { id: string; name: string; body_ids: string[]; neuron_count: number }
export interface TelemetryManifest {
  schema: 'malecns_lab.x2_telemetry_manifest.v1'; run_id: string; execution_id: string;
  checkpoint_id: string; dt_ms: number; groups: TelemetryGroup[];
  selected_neurons: { body_id: string; graph_index: number; group_id: string | null }[];
  reference_output_neuron_count?: number;
}
export interface TelemetryBatch {
  schema: 'malecns_lab.x2_telemetry_batch.v1'; run_id: string; execution_id: string;
  batch_sequence: number; phase: 'precondition' | 'forward'; observation_sequence: number | null;
  decision_sequence: number | null; step_start: number; step_end: number; dt_ms: number;
  host_monotonic_ns: number; group_rates_hz: number[]; spike_events: { body_id: string; step: number }[];
  dropped_batches: number;
}
export interface TelemetrySummary {
  schema: 'malecns_lab.x2_telemetry_summary.v1'; run_id: string; execution_id: string;
  status: 'running' | 'completed' | 'failed'; complete: boolean; batch_count: number;
  written_batches: number; dropped_batches: number; errors: string[]; error: string | null;
}
export interface ExperimentStatus {
  state: 'idle' | 'preparing' | 'awaiting_simulation' | 'running' | 'stopping' | 'completed' | 'cancelled' | 'failed';
  message: string; run_id?: string | null; simulator_ready: boolean; motion_started: boolean;
  stop_confirmed: boolean | null; resources_stopped: boolean; request_pending?: boolean;
}
export interface Snapshot { status: ExperimentStatus; manifest: TelemetryManifest | null; batch: TelemetryBatch | null; summary: TelemetrySummary | null; read_error: string | null; simulation_error: string | null; transport_delay_ms: number | null; received_at_ms: number }
export interface RecordEntry { id: string; state: string; run_id: string | null; message: string }
export interface OpenRecord { id: string; session: ExperimentStatus; manifest: TelemetryManifest; summary: TelemetrySummary | null; sequences: number[] }
export interface WorkbenchAPI {
  prepare(): Promise<ExperimentStatus>; start(): Promise<ExperimentStatus>; stop(): Promise<ExperimentStatus>;
  getSnapshot(): Promise<Snapshot>; listRecords(): Promise<RecordEntry[]>; openRecord(id: string): Promise<OpenRecord>;
  readBatch(id: string, sequence: number): Promise<TelemetryBatch>;
  setSimulationBounds(bounds: { x: number; y: number; width: number; height: number }): Promise<void>;
  setSimulationVisible(visible: boolean): Promise<void>;
}
declare global { interface Window { workbench?: WorkbenchAPI } }

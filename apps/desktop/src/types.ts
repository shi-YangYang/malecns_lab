export interface TelemetryGroup { id: string; name: string; body_ids: string[]; neuron_count: number }
export interface TelemetryManifest {
  schema: 'malecns_lab.x2_telemetry_manifest.v1' | 'malecns_lab.x2_behavior_telemetry_manifest.v1'; run_id: string; execution_id: string;
  session_id?: string; input_map_id?: string; policy_id?: string; spike_source?: string;
  checkpoint_id: string; dt_ms: number; groups: TelemetryGroup[];
  selected_neurons: { body_id: string; graph_index: number; group_id: string | null }[];
  reference_output_neuron_count?: number;
}
export interface TelemetryBatch {
  schema: 'malecns_lab.x2_telemetry_batch.v1' | 'malecns_lab.x2_behavior_telemetry_batch.v1'; run_id: string; execution_id: string;
  session_id?: string; request_id?: string | null; skill_execution_id?: string | null; checkpoint_id?: string; input_map_id?: string;
  batch_sequence: number; phase: 'precondition' | 'forward'; observation_sequence: number | null;
  decision_sequence: number | null; step_start: number; step_end: number; dt_ms: number;
  host_monotonic_ns: number; group_rates_hz: number[]; spike_events: { body_id: string; step: number }[];
  dropped_batches: number;
}
export interface TelemetrySummary {
  schema: 'malecns_lab.x2_telemetry_summary.v1' | 'malecns_lab.x2_behavior_telemetry_summary.v1'; run_id: string; execution_id: string;
  status: 'running' | 'completed' | 'failed'; complete: boolean; batch_count: number;
  written_batches: number; dropped_batches: number; errors: string[]; error: string | null;
}
export interface ExperimentStatus {
  state: 'idle' | 'preparing' | 'awaiting_simulation' | 'running' | 'stopping' | 'completed' | 'cancelled' | 'failed';
  message: string; run_id?: string | null; simulator_ready: boolean; motion_started: boolean;
  stop_confirmed: boolean | null; resources_stopped: boolean; request_pending?: boolean;
  task?: Task; phase?: string; behavior?: string; reason?: string; qualification_only?: boolean;
}
export type Task = 'GREET' | 'FACE_LEFT' | 'FACE_RIGHT' | 'SHOW_FORWARD' | 'END_INTERACTION';
export interface Snapshot { status: ExperimentStatus; manifest: TelemetryManifest | null; batch: TelemetryBatch | null; summary: TelemetrySummary | null; read_error: string | null; simulation_error: string | null; transport_delay_ms: number | null; received_at_ms: number; video?: { state: string; reason: string | null; bytes: number; complete: boolean } | null; capture?: { state: string; reason: string | null; frames: number; complete: boolean } | null }
export interface RecordEntry { id: string; state: string; run_id: string | null; message: string }
export interface OpenRecord { id: string; session: ExperimentStatus; manifest: TelemetryManifest; summary: TelemetrySummary | null; sequences: number[]; capture: CaptureArchive; video: VideoArchive }
export interface WorkbenchAPI {
  prepare(): Promise<ExperimentStatus>; start(): Promise<ExperimentStatus>; stop(): Promise<ExperimentStatus>;
  task(task: Task): Promise<ExperimentStatus>;
  getSnapshot(): Promise<Snapshot>; listRecords(): Promise<RecordEntry[]>; openRecord(id: string): Promise<OpenRecord>;
  readRecordedFrame(id: string, sequence: number): Promise<RecordedFrame>;
  readBatch(id: string, sequence: number, cursorIndex?: number): Promise<TelemetryBatch>;
  setSimulationBounds(bounds: { x: number; y: number; width: number; height: number }): Promise<void>;
  setSimulationVisible(visible: boolean): Promise<void>;
}
declare global { interface Window { workbench?: WorkbenchAPI } }

export interface CaptureIdentity { record_id: string; run_id: string; session_id: string; execution_id: string; checkpoint_id: string; policy_id: string; input_map_id: string; backend: 'sim' }
export interface CaptureFrame { sequence: number; capture_start_host_monotonic_ns: string; capture_end_host_monotonic_ns: string; width: number; height: number; bytes: number; sha256: string }
export interface ReplayEntry { kind: string; host_monotonic_ns: string; sequence?: number; original_host_monotonic_ns?: number; bytes?: number; sha256?: string; task?: string | null; state?: string | null; behavior?: string | null; request_id?: string | null; skill_execution_id?: string | null }
export interface CaptureReplay { schema: 'malecns_lab.x2_simulation_capture.replay.v1'; identity: CaptureIdentity; clock_domain: 'macos_same_host_monotonic'; max_frame_age_ns: string; entries: ReplayEntry[]; frames: CaptureFrame[]; gaps: { start_host_monotonic_ns: string; end_host_monotonic_ns: string | null; reason: string }[] }
export interface CaptureArchive { state: 'absent' | 'sealed' | 'failed' | 'incomplete' | 'invalid'; complete: boolean; reason?: string; manifest: { identity: CaptureIdentity } | null; replay: CaptureReplay | null; summary: { reason: string | null } | null }
export interface RecordedFrame { sequence: number; identity: CaptureIdentity; frame: CaptureFrame; jpeg_base64: string }

export interface VideoArchive { state: string; complete: boolean; reason?: string; media_url: string | null; alignment?: null; manifest: { identity: CaptureIdentity; media_anchor: null } | null }

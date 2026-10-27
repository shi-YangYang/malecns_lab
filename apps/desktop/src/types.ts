export interface TelemetryGroup { id: string; name: string; body_ids: string[]; neuron_count: number }
export interface NeuralStateManifest {
  schema: 'malecns_lab.neural_state_manifest.v1'; node_count: number; body_ids: string[];
  body_ids_sha256: string; graph_sha256: string;
  channels: ['voltage_mean_mv', 'voltage_delta_mean_mv', 'spike_rate_hz']; dtype: 'float32-le';
  v_rest_mv: number; v_threshold_mv: number; window_steps: 50;
  baseline: 'request_start_pre_step'; sample: 'post_step_window_mean';
}
export interface NeuralStateRef {
  schema: 'malecns_lab.neural_state_ref.v1'; file: string; bytes: number; sha256: string;
  body_ids_sha256: string; node_count: number; archived: boolean;
}
export interface NeuralStateFrame {
  batch_sequence: number; run_id: string; execution_id: string; request_id: string | null;
  body_ids_sha256: string; values: Float32Array;
}
export interface TelemetryManifest {
  schema: 'malecns_lab.x2_telemetry_manifest.v1' | 'malecns_lab.x2_behavior_telemetry_manifest.v1'; run_id: string; execution_id: string;
  session_id?: string; input_map_id?: string; policy_id?: string; spike_source?: string;
  checkpoint_id: string; dt_ms: number; groups: TelemetryGroup[];
  selected_neurons: { body_id: string; graph_index: number; group_id: string | null }[];
  reference_output_neuron_count?: number;
  neural_state?: NeuralStateManifest;
}
export interface TelemetryBatch {
  schema: 'malecns_lab.x2_telemetry_batch.v1' | 'malecns_lab.x2_behavior_telemetry_batch.v1'; run_id: string; execution_id: string;
  session_id?: string; request_id?: string | null; skill_execution_id?: string | null; checkpoint_id?: string; input_map_id?: string;
  batch_sequence: number; phase: 'precondition' | 'forward'; observation_sequence: number | null;
  decision_sequence: number | null; step_start: number; step_end: number; dt_ms: number;
  host_monotonic_ns: number; group_rates_hz: number[]; spike_events: { body_id: string; step: number }[];
  dropped_batches: number;
  neural_state?: NeuralStateRef;
}
export interface TelemetrySummary {
  schema: 'malecns_lab.x2_telemetry_summary.v1' | 'malecns_lab.x2_behavior_telemetry_summary.v1'; run_id: string; execution_id: string;
  status: 'running' | 'completed' | 'failed'; complete: boolean; batch_count: number;
  written_batches: number; dropped_batches: number; errors: string[]; error: string | null;
}
export interface ExperimentStatus {
  mode?: 'manual_demo' | 'brain_control'; brain_controlled?: boolean;
  state: 'idle' | 'preparing' | 'awaiting_simulation' | 'running' | 'stopping' | 'completed' | 'cancelled' | 'failed';
  message: string; run_id?: string | null; simulator_ready: boolean; motion_started: boolean;
  stop_confirmed: boolean | null; resources_stopped: boolean; request_pending?: boolean;
  action_busy?: boolean; active_request_id?: string | null; request_id?: string | null; closing?: boolean;
  task?: Task; phase?: string; behavior?: string; reason?: string; notice?: string; qualification_only?: boolean;
  clock_source?: 'python_runtime' | 'desktop_local'; host_clock_id?: string; host_monotonic_ns?: number;
}
export type Task = 'GREET' | 'FACE_LEFT' | 'FACE_RIGHT' | 'SHOW_FORWARD' | 'END_INTERACTION';
export interface Snapshot { status: ExperimentStatus; manifest: TelemetryManifest | null; batch: TelemetryBatch | null; summary: TelemetrySummary | null; read_error: string | null; simulation_error: string | null; neural_state?: NeuralStateFrame | null; neural_state_error?: string | null; transport_delay_ms: number | null; transport_clock?: HostClockSample | null; received_at_ms: number; video?: { state: string; reason: string | null; bytes: number; complete: boolean } | null; capture?: { state: string; reason: string | null; frames: number; complete: boolean } | null }
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
export interface HostClockSample { schema: 'malecns_lab.desktop_clock.v1'; type: 'clock'; run_id: string; clock_id: string; nonce: string; python_monotonic_ns: string; node_send_ns: string; node_receive_ns: string; clock_source: { api: 'time.monotonic_ns'; implementation: string; monotonic: true; adjustable: false } }
export interface CaptureFrame { sequence: number; capture_start_host_monotonic_ns: string; capture_end_host_monotonic_ns: string; capture_start_node_ns: string; capture_end_node_ns: string; clock_before: HostClockSample; clock_after: HostClockSample; width: number; height: number; bytes: number; sha256: string }
export interface ReplayEntry { kind: string; host_monotonic_ns: string; sequence?: number; original_host_monotonic_ns?: number; bytes?: number; sha256?: string; task?: string | null; state?: string | null; behavior?: string | null; request_id?: string | null; skill_execution_id?: string | null }
export interface CaptureReplay { schema: 'malecns_lab.x2_simulation_capture.replay.v2'; identity: CaptureIdentity; clock_domain: 'python_host_monotonic'; clock_id: string; clock_protocol: 'python_capture_outer_bounds_v1'; max_frame_age_ns: string; entries: ReplayEntry[]; frames: CaptureFrame[]; gaps: { start_host_monotonic_ns: string; end_host_monotonic_ns: string | null; reason: string }[]; node_local_events: { received_node_ns: string; state: string }[] }
export interface CaptureArchive { state: 'absent' | 'sealed' | 'failed' | 'incomplete' | 'invalid'; complete: boolean; reason?: string; manifest: { identity: CaptureIdentity } | null; replay: CaptureReplay | null; summary: { reason: string | null } | null }
export interface RecordedFrame { sequence: number; identity: CaptureIdentity; frame: CaptureFrame; jpeg_base64: string }

export interface VideoArchive { state: string; complete: boolean; reason?: string; media_url: string | null; alignment?: null; manifest: { identity: CaptureIdentity; media_anchor: null } | null }

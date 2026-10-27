import type { ExperimentStatus, Task, TelemetryBatch } from './types';

export const activeStates = new Set(['preparing', 'awaiting_simulation', 'running', 'stopping']);

/** Preparation and simulator readiness now belong to the same start request. */
export function startAction(status: ExperimentStatus, pending: boolean): 'prepare' | null {
  if (pending || status.request_pending) return null;
  return !activeStates.has(status.state) && status.resources_stopped ? 'prepare' : null;
}

/** Own the current status independently of the last telemetry snapshot. */
export class LiveStatus {
  current: ExperimentStatus | null = null;
  revision = 0;

  beginChange(): number { return ++this.revision; }

  accept(value: ExperimentStatus, revision: number): boolean {
    if (revision !== this.revision) return false;
    this.current = value;
    ++this.revision;
    return true;
  }
}

/** Display the current request only; completion feedback remains in the archive. */
export class ActivityDisplay {
  runId: string | null = null;
  requestId: string | null = null;
  mode: 'inactive' | 'initializing' | 'waiting' | 'active' | 'closing' = 'inactive';
  private awaitingRequest = false;
  private previousRequestId: string | null = null;
  private closedRequestId: string | null = null;

  reset(): void {
    this.runId = this.requestId = this.previousRequestId = this.closedRequestId = null;
    this.awaitingRequest = false; this.mode = 'inactive';
  }

  beginRequest(): void {
    this.previousRequestId = this.requestId;
    this.requestId = null; this.awaitingRequest = true; this.mode = 'waiting';
  }

  cancelRequest(): void { this.awaitingRequest = false; }

  /** Return true when the previously displayed request must be cleared. */
  update(status: ExperimentStatus): boolean {
    let clear = false;
    const runId = status.run_id ?? null;
    if (runId !== this.runId) { this.reset(); this.runId = runId; clear = true; }
    if (status.state !== 'running' || status.brain_controlled !== true) {
      this.mode = status.phase === 'initializing' && status.state === 'preparing' ? 'initializing' : 'inactive';
      return clear;
    }
    if (status.phase === 'initializing') { this.mode = 'initializing'; return clear; }
    const requestId = status.active_request_id ?? null;
    if (status.notice === 'request_rejected') this.awaitingRequest = false;
    // A local queued acknowledgement may still contain the interrupted request.
    if (this.awaitingRequest && (!requestId || requestId === this.previousRequestId)) {
      this.mode = 'waiting'; return clear;
    }
    if (!requestId || status.phase === 'waiting_task') { this.mode = 'waiting'; return clear; }
    this.awaitingRequest = false;
    if (requestId !== this.requestId) { this.requestId = requestId; clear = true; }
    if (status.closing === true && status.request_id === requestId) this.closedRequestId = requestId;
    this.mode = this.closedRequestId === requestId ? 'closing' : 'active';
    return clear;
  }

  accepts(batch: TelemetryBatch): boolean {
    if (!this.runId || batch.run_id !== this.runId) return false;
    return this.mode === 'initializing' ? batch.phase === 'precondition' :
      this.mode === 'active' && batch.phase === 'forward' && batch.request_id === this.requestId;
  }
}

export function canEmergencyStop(status: ExperimentStatus, lifecyclePending: boolean, stopPending: boolean): boolean {
  return !stopPending && status.state !== 'stopping' && (activeStates.has(status.state) || lifecyclePending);
}

export function canSubmitTask(status: ExperimentStatus, task: Task): boolean {
  if (status.state !== 'running' || status.mode !== 'brain_control' || status.brain_controlled !== true || status.phase === 'initializing') return false;
  if (task === 'END_INTERACTION') return !(status.task === task && (status.action_busy || status.request_pending));
  if (status.next_task) return false;
  if (status.can_queue_task === true && !status.request_pending) return true;
  return status.phase === 'waiting_task' && status.action_busy !== true && !status.request_pending;
}

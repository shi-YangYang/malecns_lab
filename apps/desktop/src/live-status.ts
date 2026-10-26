import type { ExperimentStatus, Task } from './types';

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

export function canEmergencyStop(status: ExperimentStatus, lifecyclePending: boolean, stopPending: boolean): boolean {
  return !stopPending && status.state !== 'stopping' && (activeStates.has(status.state) || lifecyclePending);
}

export function canSubmitTask(status: ExperimentStatus, task: Task): boolean {
  if (status.state !== 'running' || status.mode !== 'brain_control' || status.brain_controlled !== true || status.phase === 'initializing') return false;
  if (task === 'END_INTERACTION') return !(status.task === task && (status.action_busy || status.request_pending));
  return status.phase === 'waiting_task' && status.action_busy !== true && !status.request_pending;
}

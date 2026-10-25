import type { ExperimentStatus } from './types';

export const activeStates = new Set(['preparing', 'awaiting_simulation', 'running', 'stopping']);

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

export type ActivityReading =
  | { kind: 'baseline' | 'gap'; rates: null }
  | { kind: 'measured'; rates: number[] };

export interface ActivityPoint { atMs: number; reading: ActivityReading }

/** A display clock, never a source of neural telemetry or simulated time. */
export class ActivityTimeline {
  readonly windowMs = 20_000;
  readonly points: ActivityPoint[] = [];
  private pausedAt: number | null = null;
  private pausedMs = 0;

  constructor(private startedAt = performance.now()) {}

  elapsed(now = performance.now()): number {
    return Math.max(0, (this.pausedAt ?? now) - this.startedAt - this.pausedMs);
  }

  record(reading: ActivityReading, now = performance.now(), breakBefore = false): number {
    const atMs = this.elapsed(now);
    if (this.pausedAt !== null) return atMs;
    if (breakBefore) this.points.push({ atMs, reading: { kind: 'gap', rates: null } });
    const previous = this.points.at(-1);
    // Repeated polls may share the same reading. They do not create new brain samples.
    if (!previous || previous.reading.kind !== reading.kind || previous.reading.rates !== reading.rates || atMs - previous.atMs >= 100) {
      this.points.push({ atMs, reading });
    }
    const oldest = atMs - this.windowMs;
    while (this.points.length > 2 && this.points[1].atMs < oldest) this.points.shift();
    if (this.points.length > 256) this.points.splice(0, this.points.length - 256);
    return atMs;
  }

  pause(now = performance.now()): void {
    if (this.pausedAt !== null) return;
    this.record({ kind: 'gap', rates: null }, now);
    this.pausedAt = now;
  }

  resume(now = performance.now()): void {
    if (this.pausedAt === null) return;
    this.pausedMs += now - this.pausedAt;
    this.pausedAt = null;
  }

  reset(now = performance.now()): void {
    this.points.length = 0;
    this.startedAt = now; this.pausedMs = 0;
    if (this.pausedAt !== null) this.pausedAt = now;
  }
}

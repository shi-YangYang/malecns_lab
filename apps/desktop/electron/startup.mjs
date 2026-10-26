import { setTimeout as delay } from 'node:timers/promises';

export const DISPLAY_SURFACE_UNAVAILABLE = 'DISPLAY_SURFACE_UNAVAILABLE';

/** Continue only the session this click created, after the real canvas is ready. */
export async function finishStartup({ controller, confirmSource, stop, closing = () => false,
  recoverSource = null, onEvent = () => {}, wait = delay, now = () => performance.now(),
  timeoutMs = 300_000, sourceTimeoutMs = 20_000 }) {
  const output = controller.output, started = now(), deadline = started + timeoutMs;
  const active = () => !closing() && !controller.stopRequested && controller.output === output &&
    ['preparing', 'awaiting_simulation'].includes(controller.state.state) && !controller.state.resources_stopped;
  const note = (type, fields = {}) => onEvent({ type, elapsed_ms: Math.round(now() - started), ...fields });
  let sourceDeadline = null, sourceError = null, recoveryAttempted = false;
  const withinDeadline = () => now() < deadline && (sourceDeadline === null || now() < sourceDeadline);
  note('started');
  try {
    while (active()) {
      if (!withinDeadline()) throw sourceError ?? new Error('启动超时，已停止本次准备');
      if (controller.state.state === 'awaiting_simulation') {
        if (sourceDeadline === null) { sourceDeadline = now() + sourceTimeoutMs; note('source_wait_started'); }
        try { await confirmSource(); }
        catch (error) {
          sourceError = error;
          note('source_not_ready', { code: error.code === DISPLAY_SURFACE_UNAVAILABLE ? error.code : 'SOURCE_NOT_READY' });
          if (error.code === DISPLAY_SURFACE_UNAVAILABLE && !recoveryAttempted && recoverSource && active() && withinDeadline()) {
            recoveryAttempted = true;
            note('surface_recovery_requested');
            const recovered = await recoverSource();
            note(recovered === false ? 'surface_recovery_skipped' : 'surface_recovery_completed');
          }
          if (active() && withinDeadline()) await wait(200);
          continue;
        }
        // Recovery never substitutes for a fresh real capture. Stop/close/session
        // change can also happen while the asynchronous probe or recovery runs.
        if (!active()) break;
        if (!withinDeadline()) continue;
        note('source_confirmed');
        const result = controller.continue();
        note('continue_requested');
        return result;
      }
      await wait(100);
    }
    note('interrupted');
    return controller.state;
  } catch (error) {
    if (active()) stop();
    note('failed', { code: error.code === DISPLAY_SURFACE_UNAVAILABLE ? error.code : 'STARTUP_FAILED' });
    throw error;
  }
}

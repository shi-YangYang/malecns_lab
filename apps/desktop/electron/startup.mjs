import { setTimeout as delay } from 'node:timers/promises';

/** Continue only the session this click created, after the real canvas is ready. */
export async function finishStartup({ controller, confirmSource, stop, closing = () => false,
  wait = delay, now = () => performance.now(), timeoutMs = 300_000, sourceTimeoutMs = 20_000 }) {
  const output = controller.output, deadline = now() + timeoutMs;
  const active = () => !closing() && !controller.stopRequested && controller.output === output &&
    ['preparing', 'awaiting_simulation'].includes(controller.state.state) && !controller.state.resources_stopped;
  let sourceDeadline = null, sourceError = null;
  while (active()) {
    if (now() >= deadline || (sourceDeadline !== null && now() >= sourceDeadline)) {
      stop();
      throw new Error(sourceError?.message ?? '启动超时，已停止本次准备');
    }
    if (controller.state.state === 'awaiting_simulation') {
      sourceDeadline ??= now() + sourceTimeoutMs;
      try { await confirmSource(); }
      catch (error) { sourceError = error; if (active()) await wait(200); continue; }
      // Stop/close/session change can happen while the asynchronous probe runs.
      if (!active()) break;
      if (now() >= sourceDeadline || now() >= deadline) continue;
      return controller.continue();
    }
    await wait(100);
  }
  return controller.state;
}

"""Bounded task-only continuity diagnostic; preview never starts a runtime.

After the runtime reports awaiting_simulation, complete the initial VNC
Pause/Reset/Run manually, then type ``continue`` on this process's stdin.
``stop`` (or EOF/Ctrl-C) always requests the runtime's independent safe stop.
No resets, retraining, retries, SDK calls, or decision substitution occur here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

SCHEMA = 'malecns_lab.x2_continuity.v1'
ROOT = Path(__file__).resolve().parents[3]
TASKS = ('SHOW_FORWARD', 'FACE_LEFT', 'FACE_RIGHT', 'GREET')
EXPECTED = dict(zip(TASKS, ('ADVANCE', 'TURN_LEFT', 'TURN_RIGHT', 'WAVE_RIGHT')),
                END_INTERACTION='HOLD')
ARM = tuple('right_' + name + '_joint' for name in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_yaw', 'wrist_pitch', 'wrist_roll'))
MAX_LINE = 2 * 1024 * 1024
TURN_TOLERANCE_RAD = math.radians(5.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def default_sequence():
    """Linearized B(4, 2): its 17 tasks cover each directed pair exactly once."""
    a, cycle = [0] * 9, []

    def visit(t, period):
        if t > 2:
            if 2 % period == 0:
                cycle.extend(a[1:period + 1])
        else:
            a[t] = a[t - period]
            visit(t + 1, period)
            for value in range(a[t - period] + 1, 4):
                a[t] = value
                visit(t + 1, t)

    visit(1, 1)
    return [TASKS[i] for i in (*cycle, cycle[0])]


def planned_tasks(sequence=None, cycles=1):
    require(type(cycles) is int and 1 <= cycles <= 100, 'cycles must be within 1..100')
    base = default_sequence() if sequence is None else sequence.split(',')
    require(base and all(task in EXPECTED for task in base), 'Use exact runtime task enums')
    tasks = base * cycles
    if sequence is None:
        tasks += ['END_INTERACTION', TASKS[0]]
    require(len(tasks) <= 5000, 'Task count exceeds diagnostic bound')
    return tasks


class JsonlTail:
    """Bounded incremental reads, including a partial final line across polls."""
    def __init__(self, path):
        self.path, self.offset, self.buffer = Path(path), 0, b''
        self.caught_up = False

    def poll(self):
        self.caught_up = False
        try:
            with self.path.open('rb') as stream:
                require(os.fstat(stream.fileno()).st_size >= self.offset, 'Evidence file was truncated')
                stream.seek(self.offset)
                chunk = stream.read(256 * 1024)
                self.offset += len(chunk)
                self.caught_up = stream.tell() >= os.fstat(stream.fileno()).st_size
        except FileNotFoundError:
            self.caught_up = True
            return []
        parts = (self.buffer + chunk).split(b'\n')
        self.buffer = parts.pop()
        self.caught_up = self.caught_up and not self.buffer
        require(len(self.buffer) <= MAX_LINE and all(len(line) <= MAX_LINE for line in parts),
                'Evidence line exceeds diagnostic bound')
        return [json.loads(line) for line in parts if line]


def arm_sample(raw):
    sample = raw.get('sensors', {}).get('arm', {})
    joints = {j.get('name'): j for j in sample.get('message', {}).get('joints', [])}
    if not all(name in joints and finite(joints[name].get('position'))
               and finite(joints[name].get('velocity')) for name in ARM):
        return None
    return {name: dict(position=joints[name]['position'], velocity=joints[name]['velocity'],
                       error_code=joints[name].get('error_code')) for name in ARM}


class Evidence:
    def __init__(self, run_id):
        self.run_id, self.session_id, self.last_event = run_id, None, -1
        self.events, self.decisions = defaultdict(list), defaultdict(list)

    def bridge(self, row):
        if row.get('kind') != 'execution':
            return
        event = row['event']
        require(event.get('run_id') == self.run_id and event.get('session_id'), 'Foreign execution run')
        self.session_id = self.session_id or event['session_id']
        require(event['session_id'] == self.session_id and type(event.get('event_seq')) is int
                and event['event_seq'] > self.last_event, 'Reordered or foreign execution event')
        self.last_event = event['event_seq']
        raw = event.get('raw_feedback', {})
        compact = {key: event.get(key) for key in ('event_seq', 'decision_seq', 'execution_id', 'request_id',
            'behavior', 'status', 'fact', 'finish_request_id', 'finish_decision_seq', 'protection_reason')}
        compact.update(stop_confirmed=event.get('stop_evidence', {}).get('stop_confirmed'),
            reason=raw.get('reason'), metrics={key: raw.get(key) for key in (
                'forward_distance_m', 'lateral_distance_m', 'angle_rad', 'advance_direction_ok',
                'joint_return_rad', 'return_reference', 'joint_excursion_rad', 'command_seconds',
                'advance_target_reached', 'turn_target_reached')}, arm=arm_sample(raw))
        self.events[event['request_id']].append(compact)

    def brain(self, row):
        if row.get('kind') != 'raw_decision':
            return
        decision = row['decision']
        require(decision.get('run_id') == self.run_id and decision.get('session_id'), 'Foreign neural decision run')
        self.session_id = self.session_id or decision['session_id']
        require(decision['session_id'] == self.session_id, 'Foreign neural decision session')
        self.decisions[decision['request_id']].append({key: decision.get(key) for key in (
            'decision_seq', 'behavior', 'phase', 'intensity', 'checkpoint_id', 'policy_id', 'input_map_id')})

    def request(self, task, request_id):
        events = self.events[request_id]
        decisions = self.decisions[request_id]
        expected = EXPECTED[task]
        failures = []
        if any(e['status'] in ('FAILED', 'UNKNOWN', 'REJECTED', 'CANCELLED') or e['protection_reason'] for e in events):
            failures.append('execution_failed_or_cancelled')
        if any(e['behavior'] not in (expected, 'HOLD') for e in events):
            failures.append('unexpected_behavior')
        completed = [e for e in events if e['behavior'] == expected and e['status'] == 'COMPLETED'
                     and e['fact'] == 'status']
        if not completed:
            failures.append('expected_behavior_not_completed')
        if expected != 'HOLD' and len(completed) > 1:
            failures.append('repeated_motion_in_request')
        terminals = [e for e in events if e['behavior'] == expected
                     and e['status'] in ('COMPLETED', 'FAILED', 'UNKNOWN', 'REJECTED', 'CANCELLED')]
        terminal = completed[-1] if completed else terminals[-1] if terminals else None
        metrics = dict(terminal['metrics']) if terminal else {}
        if terminal:
            if terminal['stop_confirmed'] is not True:
                failures.append('physical_stop_unconfirmed')
            if not any(d['decision_seq'] == terminal['decision_seq'] and d['behavior'] == expected
                       and d['phase'] == ('FINISH' if expected == 'HOLD' else 'START') for d in decisions):
                failures.append('matching_raw_brain_decision_missing')
            if expected == 'ADVANCE':
                if not (finite(metrics.get('forward_distance_m')) and metrics['forward_distance_m'] >= 1.
                        and metrics.get('advance_direction_ok') is True):
                    failures.append('forward_distance_or_direction')
            elif expected in ('TURN_LEFT', 'TURN_RIGHT'):
                angle = metrics.get('angle_rad')
                target = math.pi / 2 * (1 if expected == 'TURN_LEFT' else -1)
                error = abs(angle - target) if finite(angle) else None
                metrics.update(target_angle_rad=target, absolute_error_rad=error,
                               tolerance_rad=TURN_TOLERANCE_RAD)
                if error is None or error > TURN_TOLERANCE_RAD:
                    failures.append('final_turn_error_exceeds_5deg')
            elif expected == 'WAVE_RIGHT':
                same = [e for e in events if e['execution_id'] == terminal['execution_id']]
                baseline = next((e['arm'] for e in same if e['status'] == 'ACCEPTED' and e['arm']), None)
                samples = [e['arm'] for e in same if e['arm']]
                excursion = max((abs(a[n]['position'] - baseline[n]['position'])
                                 for a in samples for n in ARM), default=None) if baseline else None
                stopped = terminal['arm']
                speed = max(abs(stopped[n]['velocity']) for n in ARM) if stopped else None
                metrics.update(measured_joint_excursion_rad=excursion, terminal_joint_max_speed_rad_s=speed)
                if not (excursion is not None and excursion >= .1 and finite(metrics.get('joint_return_rad'))
                        and metrics['joint_return_rad'] <= .15 and speed is not None and speed < .08
                        and all(a[n]['error_code'] == 0 for a in samples for n in ARM)):
                    failures.append('wave_joint_motion_or_return_unconfirmed')
        return dict(passed=not failures, failures=failures, metrics=metrics, events=events, brain_decisions=decisions)


class SequenceDriver:
    def __init__(self, tasks, run_path, request_timeout):
        self.run_path, self.request_timeout = Path(run_path), request_timeout
        self.results = [dict(index=i, task=t, expected_behavior=EXPECTED[t], status='not_run') for i, t in enumerate(tasks)]
        self.status, self.active, self.next_index = {}, None, 0
        self.seen_requests = set()
        self.evidence = Evidence(self.run_path.name)

    def status_event(self, value):
        require(value.get('schema') == 'malecns_lab.desktop_event.v1' and value.get('type') == 'status'
                and value.get('output') == str(self.run_path), 'Invalid runtime status identity')
        require(value.get('run_id') in (self.run_path.name, None), 'Foreign runtime status run')
        self.status = value
        if (self.active and value.get('rejected_task') == self.active['task']
                and (value.get('notice') == 'request_rejected' or value.get('phase') == 'request_rejected')):
            raise ValueError('Runtime rejected request: ' + str(value.get('reason')))
        if self.active and value.get('phase') == 'queued':
            request_id = value.get('request_id')
            require(value.get('task') == self.active['task'] and request_id and request_id not in self.seen_requests,
                    'Queued request attribution differs')
            self.seen_requests.add(request_id)
            self.active['request_id'] = request_id

    def advance(self, now, *, caught_up):
        if self.active:
            require(now - self.active['sent_monotonic'] < self.request_timeout, 'Single request timeout')
            request_id = self.active.get('request_id')
            events = self.evidence.events.get(request_id, ())
            require(not any(e['status'] in ('FAILED', 'UNKNOWN', 'REJECTED', 'CANCELLED') or e['protection_reason']
                            or e['behavior'] not in (self.active['expected_behavior'], 'HOLD') for e in events),
                    'Actual execution failed, cancelled or selected unexpected behavior')
            waiting = (request_id is not None and self.status.get('request_id') == request_id
                       and self.status.get('state') == 'running' and self.status.get('phase') == 'waiting_task'
                       and self.status.get('stop_confirmed') is True and self.status.get('action_busy') is False
                       and self.status.get('request_pending') is False and not self.status.get('active_request_id'))
            if waiting and caught_up:
                assessment = self.evidence.request(self.active['task'], request_id)
                self.active.update(assessment, status='passed' if assessment['passed'] else 'failed',
                                   elapsed_seconds=now - self.active['sent_monotonic'])
                require(assessment['passed'], 'Request failed: ' + ','.join(assessment['failures']))
                self.active = None
        if self.active is None and self.next_index < len(self.results):
            ready = (self.status.get('state') == 'running' and self.status.get('phase') == 'waiting_task'
                     and self.status.get('brain_controlled') is True and self.status.get('action_busy') is False
                     and self.status.get('request_pending') is False and not self.status.get('active_request_id'))
            if ready:
                self.active = self.results[self.next_index]
                self.next_index += 1
                self.active.update(status='running', sent_monotonic=now)
                return dict(command='task', task=self.active['task'])
        return None

    @property
    def complete(self):
        return self.next_index == len(self.results) and self.active is None

    def fail_active(self, reason, now):
        if self.active:
            if self.active.get('request_id'):
                self.active.update(self.evidence.request(self.active['task'], self.active['request_id']))
            self.active.update(status='failed', passed=False, failure_reason=reason,
                               elapsed_seconds=now - self.active['sent_monotonic'])


def runtime_command(python, run_path, session_profile, turn_profile='measured', wave_profile='baseline', sim_profile='official'):
    require(turn_profile in ('measured', 'precision', 'settled', 'quarter-turn', 'integral', 'compensated'), 'Unsupported turn profile')
    require(wave_profile in ('baseline', 'balance-sway'), 'Unsupported wave profile')
    require(wave_profile == 'baseline' or turn_profile in ('measured', 'quarter-turn', 'integral', 'compensated'), 'Wave balance-sway requires measured, quarter-turn, integral or compensated profile')
    require(turn_profile != 'quarter-turn' or wave_profile == 'balance-sway', 'Quarter-turn requires balance-sway wave profile')
    require(turn_profile != 'integral' or wave_profile == 'balance-sway', 'Integral requires balance-sway wave profile')
    require(turn_profile != 'compensated' or wave_profile == 'balance-sway', 'Compensated requires balance-sway wave profile')
    require(sim_profile in ('official', 'noslip1'), 'Unsupported simulation profile')
    command = [str(python), '-u', '-m', 'malecns_lab.runtime.x2.behavior', '--execute',
               '--sim-only', '--qualification', '--output', str(run_path)]
    # Qualification is the runtime's unchanged default; a named longer profile
    # is explicit and remains subject to the runtime's own independent limits.
    if session_profile != 'qualification':
        command += ['--session-profile', session_profile]
    if turn_profile != 'measured':
        command += ['--turn-profile', turn_profile]
    if wave_profile != 'baseline':
        command += ['--wave-profile', wave_profile]
    if sim_profile != 'official':
        command += ['--sim-profile', sim_profile]
    return command


def write_summary(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def execute(args, tasks):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_path = ROOT / 'data/x2_sim/agibot/data/var' / (
        'desktop-' + datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8])
    command = runtime_command(args.python, run_path, args.runtime_session_profile, args.turn_profile, args.wave_profile, args.sim_profile)
    driver = SequenceDriver(tasks, run_path, args.request_timeout_seconds)
    started = time.monotonic()
    work_end = started + args.max_wall_seconds - args.cleanup_timeout_seconds
    summary = dict(schema=SCHEMA, status='running', run_path=str(run_path), command=command,
        configuration=dict(sequence=tasks, cycles=args.cycles, runtime_session_profile=args.runtime_session_profile,
            turn_profile=args.turn_profile, wave_profile=args.wave_profile, sim_profile=args.sim_profile,
            max_wall_seconds=args.max_wall_seconds, cleanup_reserve_seconds=args.cleanup_timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds, turn_tolerance_rad=TURN_TOLERANCE_RAD,
            runtime_budget_override=False, resets_between_actions=False), requests=driver.results)
    summary_path = output / 'summary.json'
    write_summary(summary_path, summary)
    incoming, controls, reader_errors = queue.Queue(maxsize=1024), queue.Queue(maxsize=32), []
    interrupted = threading.Event()
    child, final_status, stop_sent, continued = None, None, False, False
    previous_handlers = {sig: signal.signal(sig, lambda *_: interrupted.set()) for sig in (signal.SIGINT, signal.SIGTERM)}

    def control_reader():
        for line in sys.stdin:
            try:
                controls.put_nowait(line.strip())
            except queue.Full:
                interrupted.set()
                return
        interrupted.set()

    def send(value):
        require(value.get('command') in ('task', 'continue', 'stop'), 'Unsupported control command')
        control_log.write(json.dumps(dict(sent_monotonic=time.monotonic(), **value)) + '\n')
        control_log.flush()
        child.stdin.write(json.dumps(value) + '\n')
        child.stdin.flush()

    def status_reader():
        try:
            while True:
                line = child.stdout.readline(MAX_LINE + 1)
                if not line:
                    return
                require(len(line) <= MAX_LINE and line.endswith('\n'), 'Oversized or incomplete runtime stdout')
                stdout_log.write(line)
                stdout_log.flush()
                incoming.put_nowait(json.loads(line))
        except Exception as error:
            reader_errors.append(repr(error))

    bridge, brain = JsonlTail(run_path / 'bridge_events.jsonl'), JsonlTail(run_path / 'behavior_events.jsonl')
    with (output / 'runtime_stdout.jsonl').open('x') as stdout_log, (output / 'runtime_stderr.log').open('x') as stderr_log, \
            (output / 'commands.jsonl').open('x') as control_log:
        try:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_log, text=True, bufsize=1)
            summary['runtime_pid'] = child.pid
            threading.Thread(target=status_reader, daemon=True).start()
            threading.Thread(target=control_reader, daemon=True).start()
            last_saved = 0.
            while True:
                require(not interrupted.is_set(), 'Operator stop or control stdin closed')
                require(time.monotonic() < work_end, 'Overall diagnostic work budget expired')
                require(not reader_errors, 'Runtime stdout reader failed: ' + str(reader_errors))
                while not incoming.empty():
                    value = incoming.get_nowait()
                    driver.status_event(value)
                    print(json.dumps(value, ensure_ascii=False), flush=True)
                    if value.get('state') in ('completed', 'cancelled', 'failed'):
                        final_status = value
                        raise ValueError('Runtime ended before diagnostic completion: ' + str(value.get('reason')))
                while not controls.empty():
                    control = controls.get_nowait()
                    require(control != 'stop', 'Operator stop')
                    require(control == 'continue' and not continued and driver.status.get('state') == 'awaiting_simulation',
                            'continue requires the initial awaiting_simulation state; only continue/stop are accepted')
                    send(dict(command='continue'))
                    continued = True
                for row in bridge.poll():
                    driver.evidence.bridge(row)
                for row in brain.poll():
                    driver.evidence.brain(row)
                command_value = driver.advance(time.monotonic(), caught_up=bridge.caught_up and brain.caught_up)
                if command_value:
                    send(command_value)
                summary.update(last_runtime_status=driver.status, elapsed_seconds=time.monotonic() - started)
                if command_value or driver.complete or time.monotonic() - last_saved >= 1.:
                    write_summary(summary_path, summary)
                    last_saved = time.monotonic()
                if driver.complete:
                    summary.update(status='passed', reason='all_requested_tasks_verified')
                    break
                require(child.poll() is None, 'Runtime process exited')
                time.sleep(.1)
        except BaseException as error:
            reason = type(error).__name__ + ':' + str(error)
            driver.fail_active(reason, time.monotonic())
            summary.update(status='failed', reason=reason)
        finally:
            if child is not None and child.poll() is None:
                try:
                    send(dict(command='stop'))
                    stop_sent = True
                except (OSError, ValueError):
                    try:
                        child.send_signal(signal.SIGINT)  # Existing safe-stop handler, owned runtime only.
                    except ProcessLookupError:
                        pass
                    summary['stop_signal_fallback'] = 'SIGINT'
                cleanup_end = min(time.monotonic() + args.cleanup_timeout_seconds, started + args.max_wall_seconds)
                while time.monotonic() < cleanup_end and child.poll() is None:
                    try:
                        value = incoming.get(timeout=.1)
                        if value.get('output') == str(run_path) and value.get('state') in ('completed', 'cancelled', 'failed'):
                            final_status = value
                    except queue.Empty:
                        pass
            # Read only the final small coordinator status, never synthesize cleanup success.
            status_path = run_path / 'desktop_session.json'
            if status_path.exists():
                try:
                    value = json.loads(status_path.read_text())
                    if value.get('output') == str(run_path):
                        final_status = value
                except (OSError, ValueError):
                    pass
            summary.update(stop_requested=stop_sent, final_runtime_status=final_status,
                runtime_alive=child is not None and child.poll() is None,
                resources_stopped=final_status.get('resources_stopped') if final_status else None,
                elapsed_seconds=time.monotonic() - started)
            if summary['resources_stopped'] is not True or summary['runtime_alive']:
                summary.update(status='failed', cleanup_unconfirmed=True)
            elif summary['status'] == 'passed' and (final_status.get('state') not in ('completed', 'cancelled')
                    or final_status.get('stop_confirmed') is not True):
                summary.update(status='failed', reason='Final runtime stop not confirmed')
            passed_pairs = [(a['task'], b['task']) for a, b in zip(driver.results, driver.results[1:])
                            if a['status'] == b['status'] == 'passed' and a['task'] in TASKS and b['task'] in TASKS]
            summary['verified_directed_pairs'] = sorted(set(passed_pairs))
            summary['verified_directed_pair_count'] = len(set(passed_pairs))
            write_summary(summary_path, summary)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    print(json.dumps(dict(summary=str(summary_path), status=summary['status'],
                         reason=summary.get('reason'), resources_stopped=summary['resources_stopped']), ensure_ascii=False))
    return 0 if summary['status'] == 'passed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--python', type=Path, default=ROOT / '.venv-train/bin/python')
    parser.add_argument('--sequence', help='Comma-separated runtime task enums; overrides default coverage sequence')
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--max-wall-seconds', type=float, default=480.)
    parser.add_argument('--request-timeout-seconds', type=float, default=120.)
    parser.add_argument('--cleanup-timeout-seconds', type=float, default=45.)
    parser.add_argument('--runtime-session-profile', choices=('qualification', 'continuous'), default='qualification')
    parser.add_argument('--turn-profile', choices=('measured', 'precision', 'settled', 'quarter-turn', 'integral', 'compensated'), default='measured')
    parser.add_argument('--wave-profile', choices=('baseline', 'balance-sway'), default='baseline')
    parser.add_argument('--sim-profile', choices=('official', 'noslip1'), default='official')
    args = parser.parse_args(argv)
    require(args.wave_profile == 'baseline' or args.turn_profile in ('measured', 'quarter-turn', 'integral', 'compensated'),
            'Wave balance-sway requires measured, quarter-turn, integral or compensated profile')
    require(args.turn_profile != 'quarter-turn' or args.wave_profile == 'balance-sway',
            'Quarter-turn requires balance-sway wave profile')
    require(args.turn_profile != 'integral' or args.wave_profile == 'balance-sway',
            'Integral requires balance-sway wave profile')
    require(args.turn_profile != 'compensated' or args.wave_profile == 'balance-sway',
            'Compensated requires balance-sway wave profile')
    tasks = planned_tasks(args.sequence, args.cycles)
    require(all(finite(v) and v > 0 for v in (args.max_wall_seconds, args.request_timeout_seconds, args.cleanup_timeout_seconds))
            and args.cleanup_timeout_seconds < args.max_wall_seconds, 'Invalid diagnostic time budgets')
    if not args.execute:
        print(json.dumps(dict(schema=SCHEMA, mode='preview', tasks=tasks,
            planned_directed_pairs=sorted(set(zip(tasks, tasks[1:]))),
            runtime_session_profile=args.runtime_session_profile, turn_profile=args.turn_profile, wave_profile=args.wave_profile,
            sim_profile=args.sim_profile,
            max_wall_seconds=args.max_wall_seconds,
            runtime_budget_override=False, brain_loaded=False, simulator_started=False), ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error('--execute requires a new --output directory')
    return execute(args, tasks)


if __name__ == '__main__':
    raise SystemExit(main())

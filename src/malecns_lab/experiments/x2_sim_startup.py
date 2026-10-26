"""Startup-only shortcuts for the owned official MuJoCo window, verified by odometry.

No simulator/ROS imports or side effects at import. This is UI automation, not an
official reset API. Never called once the behavior bridge has started.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import time


def require(value, message):
    if not value:
        raise RuntimeError(message)


def simulator_process_identity(executable, cwd, argv):
    """Verify the executable or Rosetta's actual program, never argv substrings."""
    official = Path('/workspace/sim_mujoco/bin/aima-sim-app')
    rosetta = Path('/mnt/rosetta/rosetta')
    executable, cwd = Path(executable), Path(cwd)
    require(executable.is_absolute() and cwd.is_absolute(), 'Invalid simulator process paths')
    program = executable
    prefix = []
    if executable == rosetta:
        require(bool(argv) and bool(argv[0]), 'Rosetta simulator has no program argument')
        first = (cwd/argv[0]).resolve()
        index = 1 if first == rosetta else 0
        require(len(argv) > index and bool(argv[index]), 'Rosetta simulator has no program argument')
        program = (cwd/argv[index]).resolve()
        prefix = list(argv[:index+1])
    require(program == official and executable in (official, rosetta),
            f'Unexpected simulator executable: exe={executable}, program={program}')
    return dict(executable=str(executable), program=str(program), cwd=str(cwd), argv_prefix=prefix,
                translated=executable == rosetta)


class Cancellation:
    def __init__(self, output, run_id, session_id):
        self.output, self.identity = output, (run_id, session_id)

    def __call__(self):
        for name in ('emergency.json', 'sim_startup_cancel.json'):
            path = self.output/name
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text())
                require((value.get('run_id'), value.get('session_id')) == self.identity,
                        'Foreign simulation startup cancellation')
            except (OSError, ValueError, AttributeError) as error:
                raise RuntimeError('Invalid simulation startup cancellation') from error
            raise InterruptedError('Simulation startup cancelled')


class SimulatorWindow:
    """Address one verified simulator window on the private container display."""
    def __init__(self, check_cancelled=lambda: None):
        self.check_cancelled = check_cancelled
        self.x = C.CDLL('libX11.so.6')
        self.xt = C.CDLL('libXtst.so.6')
        signatures = {
            'XOpenDisplay': (C.c_void_p, [C.c_char_p]),
            'XDefaultRootWindow': (C.c_ulong, [C.c_void_p]),
            'XQueryTree': (C.c_int, [C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_ulong), C.POINTER(C.POINTER(C.c_ulong)), C.POINTER(C.c_uint)]),
            'XFetchName': (C.c_int, [C.c_void_p, C.c_ulong, C.POINTER(C.c_void_p)]),
            'XInternAtom': (C.c_ulong, [C.c_void_p, C.c_char_p, C.c_int]),
            'XGetWindowProperty': (C.c_int, [C.c_void_p, C.c_ulong, C.c_ulong, C.c_long, C.c_long, C.c_int, C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_int), C.POINTER(C.c_ulong), C.POINTER(C.c_ulong), C.POINTER(C.c_void_p)]),
            'XSetInputFocus': (C.c_int, [C.c_void_p, C.c_ulong, C.c_int, C.c_ulong]),
            'XGetInputFocus': (C.c_int, [C.c_void_p, C.POINTER(C.c_ulong), C.POINTER(C.c_int)]),
            'XKeysymToKeycode': (C.c_uint, [C.c_void_p, C.c_ulong]),
            'XSync': (C.c_int, [C.c_void_p, C.c_int]),
            'XFree': (C.c_int, [C.c_void_p]),
            'XCloseDisplay': (C.c_int, [C.c_void_p]),
        }
        for name, (result, args) in signatures.items():
            fn = getattr(self.x, name)
            fn.restype, fn.argtypes = result, args
        self.xt.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
        self.xt.XTestFakeKeyEvent.restype = C.c_int
        require(os.environ.get('DISPLAY') == ':99', 'Unexpected simulation display')
        self.display = self.x.XOpenDisplay(b':99')
        require(self.display, 'Simulation display unavailable')
        self.window, self.pid = self.find()

    def find(self):
        pending = [self.x.XDefaultRootWindow(self.display)]
        found = []
        visited = 0
        while pending:
            window = pending.pop()
            visited += 1
            require(visited < 4096, 'Unexpected display window tree')
            name = C.c_void_p()
            self.x.XFetchName(self.display, window, C.byref(name))
            title = C.string_at(name).decode(errors='replace') if name.value else ''
            if name.value:
                self.x.XFree(name)
            if title.startswith('MuJoCo'):
                pid = self.window_pid(window)
                # A window manager decoration can repeat the client's title
                # without owning a process. Only PID-bearing clients qualify.
                if pid is not None:
                    process = Path(f'/proc/{pid}')
                    # binfmt's Rosetta executable lives in the VM mount namespace,
                    # not necessarily this container's; trust the kernel's link
                    # target, then validate the actual program and cwd below.
                    executable, cwd = Path(os.readlink(process/'exe')), (process/'cwd').resolve(strict=True)
                    argv = [os.fsdecode(value) for value in (process/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
                    try:
                        identity = simulator_process_identity(executable, cwd, argv)
                    except RuntimeError as error:
                        raise RuntimeError(f'MuJoCo window has unexpected process: pid={pid}; {error}') from error
                    self.process_identity = identity
                    found.append((window, pid))
            root, parent, count = C.c_ulong(), C.c_ulong(), C.c_uint()
            children = C.POINTER(C.c_ulong)()
            if self.x.XQueryTree(self.display, window, C.byref(root), C.byref(parent), C.byref(children), C.byref(count)):
                pending.extend(children[i] for i in range(count.value))
                if children:
                    self.x.XFree(children)
        require(len(found) == 1, 'Expected exactly one official MuJoCo window')
        return found[0]

    def window_pid(self, window):
        atom = self.x.XInternAtom(self.display, b'_NET_WM_PID', 1)
        if not atom:
            return None
        actual, count, remaining, fmt, data = C.c_ulong(), C.c_ulong(), C.c_ulong(), C.c_int(), C.c_void_p()
        result = self.x.XGetWindowProperty(self.display, window, atom, 0, 1, 0, 6,
            C.byref(actual), C.byref(fmt), C.byref(count), C.byref(remaining), C.byref(data))
        try:
            return C.cast(data, C.POINTER(C.c_ulong))[0] if result == 0 and actual.value == 6 and fmt.value == 32 and count.value == 1 and data.value else None
        finally:
            if data.value:
                self.x.XFree(data)

    def press(self, name):
        self.check_cancelled()
        require(name in ('space', 'BackSpace'), 'Unsupported startup shortcut')
        require(self.find() == (self.window, self.pid), 'Simulation window changed')
        self.x.XSetInputFocus(self.display, self.window, 1, 0)
        self.x.XSync(self.display, 0)
        focus, revert = C.c_ulong(), C.c_int()
        self.x.XGetInputFocus(self.display, C.byref(focus), C.byref(revert))
        require(focus.value == self.window, 'Simulation window has no keyboard focus')
        key = self.x.XKeysymToKeycode(self.display, 0x20 if name == 'space' else 0xff08)
        require(key, 'Simulation shortcut unavailable')
        self.check_cancelled()
        try:
            require(self.xt.XTestFakeKeyEvent(self.display, key, 1, 0), 'Simulation key press failed')
        finally:
            self.xt.XTestFakeKeyEvent(self.display, key, 0, 0)
            self.x.XSync(self.display, 0)

    def close(self):
        self.x.XCloseDisplay(self.display)


class Odometry:
    def __init__(self, check_cancelled=lambda: None):
        import rclpy
        from nav_msgs.msg import Odometry as Message
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        self.ros = rclpy
        self.check_cancelled = check_cancelled
        rclpy.init()
        self.node = rclpy.create_node('malecns_sim_startup')
        self.samples = []
        self.awaiting_first_feedback = True
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.node.create_subscription(Message, '/aima/hal/odom/state', self.receive, qos)

    def receive(self, msg):
        p, q, v, w = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear, msg.twist.twist.angular
        value = tuple(float(x) for x in (p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z))
        require(all(math.isfinite(x) for x in value), 'Invalid startup odometry')
        self.samples.append(value)

    def observe(self):
        # DDS discovery may outlast an observation window on first connection.
        # Wait only once, within the outer 15-second helper budget, then discard
        # discovery samples and assess a complete fresh observation interval.
        if getattr(self, 'awaiting_first_feedback', False):
            end = time.monotonic() + 5.
            while not self.samples and time.monotonic() < end:
                self.check_cancelled()
                self.ros.spin_once(self.node, timeout_sec=.03)
            self.check_cancelled()
            require(bool(self.samples), 'No fresh simulation odometry after initial discovery')
            self.awaiting_first_feedback = False
        # Drain queued pre-shortcut feedback, then require fresh messages across
        # a full observation interval; a stale single sample is never readiness.
        end = time.monotonic() + .3
        while time.monotonic() < end:
            self.check_cancelled()
            self.ros.spin_once(self.node, timeout_sec=.03)
        self.samples.clear()
        end = time.monotonic() + .6
        while time.monotonic() < end:
            self.check_cancelled()
            self.ros.spin_once(self.node, timeout_sec=.03)
        self.check_cancelled()
        require(len(self.samples) >= 5, 'No fresh simulation odometry')
        return dict(samples=len(self.samples), frozen=all(s == self.samples[0] for s in self.samples),
                    first=self.samples[0], last=self.samples[-1])

    def close(self):
        self.node.destroy_node()
        self.ros.shutdown()


def reset_upright(sample):
    x, y, z, qx, qy, qz, qw, *twist = sample
    # Official x2.xml: pelvis pos="0 0 0.68", euler="0 0 0".
    return abs(x) < .01 and abs(y) < .01 and abs(z - .68) < .02 and max(abs(qx), abs(qy), abs(qz)) < .01 and abs(abs(qw) - 1) < .01 and max(map(abs, twist)) < .01


def prepare(window, feedback, record, check_cancelled=lambda: None):
    check_cancelled()
    initial = feedback.observe()
    record('initial', initial)
    if not initial['frozen']:
        check_cancelled()
        window.press('space')
    paused = feedback.observe()
    record('paused', paused)
    require(paused['frozen'], 'Simulation did not pause')
    check_cancelled()
    window.press('BackSpace')
    reset = feedback.observe()
    record('reset', reset)
    require(reset['frozen'] and reset_upright(reset['last']), 'Simulation did not reset upright while paused')
    check_cancelled()
    window.press('space')
    resumed = feedback.observe()
    record('resumed', resumed)
    require(not resumed['frozen'], 'Simulation did not resume')
    check_cancelled()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--sim-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.execute and args.sim_only, 'Startup requires --execute --sim-only')
    require(os.environ.get('ROS_DOMAIN_ID') == '47' and os.environ.get('ROS_LOCALHOST_ONLY') == '1', 'Unexpected simulation ROS domain')
    output = args.output.resolve(strict=True)
    require(output.parent == Path('/agibot/data/var'), 'Unexpected startup output')
    registration = json.loads((output/'bridge_registration.json').read_text())
    require(registration.get('simulation_only') is True and registration['run_id'] == output.name, 'Foreign simulation registration')
    require(not (output/'bridge_ready.json').exists() and not (output/'sim_startup.json').exists(), 'Startup already performed')
    receipt = dict(schema='malecns_lab.sim_startup.v1', run_id=registration['run_id'], session_id=registration['session_id'],
                   status='preparing', stages=[])
    check_cancelled = Cancellation(output, registration['run_id'], registration['session_id'])
    def save():
        temporary = output/'sim_startup.json.tmp'
        temporary.write_text(json.dumps(receipt, allow_nan=False)+'\n')
        temporary.replace(output/'sim_startup.json')
    def record(stage, observation):
        receipt['stages'].append(dict(stage=stage, monotonic=time.monotonic(), **observation))
        save()
    window = feedback = None
    try:
        save()
        check_cancelled()
        window = SimulatorWindow(check_cancelled)
        receipt.update(window_id=window.window, simulator_pid=window.pid, process_identity=window.process_identity)
        feedback = Odometry(check_cancelled)
        prepare(window, feedback, record, check_cancelled)
        receipt['status'] = 'ready'
    except BaseException as error:
        receipt.update(status='failed', reason=str(error))
        raise
    finally:
        save()
        if feedback:
            feedback.close()
        if window:
            window.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

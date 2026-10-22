"""Explicit ROS bridge for the bounded virtual-marker neural loop; Python 3.10."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from .x2 import _EvidenceLog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "preview", "ros_connected": False, "brain_loaded": False,
                          "rounds": 2, "pulse_seconds": 1., "brain_timeout": 30., "wall_seconds": 120,
                          "brain_timeout_seconds": 30., "observation_timeout_seconds": 45., "pose_timeout_seconds": 40.,
                          "requires": "host ready, domain 47 localhost 1, isolated simulation"}))
        return 0
    if not args.sim_only or os.environ.get("ROS_DOMAIN_ID") != "47" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        parser.error("Execute requires --sim-only, ROS_DOMAIN_ID=47 and ROS_LOCALHOST_ONLY=1")
    if args.output is None:
        parser.error("Execute requires host-prepared shared --output")
    try:
        return execute(args.output)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


def execute(directory):
    from ..control.x2 import X2MotionAdapter, SOURCE
    from ..control.x2_protocol import (
        LoopController, atomic, read, verify, ROS_TIMEOUT, MAX_ROUNDS, BRAIN_TIMEOUT, OBSERVATION_TIMEOUT, POSE_TIMEOUT,
    )
    directory = Path(directory).resolve()
    registration = read(directory / "registration.json")
    run_id = registration["run_id"]
    ready = verify(read(directory / "ready.json"), "ready", run_id, 0)
    if ready["registration_id"] != registration["registration_id"]:
        raise ValueError("Ready registration differs")
    if not (registration.get("rounds") == ready.get("rounds") == MAX_ROUNDS
            and registration.get("brain_timeout_seconds") == BRAIN_TIMEOUT
            and registration.get("observation_timeout_seconds") == OBSERVATION_TIMEOUT
            and registration.get("pose_timeout_seconds") == POSE_TIMEOUT):
        raise ValueError("Registered loop budgets differ")
    with (directory / "bridge.claim").open("x") as stream:
        stream.write(str(os.getpid()))
    # Explicit gates and a ready complete brain precede ROS imports and IO.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from aimdk_msgs.msg import McCommonState, McLocomotionVelocity, McInputAction
    from aimdk_msgs.srv import SetMcInputSource
    from nav_msgs.msg import Odometry

    evidence = _EvidenceLog((directory / "bridge_events.jsonl").open("x"))
    started, interrupted, old_signals = time.monotonic(), [], {}
    node = core = controller = None
    failure = None
    summary = {"success": False, "run_id": run_id, "reason": "bridge_not_initialized"}

    def write(kind, **fields):
        nonlocal failure
        if not evidence.record({"kind": kind, "monotonic": time.monotonic(), "wall_time": time.time(),
                                "phase": controller.phase if controller else "registration", **fields}):
            failure = failure or evidence.failure
            if controller is not None:
                controller.fail("evidence_write_error")
            elif core is not None:
                core.request_stop("evidence_write_error", failure=True)

    def raw(message):
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return str(value) if isinstance(value, float) and not math.isfinite(value) else value
        return clean(message_to_ordereddict(message))

    def publish(command):
        message = McLocomotionVelocity()
        message.header.stamp = node.get_clock().now().to_msg()
        message.source = SOURCE
        message.forward_velocity, message.lateral_velocity, message.angular_velocity = command
        publisher.publish(message)

    def on_odom(message):
        received = time.monotonic()
        try:
            p, q = message.pose.pose.position, message.pose.pose.orientation
            v, w = message.twist.twist.linear, message.twist.twist.angular
            accepted = core.update_odom((p.x,p.y,p.z), (q.x,q.y,q.z,q.w), (v.x,v.y,v.z), (w.x,w.y,w.z), received=received)
            if controller is not None:
                controller.feedback()
            write("odom", received=received, accepted=accepted, message=raw(message))
        except Exception as error:
            core.request_stop("odom_callback_error", failure=True)
            write("callback_error", topic="odom", error=repr(error))

    def on_mc(message):
        received = time.monotonic()
        try:
            prior = core.state
            accepted = core.update_mc(message.input_source.name, message.action_info.action_desc,
                                      message.body_status.value, message.action_info.status.value, received=received)
            if controller is not None:
                controller.feedback()
            write("mc", received=received, accepted=accepted, message=raw(message))
            if prior == "CLAIMING" and core.state == "FORWARD":
                write("ownership_ack", received=core.ownership_confirmed_at, mc_sequence=core.mc.sequence)
        except Exception as error:
            core.request_stop("mc_callback_error", failure=True)
            write("callback_error", topic="mc", error=repr(error))

    def step(schedule=True):
        began = time.monotonic()
        callback_deadline = began + .008
        for _ in range(64):
            remaining = callback_deadline-time.monotonic()
            if remaining <= 0:
                break
            rclpy.spin_once(node, timeout_sec=min(.001, remaining))
        if interrupted and controller is not None:
            controller.fail("interrupted:" + str(interrupted[0]))
        if schedule and controller is not None:
            controller.update()
        write("tick", **core.tick())
        time.sleep(max(0., .02-(time.monotonic()-began)))

    def register():
        client = node.create_client(SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource")
        deadline = min(started+12., started+ROS_TIMEOUT)
        while not client.service_is_ready() and time.monotonic() < deadline and not interrupted:
            step(False)
        for action in (McInputAction.INPUTACTION_ADD, McInputAction.INPUTACTION_ENABLE):
            if time.monotonic() >= deadline or interrupted or evidence.failure or core.failures:
                return False
            request = SetMcInputSource.Request()
            request.request.header.stamp = node.get_clock().now().to_msg()
            request.action.value = action
            request.input_source.name = SOURCE
            request.input_source.priority = 40
            request.input_source.timeout = 1000
            future = client.call_async(request)
            call_deadline = min(deadline, time.monotonic()+2.)
            while not future.done() and time.monotonic() < call_deadline and not interrupted:
                step(False)
            response = future.result() if future.done() else None
            write("registration", action=action, source=SOURCE, response=raw(response) if response else None)
            if response is not None and response.response.header.code == 0:
                return True
            if response is None:
                return False
        return False

    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, lambda signum, _frame: interrupted.append(signum))
        node = Node("malecns_x2_neural_loop")
        publisher = node.create_publisher(McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10)
        core = X2MotionAdapter(publish, allow_empty_source_claim=True)
        core.arm()
        controller = LoopController(core, directory, run_id, execution_id=ready["execution_id"], record=write)
        controller.started, controller.deadline = started, started+ROS_TIMEOUT
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(Odometry, "/aima/hal/odom/state", on_odom, qos)
        node.create_subscription(McCommonState, "/aima/mc/common/state", on_mc, qos)
        write("configuration", run_id=run_id, source=SOURCE, rounds=MAX_ROUNDS, pulse_seconds=1.,
              brain_timeout_seconds=BRAIN_TIMEOUT, observation_timeout_seconds=OBSERVATION_TIMEOUT,
              pose_timeout_seconds=POSE_TIMEOUT,
              sensor="live_odometry_virtual_marker", ros_domain_id=47, ros_localhost_only=1)
        if not register():
            controller.fail("registration_failed")
        else:
            core.confirm_source_registration()
        while not controller.done:
            step()
        # A final tick records an actual zero publication before node cleanup.
        if core.state in ("FORWARD", "CLAIMING"):
            controller.fail("active_intent_at_exit")
        write("tick", **core.tick())
        summary = controller.summary()
    except BaseException as error:
        failure = "bridge_exception:" + repr(error)
        write("exception", error=repr(error))
        if controller is not None:
            controller.fail(failure)
        elif core is not None:
            core.request_stop(failure, failure=True)
    finally:
        if failure and core is not None:
            end = min(started+ROS_TIMEOUT, time.monotonic()+12.)
            while time.monotonic() < end:
                try:
                    step()
                except Exception as error:
                    core.request_stop("cleanup_executor_error", failure=True)
                    write("tick", **core.tick())
                    time.sleep(.02)
                if core.state == "STOPPED":
                    break
        if controller is not None:
            summary = controller.summary()
        for operation in ((lambda: node.destroy_node()) if node else None,
                          lambda: rclpy.shutdown() if rclpy.ok() else None):
            if operation:
                try:
                    operation()
                except Exception as error:
                    failure = failure or "ros_cleanup_error:" + repr(error)
        for signum, old in old_signals.items():
            signal.signal(signum, old)
        evidence.close()
        failure = failure or evidence.failure
        if failure:
            summary.update(success=False, reason=failure)
        summary["cleanup_completed"] = failure is None
        summary["elapsed_seconds"] = time.monotonic()-started
        if summary["elapsed_seconds"] > ROS_TIMEOUT:
            summary.update(success=False, reason="ros_total_timeout")
        atomic(directory / "bridge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

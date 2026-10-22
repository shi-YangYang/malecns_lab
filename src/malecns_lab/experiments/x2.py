"""Explicitly opt-in X2 simulation experiment; default invocation only previews."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time



class _EvidenceLog:
    """A failed evidence sink must not interrupt zero-command cleanup."""

    def __init__(self, stream):
        self.stream = stream
        self.failure = None

    def fail(self, exc):
        if self.failure is None:
            self.failure = "evidence_write_error:" + repr(exc)
            try:
                print(self.failure, file=sys.stderr, flush=True)
            except Exception:
                pass

    def record(self, event):
        if self.failure is not None:
            return False
        try:
            self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
            return True
        except Exception as exc:
            self.fail(exc)
            return False

    def summary(self, path, value):
        try:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            return True
        except Exception as exc:
            self.fail(exc)
            return False

    def close(self):
        try:
            self.stream.close()
        except Exception as exc:
            self.fail(exc)


def _bounded(value: str, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= maximum:
        raise argparse.ArgumentTypeError(f"must be > 0 and <= {maximum}")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true", help="enable ROS and a bounded motion trial")
    result.add_argument("--sim-only", action="store_true", help="confirm isolated official simulation")
    result.add_argument("--forward-seconds", type=lambda v: _bounded(v, 5), default=2.0)
    result.add_argument("--startup-timeout", type=lambda v: _bounded(v, 30), default=12.0)
    result.add_argument("--stop-timeout", type=lambda v: _bounded(v, 20), default=12.0)
    result.add_argument("--output", type=Path, help="new evidence directory (required for execute)")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "preview", "ros_connected": False, "source": "malecns.x2_mvp",
                          "mc_target": [0.2, 0, 0], "forward_seconds": args.forward_seconds,
                          "stop_timeout": args.stop_timeout}, ensure_ascii=False))
        return 0
    if not args.sim_only or os.environ.get("ROS_DOMAIN_ID") != "47" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise SystemExit("execute requires --sim-only, ROS_DOMAIN_ID=47 and ROS_LOCALHOST_ONLY=1")
    if args.output is None:
        raise SystemExit("execute requires --output for motion evidence")
    return execute(args)


def execute(args) -> int:
    # These imports happen only after explicit simulation authorization checks.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from aimdk_msgs.msg import McCommonState, McLocomotionVelocity, McInputAction
    from aimdk_msgs.srv import SetMcInputSource
    from nav_msgs.msg import Odometry
    from malecns_lab.control.x2 import X2MotionAdapter, SOURCE, STARTUP_TRANSITION_TIMEOUT

    args.output.mkdir(parents=True, exist_ok=False)
    evidence = _EvidenceLog((args.output / "events.jsonl").open("w", encoding="utf-8"))
    started = time.monotonic()
    interrupted = []
    old_signals = {}
    node = None
    core = None
    failure = None
    forward_started = forward_accepted = False
    forward_at = forward_deadline = stop_at = None
    first_position = final_position = None
    phase = "startup"
    last_reported = None

    def write(kind, **fields):
        nonlocal failure
        if not evidence.record({"kind": kind, "monotonic": time.monotonic(),
                                "wall_time": time.time(), "phase": phase, **fields}):
            failure = failure or evidence.failure
            if core is not None and phase != "finished" and "evidence_write_error" not in core.failures:
                core.request_stop("evidence_write_error", failure=True)

    def raw_message(message):
        # JSON must remain readable even when invalid numeric feedback is rejected.
        def clean(value):
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            return value
        return clean(message_to_ordereddict(message))

    def on_signal(signum, _frame):
        interrupted.append(signum)

    def publish(command):
        nonlocal forward_started
        message = McLocomotionVelocity()
        message.header.stamp = node.get_clock().now().to_msg()
        message.source = SOURCE
        message.forward_velocity, message.lateral_velocity, message.angular_velocity = command
        publisher.publish(message)
        if command[0] != 0.0:
            forward_started = True

    def on_odom(message):
        nonlocal final_position
        received = time.monotonic()
        try:
            p, q = message.pose.pose.position, message.pose.pose.orientation
            linear, angular = message.twist.twist.linear, message.twist.twist.angular
            accepted = core.update_odom((p.x,p.y,p.z), (q.x,q.y,q.z,q.w),
                                        (linear.x,linear.y,linear.z), (angular.x,angular.y,angular.z),
                                        received=received)
            if accepted:
                final_position = (p.x, p.y, p.z)
            write("odom", received=received, accepted=accepted, message=raw_message(message))
        except Exception as exc:
            core.request_stop("odom_callback_error", failure=True)
            write("callback_error", topic="odom", error=repr(exc))

    def on_mc(message):
        received = time.monotonic()
        try:
            previous_state = core.state
            accepted = core.update_mc(message.input_source.name, message.action_info.action_desc,
                                      message.body_status.value, message.action_info.status.value,
                                      received=received)
            write("mc", received=received, accepted=accepted, previous_state=previous_state,
                  state=core.state, message=raw_message(message))
            if previous_state == "CLAIMING" and core.state == "FORWARD":
                write("ownership_ack", received=core.ownership_confirmed_at,
                      source=SOURCE, mc_sequence=core.mc.sequence)
        except Exception as exc:
            core.request_stop("mc_callback_error", failure=True)
            write("callback_error", topic="mc", error=repr(exc))

    def step():
        nonlocal last_reported
        cycle_start = time.monotonic()
        callback_deadline = cycle_start + 0.008
        # Drain multiple ready messages within a bounded part of each 20 ms cycle.
        # depth=1 below also discards backlog instead of relabelling old samples fresh.
        for _ in range(64):
            remaining = callback_deadline - time.monotonic()
            if remaining <= 0:
                break
            rclpy.spin_once(node, timeout_sec=min(0.001, remaining))
        if phase == "forward" and time.monotonic() >= forward_deadline:
            core.request_stop("planned_stop")
        result = core.tick()
        write("tick", **result)
        status = (phase, result["state"], result["reason"])
        if status != last_reported:
            print(" / ".join(status), flush=True)
            last_reported = status
        time.sleep(max(0.0, 0.02 - (time.monotonic()-cycle_start)))

    def register(deadline):
        client = node.create_client(SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource")
        while time.monotonic() < deadline and not interrupted:
            if client.wait_for_service(timeout_sec=min(0.2, max(0.0, deadline-time.monotonic()))):
                break
        else:
            return False
        for action in (McInputAction.INPUTACTION_ADD, McInputAction.INPUTACTION_ENABLE):
            if time.monotonic() >= deadline or interrupted:
                return False
            request = SetMcInputSource.Request()
            request.request.header.stamp = node.get_clock().now().to_msg()
            request.action.value = action
            request.input_source.name = SOURCE
            request.input_source.priority = 40
            request.input_source.timeout = 1000
            future = client.call_async(request)
            call_deadline = min(deadline, time.monotonic()+2.0)
            while not future.done() and time.monotonic() < call_deadline and not interrupted:
                rclpy.spin_once(node, timeout_sec=0.02)
            response = future.result() if future.done() else None
            write("registration", action=action, source=SOURCE,
                  response=raw_message(response) if response is not None else None)
            if response is not None and response.response.header.code == 0:
                return True
            if response is None:
                # Unknown ADD outcome is not permission to force another source.
                return False
        return False

    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, on_signal)
        node = Node("malecns_x2_motion_mvp")
        publisher = node.create_publisher(McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10)
        core = X2MotionAdapter(publish, allow_empty_source_claim=True)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(Odometry, "/aima/hal/odom/state", on_odom, qos)
        node.create_subscription(McCommonState, "/aima/mc/common/state", on_mc, qos)
        write("configuration", source=SOURCE, priority=40, source_timeout_ms=1000,
              forward_seconds=args.forward_seconds, mc_target=[0.2, 0.0, 0.0],
              startup_timeout=args.startup_timeout, stop_timeout=args.stop_timeout,
              empty_source_claim_timeout=0.5,
              startup_transition_timeout=STARTUP_TRANSITION_TIMEOUT,
              odom_max_age=core.odom_max_age, mc_max_age=core.mc_max_age,
              ros_domain_id=os.environ["ROS_DOMAIN_ID"], ros_localhost_only=os.environ["ROS_LOCALHOST_ONLY"])
        deadline = started + args.startup_timeout
        if not register(deadline):
            failure = "registration_failed"
        else:
            core.confirm_source_registration()
            core.arm()
            while time.monotonic() < deadline and not interrupted:
                step()
                if core.failures:
                    failure = core.failures[-1]
                    break
                if core.ready_reason(allow_empty_source=True) is None:
                    break
            if failure is None and not interrupted:
                if core.ready_reason(allow_empty_source=True) is not None:
                    failure = "startup_feedback_timeout:" + core.ready_reason(allow_empty_source=True)
                else:
                    first_position = core.odom.position
                    forward_accepted = core.request_forward(args.forward_seconds)
                    if not forward_accepted:
                        failure = "forward_rejected:" + core.reason
                    else:
                        phase = "forward"
                        forward_deadline = core.intent_deadline
                        forward_at = forward_deadline - args.forward_seconds
                        write("intent", intent="FORWARD", ttl=args.forward_seconds,
                              state=core.state, deadline=forward_deadline)
                        while time.monotonic() < forward_deadline and not interrupted:
                            step()
                            if core.state not in ("CLAIMING", "FORWARD") or core.failures:
                                if core.reason != "planned_stop" or core.failures:
                                    failure = core.reason
                                break
    except Exception as exc:
        failure = "runner_exception:" + repr(exc)
        write("exception", error=repr(exc))
    finally:
        phase = "stopping"
        if interrupted:
            failure = "interrupted:" + str(interrupted[0])
        if core is not None and core.armed:
            core.request_stop(failure or "planned_stop", failure=failure is not None)
            stop_at = time.monotonic()
            write("intent", intent="STOP", reason=core.reason)
            deadline = stop_at + args.stop_timeout
            while time.monotonic() < deadline:
                try:
                    step()
                except Exception as exc:
                    failure = failure or "stop_loop_exception:" + repr(exc)
                    core.request_stop("stop_loop_exception", failure=True)
                    # A broken ROS executor still gets bounded zero publish attempts.
                    write("tick", **core.tick())
                    time.sleep(0.02)
                if core.state == "STOPPED":
                    break
            if core.state != "STOPPED":
                failure = failure or "stop_unconfirmed_timeout"
        confirmed = core is not None and core.state == "STOPPED"
        failures = [] if core is None else core.failures
        failure = failure or (failures[0] if failures else None)
        success = forward_started and confirmed and failure is None
        summary = {"success": success, "forward_accepted": forward_accepted,
                   "forward_started": forward_started,
                   "stop_confirmed": confirmed, "failure_reason": failure,
                   "failures": failures, "source": SOURCE,
                   "ownership_confirmed_monotonic": None if core is None else core.ownership_confirmed_at,
                   "forward_monotonic": forward_at, "stop_monotonic": stop_at,
                   "finished_monotonic": time.monotonic(),
                   "elapsed_seconds": time.monotonic()-started,
                   "first_position": first_position, "final_position": final_position,
                   "horizontal_net_displacement": None if first_position is None or final_position is None
                   else math.dist(first_position[:2], final_position[:2]),
                   "note": "MC target is not measured speed; full trajectory and stop evidence are in events.jsonl."}
        phase = "finished"
        write("summary", **summary)
        if evidence.failure:
            failure = failure or evidence.failure
            success = False
            summary.update(success=False, failure_reason=failure)
        # Every cleanup is attempted even when summary or an earlier cleanup fails.
        if not evidence.summary(args.output / "summary.json", summary):
            failure = failure or evidence.failure
            success = False
            summary.update(success=False, failure_reason=failure)
        cleanup = []
        if node is not None:
            cleanup.append(node.destroy_node)
        cleanup.append(lambda: rclpy.shutdown() if rclpy.ok() else None)
        cleanup.extend(lambda sig=sig, handler=handler: signal.signal(sig, handler)
                       for sig, handler in old_signals.items())
        cleanup.append(evidence.close)
        for action in cleanup:
            try:
                action()
            except Exception as exc:
                failure = failure or "cleanup_error:" + repr(exc)
                success = False
        if evidence.failure:
            failure = failure or evidence.failure
            success = False
        summary.update(success=success, failure_reason=failure)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

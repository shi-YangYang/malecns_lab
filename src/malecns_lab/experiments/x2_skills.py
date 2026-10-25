"""One bounded official X2 simulation skill probe; defaults to an inert preview.

This is an execution characterization tool, not a neural behavior policy. It
never starts the VM or simulation. A coordinator owns their bounded lifecycle.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from .x2 import _EvidenceLog

BEHAVIORS = ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT")


def validate_command_seconds(value):
    if not math.isfinite(value) or value not in (1., 2.):
        raise ValueError("command-seconds must be finite and exactly 1 or 2 seconds")


def validate_intensity(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("intensity must be finite and within (0, 1]")


def official_wave_poses(packages):
    """Read separate first/terminal references from the hash-bound official CSV.

    Fixed MC InterruptToNext(0) calls ComputeMaxDisplacement(0), which indexes
    frame 0 in the map populated by LoadAniFrames. It does not capture the
    arbitrary pre-action joint pose. Natural playback uses the terminal frame.
    """
    from . import x2_preset_identity as identity
    from ..control.x2_skills import RIGHT_ARM
    relative = 'mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/animation/SM_wave_right_hand.csv'
    path = Path(packages) / relative
    digest = identity.sha(path)
    if digest != identity.MC_CONTRACT[relative]:
        raise ValueError('Official wave reference identity mismatch')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or float(rows[0]['timeMS']) != 0 or float(rows[-1]['timeMS']) != 4280:
        raise ValueError('Unexpected official wave frame range')
    poses = [{joint: float(row['command_pos::' + joint]) for joint in RIGHT_ARM} for row in (rows[0], rows[-1])]
    if any(not math.isfinite(v) for pose in poses for v in pose.values()):
        raise ValueError('Invalid official wave reference pose')
    return dict(first=poses[0], terminal=poses[1], asset=relative, sha256=digest)


def _event_record(kind, phase, fields, *, received_monotonic, wall_time):
    """Keep the adapter's source ``monotonic`` distinct from recorder receipt.

    Both monotonic clocks are local to this probe process; ``wall_time`` is the
    recorder's Unix receipt time. ROS source timestamps remain in ``message``.
    """
    return dict(kind=kind, received_monotonic=received_monotonic,
                wall_time=wall_time, phase=phase, **fields)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--sim-only", action="store_true")
    p.add_argument("--skill", choices=BEHAVIORS, default="HOLD")
    p.add_argument("--output", type=Path)
    p.add_argument("--wave-probe", action="store_true", help="isolated characterization only; cancellation remains unverified")
    p.add_argument("--wave-cancel-probe", action="store_true", help="one fixed motion=0, 0.5 seconds after measured wave start; simulation only")
    p.add_argument("--cancel-after", type=float, help="request cancellation after this many seconds; wave cannot certify cancellation")
    p.add_argument("--command-seconds", type=float, default=1., help="continuous motion pulse: 1 or 2 seconds; does not alter preset wave playback")
    p.add_argument("--intensity", type=float, default=1., help="skill intensity in (0, 1]; continuous velocity uses the existing 0.2 scaling")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        validate_command_seconds(args.command_seconds)
        validate_intensity(args.intensity)
    except ValueError as error:
        p.error(str(error))
    if args.cancel_after is not None and (not math.isfinite(args.cancel_after) or not .1 <= args.cancel_after <= 4.):
        p.error("cancel-after must be finite and within .1..4 seconds")
    if not args.execute:
        print(json.dumps(dict(mode="preview", ros_connected=False, skill=args.skill,
            source="malecns.x2_mvp", command_seconds=args.command_seconds, intensity=args.intensity, max_target=.2,
            startup_seconds=12., wave_seconds=12., stop_seconds=12., wall_seconds=40.,
            wave_cancellation_verified=False)))
        return 0
    if (not args.sim_only or os.environ.get("ROS_DOMAIN_ID") != "47"
            or os.environ.get("ROS_LOCALHOST_ONLY") != "1"):
        p.error("execute requires --sim-only, ROS_DOMAIN_ID=47 and ROS_LOCALHOST_ONLY=1")
    if args.output is None or (args.skill == "WAVE_RIGHT" and not args.wave_probe):
        p.error("execute requires a new --output; WAVE_RIGHT also requires --wave-probe")
    if args.skill == "WAVE_RIGHT" and args.cancel_after is not None:
        p.error("Wave uses separate natural/cancel probes; arbitrary cancel-after is forbidden")
    if args.wave_cancel_probe and (args.skill != "WAVE_RIGHT" or not args.wave_probe):
        p.error("wave-cancel-probe requires WAVE_RIGHT and wave-probe")
    return execute(args)


def execute(args):
    # Explicit gates precede all ROS imports and service calls.
    validate_command_seconds(args.command_seconds)
    intensity = getattr(args, "intensity", 1.)
    validate_intensity(intensity)
    compat_binding = None
    wave_poses = None
    wave_cancel_probe = getattr(args, "wave_cancel_probe", False)
    if args.skill == "WAVE_RIGHT":
        from . import x2_preset_identity as preset_identity
        compat_binding = preset_identity.verify_build(preset_identity.LINUX_ROOT)
        wave_poses = official_wave_poses(preset_identity.LINUX_ROOT.parent)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from aimdk_msgs.msg import McCommonState, McLocomotionVelocity, McInputAction, JointStateArray
    from aimdk_msgs.srv import SetMcInputSource, GetMcPresetMotionState
    from nav_msgs.msg import Odometry
    from ..control.x2_skills import SkillExecutor, SkillCommand, SOURCE, RIGHT_ARM, SKILL_PACK_ID

    args.output.mkdir(parents=True, exist_ok=False)
    evidence = _EvidenceLog((args.output / "events.jsonl").open("x", encoding="utf-8"))
    started = time.monotonic()
    wall_deadline = started + 40.
    interrupted, signals = [], {}
    node = core = None
    failure = None
    preset_boundary = None
    query_future = None
    query_sent = 0.
    next_query = started
    phase = "startup"
    logging_failure_handled = False

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        return str(value) if isinstance(value, float) and not math.isfinite(value) else value

    def raw(message):
        return clean(message_to_ordereddict(message))

    def write(kind, **fields):
        nonlocal failure, logging_failure_handled
        event = _event_record(kind, phase, fields, received_monotonic=time.monotonic(), wall_time=time.time())
        if not evidence.record(event):
            failure = failure or evidence.failure
            if not logging_failure_handled and core and core.command and not core.terminal:
                logging_failure_handled = True
                core.cancel("evidence_write_error")

    def publish(command):
        message = McLocomotionVelocity()
        message.header.stamp = node.get_clock().now().to_msg()
        message.source = SOURCE
        message.forward_velocity, message.lateral_velocity, message.angular_velocity = command
        publisher.publish(message)

    def preset(command):
        nonlocal query_future
        if query_future is not None:
            query_future.cancel()
            query_future = None
        if preset_boundary is None:
            raise RuntimeError("isolated_preset_helper_missing")
        preset_boundary.request(command)

    def on_odom(message):
        try:
            p, q = message.pose.pose.position, message.pose.pose.orientation
            v, w = message.twist.twist.linear, message.twist.twist.angular
            ok = core.update_odom((p.x,p.y,p.z), (q.x,q.y,q.z,q.w), (v.x,v.y,v.z), (w.x,w.y,w.z), received=time.monotonic())
            write("odom", accepted=ok, message=raw(message))
        except Exception as e:
            core.cancel("odom_callback_error")
            write("callback_error", topic="odom", error=repr(e))

    def on_mc(message):
        try:
            m = message.motion_status
            ok = core.update_mc(message.input_source.name, message.action_info.action_desc,
                message.body_status.value, message.action_info.status.value,
                player_state=m.player_state.value, motion=m.motion, area=m.control_area.value,
                motion_type=m.type.value, received=time.monotonic())
            write("mc", accepted=ok, message=raw(message))
        except Exception as e:
            core.cancel("mc_callback_error")
            write("callback_error", topic="mc", error=repr(e))

    def on_joints(message):
        try:
            joints = {j.name: (j.position, j.velocity) for j in message.joints}
            ok = core.update_joints(joints, received=time.monotonic())
            if any(j.error_code for j in message.joints if j.name in RIGHT_ARM):
                core.cancel("arm_joint_error")
                ok = False
            write("arm_joints", accepted=ok, message=raw(message))
        except Exception as e:
            core.cancel("joint_callback_error")
            write("callback_error", topic="arm", error=repr(e))

    def service_updates():
        nonlocal query_future, next_query, query_sent
        now = time.monotonic()
        if preset_boundary is not None:
            for event in preset_boundary.poll():
                if event["kind"] == "error":
                    core.cancel(event["payload"]["error"])
                    if core.command is None:
                        raise RuntimeError(event["payload"]["error"])
                elif event["kind"] == "sent":
                    core.preset_sent(event["monotonic"])
                    write("preset_request", **event["payload"])
                elif event["kind"] == "response":
                    response = event["payload"]["message"]
                    write("preset_ack", response=response, cdr_hex=event["payload"]["cdr_hex"])
                    r = response["response"]
                    core.preset_ack(r["header"]["code"], r["task_id"], r["state"]["value"])
                elif event["kind"] == "cancel_sent":
                    core.preset_cancel_sent(event["monotonic"])
                    write("preset_cancel_request", **event["payload"], helper_sent_at=event["monotonic"])
                elif event["kind"] == "cancel_response":
                    response = event["payload"]["message"]
                    write("preset_cancel_ack", response=response, cdr_hex=event["payload"]["cdr_hex"])
                    r = response["response"]
                    core.preset_cancel_ack(r["header"]["code"], r["task_id"], r["state"]["value"])
        if query_future is not None:
            if query_future.done():
                response = query_future.result()
                write("preset_query", response=raw(response) if response else None, scope="global_player_state_not_task_identity")
                if response:
                    r = response.response
                    core.preset_query(r.header.code, r.task_id, r.state.value, received=now)
                elif core.command:
                    core.cancel("preset_query_unknown")
                query_future = None
            elif now-query_sent >= .5:
                if core.command:
                    core.cancel("preset_query_timeout")
                query_future.cancel()
                query_future = None
        # Query only one outstanding request and never retry an action request.
        if args.skill == "WAVE_RIGHT" and query_future is None and now >= next_query and query_client.service_is_ready():
            if core.command is None or core.task_id is not None:
                request = GetMcPresetMotionState.Request()
                request.request.header.stamp = node.get_clock().now().to_msg()
                request.request.task_id = core.task_id or 0
                query_future = query_client.call_async(request)
                query_sent, next_query = now, now + .1
                write("preset_query_request", message=raw(request))

    def step():
        began = time.monotonic()
        deadline = began+.008
        for _ in range(64):
            left = deadline-time.monotonic()
            if left <= 0:
                break
            rclpy.spin_once(node, timeout_sec=min(.001,left))
        service_updates()
        if interrupted and core.command and not core.cancel_requested:
            core.cancel("interrupted:" + str(interrupted[0]))
        if (args.cancel_after is not None and core.command and not core.cancel_requested
                and time.monotonic()-core.started_at >= args.cancel_after):
            core.cancel("probe_cancel")
        write("tick", **core.tick())
        if wave_cancel_probe:
            core.request_probe_cancel()
        time.sleep(max(0., .02-(time.monotonic()-began)))

    def register():
        deadline = started+12.
        client = node.create_client(SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource")
        while not client.service_is_ready() and time.monotonic() < deadline and not interrupted:
            step()
        for action in (McInputAction.INPUTACTION_ADD, McInputAction.INPUTACTION_ENABLE):
            if not client.service_is_ready() or time.monotonic() >= deadline or interrupted or evidence.failure:
                return False
            request = SetMcInputSource.Request()
            request.request.header.stamp = node.get_clock().now().to_msg()
            request.action.value = action
            request.input_source.name = SOURCE
            request.input_source.priority, request.input_source.timeout = 40, 1000
            future = client.call_async(request)
            end = min(deadline, time.monotonic()+2.)
            while not future.done() and time.monotonic() < end and not interrupted:
                step()
            response = future.result() if future.done() else None
            write("registration", action=action, response=raw(response) if response else None)
            if response and response.response.header.code == 0:
                return True
            if response is None:
                return False
        return False

    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signals[sig] = signal.signal(sig, lambda sig, frame: interrupted.append(sig))
        node = Node("malecns_x2_skill_probe")
        publisher = node.create_publisher(McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10)
        query_client = node.create_client(GetMcPresetMotionState, "/aimdk_5Fmsgs/srv/GetMcPresetMotionState")
        core = SkillExecutor(publish, preset, enabled=("HOLD",args.skill), wave_probe=args.wave_probe,
                             wave_contract_probe=compat_binding is not None, wave_cancel_probe=wave_cancel_probe,
                             wave_return_pose=wave_poses["first" if wave_cancel_probe else "terminal"] if wave_poses else None,
                             wave_return_reference=("official_animation_first_frame" if wave_cancel_probe
                                 else "official_animation_terminal_frame") if wave_poses else None,
                             command_seconds=args.command_seconds,
                             record=lambda event: write("execution", event=event))
        if compat_binding is not None:
            # Main process retains the new query/status types; never imports old Set.
            from .x2_preset_compat import PresetBoundary
            parent_identity = preset_identity.runtime_identity(preset_identity.LINUX_ROOT, "v1.1.0")
            write("preset_parent_identity", binding=compat_binding, identity=parent_identity)
            preset_boundary = PresetBoundary(args.output, write, binding=compat_binding, allow_cancel=wave_cancel_probe)
            preset_boundary.start()
        core.motion.arm()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(Odometry, "/aima/hal/odom/state", on_odom, qos)
        node.create_subscription(McCommonState, "/aima/mc/common/state", on_mc, qos)
        node.create_subscription(JointStateArray, "/aima/hal/joint/arm/state", on_joints, qos)
        write("configuration", skill=args.skill, skill_pack_id=SKILL_PACK_ID, source=SOURCE,
              command_seconds=core.command_seconds, intensity=intensity, max_target=.2, wave_seconds=12., wall_seconds=40.,
              cancellation_verified=False, simulation_only=True,
              wave_probe_mode="cancel" if wave_cancel_probe else "natural", cancel_delay_after_measured_start=.5,
              query_scope="global_player_state_not_task_identity",
              wave_reference=wave_poses,
              source_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in
                             (("runner",__file__), ("executor",__import__("malecns_lab.control.x2_skills",fromlist=["x"]).__file__))})
        if not register():
            raise RuntimeError("source_registration_failed")
        core.motion.confirm_source_registration()
        while time.monotonic() < started+12. and not interrupted:
            step()
            if core.standing_ready() and (args.skill != "WAVE_RIGHT" or
                    (preset_boundary.ready and core.query_state == 1 and time.monotonic()-core.query_at <= .5 and core.joints is not None)):
                break
        if interrupted or evidence.failure:
            raise RuntimeError("interrupted_before_skill")
        if preset_boundary is not None and not preset_boundary.ready:
            raise RuntimeError("preset_helper_not_ready_before_skill")
        command = SkillCommand(0, "isolated-skill-probe", args.skill,
                               "FINISH" if args.skill == "HOLD" else "START", intensity, time.monotonic())
        write("probe_command", command=command.__dict__)
        if not core.submit(command):
            raise RuntimeError("skill_preflight_rejected")
        phase = "execute"
        while not core.terminal and time.monotonic() < wall_deadline-1.:
            step()
        if not core.terminal:
            core.cancel("wall_deadline")
            failure = "wall_deadline"
    except BaseException as error:
        failure = "skill_exception:" + repr(error)
        write("exception", error=repr(error))
        if core:
            core.cancel(failure)
    finally:
        phase = "cleanup"
        if core and core.command and not core.terminal:
            core.cancel(failure or "cleanup")
            end = min(wall_deadline, time.monotonic()+12.)
            while time.monotonic() < end and not core.terminal:
                try:
                    step()
                except BaseException as error:
                    failure = failure or "cleanup_error:" + repr(error)
                    try:
                        core.motion.request_stop("cleanup_error",failure=True)
                        core.motion.tick()
                    except BaseException:
                        pass
                    time.sleep(.02)
        if core:
            # Always publish zero, without pretending this cancels arm animation.
            try:
                if core.motion.state in ("CLAIMING", "FORWARD"):
                    core.motion.request_stop("process_exit", failure=True)
                publish((0., 0., 0.))
                write("final_zero", command=[0., 0., 0.], motion_state=core.motion.state)
            except BaseException as error:
                failure = failure or "final_zero_error:" + repr(error)
        if preset_boundary is not None:
            try:
                preset_boundary.close()
            except BaseException as error:
                failure = failure or "preset_helper_cleanup:" + repr(error)
        summary = core.summary() if core else {"status":"FAILED"}
        summary.update(failure=failure or evidence.failure, wall_seconds=time.monotonic()-started,
                       command_seconds=core.command_seconds if core else args.command_seconds,
                       requested_intensity=intensity,
                       success=summary.get("status")==("CANCELLED" if wave_cancel_probe else "COMPLETED") and not failure and not evidence.failure,
                       probe_only=True, skill_pack_id=SKILL_PACK_ID)
        evidence.summary(args.output / "summary.json", summary)
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in signals.items():
            signal.signal(sig, handler)
        evidence.close()
    print(json.dumps({k:summary.get(k) for k in ("status","success","failure","reason","wall_seconds")}), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

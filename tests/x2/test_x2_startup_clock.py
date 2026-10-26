"""Startup publication clock regression; synthetic time and feedback, no IO."""
import pytest

from malecns_lab.control.x2 import FORWARD, SOURCE, ZERO, X2MotionAdapter
from malecns_lab.control.x2_skills import VelocityCore
from malecns_lab.control.x2_skill_profile import compensated_action_skill_profile
from malecns_lab.control.x2_interaction import compensated_action_interaction_profile
from .test_x2_motion import Clock, feedback, setup_adapter, stable_for
from .test_x2_action_continuity import rig, submit


def ready_request(*, ttl=1., claim=False):
    core, clock, sent = setup_adapter(allow_empty_source_claim=claim)
    core.confirm_source_registration()
    core.arm()
    feedback(core, source='' if claim else SOURCE)
    assert core.request_forward(ttl)
    assert core._first_nonzero_published_at is None
    assert core._startup_transition_deadline is None
    return core, clock, sent


def test_startup_17ms_publish_delay_preserves_real_running_arrival_inside_half_second():
    core, clock, sent = ready_request()
    request_at = clock.now
    intent_deadline = core.intent_deadline
    # 093738 trace: request447.995168046, publish448.012258046,
    # last TRANSITION448.506015963, RUNNING448.509551297.
    clock.now = request_at + .017090000
    first = core.tick()
    anchor = clock.now
    assert first['first_nonzero_published_at'] == anchor
    assert first['startup_transition_deadline'] == anchor + .5
    for offset in (.1, .2, .3, .402617084, .510847917):
        clock.now = request_at + offset
        feedback(core, action_status=200)
        core.tick()
        assert sent[-1] == FORWARD and not core.failures
    clock.now = request_at + .514383251
    feedback(core, mode='LOCOMOTION_DEFAULT')
    result = core.tick()
    assert result['state'] == 'FORWARD' and sent[-1] == FORWARD
    assert not core.failures and core.intent_deadline == intent_deadline
    assert result['startup_transition_deadline'] == anchor + .5


def test_startup_late_running_does_not_rescue_expired_publish_anchored_window():
    core, clock, sent = ready_request()
    clock.advance(.017)
    anchor = core.tick()['first_nonzero_published_at']
    for offset in (.1, .2, .3, .4):
        clock.now = anchor + offset
        feedback(core, action_status=200)
        core.tick()
    clock.now = anchor + .5
    feedback(core, mode='LOCOMOTION_DEFAULT')
    result = core.tick()
    assert result['state'] == 'STOPPING' and sent[-1] == ZERO
    assert 'startup_transition_timeout' in core.failures
    assert not result['startup_transition_pending']


def test_startup_success_timestamp_is_after_publisher_returns():
    clock, sent = Clock(), []
    def publish(command):
        clock.advance(.017)
        sent.append(command)
    core = X2MotionAdapter(publish, clock=clock)
    core.arm(); feedback(core)
    assert core.request_forward(1.)
    request_at = clock.now
    result = core.tick()
    assert sent == [FORWARD]
    assert result['first_nonzero_published_at'] == request_at + .017
    assert result['startup_transition_deadline'] == clock.now + .5
    assert result['intent_expires'] == request_at + 1.


def test_startup_publish_exception_does_not_start_window_and_zeros():
    core, clock, _ = ready_request()
    sent = []
    def publish(command):
        if command != ZERO:
            clock.advance(.017)
            raise RuntimeError('publisher failed')
        sent.append(command)
    core.publish = publish
    result = core.tick()
    assert result['first_nonzero_published_at'] is None
    assert result['startup_transition_deadline'] is None
    assert 'publish_error' in core.failures and sent == [ZERO]
    clock.advance(.1); feedback(core)
    core.tick()
    assert sent == [ZERO, ZERO] and core._first_nonzero_published_at is None


@pytest.mark.parametrize('zero_at', ['velocity_target', 'effective_sink'])
def test_startup_effective_zero_cannot_start_clock(zero_at):
    clock, sent = Clock(), []
    def publish(command):
        effective = ZERO if zero_at == 'effective_sink' else command
        sent.append(effective)
        return effective
    core = VelocityCore(publish, clock=clock)
    core.arm(); feedback(core)
    assert core.request_velocity('TURN_LEFT', .5, 1.)
    if zero_at == 'velocity_target':
        core.target = ZERO
    result = core.tick()
    assert sent == [ZERO] and result['attempts'] == [{'command':ZERO, 'ok':True}]
    assert result['first_nonzero_published_at'] is result['startup_transition_deadline'] is None


def test_startup_skill_budget_substituted_zero_does_not_start_clock():
    r = rig(compensated_action_skill_profile, compensated_action_interaction_profile)
    assert submit(r, 'TURN_LEFT')
    deadlines = (r.core.deadline, r.core.precision_hard_deadline)
    r.core.integral_nonzero_seconds = r.core.turn_precision['nonzero_budget_seconds']
    result = r.core.motion.tick()
    assert r.commands[-1] == ZERO and result['attempts'] == [{'command':ZERO, 'ok':True}]
    assert result['first_nonzero_published_at'] is result['startup_transition_deadline'] is None
    assert r.core.stop_at is not None and r.core.precision_failure == 'integral_turn_motion_budget'
    assert deadlines == (r.core.deadline, r.core.precision_hard_deadline)


def test_startup_no_successful_nonzero_has_request_anchored_wait_limit():
    core, clock, sent = ready_request(ttl=5.)
    request_at = clock.now
    # No publishing tick until the bound; fresh observations cannot renew it.
    for offset in (.2, .4, .5):
        clock.now = request_at + offset
        feedback(core)
    result = core.tick()
    assert result['startup_publish_deadline'] == request_at + .5
    assert result['startup_transition_deadline'] is None
    assert sent == [ZERO] and 'startup_publish_timeout' in core.failures


def test_startup_transition_before_first_nonzero_is_not_attributed_to_request():
    core, clock, sent = ready_request()
    clock.advance(.01); feedback(core, action_status=200)
    result = core.tick()
    assert sent == [ZERO] and 'mc_not_ready' in core.failures
    assert result['startup_transition_deadline'] is None


@pytest.mark.parametrize('bound', ['intent', 'claim'])
def test_startup_first_publish_does_not_extend_original_intent_or_claim(bound):
    core, clock, sent = ready_request(ttl=.2 if bound == 'intent' else 2., claim=bound == 'claim')
    request_at = clock.now
    original_intent = core.intent_deadline
    clock.advance(.017); core.tick()
    for offset in ((.1,) if bound == 'intent' else (.1, .2, .3, .4)):
        clock.now = request_at + offset
        feedback(core, source='' if bound == 'claim' else SOURCE)
        core.tick()
    clock.now = request_at + (.2 if bound == 'intent' else .5)
    feedback(core)  # Claim arriving at its original deadline is already late.
    result = core.tick()
    assert sent[-1] == ZERO
    assert ('intent_expired' if bound == 'intent' else 'claim_timeout') in core.failures
    if bound == 'claim':
        assert result['claim_deadline'] == request_at + .5
        assert core.ownership_confirmed_at is None
    assert original_intent == request_at + (.2 if bound == 'intent' else 2.)


def test_startup_continued_publications_and_intents_do_not_renew_segment_clock():
    core, clock, sent = ready_request(ttl=5.)
    requested_at = clock.now
    clock.advance(.017)
    initial = core.tick()
    for offset in (.1, .2, .3, .4):
        clock.now = requested_at + offset
        feedback(core, action_status=200)
        assert core.request_forward(5.)
        result = core.tick()
        assert result['first_nonzero_published_at'] == initial['first_nonzero_published_at']
        assert result['startup_transition_deadline'] == initial['startup_transition_deadline']
        assert result['startup_publish_deadline'] == requested_at + .5
        assert core.intent_deadline == requested_at + 5.
        assert sent[-1] == FORWARD


def test_startup_cancel_clears_pending_and_next_stopped_segment_gets_new_anchor():
    core, clock, sent = ready_request()
    clock.advance(.017)
    first = core.tick()['first_nonzero_published_at']
    clock.advance(.1); feedback(core, action_status=200)
    assert core._startup_transition_pending
    core.request_stop('cancelled')
    result = core.tick()
    assert sent[-1] == ZERO and not result['startup_transition_pending']
    stable_for(core, clock)
    assert core.state == 'STOPPED'
    assert core.request_forward(1.)
    assert core._first_nonzero_published_at is core._startup_transition_deadline is None
    clock.advance(.017)
    result = core.tick()
    assert result['first_nonzero_published_at'] == clock.now > first
    assert result['startup_transition_deadline'] == clock.now + .5
    assert sent[-1] == FORWARD and not core.failures

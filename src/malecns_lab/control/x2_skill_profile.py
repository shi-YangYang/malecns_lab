"""Fixed simulation qualification profile, independent of brain/task selection."""
from copy import deepcopy
import json
import math

from .x2_skills import SKILL_PACK_ID, validate_command_durations

PROFILE_ID = "malecns.x2.sim.advance1-turn2.v1"


SIM_WAVE_PROFILE_ID = "malecns.x2.sim.advance1-turn2-wave1002.v1"
SIM_FIXED_ADVANCE_PROFILE_ID = "malecns.x2.sim.advance-fixed02-1s-turn2.v1"
SIM_FROZEN_FULL_PROFILE_ID = "malecns.x2.sim.frozen-fixed02-turn04-wave1002.v1"
SIM_FEEDBACK_FULL_PROFILE_ID = "malecns.x2.sim.frozen-fixed02-turn04-feedback035-wave1002.v2"
SIM_HIGH_TURN_FULL_PROFILE_ID = "malecns.x2.sim.frozen-fixed02-turn08-feedback035-wave1002.v3"
SIM_MEASURED_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-wave1002.v4"
SIM_TAPERED_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-taper04-wave1002.v5"
SIM_PRECISION_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-pd-wave1002.v6"
SIM_SETTLED_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-settled08-wave1002.v7"
SIM_BALANCE_SWAY_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-wave-yaw006.v8"
SIM_QUARTER_TURN_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-final5-wave-yaw006.v9"
SIM_INTEGRAL_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-pid1-wave-yaw006.v10"
SIM_COMPENSATED_ACTION_PROFILE_ID = "malecns.x2.sim.frozen-advance1m-turn90-pid1-stopped-comp-wave-yaw006.v11"
MEASURED_ACTION_PROFILE_IDS = (SIM_MEASURED_ACTION_PROFILE_ID, SIM_TAPERED_ACTION_PROFILE_ID,
                               SIM_PRECISION_ACTION_PROFILE_ID, SIM_SETTLED_ACTION_PROFILE_ID,
                               SIM_BALANCE_SWAY_ACTION_PROFILE_ID, SIM_QUARTER_TURN_ACTION_PROFILE_ID,
                               SIM_INTEGRAL_ACTION_PROFILE_ID, SIM_COMPENSATED_ACTION_PROFILE_ID)
FEEDBACK_FULL_PROFILE_IDS = (SIM_FEEDBACK_FULL_PROFILE_ID, SIM_HIGH_TURN_FULL_PROFILE_ID, *MEASURED_ACTION_PROFILE_IDS)
FULL_ACTION_PROFILE_IDS = (SIM_FROZEN_FULL_PROFILE_ID, *FEEDBACK_FULL_PROFILE_IDS)


def candidate_skill_profile(*, wave_enabled=False):
    # Single intensity=1 probes only; this is not a calibration of other
    # intensities or evidence of reaching the relative heading target.
    value = dict(profile_id=PROFILE_ID, skill_pack_id=SKILL_PACK_ID,
        command_seconds_by_behavior=dict(ADVANCE=1., TURN_LEFT=2., TURN_RIGHT=2.),
        hold_mode="standing_confirmation", wave_enabled=False, qualification_only=True)
    if wave_enabled:
        value.update(profile_id=SIM_WAVE_PROFILE_ID, wave_enabled=True, simulation_only=True,
                     wave_contract=sim_wave_contract())
    return value


def validate_candidate_skill_profile(value):
    """Reject any different registration before transport can publish commands."""
    if not isinstance(value, dict):
        raise ValueError("Missing candidate skill profile")
    validate_command_durations(value.get("command_seconds_by_behavior"),
        max_seconds=10. if value.get('profile_id') in MEASURED_ACTION_PROFILE_IDS else
            5. if value.get('profile_id') in FEEDBACK_FULL_PROFILE_IDS else 2.)
    expected = skill_profile_by_id(value.get("profile_id"))
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Candidate skill profile differs from the fixed simulation configuration")
    return expected


def fixed_advance_skill_profile():
    """One qualified SIM primitive; the brain still selects behavior and phase."""
    value = candidate_skill_profile()
    value.update(profile_id=SIM_FIXED_ADVANCE_PROFILE_ID, simulation_only=True,
        advance_execution={"mode": "fixed_velocity", "velocity_m_s": [0.2, 0., 0.],
            "command_seconds": 1.,
            "intensity_semantics": "record original positive model intensity; do not scale ADVANCE velocity",
            "qualification_scope": "two consecutive fixed SIM primitives; not neural or multi-behavior qualification",
            "evidence": {"run_id": "skills-20261023-205604-c569d60d", "sha256": {
                "host_summary.json": "9621c9c8817cca68d534595fda5f387b8a7cfeec642b2ef60ab818f29245a2bd",
                "official_assets.json": "dc9e5d01d99a6a6ce747bb005e47d601d485cff91d14c5c5b759e7c980e07671",
                "transport_manifest.json": "22dab4822344aaca6563fa44a368963ca92e4182150ffd7ef13686cc2bd15505",
                "probe-00-advance/summary.json": "0aae3436502fe9bb722f06372e8e41d36f620b4bad944420aaa79b3e1c5c4ac9",
                "probe-00-advance/events.jsonl": "37a6e4bdb0cebdc21d950bf88e0de5e66f29f42d8dc16146643f1a8d1e1d7ae2",
                "probe-01-advance/summary.json": "0978160b306d69b1eace29ed07eab8383338bacc23d14dbc1f1839aff2c0532f",
                "probe-01-advance/events.jsonl": "f507bdf21a2d375135a785e34d2fc3b4bdd02163834718705eee29e9dd0695d1"}}})
    return value


def skill_profile_by_id(profile_id):
    if profile_id == SIM_COMPENSATED_ACTION_PROFILE_ID:
        return compensated_action_skill_profile()
    if profile_id == SIM_INTEGRAL_ACTION_PROFILE_ID:
        return integral_action_skill_profile()
    if profile_id == SIM_QUARTER_TURN_ACTION_PROFILE_ID:
        return quarter_turn_action_skill_profile()
    if profile_id == SIM_BALANCE_SWAY_ACTION_PROFILE_ID:
        return balance_sway_action_skill_profile()
    if profile_id == SIM_SETTLED_ACTION_PROFILE_ID:
        return settled_action_skill_profile()
    if profile_id == SIM_PRECISION_ACTION_PROFILE_ID:
        return precision_action_skill_profile()
    if profile_id == SIM_TAPERED_ACTION_PROFILE_ID:
        return tapered_action_skill_profile()
    if profile_id == SIM_MEASURED_ACTION_PROFILE_ID:
        return measured_action_skill_profile()
    if profile_id == SIM_HIGH_TURN_FULL_PROFILE_ID:
        return high_turn_full_skill_profile()
    if profile_id == SIM_FEEDBACK_FULL_PROFILE_ID:
        return feedback_full_skill_profile()
    if profile_id == SIM_FROZEN_FULL_PROFILE_ID:
        return frozen_full_skill_profile()
    if profile_id == SIM_FIXED_ADVANCE_PROFILE_ID:
        return fixed_advance_skill_profile()
    if profile_id in (PROFILE_ID, SIM_WAVE_PROFILE_ID):
        return candidate_skill_profile(wave_enabled=profile_id == SIM_WAVE_PROFILE_ID)
    raise ValueError("Unknown simulation skill profile")


def frozen_full_skill_profile():
    """Explicit execution calibration; neither task selection nor qualification."""
    value = candidate_skill_profile(wave_enabled=True)
    value.update(profile_id=SIM_FROZEN_FULL_PROFILE_ID,
        advance_execution=fixed_advance_skill_profile()['advance_execution'],
        turn_execution=dict(mode='fixed_angular_velocity', left_rad_s=.4, right_rad_s=-.4,
            command_seconds=2., measured_stop_angle_rad=.35,
            intensity_semantics='record original positive model intensity; do not scale TURN velocity',
            evidence_scope='same bounded manual SIM primitive; direction evidence is not reaching the +/-0.45 request heading',
            acceptance='retain measured direction, angular stop and spatial protection; no target-angle completion claim'),
        wave_preparation=dict(mode='single_official_SD', max_requests_per_wave=1,
            timeout_seconds=6., post_response_standing_seconds=1.,
            claim='preparation only; no verified yaw reanchoring or turn-then-wave qualification'),
        qualification_scope='all-action neural SIM integration pending actual acceptance; original model FAIL retained')
    return value


def feedback_full_skill_profile():
    """New bounded feedback candidate; keep the historical two-second contract."""
    value = frozen_full_skill_profile()
    value.update(profile_id=SIM_FEEDBACK_FULL_PROFILE_ID,
        command_seconds_by_behavior=dict(ADVANCE=1., TURN_LEFT=5., TURN_RIGHT=5.))
    value['turn_execution'].update(mode='measured_angle_bounded_velocity', command_seconds=5.,
        evidence_scope='candidate; full-brain consecutive-action SIM acceptance pending',
        acceptance='reach signed 0.35 rad before the 5 s cap, then confirm angular stop and positive signed displacement; unmet target fails; explicit cancellation stays cancellation')
    return value


def high_turn_full_skill_profile():
    """Isolated 0.8 rad/s SIM comparison; all v2 feedback guards are retained."""
    value = feedback_full_skill_profile()
    value['profile_id'] = SIM_HIGH_TURN_FULL_PROFILE_ID
    value['turn_execution'].update(left_rad_s=.8, right_rad_s=-.8,
        evidence_scope='single-speed SIM qualification candidate; startup-threshold hypothesis unverified')
    return value


def measured_action_skill_profile():
    """Measured one-metre/quarter-turn simulation actions; not qualification."""
    value = high_turn_full_skill_profile()
    value.update(profile_id=SIM_MEASURED_ACTION_PROFILE_ID,
        command_seconds_by_behavior=dict(ADVANCE=10., TURN_LEFT=10., TURN_RIGHT=10.))
    value['advance_execution'] = dict(mode='measured_distance_bounded_velocity', velocity_m_s=[.2, 0., 0.],
        command_seconds=10., measured_stop_distance_m=1.,
        intensity_semantics='record original positive model intensity; do not scale ADVANCE velocity',
        acceptance='reach 1 m along initial body forward axis, then confirm stop and final forward distance >= 1 m',
        qualification_scope='new measured-distance SIM candidate; no qualification inherited from the one-second primitive')
    value['turn_execution'].update(command_seconds=10., measured_stop_angle_rad=math.pi / 2,
        evidence_scope='new 90-degree SIM candidate; genuine heading input changed from +/-0.45 to +/-pi/2; no retraining',
        acceptance='reach signed pi/2, then confirm stop and final signed angle >= pi/2; record actual overshoot; no precision qualification')
    return value


def tapered_action_skill_profile():
    """One bounded terminal-speed reduction; preserve the original v4 evidence."""
    value = measured_action_skill_profile()
    value['profile_id'] = SIM_TAPERED_ACTION_PROFILE_ID
    value['turn_execution'].update(mode='measured_angle_two_speed_velocity',
        slowdown_remaining_angle_rad=.5, terminal_speed_rad_s=.4,
        slowdown_rule='once remaining measured signed angle is in (0, 0.5] rad, reduce magnitude to 0.4 rad/s; no acceleration or deadline renewal',
        evidence_scope='unverified terminal-speed candidate; v4 measured 1.06/1.14 rad/s at zero and 90.8/99.0 degree final turns; unchanged genuine pi/2 input and weights')
    return value


def balance_sway_action_skill_profile():
    """Opt-in SIM wave envelope candidate; final return guards are unchanged."""
    value = measured_action_skill_profile()
    value['profile_id'] = SIM_BALANCE_SWAY_ACTION_PROFILE_ID
    value['wave_execution'] = dict(active_yaw_limit_rad=.06, return_yaw_limit_rad=.03,
        evidence_scope='unverified engineering candidate, not an official safety limit; no action acceptance inherited from v4 or the helper/cancellation contract',
        acceptance='active yaw <= 0.06 rad; final return still <= 0.03 rad with all original stationary, arm-return, freshness and deadline guards')
    return value


def quarter_turn_action_skill_profile():
    """Same v8 execution, with a symmetric measured stopped-angle criterion."""
    value = balance_sway_action_skill_profile()
    value['profile_id'] = SIM_QUARTER_TURN_ACTION_PROFILE_ID
    value['turn_execution'].update(final_tolerance_rad=math.radians(5.),
        acceptance='cross the original signed pi/2 target before the unchanged 10 s cap, then confirm original measured stop, motion, direction and no protection; final absolute target error <= 5 degrees; late target feedback cannot rescue timeout',
        evidence_scope='new terminal criterion matching the existing driver +/-5 degree criterion; v8 stopped at 88.741 degrees after crossing but failed its one-sided final bound; physical v9 qualification pending')
    return value


def precision_action_skill_profile():
    """Unqualified, opt-in terminal-heading controller; no new brain decision."""
    value = measured_action_skill_profile()
    value['profile_id'] = SIM_PRECISION_ACTION_PROFILE_ID
    value['turn_execution'].update(mode='measured_heading_pd_bounded_correction',
        feedback_controller=dict(position_gain=2., angular_rate_gain=.25,
            final_tolerance_rad=math.radians(5.), settle_entry_error_rad=math.radians(2.5),
            settle_entry_speed_rad_s=.15, max_corrections=2, total_deadline_seconds=22.,
            stall_window_seconds=2., stall_min_angle_rad=.01),
        controller_rule='clip(2 * (original signed goal - measured accumulated angle) - 0.25 * measured angular speed, -0.8, +0.8); no integral or minimum speed',
        correction_rule='at most two internal corrections after confirmed stop and standing readiness; preserve origin, goal and execution identity; all motion ends by original START + 10 s; total close deadline START + 22 s',
        acceptance='final absolute heading error <= 5 degrees and original measured stop plus standing readiness; crossing pi/2 alone is insufficient; stalled, exhausted or expired control fails',
        evidence_scope='unverified PD and finite-correction candidate; MC small-command response and convergence unknown; v4 remains default; genuine heading, neural weights and history unchanged')
    return value


def integral_action_skill_profile():
    """One opt-in anti-windup PID candidate; no precision qualification claim."""
    value = balance_sway_action_skill_profile()
    value['profile_id'] = SIM_INTEGRAL_ACTION_PROFILE_ID
    value['execution_poll_seconds_by_behavior'] = dict(TURN_LEFT=36., TURN_RIGHT=36.)
    value['turn_execution'].update(mode='measured_heading_pid_bounded_correction',
        feedback_controller=dict(position_gain=2., angular_rate_gain=.25,
            integral_gain=1., integral_limit_rad_s=.8, max_sample_dt_seconds=.05,
            final_tolerance_rad=math.radians(5.), settle_entry_error_rad=math.radians(2.5),
            settle_entry_speed_rad_s=.15, max_corrections=1, nonzero_budget_seconds=10.,
            stop_timeout_seconds=12., total_deadline_seconds=34.,
            stall_window_seconds=2., stall_min_angle_rad=.01,
            max_sample_records=512, max_stage_records=64),
        controller_rule='clip(2 * original heading error - 0.25 * measured angular speed + I, -0.8, +0.8); I gain 1/s^2, bounded +/-0.8; integrate each fresh new odom once using dt <=0.05s; same-direction P+I+D saturation freezes I, opposite direction may unwind',
        integral_reset='zero on error sign change, stopping, cancellation, protection, stale feedback and each new request; no integration while stopping; a stale sample gap is skipped',
        correction_rule='one internal correction only after fresh measured STOPPED and standing; preserve original origin/goal; cumulative actual published nonzero <=10s, per segment TTL <=10s and remaining budget; each stop <=12s; START+34s hard deadline with final 12s stop-only; CONTINUE renews no budget; non-improving correction fails',
        acceptance='original measured stop and standing readiness with final absolute heading error <=5 degrees; crossing alone never completes; stall requires continuous saturation for 2s with net angle progress <0.01rad',
        evidence_scope='unverified fixed integral candidate targeting the v6 low-speed stall; zero-command recoil and closed-loop convergence remain unverified; old profiles/default and neural inputs/weights unchanged')
    return value


def compensated_action_skill_profile():
    """One same-request measured residual compensation; independently opt-in."""
    value = integral_action_skill_profile()
    value['profile_id'] = SIM_COMPENSATED_ACTION_PROFILE_ID
    value['turn_execution'].update(mode='measured_heading_pid_single_stopped_residual_compensation',
        reference_compensation=dict(max_first_stopped_error_rad=math.radians(15.),
            measured_angle_limit_rad=math.radians(105.)),
        controller_rule='same v10 bounded PID; only its tracking error and settle entry use the internal reference; original target, brain heading and final +/-5 degree scoring never change',
        correction_rule='one correction only after fresh real STOPPED plus standing, no protection/cancellation and original error >5 and <=15 degrees; internal reference = original signed target + (original signed target - first stopped angle); >15 degrees fails without clipping; v10 10s cumulative nonzero/22s motion/34s total/12s stop budgets do not renew; measured absolute angle >105 degrees latches failure and zero, not a guarantee against physical overshoot',
        evidence_scope='unverified same-request compensation candidate; no across-run bias, no gain search or inherited v10 precision qualification; old profiles/default and neural inputs/weights unchanged')
    return value


def settled_action_skill_profile():
    """Opt-in finite corrections using only the measured +/-0.8 command."""
    value = measured_action_skill_profile()
    value['profile_id'] = SIM_SETTLED_ACTION_PROFILE_ID
    value['execution_poll_seconds_by_behavior'] = dict(TURN_LEFT=48., TURN_RIGHT=48.)
    value['turn_execution'].update(mode='measured_stopped_heading_finite_correction',
        settled_controller=dict(final_tolerance_rad=math.radians(5.), max_corrections=2,
            nonzero_budget_seconds=10., stop_timeout_seconds=12., total_deadline_seconds=46.),
        controller_rule='same brain START and original signed goal; command +/-0.8 toward measured error until goal crossing, then zero and confirm full stop; no forward, integral, deadzone or fixed braking offset',
        correction_rule='at most two internal corrections after fresh stopped standing feedback; a correction must reduce absolute stopped error or fail; cumulative nonzero command time <= 10 s; each stop <= 12 s; hard deadline START + 46 s, last 12 s stop only; CONTINUE renews no budget',
        acceptance='original measured stop plus standing readiness and final absolute heading error <= 5 degrees; crossing alone is insufficient',
        evidence_scope='unverified finite-correction candidate; +/-0.8 moved both directions but short correction convergence is unknown; v4 remains default; genuine heading, neural weights and history unchanged')
    return value


# Immutable qualification facts; control never reads assets or starts a process.
_SIM_WAVE_CONTRACT = {'contract_id': 'malecns.x2.fixed-sim-wave1002-cancel0.v1',
 'simulation_only': True,
 'helper_binding': {'build_sha256': 'a8326719caa97adeba2340643e3bbc4b5d6170f25e4b4fbb7aa2d0049b96363f',
                    'inspection_sha256': '2efde8a7e2cb63b15f6b60109cecbeda4a6e35ad3baa0d90223dc0da23d30a99',
                    'mc_contract_sha256': {'mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/animation/SM_wave_right_hand.csv': '23554287aba6b32dd488a1b52ce1c9bdfd6134cf1469494a1f8a4aaa94399760',
                                           'mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/animation_player.yaml': 'd7b533981404e2ee51c7a2ddca95782f1385ecda423f65889a51bd018ffad2da',
                                           'mc/bin/mc_param/src/robot/lx2501_3_t2d5/motion_handler.yaml': 'dd7949fce4e0ba25a2d337748c181101a6760fd585cc32eb6bd52035072ca152',
                                           'mc/lib/libmc_manager_motion.so': 'a63e6d7e1c0705a0eeb29a07abad14310c97c8796eed804a51a359fe070a9b1e',
                                           'mc/lib/libmc_module_main_service.so': '8a73ec6555bcec2c37a18085fa599adf26bcceb974e370e641a880f39bc580e3',
                                           'mc/lib/libmc_runner_animation_player.so': 'b256374e35510360cc02bcda0d93b2150e85880b75ecb95e37a51fb11fd913ce'},
                    'mc_sha256': '935d4226013617a7aee2cdb525e31b6c46e44d608bf88f99eb61467b746cb931',
                    'prepared_sha256': '08a5b5f169b77721fce916346256b01b8baf886bfc09c256c82f4692a82bb6ac',
                    'schema': 'malecns.x2.preset-compat.v2',
                    'zip_sha256': '5bbcf724d54fb28f153db0d272f9acb7906bb1d2cac7dd7ccdc699a5c7eeab35'},
 'qualification_evidence': {'skills-20261023-233100-1ef27ab8': {'host_summary.json': 'd8a3405b51ce50d3959bfa87830e36e41cca48bc1592441c4eee9aa98244974b',
                                                                'probe-00-wave_right/summary.json': '65ea0372dc4d8cdf3fdb1f8e06070dcae9681adf1fdb6b315aafbffbc3cd1ce8',
                                                                'probe-00-wave_right/events.jsonl': '5d0ee1492b3d0eadf6b3d8878ba1aa805d229e83752528e709a7f0ca7bf4b968',
                                                                'preset_compat/binding.json': 'dc36fbe855bc8e18b9d6cf6db99f8973651341001f34d942dc4f28f5b8531d82'},
                            'skills-20261023-233900-98dcab71': {'host_summary.json': 'd840e7ca90310c94acf1bf6cf15eaee52bfc7b26826ce41d5f4c709b9f2d6597',
                                                                'probe-00-wave_right/summary.json': 'c2da509a17fcf0e630b7f78418225ed9b629faa8998161123cbc1e6a0282432e',
                                                                'probe-00-wave_right/events.jsonl': '205ebfbbaa3dda0b34b3f8030cc2f3c9d6a6b49942e090ad6951a949db308136',
                                                                'preset_compat/binding.json': 'dc36fbe855bc8e18b9d6cf6db99f8973651341001f34d942dc4f28f5b8531d82'}},
 'attribution': 'exclusive_per_action_helper; post_send_common_motion_area_type_and_joints; '
                'global_query_not_task_identity',
 'natural_return': 'official_animation_terminal_frame',
 'cancel_return': 'official_animation_first_frame',
 'cancel_request': {'motion': 0, 'area': 2, 'interrupt': True},
 'qualification_scope': 'fixed simulation primitives; neural session integration requires actual acceptance',
 'observed_cancel_delay_s': 0.503113042,
 'helper_preparation_seconds': 12.0,
 'action_seconds': 12.0}


def sim_wave_contract():
    return deepcopy(_SIM_WAVE_CONTRACT)

"""Fixed simulation qualification profile, independent of brain/task selection."""
from copy import deepcopy
import json

from .x2_skills import SKILL_PACK_ID, validate_command_durations

PROFILE_ID = "malecns.x2.sim.advance1-turn2.v1"


SIM_WAVE_PROFILE_ID = "malecns.x2.sim.advance1-turn2-wave1002.v1"
SIM_FIXED_ADVANCE_PROFILE_ID = "malecns.x2.sim.advance-fixed02-1s-turn2.v1"
SIM_FROZEN_FULL_PROFILE_ID = "malecns.x2.sim.frozen-fixed02-turn04-wave1002.v1"


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
    validate_command_durations(value.get("command_seconds_by_behavior"))
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

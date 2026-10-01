# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-4 configurations (2026-09-30): literature-style versions of earlier negative ideas, on top of E4/E6.

(E6 = the E4 environment trained with entropy_coef 0.005, passed as an agent override.)

Air       E16  legged_gym-style feet air time: threshold 0.5 s, paid at touchdown only, weight 1.0 (x dt)
FallRamp  E17  fall penalty (1 on the falling step) whose weight ramps linearly 0 -> -30 over the first 300 iterations
               (x dt: <= -0.5 per fall)
FallConst      diagnostic: the same penalty at -30 from the first step (used to verify the per-episode sum)
Asym      E18  asymmetric critic: an extra "critic" observation group = actor terms + forward height scan (45 points);
               rsl_rl uses an env group named "critic" for the critic when agent obs_groups is empty
Push      E19  weak pushes ±0.15 m/s every 10-15 s, no mass randomization
EyesNoise E20  C1+Eyes+FlatLanes with uniform scan noise ±0.1 m during training only
OracleVar F2   E15 terrain with box height random per tile (±3-15 cm) instead of fixed ±10 cm
Teacher   T1   E15 (boxes-only) + privileged actor inputs: forward scan (no noise) + robot friction (teacher check)

All reward changes, pushes and noise are training-only; Play variants remove them.
"""

import copy

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_rough2_env_cfg import _r2_to_play, feet_air_time_ant
from .ant_rough3_env_cfg import _apply_r3, forward_height_scan
from .ant_rough_env_cfg import AntRoughE4EnvCfg, height_above_ground_below_minimum

ITER_STEPS = 32  # env steps per PPO iteration (num_steps_per_env)


def ramp_reward_weight(
    env: ManagerBasedRLEnv, env_ids, term_name: str, final_weight: float, num_steps: int
) -> torch.Tensor:
    """Set the weight of ``term_name`` to final_weight * min(1, steps / num_steps) (linear ramp)."""
    frac = min(1.0, env.common_step_counter / max(num_steps, 1))
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = final_weight * frac
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(term_cfg.weight)


def fell_this_step(env: ManagerBasedRLEnv, minimum_height: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 on the step the fall termination fires (same condition as the termination term), else 0.

    ``mdp.is_terminated_term`` is not used: in Isaac Lab 2.3 it reads ``TerminationManager._term_dones``, which is
    latched until the environment's next termination (it is not cleared on reset). A robot whose previous episode
    ended by falling is then penalised on every step of its next episode (measured: 588x the intended sum; this is
    what collapsed E11).
    """
    return height_above_ground_below_minimum(env, minimum_height, sensor_cfg).float()


def _add_fall_penalty(env_cfg, ramp_iterations: int) -> None:
    env_cfg.rewards.fall_penalty = RewTerm(
        func=fell_this_step, weight=0.0, params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_ray")}
    )
    env_cfg.curriculum.fall_ramp = CurrTerm(
        func=ramp_reward_weight,
        params={"term_name": "fall_penalty", "final_weight": -30.0, "num_steps": ramp_iterations * ITER_STEPS},
    )


def _add_height_scan_sensor(env_cfg) -> None:
    env_cfg.scene.height_scan = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.25, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def robot_friction_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mean static friction of the robot's shapes per env (privileged; randomized once at startup).

    Read from PhysX and cached after the first environment step (before that the startup randomization may not
    have been applied yet), so it costs nothing per step afterwards.
    """
    cache = getattr(env, "_robot_friction_cache", None)
    if cache is None or env.common_step_counter == 0:
        mats = env.scene[asset_cfg.name].root_physx_view.get_material_properties()
        cache = mats[..., 0].mean(dim=1, keepdim=True).to(env.device)
        env._robot_friction_cache = cache
        if env.common_step_counter > 0:
            print(f"[T1] robot friction obs cached: min {cache.min().item():.3f} max {cache.max().item():.3f} "
                  f"mean {cache.mean().item():.3f} ({cache.shape[0]} envs)", flush=True)
    return cache


def _apply_r4(env_cfg, variant: str) -> None:
    if variant == "Air":
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
        env_cfg.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=True
        )
        env_cfg.rewards.feet_air_time = RewTerm(
            func=feet_air_time_ant,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 0.5},
        )
    elif variant == "FallRamp":
        _add_fall_penalty(env_cfg, ramp_iterations=300)
    elif variant == "FallConst":
        _add_fall_penalty(env_cfg, ramp_iterations=0)
    elif variant == "Asym":
        _add_height_scan_sensor(env_cfg)
        critic = copy.deepcopy(env_cfg.observations.policy)
        critic.height_scan = ObsTerm(func=forward_height_scan, params={"sensor_cfg": SceneEntityCfg("height_scan")})
        critic.enable_corruption = False
        critic.concatenate_terms = True
        env_cfg.observations.critic = critic
    elif variant == "Push":
        env_cfg.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
        )
    elif variant == "Teacher":
        # T1: E15 (boxes-only Oracle) with privileged actor inputs: forward scan (no noise) + robot friction
        from .ant_rough2_env_cfg import _apply_r2

        _apply_r2(env_cfg, ("Oracle",))
        _add_height_scan_sensor(env_cfg)
        env_cfg.observations.policy.height_scan = ObsTerm(
            func=forward_height_scan, params={"sensor_cfg": SceneEntityCfg("height_scan")}
        )
        env_cfg.observations.policy.robot_friction = ObsTerm(func=robot_friction_obs)
    elif variant == "OracleVar":
        # F2: E15 terrain (boxes only, no curriculum) with box height random per tile in ±3-15 cm
        from .ant_rough2_env_cfg import _apply_r2

        _apply_r2(env_cfg, ("Oracle",))
        gen = env_cfg.scene.terrain.terrain_generator
        gen.sub_terrains["boxes"].grid_height_range = (0.03, 0.15)
        gen.difficulty_range = (0.0, 1.0)
    elif variant == "EyesNoise":
        _apply_r3(env_cfg, eyes=True, flat_lanes=True)
        env_cfg.observations.policy.height_scan.noise = Unoise(n_min=-0.1, n_max=0.1)
        env_cfg.observations.policy.enable_corruption = True
    else:
        raise ValueError(variant)


def _r4_to_play(env_cfg) -> None:
    _r2_to_play(env_cfg)
    env_cfg.observations.policy.enable_corruption = False


def _make(variant: str, play: bool):
    def __post_init__(self):
        AntRoughE4EnvCfg.__post_init__(self)
        _apply_r4(self, variant)
        if play:
            _r4_to_play(self)

    name = f"AntR4{variant}{'Play' if play else ''}EnvCfg"
    cls = configclass(type(name, (AntRoughE4EnvCfg,), {"__post_init__": __post_init__, "__module__": __name__}))
    globals()[name] = cls


R4_VARIANTS = ("Air", "FallRamp", "FallConst", "Asym", "Push", "EyesNoise", "Teacher", "OracleVar")
for _v in R4_VARIANTS:
    _make(_v, play=False)
    _make(_v, play=True)

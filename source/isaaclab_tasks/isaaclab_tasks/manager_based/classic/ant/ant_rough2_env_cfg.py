# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-2 variants (2026-09-30) on top of E4, stacked on the overnight winner.

Task ids: ``Isaac-Ant-R2-<Flags>-v0`` / ``-Play-v0`` where <Flags> is any ordered subset of ``R2_FLAGS``.
Config classes are built lazily on attribute access (``AntR2<Flags>[Play]EnvCfg``) so that only the combinations
actually used are created.

Flags (round 1, reused):  Flat, Hist, Push                        (see ant_rough_env_cfg._apply_variant)
Flags (round 2):
    Safe     fall penalty, ground-clearance margin, roll/pitch rate penalty   (training-only rewards, ④)
    LowFric  ground combined by multiply + robot friction 0.05–1.2 -> effective friction down to 0.05 (⑦)
    Air      contact sensor on the feet + air-time reward without velocity command    (training-only reward, ①)
    NoCurr   no lane curriculum, random difficulty per tile                            (verification ②)
    Oracle   training terrain = boxes ±10 cm only                                      (verification ⑥)

Scores are always computed with the original rewards: every training-only reward is removed in Play.
"""

import copy

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGenerator
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_rough_env_cfg import AntRoughE4EnvCfg, _apply_variant, _to_play, base_height_above_ground

R2_FLAGS = ("Flat", "Hist", "Push", "Safe", "LowFric", "Air", "NoCurr", "Oracle")
_TRAIN_ONLY_REWARDS = ("fall_penalty", "clearance", "ang_vel_xy", "feet_air_time")


def clearance_deficit(env: ManagerBasedRLEnv, target: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """How far the torso is below ``target`` above the ground (0 when above)."""
    return torch.clamp(target - base_height_above_ground(env, sensor_cfg)[:, 0], min=0.0)


def feet_air_time_ant(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Air time of each foot minus ``threshold``, paid when the foot touches down (no velocity command gating)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.sum((air_time - threshold) * first_contact, dim=1)


def _apply_r2(env_cfg, flags):
    _apply_variant(env_cfg, tuple(f for f in flags if f in ("Flat", "Hist", "Push")))
    gen = env_cfg.scene.terrain.terrain_generator
    if "Safe" in flags:
        env_cfg.rewards.fall_penalty = RewTerm(
            func=mdp.is_terminated_term, weight=-300.0, params={"term_keys": "torso_height"}
        )
        env_cfg.rewards.clearance = RewTerm(
            func=clearance_deficit, weight=-5.0, params={"target": 0.45, "sensor_cfg": SceneEntityCfg("height_ray")}
        )
        env_cfg.rewards.ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    if "LowFric" in flags:
        # multiply has priority over average in PhysX, so effective friction = robot friction (ground is 1.0)
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        env_cfg.events.robot_friction.params["static_friction_range"] = (0.05, 1.2)
        env_cfg.events.robot_friction.params["dynamic_friction_range"] = (0.05, 1.2)
    if "Air" in flags:
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
        env_cfg.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=True
        )
        env_cfg.rewards.feet_air_time = RewTerm(
            func=feet_air_time_ant,
            weight=2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 0.1},
        )
    if "NoCurr" in flags or "Oracle" in flags:
        gen.class_type = TerrainGenerator
        gen.curriculum = False
        env_cfg.curriculum = None
    if "Oracle" in flags:
        boxes = copy.deepcopy(gen.sub_terrains["boxes"])
        boxes.proportion = 1.0
        boxes.grid_height_range = (0.10, 0.10)
        gen.sub_terrains = {"boxes": boxes}
        gen.difficulty_range = (1.0, 1.0)


def _r2_to_play(env_cfg):
    _to_play(env_cfg)
    for name in _TRAIN_ONLY_REWARDS:
        if hasattr(env_cfg.rewards, name):
            setattr(env_cfg.rewards, name, None)


def _parse(name):
    body = name[len("AntR2"):-len("EnvCfg")]
    play = body.endswith("Play")
    if play:
        body = body[: -len("Play")]
    flags, rest = [], body
    for flag in R2_FLAGS:
        if rest.startswith(flag):
            flags.append(flag)
            rest = rest[len(flag):]
    if rest:
        raise AttributeError(name)
    return tuple(flags), play


def __getattr__(name):
    if not (name.startswith("AntR2") and name.endswith("EnvCfg")):
        raise AttributeError(name)
    flags, play = _parse(name)

    def __post_init__(self):
        AntRoughE4EnvCfg.__post_init__(self)
        _apply_r2(self, flags)
        if play:
            _r2_to_play(self)

    cls = configclass(type(name, (AntRoughE4EnvCfg,), {"__post_init__": __post_init__, "__module__": __name__}))
    globals()[name] = cls
    return cls

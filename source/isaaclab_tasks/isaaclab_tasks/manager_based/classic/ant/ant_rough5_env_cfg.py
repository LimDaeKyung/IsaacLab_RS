# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch 10 (prepared 2026-10-01, share-fb): train on every evaluation terrain type ("AllMix").

Same as E15 (R2 Oracle: E4 observations/termination, robot friction randomization, no curriculum, entropy 0.005
via the agent override) except the training terrain: instead of boxes ±10 cm only, every tile samples one of the
11 EVAL_TERRAINS shapes with boxes kept as the largest share. Lockbox 1 and lockbox 2 types are NOT trained on.
The Play variant is identical to the E15/F3a Play config (boxes ±10 cm, no training-only randomization).
"""

import copy

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGenerator
from isaaclab.utils import configclass

from .ant_eval_env_cfg import EVAL_TERRAINS
from .ant_rough2_env_cfg import AntR2OracleEnvCfg, AntR2OraclePlayEnvCfg

# share of each training shape; boxes ±10 cm (the E15 terrain) stays the largest share
ALLMIX_PROPORTIONS = {
    "boxes_mid": 0.30,
    "boxes_low": 0.07,
    "boxes_high": 0.07,
    "flat": 0.07,
    "rough_low": 0.07,
    "rough_high": 0.07,
    "slope": 0.07,
    "stairs": 0.07,
    "ho_boxes_fine": 0.07,
    "ho_obstacles": 0.07,
    "ho_wave": 0.07,
}


def allmix_sub_terrains():
    subs = {}
    for name, share in ALLMIX_PROPORTIONS.items():
        cfg = EVAL_TERRAINS[name]
        cfg = terrain_gen.MeshPlaneTerrainCfg() if cfg is None else copy.deepcopy(cfg)
        cfg.proportion = share
        subs[name] = cfg
    return subs


@configclass
class AntR5AllMixEnvCfg(AntR2OracleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.class_type = TerrainGenerator
        gen.curriculum = False
        # evaluation presets are defined at difficulty 1.0 (fixed parameters), so keep it fixed here as well
        gen.difficulty_range = (1.0, 1.0)
        gen.sub_terrains = allmix_sub_terrains()


@configclass
class AntR5AllMixPlayEnvCfg(AntR2OraclePlayEnvCfg):
    """Same Play config as E15/F3a (boxes ±10 cm, training-only terms removed)."""

    pass


@configclass
class AntR5AllMixLowFricEnvCfg(AntR5AllMixEnvCfg):
    """Batch 12 P1: AllMix + LowFric (ground friction combined by multiply, robot friction 0.05-1.2)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.events.robot_friction.params["static_friction_range"] = (0.05, 1.2)
        self.events.robot_friction.params["dynamic_friction_range"] = (0.05, 1.2)


def _moderate_low_friction(env_cfg, lo):
    env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="average",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    env_cfg.events.robot_friction.params["static_friction_range"] = (lo, 1.2)
    env_cfg.events.robot_friction.params["dynamic_friction_range"] = (lo, 1.2)


@configclass
class AntR5AllMixMF02EnvCfg(AntR5AllMixEnvCfg):
    """Batch 13: AllMix + moderate low friction (effective 0.2-1.2, multiply)."""

    def __post_init__(self):
        super().__post_init__()
        _moderate_low_friction(self, 0.2)


@configclass
class AntR5AllMixMF04EnvCfg(AntR5AllMixEnvCfg):
    """Batch 13: AllMix + mild low friction (effective 0.4-1.2, multiply)."""

    def __post_init__(self):
        super().__post_init__()
        _moderate_low_friction(self, 0.4)


def allmix_wide_sub_terrains():
    """Same 11 shape families and shares as AllMix, but each tile samples its difficulty in [0, 1] so heights
    vary continuously (ranges stay below lockbox 5: stairs ≤ 12 cm, slope ≤ 0.30)."""
    from .ant_rough_env_cfg import ScaledRandomUniformTerrainCfg

    return {
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.44, grid_width=0.45, grid_height_range=(0.03, 0.15), platform_width=2.0
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.07),
        "rough": ScaledRandomUniformTerrainCfg(proportion=0.14, noise_range=(0.0, 0.12), noise_step=0.01, border_width=0.25),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.07, slope_range=(0.0, 0.30), platform_width=2.0, border_width=0.25
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07, step_height_range=(0.03, 0.12), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "boxes_fine": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.07, grid_width=0.30, grid_height_range=(0.03, 0.12), platform_width=2.0
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.07, obstacle_height_mode="choice", obstacle_width_range=(0.4, 1.2),
            obstacle_height_range=(0.03, 0.12), num_obstacles=40, platform_width=2.0, border_width=0.25,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.07, amplitude_range=(0.03, 0.20), num_waves=4, border_width=0.25),
    }


@configclass
class AntR5AllMixWideEnvCfg(AntR5AllMixEnvCfg):
    """Batch 14 AX: AllMix with per-tile random difficulty (continuous heights) instead of fixed presets."""

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.difficulty_range = (0.0, 1.0)
        gen.sub_terrains = allmix_wide_sub_terrains()


@configclass
class AntR5AllMixWideMF04EnvCfg(AntR5AllMixWideEnvCfg):
    """Batch 14 AXF: AllMix-Wide + mild low friction (0.4-1.2, multiply)."""

    def __post_init__(self):
        super().__post_init__()
        _moderate_low_friction(self, 0.4)


@configclass
class AntR5AllMixWideMF02EnvCfg(AntR5AllMixWideEnvCfg):
    """Batch 15 AXF20/SPF20: AllMix-Wide + moderate low friction (0.2-1.2, multiply)."""

    def __post_init__(self):
        super().__post_init__()
        _moderate_low_friction(self, 0.2)


@configclass
class AntR5AllMixWideMF03EnvCfg(AntR5AllMixWideEnvCfg):
    """Batch 16 CF30: AllMix-Wide + low friction between mild and moderate (0.3-1.2, multiply)."""

    def __post_init__(self):
        super().__post_init__()
        _moderate_low_friction(self, 0.3)


def allmix_hard_sub_terrains():
    """Batch 16 CH: AllMix-Wide with the bump-type heights raised by one third (boxes ≤ 20 cm, boxes_fine and
    obstacles ≤ 16 cm, wave ≤ 25 cm). Rough, slope and stairs stay as in Wide so that lockbox 5/6 (stairs 15 cm,
    slope 0.35, inverted stairs/slope, rough 15 cm) remain beyond the training ranges."""
    subs = allmix_wide_sub_terrains()
    subs["boxes"].grid_height_range = (0.03, 0.20)
    subs["boxes_fine"].grid_height_range = (0.03, 0.16)
    subs["obstacles"].obstacle_height_range = (0.03, 0.16)
    subs["wave"].amplitude_range = (0.03, 0.25)
    return subs


@configclass
class AntR5AllMixHardMF04EnvCfg(AntR5AllMixWideMF04EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = allmix_hard_sub_terrains()


@configclass
class AntR5AllMixHardMF02EnvCfg(AntR5AllMixWideMF02EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = allmix_hard_sub_terrains()


def _add_fall_penalty_const(env_cfg, weight):
    """Batch 17 (user request 10:20: fewer falls): training-only penalty on the step the fall termination fires.

    Uses ``fell_this_step`` (same condition as the termination; not the latched ``is_terminated_term``). The Play
    config removes ``fall_penalty`` (``_TRAIN_ONLY_REWARDS``), so the scored reward is unchanged. Per fall the
    return changes by weight x step_dt (1/60 s): -1200 -> -20, -3000 -> -50 (SP50 returns are about 100-110).
    """
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg

    from .ant_rough4_env_cfg import fell_this_step

    env_cfg.rewards.fall_penalty = RewTerm(
        func=fell_this_step, weight=weight, params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_ray")}
    )


@configclass
class AntR5AllMixWideMF04FP20EnvCfg(AntR5AllMixWideMF04EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -1200.0)


@configclass
class AntR5AllMixWideMF04FP50EnvCfg(AntR5AllMixWideMF04EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -3000.0)


@configclass
class AntR5AllMixWideMF04FP30EnvCfg(AntR5AllMixWideMF04EnvCfg):
    """Batch 18 FPU: fall penalty -30 per fall (weight -1800 x 1/60 s)."""

    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -1800.0)


# Batch 19: the batch-16 ideas (friction 0.3, taller bumps), now on the fall-penalised line (-20 or -30 per fall)
@configclass
class AntR5AllMixWideMF03FP20EnvCfg(AntR5AllMixWideMF03EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -1200.0)


@configclass
class AntR5AllMixHardMF04FP20EnvCfg(AntR5AllMixHardMF04EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -1200.0)


@configclass
class AntR5AllMixWideMF03FP30EnvCfg(AntR5AllMixWideMF03EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -1800.0)


@configclass
class AntR5AllMixHardMF04FP30EnvCfg(AntR5AllMixHardMF04EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -1800.0)


@configclass
class AntR5AllMixHardMF03FP20EnvCfg(AntR5AllMixHardMF04FP20EnvCfg):
    """Batch 20 HF: taller bumps (FPH) + friction 0.3-1.2 (FPF) + fall penalty -20 per fall."""

    def __post_init__(self):
        super().__post_init__()
        _moderate_low_friction(self, 0.3)


# Batch 22 (descriptive penalty sweep from SP50): -10 and -35 per fall, next to C60 (0), FP20 (-20) and FP50 (-50)
@configclass
class AntR5AllMixWideMF04FP10EnvCfg(AntR5AllMixWideMF04EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -600.0)


@configclass
class AntR5AllMixWideMF04FP35EnvCfg(AntR5AllMixWideMF04EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _add_fall_penalty_const(self, -2100.0)

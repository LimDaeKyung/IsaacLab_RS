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

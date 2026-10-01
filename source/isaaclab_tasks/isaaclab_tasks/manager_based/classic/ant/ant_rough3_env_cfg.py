# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-3 configurations (2026-09-30): C1 baseline and the 2×2 "flat speed" study.

C1        = E4 + no curriculum + low-friction range (entropy_coef 0.005 is passed as an agent override)
Eyes      = forward height scan (separate ray caster ``height_scan``, 2.0 m × 1.0 m, 0.25 m → 45 points) in the actor obs
FlatLanes = 5 of the 20 lanes (the central block, columns 8-12) are flat over all 16 tiles; others are random per tile.
            A contiguous central block keeps robots on flat ground while they drift sideways toward the target
            (1000, 0, 0); edge lanes (|y| up to 95 m) drift out of a 10 m lane at high speed.

The single-ray ``height_ray`` used by the fall termination and the base-height observation is left unchanged.
"""

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGenerator
from isaaclab.utils import configclass

from .ant_rough2_env_cfg import _apply_r2, _r2_to_play
from .ant_rough_env_cfg import AntRoughE4EnvCfg

FLAT_LANES = (8, 9, 10, 11, 12)


class FlatLaneTerrainGenerator(TerrainGenerator):
    """Random terrain per tile, except the columns (lanes) in ``FLAT_LANES``, which are flat end to end."""

    def _generate_random_terrains(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        flat_cfg = self.cfg.sub_terrains["flat"]
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            if sub_col in FLAT_LANES:
                sub_cfg, difficulty = flat_cfg, 0.0
            else:
                sub_cfg = sub_terrains_cfgs[self.np_rng.choice(len(proportions), p=proportions)]
                difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)


def forward_height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso height above each scanned ground point (a miss counts as a deep hole), clipped to [-1, 1]."""
    sensor = env.scene.sensors[sensor_cfg.name]
    torso_z = env.scene["robot"].data.root_pos_w[:, 2:3]
    height = torso_z - sensor.data.ray_hits_w[..., 2]
    return torch.nan_to_num(height, nan=1.0, posinf=1.0, neginf=1.0).clamp(-1.0, 1.0)


def _apply_r3(env_cfg, eyes: bool, flat_lanes: bool) -> None:
    _apply_r2(env_cfg, ("NoCurr", "LowFric"))
    if eyes:
        env_cfg.scene.height_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso",
            offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.25, size=[2.0, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        env_cfg.observations.policy.height_scan = ObsTerm(
            func=forward_height_scan, params={"sensor_cfg": SceneEntityCfg("height_scan")}
        )
    if flat_lanes:
        env_cfg.scene.terrain.terrain_generator.class_type = FlatLaneTerrainGenerator


def _make(name: str, eyes: bool, flat_lanes: bool, play: bool):
    def __post_init__(self):
        AntRoughE4EnvCfg.__post_init__(self)
        _apply_r3(self, eyes, flat_lanes)
        if play:
            _r2_to_play(self)

    cls = configclass(type(name, (AntRoughE4EnvCfg,), {"__post_init__": __post_init__, "__module__": __name__}))
    globals()[name] = cls


R3_VARIANTS = {"C1": (False, False), "C1Eyes": (True, False), "C1FlatLanes": (False, True), "C1EyesFlatLanes": (True, True)}
for _name, (_eyes, _flat) in R3_VARIANTS.items():
    _make(f"AntR3{_name}EnvCfg", _eyes, _flat, play=False)
    _make(f"AntR3{_name}PlayEnvCfg", _eyes, _flat, play=True)

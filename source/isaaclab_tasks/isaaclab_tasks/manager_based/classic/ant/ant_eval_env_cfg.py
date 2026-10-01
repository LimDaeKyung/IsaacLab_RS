# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unseen-terrain evaluation environments for the Ant task.

Only the terrain (shape + physics material) differs from :class:`AntEnvCfg`; observations, rewards,
terminations and episode length are kept identical so that returns are directly comparable.

The terrain is a long strip along +x (the Ant walks toward x=1000). All robots start on the first row
of tiles and are spread over the columns, so every episode is spent on the evaluated terrain.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg

# tile layout: 16 rows (x) x 20 cols (y) of 10 m tiles -> 160 m of terrain ahead of each robot
_TILE_SIZE = 10.0
_NUM_ROWS = 16
_NUM_COLS = 20


def _boxes(height: float) -> terrain_gen.MeshRandomGridTerrainCfg:
    # grid cells are shifted uniformly in [-height, height]
    return terrain_gen.MeshRandomGridTerrainCfg(
        proportion=1.0, grid_width=0.45, grid_height_range=(height, height), platform_width=2.0
    )


def _rough(noise: float) -> terrain_gen.HfRandomUniformTerrainCfg:
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(0.0, noise), noise_step=0.01, border_width=0.25
    )


def _slope(slope: float) -> terrain_gen.HfPyramidSlopedTerrainCfg:
    return terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.0, slope_range=(slope, slope), platform_width=2.0, border_width=0.25
    )


def _stairs(step: float) -> terrain_gen.MeshPyramidStairsTerrainCfg:
    return terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=1.0, step_height_range=(step, step), step_width=0.3, platform_width=3.0, border_width=1.0
    )


def _boxes_fine(height: float) -> terrain_gen.MeshRandomGridTerrainCfg:
    # narrower cells than the 0.45 m used in training
    return terrain_gen.MeshRandomGridTerrainCfg(
        proportion=1.0, grid_width=0.30, grid_height_range=(height, height), platform_width=2.0
    )


def _obstacles(height: float) -> terrain_gen.HfDiscreteObstaclesTerrainCfg:
    return terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=1.0,
        obstacle_height_mode="choice",
        obstacle_width_range=(0.4, 1.2),
        obstacle_height_range=(height, height),
        num_obstacles=40,
        platform_width=2.0,
        border_width=0.25,
    )


def _wave(amplitude: float) -> terrain_gen.HfWaveTerrainCfg:
    return terrain_gen.HfWaveTerrainCfg(
        proportion=1.0, amplitude_range=(amplitude, amplitude), num_waves=4, border_width=0.25
    )


# lockbox: fixed 2026-09-30 00:04 before reading any overnight result. Never used for model selection;
# evaluated exactly once at the very end (E0, E4, final) via experiments/lockbox_eval.sh, not by eval_sweep.
LOCKBOX_TERRAINS = {
    # 8 cm rails (20 cm thick) around the tile platform
    "lock_rails": terrain_gen.MeshRailsTerrainCfg(
        proportion=1.0, rail_thickness_range=(0.2, 0.2), rail_height_range=(0.08, 0.08), platform_width=2.0
    ),
    # 20 cm wide gap ring around the tile platform
    "lock_gaps": terrain_gen.MeshGapTerrainCfg(proportion=1.0, gap_width_range=(0.2, 0.2), platform_width=3.0),
    # start inside a 15 cm deep pit
    "lock_pits": terrain_gen.MeshPitTerrainCfg(proportion=1.0, pit_depth_range=(0.15, 0.15), platform_width=3.0),
    # stepping stones 0.8-1.2 m wide, 5-10 cm apart, holes 30 cm deep
    "lock_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        proportion=1.0,
        stone_height_max=0.03,
        stone_width_range=(0.8, 1.2),
        stone_distance_range=(0.05, 0.10),
        holes_depth=-0.3,
        platform_width=2.0,
        border_width=0.25,
    ),
}


# Lockbox 2 (fixed 2026-10-01 11:00, before batch 10 trains on every EVAL_TERRAINS type).
# Terrain types never used anywhere so far (training, held-out or lockbox 1). Evaluate only at the end of batch 10.
LOCKBOX2_TERRAINS = {
    "lock2_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
        proportion=1.0,
        platform_width=2.0,
        object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=40, height=0.10, radius=0.15),
        object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=40, height=0.10, radius=0.15),
    ),
    "lock2_cones": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
        proportion=1.0,
        platform_width=2.0,
        object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=30, height=0.12, radius=0.30),
        object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=30, height=0.12, radius=0.30),
    ),
    "lock2_tilted_blocks": terrain_gen.MeshRepeatedBoxesTerrainCfg(
        proportion=1.0,
        platform_width=2.0,
        object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            num_objects=40, height=0.08, size=(0.4, 0.4), max_yx_angle=15.0
        ),
        object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            num_objects=40, height=0.08, size=(0.4, 0.4), max_yx_angle=15.0
        ),
    ),
    "lock2_platform": terrain_gen.MeshBoxTerrainCfg(proportion=1.0, box_height_range=(0.10, 0.10), platform_width=5.0),
}


EVAL_TERRAINS = {
    "flat": None,
    "boxes_low": _boxes(0.05),
    "boxes_mid": _boxes(0.10),
    "boxes_high": _boxes(0.15),
    "rough_low": _rough(0.05),
    "rough_high": _rough(0.10),
    "slope": _slope(0.2),
    "stairs": _stairs(0.10),
    # held-out: terrain types / parameters that are never used for training
    "ho_boxes_fine": _boxes_fine(0.10),
    "ho_obstacles": _obstacles(0.10),
    "ho_wave": _wave(0.15),
}
"""Named terrain presets. ``flat`` keeps the original plane; ``ho_*`` are held out from training."""


def _all_terrains():
    return {**EVAL_TERRAINS, **LOCKBOX_TERRAINS, **LOCKBOX2_TERRAINS}


def apply_eval_terrain(
    env_cfg: AntEnvCfg, terrain: str, friction: float = 1.0, seed: int = 0, combine_mode: str = "average"
) -> None:
    """Replace the terrain of ``env_cfg`` in place with the named preset and ground friction."""
    if terrain not in _all_terrains():
        raise ValueError(f"Unknown terrain '{terrain}'. Available: {list(_all_terrains())}")
    terrain_cfg = env_cfg.scene.terrain
    terrain_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode=combine_mode,
        restitution_combine_mode="average",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )
    sub_terrain = _all_terrains()[terrain]
    if sub_terrain is None:
        terrain_cfg.terrain_type = "plane"
        terrain_cfg.terrain_generator = None
        return
    terrain_cfg.terrain_type = "generator"
    terrain_cfg.terrain_generator = TerrainGeneratorCfg(
        seed=seed,
        size=(_TILE_SIZE, _TILE_SIZE),
        border_width=20.0,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        difficulty_range=(1.0, 1.0),
        sub_terrains={terrain: sub_terrain},
    )
    # spawn every robot on the first row so the whole strip lies ahead of it
    terrain_cfg.max_init_terrain_level = 0
    terrain_cfg.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12))


@configclass
class AntEvalEnvCfg(AntEnvCfg):
    """Ant on random boxes (similar to the example unseen terrain)."""

    def __post_init__(self):
        super().__post_init__()
        apply_eval_terrain(self, "boxes_mid")

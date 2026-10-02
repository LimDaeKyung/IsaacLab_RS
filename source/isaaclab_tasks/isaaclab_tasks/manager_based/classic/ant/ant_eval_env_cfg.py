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


# Lockbox 3 (fixed 2026-10-01 15:20, before any batch-11 result): harder parameter variants of known shapes, never
# trained on and never used for selection. Evaluated once per model at the very end of batch 11 (final check).
LOCKBOX3_TERRAINS = {
    "lock3_obstacles_high": _obstacles(0.15),
    "lock3_wave_big": _wave(0.25),
    "lock3_rails_high": terrain_gen.MeshRailsTerrainCfg(
        proportion=1.0, rail_thickness_range=(0.2, 0.2), rail_height_range=(0.12, 0.12), platform_width=2.0
    ),
    "lock3_pits_deep": terrain_gen.MeshPitTerrainCfg(proportion=1.0, pit_depth_range=(0.25, 0.25), platform_width=3.0),
}


# Lockbox 4 (fixed 2026-10-01 21:00, before any batch-12 result): final check for batch 12. lock4_wave_lowfric is
# evaluated with ground friction 0.3 and multiply combine (passed on the evaluate.py command line).
LOCKBOX4_TERRAINS = {
    "lock4_stones_hard": terrain_gen.HfSteppingStonesTerrainCfg(
        proportion=1.0,
        stone_height_max=0.03,
        stone_width_range=(0.7, 1.0),
        stone_distance_range=(0.10, 0.15),
        holes_depth=-0.3,
        platform_width=2.0,
        border_width=0.25,
    ),
    "lock4_cylinders_tall": terrain_gen.MeshRepeatedCylindersTerrainCfg(
        proportion=1.0,
        platform_width=2.0,
        object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=40, height=0.15, radius=0.10),
        object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(num_objects=40, height=0.15, radius=0.10),
    ),
    "lock4_boxes_fine12": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=1.0, grid_width=0.35, grid_height_range=(0.12, 0.12), platform_width=2.0
    ),
    "lock4_wave_lowfric": _wave(0.15),
}


# Lockbox 5 (fixed 2026-10-02 00:30, before any batch-13 result). lock5_boxes_lowfric is evaluated with ground
# friction 0.3 and multiply combine (passed on the evaluate.py command line).
LOCKBOX5_TERRAINS = {
    "lock5_stairs_high": terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=1.0, step_height_range=(0.15, 0.15), step_width=0.3, platform_width=3.0, border_width=1.0
    ),
    "lock5_slope_steep": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.0, slope_range=(0.35, 0.35), platform_width=2.0, border_width=0.25
    ),
    "lock5_cones_dense": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
        proportion=1.0,
        platform_width=2.0,
        object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=50, height=0.10, radius=0.25),
        object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=50, height=0.10, radius=0.25),
    ),
    "lock5_boxes_lowfric": _boxes(0.08),
}


# Lockbox 6 (fixed 2026-10-02 08:12, before any lockbox-5 result and before batch 16): final check for batch 16.
# Downward shapes (inverted stairs / slope) and box obstacles were never trained on or evaluated before.
# lock6_boxobst_midfric is evaluated with ground friction 0.5 and average combine (passed on the command line).
_BOX6 = terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(num_objects=40, height=0.12, size=(0.4, 0.4), max_yx_angle=15.0)
LOCKBOX6_TERRAINS = {
    "lock6_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=1.0, step_height_range=(0.10, 0.10), step_width=0.3, platform_width=3.0, border_width=1.0
    ),
    "lock6_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=1.0, slope_range=(0.25, 0.25), platform_width=2.0, border_width=0.25
    ),
    "lock6_rough_heavy": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(0.0, 0.15), noise_step=0.01, border_width=0.25
    ),
    "lock6_boxobst_midfric": terrain_gen.MeshRepeatedBoxesTerrainCfg(
        proportion=1.0, platform_width=2.0, object_params_start=_BOX6, object_params_end=_BOX6
    ),
}


# Lockbox 7 (fixed 2026-10-02 12:08, before any lockbox-6 result and before batch 18): final check for batch 18.
# lock7_rough_lowfric is evaluated with ground friction 0.4 and multiply combine (passed on the command line).
_PYR7 = terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=20, height=0.15, radius=0.4)
LOCKBOX7_TERRAINS = {
    "lock7_gaps_wide": terrain_gen.MeshGapTerrainCfg(proportion=1.0, gap_width_range=(0.35, 0.35), platform_width=3.0),
    "lock7_pyramids_high": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
        proportion=1.0, platform_width=2.0, object_params_start=_PYR7, object_params_end=_PYR7
    ),
    "lock7_wave_short": terrain_gen.HfWaveTerrainCfg(
        proportion=1.0, amplitude_range=(0.12, 0.12), num_waves=8, border_width=0.25
    ),
    "lock7_rough_lowfric": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(0.0, 0.08), noise_step=0.01, border_width=0.25
    ),
}


# Lockbox 8 (fixed 2026-10-02 14:12, before any lockbox-7 result and before batch 19): final check for batch 19 if
# lockbox 7 is used by batch 18. lock8_slope_lowfric is evaluated with ground friction 0.5 and multiply combine.
_BOX8 = terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(num_objects=25, height=0.36,  # half is buried: 18 cm visible
                                                            size=(0.6, 0.6), max_yx_angle=0.0)
LOCKBOX8_TERRAINS = {
    "lock8_boxes_tall_sparse": terrain_gen.MeshRepeatedBoxesTerrainCfg(
        proportion=1.0, platform_width=2.0, object_params_start=_BOX8, object_params_end=_BOX8
    ),
    "lock8_stairs_down_hf": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
        proportion=1.0, step_height_range=(0.13, 0.13), step_width=0.4, platform_width=3.0, border_width=1.0
    ),
    "lock8_stones_wide": terrain_gen.HfSteppingStonesTerrainCfg(
        proportion=1.0, stone_height_max=0.03, stone_width_range=(1.0, 1.4), stone_distance_range=(0.15, 0.25),
        holes_depth=-0.3, platform_width=2.0, border_width=0.25,
    ),
    "lock8_slope_lowfric": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.0, slope_range=(0.20, 0.20), platform_width=2.0, border_width=0.25
    ),
}


def _all_terrains():
    return {**EVAL_TERRAINS, **LOCKBOX_TERRAINS, **LOCKBOX2_TERRAINS, **LOCKBOX3_TERRAINS, **LOCKBOX4_TERRAINS,
            **LOCKBOX5_TERRAINS, **LOCKBOX6_TERRAINS,
            **LOCKBOX7_TERRAINS, **LOCKBOX8_TERRAINS}


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

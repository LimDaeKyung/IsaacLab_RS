# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant training environments for walking on unseen terrain (assignment 1).

Every experiment changes exactly one thing on top of the previous one. Rewards, actions, episode length and
PPO settings are identical to ``Isaac-Ant-v0`` so that returns stay comparable.

Common rule for every experiment (and for re-scoring the baseline): the fall termination uses the torso height
above the ground below it instead of world z. On a flat plane both are identical; on uneven terrain the
world-z rule terminates robots standing in depressions and misses robots flipped on raised terrain.

E1  terrain randomization  : mixed rough terrains, random difficulty per tile
E2  + lane curriculum      : one difficulty per lane, robots move to harder lanes when they walk far
E3  + friction randomization: robot-side friction randomized per environment
E4  + relative height obs  : ``base_height`` measured from the ground below (ray cast) instead of world z

Each ``*PlayEnvCfg`` is the evaluation/submission variant: no curriculum and no training-only randomization,
so that the terrain can be swapped freely (plane or generator) as the TA does.
"""

import copy
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_env_cfg import AntEnvCfg
from .ant_eval_env_cfg import apply_eval_terrain

##
# Terrain
##


def _scaled_random_uniform_terrain(difficulty: float, cfg: HfRandomUniformTerrainCfg):
    """Random uniform height field whose maximum noise grows with the difficulty.

    The stock term ignores the difficulty, which would make every lane equally rough.
    """
    scaled = copy.copy(cfg)
    lo, hi = cfg.noise_range
    top = lo + (hi - lo) * difficulty
    scaled.noise_range = (lo, max(lo + cfg.noise_step, round(top / cfg.noise_step) * cfg.noise_step))
    return hf_terrains.random_uniform_terrain(difficulty, scaled)


@configclass
class ScaledRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
    function = _scaled_random_uniform_terrain


class LaneTerrainGenerator(TerrainGenerator):
    """Curriculum generator whose difficulty grows along the columns (y) instead of the rows (x).

    The Ant walks along +x for up to ~130 m per episode, so each column is a 160 m lane of constant
    difficulty. Sub-terrain types are sampled per tile from the proportions, so every lane mixes all types.
    """

    def _generate_curriculum_terrains(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        lower, upper = self.cfg.difficulty_range
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                difficulty = (sub_col + self.np_rng.uniform()) / self.cfg.num_cols
                difficulty = lower + (upper - lower) * difficulty
                sub_index = self.np_rng.choice(len(proportions), p=proportions)
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])


TRAIN_SUB_TERRAINS = {
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.25, grid_width=0.45, grid_height_range=(0.02, 0.18), platform_width=2.0
    ),
    "rough": ScaledRandomUniformTerrainCfg(
        proportion=0.20, noise_range=(0.0, 0.10), noise_step=0.01, border_width=0.25
    ),
    "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.10, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
    ),
    "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=0.10, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
    ),
    "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=0.125, step_height_range=(0.03, 0.15), step_width=0.3, platform_width=3.0, border_width=1.0
    ),
    "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=0.125, step_height_range=(0.03, 0.15), step_width=0.3, platform_width=3.0, border_width=1.0
    ),
}
"""Training terrain types. Held-out evaluation types (``ho_*`` in ant_eval_env_cfg) are deliberately absent."""


def _train_terrain_generator(curriculum: bool) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        class_type=LaneTerrainGenerator if curriculum else TerrainGenerator,
        seed=42,
        size=(10.0, 10.0),
        border_width=20.0,
        num_rows=16,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=curriculum,
        difficulty_range=(0.0, 1.0),
        sub_terrains=TRAIN_SUB_TERRAINS,
    )


##
# MDP terms
##


def lane_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    up_distance: float,
    down_distance: float,
    init_max_lane: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Move robots to a harder lane when they walked far along +x, to an easier one when they did not."""
    terrain = env.scene.terrain
    asset = env.scene[asset_cfg.name]
    num_lanes = terrain.terrain_origins.shape[1]
    if not getattr(env, "_lane_curriculum_initialized", False):
        # start every robot on one of the easiest lanes
        terrain.terrain_types[:] = torch.randint_like(terrain.terrain_types, init_max_lane + 1)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env._lane_curriculum_initialized = True
        return torch.mean(terrain.terrain_types.float())
    # called before the reset, so the root still holds the final pose of the episode
    distance = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = distance > up_distance
    move_down = (distance < down_distance) & ~move_up
    lanes = terrain.terrain_types[env_ids] + move_up.long() - move_down.long()
    terrain.terrain_types[env_ids] = torch.clip(lanes, 0, num_lanes - 1)
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    return torch.mean(terrain.terrain_types.float())


def base_height_above_ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso height above the terrain directly below it (single downward ray)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene["robot"]
    height = asset.data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[:, :1, 2]
    # no hit (e.g. outside the terrain mesh) -> fall back to a nominal value
    return torch.nan_to_num(height, nan=0.5, posinf=0.5, neginf=0.5).clamp(-1.0, 2.0)


def height_above_ground_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Fall termination measured from the ground below the torso (identical to world z on a flat plane)."""
    return base_height_above_ground(env, sensor_cfg)[:, 0] < minimum_height


def _add_height_ray(env_cfg: AntEnvCfg) -> None:
    """Single downward ray from above the torso onto the terrain mesh."""
    env_cfg.scene.height_ray = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.0, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # the ray caster finds its prims on the USD stage, which fabric-only clones do not populate
    env_cfg.scene.clone_in_fabric = False


def use_relative_termination(env_cfg: AntEnvCfg) -> None:
    _add_height_ray(env_cfg)
    env_cfg.terminations.torso_height = DoneTerm(
        func=height_above_ground_below_minimum,
        params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_ray")},
    )


##
# Experiments
##


@configclass
class CurriculumCfg:
    lane_level = CurrTerm(
        func=lane_curriculum, params={"up_distance": 40.0, "down_distance": 10.0, "init_max_lane": 3}
    )


@configclass
class AntRoughE1EnvCfg(AntEnvCfg):
    """E1: train on mixed rough terrains with random difficulty per tile."""

    def __post_init__(self):
        super().__post_init__()
        use_relative_termination(self)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _train_terrain_generator(curriculum=False)
        # every robot starts at the beginning of the 160 m strip, spread over the lanes
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12))


@configclass
class AntRoughE2EnvCfg(AntRoughE1EnvCfg):
    """E2: E1 + lane curriculum (constant difficulty per lane, adaptive lane assignment)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _train_terrain_generator(curriculum=True)
        self.curriculum = CurriculumCfg()


@configclass
class AntRoughE3EnvCfg(AntRoughE2EnvCfg):
    """E3: E2 + robot-side friction randomization.

    The terrain has a single material and PhysX averages the two frictions, so randomizing the robot
    material in [0.1, 1.2] against a ground of 1.0 spans an effective friction of about [0.55, 1.1].
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.1, 1.2),
                "dynamic_friction_range": (0.1, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )


def _use_relative_height(env_cfg: AntEnvCfg) -> None:
    _add_height_ray(env_cfg)
    env_cfg.observations.policy.base_height = ObsTerm(
        func=base_height_above_ground, params={"sensor_cfg": SceneEntityCfg("height_ray")}
    )


@configclass
class AntRoughE4EnvCfg(AntRoughE3EnvCfg):
    """E4: E3 + base height measured relative to the ground below instead of world z."""

    def __post_init__(self):
        super().__post_init__()
        _use_relative_height(self)


@configclass
class AntRoughE4bEnvCfg(AntRoughE4EnvCfg):
    """E4b: E4 without the lane curriculum (random difficulty per tile as in E1), to re-test E2 on the best stack."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _train_terrain_generator(curriculum=False)
        self.curriculum = None


##
# Evaluation / submission variants
##


def _to_play(env_cfg: AntEnvCfg) -> None:
    """Drop training-only terms and put the robots on the example-like boxes terrain."""
    env_cfg.curriculum = None
    for name in ("robot_friction", "push_robot", "torso_mass"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    apply_eval_terrain(env_cfg, "boxes_mid")


@configclass
class AntE0PlayEnvCfg(AntEnvCfg):
    """Baseline scored with the common relative-height termination (policy trained on the flat plane)."""

    def __post_init__(self):
        super().__post_init__()
        use_relative_termination(self)
        apply_eval_terrain(self, "boxes_mid")


@configclass
class AntRoughE1PlayEnvCfg(AntRoughE1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


@configclass
class AntRoughE2PlayEnvCfg(AntRoughE2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


@configclass
class AntRoughE3PlayEnvCfg(AntRoughE3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


@configclass
class AntRoughE4PlayEnvCfg(AntRoughE4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


@configclass
class AntRoughE4bPlayEnvCfg(AntRoughE4bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


##
# Overnight variants on top of E4 (2026-09-29): environment-side flags, combined freely.
# Agent-side changes (entropy_coef, obs normalization) are passed as hydra overrides at train/eval time.
##

VARIANT_FLAGS = ("Flat", "Hist", "Push")


def _apply_variant(env_cfg: AntEnvCfg, flags: tuple[str, ...]) -> None:
    if "Flat" in flags:
        # E7: raise the flat share from 0.10 to 0.25, scale the other types down proportionally
        subs = copy.deepcopy(env_cfg.scene.terrain.terrain_generator.sub_terrains)
        others = sum(cfg.proportion for name, cfg in subs.items() if name != "flat")
        for name, cfg in subs.items():
            cfg.proportion = 0.25 if name == "flat" else cfg.proportion * 0.75 / others
        env_cfg.scene.terrain.terrain_generator.sub_terrains = subs
    if "Hist" in flags:
        # E8: 3-step history of the dynamic terms only
        policy = env_cfg.observations.policy
        for term in (policy.base_lin_vel, policy.base_ang_vel, policy.joint_pos_norm, policy.joint_vel_rel):
            term.history_length = 3
    if "Push" in flags:
        # E10: weak pushes and torso mass scaling (training only, removed in Play)
        env_cfg.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
        env_cfg.events.torso_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )


def _variant_name(flags: tuple[str, ...], play: bool) -> str:
    return "AntRoughE4" + "".join(flags) + ("Play" if play else "") + "EnvCfg"


def _make_variant(flags: tuple[str, ...], play: bool):
    def __post_init__(self):
        AntRoughE4EnvCfg.__post_init__(self)
        _apply_variant(self, flags)
        if play:
            _to_play(self)

    name = _variant_name(flags, play)
    cls = configclass(type(name, (AntRoughE4EnvCfg,), {"__post_init__": __post_init__, "__module__": __name__}))
    globals()[name] = cls
    return name


VARIANT_COMBOS = []
for _mask in range(1, 2 ** len(VARIANT_FLAGS)):
    _flags = tuple(f for i, f in enumerate(VARIANT_FLAGS) if _mask & (1 << i))
    _make_variant(_flags, play=False)
    _make_variant(_flags, play=True)
    VARIANT_COMBOS.append(_flags)

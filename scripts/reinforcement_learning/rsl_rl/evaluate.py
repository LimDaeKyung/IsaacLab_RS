# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless quantitative evaluation of an RSL-RL checkpoint on (unseen) terrains.

Every environment runs exactly one full episode; per-episode return, length, fall flag and forward
distance are aggregated and written to a JSON file.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint on unseen terrains (headless).")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments (= evaluated episodes).")
parser.add_argument("--task", type=str, default="Isaac-Ant-Eval-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=24, help="Seed used for the environment.")
parser.add_argument("--terrain", type=str, default=None, help="Terrain preset from ant_eval_env_cfg.EVAL_TERRAINS.")
parser.add_argument("--friction", type=float, default=1.0, help="Static/dynamic friction of the ground.")
parser.add_argument("--terrain_seed", type=int, default=0, help="Seed of the terrain generator.")
parser.add_argument(
    "--combine_mode", type=str, default="average", help="Ground friction combine mode (average, min, multiply, max)."
)
parser.add_argument("--output", type=str, default=None, help="Path of the JSON result file.")
parser.add_argument(
    "--diag_height_relative",
    action="store_true",
    default=False,
    help="Diagnostic only: feed base height relative to the spawn level (env origin z) to the policy.",
)
parser.add_argument(
    "--diag_world_z_termination",
    action="store_true",
    default=False,
    help="Diagnostic only: use the original world-z fall termination (root z < 0.31).",
)
parser.add_argument(
    "--diag_scan", action="store_true", default=False, help="Diagnostic: print height-scan stats and lane assignment."
)
parser.add_argument(
    "--diag_friction", action="store_true", default=False, help="Diagnostic: print the cached robot-friction observation."
)
parser.add_argument("--video", action="store_true", default=False, help="Record the first episode of env 0 (headless).")
parser.add_argument("--video_length", type=int, default=960, help="Length of the recorded video (in steps).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# evaluation is always headless
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant.ant_eval_env_cfg import apply_eval_terrain
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.terrain is not None:
        apply_eval_terrain(env_cfg, args_cli.terrain, args_cli.friction, args_cli.terrain_seed, args_cli.combine_mode)

    if args_cli.diag_height_relative:
        def base_height_above_origin(env, asset_cfg=SceneEntityCfg("robot")):
            return env.scene[asset_cfg.name].data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]

        env_cfg.observations.policy.base_height.func = base_height_above_origin

    if args_cli.diag_world_z_termination:
        from isaaclab.envs.mdp import root_height_below_minimum
        from isaaclab.managers import TerminationTermCfg

        env_cfg.terminations.torso_height = TerminationTermCfg(
            func=root_height_below_minimum, params={"minimum_height": 0.31}
        )

    resume_path = retrieve_file_path(args_cli.checkpoint)

    if args_cli.video:
        # follow env 0 with the camera
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-4.0, 4.0, 2.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        out_dir = os.path.dirname(os.path.abspath(args_cli.output)) if args_cli.output else os.getcwd()
        name = os.path.splitext(os.path.basename(args_cli.output))[0] if args_cli.output else "eval"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(out_dir, "videos"),
            name_prefix=name,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    num_envs = env.num_envs
    device = env.unwrapped.device

    ep_return = torch.zeros(num_envs, device=device)
    ep_length = torch.zeros(num_envs, dtype=torch.long, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    distance = torch.zeros(num_envs, device=device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    upside_down = torch.zeros(num_envs, dtype=torch.bool, device=device)

    obs = env.get_observations()
    start_pos = robot.data.root_pos_w.clone()
    last_pos = start_pos.clone()
    # z of gravity in the body frame: -1 upright, +1 upside down
    last_grav_z = robot.data.projected_gravity_b[:, 2].clone()
    max_steps = env.unwrapped.max_episode_length + 5
    if args_cli.diag_scan:
        terrain = env.unwrapped.scene.terrain
        if getattr(terrain, "terrain_types", None) is not None:
            types = terrain.terrain_types
            from isaaclab_tasks.manager_based.classic.ant.ant_rough3_env_cfg import FLAT_LANES

            flat = torch.isin(types, torch.tensor(FLAT_LANES, device=types.device))
            print(f"[DIAG] lanes: {types.unique().numel()} used, flat-lane share {flat.float().mean().item():.3f}")
    for step in range(max_steps):
        if args_cli.diag_friction and step == 3:
            c = getattr(env.unwrapped, "_robot_friction_cache", None)
            mats = env.unwrapped.scene["robot"].root_physx_view.get_material_properties()[..., 0].mean(dim=1)
            print(f"[DIAG] friction obs cache: {None if c is None else (round(c.min().item(), 3), round(c.max().item(), 3), round(c.std().item(), 3))} "
                  f"| PhysX now: min {mats.min().item():.3f} max {mats.max().item():.3f}", flush=True)
        if args_cli.diag_scan and step in (1, 30):
            from isaaclab_tasks.manager_based.classic.ant.ant_rough3_env_cfg import forward_height_scan

            scan = forward_height_scan(env.unwrapped, SceneEntityCfg("height_scan"))
            hits = env.unwrapped.scene.sensors["height_scan"].data.ray_hits_w[..., 2]
            print(f"[DIAG] step {step}: scan shape {tuple(scan.shape)} mean {scan.mean().item():.3f} "
                  f"min {scan.min().item():.3f} max {scan.max().item():.3f} std {scan.std().item():.3f} | "
                  f"hit z mean {hits.mean().item():.3f} finite {torch.isfinite(hits).float().mean().item():.3f}")
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
        active = ~finished
        ep_return[active] += rewards[active]
        ep_length[active] += 1
        done = dones.bool() & active
        # done envs are already reset: use the pose from the previous step as the final pose
        distance[done] = last_pos[done, 0] - start_pos[done, 0]
        fell[done] = ~extras["time_outs"].bool()[done]
        upside_down[done] = last_grav_z[done] > 0.0
        finished |= done
        last_pos = robot.data.root_pos_w.clone()
        last_grav_z = robot.data.projected_gravity_b[:, 2].clone()
        if finished.all():
            break

    step_dt = env.unwrapped.step_dt
    ret, length, dist = ep_return.cpu(), ep_length.cpu().float(), distance.cpu()
    result = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "terrain": args_cli.terrain,
        "friction": args_cli.friction,
        "terrain_seed": args_cli.terrain_seed,
        "combine_mode": args_cli.combine_mode,
        "hydra_overrides": hydra_args,
        "seed": agent_cfg.seed,
        "diag_height_relative": args_cli.diag_height_relative,
        "diag_world_z_termination": args_cli.diag_world_z_termination,
        "num_episodes": int(finished.sum().item()),
        "return_mean": ret.mean().item(),
        "return_std": ret.std(unbiased=False).item(),
        "return_min": ret.min().item(),
        "return_median": ret.median().item(),
        "episode_length_mean": length.mean().item(),
        "fall_rate": fell.float().mean().item(),
        "upside_down_at_end_rate": upside_down.float().mean().item(),
        "distance_x_mean": dist.mean().item(),
        "distance_x_max": dist.max().item(),
        "speed_x_mean": (dist / (length * step_dt)).mean().item(),
    }
    print("[EVAL] " + json.dumps(result))
    if args_cli.output:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        with open(args_cli.output, "w") as f:
            json.dump(result, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

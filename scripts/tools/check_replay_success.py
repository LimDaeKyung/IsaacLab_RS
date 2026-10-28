# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Replay recorded demos and check that each one succeeds again (success held for 10 steps in a row).

The stock replay_demos.py in this repo only replays; this script also judges success with the
task's own success term and can save an mp4 per demo.

Example:
    ./isaaclab.sh -p scripts/tools/check_replay_success.py \
        --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
        --dataset_file ./datasets/toy2box_auto_30.hdf5 --headless
    # with videos of demos 0 and 5 (needs --enable_cameras):
    ... --video --video_folder ./videos/toy2box_auto --select_episodes 0 5 --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay demos and check success.")
parser.add_argument("--task", type=str, default="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0")
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--select_episodes", type=int, nargs="+", default=[])
parser.add_argument("--num_success_steps", type=int, default=10)
parser.add_argument("--extra_steps", type=int, default=20, help="Idle steps after the last action.")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_folder", type=str, default="./videos/replay_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from isaaclab.utils.datasets import HDF5DatasetFileHandler

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.dataset_file)
    names = sorted(handler.get_episode_names(), key=lambda k: int(k.split("_")[-1]))
    indices = args_cli.select_episodes or list(range(len(names)))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    success_term = env_cfg.terminations.success
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None).unwrapped
    env.reset()
    if args_cli.video:
        import imageio

        os.makedirs(args_cli.video_folder, exist_ok=True)

    results = []
    with torch.inference_mode():
        for idx in indices:
            ep = handler.load_episode(names[idx], env.device)
            env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
            frames = []
            streak, best, last = 0, 0, None
            n = 0
            while True:
                a = ep.get_next_action()
                if a is None:
                    if n >= ep.data["actions"].shape[0] + args_cli.extra_steps:
                        break
                    a = last.clone()
                    a[:6] = 0.0  # keys released, gripper stays as it was
                last = a
                env.step(a.unsqueeze(0))
                n += 1
                if bool(success_term.func(env, **success_term.params)[0]):
                    streak += 1
                    best = max(best, streak)
                else:
                    streak = 0
                if args_cli.video:
                    frames.append(env.render())
            ok = best >= args_cli.num_success_steps
            results.append(ok)
            print(f"  {names[idx]:<8s} replay {'SUCCESS' if ok else 'FAIL'} (longest success streak {best} steps)")
            if args_cli.video and frames:
                path = os.path.join(args_cli.video_folder, f"{names[idx]}.mp4")
                imageio.mimsave(path, frames, fps=20)
                print(f"    video: {path}")

    print(f"Replay success: {sum(results)}/{len(results)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

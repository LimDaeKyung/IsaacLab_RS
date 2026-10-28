# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Record Toy2Box demonstrations with a scripted "virtual keyboard" operator.

The recording pipeline is the same as scripts/tools/record_demos.py (ActionStateRecorder,
EXPORT_SUCCEEDED_ONLY, success = 3 conditions held for --num_success_steps steps). The only
difference is who presses the keys: instead of a human on Se3Keyboard, a scripted operator looks
at the scene and "holds" the same keys a person would (W/S A/D Q/E, Z/X T/G C/V, K toggle, R reset).
Every action therefore has exactly the keyboard format: each axis is -0.05 / 0 / +0.05 and the
gripper is +1 (open) / -1 (closed).

Operator strategy (same order as the course slides):
  approach above truck + turn wrist (C/V) -> lower (E) -> stop -> K close -> lift (Q)
  -> carry over box (W/A ...) -> lower a little (E) -> K open -> wait until saved.
Bad attempts (missed grasp, truck knocked over, too slow) are discarded with R.

Example:
    ./isaaclab.sh -p scripts/tools/record_demos_auto.py \
        --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
        --dataset_file ./datasets/toy2box_auto_30.hdf5 --num_demos 30 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Toy2Box demos with a scripted keyboard operator.")
parser.add_argument("--task", type=str, default="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0")
parser.add_argument("--dataset_file", type=str, default="./datasets/toy2box_auto.hdf5")
parser.add_argument("--num_demos", type=int, default=30, help="Number of successful demos to save.")
parser.add_argument("--num_success_steps", type=int, default=10)
parser.add_argument("--step_hz", type=int, default=0, help="Real-time pacing in Hz (0 = as fast as possible).")
parser.add_argument("--seed", type=int, default=0, help="Seed for the scene randomization and the operator.")
parser.add_argument("--max_attempts", type=int, default=200, help="Give up after this many attempts.")
parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing dataset file.")
parser.add_argument("--closing_yaw", type=float, default=-1.57, help="Preferred finger closing-axis yaw (rad, base frame).")
parser.add_argument("--show_phases", action="store_true", help="Print operator phase changes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import time
import torch
from scipy.spatial.transform import Rotation

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Same sensitivity as the task's Se3KeyboardCfg (pos_sensitivity = rot_sensitivity = 0.05).
SENS = 0.05


class ScriptedKeyboardOperator:
    """Decides which keys to hold at every step, like a person watching the scene.

    The output of advance() is computed with the exact formula of Se3Keyboard.advance(),
    so the recorded actions cannot be told apart from real key presses by their format.
    """

    # Wrist-yaw preference for the finger closing axis (hand local y) in the base frame.
    # With the hand pitched down (T key) it naturally sits near -pi/2; picking the closer of the two
    # equivalent grasps keeps the wrist turn below 90 deg.
    PREFERRED_CLOSING_YAW = -1.57

    def __init__(self, rng: random.Random, verbose: bool = False):
        self.rng = rng
        self.verbose = verbose
        self.reset()

    # ------------------------------------------------------------------ keyboard state
    def reset(self):
        self.close_gripper = False  # Se3Keyboard starts open
        self.delta_pos = [0.0, 0.0, 0.0]
        self.delta_rot = [0.0, 0.0, 0.0]
        self.phase = "approach"
        self.phase_steps = 0
        self.steps = 0
        self.wait = 0
        self.want_reset = False
        self.reason = ""
        self.truck_z0 = None
        # per-episode "operator style" so that the demos are not all identical
        self.hover_dz = self.rng.uniform(0.10, 0.16)  # height above truck before lowering
        self.grasp_dz = self.rng.uniform(-0.012, 0.0)  # gripper center vs truck center
        self.carry_clear = self.rng.uniform(0.05, 0.09)  # truck bottom above box rim while carrying
        self.release_clear = self.rng.uniform(-0.05, -0.02)  # truck bottom vs rim when opening (inside the box)
        self.pos_tol = self.rng.uniform(0.007, 0.011)
        self.yaw_tol = 0.04

    def advance(self) -> torch.Tensor:
        """Same as Se3Keyboard.advance()."""
        rot_vec = Rotation.from_euler("XYZ", self.delta_rot).as_rotvec()
        grip = -1.0 if self.close_gripper else 1.0
        return torch.tensor([*self.delta_pos, *rot_vec, grip], dtype=torch.float32)

    def _release_all(self):
        self.delta_pos = [0.0, 0.0, 0.0]
        self.delta_rot = [0.0, 0.0, 0.0]

    def _press_toward(self, err_pos, tol_pos, err_rot=None, tol_rot=(0.04, 0.08, 0.08)):
        """Hold W/S, A/D, Q/E (and C/V, Z/X, T/G) while the error on that axis is large."""
        for i in range(3):
            self.delta_pos[i] = SENS * (1 if err_pos[i] > 0 else -1) if abs(err_pos[i]) > tol_pos else 0.0
        if err_rot is None:
            self.delta_rot = [0.0, 0.0, 0.0]
        else:
            # index 0 roll (Z/X), 1 pitch (T/G), 2 yaw (C/V); roll/pitch only fixed when clearly tilted
            for i, tol in zip((2, 0, 1), (tol_rot[0], tol_rot[1], tol_rot[2])):
                self.delta_rot[i] = SENS * (1 if err_rot[i] > 0 else -1) if abs(err_rot[i]) > tol else 0.0

    def _goto(self, phase):
        if self.verbose:
            print(f"    [operator] step {self.steps:4d}: {self.phase} -> {phase}")
        self.phase = phase
        self.phase_steps = 0

    def _abort(self, reason):
        self.want_reset = True
        self.reason = reason

    # ------------------------------------------------------------------ perception helpers
    def _target_quat(self, truck_quat, cur_quat):
        """Top-down grasp: fingers point down (hand local z = -Z), closing axis (local y) across the truck."""
        t_yaw = math_utils.euler_xyz_from_quat(truck_quat.unsqueeze(0))[2].item()
        best = None
        for theta in (t_yaw + math.pi / 2, t_yaw - math.pi / 2):  # closing-axis yaw is defined mod pi
            theta = math.atan2(math.sin(theta), math.cos(theta))
            d = abs(math.atan2(math.sin(theta - self.PREFERRED_CLOSING_YAW), math.cos(theta - self.PREFERRED_CLOSING_YAW)))
            if best is None or d < best[0]:
                best = (d, theta)
        theta = best[1]
        c, s = math.cos(theta), math.sin(theta)
        # columns: local x = y cross z, local y = (c, s, 0), local z = (0, 0, -1)
        R = torch.tensor([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]], device=cur_quat.device)
        return math_utils.quat_from_matrix(R.unsqueeze(0))[0]

    @staticmethod
    def _rot_err(q_target, q_cur):
        dq = math_utils.quat_mul(q_target.unsqueeze(0), math_utils.quat_conjugate(q_cur.unsqueeze(0)))
        dq = torch.where(dq[:, :1] < 0, -dq, dq)
        return math_utils.axis_angle_from_quat(dq)[0].tolist()

    # ------------------------------------------------------------------ main decision
    def observe(self, obs: dict):
        """Look at the scene (policy observations, robot base frame) and update the held keys."""
        self.steps += 1
        self.phase_steps += 1
        eef = obs["eef_pos"][0]
        eef_q = obs["eef_quat"][0]
        truck = obs["toy_truck_positions"][0]
        truck_q = obs["toy_truck_orientations"][0]
        box = obs["box_positions"][0]
        if self.truck_z0 is None:
            self.truck_z0 = truck[2].item()

        # truck "up" axis: knocked over?
        truck_up_z = math_utils.matrix_from_quat(truck_q.unsqueeze(0))[0, 2, 2].item()
        if self.phase in ("approach", "descend", "settle", "grasp") and truck_up_z < 0.8:
            return self._abort("truck knocked over")
        if self.steps > 1100:
            return self._abort("too slow")
        if self.phase_steps > 350:
            return self._abort(f"stuck in {self.phase}")

        # short "human" pauses between phases (nothing pressed)
        if self.wait > 0:
            self.wait -= 1
            self._release_all()
            return

        q_tgt = self._target_quat(truck_q, eef_q) if self.phase in ("approach", "descend") else None
        rot_err = self._rot_err(q_tgt, eef_q) if q_tgt is not None else None
        box_top = box[2].item() + 0.12  # rim height (box half height ~0.12 m once settled)
        truck_half_h = 0.032

        if self.phase == "approach":
            tgt = torch.stack([truck[0], truck[1], truck[2] + self.hover_dz])
            err = (tgt - eef).tolist()
            # keep the hand high until roughly above the truck
            if math.hypot(err[0], err[1]) > 0.05 and err[2] < 0:
                err[2] = 0.0
            self._press_toward(err, self.pos_tol, rot_err)
            if math.hypot(err[0], err[1]) < 0.012 and abs(err[2]) < 0.03 and abs(rot_err[2]) < self.yaw_tol and max(abs(rot_err[0]), abs(rot_err[1])) < 0.12:
                self._release_all()
                self.wait = self.rng.randint(2, 8)
                self._goto("descend")
        elif self.phase == "descend":
            tgt = torch.stack([truck[0], truck[1], truck[2] + self.grasp_dz])
            err = (tgt - eef).tolist()
            self._press_toward(err, 0.006, rot_err)
            if abs(err[2]) < 0.008 and math.hypot(err[0], err[1]) < 0.012:
                self._release_all()
                self._goto("settle")
        elif self.phase == "settle":
            # stop completely before closing (slide: "완전히 멈춘 뒤 K")
            self._release_all()
            if self.phase_steps >= self.rng.randint(4, 8):
                self.close_gripper = True  # press K
                self._goto("grasp")
        elif self.phase == "grasp":
            self._release_all()
            if self.phase_steps >= 15:
                self._goto("lift")
        elif self.phase == "lift":
            lift_z = box_top + truck_half_h + self.carry_clear
            err = [0.0, 0.0, lift_z - eef[2].item()]
            self._press_toward(err, self.pos_tol)
            if self.phase_steps == 25 and truck[2].item() < self.truck_z0 + 0.03:
                return self._abort("missed grasp")
            if abs(err[2]) < 0.015:
                self._release_all()
                self.wait = self.rng.randint(0, 5)
                self._goto("carry")
        elif self.phase == "carry":
            lift_z = box_top + truck_half_h + self.carry_clear
            # put the truck (not the hand) over the box center
            off = (truck - eef)[:2]
            tgt_xy = box[:2] - off
            err = [(tgt_xy[0] - eef[0]).item(), (tgt_xy[1] - eef[1]).item(), lift_z - eef[2].item()]
            self._press_toward(err, self.pos_tol)
            if truck[2].item() < self.truck_z0 + 0.03:
                return self._abort("dropped truck")
            if math.hypot(err[0], err[1]) < 0.015:
                self._release_all()
                self._goto("lower")
        elif self.phase == "lower":
            rel_z = box_top + truck_half_h + self.release_clear
            err = [0.0, 0.0, rel_z - eef[2].item()]
            self._press_toward(err, 0.008)
            if abs(err[2]) < 0.01:
                self._release_all()
                self.wait = self.rng.randint(2, 5)
                self._goto("release")
        elif self.phase == "release":
            self._release_all()
            self.close_gripper = False  # press K (open), then keep still until saved
            self._goto("hold")
        elif self.phase == "hold":
            self._release_all()
            if self.phase_steps > 120:
                return self._abort("not in box after release")


def main():
    rng = random.Random(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    if os.path.exists(args_cli.dataset_file) and not args_cli.overwrite:
        raise SystemExit(f"{args_cli.dataset_file} already exists (use --overwrite or a new name)")
    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # ---- environment: identical settings to record_demos.py
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.seed = args_cli.seed
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    operator = ScriptedKeyboardOperator(rng, verbose=args_cli.show_phases)
    operator.PREFERRED_CLOSING_YAW = args_cli.closing_yaw

    def hard_reset():
        env.sim.reset()
        env.recorder_manager.reset()
        env.reset()
        operator.reset()

    env.sim.reset()
    env.reset()
    operator.reset()

    saved, attempts, success_steps = 0, 1, 0
    fail_reasons: dict[str, int] = {}
    t0 = time.time()
    period = 1.0 / args_cli.step_hz if args_cli.step_hz > 0 else 0.0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and saved < args_cli.num_demos:
            tick = time.time()
            operator.observe(env.obs_buf["policy"])
            if operator.want_reset:  # press R: discard this attempt
                fail_reasons[operator.reason] = fail_reasons.get(operator.reason, 0) + 1
                print(f"  attempt {attempts:3d}: discarded ({operator.reason}, {operator.steps} steps)")
                attempts += 1
                if attempts > args_cli.max_attempts:
                    break
                hard_reset()
                success_steps = 0
                continue

            action = operator.advance().to(env.device).repeat(env.num_envs, 1)
            _, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0] | truncated[0]):  # e.g. truck fell off the table -> auto reset
                fail_reasons["truck fell off table"] = fail_reasons.get("truck fell off table", 0) + 1
                print(f"  attempt {attempts:3d}: discarded (truck fell off table)")
                attempts += 1
                operator.reset()
                success_steps = 0
                continue

            # success check, exactly as record_demos.process_success_condition
            if bool(success_term.func(env, **success_term.params)[0]):
                success_steps += 1
                if success_steps >= args_cli.num_success_steps:
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    saved = env.recorder_manager.exported_successful_episode_count
                    print(
                        f"  attempt {attempts:3d}: SAVED demo {saved}/{args_cli.num_demos} "
                        f"({operator.steps} steps = {operator.steps / 20:.1f}s)"
                    )
                    attempts += 1
                    if saved >= args_cli.num_demos:
                        break
                    hard_reset()
                    success_steps = 0
            else:
                success_steps = 0

            if period > 0:
                time.sleep(max(0.0, period - (time.time() - tick)))

    n_att = attempts - 1
    print("=" * 60)
    print(f"Saved {saved} demos to {args_cli.dataset_file} from {n_att} attempts in {time.time() - t0:.0f}s")
    print(f"Discarded: {fail_reasons}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch 11 (share-fb, 2026-10-01) PPO variants on top of AntPPORunnerCfg. entropy_coef 0.005 as in E15/F3a.

- AntPPOLSTMRunnerCfg: recurrent actor-critic (LSTM 256, one layer) in front of the same MLP head
- AntPPOSymRunnerCfg: left-right mirror data augmentation (ant_symmetry.compute_symmetric_states)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg, RslRlSymmetryCfg

from .rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntPPOLSTMRunnerCfg(AntPPORunnerCfg):
    def __post_init__(self):
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[400, 200, 100],
            critic_hidden_dims=[400, 200, 100],
            activation="elu",
            rnn_type="lstm",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        )
        self.algorithm.entropy_coef = 0.005


@configclass
class AntPPOSymRunnerCfg(AntPPORunnerCfg):
    def __post_init__(self):
        self.algorithm.entropy_coef = 0.005
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func="isaaclab_tasks.manager_based.classic.ant.ant_symmetry:compute_symmetric_states",
        )

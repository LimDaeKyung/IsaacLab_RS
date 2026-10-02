# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the Ant (batch 11, share-fb, 2026-10-01).

Only the left-right mirror (y -> -y) is valid: the target is fixed at +x, so front-back or 90 degree symmetries
would turn the task into walking in another direction.

The mirror is a signed permutation of the policy observation and of the actions. It is not hand-derived: the
permutation pairs left/right joints and feet by name, and the signs are measured in simulation by
``experiments/check_symmetry.py`` (mirrored initial states + mirrored actions must give mirrored observations).
That script writes ``ant_symmetry_map.json`` next to this file; this module only applies it.
"""

from __future__ import annotations

import json
import os

import torch

_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ant_symmetry_map.json")
_CACHE: dict = {}


def _load(device):
    key = str(device)
    if key not in _CACHE:
        with open(_MAP_FILE) as f:
            m = json.load(f)
        _CACHE[key] = {
            "obs_matrix": torch.tensor(m["obs_matrix"], dtype=torch.float32, device=device) if "obs_matrix" in m else None,
            "obs_perm": torch.tensor(m["obs_perm"], dtype=torch.long, device=device),
            "obs_sign": torch.tensor(m["obs_sign"], dtype=torch.float32, device=device),
            "act_perm": torch.tensor(m["act_perm"], dtype=torch.long, device=device),
            "act_sign": torch.tensor(m["act_sign"], dtype=torch.float32, device=device),
        }
    return _CACHE[key]


def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    m = _load(obs.device)
    if m["obs_matrix"] is not None:  # signed permutation plus fitted 6x6 blocks for the foot wrenches
        return obs @ m["obs_matrix"]
    return obs[:, m["obs_perm"]] * m["obs_sign"]


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    m = _load(actions.device)
    return actions[:, m["act_perm"]] * m["act_sign"]


def compute_symmetric_states(env=None, obs=None, actions=None, **kwargs):
    """RSL-RL data-augmentation hook: returns (original + left-right mirror) observations and actions."""
    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs_aug.keys():
            obs_aug[group][:n] = obs[group][:]
            obs_aug[group][n:] = mirror_obs(obs[group])
    act_aug = None
    if actions is not None:
        act_aug = torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs_aug, act_aug

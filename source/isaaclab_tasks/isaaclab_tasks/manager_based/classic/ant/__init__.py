# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ant-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_eval_env_cfg:AntEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

# assignment 1: unseen-terrain experiments (see ant_rough_env_cfg.py)
gym.register(
    id="Isaac-Ant-E0-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntE0PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)
for _exp in ("E1", "E2", "E3", "E4", "E4b"):
    gym.register(
        id=f"Isaac-Ant-Rough-{_exp}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRough{_exp}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )
    gym.register(
        id=f"Isaac-Ant-Rough-{_exp}-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRough{_exp}PlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# overnight variants on top of E4, e.g. Isaac-Ant-Rough-E4-FlatHist-v0 (see ant_rough_env_cfg.VARIANT_COMBOS)
from .ant_rough_env_cfg import VARIANT_COMBOS  # noqa: E402

for _flags in VARIANT_COMBOS:
    _suffix = "".join(_flags)
    for _play in (False, True):
        gym.register(
            id=f"Isaac-Ant-Rough-E4-{_suffix}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    f"{__name__}.ant_rough_env_cfg:AntRoughE4{_suffix}{'Play' if _play else ''}EnvCfg"
                ),
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
            },
        )

# round-2 variants (ant_rough2_env_cfg.py), e.g. Isaac-Ant-R2-SafeAir-v0; config classes are created lazily
_R2_FLAGS = ("Flat", "Hist", "Push", "Safe", "LowFric", "Air", "NoCurr", "Oracle")
for _mask in range(1, 2 ** len(_R2_FLAGS)):
    _suffix = "".join(f for i, f in enumerate(_R2_FLAGS) if _mask & (1 << i))
    for _play in (False, True):
        gym.register(
            id=f"Isaac-Ant-R2-{_suffix}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.ant_rough2_env_cfg:AntR2{_suffix}{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
            },
        )

# round 3 (ant_rough3_env_cfg.py): C1 and the 2×2 flat-speed study, e.g. Isaac-Ant-R3-C1EyesFlatLanes-v0
for _name in ("C1", "C1Eyes", "C1FlatLanes", "C1EyesFlatLanes"):
    for _play in (False, True):
        gym.register(
            id=f"Isaac-Ant-R3-{_name}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.ant_rough3_env_cfg:AntR3{_name}{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
            },
        )

# round 4 (ant_rough4_env_cfg.py): e.g. Isaac-Ant-R4-Air-v0, Isaac-Ant-R4-FallRamp-v0, Isaac-Ant-R4-Asym-v0
for _name in ("Air", "FallRamp", "FallConst", "Asym", "Push", "EyesNoise"):
    for _play in (False, True):
        gym.register(
            id=f"Isaac-Ant-R4-{_name}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.ant_rough4_env_cfg:AntR4{_name}{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
            },
        )

# T1 teacher check (ant_rough4_env_cfg.py, variant "Teacher")
for _play in (False, True):
    gym.register(
        id=f"Isaac-Ant-R4-Teacher{'-Play' if _play else ''}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough4_env_cfg:AntR4Teacher{'Play' if _play else ''}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# F2 box-height variety (ant_rough4_env_cfg.py, variant "OracleVar")
for _play in (False, True):
    gym.register(
        id=f"Isaac-Ant-R4-OracleVar{'-Play' if _play else ''}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough4_env_cfg:AntR4OracleVar{'Play' if _play else ''}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# batch 10 (prepared 2026-10-01, share-fb): train on every evaluation terrain type, see ant_rough5_env_cfg.py
for _play in (False, True):
    gym.register(
        id=f"Isaac-Ant-R5-AllMix{'-Play' if _play else ''}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough5_env_cfg:AntR5AllMix{'Play' if _play else ''}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# batch 11 (share-fb, 2026-10-01): E15 environment (R2 Oracle) with an LSTM policy or left-right symmetry augmentation
for _name, _agent in (("OracleLSTM", "AntPPOLSTMRunnerCfg"), ("OracleSym", "AntPPOSymRunnerCfg")):
    for _play in (False, True):
        gym.register(
            id=f"Isaac-Ant-R6-{_name}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.ant_rough2_env_cfg:AntR2Oracle{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_extra_cfg:{_agent}",
            },
        )

# batch 12 (share-fb, 2026-10-01): AllMix + low friction, AllMix + left-right symmetry augmentation
for _id, _env, _agent in (
    ("R7-AllMixLowFric", "ant_rough5_env_cfg:AntR5AllMixLowFric", "rsl_rl_ppo_cfg:AntPPORunnerCfg"),
    ("R7-AllMixSym", "ant_rough5_env_cfg:AntR5AllMix", "rsl_rl_ppo_extra_cfg:AntPPOSymRunnerCfg"),
):
    for _play in (False, True):
        _env_cls = _env if not _play else "ant_rough5_env_cfg:AntR5AllMix"  # Play is the AllMix Play config
        gym.register(
            id=f"Isaac-Ant-{_id}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{_env_cls}{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.{_agent}",
            },
        )

# batch 13 (share-fb, 2026-10-02): AllMix + moderate / mild low friction (Play = AllMix Play)
for _name in ("MF02", "MF04"):
    for _play in (False, True):
        gym.register(
            id=f"Isaac-Ant-R8-AllMix{_name}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.ant_rough5_env_cfg:AntR5AllMix{_name if not _play else ''}{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
            },
        )

# batch 14 (share-fb, 2026-10-02): AllMix with per-tile random difficulty, alone and with mild low friction
for _name in ("Wide", "WideMF04", "WideMF02", "WideMF03", "HardMF04", "HardMF02", "WideMF04FP20", "WideMF04FP50", "WideMF04FP30",
              "WideMF03FP20", "HardMF04FP20", "WideMF03FP30", "HardMF04FP30", "HardMF03FP20", "WideMF04FP10", "WideMF04FP35"):  # batch 15-22
    for _play in (False, True):
        gym.register(
            id=f"Isaac-Ant-R9-AllMix{_name}{'-Play' if _play else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.ant_rough5_env_cfg:AntR5AllMix{_name if not _play else ''}{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
            },
        )

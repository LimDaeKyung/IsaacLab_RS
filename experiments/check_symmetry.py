"""Measure and verify the Ant left-right mirror in simulation (batch 11, share-fb).

Half of the envs start from random joint states and get random actions; the other half start from the mirrored
joint states and get the mirrored actions. If the mirror is right, the second half's observations equal the mirrored
observations of the first half. Joint signs (hips, feet) are searched over the 4 combinations; observation signs are
then read from the data. Writes source/.../classic/ant/ant_symmetry_map.json only if the check passes.

usage: ./isaaclab.sh -p experiments/check_symmetry.py   (headless)
"""

import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import itertools  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.manager_based.classic.ant.ant_eval_env_cfg import apply_eval_terrain  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Ant-R2-Oracle-Play-v0"
N = int(os.environ.get("SYM_N", 64))
T = int(os.environ.get("SYM_T", 8))  # short horizon: contact dynamics amplify tiny solver differences over time
DEVICE = os.environ.get("SYM_DEVICE", "cuda:0")
ACT = float(os.environ.get("SYM_ACT", 0.3))
OUT = os.path.join(os.path.dirname(isaaclab_tasks.__file__), "manager_based/classic/ant/ant_symmetry_map.json")


def swap_lr(name):
    return name.replace("left", "@").replace("right", "left").replace("@", "right")


def main():
    cfg = parse_env_cfg(TASK, device=DEVICE, num_envs=N)
    apply_eval_terrain(cfg, "flat")
    env = gym.make(TASK, cfg=cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    jn = list(robot.joint_names)
    # Pair legs by geometry, not by name: Ant legs sit on the diagonals, so the name swap left<->right is a
    # front-back (x) mirror. The valid mirror for a +x target is y -> -y. Read foot positions in the root frame.
    env.reset()
    feet_all = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
    bids = [robot.body_names.index(f) for f in feet_all]
    rel = (robot.data.body_pos_w[0, bids] - robot.data.root_pos_w[0]).cpu()
    leg_of = {}
    for i, f in enumerate(feet_all):
        target = rel[i].clone()
        target[1] = -target[1]
        j = int(((rel[:, :2] - target[:2]).norm(dim=1)).argmin())
        leg_of[f.replace("_foot", "")] = feet_all[j].replace("_foot", "")
    print("[SYM] foot positions (root frame)", {f: [round(x, 3) for x in rel[i, :2].tolist()] for i, f in enumerate(feet_all)},
          "y-mirror partners", leg_of, flush=True)

    def partner(name):
        for leg, other in leg_of.items():
            if name.startswith(leg + "_"):
                return name.replace(leg, other, 1)
        return name

    jperm = [jn.index(partner(n)) for n in jn]
    om = uw.observation_manager
    terms, dims = om.active_terms["policy"], om.group_obs_term_dim["policy"]
    print("[SYM] joints", jn, "perm", jperm, flush=True)
    print("[SYM] obs terms", list(zip(terms, dims)), flush=True)

    # observation permutation (partner index) by term semantics
    perm, start = [], 0
    feet_order = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
    for name, d in zip(terms, dims):
        size = d[0]
        if size == len(jn):  # per-joint terms: joint_pos_norm, joint_vel_rel, actions
            perm += [start + p for p in jperm]
        elif name == "feet_body_forces":
            per = size // len(feet_order)
            fperm = [feet_order.index(partner(f)) for f in feet_order]
            perm += [start + fperm[i // per] * per + i % per for i in range(size)]
        else:
            perm += list(range(start, start + size))
        start += size
    perm = torch.tensor(perm, device=uw.device)
    jperm_t = torch.tensor(jperm, device=uw.device)
    half = N // 2

    pair_a = ("front_left", leg_of["front_left"])

    fstart = sum(d[0] for name, d in zip(terms, dims) if terms.index(name) < terms.index("feet_body_forces"))
    fsize = dims[terms.index("feet_body_forces")][0]
    per = fsize // 4
    D = sum(d[0] for d in dims)
    force = torch.zeros(D, dtype=torch.bool, device=uw.device)
    force[fstart:fstart + fsize] = True

    def trial(hip_a, foot_a, hip_b, foot_b):
        def sgn(n):
            in_a = any(n.startswith(p + "_") for p in pair_a)
            return (hip_a if in_a else hip_b) if "_leg" in n else (foot_a if in_a else foot_b)
        jsign = torch.tensor([sgn(n) for n in jn], dtype=torch.float32, device=uw.device)
        torch.manual_seed(0)
        env.reset()
        q0 = robot.data.default_joint_pos.clone()
        qa = q0[:half] + 0.3 * (torch.rand_like(q0[:half]) - 0.5)
        q = torch.cat([qa, qa[:, jperm_t] * jsign])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        # mirrored robots stand at the mirrored world position (x, -y) so that the target direction is mirrored too
        root = robot.data.default_root_state.clone()
        root[:, :3] += uw.scene.env_origins
        root[half:, 0] = root[:half, 0]
        root[half:, 1] = -root[:half, 1]
        root[half:, 2] = root[:half, 2]
        robot.write_root_state_to_sim(root)
        xs, ys = [], []
        a_prev = torch.zeros(half, len(jn), device=uw.device)
        for _ in range(T):
            a_prev = 0.7 * a_prev + 0.3 * torch.randn(half, len(jn), device=uw.device) * ACT
            a = torch.cat([a_prev, a_prev[:, jperm_t] * jsign])
            obs, _, term, trunc, _ = env.step(a)
            o = obs["policy"]
            if (term | trunc).any():
                break
            xs.append(o[:half])
            ys.append(o[half:])
        x, y = torch.cat(xs), torch.cat(ys)
        xp = x[:, perm]
        x0, y0 = xp - xp.mean(0), y - y.mean(0)
        corr = (x0 * y0).sum(0) / (x0.norm(dim=0) * y0.norm(dim=0) + 1e-9)
        sign = torch.where(corr >= 0, 1.0, -1.0)
        mt = torch.zeros(D, D, device=uw.device)
        for d in range(D):
            if not force[d]:
                mt[perm[d], d] = sign[d]
        # foot wrenches live in rotated foot frames: fit a 6x6 linear map per foot pair (least squares)
        for f in range(4):
            cols = list(range(fstart + f * per, fstart + (f + 1) * per))
            src = [int(perm[c]) for c in cols]
            w = torch.linalg.lstsq(x[:, src].cpu(), y[:, cols].cpu()).solution.to(uw.device)
            for i, r in enumerate(src):
                mt[r, cols] = w[i]
        pred = x @ mt
        active = y.std(0) > 1e-4
        resid = ((y - pred).pow(2).mean(0).sqrt() / (y.pow(2).mean(0).sqrt() + 1e-6))
        score = resid[active].mean().item()
        return score, (sign, mt), resid, active, jsign, len(xs)

    best = None
    for combo in itertools.product((1.0, -1.0), repeat=4):
        score, sign, resid, active, jsign, steps = trial(*combo)
        print(f"[SYM] signs (hipA, footA, hipB, footB) {combo}: mean relative residual {score:.4f} over {steps} steps", flush=True)
        if best is None or score < best[0]:
            best = (score, sign, resid, active, jsign, combo, None)
    score, sign, resid, active, jsign, hip_s, foot_s = best
    worst = resid[active].max().item()
    start = 0
    for name, d in zip(terms, dims):
        r = resid[start:start + d[0]]
        a = active[start:start + d[0]]
        print(f"[SYM] term {name:24s} max resid {r[a].max().item() if a.any() else 0:.4f}  signs {sign[0][start:start + d[0]].int().tolist()}", flush=True)
        start += d[0]
    ok = score < 0.05 and worst < 0.2
    approx = os.environ.get("SYM_ACCEPT_APPROX") == "1" and not ok
    if approx:
        print(f"[SYM] writing APPROXIMATE map (mean {score:.4f}, worst {worst:.4f}) because SYM_ACCEPT_APPROX=1", flush=True)
    ok = ok or approx
    print(f"[SYM] best signs {hip_s}: mean {score:.4f}, worst {worst:.4f} → {'PASS' if ok else 'FAIL'}", flush=True)
    if ok:
        with open(OUT, "w") as f:
            sign, mt = sign
            json.dump({"obs_matrix": mt.cpu().tolist(), "obs_perm": perm.tolist(), "obs_sign": sign.tolist(), "act_perm": jperm,
                       "act_sign": jsign.tolist(), "joint_names": jn, "obs_terms": terms,
                       "check": {"mean_resid": score, "worst_resid": worst, "signs_hipA_footA_hipB_footB": list(hip_s),
                                 "steps": T, "device": DEVICE, "approximate": approx}},
                      f, indent=1)
        print(f"[SYM] wrote {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()

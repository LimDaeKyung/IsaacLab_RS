"""T1 teacher check (EXPERIMENTS.md §12, pre-registered 2026-09-30 20:19). Runs after queue_e22_entropy.py."""
import os, subprocess, sys, time, statistics as st
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight4 as o4  # noqa: E402

ENT = ["agent.algorithm.entropy_coef=0.005"]
TASK, PLAY = "Isaac-Ant-R4-Teacher-v0", "Isaac-Ant-R4-Teacher-Play-v0"
LIMIT = datetime(2026, 10, 1, 8, 25)
WAIT_FOR = ("experiments/overnight4.py", "experiments/queue_dt_check.py", "experiments/queue_e21_energy.py",
            "experiments/queue_e22_entropy.py")


def busy():
    return any(subprocess.run(["pgrep", "-f", p], stdout=subprocess.DEVNULL).returncode == 0 for p in WAIT_FOR)


while busy():
    time.sleep(30)
if datetime.now() + timedelta(minutes=2 * o4.EST_RUN + 5) > LIMIT:
    o.log("T1 not run: not enough time before 08:25 (시간 부족으로 미실행)")
    sys.exit(0)
o.log("## T1 teacher check start (E15 + privileged actor inputs: scan 45 + robot friction)")
ok, d = o4.smoke("t1_teacher", TASK, ENT)
if d:
    os.rename(d, f"logs/rsl_rl/_smoke/{os.path.basename(d)}")
fric = [l.strip() for l in open("experiments/train_logs/smoke_t1_teacher.log") if "[T1]" in l]
o.log(f"T1 smoke: {'OK' if ok else 'FAILED'} | {fric[-1] if fric else 'friction obs line not found'}")
if not ok:
    o.log("T1 not run: smoke test failed")
    sys.exit(0)
res = {}
for seed in (42, 43):
    res[seed] = o4.train_eval("T1", TASK, PLAY, f"t1_teacher_s{seed}", seed, ENT)
if all(res.values()):
    ref = [o4.full_metrics(f"e15_s{s}") for s in (42, 43)]
    avg = lambda ms, k: st.mean(m[k] for m in ms)  # noqa: E731
    t = list(res.values())
    ho, e15ho = avg(t, "held_out"), avg(ref, "held_out")
    met = ho >= e15ho + 3.0
    o.log(f"T1 vs E15 (2-seed): held-out {ho:.1f} vs {e15ho:.1f} (need ≥ {e15ho + 3:.1f}), held-out fall "
          f"{100 * avg(t, 'ho_fall'):.0f}% vs {100 * avg(ref, 'ho_fall'):.0f}%, flat {avg(t, 'flat'):.1f} vs {avg(ref, 'flat'):.1f}, "
          f"flat speed {avg(t, 'flat_speed'):.2f} vs {avg(ref, 'flat_speed'):.2f} → "
          + ("선생 전제 충족: 선생-학생을 진행할 가치 있음" if met else
             "선생 전제 불충족: 특권 정보를 줘도 E15보다 나아지지 않으므로 선생-학생 효과 기대 낮음"))
    o.record({"T1": "verdict", "held_out": ho, "e15": e15ho, "met": met})
else:
    o.log("T1: incomplete runs → no verdict")
o.log("## T1 done")

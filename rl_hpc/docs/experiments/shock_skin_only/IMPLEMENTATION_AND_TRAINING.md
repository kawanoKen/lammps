# Shock/NEMD continuous-skin-only online study

Status: paired training collection **completed** on 2026-10-04 at 22:57 JST.
Idle and CPU-jitter studies both finished 36 episodes and 4,320 transitions
(8,640 total). The service exited successfully after 19 h 48 min. On
2026-10-04 at 03:09 JST the initial
temporary service was replaced by the linked, enabled
`lammps-shock-skin-continuous-resilient-v1.service` with `Restart=on-failure`,
`RestartSec=30s` and a maximum of three rapid restart attempts per hour.
User linger is enabled, so the user service survives logout. If the host
reboots, systemd starts the service again and the script resumes from the
last fully completed episode. The interrupted first attempt was preserved
and excluded from replay. After all 36+36 episodes complete, a later service
start checks progress and exits without recollecting data. This study does not
reuse the previous full-action checkpoints. It retrains separately with no
deliberate contention and with controlled CPU contention, using the same
learning budget in both conditions.

The completed-data audit found 36 complete statuses per condition, zero
dangerous builds, zero invalid/nonfinite rewards, and all executed skins
inside `[0.25, 1.0]`. The CPU co-runner was active in 2,155 of its 4,320
measured intervals. Each learner made 145 model updates. Among greedy
decisions, mean selected skin was 0.695 (idle-trained) and 0.653
(CPU-jitter-trained); about 5.1% and 8.8%, respectively, were within 0.005
of an action bound. These are training-action summaries, not live policy
comparisons.

Mean total reward over the final four training episodes was -936.5 s (idle)
and -1003.2 s (CPU jitter). The initial four exploration episodes averaged
-950.8 s and -1013.0 s respectively. Because exploration/action choices
and external schedules differ, these curves do **not** establish a learning
speedup or a useful load-responsive skin policy. The next required test is
paired, frozen-policy evaluation on the two held-out physical restarts,
with the same contention schedules and a training-selected fixed-skin
baseline.

## Controlled formulation

- Workload: 491,520-atom shock/NEMD, MPI 32, one thread/rank, original CPU
  SHOCK build and the same four physical training restarts as the preceding
  online study. Two further physical restarts remain held out for live
  evaluation.
- Absolute action: **real-valued** neighbour skin in `[0.25, 1.0]`, not a
  five-value action grid. `neigh_modify every 1 delay 0 check yes` is fixed.
  `balance 1.0 shift x 10 1.0 weight neigh 1.5` executes at every decision.
  There is no factor or skip-balance action. The exact current skin is also
  a candidate, allowing no change. These bounds include the previously
  tested safe *grid*; they are not a mathematical safety proof for all real
  values. Each transition aborts on LAMMPS error, nonfinite thermo, dangerous
  build, atom-count change, or unexpected step advance.
- State: temperature, pressure, atom imbalance, current skin, prior segment
  Pair/Neigh/Comm/Modify times and neighbour-build rate, rank nlocal and
  neighbour-count imbalance, and current domain-width minimum/maximum/std.
  No decision index, MD step, shock-front position, CPU-load label, host
  load, or future data enters the policy features. Shock position remains
  only in the safety guard and analysis log.
- Reward: negative seconds for applying the action, balancing, and running
  the next 500 MD steps. Initialization/warm-up and co-runner switching are
  excluded, as in the previous study.
- Experience: 36 episodes per condition, with 12 exploration and 24 online
  FQI episodes; 120 decisions/episode; four training restarts in the same
  order; same seed and learner hyperparameters in the two conditions.
  The CPU co-runner shares cores 0--7 with eight MPI ranks and switches
  on/off in 2--5-decision blocks. Its label is logged but hidden from policy.
- Exploration samples across 15 equal skin strata (with 20% explicit
  unchanged-skin actions). Greedy action selection maximizes the learned
  piecewise-quadratic Q response over the continuous interval, comparing
  interior vertices, endpoints, and the unchanged-skin action. FQI uses a
  41-point integration grid for next-state backup, but the **executed greedy
  skin is not restricted to that grid**.

The small two-episode/four-decision smoke completed separately in idle and
CPU-jitter modes with no dangerous builds or crashes. It validates the
mechanics, not the safety of every skin/state combination or a speedup.

## Collection and later evaluation

The long run is sequential: idle training finishes before CPU-jitter
training starts. The runner resumes completed episodes and preserves
incomplete attempts. Generated logs/checkpoints stay in ignored data folders.

```bash
python3 rl_hpc/characterization/shock/run_skin_study.py
systemctl --user status lammps-shock-skin-continuous-resilient-v1.service
journalctl --user -u lammps-shock-skin-continuous-resilient-v1.service -f
```

Outputs:

```text
rl_hpc/characterization/data/shock_skin_continuous_idle_v1/
rl_hpc/characterization/data/shock_skin_continuous_cpu_jitter_v1/
```

Training reward is not an evaluation. After both policies freeze, they
must be tested live on the same two held-out restarts and paired contention
schedules, versus a training-selected fixed-skin baseline and the previous
full-action policy. A no-step phase-schedule control is also needed before
claiming long-horizon RL adds value.

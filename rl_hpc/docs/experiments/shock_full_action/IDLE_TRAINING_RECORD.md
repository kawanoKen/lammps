# Shock/NEMD online-learning run (no deliberate jitter)

Status: **main jitter-free learning collection completed successfully** on
2026-10-02 at 10:23 JST as user service
`lammps-shock-online-idle-v1.service`, writing to ignored
`rl_hpc/characterization/data/shock_online_idle_main_v1/`. It was started
with 36 episodes (12 stratified exploration, 24 online learning), 120
decisions per episode, from four training restarts. Two additional independent
thermal-velocity restarts are reserved for later evaluation and are not used
for training. The final audit found 36/36 complete episodes, 4,320/4,320
valid 500-step transitions, 145 model updates, and zero dangerous builds.
The revised integration smoke passed. The earlier long pilot completed
one full 120-decision exploration episode; its second episode was interrupted
after 36 decisions when the interactive session ended. Those partial data are
preserved but are **not** counted as a completed learning episode. The main
study is intended to run as a user-level background service so it survives
interactive-session closure. **No policy speedup claim is made before paired
frozen-policy live evaluation.**

## Environment

* CPU SHOCK build `build_shock_char/lmp`; MPI 32, OpenMP 1.
* Shock/NEMD with `nx=480, ny=nz=16` (491,520 atoms). The physical restart
  begins at step 1,200. Each episode has 50 warm-up + 50 first-observation
  steps, then 120 decisions × 500 measured steps, ending at step 61,300.
* No deliberate CPU or GPU co-runner. Natural host load is not guaranteed to
  be zero; the current runner logs host load and available memory before each
  action. The pilot launched before this field was added does not have it.
* Absolute action grid: skin `{0.25,0.35,0.50,0.70,1.00}` × neighbor-weight
  factor `{0.50,0.65,0.80,1.00,1.20,1.50}`, plus skip-balance at each skin:
  **35 actions**. A skipped balance keeps the existing decomposition. The
  neighbor policy remains `every 1 delay 0 check yes`.
* Reward `r_t = -T_t`, where `T_t` includes skin/neighbor preparation, the
  optional balance, and the next 500-step run. Startup, initial warm-up and
  first observation are outside reward. No reward penalty for action changes,
  energy, or intervention is added.

## Learner and exploration

`online_learn.py` implements an experimental low-dimensional **online fitted-Q
iteration** controller (`gamma=0.95`, ridge 100, five fitted-Q passes per
update, replay updated every 20 transitions). A shared quadratic action
surface plus state/action interactions allows numerical actions to share data.
Rewards are divided by 10 only inside training; logs and comparisons remain
seconds. Q bootstrap is clipped to `[-50,0]` in scaled units as a numerical
guard, not a certified return bound. A prior offline FQI prototype showed
out-of-support Q instability; inspect Q scales and live results before using
this as evidence for RL superiority.

The first 12 episodes of the planned main run are an **explicit exploration
phase**. Each 120-decision exploration episode cycles through randomized
permutations of all 35 actions, so all actions appear multiple times. The
next 24 episodes update the model online and use epsilon-greedy behavior,
linearly decreasing exploration from 0.25 to 0.05. Each episode resets the
LAMMPS process. The main run should cycle among independent thermal-velocity
restarts; a single-restart pilot only validates mechanics and is not a
generalization test.

The 12+24 episodes are an initial matched experience budget for the later
idle-vs-CPU-load comparison, **not an eight-hour wall-clock limit**. If
coverage or learning curves are inadequate, extend both conditions by equal
episode blocks. Policy parameters update only after measured transitions.

Monitor the detached run and audit reward progression with:

```bash
systemctl --user status lammps-shock-online-idle-v1.service
journalctl --user -u lammps-shock-online-idle-v1.service -f
python3 rl_hpc/characterization/shock/analyze_online.py \
  rl_hpc/characterization/data/shock_online_idle_main_v1
```

The analyzer can run during collection: it marks the current episode partial
and writes per-decision reward, cumulative reward, rolling-ten reward,
episode summaries, and update traces. A failed/interrupted episode is not
replayed as completed experience; `online_learn.py --resume` preserves the
partial attempt and reconstructs replay from complete episodes. Do not start
a second MPI32 experiment while a training/evaluation service is active,
because it would contaminate its runtime reward.

The first exploration episode took 1,131.95 s of measured action/segment
time and is unusually slow relative to subsequent exploration episodes
(mostly 852--885 s). The final learning episode took 792.17 s. This is a
descriptive reward trajectory, **not a causal speedup estimate**: action
distribution, shock phase, physical seed, and natural host load may differ.
The [PNG reward plot](assets/shock_online_reward.png) shows both episode totals and the
within-episode ten-decision rolling reward. The lower panel averages the
trailing-ten reward curves of episodes 8--11 and 32--35 separately. These
four-episode groups use the same set of four physical restarts, but are not
paired live policy evaluations. The reward becomes more negative as the shock
evolves; the blue/brown separation alone is not a causal learning effect.
Rebuild it from the audited CSV:

```bash
python3 rl_hpc/characterization/shock/plot_online_reward.py \
  rl_hpc/characterization/data/shock_online_idle_main_v1 \
  --output rl_hpc/docs/experiments/shock_full_action/assets/shock_online_reward.svg
ffmpeg -hide_banner -loglevel error -y \
  -i rl_hpc/docs/experiments/shock_full_action/assets/shock_online_reward.svg -frames:v 1 \
  rl_hpc/docs/experiments/shock_full_action/assets/shock_online_reward.png
```

The [PNG action heatmap](assets/shock_online_actions.png) shows the *selected* skin
and `weight neigh` factor at every decision of learning episodes 12--35.
Gray cells are epsilon-random actions, not greedy policy choices. In the
last four episodes, the 452 greedy decisions choose skin 0.5 throughout the
first 80 decisions; later, skin 0.7 appears in 25/150 greedy decisions.
The neighbor factor is mostly 1.2 early, rises toward 1.5 around decisions
60--79, and is 0.5 in all 74 greedy decisions at 100--119. Greedy choices
never skip balancing. This is a learned *phase-dependent pattern*, but the
model includes both decision index and shock position, so the plot alone
cannot distinguish a reactive policy from a time/phase schedule.

```bash
python3 rl_hpc/characterization/shock/plot_online_actions.py \
  rl_hpc/characterization/data/shock_online_idle_main_v1 \
  --output rl_hpc/docs/experiments/shock_full_action/assets/shock_online_actions.svg
ffmpeg -hide_banner -loglevel error -y \
  -i rl_hpc/docs/experiments/shock_full_action/assets/shock_online_actions.svg -frames:v 1 \
  rl_hpc/docs/experiments/shock_full_action/assets/shock_online_actions.png
```

The next required check is paired live evaluation of a frozen learned policy
against strong fixed-action and heuristic controls from identical held-out
restarts. No RL superiority claim is made from the training trace alone.

## Logging and audit

Each episode directory stores `transitions.jsonl`, the complete LAMMPS log,
and one text file per segment. Every transition records:

* pre-action state; chosen action and selection source; full pre-action Q
  vector; epsilon and update count;
* reward, cumulative reward, rolling ten-decision reward, total/action/
  balance/run time breakdown;
* next state including thermo, shock position, rank load statistics and
  Pair/Neigh/Comm/Modify timings; neighbor and dangerous build counts;
* timestamps, physical and behavior seeds, process/build provenance, and
  (for main runs) host load/memory before the decision.

`model_updates.jsonl` records replay size, TD loss, logged Q range,
coefficient norm and update wall time. `progress.json` and
`latest_policy.npz` are written after complete episodes. All large data are
under ignored `rl_hpc/characterization/data/`.

The analyzer validates every transition and writes `reward_trace.csv`,
`episode_reward_summary.csv`, `reward_by_decision.csv` and `analysis.json`:

```bash
python3 rl_hpc/characterization/shock/analyze_online.py \
  rl_hpc/characterization/data/shock_online_long_pilot_v1
```

Raw reward normally gets more negative as the shock changes the workload;
**that trend alone is not a learning curve**. A valid performance conclusion
requires frozen-policy and strong fixed-action runs from paired restarts,
with matching decision horizons and no online updates during evaluation.

## Safety and scope

The original short integration smoke completed 2 episodes × 4 decisions,
including online model updates. A revised 2-episode × 2-decision smoke also
passed after adding host-load logging and partition-width features. The
completed long exploration episode screened all 35 actions across one
trajectory with zero dangerous builds. The interrupted learning episode is
not evidence for or against speedup.
No guarantee of scientific equivalence is inferred from zero dangerous builds
alone. The pilot/restart setup omits the original CVCF/history diagnostics in
the continuation, consistently with the earlier counterfactual experiments.

Factor values above 1.5 and skin outside 0.25–1.0 are **not in this first
learner** because their large-system safety has not been preflighted. The
learning code and action grid are isolated from the existing CPU/GPU builds.

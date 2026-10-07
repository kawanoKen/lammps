# Shock/NEMD online-learning study: matched-sample design

Status: **length/timing pilot complete**, action-range safety and online
training still pending; not a completed training/evaluation result.
The earlier 50-decision trajectory study is not evidence that an online policy
learns or improves runtime. This protocol keeps CPU-load-free and CPU-loaded
training comparable by *experience count*, not by elapsed time. Eight hours is
not a stop criterion for one condition.

## Workload and episode

Use the CPU SHOCK-enabled `build_shock_char/lmp`, MPI 32, one thread per rank.
Keep shock/NEMD physical parameters unchanged and lengthen only x: candidate
`nx=480, ny=nz=16` (491,520 atoms). Generate a separate restart at step 1,200
(the bundled setup advances 100 steps before the requested 1,100-step run);
do not replace earlier 245,760-atom data or builds. A process persists for one
episode. Initial 50-step warm-up and 50-step observation are common, followed
by 120 decisions × 500 MD steps (60,000 MD steps). A fixed-action long-run
probe passed the length check, but action-changing episodes still need their
own safety preflight.
Do not extend the old `nx=240` episode beyond its valid propagation window.

## Action domain and safety gate

The action is an **absolute** neighbor skin and a balance choice; changing
partitioning has carry-over effects, so record the current skin and partition
in the observation. Keep `neigh_modify every 1 delay 0 check yes` fixed.

The initial numerical screening grid is skin
`{0.25, 0.35, 0.50, 0.70, 1.00}` and `weight neigh` factor
`{0.50, 0.65, 0.80, 1.00, 1.20, 1.50}` (30 settings), plus **skip balance**
at each skin (five more settings). Test factor `2.0` and skin `0.15`/`1.20`
at representative early/late states before admitting them to exploration.
The tested action set, not this candidate list, defines the safe domain.
Do not mistake factor 1.0 for unweighted atom balancing: the neighbor weight
is still applied at 1.0. For true atom-count weighting the `weight neigh`
keyword must be omitted.

Record full action wall time: skin application, any neighbor-list
reinitialization, optional balance, and the following 500-step run. A skipped
balance means *keep the current partition*, not undo previous balancing.
Reject dangerous builds, failed runs and nonfinite thermo; inspect pressure,
temperature, energy and shock position against a safe reference. Numeric
factor 2.0 and skin range extensions are **not yet approved as safe**.

## Paired conditions and learning budget

Run separate online learners from scratch for (a) no deliberate CPU load and
(b) a specified pinned CPU co-runner. Use the **same** restart/physical seed,
episode length, number of episodes, action constraints, learner architecture,
optimizer, update-to-data ratio, replay capacity, and evaluation schedule.
Use paired policy/exploration seeds. In an initial explicit exploration phase,
the random action schedule should be identical between conditions where safe,
to isolate the load effect. The co-runner's affinity must deliberately overlap
the targeted LAMMPS core/socket resources in the loaded condition; log the
exact overlap, intensity and duty cycle. Any intentional core sharing must be
distinguished from accidental MPI-rank oversubscription.

Provisional initial budget **per condition and per learner seed**:

* 12 complete exploration episodes, then 24 online-learning episodes;
* 120 decisions per episode if the long-run pilot passes, giving 4,320
  environment transitions per condition/seed;
* exploration samples the entire approved action domain, stratified by
  episode phase rather than a random walk that concentrates at boundaries;
* repeat with independent learner seeds after a one-seed integration pilot.

These are matched *minimum* counts, not an eight-hour cap or a guarantee of
convergence. Inspect phase × action visitation and learning curves. If either
condition lacks coverage or convergence, extend **both** conditions in paired
episode blocks; do not grant extra episodes only to the slower condition or
stop it early because it consumed more wall time. Reserve separate paired
episodes for frozen-policy live evaluation. Report episode totals and wall
times, not only reward curves.

## Completed length/timing pilot (2026-10-01)

The 491,520-atom restart was produced successfully in 109.0 s; its size is
133,694,669 bytes. The clean MPI32 fixed-action probe started with a 50-step
warm-up, balanced once with skin 0.5 / neighbor factor 1.5, and then ran six
10,000-step blocks without changing action. It completed with exit code 0,
finite thermo, and **zero dangerous builds in every block**. This tests
episode length, not online action safety or policy performance.

| End step | Shock-position observable | Fraction of x box | 10,000-step loop time, s |
| ---: | ---: | ---: | ---: |
| 11,250 | 128.25 | 0.150 | 245.6 |
| 21,250 | 243.54 | 0.285 | 334.5 |
| 31,250 | 359.15 | 0.420 | 355.4 |
| 41,250 | 474.38 | 0.555 | 380.8 |
| 51,250 | 589.12 | 0.689 | 416.6 |
| 61,250 | 648.25 | 0.758 | 373.7 |

The x box spans 0–855.26. End-of-episode front-position headroom is about
207 x units (24.2% of x extent) on this **one** baseline trajectory. Total
process wall time was 2,108.8 s (35.1 min). Thirty-six episodes at this
rate would be about 21.1 h for *one* condition, before per-decision balance
costs or load-induced slowdown. This is a planning estimate, not a runtime
cap: loaded and idle conditions must still receive the same experiences.

A separate two-decision persistent-process smoke test from the new restart
applied `(skin=1.0, factor=0.8)` and then `(skin=0.7, factor=1.2)` for 500
steps each. It completed with process exit 0; both transitions passed the
trajectory audit (step continuity, finite thermo, zero dangerous builds,
timer identity). The measured action-plus-run times were 5.06 and 4.06 s.
This only validates the start of a segmented episode; it does **not** certify
all candidate actions over 120 decisions.

An initial probe was interrupted after 10,000 steps by a faulty extra print
path. Its MPI children remained alive and contaminated a retry. Exactly those
64 stale `lmp` processes from the two aborted probe directories were
terminated and verified absent before the clean probe. The probe runner now
terminates its MPI process group on interruption/timeout. The clean data are
in ignored `rl_hpc/characterization/data/shock_long_probe_v4/result.json`;
the segmented smoke data are in ignored
`rl_hpc/characterization/data/shock_long_segment_smoke_v1/`.

Before main collection, extend the existing trajectory collector's 50-decision
guard and action representation to support 120 decisions and the skip-balance
branch; verify interruption cleanup and the front-position guard. These are
**implementation prerequisites**, not assumptions that the present script
already supports the whole protocol. Also finish the early/late action-range
safety screen before accepting skin/factor endpoints beyond the previous grid.

The experimental question is whether the learned action differs with the
observed shock/load state and reduces live runtime beyond strong fixed and
simple observable baselines. More transitions alone do not establish that.

## Pilot artifacts and commands

The axial restart generator now accepts lattice dimensions without modifying
the bundled example:

```bash
python3 rl_hpc/characterization/shock/prepare_heavy.py \
  --nx 480 --ny 16 --nz 16 \
  --output rl_hpc/characterization/data/shock_long_preflight_v1
```

The long-run probe reads this restart in a fresh MPI process and records the
shock-position/box-length relation at six 10,000-step boundaries:

```bash
python3 rl_hpc/characterization/shock/long_episode_preflight.py \
  --restart rl_hpc/characterization/data/shock_long_preflight_v1/checkpoint-S1.restart \
  --output rl_hpc/characterization/data/shock_long_probe_v1 \
  --blocks 6 --block-steps 10000
```

Both output directories are ignored by Git. The probe is a single fixed-action
trajectory, not an action safety sweep and not an online-learning run.

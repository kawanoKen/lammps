# Continuous-skin-only frozen-policy evaluation

Status: **32/32 live runs completed successfully** on 2026-10-05 at 07:45
JST after 8 h 24 min. The service had no restarts. Two short smokes (idle
and CPU-jitter) passed before launch. This was evaluation only: no
exploration or online parameter updates.

## Matched design

- Policies: continuous-skin policy trained without contention; continuous-
  skin policy trained with CPU jitter; fixed skin 0.50 with balance weight
  1.5; previous full-action idle-trained policy. The two continuous policies
  always use balance weight 1.5, and their features exclude step, decision
  number, shock position and contention label. The old full-action policy
  has a different action/state space and is a reference, not a controlled
  skin-only ablation.
- Restart seeds: 47287 and 57287, excluded from all policy training. They
  were used in the previous full-action evaluation, so they are not entirely
  untouched for the broader research program. Baseline skin 0.50 was
  preregistered from earlier characterization; this is not a proof of the
  globally best fixed skin in `[0.25,1.0]`.
- Each seed is run twice in idle and twice under controlled piecewise CPU
  contention. Four policies per matched `(seed, repetition, regime)` group
  get the same contention schedule, initial restart and setup. Policy order
  is randomized within groups: 2 x 2 x 2 x 4 = **32 independent processes**.
- Each run is 120 decisions x 500 MD steps at MPI32, OMP1. The measured
  outcome is total action-application plus MD-run time for the fixed 60,000
  steps. Process startup and common warm-up are excluded. Invalid physics,
  dangerous neighbour builds, unexpected step advance or MPI failure stop
  the run. Incomplete attempts are preserved and excluded from comparisons.
- The manifest records restart/model/binary hashes, schedules and run order.
  Raw results are in ignored
  `rl_hpc/characterization/data/shock_skin_continuous_eval_main_v1/`.

The service has `Restart=on-failure` (30-second delay, three rapid attempts
per hour) and is enabled for reboot recovery. User linger is enabled.
`run_skin_evaluation.py` resumes after the last complete run and is a no-op
after all 32 runs complete.

```bash
systemctl --user status lammps-shock-skin-evaluation-v1.service
journalctl --user -u lammps-shock-skin-evaluation-v1.service -f
python3 rl_hpc/characterization/shock/analyze_online_cross.py \
  rl_hpc/characterization/data/shock_skin_continuous_eval_main_v1
```

The analysis must compare paired episode wall times, not training reward
curves. Two unique physical restarts remain a small generalization sample.
An improvement over the fixed 0.50 baseline would not alone establish RL
superiority over a better tuned fixed skin, a phase rule or a supervised
predictor. The crucial CPU-load question is whether the relative ordering
of the **two skin-only policies** changes under contention.

## Results

The analyzer checked all 32 independent-process runs, each with 120 valid
500-step decisions, matching contention schedules, normal MPI exit, and
zero dangerous builds. Values are mean +/- sample standard deviation over
four runs per regime/policy (two physical restart seeds, two repeated
schedule seeds). The repetitions do not increase the number of independent
physical initial conditions beyond two.

| Regime | Skin-only idle-trained | Skin-only CPU-jitter-trained | Fixed skin 0.50 | Previous full-action idle-trained |
| --- | ---: | ---: | ---: | ---: |
| Idle | 938.10 +/- 1.39 s | 937.70 +/- 2.02 s | **932.77 +/- 5.25 s** | **791.65 +/- 3.66 s** |
| CPU jitter | 1000.56 +/- 4.73 s | 994.08 +/- 8.11 s | **987.07 +/- 4.81 s** | **916.87 +/- 4.32 s** |

Within the **skin-only/fixed-factor comparison**, neither learned policy
beat fixed skin 0.50. Relative to that fixed policy, the idle-trained
skin policy was 5.32 s slower in idle (+0.57%) and 13.49 s slower under
CPU jitter (+1.37%). The jitter-trained skin policy was 4.92 s slower in
idle (+0.53%) and 7.01 s slower under jitter (+0.71%). Every matched pair
under jitter favoured fixed 0.50 over either skin-only policy. In idle,
the small pair differences were mixed for idle-trained skin and all four
favoured fixed 0.50 over jitter-trained skin.

Directly comparing the two skin-only policies, jitter training gained
only 0.40 +/- 3.06 s (paired SD) in idle and 6.48 +/- 7.01 s under
jitter. The idle difference is negligible; the jitter difference is small
and inconsistent across the four pairs (one favoured idle training).
There is no convincing state-conditioned action reversal or positive
fixed-policy regret to exploit in this skin-only setting.

The idle-trained skin-only policy changed skin in **476/476** opportunities
between consecutive decisions, adding about 7.1 s/episode of neighbour
preparation in idle and 7.5 s under jitter. Its MD-run component was
approximately 930.0 s vs 931.8 s for fixed 0.50 in idle, so the small
run-time gain was outweighed by repeated action preparation. The
jitter-trained skin-only policy changed skin 73/476 times and spent about
1.2 s/episode preparing; its MD-run component was itself slower than
fixed 0.50. Balance command time was about 0.9--1.0 s/episode for all
three fixed-factor policies.

The previous full-action policy was much faster than fixed 0.50 (about
15.1% in idle and 7.1% under jitter), reproducing the earlier result.
**This is not an isolated factor effect:** that older policy can change
both skin and neighbor-weight factor and receives progress/front-position
features that the new skin-only policies deliberately lack. Its advantage
cannot be attributed to factor, progress information, or their interaction
without further ablations. The continuous-skin learners' training curves
therefore did not translate into a live performance gain.

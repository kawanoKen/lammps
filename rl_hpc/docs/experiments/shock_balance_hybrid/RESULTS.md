# Shock/NEMD hybrid-balance results

## Data integrity

- Training: 48 idle + 48 CPU-jitter episodes; 11,520 transitions total.
- Evaluation: 40/40 held-out runs completed.
- Every evaluation run contains 120 valid 500-step decisions.
- All processes exited normally.
- Dangerous builds, nonfinite values, and recorded LAMMPS errors: zero.

Values are mean +/- sample standard deviation over four paired runs per cell
(two held-out physical seeds and two schedule repetitions).

| Regime | Idle-trained | CPU-jitter-trained | Always skip | Fixed 1.5 | Threshold 1.5 |
|---|---:|---:|---:|---:|---:|
| Idle | **788.29 +/- 4.15 s** | 814.88 +/- 5.73 s | 2120.85 +/- 5.57 s | 932.16 +/- 5.84 s | 972.52 +/- 9.41 s |
| CPU jitter | 920.18 +/- 5.18 s | **909.58 +/- 11.25 s** | 2587.67 +/- 9.91 s | 988.25 +/- 6.94 s | 1035.29 +/- 10.80 s |

Relative to fixed factor 1.5, idle training saved 143.86 s (15.43%) in idle
and 68.06 s (6.89%) under jitter. CPU-jitter training saved 117.28 s (12.58%)
in idle and 78.67 s (7.96%) under jitter. Every paired run favored both
learned policies over fixed 1.5 in both regimes.

## Policy ordering

In idle, idle training beat jitter training by 26.58 s (3.37%). The paired
95% t interval for `jitter - idle` was [13.66, 39.51] s.

Under CPU jitter, jitter training was 10.60 s (1.15%) faster on average, but
the paired differences changed sign. The corresponding 95% interval was
[-36.45, 15.24] s. The mean ordering therefore reverses, but the CPU-jitter
side does not yet establish a statistically resolved need for a distinct
policy.

## Action interpretation

Always-skip is extremely poor, showing that redistribution is essential in
this growing shock imbalance. The threshold rule is also slower than always
balancing at factor 1.5.

Learned policies balance in about 113--116 of 120 intervals. Their advantage
therefore comes mainly from continuous factor selection rather than skip
decisions. The idle-trained policy uses a mean executed factor near 0.97; the
CPU-jitter-trained policy uses about 1.24. Both use the full approved
`[0.5,1.5]` interval.

## Scope

The result establishes superiority over the tested fixed and threshold
baselines. It does not establish superiority over the globally best possible
fixed factor, a stronger supervised contextual optimizer, or an oracle. Only
two independent held-out physical initial states were used.

The next decisive experiment is a larger paired confirmation of the two
frozen policies under CPU jitter, with a denser fixed-factor baseline selected
without using confirmation outcomes.

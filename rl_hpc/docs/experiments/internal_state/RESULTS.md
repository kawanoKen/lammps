# Internal-state checkpoint-fork characterization

## Result

This is a direct counterfactual measurement: each candidate action starts from
the *same binary restart state* in a separate LAMMPS process. No GPU co-runner
or external contention was used. Across five evolving checkpoints, neither
workload shows a meaningful change of the best safe neighbor configuration.
The state-conditioned oracle equals the best fixed configuration in both
workloads, so the measured adaptation opportunity is **0.00%**.

`delay=0` and `check=yes` were fixed. The primary metric is LAMMPS loop time
for the same 500 steps; process startup is not used for ranking.

## Interpretation

- **LJ (control):** `skin=0.4, every=20` has the lowest mean at every
  checkpoint. Its difference from `skin=0.4, every=10` is sometimes below
  the 3-run noise, so those two should be treated as a stable near-optimal
  set—not evidence of a changing optimum. The large skin effect is stable.
- **SPC/E + PPPM:** `skin=2.0, every=5` is the winner at every checkpoint.
  It is about 1.0--1.2% faster than the runner-up (`skin=2.0, every=1`),
  larger than the within-action CVs (roughly 0.1--0.3%), and the ordering does
  not reverse. Although temperature, pressure, Neigh, Comm, and KSpace timing
  fluctuate across checkpoints, they do not correspond to a new best action.
- `skin=4.0, every=20` produced dangerous neighbor builds at SPC/E checkpoint
  S3 in all three repeats. It is invalid there and excluded from all
  performance conclusions and from the fixed-policy candidate set.

Consequently, these natural internal-state changes do **not** justify an
internal-state-only RL controller for the tested neighbor controls. The prior
same-GPU-contention result remains an external-resource phenomenon rather
than evidence of a changing intrinsic LAMMPS optimum.

## Lennard-Jones (GPU, idle)

| Checkpoint | Step | Action | Mean loop (s) | Std | CV | n | Rank |
|---|---:|---|---:|---:|---:|---:|---:|
| S1 | 1000 | skin=0.4, every=20 | 0.483107 | 0.000633 | 0.13% | 3 | 1 |
| S1 | 1000 | skin=0.4, every=10 | 0.483115 | 0.000622 | 0.13% | 3 | 2 |
| S1 | 1000 | skin=0.4, every=5 | 0.484024 | 0.000953 | 0.20% | 3 | 3 |
| S1 | 1000 | skin=0.4, every=1 | 0.495016 | 0.000199 | 0.04% | 3 | 4 |
| S1 | 1000 | skin=0.6, every=20 | 0.591917 | 0.000258 | 0.04% | 3 | 5 |
| S1 | 1000 | skin=0.6, every=10 | 0.592355 | 0.000275 | 0.05% | 3 | 6 |
| S1 | 1000 | skin=0.6, every=5 | 0.594135 | 0.000110 | 0.02% | 3 | 7 |
| S1 | 1000 | skin=0.6, every=1 | 0.603654 | 0.001262 | 0.21% | 3 | 8 |
| S1 | 1000 | skin=0.8, every=20 | 0.701375 | 0.002576 | 0.37% | 3 | 9 |
| S1 | 1000 | skin=0.8, every=10 | 0.701844 | 0.002262 | 0.32% | 3 | 10 |
| S1 | 1000 | skin=0.8, every=5 | 0.704896 | 0.000350 | 0.05% | 3 | 11 |
| S1 | 1000 | skin=0.8, every=1 | 0.714030 | 0.002898 | 0.41% | 3 | 12 |
| S1 | 1000 | skin=1, every=20 | 0.818779 | 0.000752 | 0.09% | 3 | 13 |
| S1 | 1000 | skin=1, every=10 | 0.819226 | 0.000824 | 0.10% | 3 | 14 |
| S1 | 1000 | skin=1, every=5 | 0.820431 | 0.000336 | 0.04% | 3 | 15 |
| S1 | 1000 | skin=1, every=1 | 0.830882 | 0.000485 | 0.06% | 3 | 16 |
| S1 | 1000 | skin=1.2, every=10 | 0.945841 | 0.002352 | 0.25% | 3 | 17 |
| S1 | 1000 | skin=1.2, every=20 | 0.946579 | 0.000228 | 0.02% | 3 | 18 |
| S1 | 1000 | skin=1.2, every=5 | 0.947751 | 0.000687 | 0.07% | 3 | 19 |
| S1 | 1000 | skin=1.2, every=1 | 0.958670 | 0.000666 | 0.07% | 3 | 20 |
| S2 | 2000 | skin=0.4, every=20 | 0.483080 | 0.000289 | 0.06% | 3 | 1 |
| S2 | 2000 | skin=0.4, every=10 | 0.483883 | 0.000318 | 0.07% | 3 | 2 |
| S2 | 2000 | skin=0.4, every=5 | 0.485459 | 0.000339 | 0.07% | 3 | 3 |
| S2 | 2000 | skin=0.4, every=1 | 0.496265 | 0.000311 | 0.06% | 3 | 4 |
| S2 | 2000 | skin=0.6, every=20 | 0.591974 | 0.000150 | 0.03% | 3 | 5 |
| S2 | 2000 | skin=0.6, every=10 | 0.592062 | 0.000249 | 0.04% | 3 | 6 |
| S2 | 2000 | skin=0.6, every=5 | 0.593526 | 0.000059 | 0.01% | 3 | 7 |
| S2 | 2000 | skin=0.6, every=1 | 0.603960 | 0.000135 | 0.02% | 3 | 8 |
| S2 | 2000 | skin=0.8, every=20 | 0.700805 | 0.001555 | 0.22% | 3 | 9 |
| S2 | 2000 | skin=0.8, every=10 | 0.702722 | 0.000279 | 0.04% | 3 | 10 |
| S2 | 2000 | skin=0.8, every=5 | 0.702969 | 0.001891 | 0.27% | 3 | 11 |
| S2 | 2000 | skin=0.8, every=1 | 0.714004 | 0.001789 | 0.25% | 3 | 12 |
| S2 | 2000 | skin=1, every=20 | 0.817000 | 0.001469 | 0.18% | 3 | 13 |
| S2 | 2000 | skin=1, every=10 | 0.817441 | 0.000706 | 0.09% | 3 | 14 |
| S2 | 2000 | skin=1, every=5 | 0.818658 | 0.001982 | 0.24% | 3 | 15 |
| S2 | 2000 | skin=1, every=1 | 0.828280 | 0.002602 | 0.31% | 3 | 16 |
| S2 | 2000 | skin=1.2, every=20 | 0.945321 | 0.000853 | 0.09% | 3 | 17 |
| S2 | 2000 | skin=1.2, every=5 | 0.945537 | 0.003584 | 0.38% | 3 | 18 |
| S2 | 2000 | skin=1.2, every=10 | 0.945749 | 0.001254 | 0.13% | 3 | 19 |
| S2 | 2000 | skin=1.2, every=1 | 0.958597 | 0.000620 | 0.06% | 3 | 20 |
| S3 | 3000 | skin=0.4, every=20 | 0.482319 | 0.000546 | 0.11% | 3 | 1 |
| S3 | 3000 | skin=0.4, every=10 | 0.483097 | 0.000607 | 0.13% | 3 | 2 |
| S3 | 3000 | skin=0.4, every=5 | 0.483700 | 0.001684 | 0.35% | 3 | 3 |
| S3 | 3000 | skin=0.4, every=1 | 0.494802 | 0.001219 | 0.25% | 3 | 4 |
| S3 | 3000 | skin=0.6, every=20 | 0.590943 | 0.000243 | 0.04% | 3 | 5 |
| S3 | 3000 | skin=0.6, every=10 | 0.591541 | 0.000143 | 0.02% | 3 | 6 |
| S3 | 3000 | skin=0.6, every=5 | 0.593105 | 0.000309 | 0.05% | 3 | 7 |
| S3 | 3000 | skin=0.6, every=1 | 0.603853 | 0.000901 | 0.15% | 3 | 8 |
| S3 | 3000 | skin=0.8, every=10 | 0.700210 | 0.002599 | 0.37% | 3 | 9 |
| S3 | 3000 | skin=0.8, every=20 | 0.700572 | 0.000173 | 0.02% | 3 | 10 |
| S3 | 3000 | skin=0.8, every=5 | 0.702163 | 0.002146 | 0.31% | 3 | 11 |
| S3 | 3000 | skin=0.8, every=1 | 0.713538 | 0.000175 | 0.02% | 3 | 12 |
| S3 | 3000 | skin=1, every=20 | 0.816318 | 0.000520 | 0.06% | 3 | 13 |
| S3 | 3000 | skin=1, every=10 | 0.817237 | 0.000467 | 0.06% | 3 | 14 |
| S3 | 3000 | skin=1, every=5 | 0.818321 | 0.000413 | 0.05% | 3 | 15 |
| S3 | 3000 | skin=1, every=1 | 0.828929 | 0.000422 | 0.05% | 3 | 16 |
| S3 | 3000 | skin=1.2, every=20 | 0.944311 | 0.000632 | 0.07% | 3 | 17 |
| S3 | 3000 | skin=1.2, every=10 | 0.945413 | 0.000251 | 0.03% | 3 | 18 |
| S3 | 3000 | skin=1.2, every=5 | 0.945666 | 0.001883 | 0.20% | 3 | 19 |
| S3 | 3000 | skin=1.2, every=1 | 0.957342 | 0.001762 | 0.18% | 3 | 20 |
| S4 | 4000 | skin=0.4, every=20 | 0.482767 | 0.000429 | 0.09% | 3 | 1 |
| S4 | 4000 | skin=0.4, every=10 | 0.483359 | 0.000592 | 0.12% | 3 | 2 |
| S4 | 4000 | skin=0.4, every=5 | 0.485059 | 0.000484 | 0.10% | 3 | 3 |
| S4 | 4000 | skin=0.4, every=1 | 0.495337 | 0.000784 | 0.16% | 3 | 4 |
| S4 | 4000 | skin=0.6, every=20 | 0.589586 | 0.001890 | 0.32% | 3 | 5 |
| S4 | 4000 | skin=0.6, every=10 | 0.591199 | 0.000221 | 0.04% | 3 | 6 |
| S4 | 4000 | skin=0.6, every=5 | 0.593124 | 0.000365 | 0.06% | 3 | 7 |
| S4 | 4000 | skin=0.6, every=1 | 0.603915 | 0.000277 | 0.05% | 3 | 8 |
| S4 | 4000 | skin=0.8, every=20 | 0.700711 | 0.000201 | 0.03% | 3 | 9 |
| S4 | 4000 | skin=0.8, every=10 | 0.701977 | 0.000271 | 0.04% | 3 | 10 |
| S4 | 4000 | skin=0.8, every=5 | 0.702878 | 0.000647 | 0.09% | 3 | 11 |
| S4 | 4000 | skin=0.8, every=1 | 0.713692 | 0.000502 | 0.07% | 3 | 12 |
| S4 | 4000 | skin=1, every=20 | 0.816883 | 0.000423 | 0.05% | 3 | 13 |
| S4 | 4000 | skin=1, every=10 | 0.817419 | 0.000260 | 0.03% | 3 | 14 |
| S4 | 4000 | skin=1, every=5 | 0.818682 | 0.000248 | 0.03% | 3 | 15 |
| S4 | 4000 | skin=1, every=1 | 0.829320 | 0.000430 | 0.05% | 3 | 16 |
| S4 | 4000 | skin=1.2, every=20 | 0.945475 | 0.000883 | 0.09% | 3 | 17 |
| S4 | 4000 | skin=1.2, every=10 | 0.946891 | 0.001021 | 0.11% | 3 | 18 |
| S4 | 4000 | skin=1.2, every=5 | 0.948432 | 0.000770 | 0.08% | 3 | 19 |
| S4 | 4000 | skin=1.2, every=1 | 0.956854 | 0.004051 | 0.42% | 3 | 20 |
| S5 | 5000 | skin=0.4, every=20 | 0.483092 | 0.000579 | 0.12% | 3 | 1 |
| S5 | 5000 | skin=0.4, every=10 | 0.483387 | 0.000652 | 0.13% | 3 | 2 |
| S5 | 5000 | skin=0.4, every=5 | 0.484332 | 0.000567 | 0.12% | 3 | 3 |
| S5 | 5000 | skin=0.4, every=1 | 0.495153 | 0.001239 | 0.25% | 3 | 4 |
| S5 | 5000 | skin=0.6, every=20 | 0.590873 | 0.000256 | 0.04% | 3 | 5 |
| S5 | 5000 | skin=0.6, every=10 | 0.591296 | 0.000445 | 0.08% | 3 | 6 |
| S5 | 5000 | skin=0.6, every=5 | 0.593242 | 0.000302 | 0.05% | 3 | 7 |
| S5 | 5000 | skin=0.6, every=1 | 0.604030 | 0.000722 | 0.12% | 3 | 8 |
| S5 | 5000 | skin=0.8, every=20 | 0.701196 | 0.000481 | 0.07% | 3 | 9 |
| S5 | 5000 | skin=0.8, every=10 | 0.702223 | 0.000358 | 0.05% | 3 | 10 |
| S5 | 5000 | skin=0.8, every=5 | 0.703365 | 0.000375 | 0.05% | 3 | 11 |
| S5 | 5000 | skin=0.8, every=1 | 0.714600 | 0.000102 | 0.01% | 3 | 12 |
| S5 | 5000 | skin=1, every=20 | 0.817632 | 0.000211 | 0.03% | 3 | 13 |
| S5 | 5000 | skin=1, every=10 | 0.818435 | 0.000301 | 0.04% | 3 | 14 |
| S5 | 5000 | skin=1, every=5 | 0.819280 | 0.000022 | 0.00% | 3 | 15 |
| S5 | 5000 | skin=1, every=1 | 0.829451 | 0.002291 | 0.28% | 3 | 16 |
| S5 | 5000 | skin=1.2, every=20 | 0.942487 | 0.004139 | 0.44% | 3 | 17 |
| S5 | 5000 | skin=1.2, every=10 | 0.945102 | 0.000842 | 0.09% | 3 | 18 |
| S5 | 5000 | skin=1.2, every=5 | 0.945741 | 0.002631 | 0.28% | 3 | 19 |
| S5 | 5000 | skin=1.2, every=1 | 0.955897 | 0.003950 | 0.41% | 3 | 20 |

Best action by checkpoint: S1=(0.4,20), S2=(0.4,20), S3=(0.4,20), S4=(0.4,20), S5=(0.4,20).

- State-conditioned oracle total: **2.414364 s**
- Best fixed action: **skin=0.4, every=20**; total **2.414364 s**
- Theoretical checkpoint-conditioned improvement: **0.00%**.
- Invalid/unsafe branch records excluded: **0**.

Checkpoint state before actions:

| Checkpoint | Step | Temp | Press | Density | Pair | Neigh | Comm | KSpace | Neighbor builds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1000 | 0.7037 | 0.70685 | 0.8442 | 0.01358 | 0.35765 | 0.18715 | 0.00000 | 65 |
| S2 | 2000 | 0.69786 | 0.74973 | 0.8442 | 0.01331 | 0.33099 | 0.15230 | 0.00000 | 63 |
| S3 | 3000 | 0.69883 | 0.74625 | 0.8442 | 0.01330 | 0.32697 | 0.14582 | 0.00000 | 62 |
| S4 | 4000 | 0.6993 | 0.74244 | 0.8442 | 0.01331 | 0.32486 | 0.14666 | 0.00000 | 62 |
| S5 | 5000 | 0.69735 | 0.75197 | 0.8442 | 0.01325 | 0.32543 | 0.14877 | 0.00000 | 62 |

## SPC/E + PPPM (GPU, idle)

| Checkpoint | Step | Action | Mean loop (s) | Std | CV | n | Rank |
|---|---:|---|---:|---:|---:|---:|---:|
| S1 | 1000 | skin=2, every=5 | 5.705100 | 0.013430 | 0.24% | 3 | 1 |
| S1 | 1000 | skin=2, every=1 | 5.773983 | 0.008787 | 0.15% | 3 | 2 |
| S1 | 1000 | skin=3, every=10 | 6.307363 | 0.022441 | 0.36% | 3 | 3 |
| S1 | 1000 | skin=3, every=5 | 6.337967 | 0.025329 | 0.40% | 3 | 4 |
| S1 | 1000 | skin=3, every=1 | 6.426550 | 0.011761 | 0.18% | 3 | 5 |
| S1 | 1000 | skin=4, every=20 | 6.749677 | 0.015223 | 0.23% | 3 | 6 |
| S1 | 1000 | skin=4, every=10 | 6.824950 | 0.013725 | 0.20% | 3 | 7 |
| S1 | 1000 | skin=4, every=5 | 6.828203 | 0.004297 | 0.06% | 3 | 8 |
| S1 | 1000 | skin=4, every=1 | 6.867240 | 0.010033 | 0.15% | 3 | 9 |
| S2 | 2000 | skin=2, every=5 | 5.696493 | 0.003136 | 0.06% | 3 | 1 |
| S2 | 2000 | skin=2, every=1 | 5.763897 | 0.013462 | 0.23% | 3 | 2 |
| S2 | 2000 | skin=3, every=10 | 6.134347 | 0.020096 | 0.33% | 3 | 3 |
| S2 | 2000 | skin=3, every=5 | 6.182107 | 0.024501 | 0.40% | 3 | 4 |
| S2 | 2000 | skin=3, every=1 | 6.245397 | 0.009074 | 0.15% | 3 | 5 |
| S2 | 2000 | skin=4, every=20 | 6.750043 | 0.010101 | 0.15% | 3 | 6 |
| S2 | 2000 | skin=4, every=10 | 6.812577 | 0.013759 | 0.20% | 3 | 7 |
| S2 | 2000 | skin=4, every=5 | 6.829977 | 0.004991 | 0.07% | 3 | 8 |
| S2 | 2000 | skin=4, every=1 | 6.876167 | 0.018336 | 0.27% | 3 | 9 |
| S3 | 3000 | skin=2, every=5 | 5.703183 | 0.007268 | 0.13% | 3 | 1 |
| S3 | 3000 | skin=2, every=1 | 5.765093 | 0.015443 | 0.27% | 3 | 2 |
| S3 | 3000 | skin=3, every=10 | 6.304023 | 0.016218 | 0.26% | 3 | 3 |
| S3 | 3000 | skin=3, every=5 | 6.369787 | 0.017762 | 0.28% | 3 | 4 |
| S3 | 3000 | skin=3, every=1 | 6.465117 | 0.026385 | 0.41% | 3 | 5 |
| S3 | 3000 | skin=4, every=10 | 6.840873 | 0.013520 | 0.20% | 3 | 6 |
| S3 | 3000 | skin=4, every=5 | 6.848367 | 0.017440 | 0.25% | 3 | 7 |
| S3 | 3000 | skin=4, every=1 | 6.888597 | 0.012665 | 0.18% | 3 | 8 |
| S4 | 4000 | skin=2, every=5 | 5.700590 | 0.018480 | 0.32% | 3 | 1 |
| S4 | 4000 | skin=2, every=1 | 5.756863 | 0.019701 | 0.34% | 3 | 2 |
| S4 | 4000 | skin=3, every=10 | 6.120790 | 0.019318 | 0.32% | 3 | 3 |
| S4 | 4000 | skin=3, every=5 | 6.181280 | 0.016081 | 0.26% | 3 | 4 |
| S4 | 4000 | skin=3, every=1 | 6.246960 | 0.016711 | 0.27% | 3 | 5 |
| S4 | 4000 | skin=4, every=20 | 6.743660 | 0.017859 | 0.26% | 3 | 6 |
| S4 | 4000 | skin=4, every=10 | 6.810943 | 0.016835 | 0.25% | 3 | 7 |
| S4 | 4000 | skin=4, every=5 | 6.845947 | 0.012565 | 0.18% | 3 | 8 |
| S4 | 4000 | skin=4, every=1 | 6.863410 | 0.020645 | 0.30% | 3 | 9 |
| S5 | 5000 | skin=2, every=5 | 5.695270 | 0.011097 | 0.19% | 3 | 1 |
| S5 | 5000 | skin=2, every=1 | 5.753070 | 0.015192 | 0.26% | 3 | 2 |
| S5 | 5000 | skin=3, every=10 | 6.154657 | 0.008364 | 0.14% | 3 | 3 |
| S5 | 5000 | skin=3, every=5 | 6.178120 | 0.008550 | 0.14% | 3 | 4 |
| S5 | 5000 | skin=3, every=1 | 6.252223 | 0.020994 | 0.34% | 3 | 5 |
| S5 | 5000 | skin=4, every=20 | 6.758927 | 0.006974 | 0.10% | 3 | 6 |
| S5 | 5000 | skin=4, every=5 | 6.812613 | 0.008298 | 0.12% | 3 | 7 |
| S5 | 5000 | skin=4, every=10 | 6.819957 | 0.006867 | 0.10% | 3 | 8 |
| S5 | 5000 | skin=4, every=1 | 6.871330 | 0.019791 | 0.29% | 3 | 9 |

Best action by checkpoint: S1=(2,5), S2=(2,5), S3=(2,5), S4=(2,5), S5=(2,5).

- State-conditioned oracle total: **28.500637 s**
- Best fixed action: **skin=2, every=5**; total **28.500637 s**
- Theoretical checkpoint-conditioned improvement: **0.00%**.
- Invalid/unsafe branch records excluded: **3**.

Checkpoint state before actions:

| Checkpoint | Step | Temp | Press | Density | Pair | Neigh | Comm | KSpace | Neighbor builds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1000 | 298.44 | 463.9 | 1.0041 | 0.04103 | 0.74900 | 0.11568 | 11.06500 | 53 |
| S2 | 2000 | 300.35 | 467.41 | 1.0041 | 0.04099 | 0.75002 | 0.10065 | 11.05500 | 53 |
| S3 | 3000 | 300.67 | 391.93 | 1.0041 | 0.04121 | 0.77827 | 0.10428 | 11.06800 | 55 |
| S4 | 4000 | 297.48 | 420.27 | 1.0041 | 0.04115 | 0.78350 | 0.11402 | 11.09800 | 55 |
| S5 | 5000 | 300.95 | 462.91 | 1.0041 | 0.04102 | 0.77195 | 0.10153 | 11.09200 | 54 |

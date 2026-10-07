# Shock: numerical action sweep

32 MPI ranks; six reused physical checkpoints; common 50-step warm-up; 500 measured steps.
Actions: skin × neighbor-weight factor. CVCF/history diagnostics excluded in every fork.
Total includes list setup, one-time balance, and simulation. No time-weight action is used.

| State | Best skin | Best factor | Mean seconds | Std | CV % |
|---|---:|---:|---:|---:|---:|
| S1 | 0.35 | 1.5 | 1.179236 | 0.001330 | 0.113 |
| S2 | 0.35 | 1.5 | 1.375094 | 0.002427 | 0.177 |
| S3 | 0.5 | 0.8 | 1.554257 | 0.003580 | 0.230 |
| S4 | 0.5 | 1.5 | 1.683156 | 0.002870 | 0.171 |
| S5 | 0.35 | 1.5 | 1.703631 | 0.006507 | 0.382 |
| S6 | 0.5 | 1.2 | 1.566420 | 0.009902 | 0.632 |

Best fixed discovery action: skin=0.35, factor=1.5.
Summed fixed means: 9.307376 s; selected oracle: 9.061794 s.
Descriptive improvement (fixed-oracle)/fixed: 2.639%.
This is a selection-biased discovery estimate, not a live-policy gain or a proven upper bound.
Full state-action means, variability, ranks and regrets: matrix.csv.
Response surfaces: response.svg (per-state relative runtime; numbers are seconds).

Frozen confirmation pair: states S4/S5, actions (0.5, 1.5)/(0.35, 1.5).
Positive A-minus-B means B is faster; negative means A is faster.
S4: A-B=-0.130306 s; bootstrap 95% CI [-0.134159, -0.126727].
S5: A-B=0.154804 s; bootstrap 95% CI [0.150988, 0.158496].
Independent confirmation of opposite ordering with both intervals excluding zero: True.

Interpretation is limited to these checkpoints and the sampled parameter grid.
Boundary optima require range extension before claiming an optimum.
No RL or SMDP training was performed.

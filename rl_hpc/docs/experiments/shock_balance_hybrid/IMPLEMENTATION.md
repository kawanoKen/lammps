# Shock/NEMD hybrid-balance implementation record

## Code

- `rl_hpc/characterization/shock/online_balance_hybrid.py`: data collection
  and online FQI.
- `rl_hpc/characterization/shock/run_balance_hybrid_study.py`: sequential,
  resumable idle and CPU-jitter training.
- `rl_hpc/characterization/shock/evaluate_balance_hybrid.py`: paired frozen
  live evaluation.
- `rl_hpc/characterization/shock/analyze_balance_hybrid_evaluation.py`:
  independent result and schedule audit.
- `rl_hpc/characterization/shock/run_balance_hybrid_evaluation.py`: resumable
  main evaluation and final analysis.

## Generated data

Large generated files are ignored under:

```text
rl_hpc/characterization/data/shock_balance_hybrid_idle_v1/
rl_hpc/characterization/data/shock_balance_hybrid_cpu_jitter_v1/
rl_hpc/characterization/data/shock_balance_hybrid_eval_main_v1/
```

Each training condition completed 48 episodes and 5,760 transitions. The
held-out evaluation completed 40 independent processes. Manifests record
binary, restart, model, script, schedule, and policy provenance.

The training runner reconstructs replay from committed complete episodes when
resumed. Incomplete episode directories are preserved separately. Evaluation
similarly resumes after the last committed result without rerunning it.

Historical implementation and progress narration is retained in
[the archived combined record](../../archive/SHOCK_BALANCE_HYBRID_COMBINED.md).

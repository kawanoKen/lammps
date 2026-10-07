# Baseline timing sanity check

Workload: `rl_hpc/bench_lj.in`, derived from bundled `bench/in.lj`.

- 32,000 atoms (`x=y=z=1`)
- 500 LJ timesteps
- velocity seed `87287`
- `neigh_modify delay 0 every 20 check no`, matching the bundled benchmark
- `OMP_NUM_THREADS=1`
- same executable: `build_rl/lmp`
- three sequential repetitions per MPI count
- standard deviation below is the sample standard deviation (`n-1`)
- wall time is the launcher-measured process wall time; LAMMPS loop time excludes
  some process startup and dynamic-loader overhead

## Results

| MPI ranks | Run 1 wall (s) | Run 2 wall (s) | Run 3 wall (s) | Mean (s) | Std. dev. (s) | CV |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.7246 | 7.6990 | 7.6952 | 7.7062 | 0.0160 | 0.207% |
| 4 | 2.9921 | 2.9917 | 3.0016 | 2.9951 | 0.0056 | 0.187% |

LAMMPS-reported loop-time means were 6.6170 s (MPI 1) and 1.8891 s (MPI 4).
The corresponding mean reported throughputs were approximately 75.56 and
264.68 timesteps/s.  The 4-rank run is faster for this fixed-size workload,
but this is only a two-point sanity check, not a scaling study.

The low short-run variation is useful for initial experiments, but measurements
remain sensitive to CPU frequency policy (`powersave` at capture time), system
load, process affinity, and concurrent GPU/CPU jobs.  Keep the process/thread
configuration fixed and use longer segments for controller decisions.

## Reproduction commands

Use new output directory names each time because the launcher refuses to
overwrite an existing directory:

```bash
python3 rl_hpc/run_experiment.py --mpi 1 --steps 500 --seed 87287 \
  --output rl_hpc/runs/new-mpi1
python3 rl_hpc/run_experiment.py --mpi 4 --steps 500 --seed 87287 \
  --output rl_hpc/runs/new-mpi4
```

The individual raw records used for this summary are under the ignored
directories `rl_hpc/runs/baseline-mpi1-*` and `rl_hpc/runs/baseline-mpi4-*`.

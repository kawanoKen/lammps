# LAMMPS RL/HPC research documentation

This directory separates research assumptions, machine records, experiment
protocols, implementation notes, and measured results.

Start with [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) for the current evidence
and the winning policy or configuration in each experiment.

## Layout

- `research/`: problem formulation, source-level analysis, and candidate
  parameter space.
- `environment/`: machine, software, network, build, and baseline records.
- `experiments/`: experiment-specific protocol, implementation, training, and
  result documents.
- `archive/`: superseded or combined historical records retained for
  traceability. These are not the current source of conclusions.

Generated datasets and large logs remain under the ignored
`rl_hpc/characterization/data/` tree. Experiment code remains under
`rl_hpc/characterization/`.

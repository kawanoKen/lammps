# Offline RL pilot

`env.py` provides a persistent, GPU-0, 500-step delta-control environment.
It intentionally contains no learner. Its provisional safe domain is
`skin=[0.6,1.2]`, `every=[1,15]`, `delay=0`, `check=yes`; it excludes known
dangerous low-skin/high-every combinations pending a dedicated preflight.

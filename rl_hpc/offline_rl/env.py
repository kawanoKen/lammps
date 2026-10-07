"""Persistent GPU-LJ environment for offline-data collection (no learning)."""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
# Workers are launched as scripts from this directory, so make the repository
# package root explicit before importing the shared log parser.
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from rl_hpc.characterization.characterize import parse_log

SAFE_SKIN=(0.6,1.2)  # 0.4--0.6 was not yet validated for every=20
# `every=15` produced a dangerous build at skin=0.672 in the fresh pilot.
# Keep a conservative rectangular region for the first offline study.
SAFE_EVERY=(1,10)

def telemetry():
    q="index,temperature.gpu,power.draw,enforced.power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used"
    try: out=subprocess.check_output(["nvidia-smi",f"--query-gpu={q}","--format=csv,noheader,nounits"],text=True,timeout=3)
    except Exception:return {}
    for line in out.splitlines():
        x=[v.strip() for v in line.split(',')]
        if x and x[0]=='0': return dict(zip(("gpu","temperature","power","power_limit","sm_clock","mem_clock","util","mem_util","mem_used"),map(float,x)))
    return {}

class LammpsGpuEnv:
    """No ground-truth contention label is ever included in observations."""
    def __init__(self, steps=500, log=None):
        self.steps=steps; self.log=Path(log or tempfile.mktemp(prefix="lmp-rl-",suffix=".log")); self.lmp=None; self.skin=.6; self.every=5; self.previous={}
    def _load(self):
        d=tempfile.mkdtemp(prefix="lmp-py-"); pkg=Path(d)/"lammps"; shutil.copytree(ROOT/"python/lammps",pkg); (pkg/"liblammps.so").symlink_to(ROOT/"build_kokkos_cuda_rl/liblammps.so"); sys.path.insert(0,d); from lammps import lammps; return lammps
    def reset(self, skin=.6, every=5):
        self.close(); self.skin=float(skin); self.every=int(every); L=self._load()
        self.lmp=L(cmdargs=["-k","on","g","1","-sf","kk","-pk","kokkos","gpu/aware","off","-log",str(self.log),"-screen","none"])
        c=["units lj","atom_style atomic","lattice fcc 0.8442","region box block 0 40 0 40 0 20","create_box 1 box","create_atoms 1 box","mass 1 1.0","velocity all create 1.44 87287 loop geom","pair_style lj/cut 2.5","pair_coeff 1 1 1.0 1.0 2.5",f"neighbor {self.skin} bin",f"neigh_modify delay 0 every {self.every} check yes","fix integrator all nve","thermo 500","thermo_style custom step temp pe ke etotal press","timer normal"]
        self.lmp.commands_list(c); self._run(); return self.obs()
    def obs(self): return {"skin":self.skin,"every":self.every,"previous":self.previous,"gpu":telemetry()}
    def _run(self):
        t=time.perf_counter(); self.lmp.command(f"run {self.steps}"); wall=time.perf_counter()-t
        temp=float(self.lmp.get_thermo("temp")); pe=float(self.lmp.get_thermo("pe"))
        lammps_metrics=parse_log(self.log)
        dangerous=int(lammps_metrics.get("dangerous_builds",0))
        bad=(not all(map(lambda x:x==x and abs(x)<1e100,(temp,pe)))) or dangerous > 0
        self.previous={
            "wall_seconds":wall,
            "temperature":temp,
            "pe":pe,
            "throughput":self.steps/wall,
            "lammps":lammps_metrics,
            "neighbor_builds":int(lammps_metrics.get("neighbor_builds",0)),
            "dangerous_builds":dangerous,
        }
        return bad
    def step(self, action):
        ds=float(action[0]); de=int(action[1]); ns=min(max(self.skin+ds,SAFE_SKIN[0]),SAFE_SKIN[1]); ne=min(max(self.every+de,SAFE_EVERY[0]),SAFE_EVERY[1]);
        applied=(ns-self.skin,ne-self.every); self.skin,self.every=ns,ne
        self.lmp.command(f"neighbor {ns} bin"); self.lmp.command(f"neigh_modify delay 0 every {ne} check yes")
        bad=self._run(); info={"requested_action":(ds,de),"applied_action":applied,"configuration":(ns,ne),"unsafe":bad}
        return self.obs(),-self.previous["wall_seconds"],bad,info
    def close(self):
        if self.lmp:
            try:self.lmp.close()
            finally:self.lmp=None

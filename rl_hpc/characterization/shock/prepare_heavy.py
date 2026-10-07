#!/usr/bin/env python3
"""Make a shock restart, optionally lengthening only the propagation axis."""
import argparse
import json
from pathlib import Path
import shutil

import shock_counterfactual as base


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--nx',type=int,default=240)
    ap.add_argument('--ny',type=int,default=16)
    ap.add_argument('--nz',type=int,default=16)
    ap.add_argument('--velocity-seed',type=int,default=87287)
    a=ap.parse_args()
    if min(a.nx,a.ny,a.nz)<1:ap.error('lattice dimensions must be positive')
    if a.velocity_seed<1:ap.error('velocity seed must be positive')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    src=base.SOURCE
    params=(src/'shockparams.mod').read_text()
    import re
    params,n1=re.subn(r'(variable\s+nx\s+index\s+)\d+',lambda m:m.group(1)+str(a.nx),params)
    params,n2=re.subn(r'(variable\s+ny\s+index\s+)\d+',lambda m:m.group(1)+str(a.ny),params)
    params,n3=re.subn(r'(variable\s+nz\s+index\s+)\d+',lambda m:m.group(1)+str(a.nz),params)
    assert (n1,n2,n3)==(1,1,1)
    (out/'shockparams.mod').write_text(params)
    setup=(src/'shocksetup.mod').read_text()
    setup,nseed=re.subn(r'(velocity\s+all\s+create\s+\$\{temp\}\s+)87287',
                        lambda m:m.group(1)+str(a.velocity_seed),setup)
    assert nseed==1
    (out/'shocksetup.mod').write_text(setup)
    script=base.baseline_input(out,(1100,))
    (out/'in.baseline').write_text(script)
    cmd=base.mpi_command(32,out/'in.baseline',out/'log.lammps')
    rc,wall=base.run_command(cmd,out,out/'stdout.txt',out/'stderr.txt',timeout=1800)
    status={'return_code':rc,'wall_seconds':wall,'atoms_expected':4*a.nx*a.ny*a.nz,
            'nx':a.nx,'ny':a.ny,'nz':a.nz,
            'velocity_seed':a.velocity_seed,
            'checkpoint':str(out/'checkpoint-S1.restart'),'command':cmd,
            'size_bytes':(out/'checkpoint-S1.restart').stat().st_size if (out/'checkpoint-S1.restart').exists() else None}
    (out/'result.json').write_text(json.dumps(status,indent=2))
    if rc:raise SystemExit(rc)
    print(status)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Install the COMPLETE new workflow into a local symm-fm clone, without overwriting originals.

Creates extensions/trjepa and a root run_trjepa.py launcher. Does not modify
old experimental contracts, checkpoints, patient data or other workflow packages.
"""
from pathlib import Path
import argparse
import shutil
import subprocess

PIN="92265d3b1749ae3b686f2089843c49da129fd4d2"

def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True);p.add_argument("--allow-other-revision",action="store_true")
    a=p.parse_args();root=Path(a.repo).resolve();source=Path(__file__).resolve().parents[1]
    if not (root/"workflows/first_post_three_phase/mewm_ispy2/three_phase_symmflow.py").exists():
        raise SystemExit("Not the expected symm-fm repository layout")
    revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if revision!=PIN and not a.allow_other_revision:
        raise SystemExit(f"Inspected revision is {PIN}; current revision is {revision}. Review compatibility before overriding.")
    dest=root/"extensions/trjepa";launcher=root/"run_trjepa.py"
    if dest.exists() or launcher.exists():raise SystemExit("Refusing to overwrite an existing extension/launcher")
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","runs","*.pyc"))
    launcher.write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/"extensions/trjepa"))\nfrom trjepa.cli import main\nif __name__ == "__main__": main()\n')
    print(f"Installed {dest}; original files were not changed. Run: python {launcher} --help")

if __name__=="__main__":main()

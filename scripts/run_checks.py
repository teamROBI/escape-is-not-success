#!/usr/bin/env python3
"""Run the release's CPU-only checks without importing the simulator."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STDLIB = [
    'escape_ood/suites/tests/test_spec.py',
    'escape_ood/suites/tests/test_layout_gen.py',
    'escape_ood/collect/tests/test_context.py',
    'escape_ood/collect/tests/test_tasks.py',
    'escape_ood/metrics/tests/test_extent.py',
    'escape_ood/sim/env/tests/test_scene_symmetry.py',
    'escape_ood/sim/tools/tests/test_spawn_rotation.py',
]
NUMPY = [
    'escape_ood/collect/tests/test_recorder.py',
    'escape_ood/splits/tests/test_archive_naming.py',
    'escape_ood/splits/tests/test_eef_abs_rotvec.py',
    'escape_ood/policy/act/tests/test_chunk_anchor.py',
]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--numpy', action='store_true', help='include numpy/h5py tests')
    args = ap.parse_args()
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    failures = []
    tests = STDLIB + (NUMPY if args.numpy else [])
    for name in tests:
        run = subprocess.run([sys.executable, str(ROOT / name)], cwd=ROOT, env=env,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(('PASS' if run.returncode == 0 else 'FAIL'), name)
        if run.returncode:
            print(run.stdout)
            failures.append(name)
    print(f'{len(tests) - len(failures)}/{len(tests)} test programs passed')
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())

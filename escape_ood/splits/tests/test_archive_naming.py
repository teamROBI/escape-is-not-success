# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Partitioned training archives must be selected without admitting evaluation archives."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/bench'))
from convert_act import _select_archives

CASES = {
    'lift': True, 'lift_p0of4': True, 'lift_p3of4': True,
    'lift_p0of2': True, 'lift_p11of16': True,
    'lift_interior': False, 'lift_test': False,
    'lift_interior+test': False, 'lift_test_p0of2': False,
    'lift_test_p1of2': False, 'lift_interior+test_p2of4': False,
}

def main():
    with tempfile.TemporaryDirectory() as td:
        col = Path(td)
        for stem in CASES:
            (col / f'{stem}.hdf5').touch()
        (col / 'lift-unrelated_p0of4.hdf5').touch()
        selected = {p.stem for p in _select_archives(col, 'lift', 'train')}
        assert selected == {stem for stem, train in CASES.items() if train}, selected
        for split in ('interior', 'test'):
            selected = {p.stem for p in _select_archives(col, 'lift', split)}
            assert {stem for stem in CASES if split in stem} <= selected
            assert not any(stem.startswith('lift-') for stem in selected)
    print(f'PASS: {len(CASES)} archive names; unrelated archives excluded')

if __name__ == '__main__':
    main()

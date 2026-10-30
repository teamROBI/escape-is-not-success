# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
import re
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
SCENE_PY = REPO_ROOT / 'escape_ood' / 'sim' / 'env' / 'scene.py'
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    print(f"  {('ok   ' if cond else 'FAIL ')} {label}")
    if not cond:
        FAILURES.append(label)

def light_direction(q) -> list[float]:
    """Emission direction of a USD DistantLight with orientation ``q`` = (w, x, y, z).

    A DistantLight emits along its own -Z, so the direction is the third column of the rotation
    matrix, negated.
    """
    (w, x, y, z) = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    (w, x, y, z) = (w / n, x / n, y / n, z / n)
    return [-(2 * (x * z + w * y)), -(2 * (y * z - w * x)), -(1 - 2 * (x * x + y * y))]

def read_sun_rot() -> tuple[float, ...]:
    src = SCENE_PY.read_text()
    m = re.search('^SUN_ROT\\s*=\\s*\\(([^)]*)\\)', src, re.M)
    if not m:
        raise AssertionError('SUN_ROT not found in scene.py')
    return tuple((float(v) for v in m.group(1).split(',')))

def main() -> int:
    print('\nthe sun carries no lateral component')
    q = read_sun_rot()
    check(len(q) == 4, f'SUN_ROT is a quaternion: {q}')
    norm = math.sqrt(sum((c * c for c in q)))
    check(abs(norm - 1.0) < 1e-05, f'it is normalised: |q| = {norm:.6f} (the previous value was 0.9807)')
    check(abs(q[1]) < 1e-12 and abs(q[3]) < 1e-12, f'it is a pure rotation about Y, so no lateral tilt is possible: x={q[1]}, z={q[3]}')
    d = light_direction(q)
    check(abs(d[1]) < 1e-09, f'light direction has zero y: ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})')
    check(d[2] < -0.1, f'it points downward: z = {d[2]:+.4f}')
    elev = math.degrees(math.asin(abs(d[2])))
    check(10.0 < elev < 80.0, f'elevation is a usable {elev:.1f} degrees above the horizon')
    print('\nthe scene the sun lights is itself symmetric')
    src = SCENE_PY.read_text()
    directional = re.findall('(DistantLightCfg|SphereLightCfg|DiskLightCfg|RectLightCfg)', src)
    check(directional.count('DistantLightCfg') == 1 and len(directional) == 1, f'exactly one directional light to keep symmetric: {directional}')
    old = light_direction((0.87, 0.32, 0.32, 0.0))
    check(abs(old[1]) > 0.5, f'the previous orientation really was asymmetric (y = {old[1]:+.4f}), so this test would have caught it')
    print()
    if FAILURES:
        print(f'[FAIL] {len(FAILURES)} check(s):')
        for f in FAILURES:
            print('  -', f)
        return 1
    print('[PASS] scene lighting is left-right symmetric')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.metrics.extent import Band, action_extent, curve, direction_of, escape_distance, reach_fraction, strict_success, summarise, wilson
from escape_ood.suites.spec import ENVELOPE, Envelope, polar_xy
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    print(f"  {('ok   ' if cond else 'FAIL ')} {label}")
    if not cond:
        FAILURES.append(label)

def test_wilson() -> None:
    print('\n[wilson]')
    (lo, hi) = wilson(0, 10)
    check(lo == 0.0 and 0.25 < hi < 0.35, f'0/10 -> [0, ~0.28] (got {lo:.3f},{hi:.3f})')
    (lo, hi) = wilson(10, 10)
    check(hi == 1.0 and 0.65 < lo < 0.75, f'10/10 -> [~0.72, 1] (got {lo:.3f},{hi:.3f})')
    (lo, hi) = wilson(5, 10)
    check(abs((lo + hi) / 2 - 0.5) < 0.01, '5/10 is symmetric about 0.5')
    check(wilson(0, 0) == (0.0, 0.0), 'zero trials -> (0,0), not a division error')

def test_reach_fraction() -> None:
    print('\n[reach fraction = target approach]')
    e = Envelope(r_min=0.325, r_max=0.525, theta_max_deg=35.0)
    line = lambda r0, r1, th, n=60: [polar_xy(r0 + (r1 - r0) * i / n, th) for i in range(n + 1)]
    tgt = polar_xy(0.725, 0.0)
    inside = [polar_xy(r, th) for r in (0.35, 0.45, 0.5) for th in (-20.0, 0.0, 20.0)]
    check(reach_fraction(inside, tgt, e) == 0.0, 'a path that never leaves the envelope -> 0')
    check(abs(reach_fraction(line(0.4, 0.725, 0.0), tgt, e) - 1.0) < 1e-09, 'a path that reaches the target -> 1')
    check(abs(reach_fraction(line(0.4, 0.625, 0.0), tgt, e) - 0.5) < 1e-09, 'a path stopping 10 cm out of 20 -> 0.5')
    check(abs(reach_fraction(line(0.4, 0.9, 0.0), tgt, e) - 1.0) < 1e-09, 'passing through the target and beyond -> 1')
    check(reach_fraction(line(0.4, 0.725, 0.0), polar_xy(0.45, 0.0), e) is None, 'an interior target has no reach fraction (None)')
    tgt_a = polar_xy(0.5, 65.0)
    d_a = 0.5 * math.sin(math.radians(30.0))
    swing = [polar_xy(0.5, th) for th in (0.0, 20.0, 40.0, 50.0)]
    want = 1.0 - 2 * 0.5 * math.sin(math.radians(7.5)) / d_a
    check(abs(reach_fraction(swing, tgt_a, e) - want) < 1e-09, f'angular partial swing -> {want:.3f}')
    tgt_r5 = polar_xy(0.575, 0.0)
    retract = [polar_xy(0.45, 0.0), polar_xy(0.257, 13.8)]
    check(reach_fraction(retract, tgt_r5, e) == 0.0, 'pulling in toward the base past r_min is not reach (any-direction numerator: 1.00)')
    sideways = [polar_xy(0.45, 0.0), polar_xy(0.5, 60.0)]
    check(reach_fraction(sideways, tgt_r5, e) == 0.0, 'swinging out sideways is not reach toward a radial target (any-direction: 1.00)')
    check(reach_fraction([polar_xy(0.55, 60.0)], tgt_r5, e) == 0.0, 'a corner exit far from the target is not reach (any-direction: 1.00; radial-axis: 0.50)')
    tgt_r10 = polar_xy(0.625, 0.0)
    check(reach_fraction(line(0.4, 0.625, 30.0), tgt_r10, e) == 0.0, 'the right distance out on the WRONG bearing is not reach (radial-axis numerator: 1.00)')
    check(reach_fraction([polar_xy(0.5, -50.0)], tgt_a, e) == 0.0, 'an angular target is not reached by swinging out the other side (any-direction: 0.50)')
    check(reach_fraction([polar_xy(0.2, 60.0)], tgt_a, e) == 0.0, 'a wide bearing inside r_min is not reach toward an angular target')
    check(escape_distance(inside, e) == 0.0, 'escape: a path inside the envelope -> 0')
    check(abs(escape_distance(line(0.4, 0.575, 0.0), e) - 0.05) < 1e-09, 'escape: 5 cm straight out -> 0.05')
    check(abs(escape_distance(retract, e) - (0.325 - 0.257)) < 1e-09, 'escape counts pulling in past r_min (it is not target-directed)')
    check(direction_of(tgt, e) == 'radial' and direction_of(tgt_a, e) == 'angular' and (direction_of(polar_xy(0.45, 0.0), e) == 'inside'), 'direction_of classifies targets')

def test_strict_success() -> None:
    print('\n[strict success]')
    check(strict_success({'success': True, 'steps': 300}), 'early exit after the hold -> success')
    check(not strict_success({'success': True, 'steps': 600}), 'legacy record that ended AT the cap -> not certified (conservative)')
    check(not strict_success({'success': False, 'steps': 200}), 'a failure stays a failure')
    check(strict_success({'success': True, 'steps': 600, 'success_sustained': True}), 'a logged sustained hold at the cap -> success')
    check(not strict_success({'success': True, 'steps': 600, 'success_sustained': False}), 'a logged one-check pass at the cap -> not success')
    recs = [{'sweep_coord': 5.0, 'success': True, 'steps': 600, 'target_xy': [], 'eef_path': []}, {'sweep_coord': 5.0, 'success': True, 'steps': 200, 'target_xy': [], 'eef_path': []}]
    check(curve(recs)[0].n_success == 1, 'curve() counts through strict_success')

def _rec(coord, ok, tgt, path):
    return {'sweep_coord': coord, 'success': ok, 'target_xy': [list(tgt)], 'eef_path': [list(p) for p in path]}

def test_curve_and_extent() -> None:
    print('\n[curve and extent]')
    e = ENVELOPE
    recs = []
    for (coord, k) in ((5.0, 7), (10.0, 7), (15.0, 6), (20.0, 2), (25.0, 0)):
        tgt = polar_xy(0.525 + coord / 100, 0.0)
        for i in range(7):
            ok = i < k
            path = [polar_xy(0.4, 0.0), tgt] if ok else [polar_xy(0.4, 0.0), polar_xy(0.5, 0.0)]
            recs.append(_rec(coord, ok, tgt, path))
    for i in range(10):
        recs.append(_rec(0.0, True, polar_xy(0.425, 0.0), [polar_xy(0.425, 0.0)]))
    bands = curve(recs, e)
    check([b.sweep_coord for b in bands] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0], 'bands sorted by coord')
    b5 = next((b for b in bands if b.sweep_coord == 5.0))
    check(b5.n == 7 and b5.n_success == 7 and (b5.reach_median == 1.0) and (b5.reach_zero_frac == 0.0), 'a fully reached band: median reach 1, never-tried 0')
    b25 = next((b for b in bands if b.sweep_coord == 25.0))
    check(b25.reach_median == 0.0 and b25.reach_zero_frac == 1.0, 'an orbiting band: reach 0, never-tried 1')
    b0 = next((b for b in bands if b.sweep_coord == 0.0))
    check(b0.reach_median is None, 'the inside band has no reach fraction')
    ext = action_extent(bands, direction='radial', threshold=0.5)
    check(ext.extent_cm == 10.0 and ext.bands_passed == 2 and (not ext.saturated), f'extent walks outward and stops at the first failing band ({ext.summary()})')
    ext_lo = action_extent(bands, direction='radial', threshold=0.3)
    check(ext_lo.extent_cm == 15.0, 'a lower threshold admits the 6/7 band')
    all_pass = [Band(c, 7, 7, 1.0, 0.65, 1.0, 1.0, 0.0) for c in (5.0, 10.0)]
    check(action_extent(all_pass, direction='radial').saturated, 'every band passing is flagged saturated')
    s = summarise({'records': recs}, e)
    check(s['interior_control']['n'] == 10 and abs(s['interior_control']['rate'] - 1.0) < 1e-09, 'summary carries the interior control')
    check('radial' in s['directions'] and 'angular' not in s['directions'], 'summary groups by direction and omits absent ones')

def main() -> int:
    test_wilson()
    test_reach_fraction()
    test_strict_success()
    test_curve_and_extent()
    print()
    if FAILURES:
        print(f'[FAIL] {len(FAILURES)} check(s) failed:')
        for f in FAILURES:
            print('  -', f)
        return 1
    print('[PASS] all checks passed')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

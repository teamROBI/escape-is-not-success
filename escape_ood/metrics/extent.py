# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Action extent: how far beyond its training envelope a policy still acts.

Three numbers, computed from the evaluator's per-episode records (``eef_path``,
``target_xy``, ``success``, ``sweep_coord``) and the spec's envelope. No re-rollout needed.

Three per-episode stages, because one success bit collapses them:

**Escape** -- did the grasp point leave the envelope at all, in any direction, and how far.
A policy can leave it without heading for the object.

**Reach fraction** (target approach) -- ``1 - min_t |p_t - o| / d(o, E)``, clipped to [0, 1], in
the table plane: how much of the gap between the envelope and the object the grasp point
closed at its closest. ~0: it never got nearer the object than the envelope edge already is.
~1: it got to the object (and may still have failed to lift it). It is what separates *never
approached* from *approached and failed*, which a success rate cannot see.

It was first defined as the farthest excursion outside the envelope over the target's own
distance outside. That credited leaving the envelope the wrong way -- inward past ``r_min``,
out the far side, or out at the right distance on the wrong bearing -- and so measured escape,
not approach. Replaced 2026-09-28; ``tests/test_extent.py`` pins each case.

**Success** -- the evaluator's predicate.

**Success curve** -- per band (distinct ``sweep_coord``), with Wilson intervals. Wilson rather
than normal because these rates live at 0 and 1.

**Action extent** -- the distance beyond the envelope at which success stops. Walks the bands
outward from the boundary and reports the last band whose Wilson LOWER bound clears the
threshold before the first that does not; a lone far band that passes by luck does not extend
it. Reported separately per direction (radial, angular), because the two need not agree.

Stdlib only, like every metric here; the tests are a self-contained script.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence
from escape_ood.suites.spec import ENVELOPE, Envelope
XY = tuple[float, float]

def wilson(successes: int, trials: int, z: float=1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate. (0, 0) for zero trials."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def reach_fraction(eef_path: Sequence[Sequence[float]], target_xy: XY, envelope: Envelope=ENVELOPE) -> float | None:
    """Target approach: ``1 - (closest the grasp point came to the object) / (object's distance
    from the envelope)``, clipped to [0, 1], in the table plane.

    ``eef_path`` is a sequence of ``[x, y, (z)]`` in the base frame. Returns ``None`` when the
    target is inside the envelope (an interior/train condition: the ratio is undefined and the
    number would mean nothing). Leaving the envelope in some other direction scores nothing;
    that is :func:`escape_distance`.
    """
    (tx, ty) = (float(target_xy[0]), float(target_xy[1]))
    d_target = envelope.distance_outside(tx, ty)
    if d_target <= 1e-06:
        return None
    closest = min((math.hypot(float(p[0]) - tx, float(p[1]) - ty) for p in eef_path), default=math.inf)
    return min(max(1.0 - closest / d_target, 0.0), 1.0)

def escape_distance(eef_path: Sequence[Sequence[float]], envelope: Envelope=ENVELOPE) -> float:
    """Metres the grasp point ever got outside the envelope, in ANY direction; 0 if it never
    left. Deliberately not target-directed: it answers "did the policy leave its training
    region at all", which a policy can do without approaching the object."""
    return max([0.0] + [envelope.boundary_distance(float(p[0]), float(p[1])) for p in eef_path])
MAX_STEPS = 600

def strict_success(record: dict, max_steps: int=MAX_STEPS) -> bool:
    """Success that certifies a SUSTAINED lift (hold_checks consecutive passing checks).

    The evaluator's ``success`` also accepts a single passing check at the final step. Records
    written from 2026-09-29 carry ``success_sustained``, which is exact. Older records do not;
    for them ``success and steps < max_steps`` is used -- a conservative lower bound, since a hold
    completed exactly at the cap is indistinguishable from a one-check pass there. Every table and
    figure counts success through this one function.
    """
    sustained = record.get('success_sustained')
    if sustained is not None:
        return bool(sustained)
    steps = record.get('steps')
    return bool(record.get('success')) and (steps is None or int(steps) < max_steps)

def direction_of(target_xy: XY, envelope: Envelope=ENVELOPE) -> str:
    """'radial' | 'angular' | 'inside' | 'corner' | 'near' for a target."""
    return envelope.region(float(target_xy[0]), float(target_xy[1]))

@dataclass(frozen=True)
class Band:
    sweep_coord: float
    n: int
    n_success: int
    rate: float
    ci_low: float
    ci_high: float
    reach_median: float | None
    reach_zero_frac: float | None
    escape_frac: float | None = None

    def summary(self) -> str:
        rf = '   --  ' if self.reach_median is None else f'{self.reach_median:5.2f}'
        nz = '   --  ' if self.reach_zero_frac is None else f'{self.reach_zero_frac:5.2f}'
        ef = '   --  ' if self.escape_frac is None else f'{self.escape_frac:5.2f}'
        return f'd={self.sweep_coord:5.1f}cm  {self.n_success:3d}/{self.n:<3d} {100 * self.rate:5.1f}%  [{100 * self.ci_low:5.1f},{100 * self.ci_high:5.1f}]  left {ef}  approach {rf}  no-approach {nz}'

@dataclass(frozen=True)
class Extent:
    direction: str
    threshold: float
    extent_cm: float
    bands_passed: int
    bands_total: int
    saturated: bool
    note: str = ''

    def summary(self) -> str:
        s = f'{self.direction}: extent {self.extent_cm:.0f} cm ({self.bands_passed}/{self.bands_total} bands with Wilson-lower >= {self.threshold})'
        return s + (' -- SATURATED, extend the sweep' if self.saturated else '') + (f' -- {self.note}' if self.note else '')

def curve(records: Iterable[dict], envelope: Envelope=ENVELOPE) -> list[Band]:
    """Per-band success and reach statistics from evaluator records."""
    groups: dict[float, list[dict]] = {}
    for r in records:
        groups.setdefault(round(float(r['sweep_coord']), 4), []).append(r)
    out: list[Band] = []
    for (coord, rs) in sorted(groups.items()):
        n = len(rs)
        k = sum((1 for r in rs if strict_success(r)))
        (lo, hi) = wilson(k, n)
        (reaches, escapes) = ([], [])
        for r in rs:
            tgt = r.get('target_xy') or []
            path = r.get('eef_path') or []
            if not tgt or not path:
                continue
            rf = reach_fraction(path, tuple(tgt[0]), envelope)
            if rf is not None:
                reaches.append(rf)
                escapes.append(escape_distance(path, envelope))
        if reaches:
            srt = sorted(reaches)
            med = srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2])
            nz = sum((1 for x in reaches if x < 0.25)) / len(reaches)
            esc = sum((1 for x in escapes if x > 0.01)) / len(escapes)
        else:
            (med, nz, esc) = (None, None, None)
        out.append(Band(coord, n, k, k / n, lo, hi, med, nz, esc))
    return out

def action_extent(bands: Sequence[Band], *, direction: str, threshold: float=0.5) -> Extent:
    """Largest band distance reached with the Wilson lower bound still >= threshold, walking
    outward and stopping at the first failure. Bands at coord 0 (inside) are not counted."""
    outside = [b for b in bands if b.sweep_coord > 0]
    if not outside:
        return Extent(direction, threshold, 0.0, 0, 0, False, 'no test bands')
    (passed, last) = (0, 0.0)
    for b in sorted(outside, key=lambda b: b.sweep_coord):
        if b.ci_low >= threshold:
            passed += 1
            last = b.sweep_coord
        else:
            break
    return Extent(direction, threshold, last, passed, len(outside), passed == len(outside))

def summarise(result: dict, envelope: Envelope=ENVELOPE, threshold: float=0.5) -> dict:
    """Everything the report needs from one evaluator JSON: per-direction curves and extents,
    plus the interior control the curves are read against."""
    recs = list(result.get('records', []))
    by_dir: dict[str, list[dict]] = {}
    for r in recs:
        tgt = r.get('target_xy') or []
        d = direction_of(tuple(tgt[0]), envelope) if tgt else 'inside'
        by_dir.setdefault(d, []).append(r)
    out: dict = {'n': len(recs), 'directions': {}}
    inside = by_dir.get('inside', [])
    if inside:
        k = sum((1 for r in inside if strict_success(r)))
        (lo, hi) = wilson(k, len(inside))
        out['interior_control'] = {'n': len(inside), 'n_success': k, 'rate': k / len(inside), 'ci_low': lo, 'ci_high': hi}
    for d in ('radial', 'angular'):
        rs = by_dir.get(d, [])
        if not rs:
            continue
        bands = curve(rs, envelope)
        ext = action_extent(bands, direction=d, threshold=threshold)
        out['directions'][d] = {'bands': [b.__dict__ for b in bands], 'extent': ext.__dict__}
    return out

def format_summary(s: dict) -> str:
    lines = []
    ic = s.get('interior_control')
    if ic:
        lines.append(f"interior control: {ic['n_success']}/{ic['n']} = {100 * ic['rate']:.1f}% [{100 * ic['ci_low']:.1f},{100 * ic['ci_high']:.1f}]" + ('   ** below 0.8: extents are not interpretable **' if ic['rate'] < 0.8 else ''))
    for (d, blk) in s.get('directions', {}).items():
        lines.append(f'[{d}]')
        for b in blk['bands']:
            lines.append('  ' + Band(**b).summary())
        lines.append('  ' + Extent(**blk['extent']).summary())
    return '\n'.join(lines)

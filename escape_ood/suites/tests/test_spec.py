# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
import re
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.suites import spec
from escape_ood.suites.spec import ANGULAR_BAND_THETA, ENVELOPE, INTERIOR_R, RADIAL_BAND_R, SUITES, TRAIN_R, TRAIN_THETA, WORKSPACE, Condition, Envelope, band_label, enumerate_conditions, expected_train_episodes, instruction_from_attrs, is_withheld_from_training, lang_obs_key, object_keys_for, object_start_xy, params_from_attrs, polar_xy, sample_sweep, suite, target_xys, xy_polar
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    if cond:
        print(f'  ok    {label}')
    else:
        print(f'  FAIL  {label}')
        FAILURES.append(label)

def _cells(s, split=None):
    return [p for p in s.grid() if split is None or s.split(p) == split]

def test_registry() -> None:
    print('\n[registry]')
    check(len(SUITES) >= 1, f'{len(SUITES)} suite(s) registered')
    check(all((s.name == n for (n, s) in SUITES.items())), 'registry keys match suite names')
    check(all((s.success_kind for s in SUITES.values())), 'every suite names its success kind')
    check(all((object_keys_for(n) for n in SUITES)), 'every suite names at least one object')
    try:
        suite('nope')
        check(False, 'unknown suite raises KeyError')
    except KeyError:
        check(True, 'unknown suite raises KeyError')

def test_grids_and_splits() -> None:
    print('\n[grids]')
    for (name, s) in SUITES.items():
        grid = list(s.grid())
        n = {sp: sum((1 for p in grid if s.split(p) == sp)) for sp in ('train', 'interior', 'test')}
        check(len(grid) > 0, f'{name}: grid non-empty ({len(grid)})')
        check(n['train'] > 0, f"{name}: has training cells ({n['train']})")
        check(n['interior'] > 0, f"{name}: has interior-ID cells ({n['interior']})")
        check(n['test'] > 0, f"{name}: has test cells ({n['test']})")
        keys = [(round(float(p['r']), 6), round(float(p['theta_deg']), 6)) for p in grid]
        check(len(set(keys)) == len(keys), f'{name}: no duplicate cells')

def test_sweep_coord() -> None:
    print('\n[sweep_coord]')
    for (name, s) in SUITES.items():
        inside = _cells(s, 'train') + _cells(s, 'interior')
        worst = max((abs(s.sweep_coord(p)) for p in inside))
        check(worst < 1e-06, f'{name}: coord==0 for all {len(inside)} train+interior cells')
        test = _cells(s, 'test')
        check(all((s.sweep_coord(p) >= 5.0 - 1e-06 for p in test)), f'{name}: every test cell is >= 5 cm outside ({len(test)} cells)')

def test_envelope_geometry() -> None:
    """Hand-computed points against boundary_distance, so the metric's definition is pinned."""
    print('\n[envelope: boundary_distance]')
    e = Envelope(r_min=0.3, r_max=0.5, theta_max_deg=30.0)
    (x, y) = polar_xy(0.6, 0.0)
    check(e.region(x, y) == 'radial' and math.isclose(e.boundary_distance(x, y), 0.1, abs_tol=1e-09), 'radial: 0.60 m at 0 deg is 10 cm out')
    (x, y) = polar_xy(0.4, 45.0)
    want = 0.4 * math.radians(15.0)
    check(e.region(x, y) == 'angular' and math.isclose(e.boundary_distance(x, y), want, abs_tol=1e-09), f'angular: 0.40 m at 45 deg is arc {want * 100:.2f} cm out')
    check(math.isclose(e.boundary_distance(*polar_xy(0.4, -45.0)), want, abs_tol=1e-09), 'angular distance is symmetric in theta')
    (x, y) = polar_xy(0.4, 0.0)
    check(math.isclose(e.boundary_distance(x, y), -0.1, abs_tol=1e-09), 'inside: 0.40 m at 0 deg is 10 cm from the nearest (radial) boundary')
    (x, y) = polar_xy(0.4, 25.0)
    want = -0.4 * math.radians(5.0)
    check(math.isclose(e.boundary_distance(x, y), want, abs_tol=1e-09), 'inside: near the sector edge the nearest boundary is angular (arc length)')
    (x, y) = polar_xy(0.6, 60.0)
    (cx, cy) = polar_xy(0.5, 30.0)
    check(e.region(x, y) == 'corner' and math.isclose(e.boundary_distance(x, y), math.hypot(x - cx, y - cy), abs_tol=1e-09), 'corner: euclidean distance to the nearest envelope corner')
    check(e.region(*polar_xy(0.2, 0.0)) == 'near' and math.isclose(e.boundary_distance(*polar_xy(0.2, 0.0)), 0.1, abs_tol=1e-09), 'near: closer than r_min is 10 cm out (inward)')
    (r, th) = xy_polar(*polar_xy(0.42, -17.0))
    check(math.isclose(r, 0.42, abs_tol=1e-09) and math.isclose(th, -17.0, abs_tol=1e-09), 'polar_xy / xy_polar round-trip')

def test_distance_outside() -> None:
    """The reach-fraction denominator: straight-line distance from a point to the envelope."""
    print('\n[envelope: distance_outside]')
    e = Envelope(r_min=0.3, r_max=0.5, theta_max_deg=30.0)
    check(e.distance_outside(*polar_xy(0.4, 10.0)) == 0.0, 'inside -> 0')
    pts = [polar_xy(r, th) for r in (0.52, 0.6, 0.8) for th in (-25.0, 0.0, 25.0)]
    check(all((math.isclose(e.distance_outside(*p), e.boundary_distance(*p), abs_tol=1e-12) for p in pts)), 'radial region: equals boundary_distance (r - r_max)')
    check(math.isclose(e.distance_outside(*polar_xy(0.45, 50.0)), 0.45 * math.sin(math.radians(20.0)), abs_tol=1e-12), 'angular: perpendicular distance to the theta_max ray')
    pts = [polar_xy(r, th) for r in (0.32, 0.4, 0.48) for th in (35.0, 50.0, -70.0)]
    check(all((e.distance_outside(*p) <= e.boundary_distance(*p) + 1e-12 for p in pts)), 'angular: the chord never exceeds the arc')
    (x, y) = polar_xy(0.6, 60.0)
    (cx, cy) = polar_xy(0.5, 30.0)
    check(math.isclose(e.distance_outside(x, y), math.hypot(x - cx, y - cy), abs_tol=1e-12), 'corner: Euclidean to the envelope corner')
    check(math.isclose(e.distance_outside(*polar_xy(0.2, 0.0)), 0.1, abs_tol=1e-12), 'near: r_min - r on a trained bearing')
    (x, y) = polar_xy(0.31, 60.0)
    (cx, cy) = polar_xy(0.3, 30.0)
    check(math.isclose(e.distance_outside(x, y), math.hypot(x - cx, y - cy), abs_tol=1e-12) and e.distance_outside(x, y) > 0.31 * math.sin(math.radians(30.0)), 'angular near r_min: the perpendicular foot falls short of r_min, so the inner corner is nearest')
    tm = math.radians(30.0)
    bd = []
    for i in range(4001):
        t = -tm + 2 * tm * i / 4000
        bd += [polar_xy(0.3, math.degrees(t)), polar_xy(0.5, math.degrees(t))]
        r = 0.3 + 0.2 * i / 4000
        bd += [polar_xy(r, 30.0), polar_xy(r, -30.0)]
    probes = [polar_xy(r, th) for r in (0.1, 0.25, 0.35, 0.55, 0.9) for th in (-170.0, -60.0, -31.0, 45.0, 90.0, 150.0)]
    worst = max((abs(e.distance_outside(*p) - min((math.hypot(p[0] - bx, p[1] - by) for (bx, by) in bd))) for p in probes if e.region(*p) != 'inside'))
    check(worst < 0.0002, f'matches brute force over the boundary (worst {worst:.1e} m)')

def test_tier1_invariants() -> None:
    print('\n[tier 1 invariants]')
    for (name, s) in SUITES.items():
        train = _cells(s, 'train')
        interior = _cells(s, 'interior')
        test = _cells(s, 'test')
        xy = lambda p: polar_xy(float(p['r']), float(p['theta_deg']))
        margins = [ENVELOPE.boundary_distance(*xy(p)) for p in train]
        check(all((m <= -0.02 for m in margins)), f'{name}: every training cell is >= 2 cm inside the envelope (shallowest {max(margins) * 100:.1f} cm)')
        check(all((ENVELOPE.contains(*xy(p)) for p in interior)), f'{name}: every interior cell is inside the envelope')
        tr_xy = [xy(p) for p in train]
        near = min((min((math.hypot(x - tx, y - ty) for (tx, ty) in tr_xy)) for (x, y) in map(xy, interior)))
        check(near >= 0.01, f'{name}: no interior cell within 1 cm of a training cell (nearest {near * 100:.1f} cm)')
        edge = {abs(float(p['theta_deg'])) for p in interior}
        check(max(TRAIN_THETA) in edge, f'{name}: interior-ID includes the +/-{max(TRAIN_THETA):g} deg edge')
        regions = {ENVELOPE.region(*xy(p)) for p in test}
        check(regions <= {'radial', 'angular'}, f'{name}: test regions are radial/angular only ({sorted(regions)})')
        labels = [band_label(p) for p in test]
        check(all(labels), f'{name}: every test cell has a band label')
        check(all((band_label(p) is None for p in train + interior)), f'{name}: no train/interior cell has a band label')
        radial = [p for p in test if ENVELOPE.region(*xy(p)) == 'radial']
        angular = [p for p in test if ENVELOPE.region(*xy(p)) == 'angular']
        check(all((any((abs(float(p['theta_deg']) - t) < 1e-06 for t in TRAIN_THETA)) for p in radial)), f'{name}: radial bands use only trained bearings ({len(radial)} cells)')
        check(all((any((abs(float(p['r']) - r) < 1e-06 for r in TRAIN_R)) for p in angular)), f'{name}: angular bands use only trained radii ({len(angular)} cells)')
        rd = sorted({round(s.sweep_coord(p), 6) for p in radial})
        check(rd == [round(5.0 * (i + 1), 6) for i in range(len(RADIAL_BAND_R))], f'{name}: radial bands sit at exactly 5 cm steps ({rd})')
        mirrored_inside = [p for p in test if ENVELOPE.contains(*polar_xy(float(p['r']), -float(p['theta_deg'])))]
        check(not mirrored_inside, f"{name}: no test cell's mirror lies inside the envelope")
        check(all((WORKSPACE.contains(*xy(p), margin=0.05) for p in s.grid())), f'{name}: every cell is >= 5 cm inside WORKSPACE')

def test_every_cell_passed_the_gate() -> None:
    pass
    import json
    print('\n[gate: every cell demonstrated by the expert]')
    for (name, s) in SUITES.items():
        path = Path(spec.__file__).parent / 'reachability' / name / 'gate.json'
        if not path.is_file():
            print(f'  skip  {name}: {path.relative_to(REPO_ROOT)} not present -- run the gate and copy its per-cell summary there')
            continue
        g = json.load(open(path))
        by_key = {r['key']: bool(r['ok']) for r in g['results']}
        (missing, failed) = ([], [])
        for c in enumerate_conditions(s, object_keys_for(name)[:1]):
            if c.key() not in by_key:
                missing.append(c.key())
            elif not by_key[c.key()]:
                failed.append(c.key())
        check(not missing, f'{name}: every cell was gated ({len(missing)} missing' + (f', e.g. {missing[0]}' if missing else '') + ')')
        check(not failed, f'{name}: every gated cell passed ({len(failed)} failed' + (f', e.g. {failed[0]}' if failed else '') + ')')
        check(g.get('object') == object_keys_for(name)[0], f"{name}: gate was run on the suite's current object ({g.get('object')})")

def test_workspace_not_used_for_splits() -> None:
    """WORKSPACE sizes the table and the probe grid. It must never decide a split."""
    print('\n[layering: WORKSPACE]')
    src = Path(spec.__file__).read_text(encoding='utf-8')
    body = src.split('WORKSPACE = Workspace()', 1)[1]
    check('WORKSPACE.contains' not in body, 'spec.py never calls WORKSPACE.contains after defining it')
    check(re.search('def envelope_split.*?\\n\\n', src, re.S) is not None and 'WORKSPACE' not in re.search('def envelope_split.*?\\n\\n', src, re.S).group(0), 'envelope_split does not reference WORKSPACE')

def test_withholding_and_counts() -> None:
    print('\n[withholding]')
    s = SUITES['lift']
    train = _cells(s, 'train')[0]
    inter = _cells(s, 'interior')[0]
    tests = _cells(s, 'test')
    check(not is_withheld_from_training('lift', train), 'train cell is trained on')
    check(is_withheld_from_training('lift', inter), 'interior cell is withheld')
    check(all((is_withheld_from_training('lift', p) for p in tests)), 'every test cell is withheld by default')
    r2 = [p for p in tests if band_label(p) == 'R2']
    check(r2 and all((not is_withheld_from_training('lift', p, include_bands=('R2',)) for p in r2)), "skyline: include_bands=('R2',) admits exactly that band")
    check(all((is_withheld_from_training('lift', p, include_bands=('R2',)) for p in tests if band_label(p) != 'R2')), 'skyline: other bands stay withheld')
    n_train = len(_cells(s, 'train'))
    check(expected_train_episodes('lift', 15) == n_train * 15, f'expected_train_episodes = {n_train} cells x demos')

def test_attrs_roundtrip() -> None:
    print('\n[attrs round-trip]')
    for (name, s) in SUITES.items():
        conds = enumerate_conditions(s, ['red_block'])
        ok = all((params_from_attrs(c.to_attrs()) == c.params for c in conds))
        check(ok, f'{name}: params round-trip through to_attrs/params_from_attrs ({len(conds)})')
        c = conds[0]
        attrs = c.to_attrs()
        for k in ('suite', 'object_key', 'split', 'sweep_coord', 'instruction'):
            check(k in attrs, f"{name}: attrs carry '{k}'")
        check(all((isinstance(v, (int, float, str)) for v in attrs.values())), f'{name}: all attr values are HDF5-safe scalars')
        check(instruction_from_attrs(attrs) == c.instruction, f'{name}: instruction re-derives from attrs')

def test_enumerate_and_placement() -> None:
    print('\n[enumerate]')
    s = SUITES['lift']
    objs = ['red_block', 'blue_block']
    conds = enumerate_conditions(s, objs)
    check(len(conds) == len(list(s.grid())) * len(objs), 'one condition per (cell x object)')
    check(len({c.key() for c in conds}) == len(conds), 'condition keys are unique')
    for sp in ('train', 'interior', 'test'):
        only = enumerate_conditions(s, objs, splits=(sp,))
        check(only and all((c.split == sp for c in only)), f'split filter honours {sp!r}')
    check(all((c.sweep_coord == 0.0 for c in enumerate_conditions(s, objs, splits=('train', 'interior')))), 'train and interior conditions report coord 0')
    c = conds[0]
    (x, y) = object_start_xy(c)
    (r, th) = xy_polar(x, y)
    check(math.isclose(r, float(c.params['r']), abs_tol=1e-09) and math.isclose(th, float(c.params['theta_deg']), abs_tol=1e-09), 'object_start_xy places the object at the swept polar cell')
    check(target_xys(c) == [], 'lift has no target')
    check(isinstance(c, Condition), 'enumerate returns Condition instances')
    try:
        c.split = 'train'
        check(False, 'Condition is frozen')
    except Exception:
        check(True, 'Condition is frozen')

def test_sample_sweep() -> None:
    print('\n[sample_sweep]')
    s = SUITES['lift']
    objs = ['red_block']
    ss = sample_sweep(s, objs, n_bins=10, per_bin=3, seed=0)
    print(f'        {ss.summary()}')
    n_train = len(_cells(s, 'train'))
    n_int = len(_cells(s, 'interior'))
    check(ss.n_train == n_train and ss.n_interior == n_int, 'all train and interior conditions kept')
    check(all((c.split != 'test' for c in ss.conditions[:n_train + n_int])), 'controls come first')
    check(all((c > 0 for c in ss.bin_counts)), f'no empty sweep bin (counts {ss.bin_counts})')
    again = sample_sweep(s, objs, n_bins=10, per_bin=3, seed=0)
    check([c.key() for c in ss.conditions] == [c.key() for c in again.conditions], 'same seed gives an identical evaluation set')

def test_language() -> None:
    print('\n[language]')
    for (name, s) in SUITES.items():
        texts = {c.instruction for c in enumerate_conditions(s, object_keys_for(name))}
        check(len(texts) == 1, f'{name}: instruction is constant across cells (got {len(texts)})')
        check(not any(('target' in t for t in texts)) or s.target_xy(next(s.grid())) is not None, f'{name}: instruction promises a target only if the suite has one')
    one = lambda n, s: next(iter({c.instruction for c in enumerate_conditions(s, object_keys_for(n))}))
    all_texts = [one(n, s) for (n, s) in SUITES.items()]
    check(len(set(all_texts)) == len(all_texts), 'instructions differ across (non-variant) suites')
    check(lang_obs_key('task') == 'annotation.human.action.task_description', 'task channel key')
    try:
        lang_obs_key('sub_task')
        check(False, 'sub_task channel is gone')
    except ValueError:
        check(True, 'sub_task channel is gone')

def test_obj_phrases_are_curated() -> None:
    print('\n[phrases]')
    from escape_ood.suites.spec import _OBJ_PHRASE
    used = {o for n in SUITES for o in object_keys_for(n)}
    missing = sorted(used - set(_OBJ_PHRASE))
    check(not missing, f'every used asset has a curated phrase (missing {missing})')
    check(all(('_' not in _OBJ_PHRASE[a] and 'escape_ood' not in _OBJ_PHRASE[a] for a in used)), 'phrases carry no prefix or underscore')

def test_no_heavy_imports() -> None:
    print('\n[layering: imports]')
    for mod in ('h5py', 'numpy', 'torch', 'isaaclab'):
        check(mod not in sys.modules, f'importing spec did not pull in {mod}')

def main() -> int:
    test_registry()
    test_grids_and_splits()
    test_sweep_coord()
    test_envelope_geometry()
    test_distance_outside()
    test_tier1_invariants()
    test_every_cell_passed_the_gate()
    test_workspace_not_used_for_splits()
    test_withholding_and_counts()
    test_attrs_roundtrip()
    test_enumerate_and_placement()
    test_sample_sweep()
    test_language()
    test_obj_phrases_are_curated()
    test_no_heavy_imports()
    print()
    if FAILURES:
        print(f'[FAIL] {len(FAILURES)} check(s) failed:')
        for f in FAILURES:
            print(f'  - {f}')
        return 1
    print('[PASS] all checks passed')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

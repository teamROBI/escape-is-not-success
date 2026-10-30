# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import h5py
    import numpy as np
except ImportError:
    print('[ERROR] needs h5py. Run: venvs/sim/bin/python scripts/bench/verify_collection.py', flush=True)
    sys.exit(1)
CAMERA_MIN_SPATIAL_STD = 3.0
CAMERA_MIN_TEMPORAL_DELTA = 0.5
from escape_ood.collect.constants import MAX_TRANSLATION_STEP_M as MAX_STEP, PHASE_ADVANCE_TOLERANCE_M as ADVANCE_TOL
from escape_ood.suites.spec import SUITES, enumerate_conditions, object_keys_for, params_from_attrs
SPLITS = ('train', 'interior', 'test')
FAILURES: list[str] = []
WARNINGS: list[str] = []

def check(cond: bool, label: str, *, warn_only: bool=False) -> None:
    tag = 'ok   ' if cond else 'warn ' if warn_only else 'FAIL '
    print(f'  {tag} {label}')
    if not cond:
        (WARNINGS if warn_only else FAILURES).append(label)

def declared(suite: str, split: str) -> int:
    return len(enumerate_conditions(SUITES[suite], object_keys_for(suite), splits=(split,)))

def condition_signature(attrs) -> tuple:
    """Identify a condition from its recorded attrs, on the parameter vector rather than a key
    string: HDF5 returns numpy scalars and bytes, so a reconstructed key would not match the
    collector's spelling character for character even when the condition is the same."""
    params = params_from_attrs(dict(attrs))
    return (tuple(sorted(((k, str(v)) for (k, v) in params.items()))), str(attrs.get('object_key', '')))

def check_cameras_see_something(group, name, report) -> None:
    pass
    obs = group.get('obs')
    if obs is None:
        return
    for cam in sorted((k for k in obs.keys() if str(k).startswith('cam'))):
        d = obs[cam]
        if len(d.shape) != 4 or d.shape[0] < 2:
            continue
        idx = [0, d.shape[0] // 2, d.shape[0] - 1]
        frames = [np.asarray(d[i], dtype=np.float32) for i in idx]
        spatial = max((float(f.std()) for f in frames))
        temporal = float(np.abs(frames[-1] - frames[0]).mean())
        if spatial < CAMERA_MIN_SPATIAL_STD:
            report(f'{name}: {cam} is a flat image (std {spatial:.2f} < {CAMERA_MIN_SPATIAL_STD}) -- the lens is probably inside geometry')
        elif temporal < CAMERA_MIN_TEMPORAL_DELTA:
            report(f'{name}: {cam} never changes across the episode (mean |Δ| {temporal:.3f} < {CAMERA_MIN_TEMPORAL_DELTA}) -- the stream is frozen')

def check_no_jams(group, name, report) -> None:
    pass
    a = group['actions'][:]
    q = group['obs']['joint_pos'][:]
    if len(a) < 30:
        return
    err = np.abs(a[:, :7] - q[:, :7]).max(axis=1)
    vel = np.concatenate([[0.0], np.abs(np.diff(q[:, :7], axis=0)).max(axis=1)])
    bad = (err > 0.15) & (vel < 0.006)
    (runs, start) = ([], None)
    for (i, v) in enumerate(bad):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 45:
                runs.append((start, i))
            start = None
    if start is not None and len(bad) - start >= 45:
        runs.append((start, len(bad)))
    for (s0, e0) in runs:
        report(f'{name}: JAM frames {s0}-{e0} ({(e0 - s0) / 30:.1f}s, peak tracking error {err[s0:e0].max():.3f} rad)')

def check_eef_and_phases(group, name, report) -> None:
    pass
    n = int(group['actions'].shape[0])
    if 'actions_eef' not in group or 'eef_pose' not in group['obs']:
        report(f'{name}: missing actions_eef / obs/eef_pose')
        return
    (ae, ep) = (group['actions_eef'][:], group['obs']['eef_pose'][:])
    if ae.shape != (n, 8) or ep.shape != (n, 7):
        report(f'{name}: EEF shapes {ae.shape} / {ep.shape}, expected ({n},8) / ({n},7)')
        return
    if not np.allclose(ae[:, 7], group['actions'][:, 7], atol=1e-06):
        report(f'{name}: gripper column differs between actions and actions_eef')
    raw = group.attrs.get('phases')
    if raw is None:
        report(f"{name}: no 'phases' attr")
        return
    spans = json.loads(raw if isinstance(raw, str) else raw.decode())
    if not spans or spans[0][1] != 0 or spans[-1][2] != n:
        report(f'{name}: phase spans do not tile [0,{n}): {spans[:2]}...{spans[-1:]}')
        return
    for ((la, a0, a1, _), (lb, b0, b1, _)) in zip(spans, spans[1:]):
        if a1 != b0:
            report(f'{name}: gap/overlap between phases {la!r} and {lb!r} ({a1} vs {b0})')
            return
    if bool(group.attrs.get('success', True)) and (not spans[-1][3]):
        report(f'{name}: marked success but its last phase {spans[-1][0]!r} is not ok')
    edges = {b for (_, _, b, _) in spans} | {a for (_, a, _, _) in spans}
    step = np.linalg.norm(np.diff(ae[:, :3], axis=0), axis=1)
    for (i, sv) in enumerate(step):
        if i + 1 in edges:
            limit = max(2.0 * MAX_STEP, MAX_STEP + float(np.linalg.norm(ep[i + 1, :3] - ae[i, :3])))
        else:
            limit = MAX_STEP
        if sv > limit + 0.0001:
            where = 'at a phase edge' if i + 1 in edges else 'within a phase'
            report(f'{name}: commanded grasp point moved {sv * 1000:.1f} mm {where} at frame {i} (cap {limit * 1000:.0f} mm) -- actions_eef may be misaligned')
            break
    gap = np.linalg.norm(ae[:, :3] - ep[:, :3], axis=1)
    for (lbl, a0, a1, _) in spans:
        end = float(gap[a1 - 1])
        if end > ADVANCE_TOL + 0.01:
            report(f'{name}: phase {lbl!r} ended {end * 100:.1f} cm from its commanded pose (advance tolerance {ADVANCE_TOL * 100:.0f} cm)')
            break
    if float(gap.max()) > 0.12:
        report(f'{name}: commanded pose ran {gap.max() * 100:.1f} cm ahead of the arm (> 12 cm) -- the arm is not tracking')

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='data/output/collect')
    ap.add_argument('--in-progress', action='store_true', help='a run is still going; report unequal demo counts as a warning')
    ap.add_argument('--expect-demos', type=int, default=0, help='required demonstrations per condition; 0 infers from the data')
    args = ap.parse_args()
    root = Path(args.dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    archives = sorted(glob.glob(str(root / '*.hdf5')))
    if not archives:
        print(f'[ERROR] no archives under {root}')
        return 1
    cells: dict[tuple[str, str], dict] = {}
    problems: list[str] = []
    for path in archives:
        with h5py.File(path, 'r') as f:
            if 'data' not in f:
                continue
            for name in f['data']:
                g = f['data'][name]
                tag = f'{Path(path).stem}/{name}'
                check_no_jams(g, tag, problems.append)
                check_eef_and_phases(g, tag, problems.append)
                check_cameras_see_something(g, tag, problems.append)
                attrs = g.attrs
                (suite, split) = (str(attrs['suite']), str(attrs['split']))
                d = cells.setdefault((suite, split), {'n': 0, 'ok': 0, 'frames': 0, 'conds': {}, 'seeds': set(), 'no_params': 0})
                d['n'] += 1
                d['frames'] += int(attrs.get('n_frames', 0))
                if 'episode_seed' in attrs:
                    d['seeds'].add(int(attrs['episode_seed']))
                if not params_from_attrs(dict(attrs)):
                    d['no_params'] += 1
                if bool(attrs.get('success', True)):
                    d['ok'] += 1
                    sig = condition_signature(attrs)
                    d['conds'][sig] = d['conds'].get(sig, 0) + 1
    print(f"\n{'suite':8s} {'split':9s} {'eps':>5s} {'ok':>5s} {'conds':>6s} {'declared':>9s} {'demos':>9s} {'frames':>8s}")
    total_eps = total_frames = 0
    for (suite, split) in sorted(cells):
        d = cells[suite, split]
        counts = sorted(d['conds'].values())
        rng = f'{counts[0]}-{counts[-1]}' if counts else '-'
        total_eps += d['n']
        total_frames += d['frames']
        print(f"{suite:8s} {split:9s} {d['n']:5d} {d['ok']:5d} {len(d['conds']):6d} {declared(suite, split):9d} {rng:>9s} {d['frames']:8d}")
    print(f"{'TOTAL':8s} {'':9s} {total_eps:5d} {'':5s} {'':6s} {'':9s} {'':9s} {total_frames:8d}")
    print('\nevery declared condition of each recorded split was collected')
    for (suite, split) in sorted(cells):
        (got, want) = (len(cells[suite, split]['conds']), declared(suite, split))
        check(got == want, f'{suite}/{split}: {got} of {want} declared conditions', warn_only=args.in_progress)
    print('\nequal demonstration depth per condition')
    for (suite, split) in sorted(cells):
        counts = sorted(cells[suite, split]['conds'].values())
        if not counts:
            continue
        equal = counts[0] == counts[-1]
        msg = f'{suite}/{split}: {counts[0]}-{counts[-1]} demos per condition'
        if args.expect_demos and equal:
            equal = counts[0] == args.expect_demos
            msg += f' (expected {args.expect_demos})'
        check(equal, msg, warn_only=args.in_progress)
    print('\nevery episode succeeded')
    for (suite, split) in sorted(cells):
        d = cells[suite, split]
        check(d['ok'] == d['n'], f"{suite}/{split}: {d['ok']}/{d['n']}")
    print('\nthe parameter vector survived on every episode')
    for (suite, split) in sorted(cells):
        d = cells[suite, split]
        check(d['no_params'] == 0, f"{suite}/{split}: {d['no_params']} episodes missing param_*")
    print('\nseeds are unique (a reused seed reproduces a trajectory exactly)')
    for (suite, split) in sorted(cells):
        d = cells[suite, split]
        if d['seeds']:
            check(len(d['seeds']) == d['n'], f"{suite}/{split}: {len(d['seeds'])} distinct seeds for {d['n']} episodes")
    print('\nrepeats of one condition are genuinely different trajectories')
    for path in archives:
        with h5py.File(path, 'r') as f:
            if 'data' not in f:
                continue
            groups: dict[tuple, list[str]] = {}
            for name in f['data']:
                groups.setdefault(condition_signature(f['data'][name].attrs), []).append(name)
            worst = None
            for names in groups.values():
                if len(names) < 2:
                    continue
                base = f['data'][names[0]]['actions'][:]
                for other in names[1:]:
                    arr = f['data'][other]['actions'][:]
                    m = min(len(base), len(arr))
                    diff = float(np.abs(base[:m] - arr[:m]).max())
                    worst = diff if worst is None else min(worst, diff)
            if worst is not None:
                check(worst > 1e-06, f'{Path(path).stem}: smallest difference across repeats = {worst:.5f} rad')
    print('\nper-episode: no jams, EEF datasets sane, phase spans tile the episode')
    if problems:
        for pr in problems[:25]:
            check(False, pr)
        if len(problems) > 25:
            print(f'  ... and {len(problems) - 25} more')
    else:
        check(True, f'{total_eps} episodes clean')
    print()
    if FAILURES:
        print(f'[FAIL] {len(FAILURES)} check(s):')
        for x in FAILURES:
            print('  -', x)
        for w in WARNINGS:
            print('  ~', w)
        return 1
    if WARNINGS:
        print(f'[PASS with {len(WARNINGS)} warning(s)]')
        for w in WARNINGS:
            print('  ~', w)
        return 0
    print(f'[PASS] {total_eps} episodes, {total_frames} frames verified')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

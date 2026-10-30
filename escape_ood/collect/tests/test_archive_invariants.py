# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.collect.constants import MAX_TRANSLATION_STEP_M as MAX_STEP, PHASE_ADVANCE_TOLERANCE_M as ADVANCE_TOL
try:
    import h5py
    import numpy as np
except ImportError:
    print('[ERROR] needs h5py (venvs/sim/bin/python)', flush=True)
    sys.exit(1)
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    print(f"  {('ok   ' if cond else 'FAIL ')} {label}")
    if not cond:
        FAILURES.append(label)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/output/collect')
    ap.add_argument('--max-lag-m', type=float, default=0.12, help='ceiling on mid-move command-vs-arm lag; a stuck arm diverges')
    args = ap.parse_args()
    root = Path(args.dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    archives = sorted(glob.glob(str(root / '*.hdf5')))
    if not archives:
        print(f'  skip  no archives under {root} -- nothing collected yet')
        print('[PASS] (skipped)')
        return 0
    n_eps = 0
    worst_step = worst_end = worst_lag = 0.0
    for path in archives:
        with h5py.File(path, 'r') as f:
            for name in f['data']:
                g = f['data'][name]
                tag = f'{Path(path).stem}/{name}'
                n = int(g['actions'].shape[0])
                n_eps += 1
                ok_shapes = 'actions_eef' in g and 'eef_pose' in g['obs'] and (g['actions_eef'].shape == (n, 8)) and (g['obs']['eef_pose'].shape == (n, 7))
                if not ok_shapes:
                    check(False, f'{tag}: EEF datasets present with shapes ({n},8)/({n},7)')
                    continue
                (ae, ep) = (g['actions_eef'][:], g['obs']['eef_pose'][:])
                raw = g.attrs.get('phases')
                if raw is None:
                    check(False, f'{tag}: no phases attr')
                    continue
                spans = json.loads(raw if isinstance(raw, str) else raw.decode())
                tiled = bool(spans) and spans[0][1] == 0 and (spans[-1][2] == n) and all((a[2] == b[1] for (a, b) in zip(spans, spans[1:])))
                if not tiled:
                    check(False, f'{tag}: phase spans do not tile [0,{n})')
                    continue
                if bool(g.attrs.get('success', True)) and (not spans[-1][3]):
                    check(False, f'{tag}: success but last phase {spans[-1][0]!r} not ok')
                if not np.allclose(ae[:, 7], g['actions'][:, 7], atol=1e-06):
                    check(False, f'{tag}: gripper column differs between the two action arrays')
                edges = {b for (_, _, b, _) in spans} | {a for (_, a, _, _) in spans}
                step = np.linalg.norm(np.diff(ae[:, :3], axis=0), axis=1)
                ep_ = g['obs']['eef_pose'][:]
                for (i, sv) in enumerate(step):
                    limit = max(2.0 * MAX_STEP, MAX_STEP + float(np.linalg.norm(ep_[i + 1, :3] - ae[i, :3]))) if i + 1 in edges else MAX_STEP
                    worst_step = max(worst_step, sv / limit)
                    if sv > limit + 0.0001:
                        check(False, f'{tag}: command moved {sv * 1000:.1f} mm at frame {i} (cap {limit * 1000:.0f} mm) -- actions_eef misaligned?')
                        break
                gap = np.linalg.norm(ae[:, :3] - ep[:, :3], axis=1)
                for (lbl, a0, a1, _) in spans:
                    e = float(gap[a1 - 1])
                    worst_end = max(worst_end, e)
                    if e > ADVANCE_TOL + 0.01:
                        check(False, f'{tag}: phase {lbl!r} ended {e * 100:.1f} cm from its commanded pose')
                        break
                worst_lag = max(worst_lag, float(gap.max()))
    check(n_eps > 0, f'{n_eps} episodes inspected across {len(archives)} archive(s)')
    check(worst_step <= 1.0 + 0.001, f'command step never exceeds its cap (worst {100 * worst_step:.2f}% of the limit)')
    check(worst_end <= ADVANCE_TOL + 0.01, f'every phase converged (worst end-of-phase gap {worst_end * 100:.2f} cm)')
    check(worst_lag <= args.max_lag_m, f'mid-move lag stays bounded (worst {worst_lag * 100:.2f} cm <= {args.max_lag_m * 100:.0f} cm)')
    print()
    if FAILURES:
        print(f'[FAIL] {len(FAILURES)} check(s) failed:')
        for x in FAILURES:
            print('  -', x)
        return 1
    print('[PASS] all checks passed')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

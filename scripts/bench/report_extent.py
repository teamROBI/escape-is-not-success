# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.metrics.extent import ENVELOPE, Band, summarise
_ORDER = {'expert': 0, 'replay': 1, 'hold': 2, 'random': 3}

def load(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if 'records' not in d:
        return None
    d['_path'] = path
    d['_tag'] = d.get('policy') or path.stem.split('_')[-1]
    return d

def rank(d: dict) -> tuple:
    return (_ORDER.get(d['_tag'], 9), str(d['_tag']), str(d.get('split', '')))

def render(runs: list[dict], floor: float) -> list[str]:
    out: list[str] = []
    rows: list[tuple] = []
    for d in sorted(runs, key=rank):
        s = summarise(d, ENVELOPE)
        ic = s.get('interior_control')
        ctrl = f"{100 * ic['rate']:5.1f}% ({ic['n_success']}/{ic['n']})" if ic else '  none'
        why = '' if ic and ic['rate'] >= floor else '   ** below floor: extents not interpretable **' if ic else '   ** no interior control: evaluate --split interior **'
        valid = not why
        head = f"{d['_tag']:<12s} suite={d.get('suite', '?'):<6s} split={d.get('split', '?'):<9s} n={s['n']:<4d} interior control {ctrl}"
        out.append(head + why)
        for (direction, blk) in s.get('directions', {}).items():
            ext = blk['extent']
            note = '' if valid else '  (uninterpretable)'
            sat = ' SATURATED' if ext['saturated'] else ''
            out.append(f"    {direction:<8s} extent {ext['extent_cm']:5.1f} cm  {ext['bands_passed']}/{ext['bands_total']} bands{sat}{note}")
            for b in blk['bands']:
                out.append('      ' + Band(**b).summary())
            if valid:
                rows.append((d['_tag'], d.get('suite', '?'), direction, ext['extent_cm'], ext['saturated'], ic['rate']))
        out.append('')
    if rows:
        out.append('=' * 78)
        out.append(f"{'model':<12s} {'suite':<7s} {'dir':<9s} {'extent':>8s}  {'ID ctrl':>8s}")
        out.append('-' * 78)
        for (tag, suite, direction, cm, sat, ctrl) in rows:
            out.append(f"{tag:<12s} {suite:<7s} {direction:<9s} {cm:6.1f}cm{('*' if sat else ' ')} {100 * ctrl:7.1f}%")
        out.append('')
        out.append('* saturated: every band passed, so the sweep did not find the limit and must be extended.')
    else:
        out.append('No run passed the interior-ID control; nothing is comparable yet.')
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval', default='data/output/eval')
    ap.add_argument('--suite', default=None, help='restrict to one suite')
    ap.add_argument('--control-floor', type=float, default=0.8)
    ap.add_argument('--markdown', default=None, help='also write the table to this path')
    args = ap.parse_args()
    root = Path(args.eval)
    if not root.is_absolute():
        root = REPO_ROOT / root
    files = [d for d in (load(p) for p in sorted(root.glob('*.json'))) if d]
    if args.suite:
        files = [d for d in files if d.get('suite') == args.suite]
    if not files:
        print(f'[REPORT] no evaluation JSONs under {root}')
        return 1
    merged: dict[tuple, dict] = {}
    for d in files:
        key = (d.get('suite'), d['_tag'])
        if key not in merged:
            merged[key] = {**d, 'records': [], '_splits': []}
        seen = {(r.get('key'), r.get('episode'), r.get('split')) for r in merged[key]['records']}
        merged[key]['records'] += [r for r in d['records'] if (r.get('key'), r.get('episode'), r.get('split')) not in seen]
        merged[key]['_splits'].append(str(d.get('split')))
    runs = list(merged.values())
    for d in runs:
        d['split'] = '+'.join(sorted(set(d['_splits'])))
    lines = render(runs, args.control_floor)
    print(f'[REPORT] {len(runs)} run(s) under {root}\n')
    print('\n'.join(lines))
    if args.markdown:
        out = Path(args.markdown)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('# Action extent diagnostics\n\n```\n' + '\n'.join(lines) + '\n```\n')
        print(f'\n[REPORT] wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

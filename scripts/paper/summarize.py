#!/usr/bin/env python3
"""Recompute published per-seed stages from evaluation JSONs; no simulator is needed."""
import argparse
import json
import statistics
from pathlib import Path
import paper_data as D
from escape_ood.metrics.extent import reach_fraction, escape_distance, direction_of, strict_success


def summarize_cell(c):
    runs = []
    for seed, run in c['runs'].items():
        entry = {'seed': int(seed), 'run': run, 'eval_tag': D.MANIFEST['provenance'][run]['eval_tag']}
        for direction in ('radial', 'angular'):
            scored = D.scored(run, direction)
            entry[direction] = {
                'n': len(scored), 'escaped': sum(x['escape'] > D.ESCAPE_M for x in scored),
                'approached': sum(x['approach'] >= D.APPROACH_STAGE for x in scored),
                'completed': sum(x['success'] for x in scored),
                'mean_approach': statistics.mean(x['approach'] for x in scored),
                'median_approach': statistics.median(x['approach'] for x in scored),
                'approach_threshold_counts': {str(t): sum(x['approach'] >= t for x in scored)
                                              for t in (.1, .25, .75)},
                'bands': {str(b): {'n': sum(x['band'] == b for x in scored),
                                  'median_approach': statistics.median(x['approach'] for x in scored if x['band'] == b),
                                  'completed': sum(x['success'] for x in scored if x['band'] == b)}
                          for b in sorted({x['band'] for x in scored})},
            }
        for split in ('train', 'interior'):
            success, n = D.split_success(run, split)
            entry[split] = {'n': n, 'completed': success}
        runs.append(entry)
    return {**{k: v for k, v in c.items() if k != 'runs'}, 'seeds': runs}


def diagnostics(repeatability=False):
    violations = []
    native = strict = failures = reclassified = carried = 0
    heights = []
    for c in D.MANIFEST['cells']:
        for run in c['runs'].values():
            for r in D.records(run):
                target = r['target_xy'][0]
                e, a, ok = escape_distance(r['eef_path']), reach_fraction(r['eef_path'], target), strict_success(r)
                native += bool(r['success'])
                strict += ok
                if ok:
                    heights.append(r['eef_path'][-1][2] - D.CUBE_CENTRE_Z)
                if ok and not (e > D.ESCAPE_M and a >= D.APPROACH_STAGE):
                    violations.append({'run': run, 'params': r['params'], 'episode': r['episode'],
                                       'escape_m': e, 'approach': a, 'strict_completion': ok,
                                       'target_xy': target, 'final_object_xy': r['final_object_xy']})
    return {'expert': D.expert_summary(), 'stage_order_violations': violations,
            'native_test_successes': native, 'strict_test_completions': strict,
            'approach_3d_radial': D.approach_3d_check(), 'push_check': D.approach_push_check(),
            'grasp_height': D.grasp_height_check(),
            'envelope_sensitivity': D.envelope_sensitivity(),
            **({'repeatability': D.eval_repeatability_detail()} if repeatability else {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=D.REPO / 'outputs/paper_cells.json')
    ap.add_argument('--repeatability', action='store_true', help='include optional repeated and alternate-mount evaluations')
    ap.add_argument('--published', action='store_true', help='print the committed main table without raw outputs')
    ap.add_argument('--diagnostics', action='store_true', help='also recompute calibration and sensitivity statements')
    args = ap.parse_args()
    if args.published:
        saved = json.loads((D.REPO / 'results/paper_cells.json').read_text())
        print('| frame | arm | cameras | escape | approach | strict completion |')
        print('|---|---|---|---|---|---|')
        for c in saved['cells']:
            counts = ['/'.join(str(s['radial'][stage]) for s in c['seeds'])
                      for stage in ('escaped', 'approached', 'completed')]
            arm = c['arm_command'] + (' (query anchor)' if c.get('anchor') == 'query' else '')
            print('| ' + ' | '.join([c['frame'], arm, c['cameras'], *counts]) + ' |')
        return
    out = {'thresholds': {'escape_m': D.ESCAPE_M, 'approach': D.APPROACH_STAGE},
           'cells': [summarize_cell(c) for c in D.MANIFEST['cells']],
           'controls': [summarize_cell(c) for c in D.MANIFEST['controls']]}
    if args.diagnostics:
        out['diagnostics'] = diagnostics(args.repeatability)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(f'Wrote {args.out}: {sum(len(c["seeds"]) for c in out["cells"])} table runs, '
          f'{sum(len(c["seeds"]) for c in out["controls"])} zeroed-state control runs')

if __name__ == '__main__':
    main()

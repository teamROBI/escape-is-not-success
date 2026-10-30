#!/usr/bin/env python3
"""Print or execute paper recipes. Each command runs in the foreground; servers use a separate terminal."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RECIPES = json.loads((ROOT / 'results/recipes.json').read_text())
SIM = 'venvs/sim/bin/python'
ACT = 'venvs/act/bin/python'

def commands(stage, rows, port):
    if stage == 'scene':
        for tool in ('create_scene', 'create_assets'):
            yield {}, [SIM, f'escape_ood/sim/tools/{tool}.py']
    elif stage == 'gate':
        yield {}, [SIM, 'escape_ood/sim/tools/achievability_gate.py', '--suite', 'lift',
                   '--exhaustive', '--attempts', '2', '--out', 'data/output/gate']
    elif stage == 'collect':
        for collection, parts, legacy in [('collect', 4, '1'), ('collect_wrist', 3, '0')]:
            for part in range(parts):
                yield {'ESCAPE_WRIST_LEGACY': legacy}, [SIM, 'escape_ood/sim/tools/collect_suite.py',
                    '--suite', 'lift', '--demos', '15', '--partition-total', str(parts),
                    '--partition-idx', str(part), '--out', f'data/output/{collection}']
    elif stage == 'verify':
        for collection in ('collect', 'collect_wrist'):
            yield {}, [SIM, 'scripts/bench/verify_collection.py', '--dir', f'data/output/{collection}', '--expect-demos', '15']
            yield {}, [SIM, 'escape_ood/collect/tests/test_archive_invariants.py', '--dir', f'data/output/{collection}']
    elif stage == 'convert':
        seen = set()
        for r in rows:
            if r['dataset'] in seen:
                continue
            seen.add(r['dataset'])
            col = 'collect' if r['collection'] == 'A' else 'collect_wrist'
            cmd = [SIM, 'scripts/bench/convert_act.py', '--suite', 'lift', '--collect', f'data/output/{col}',
                   '--out', f'data/output/{r["dataset"]}', '--action-space', r['convert_space'],
                   '--cameras', *r['cameras'], '--scale', '2']
            if r['no_proprio']:
                cmd += ['--no-proprio']
            yield {}, cmd
    elif stage == 'calibrate':
        yield {}, [SIM, 'escape_ood/sim/tools/eval_policy.py', '--suite', 'lift', '--policy', 'expert',
                   '--split', 'all', '--episodes', '2']
        for space in ('joint', 'eef_abs', 'eef_abs_rotvec', 'joint_chunk_absgrip', 'eef_chunk'):
            yield {'ESCAPE_WRIST_LEGACY': '1'}, [SIM, 'escape_ood/sim/tools/eval_policy.py', '--suite', 'lift',
                '--policy', 'replay', '--split', 'train', '--episodes', '2', '--replay-space', space,
                '--tag', f'replay_{space}']
    elif stage == 'report':
        yield {}, [SIM, 'scripts/paper/summarize.py', '--diagnostics']
        yield {}, [SIM, 'scripts/paper/gen_table1_cells.py']
        yield {}, [SIM, 'scripts/paper/gen_fig1_hero.py']
        yield {}, [SIM, 'scripts/paper/gen_fig2_approach.py']
        yield {}, [SIM, 'scripts/paper/gen_fig3_overlay.py']
    else:
        for r in rows:
            dims = {'ESCAPE_STATE_DIM': '8', 'ESCAPE_ACTION_DIM': str(7 if r['action_space'].startswith('eef_') and r['action_space'] != 'eef_abs' else 8)}
            ckpt = f'data/output/train/{r["run"]}/lift/policy_best.ckpt'
            if stage == 'train':
                cmd = [ACT, 'scripts/bench/run_act_suite.py', '--suite', 'lift', '--seed', str(r['seed']),
                       '--data', f'data/output/{r["dataset"]}', '--out', f'data/output/train/{r["run"]}']
                if r['anchor'] == 'query':
                    cmd += ['--anchor-chunk', r['action_space']]
                yield dims, cmd
            elif stage == 'serve':
                cmd = [ACT, 'scripts/bench/run_act_policy_server.py', '--ckpt', ckpt, '--port', str(port),
                       '--action-space', r['action_space'], '--cameras', *r['cameras']]
                if r['no_proprio']:
                    cmd += ['--no-proprio']
                yield dims, cmd
            elif stage in ('evaluate', 'fig3', 'repeatability'):
                env = {'ESCAPE_WRIST_LEGACY': '1' if r['collection'] == 'A' else '0'}
                if stage == 'repeatability':
                    if r['eval_tag'] == r['run']:
                        continue
                    env['ESCAPE_WRIST_LEGACY'] = '0'
                    tags = [r['run']]
                    if r['seed'] == 0 and r['action_space'] in ('joint_chunk_absgrip', 'eef_chunk'):
                        tags.append(r['run'] + '_rep')
                    for tag in tags:
                        for split in ('interior', 'test'):
                            yield env, [SIM, 'escape_ood/sim/tools/eval_policy.py', '--suite', 'lift',
                                '--split', split, '--episodes', '2', '--policy', 'server', '--port', str(port), '--tag', tag]
                elif stage == 'fig3':
                    env.update(ESCAPE_FIG_CAM='1', ESCAPE_FIG_STRIDE='5', ESCAPE_EVAL_CELLS='0.725,20')
                    yield env, [SIM, 'escape_ood/sim/tools/eval_policy.py', '--suite', 'lift', '--split', 'test',
                                '--episodes', '2', '--policy', 'server', '--port', str(port), '--save-video', 'all',
                                '--tag', f'fig_{r["run"]}', '--out', f'data/output/media/fig3_rollouts/{r["run"]}']
                else:
                    for split in ('interior', 'test', 'train'):
                        yield env, [SIM, 'escape_ood/sim/tools/eval_policy.py', '--suite', 'lift', '--split', split,
                                    '--episodes', '2', '--policy', 'server', '--port', str(port),
                                    '--tag', r['eval_tag'] + ('_trn' if split == 'train' else '')]
            else:
                raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('list', 'scene', 'gate', 'collect', 'verify', 'convert', 'train',
                                     'serve', 'evaluate', 'calibrate', 'fig3', 'repeatability', 'report'))
    ap.add_argument('--run', help='run name in results/recipes.json; omit to print all runs')
    ap.add_argument('--port', type=int, default=6000)
    ap.add_argument('--execute', action='store_true', help='execute commands; default only prints')
    args = ap.parse_args()
    rows = [r for r in RECIPES if not args.run or r['run'] == args.run]
    if not rows:
        ap.error('Unknown run; see the list stage')
    if args.stage == 'list':
        for r in rows:
            print(r['run'], r['collection'], r['action_space'], '+'.join(r['cameras']),
                  'zeroed joints' if r['no_proprio'] else 'measured joints')
        return
    if args.execute and args.stage in ('serve', 'evaluate', 'fig3', 'repeatability') and not args.run:
        ap.error('Select --run; start its server in a separate terminal')
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for extra, cmd in commands(args.stage, rows, args.port):
        print(shlex.join(['env', *(f'{k}={v}' for k, v in extra.items()), *cmd]), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, env=env | extra, check=True)

if __name__ == '__main__':
    main()

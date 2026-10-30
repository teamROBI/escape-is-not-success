# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument('--suite', required=True, help='suite to gate')
_parser.add_argument('--per-suite', type=int, default=24, help='test cells to attempt when sampling (train and interior are always gated in full)')
_parser.add_argument('--splits', default='train,interior,test', help='which splits to gate; all three by default, because a training cell the expert cannot demonstrate is a collection bug, an interior cell it cannot demonstrate invalidates the ID control, and a test cell it cannot demonstrate invalidates the band')
_parser.add_argument('--objects', default=None, help="comma-separated graspable assets; default: the suite's own")
_parser.add_argument('--max-steps', type=int, default=3000)
_parser.add_argument('--partition-idx', type=int, default=0)
_parser.add_argument('--partition-total', type=int, default=1, help='split the suite across parallel launches by striding, so one suite can use several GPUs. Strided rather than contiguous: a contiguous split would give one worker only the far bands, so a crashed worker would bias coverage along the very direction being measured.')
_parser.add_argument('--attempts', type=int, default=1, help='attempts per cell, each from a differently jittered home pose. 1 measures achievability; >1 measures reliability, which is what the protocol requires of the expert at every cell.')
_parser.add_argument('--exhaustive', action='store_true', help="run EVERY cell of the suite instead of sampling the test split. The only form of the gate whose pass rate is literally 'all of them'.")
_parser.add_argument('--seed', type=int, default=0)
_parser.add_argument('--out', default=None)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app
from escape_ood.collect.context import ObjectDropped, PhaseTimeout, TaskContext, TaskFailed
from escape_ood.collect.tasks import task_for
from escape_ood.sim.env.isaac_backend import IsaacMotionBackend
from escape_ood.sim.env.scene import assert_condition_realised, build_scene, episode_seed, jittered_home, reset_condition_scene, suite_home
from escape_ood.suites.spec import SUITES, band_label, enumerate_conditions, object_keys_for, sample_sweep, suite as get_suite

def pick_conditions(suite_name: str, object_keys: list[str]) -> list:
    """Cells to attempt. Train and interior always in full; test in full with --exhaustive,
    otherwise stratified across the sweep so both ends of every direction are represented."""
    s = get_suite(suite_name)
    splits = tuple((x.strip() for x in args_cli.splits.split(',') if x.strip()))
    if args_cli.exhaustive:
        conds = enumerate_conditions(s, object_keys, splits=splits)
    else:
        n_bins = max(4, args_cli.per_suite // 4)
        ss = sample_sweep(s, object_keys, n_bins=n_bins, per_bin=max(1, args_cli.per_suite // n_bins), seed=args_cli.seed)
        conds = [c for c in ss.conditions if c.split in splits]
    if args_cli.partition_total > 1:
        (pt, pi) = (args_cli.partition_total, args_cli.partition_idx)
        if not 0 <= pi < pt:
            raise SystemExit(f'[GATE] bad partition {pi}/{pt}')
        conds = [c for (i, c) in enumerate(conds) if i % pt == pi]
    return conds

def run_one(scene, sim, info, backend, cond, suite_name: str, attempt: int=0) -> dict:
    """One episode. Returns the outcome plus the phase ladder the expert reached."""
    assert_condition_realised(suite_name, cond.params, info['built'])
    base = suite_home(suite_name)
    home = base if attempt == 0 else jittered_home(episode_seed(cond.key(), attempt), base=base)
    backend.reset_to(home)
    reset_condition_scene(scene, sim, info, cond)
    ctx = TaskContext(backend)
    ctx.object_name = info['instance_name'][cond.object_key]
    task = task_for(cond)(cond)
    out = {'key': cond.key(), 'split': cond.split, 'band': band_label(cond.params), 'params': dict(cond.params), 'sweep_coord': cond.sweep_coord, 'ok': False, 'reason': '', 'steps': 0}
    try:
        out['steps'] = backend.drive(task.run(ctx), max_steps=args_cli.max_steps)
        out['ok'] = True
    except (PhaseTimeout, ObjectDropped, TaskFailed, TimeoutError) as e:
        out['reason'] = f'{type(e).__name__}: {e}'
        out['steps'] = sum((p.steps for p in ctx.phases))
    out['phases'] = [{'label': p.label, 'ok': p.ok, 'steps': p.steps, 'err': None if p.final_error is None else round(p.final_error, 4)} for p in ctx.phases]
    out['last_phase'] = ctx.last_phase
    return out

def main() -> int:
    suite_name = args_cli.suite
    if suite_name not in SUITES:
        print(f"[GATE] unknown suite '{suite_name}'. Known: {sorted(SUITES)}", flush=True)
        return 2
    object_keys = [k.strip() for k in args_cli.objects.split(',') if k.strip()] if args_cli.objects else object_keys_for(suite_name)
    (scene, sim, info) = build_scene(suite_name, device=args_cli.device, object_keys=object_keys, with_cameras=False)
    backend = IsaacMotionBackend(scene, sim)
    conds = pick_conditions(suite_name, object_keys)
    print(f'[GATE] suite={suite_name} cells={len(conds)} attempts={args_cli.attempts}', flush=True)
    n_att = max(1, args_cli.attempts)
    results = []
    per_cond: dict[str, list[bool]] = {}
    for (i, cond) in enumerate(conds):
        for attempt in range(n_att):
            r = run_one(scene, sim, info, backend, cond, suite_name, attempt=attempt)
            r['attempt'] = attempt
            r['suite'] = suite_name
            results.append(r)
            per_cond.setdefault(cond.key(), []).append(bool(r['ok']))
            tag = 'ok  ' if r['ok'] else 'FAIL'
            suffix = f' a{attempt}' if n_att > 1 else ''
            print(f"[GATE] {i + 1:3d}/{len(conds)}{suffix} {tag} d={r['sweep_coord']:5.1f}cm {r['split']:8s} {str(r['band'] or '-'):3s} steps={r['steps']:4d} {r['key'][:52]}" + ('' if r['ok'] else f"\n[GATE]        -> {r['reason'][:110]}"), flush=True)
    n_ok = sum((1 for r in results if r['ok']))
    by_split: dict[str, list] = {}
    by_band: dict[str, list] = {}
    for r in results:
        by_split.setdefault(r['split'], []).append(r)
        if r['band']:
            by_band.setdefault(r['band'], []).append(r)
    print(f'\n[GATE] ==== {suite_name} ====', flush=True)
    if n_att > 1:
        full = sum((1 for v in per_cond.values() if all(v)))
        none = sum((1 for v in per_cond.values() if not any(v)))
        flaky = {k: v for (k, v) in per_cond.items() if any(v) and (not all(v))}
        print(f'[GATE] reliability over {n_att} attempts: {full}/{len(per_cond)} cells succeeded every time, {len(flaky)} flaky, {none} never', flush=True)
        for (k, v) in sorted(flaky.items())[:20]:
            print(f'[GATE]   flaky {sum(v)}/{len(v)}  {k[:70]}', flush=True)
    print(f'[GATE] {n_ok}/{len(results)} achievable ({100 * n_ok / max(len(results), 1):.1f}%)', flush=True)
    for (split, rs) in sorted(by_split.items()):
        k = sum((1 for r in rs if r['ok']))
        print(f'[GATE]   {split:8s} {k}/{len(rs)}', flush=True)
    for (band, rs) in sorted(by_band.items()):
        k = sum((1 for r in rs if r['ok']))
        print(f'[GATE]   band {band}: {k}/{len(rs)}', flush=True)
    fails = [r for r in results if not r['ok']]
    if fails:
        stops: dict[str, int] = {}
        for r in fails:
            stops[r['last_phase']] = stops.get(r['last_phase'], 0) + 1
        print(f'[GATE]   failures by last phase: {dict(sorted(stops.items(), key=lambda kv: -kv[1]))}', flush=True)
        for r in sorted(fails, key=lambda r: -r['sweep_coord'])[:3]:
            print(f"[GATE]   d={r['sweep_coord']:.1f} {r['reason'][:100]}", flush=True)
    if not results:
        print('[GATE] VERDICT FAIL: no cells matched the filters, so nothing was gated.', flush=True)
        verdict = 'FAIL'
    else:
        verdict = 'PASS' if n_ok == len(results) else 'FAIL'
    print(f"[GATE] VERDICT {verdict}: the suite {('is' if verdict == 'PASS' else 'IS NOT')} fully achievable by the scripted expert", flush=True)
    if args_cli.out:
        out = Path(args_cli.out)
        out.mkdir(parents=True, exist_ok=True)
        tag = suite_name
        if args_cli.partition_total > 1:
            tag += f'_p{args_cli.partition_idx}of{args_cli.partition_total}'
        (out / f'{tag}.json').write_text(json.dumps({'suite': suite_name, 'attempts': n_att, 'exhaustive': bool(args_cli.exhaustive), 'n': len(results), 'n_ok': n_ok, 'results': results}, indent=2) + '\n')
        print(f'[GATE] wrote {out}/{tag}.json', flush=True)
    sys.stdout.flush()
    import os
    os._exit(0 if verdict == 'PASS' else 1)

def _hard_exit(code: int) -> None:
    """Terminate for real; Isaac's non-daemon threads outlive a bare raise."""
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(code)
if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        _hard_exit(int(e.code) if isinstance(e.code, int) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _hard_exit(1)

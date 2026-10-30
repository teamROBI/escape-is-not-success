#!/usr/bin/env python3
"""Compare geometry and reset seeds with a read-only reference checkout, without loading Isaac."""
import argparse
import ast
import importlib.util
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from escape_ood.suites import spec


def load_spec(path):
    module_spec = importlib.util.spec_from_file_location('_reference_spec', path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def reset_helpers(path):
    tree = ast.parse(path.read_text())
    wanted = {'HOME_JOINT_POS', 'HOME_JITTER_RAD', 'jittered_home', 'episode_seed'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='read-only reference checkout')
    args = ap.parse_args()
    packages = [p for p in args.source.iterdir() if p.is_dir() and (p / 'suites/spec.py').is_file()]
    if len(packages) != 1:
        raise SystemExit('Expected one reference package with suites/spec.py')
    source_pkg = packages[0]
    original = load_spec(source_pkg / 'suites/spec.py')
    old_reset = reset_helpers(source_pkg / 'sim/env' / (source_pkg.name + '_scene.py'))
    new_reset = reset_helpers(ROOT / 'escape_ood/sim/env/scene.py')
    old, new = original.suite('lift'), spec.suite('lift')
    grid = list(old.grid())
    assert grid == list(new.grid())
    old_conds = original.enumerate_conditions(old, original.object_keys_for('lift'))
    new_conds = spec.enumerate_conditions(new, spec.object_keys_for('lift'))
    seed_count = 0
    for p, a, b in zip(grid, old_conds, new_conds):
        assert old.split(p) == new.split(p)
        assert old.sweep_coord(p) == new.sweep_coord(p)
        assert a.key() == b.key()
        for repeat in range(15):
            sa = old_reset['episode_seed'](a.key(), repeat)
            sb = new_reset['episode_seed'](b.key(), repeat)
            assert sa == sb
            assert old_reset['jittered_home'](sa) == new_reset['jittered_home'](sb)
            seed_count += 1
    rng = random.Random(2026)
    points = [(0.0, 0.0)] + [spec.polar_xy(r, t) for r in (0.0, .2, .325, .35, .5, .525, .775, 1.0)
                              for t in (-180, -65, -35, -30, 0, 30, 35, 65, 180)]
    points += [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(10000)]
    for p in points:
        assert original.ENVELOPE.boundary_distance(*p) == spec.ENVELOPE.boundary_distance(*p)
        assert original.ENVELOPE.distance_outside(*p) == spec.ENVELOPE.distance_outside(*p)
    print(f'PASS: {len(grid)} grid/split cells; {len(points)} exact boundary/Euclidean distances; '
          f'{seed_count} identical seeds and jittered homes')

if __name__ == '__main__':
    main()

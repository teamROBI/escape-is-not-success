# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.suites.spec import SUITES, WORKSPACE, enumerate_conditions, object_keys_for, parking_slot, suite as get_suite
from escape_ood.suites.layout_gen import HEAD_CAM, OBJECT_CENTRE_Z, SCHEMA, TABLE_TOP_Z, build_layout, in_head_frame, project_to_head_cam
OBJ_USD_DIR = REPO_ROOT / 'data' / 'output' / 'assets' / 'escape_ood_assets'
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    print(f"  {('ok   ' if cond else 'FAIL ')} {label}")
    if not cond:
        FAILURES.append(label)

def test_structure() -> None:
    print('\n[required keys]')
    required = {'camera', 'object_pool', 'object_spawn_areas', 'props', 'robot', 'scene', 'custom_cams', 'workspace', 'suite'}
    for name in sorted(SUITES):
        s = build_layout(get_suite(name))['session']
        missing = required - set(s)
        check(not missing, f'{name}: all session keys present' + (f' (missing {sorted(missing)})' if missing else ''))

def test_portable_scene_references() -> None:
    print('\n[portable scene references]')
    for name in sorted(SUITES):
        d = build_layout(get_suite(name))
        check(d['session']['scene']['path'] == '', f'{name}: scene.path is empty, forcing portable resolution')
        check(d['schema'] == SCHEMA, f'{name}: schema identifies the scene format')
        check(d['session']['scene']['file_name'].endswith('.usd'), f'{name}: scene resolves by file_name')

def test_pool_references_real_assets() -> None:
    """A layout must not name an asset the generator never wrote."""
    print('\n[object pool references assets that exist]')
    if not OBJ_USD_DIR.is_dir():
        print(f'  skip  {OBJ_USD_DIR} not present (assets not generated on this machine)')
        return
    for name in sorted(SUITES):
        s = build_layout(get_suite(name))['session']
        missing = [o['asset_name'] for o in s['object_pool'] if not (OBJ_USD_DIR / f"{o['asset_name']}.usd").is_file()]
        check(not missing, f'{name}: every pool asset has a USD on disk' + (f' (missing: {missing})' if missing else ''))
        check(all((o['asset_name'] in {p['asset_name'] for p in s['object_pool']} for o in [{'asset_name': k} for k in object_keys_for(name)])), f'{name}: pool contains every object the suite uses')
        check(s['props'] == [], f'{name}: no props on the table (the envelope is invisible)')

def test_spawn_areas_inside_workspace() -> None:
    print('\n[training positions are on the table]')
    for name in sorted(SUITES):
        s = build_layout(get_suite(name))['session']
        sp = get_suite(name)
        outside = [a['id'] for a in s['object_spawn_areas'] if not WORKSPACE.contains(a['pos'][0], a['pos'][1])]
        check(not outside, f'{name}: all training positions inside the workspace' + (f' (outside: {outside})' if outside else ''))
        check(all((abs(a['pos'][2] - TABLE_TOP_Z) < 1e-09 for a in s['object_spawn_areas'])), f'{name}: positions sit on the table surface')
        n_train = sum((1 for p in sp.grid() if sp.split(p) == 'train'))
        check(len(s['object_spawn_areas']) == n_train, f"{name}: lists exactly its {n_train} training cells (got {len(s['object_spawn_areas'])})")

def test_no_enumerated_sweep() -> None:
    """A layout must not enumerate swept positions -- that duplicates the suite definition."""
    print('\n[layouts describe the scene, not the sweep]')
    for name in sorted(SUITES):
        sp = get_suite(name)
        n_grid = len(list(sp.grid()))
        n_areas = len(build_layout(sp)['session']['object_spawn_areas'])
        n_train = sum((1 for p in sp.grid() if sp.split(p) == 'train'))
        check(n_areas <= max(n_train, 1), f'{name}: {n_areas} areas listed, not {n_grid} grid points')

def test_camera_symmetric() -> None:
    """A camera off the centreline makes +theta and -theta unequally visible."""
    print('\n[camera is left-right symmetric]')
    for name in sorted(SUITES):
        cams = build_layout(get_suite(name))['session']['custom_cams']
        head = [c for c in cams if c['name'] == 'cam_head']
        check(len(head) == 1, f'{name}: exactly one head camera')
        cam = head[0]
        check(abs(cam['pos'][1]) < 1e-09, f"{name}: head camera sits on y=0 (y={cam['pos'][1]})")
        check(abs(cam['look_at'][1]) < 1e-09, f"{name}: head camera looks at y=0 (y={cam['look_at'][1]})")
        for w in [c for c in cams if c['name'] == 'cam_wrist']:
            check(w.get('parent_body') == 'panda_hand', f'{name}: wrist camera is attached to panda_hand')
            check('look_at' not in w, f'{name}: wrist camera has no fixed look_at')
    check(abs(WORKSPACE.y_min + WORKSPACE.y_max) < 1e-09, 'workspace is centred on y=0, so the whole scene mirrors cleanly')

def test_camera_streams() -> None:
    print('\n[camera streams]')
    for name in sorted(SUITES):
        cams = build_layout(get_suite(name))['session']['custom_cams']
        names = [c['name'] for c in cams]
        check(names == ['cam_head', 'cam_wrist'], f'{name}: both streams, head first ({names})')
        check(all((c.get('width') and c.get('height') for c in cams)), f'{name}: every camera declares a resolution')
        check(all((c.get('focal_mm') for c in cams)), f'{name}: every camera declares a focal length')

def test_reproducible() -> None:
    print('\n[reproducible output]')
    for name in sorted(SUITES):
        a = json.dumps(build_layout(get_suite(name)), indent=2, sort_keys=True)
        b = json.dumps(build_layout(get_suite(name)), indent=2, sort_keys=True)
        check(a == b, f'{name}: two builds are byte-identical')
    d = build_layout(get_suite(sorted(SUITES)[0]))
    check(d['saved_at'] == 0.0, 'no wall-clock stamp, so regeneration is not a spurious diff')

def test_parking_slots_are_out_of_frame() -> None:
    """A parked object must be behind the head camera, not merely off the table."""
    print('\n[parking slots are behind the camera]')
    cam = tuple((float(v) for v in HEAD_CAM['pos']))
    tgt = tuple((float(v) for v in HEAD_CAM['look_at']))
    d = math.dist(cam, tgt)
    fwd = tuple(((t - c) / d for (t, c) in zip(tgt, cam)))
    for i in range(8):
        p = parking_slot(i)
        v = tuple((a - b for (a, b) in zip(p, cam)))
        dot = sum((a * b for (a, b) in zip(v, fwd)))
        check(dot < -0.25, f'parking slot {i} at {tuple((round(c, 2) for c in p))} is behind the head camera (dot={dot:+.3f}, must be < -0.25)')

def test_all_conditions_in_head_fov() -> None:
    """Every condition's object (and target, if any) projects inside the head image with margin.

    Margin is 10% of each image dimension. The projection is of the object's centre; a 5 cm
    cube at the margin still has half its body inside the frame, and the wrist camera covers
    the final approach. The arm occluding the object at far radial cells is NOT covered by a
    projection test -- check it with render_scene.py at the farthest band.
    """
    print("\n[every condition is in the head camera's frame]")
    (w, h) = (HEAD_CAM['width'], HEAD_CAM['height'])
    for name in sorted(SUITES):
        sp = get_suite(name)
        bad: list[str] = []
        extremes = {'u': (math.inf, -math.inf), 'v': (math.inf, -math.inf)}
        for c in enumerate_conditions(sp, object_keys_for(name)[:1]):
            pts = [('object', sp.object_xy(c.params))]
            t = sp.target_xy(c.params)
            if t is not None:
                pts.append(('target', t))
            for (role, (x, y)) in pts:
                (u, v, depth) = project_to_head_cam(x, y, TABLE_TOP_Z + OBJECT_CENTRE_Z)
                if not math.isnan(u):
                    extremes['u'] = (min(extremes['u'][0], u), max(extremes['u'][1], u))
                    extremes['v'] = (min(extremes['v'][0], v), max(extremes['v'][1], v))
                if not in_head_frame(x, y, TABLE_TOP_Z + OBJECT_CENTRE_Z, margin_frac=0.1):
                    bad.append(f'{c.key()}[{role}] -> u={u:.0f} v={v:.0f} depth={depth:.2f}')
        check(not bad, f'{name}: every condition inside the frame with 10% margin ({len(bad)} outside)')
        for b in bad[:6]:
            print(f'          {b}')
        print(f"          u range [{extremes['u'][0]:.0f}, {extremes['u'][1]:.0f}] of {w}; v range [{extremes['v'][0]:.0f}, {extremes['v'][1]:.0f}] of {h}")

def test_suite_metadata_matches() -> None:
    print('\n[suite metadata]')
    for name in sorted(SUITES):
        meta = build_layout(get_suite(name))['session']['suite']
        sp = get_suite(name)
        check(meta['name'] == sp.name and meta['success_kind'] == sp.success_kind and (meta['unit'] == sp.unit), f'{name}: layout suite block matches the spec')

def main() -> int:
    test_structure()
    test_portable_scene_references()
    test_pool_references_real_assets()
    test_spawn_areas_inside_workspace()
    test_no_enumerated_sweep()
    test_camera_symmetric()
    test_camera_streams()
    test_reproducible()
    test_parking_slots_are_out_of_frame()
    test_all_conditions_in_head_fov()
    test_suite_metadata_matches()
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

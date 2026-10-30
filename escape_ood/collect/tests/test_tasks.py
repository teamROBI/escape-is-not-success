# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.collect.context import TaskContext, TaskFailed
from escape_ood.collect.grasp import canonical_azimuth, grasp_frame, grasp_poses, grasp_quat, mat_to_quat
from escape_ood.collect.tasks import LiftTask, TASK_FOR_SUITE, task_for
from escape_ood.collect.tests.test_context import FakeArm, drive
from escape_ood.suites.spec import SUITES, enumerate_conditions, object_keys_for, object_start_xy
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    print(f"  {('ok   ' if cond else 'FAIL ')} {label}")
    if not cond:
        FAILURES.append(label)

def _dot(a, b):
    return sum((x * y for (x, y) in zip(a, b)))

def _col(m, j):
    return [m[0][j], m[1][j], m[2][j]]

def test_grasp_frame_orthonormal() -> None:
    print('\n[grasp frame is a proper rotation]')
    worst_orth = worst_norm = worst_det = 0.0
    for az in range(0, 360, 15):
        for tilt in (0.0, 12.0, 30.0):
            m = grasp_frame(az, tilt)
            (x, y, z) = (_col(m, 0), _col(m, 1), _col(m, 2))
            worst_orth = max(worst_orth, abs(_dot(x, y)), abs(_dot(y, z)), abs(_dot(x, z)))
            for v in (x, y, z):
                worst_norm = max(worst_norm, abs(math.sqrt(_dot(v, v)) - 1.0))
            cx = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
            worst_det = max(worst_det, max((abs(a - b) for (a, b) in zip(cx, z))))
    check(worst_orth < 1e-09, f'axes mutually orthogonal (worst {worst_orth:.2e})')
    check(worst_norm < 1e-09, f'axes unit length (worst {worst_norm:.2e})')
    check(worst_det < 1e-09, f'right-handed, x cross y == z (worst {worst_det:.2e})')

def test_topdown_approach_points_down() -> None:
    print('\n[top-down grasp approaches downward]')
    for az in (0, 45, 90, 137, 180, 271):
        z = _col(grasp_frame(az, 0.0), 2)
        check(abs(z[2] + 1.0) < 1e-09, f'az={az}: approach is straight down (z={[round(c, 3) for c in z]})')
    z = _col(grasp_frame(0.0, 30.0), 2)
    check(z[2] < 0.0 and z[0] > 0.0, f'tilt leans toward the heading while still descending ({[round(c, 3) for c in z]})')
    z = _col(grasp_frame(90.0, 30.0), 2)
    check(z[2] < 0.0 and z[1] > 0.0 and (abs(z[0]) < 1e-09), 'at azimuth 90 the lean is along +y')

def test_jaw_axis_follows_azimuth() -> None:
    print('\n[jaw axis follows the azimuth]')
    for az in (0, 30, 60, 90, 120):
        x = _col(grasp_frame(az, 0.0), 0)
        want = [math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0]
        check(max((abs(a - b) for (a, b) in zip(x, want))) < 1e-09, f'az={az}: jaw axis is the azimuth heading')

def test_quat_conversion() -> None:
    print('\n[quaternion conversion]')
    q = mat_to_quat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    check(abs(q[0] - 1.0) < 1e-09 and max((abs(c) for c in q[1:])) < 1e-09, 'identity matrix gives the identity quaternion')
    for az in range(0, 360, 30):
        qq = grasp_quat(az, 0.0)
        n = math.sqrt(sum((c * c for c in qq)))
        check(abs(n - 1.0) < 1e-09, f'az={az}: quaternion is unit length (|q|={n:.9f})')
    q180 = mat_to_quat([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    check(abs(math.sqrt(sum((c * c for c in q180))) - 1.0) < 1e-09, '180-degree rotation converts without precision loss')

def test_azimuth_period() -> None:
    print('\n[azimuth is periodic at 180, not 360]')
    check(canonical_azimuth(200.0) == 20.0, '200 folds to 20')
    check(canonical_azimuth(180.0) == 0.0, '180 folds to 0')
    check(canonical_azimuth(-30.0) == 150.0, 'negative folds into range')
    a = _col(grasp_frame(30.0), 0)
    b = _col(grasp_frame(210.0), 0)
    check(max((abs(x + y) for (x, y) in zip(a, b))) < 1e-09, 'az and az+180 give opposite jaw axes, i.e. the same grasp')

def test_grasp_poses_geometry() -> None:
    print('\n[grasp / pregrasp geometry]')
    obj = [0.5, 0.1, 0.027]
    (grasp, pregrasp, _) = grasp_poses(obj, azimuth_deg=0.0)
    check(abs(grasp[2] - 0.027) < 1e-09, 'grasp target sits at the object CENTRE (centre-origin assets)')
    (base_origin, _, _) = grasp_poses([0.5, 0.1, 0.0], origin_to_centre_z=0.025)
    check(abs(base_origin[2] - 0.025) < 1e-09, 'origin_to_centre_z supports base-origin assets (the YCB mug will need it)')
    check(abs(grasp[0] - obj[0]) < 1e-09 and abs(grasp[1] - obj[1]) < 1e-09, 'top-down grasp is directly over the object in XY')
    check(pregrasp[2] > grasp[2], f'pregrasp stands off above the grasp ({pregrasp[2]:.3f} > {grasp[2]:.3f})')
    check(abs(pregrasp[0] - grasp[0]) < 1e-09 and abs(pregrasp[1] - grasp[1]) < 1e-09, 'top-down pregrasp has no lateral offset')
    (_, pre_tilt, _) = grasp_poses(obj, azimuth_deg=0.0, tilt_deg=30.0)
    check(abs(pre_tilt[0] - grasp[0]) > 0.001, 'a tilted pregrasp IS offset laterally, along the approach')

def _arm_for(cond, objects=('cube',)) -> FakeArm:
    """A stub arm with the target object placed where the SPEC says the condition puts it."""
    (x, y) = object_start_xy(cond)
    arm = FakeArm(pos=(0.3, 0.0, 0.3), tracking=0.5)
    arm.objects = {name: [x, y, 0.0] for name in objects}
    return arm

def _run_task(cond, cls=None, **kw) -> tuple[TaskContext, FakeArm]:
    objects = kw.pop('objects', ('cube',))
    arm = _arm_for(cond, objects)
    ctx = TaskContext(arm)
    ctx.object_name = objects[0]
    task = (cls or task_for(cond))(cond, **kw)
    drive(task.run(ctx))
    return (ctx, arm)

def test_lift_task_every_cell() -> None:
    """The stub arm reaches anything, so this checks the expert's LOGIC on every cell: it
    derives a grasp from the spec's placement, runs the full phase sequence, and lands its
    grasp on the object -- for train, interior and every band alike."""
    print('\n[LiftTask on every LIFT cell]')
    s = SUITES['lift']
    conds = enumerate_conditions(s, ['cube'])
    ok = 0
    worst = 0.0
    for cond in conds:
        (ctx, arm) = _run_task(cond)
        (x, y) = object_start_xy(cond)
        if ctx.reached('grasp') and ctx.reached('lift') and ctx.reached('retract'):
            ok += 1
        gp = ctx.phases[[p.label for p in ctx.phases].index('grasp')]
        worst = max(worst, gp.final_error or 0.0)
    check(ok == len(conds), f'all {len(conds)} cells run grasp -> lift -> retract ({ok} ok)')
    check(worst < 0.03, f'grasp phase converges on every cell (worst final error {worst:.3f} m)')
    c0 = conds[0]
    check(task_for(c0) is LiftTask, 'lift dispatches to LiftTask')
    check(c0.instruction == 'pick up the red block' if c0.object_key == 'escape_ood_cube_red' else True, 'instruction wording')

def test_tilt_is_honoured() -> None:
    """If the spec ever puts tilt_deg on a condition, the expert must lean, not ignore it."""
    print('\n[tilt_deg on a condition reaches the grasp]')
    s = SUITES['lift']
    cond = enumerate_conditions(s, ['cube'], splits=('train',))[0]
    from dataclasses import replace
    tilted = replace(cond, params={**cond.params, 'tilt_deg': 25.0})
    (_, arm_flat) = _run_task(cond)
    (_, arm_tilt) = _run_task(tilted)
    (x, y) = object_start_xy(cond)
    obj = [x, y, 0.0]
    (_, pre_flat, _) = grasp_poses(obj, azimuth_deg=0.0, tilt_deg=0.0)
    (_, pre_tilt, _) = grasp_poses(obj, azimuth_deg=0.0, tilt_deg=25.0)

    def commanded(arm, target, tol=0.01):
        return any((math.dist(c[0], target) <= tol for c in arm.commands))
    check(commanded(arm_flat, pre_flat), "flat run commands the spec's top-down pregrasp point")
    check(commanded(arm_tilt, pre_tilt), "tilted run commands the spec's tilted pregrasp point")
    check(math.hypot(pre_tilt[0] - x, pre_tilt[1] - y) > 0.02, f'tilted pregrasp is off to the side ({math.hypot(pre_tilt[0] - x, pre_tilt[1] - y) * 100:.1f} cm)')

def test_every_suite_has_an_expert() -> None:
    print('\n[coverage]')
    missing = [n for n in SUITES if n not in TASK_FOR_SUITE]
    check(not missing, f'every suite has a registered expert (missing: {missing})')
    extra = [n for n in TASK_FOR_SUITE if n not in SUITES]
    check(not extra, f'no expert registered for a suite the spec lacks (extra: {extra})')
    try:

        class Fake:
            suite = 'nonexistent'
        task_for(Fake())
        check(False, 'an unregistered suite raises rather than silently doing nothing')
    except KeyError:
        check(True, 'an unregistered suite raises rather than silently doing nothing')
    for name in SUITES:
        check(object_keys_for(name), f'{name}: names its object')

def main() -> int:
    test_grasp_frame_orthonormal()
    test_topdown_approach_points_down()
    test_jaw_axis_follows_azimuth()
    test_quat_conversion()
    test_azimuth_period()
    test_grasp_poses_geometry()
    test_lift_task_every_cell()
    test_tilt_is_honoured()
    test_every_suite_has_an_expert()
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

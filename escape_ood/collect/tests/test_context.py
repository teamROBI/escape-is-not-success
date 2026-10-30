# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.collect import constants as K
from escape_ood.collect.backend import dist, movement_timeout_steps, quat_angle_between, quat_normalize, quat_slerp, step_orientation_towards, step_position_towards
from escape_ood.collect.context import ObjectDropped, PhaseTimeout, TaskContext, TaskFailed
from escape_ood.collect.task import ConditionQueue, Task
from escape_ood.suites.spec import LIFT
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    print(f"  {('ok   ' if cond else 'FAIL ')} {label}")
    if not cond:
        FAILURES.append(label)

class FakeArm:
    """First-order EEF tracker standing in for differential IK.

    ``tracking`` < 1 makes convergence take several steps, which is what the budget and stall
    logic must cope with. ``blocked`` freezes motion to exercise the failure paths.
    """

    def __init__(self, pos=(0.4, 0.0, 0.3), quat=(1.0, 0.0, 0.0, 0.0), *, tracking: float=0.35, blocked: bool=False) -> None:
        self.pos = list(pos)
        self.quat = list(quat)
        self.grip = 0.04
        self.tracking = float(tracking)
        self.blocked = bool(blocked)
        self.steps = 0
        self.commands: list[tuple[list[float], list[float], float]] = []
        self.cache_cleared = 0
        self.objects: dict[str, list[float]] = {'cube': [0.5, 0.0, 0.05]}
        self.attached: str | None = None
        self.attach_offset: list[float] | None = None

    def getp(self):
        return (list(self.pos), list(self.quat))

    def getj(self):
        return [0.0] * 7
    GRASP_WIDTH = 0.02
    GRASP_REACH_M = 0.05

    def movep(self, pos_b, quat_b, gripper):
        self.commands.append((list(pos_b), list(quat_b), float(gripper)))
        self.steps += 1
        if self.blocked:
            return
        t = self.tracking
        self.pos = [c + t * (d - c) for (c, d) in zip(self.pos, pos_b)]
        self.quat = quat_slerp(self.quat, list(quat_b), t)
        self.grip = self.grip + t * (float(gripper) - self.grip)
        self._update_attachment()

    def _update_attachment(self):
        if self.grip <= self.GRASP_WIDTH:
            if self.attached is None:
                for (name, p) in self.objects.items():
                    if math.dist(self.pos, p) <= self.GRASP_REACH_M:
                        self.attached = name
                        self.attach_offset = [o - e for (o, e) in zip(p, self.pos)]
                        break
        else:
            if self.attached is not None:
                self.objects[self.attached][2] = 0.0
            self.attached = None
            self.attach_offset = None
        if self.attached is not None and self.attach_offset is not None:
            self.objects[self.attached] = [e + o for (e, o) in zip(self.pos, self.attach_offset)]

    def gripper_pos(self):
        return self.grip

    def object_pos_world(self, name):
        return list(self.objects[name])

    def object_pos_in_base(self, name):
        return list(self.objects[name])

    def step(self):
        pass

    def clear_recording_cache(self):
        self.cache_cleared += 1

def drive(gen) -> int:
    """Run a primitive generator to completion, counting control steps."""
    n = 0
    for _ in gen:
        n += 1
        if n > 5000:
            raise AssertionError('generator did not terminate')
    return n

def test_rate_caps() -> None:
    print("\n[rate caps — the expert must not command more than one step's motion]")
    cur = [0.0, 0.0, 0.0]
    far = [1.0, 0.0, 0.0]
    nxt = step_position_towards(cur, far)
    check(abs(dist(cur, nxt) - K.MAX_TRANSLATION_STEP_M) < 1e-09, f'far target steps exactly {K.MAX_TRANSLATION_STEP_M} m')
    near = [0.001, 0.0, 0.0]
    check(step_position_towards(cur, near) == near, 'target inside one step snaps exactly, so the expert converges rather than dithering')
    diag = step_position_towards(cur, [1.0, 1.0, 1.0])
    check(abs(dist(cur, diag) - K.MAX_TRANSLATION_STEP_M) < 1e-09, 'cap applies to the 3D norm, not per-axis')
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = quat_normalize([0.0, 1.0, 0.0, 0.0])
    stepped = step_orientation_towards([0.0] * 3, q0, [0.0] * 3, q1)
    arc = quat_angle_between(q0, stepped) * 0.5
    check(arc <= K.MAX_ORIENTATION_STEP_RAD + 1e-06, f'orientation step capped at {K.MAX_ORIENTATION_STEP_RAD} rad (got {arc:.4f})')
    slow = step_orientation_towards([0.0] * 3, q0, [1.0, 0.0, 0.0], q1)
    check(quat_angle_between(q0, slow) * 0.5 <= K.MAX_ORIENTATION_STEP_RAD + 1e-06, 'orientation stays capped when position is far too')

def test_slerp_shortest_arc() -> None:
    print('\n[slerp takes the shortest arc]')
    q = quat_normalize([0.7071, 0.0, 0.7071, 0.0])
    neg = [-c for c in q]
    a = quat_slerp([1.0, 0.0, 0.0, 0.0], q, 0.5)
    b = quat_slerp([1.0, 0.0, 0.0, 0.0], neg, 0.5)
    check(quat_angle_between(a, b) < 1e-06, 'q and -q are the same rotation, so they interpolate identically')
    ident = quat_slerp([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5)
    check(quat_angle_between(ident, [1.0, 0.0, 0.0, 0.0]) < 1e-06, 'identical rotations interpolate to themselves (no NaN from the sin(0) path)')
    check(all((math.isfinite(c) for c in a)), 'slerp output is finite')

def test_timeout_budget() -> None:
    print('\n[timeout budget scales with distance]')
    near = movement_timeout_steps(0.001)
    far = movement_timeout_steps(0.5)
    check(near == K.PHASE_TIMEOUT_STEPS, f'tiny move gets the floor budget ({near})')
    check(far > near, f'long move gets more budget ({far} > {near})')
    check(far <= K.MOVEMENT_TIMEOUT_MAX_STEPS, f'budget clamped at the ceiling ({far})')
    check(movement_timeout_steps(100.0) == K.MOVEMENT_TIMEOUT_MAX_STEPS, 'absurd distance saturates rather than running forever')
    check(movement_timeout_steps(0.0, math.pi) > K.PHASE_TIMEOUT_STEPS, 'orientation-only move gets a budget from the rotation, not just the translation')

def test_move_to_converges() -> None:
    print('\n[move_to]')
    arm = FakeArm()
    ctx = TaskContext(arm)
    target = [0.45, 0.05, 0.25]
    n = drive(ctx.move_to(target, [1.0, 0.0, 0.0, 0.0], label='approach'))
    check(dist(arm.pos, target) <= K.PHASE_ADVANCE_TOLERANCE_M, f'converged inside the advance tolerance in {n} steps')
    check(ctx.reached('approach'), 'phase recorded as reached')
    check(ctx.phases[-1].steps == n, 'phase step count matches yields')
    check(len(arm.commands) == n, 'exactly one command per yielded step')
    biggest = max((dist(arm.commands[i][0], arm.commands[i - 1][0]) for i in range(1, len(arm.commands))))
    check(biggest <= K.MAX_TRANSLATION_STEP_M + 1e-06, f'no commanded step exceeded the rate cap (max {biggest:.4f} m)')
    cmds = [c[0] for c in arm.commands]
    errs = [dist(c, target) for c in cmds]
    check(all((errs[i] <= errs[i - 1] + 1e-09 for i in range(1, len(errs)))), 'commanded pose approaches the target MONOTONICALLY')
    check(errs[-1] < K.MAX_TRANSLATION_STEP_M, f'the command actually reaches the target rather than asymptoting short (final command error {errs[-1]:.5f} m)')

def test_move_to_timeout_and_soft_accept() -> None:
    print('\n[move_to failure paths]')
    arm = FakeArm(blocked=True)
    ctx = TaskContext(arm)
    try:
        drive(ctx.move_to([0.9, 0.4, 0.5], label='transit', check_stall=False))
        check(False, 'blocked arm raises PhaseTimeout')
    except PhaseTimeout as e:
        check('timeout' in str(e), f'blocked arm raises PhaseTimeout ({e})')
    check(ctx.phases[-1].ok is False, 'failed phase recorded as not ok')
    arm2 = FakeArm(pos=(0.4, 0.0, 0.3))
    ctx2 = TaskContext(arm2)
    target = [0.4 + 0.025, 0.0, 0.3]
    arm2.blocked = True
    try:
        drive(ctx2.move_to(target, label='nudge', check_stall=False))
        check(True, 'a miss inside POSITION_TOLERANCE_M is accepted, not discarded')
    except PhaseTimeout:
        check(False, 'a miss inside POSITION_TOLERANCE_M is accepted, not discarded')

def test_stall_detection() -> None:
    print('\n[stall detection]')
    arm = FakeArm(blocked=True)
    ctx = TaskContext(arm)
    try:
        drive(ctx.move_to([0.9, 0.4, 0.5], label='grasp'))
        check(False, 'stalled grasp raises')
    except PhaseTimeout as e:
        steps = ctx.phases[-1].steps
        check('stall' in str(e), f'stalled grasp raises a stall error ({e})')
        check(steps <= K.STALL_WINDOW_STEPS + 2, f'failed at ~{K.STALL_WINDOW_STEPS} steps rather than the full budget (got {steps})')
    arm2 = FakeArm(blocked=True)
    ctx2 = TaskContext(arm2)
    try:
        drive(ctx2.move_to([0.9, 0.4, 0.5], label='grasp', check_stall=False))
    except PhaseTimeout as e:
        check('stall' not in str(e), 'stall check is suppressible for transit phases')

def test_drop_detection() -> None:
    print('\n[drop detection needs BOTH conditions]')
    arm = FakeArm()
    ctx = TaskContext(arm)
    ctx.object_name = 'cube'
    ctx.snapshot_object_z()
    arm.objects['cube'] = [0.5, 0.0, 0.04]
    try:
        drive(ctx.hold(3))
        check(True, 'near floor but barely moved: not a drop')
    except ObjectDropped:
        check(False, 'near floor but barely moved: not a drop')
    ctx2 = TaskContext(arm)
    ctx2.object_name = 'cube'
    arm.objects['cube'] = [0.5, 0.0, 0.9]
    ctx2.snapshot_object_z()
    arm.objects['cube'] = [0.5, 0.0, 0.6]
    try:
        drive(ctx2.hold(3))
        check(True, 'large relative drop but high up: not a drop')
    except ObjectDropped:
        check(False, 'large relative drop but high up: not a drop')
    arm.objects['cube'] = [0.5, 0.0, 0.02]
    try:
        drive(ctx2.hold(3))
        check(False, 'near floor AND far below start: IS a drop')
    except ObjectDropped:
        check(True, 'near floor AND far below start: IS a drop')

def test_gripper_and_hold() -> None:
    print('\n[gripper + hold]')
    arm = FakeArm()
    ctx = TaskContext(arm)
    n = drive(ctx.close_gripper())
    check(n == K.CLOSE_HOLD_STEPS + K.POST_CLOSE_SETTLE_STEPS, f'close runs close+settle steps ({n})')
    check(ctx.settled_gripper is not None, 'settled gripper width recorded')
    grips = [c[2] for c in arm.commands]
    biggest = max((abs(grips[i] - grips[i - 1]) for i in range(1, len(grips))))
    check(biggest <= K.MAX_GRIPPER_STEP + 1e-09, f'gripper respects its rate cap (max {biggest:.5f})')
    check(all((grips[i] <= grips[i - 1] + 1e-09 for i in range(1, len(grips)))), 'gripper command is MONOTONIC: re-reading the actual position would cap grip force at one step once the fingers contact the object')
    check(abs(grips[-1] - ctx.closed_value) < 1e-06, f'command reaches fully closed and is held there (final {grips[-1]:.4f})')
    arm2 = FakeArm()
    ctx2 = TaskContext(arm2)
    n2 = drive(ctx2.hold(7, discard=True))
    check(n2 == 7, 'hold yields exactly the requested steps')
    check(arm2.cache_cleared == 1, 'discard clears the recording cache once')
    ctx3 = TaskContext(arm2)
    drive(ctx3.hold(3))
    check(arm2.cache_cleared == 1, 'a non-discard hold does not clear the cache')


def test_success_checks() -> None:
    print('\n[success checks]')
    arm = FakeArm()
    ctx = TaskContext(arm)
    ctx.object_name = 'cube'
    arm.objects['cube'] = list(arm.pos)
    try:
        ctx.check_lift()
        check(False, 'check_lift without snapshot_object_z() must refuse')
    except TaskFailed as e:
        check(not e.retriable, 'missing baseline is a bug, not an unlucky attempt')
    arm.objects['cube'] = [arm.pos[0], arm.pos[1], 0.0]
    ctx.snapshot_object_z()
    arm.objects['cube'] = [arm.pos[0], arm.pos[1], 0.1]
    try:
        ctx.check_lift()
        check(True, 'risen object under the gripper passes check_lift')
    except TaskFailed as e:
        check(False, f'risen object under the gripper passes check_lift ({e})')
    arm.objects['cube'] = [arm.pos[0], arm.pos[1], 0.0]
    try:
        ctx.check_lift()
        check(False, 'an object that never rose fails check_lift')
    except TaskFailed as e:
        check(e.retriable, 'a failed grasp is retriable')
    arm.objects['cube'] = [arm.pos[0] + 0.5, arm.pos[1], 0.1]
    try:
        ctx.check_lift()
        check(False, 'object far from the gripper fails check_lift')
    except TaskFailed as e:
        check(e.retriable, 'check_lift failure is retriable (an unlucky attempt, not a bad condition)')

def test_condition_queue() -> None:
    print('\n[condition queue — advances only on confirmed success]')
    q = ConditionQueue(LIFT, ['cube_a', 'cube_b'], demos_per_condition=3)
    total = q.total_episodes
    check(total > 0, f'queue non-empty ({total} episodes)')
    c = q.next_condition()
    check(c is not None and c.split == 'train', 'only train conditions are collected')
    before = q.collected
    for _ in range(5):
        q.record_failure(q.next_condition())
    check(q.collected == before, 'failures do NOT advance progress')
    q.confirm_success(q.next_condition())
    check(q.collected == before + 1, 'a confirmed success advances progress by one')
    q2 = ConditionQueue(LIFT, ['cube_a'], demos_per_condition=2)
    n_conds = len(q2._order)
    seen = []
    for _ in range(n_conds):
        cc = q2.next_condition()
        seen.append(cc.key())
        q2.confirm_success(cc)
    check(len(set(seen)) == n_conds, f'first {n_conds} episodes cover all {n_conds} conditions once (round-robin, not drain-one-first)')
    q3 = ConditionQueue(LIFT, ['cube_a'], demos_per_condition=1)
    guard = 0
    while not q3.done and guard < 10000:
        q3.confirm_success(q3.next_condition())
        guard += 1
    check(q3.done, 'queue drains to done')
    check(q3.next_condition() is None, 'an exhausted queue returns None')
    check(q3.collected == q3.total_episodes, 'collected == total when done')
    parts = [ConditionQueue(LIFT, ['cube_a'], demos_per_condition=1, partition_idx=i, partition_total=3) for i in range(3)]
    keys = [k for p in parts for k in p._remaining]
    whole = ConditionQueue(LIFT, ['cube_a'], demos_per_condition=1)
    check(sorted(keys) == sorted(whole._remaining), '3 partitions tile the suite exactly once')
    check(len(set(keys)) == len(keys), 'no condition appears in two partitions')
    try:
        ConditionQueue(LIFT, ['cube_a'], partition_idx=3, partition_total=3)
        check(False, 'an out-of-range partition is rejected')
    except ValueError:
        check(True, 'an out-of-range partition is rejected')

def test_task_base() -> None:
    print('\n[task base]')
    c = ConditionQueue(LIFT, ['cube_a']).next_condition()

    class Lift(Task):

        def run(self, ctx):
            yield from ctx.hold(2)
    t = Lift(c)
    check(t.instruction == c.instruction, 'instruction comes from the condition, so it matches the suite')
    check(drive(t.run(TaskContext(FakeArm()))) == 2, 'task generator drives')
    try:
        drive(Task(c).run(TaskContext(FakeArm())))
        check(False, 'the base Task refuses to run unimplemented')
    except NotImplementedError:
        check(True, 'the base Task refuses to run unimplemented')

def main() -> int:
    test_rate_caps()
    test_slerp_shortest_arc()
    test_timeout_budget()
    test_move_to_converges()
    test_move_to_timeout_and_soft_accept()
    test_stall_detection()
    test_drop_detection()
    test_gripper_and_hold()
    test_success_checks()
    test_condition_queue()
    test_task_base()
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

# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from typing import Generator
from escape_ood.collect import constants as K
from escape_ood.collect.context import TaskContext, TaskFailed
from escape_ood.collect.grasp import grasp_poses
from escape_ood.collect.task import Task
from escape_ood.suites.spec import WORKSPACE, target_xys
Vec3 = list[float]

class LiftTask(Task):
    """Pick up the target object and lift it clear of the table.

    The object sits at the swept cell; the grasp is top-down with the jaw azimuth along the
    cell's bearing. ``tilt_deg`` on the condition (if the reach probe decides the far bands
    need a leaning approach) is honoured here and nowhere else, so the expert and the probe
    agree on what "the same grasp" means across bands.
    """

    def run(self, ctx: TaskContext) -> Generator:
        cond = self.condition
        azimuth = float(cond.params.get('phi_deg', 0.0))
        tilt = float(cond.params.get('tilt_deg', 0.0))
        obj = ctx.b.object_pos_in_base(ctx.object_name)
        (grasp, pregrasp, quat) = grasp_poses(obj, azimuth_deg=azimuth, tilt_deg=tilt)
        ctx.snapshot_object_z()
        yield from ctx.hold(K.DISCARD_HOLD_STEPS, discard=True)
        yield from ctx.hold(K.RECORD_HOLD_STEPS)
        yield from ctx.move_to(pregrasp, quat, gripper='open', label='pregrasp')
        yield from ctx.move_to(grasp, quat, gripper='open', label='grasp')
        yield from ctx.close_gripper()
        yield from ctx.lift(K.LIFT_HEIGHT_M)
        yield from ctx.retract()
        yield from ctx.hold(K.SETTLE_HOLD_STEPS, label='settle')
        ctx.check_lift()
TASK_FOR_SUITE: dict[str, type[Task]] = {'lift': LiftTask}

def task_for(condition) -> type[Task]:
    name = condition.suite
    if name not in TASK_FOR_SUITE:
        raise KeyError(f"no scripted expert registered for suite '{name}'. Known: {', '.join(sorted(TASK_FOR_SUITE))}")
    return TASK_FOR_SUITE[name]

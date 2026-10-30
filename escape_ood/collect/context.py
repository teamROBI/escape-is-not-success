# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Generator-based action primitives for scripted demonstration collection.

A task composes primitives with ``yield from``, and each primitive yields exactly once per
control step. That inversion is what lets one driver loop own stepping, recording, camera
capture, and abort handling, while a task reads as a straight-line script:

    def run(self, ctx):
        yield from ctx.hold(DISCARD_HOLD_STEPS, discard=True)
        yield from ctx.move_to(pregrasp_pos, grasp_quat, gripper="open", label="pregrasp")
        yield from ctx.move_to(grasp_pos, grasp_quat, gripper="open", label="grasp")
        yield from ctx.close_gripper()
        yield from ctx.lift(LIFT_HEIGHT_M)
        yield from ctx.retract()

Adapted from the collection stack this repo was split out of. Dropped on the way in: every
bimanual primitive, the mobile-base and head/lift primitives, the GraspGen grasp predictor,
and teleop — none of which apply to a fixed single-arm Franka, and all of which were dead
branches waiting to confuse someone.

Added: :meth:`TaskContext.route`, which the TOPOLOGY axis needs. Phase state lives on the
context rather than being pushed into a controller, so the driver can read it for the staged
progress score without a second bookkeeping path.

Three failure modes are distinguished deliberately, because the collection queue treats them
differently: :class:`PhaseTimeout` and :class:`ObjectDropped` mean *this attempt* failed and is
worth retrying, while :class:`TaskFailed` carries an explicit ``retriable`` flag for a task
that has decided the condition itself is bad.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Generator
from escape_ood.collect import constants as K
from escape_ood.collect import success
from escape_ood.collect.backend import MotionBackend, Quat, Vec3, dist, movement_timeout_steps, quat_angle_between, step_orientation_towards, step_position_towards

class PhaseTimeout(Exception):
    """A phase did not converge inside its step budget, or stalled short of the target."""

class ObjectDropped(Exception):
    """The manipulated object fell — near the floor and well below where it started."""

class TaskFailed(Exception):
    """A task declared failure. ``retriable`` decides whether the queue re-attempts."""

    def __init__(self, message: str, *, retriable: bool=False) -> None:
        super().__init__(message)
        self.retriable = bool(retriable)

@dataclass
class Phase:
    """One phase of an episode. The driver reads these for the staged progress score."""
    label: str
    budget: int
    steps: int = 0
    final_error: float | None = None
    ok: bool = False

class TaskContext:
    pass

    def __init__(self, backend: MotionBackend, *, open_gripper_value: float=0.04, closed_gripper_value: float=0.0) -> None:
        self.b = backend
        self.open_value = float(open_gripper_value)
        self.closed_value = float(closed_gripper_value)
        (pos, quat) = backend.getp()
        self.init_eef_pos: Vec3 = list(pos)
        self.init_eef_quat: Quat = list(quat)
        self.init_joints: list[float] = list(backend.getj())
        self.object_name: str = ''
        self.rest_object_z: float | None = None
        self.settled_gripper: float | None = None
        self.phases: list[Phase] = []
        self._phase: Phase | None = None

    @property
    def eef_pos(self) -> Vec3:
        return list(self.b.getp()[0])

    @property
    def eef_quat(self) -> Quat:
        return list(self.b.getp()[1])

    def object_z(self, name: str | None=None) -> float:
        return float(self.b.object_pos_world(name or self.object_name)[2])

    def snapshot_object_z(self) -> None:
        """Record the object's resting height, the baseline for lift and drop checks."""
        self.rest_object_z = self.object_z()

    def _begin(self, label: str, budget: int) -> Phase:
        self._phase = Phase(label=label, budget=int(budget))
        self.phases.append(self._phase)
        return self._phase

    def _end(self, ok: bool, error: float | None=None) -> None:
        if self._phase is not None:
            self._phase.ok = ok
            self._phase.final_error = error

    def reached(self, label: str) -> bool:
        """Did a phase with this label complete successfully?"""
        return any((p.label == label and p.ok for p in self.phases))

    @property
    def last_phase(self) -> str:
        return self.phases[-1].label if self.phases else ''

    def _check_drop(self) -> None:
        """Both conditions must hold: near the floor *and* far below the start height.

        Either alone gives false positives — a low table trips the floor test, a deliberate
        place trips the relative-drop test.
        """
        if not self.object_name or self.rest_object_z is None:
            return
        z = self.object_z()
        if z <= K.FALL_FLOOR_Z_THRESHOLD_M and z <= self.rest_object_z - K.FALL_RELATIVE_DROP_M:
            raise ObjectDropped(f'object dropped (z={z:.3f}m, start={self.rest_object_z:.3f}m)')

    def _resolve_gripper(self, gripper: float | str | None) -> float:
        if gripper == 'open':
            return self.open_value
        if gripper == 'close':
            return self.closed_value
        if gripper is None:
            return self.open_value if self.settled_gripper is None else float(self.settled_gripper)
        return float(gripper)

    def hold(self, steps: int, *, discard: bool=False, gripper: float | str | None=None, label: str | None=None) -> Generator:
        """Hold the current pose. ``discard`` drops the frames from the recording."""
        lbl = label or ('hold_discard' if discard else 'hold_record')
        ph = self._begin(lbl, steps)
        (pos, quat) = self.b.getp()
        (pos, quat) = (list(pos), list(quat))
        g = self._resolve_gripper(gripper)
        for i in range(steps):
            self._check_drop()
            self.b.movep(pos, quat, g)
            ph.steps = i + 1
            yield
        if discard:
            self.b.clear_recording_cache()
        self._end(True)

    def move_to(self, pos_b: Vec3, quat_b: Quat | None=None, *, gripper: float | str | None=None, label: str='move', orientation_tolerance: float | None=None, check_stall: bool | None=None) -> Generator:
        pass
        target_pos = list(pos_b)
        target_quat = list(quat_b) if quat_b is not None else None
        (cur_pos, cur_quat) = self.b.getp()
        orient_err = quat_angle_between(list(cur_quat), target_quat) if target_quat is not None else None
        budget = movement_timeout_steps(dist(list(cur_pos), target_pos), orient_err)
        ph = self._begin(label, budget)
        g = self._resolve_gripper(gripper)
        stall = check_stall if check_stall is not None else label in ('pregrasp', 'grasp')
        window: list[float] = []
        error = 0.0
        (cmd_pos, cmd_quat) = self.b.getp()
        (cmd_pos, cmd_quat) = (list(cmd_pos), list(cmd_quat))
        for i in range(budget):
            self._check_drop()
            (cur_pos, cur_quat) = self.b.getp()
            (cur_pos, cur_quat) = (list(cur_pos), list(cur_quat))
            error = dist(cur_pos, target_pos)
            if error <= K.PHASE_ADVANCE_TOLERANCE_M:
                converged = True
                if orientation_tolerance is not None and target_quat is not None:
                    converged = quat_angle_between(cur_quat, target_quat) <= orientation_tolerance
                if converged:
                    self._end(True, error)
                    return
            elif stall:
                window.append(error)
                if len(window) > K.STALL_WINDOW_STEPS:
                    window.pop(0)
                if len(window) == K.STALL_WINDOW_STEPS and window[0] - min(window) < K.STALL_THRESHOLD_M:
                    self._end(False, error)
                    raise PhaseTimeout(f'{label} stall (err={error:.3f}m, no progress in {K.STALL_WINDOW_STEPS} steps)')
            cmd_pos = step_position_towards(cmd_pos, target_pos)
            if target_quat is not None:
                cmd_quat = step_orientation_towards(cmd_pos, cmd_quat, target_pos, target_quat)
            self.b.movep(cmd_pos, cmd_quat, g)
            ph.steps = i + 1
            yield
        error = dist(list(self.b.getp()[0]), target_pos)
        if error <= K.POSITION_TOLERANCE_M:
            self._end(True, error)
            return
        self._end(False, error)
        raise PhaseTimeout(f'{label} timeout (err={error:.3f}m after {budget} steps)')

    def close_gripper(self, *, label: str='close') -> Generator:
        pass
        ph = self._begin(label, K.CLOSE_HOLD_STEPS + K.POST_CLOSE_SETTLE_STEPS)
        (pos, quat) = self.b.getp()
        (pos, quat) = (list(pos), list(quat))
        cmd = self.b.gripper_pos()
        target = self.closed_value
        direction = 1.0 if target > cmd else -1.0
        for i in range(K.CLOSE_HOLD_STEPS):
            self._check_drop()
            if abs(cmd - target) > 1e-06:
                cmd += direction * K.MAX_GRIPPER_STEP
                cmd = min(cmd, target) if direction > 0 else max(cmd, target)
            self.b.movep(pos, quat, cmd)
            ph.steps = i + 1
            yield
        for i in range(K.POST_CLOSE_SETTLE_STEPS):
            self._check_drop()
            self.b.movep(pos, quat, cmd)
            ph.steps = K.CLOSE_HOLD_STEPS + i + 1
            yield
        self.settled_gripper = cmd
        self._end(True)

    def lift(self, height_m: float=K.LIFT_HEIGHT_M, *, label: str='lift') -> Generator:
        """Raise the EEF straight up, holding orientation."""
        (pos, quat) = self.b.getp()
        target = list(pos)
        target[2] += float(height_m)
        yield from self.move_to(target, list(quat), label=label)

    def retract(self, *, gripper: float | str | None=None, label: str='retract') -> Generator:
        """Return to the pose the episode started from."""
        yield from self.move_to(list(self.init_eef_pos), list(self.init_eef_quat), gripper=gripper, label=label)

    def check_lift(self, min_delta: float=K.SUCCESS_RETRACT_DIST_M) -> None:
        pass
        if not self.object_name:
            raise TaskFailed('check_lift called with no object_name set')
        if self.rest_object_z is None:
            raise TaskFailed('check_lift needs snapshot_object_z() first', retriable=False)
        obj_b = self.b.object_pos_in_base(self.object_name)
        (ok, why) = success.lifted(object_z_world=self.object_z(), rest_z_world=self.rest_object_z, object_xy=obj_b, eef_xy=self.eef_pos, max_xy=min_delta + K.POSITION_TOLERANCE_M)
        if not ok:
            raise TaskFailed(why, retriable=True)

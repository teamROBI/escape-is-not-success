# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from escape_ood.collect.grasp import GRASP_POINT_OFFSET_M
Vec3 = list[float]
Quat = list[float]

class IsaacMotionBackend:
    """Drive one Franka in one environment through differential IK.

    Single-environment by design. Demonstration collection is inherently sequential -- the
    queue advances only on a confirmed success, and an episode's outcome decides whether the
    next attempt repeats the condition -- so batching envs would buy nothing and would make the
    deterministic queue order meaningless.
    """
    DECIMATION = 4

    def __init__(self, scene, sim, *, env_index: int=0, arm_joint_expr: str='panda_joint.*', hand_body: str='panda_hand', gripper_joint_expr: str='panda_finger_joint.*') -> None:
        self._scene = scene
        self._sim = sim
        self._i = env_index
        self.robot = scene['robot']
        self._arm = SceneEntityCfg('robot', joint_names=[arm_joint_expr], body_names=[hand_body])
        self._arm.resolve(scene)
        self._grip = SceneEntityCfg('robot', joint_names=[gripper_joint_expr])
        self._grip.resolve(scene)
        self._jac_idx = self._arm.body_ids[0] - 1 if self.robot.is_fixed_base else self._arm.body_ids[0]
        self._ik = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False, ik_method='dls'), num_envs=scene.num_envs, device=sim.device)
        self._dev = sim.device
        self._cmd = torch.zeros(scene.num_envs, self._ik.action_dim, device=self._dev)
        self._last_action: list[float] | None = None
        self._last_pose: tuple[list[float], list[float], float] | None = None
        self._recorder = None

    def _wrist_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_w = self.robot.data.body_pose_w[:, self._arm.body_ids[0]]
        root_w = self.robot.data.root_pose_w
        return subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])

    @staticmethod
    def _approach_from_quat(q: Quat) -> Vec3:
        """Third column of the rotation -- the gripper's approach axis (eef_z)."""
        (w, x, y, z) = q
        return [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]

    def getp(self) -> tuple[Vec3, Quat]:
        """Grasp-point pose in the base frame, not the wrist pose.

        A task reasons about where the fingertips are. Returning the wrist here would make
        every `move_to` target silently off by the 10 cm hand offset in whatever direction the
        gripper happened to be pointing.
        """
        (pos_b, quat_b) = self._wrist_in_base()
        q = quat_b[self._i].detach().cpu().tolist()
        p = pos_b[self._i].detach().cpu().tolist()
        a = self._approach_from_quat(q)
        return ([p[j] + a[j] * GRASP_POINT_OFFSET_M for j in range(3)], q)

    def getj(self) -> list[float]:
        return self.robot.data.joint_pos[self._i, self._arm.joint_ids].detach().cpu().tolist()

    def movep(self, pos_b: Vec3, quat_b: Quat, gripper: float) -> None:
        """Set an IK target for the grasp point and a gripper width. Does not step."""
        a = self._approach_from_quat(list(quat_b))
        wrist = [float(pos_b[j]) - a[j] * GRASP_POINT_OFFSET_M for j in range(3)]
        self._cmd[self._i, 0:3] = torch.tensor(wrist, device=self._dev)
        self._cmd[self._i, 3:7] = torch.tensor([float(c) for c in quat_b], device=self._dev)
        self._ik.set_command(self._cmd)
        jac = self.robot.root_physx_view.get_jacobians()[:, self._jac_idx, :, self._arm.joint_ids]
        (pos_b_now, quat_b_now) = self._wrist_in_base()
        jp = self.robot.data.joint_pos[:, self._arm.joint_ids]
        des = self._ik.compute(pos_b_now, quat_b_now, jac, jp)
        self.robot.set_joint_position_target(des, joint_ids=self._arm.joint_ids)
        g = torch.full((self._scene.num_envs, len(self._grip.joint_ids)), float(gripper), device=self._dev)
        self.robot.set_joint_position_target(g, joint_ids=self._grip.joint_ids)
        self._last_action = des[self._i].detach().cpu().tolist() + [float(gripper)]
        self._last_pose = ([float(c) for c in pos_b], [float(c) for c in quat_b], float(gripper))

    def movej(self, arm_targets, gripper: float) -> None:
        """Command joint position targets directly: 7 arm joints + gripper width.

        This is the action space the dataset records, so it is the one a policy acts in.
        `movep` cannot serve here -- it solves IK from an end-effector pose, so feeding it a
        policy's joint output would first have to convert joints to a pose and back, and the
        round trip is not the identity. The recorded action is what `movep` *sent*, and this
        replays that same quantity.
        """
        if len(arm_targets) != len(self._arm.joint_ids):
            raise ValueError(f'expected {len(self._arm.joint_ids)} arm targets, got {len(arm_targets)}')
        des = torch.tensor([[float(v) for v in arm_targets]], device=self._dev, dtype=self.robot.data.joint_pos.dtype)
        des = des.repeat(self._scene.num_envs, 1)
        self.robot.set_joint_position_target(des, joint_ids=self._arm.joint_ids)
        g = torch.full((self._scene.num_envs, len(self._grip.joint_ids)), float(gripper), device=self._dev)
        self.robot.set_joint_position_target(g, joint_ids=self._grip.joint_ids)
        self._last_action = [float(v) for v in arm_targets] + [float(gripper)]
        self._last_pose = None

    def gripper_pos(self) -> float:
        return float(self.robot.data.joint_pos[self._i, self._grip.joint_ids[0]].detach().cpu().item())

    def object_pos_world(self, name: str) -> Vec3:
        return self._scene[name].data.root_pos_w[self._i, :3].detach().cpu().tolist()

    def object_pos_in_base(self, name: str) -> Vec3:
        obj_w = self._scene[name].data.root_pos_w[:, :3]
        root_w = self.robot.data.root_pose_w
        quat_w = self._scene[name].data.root_quat_w
        (pos_b, _) = subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], obj_w, quat_w)
        return pos_b[self._i].detach().cpu().tolist()

    def step(self) -> None:
        """One control step: DECIMATION physics steps against the current joint targets."""
        for _ in range(self.DECIMATION):
            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self._sim.get_physics_dt())

    def last_action(self) -> list[float]:
        """The most recent commanded action: 7 arm joint targets + gripper width.

        Raises rather than returning a zero vector if nothing has been commanded yet: a silent
        zero would be recorded as a real action and teach the policy a spurious first step.
        """
        if self._last_action is None:
            raise RuntimeError('no action commanded yet; call movep() before last_action()')
        return list(self._last_action)

    def last_pose(self) -> tuple[list[float], list[float], float] | None:
        """The most recent commanded grasp-point pose ``(pos_xyz, quat_wxyz, gripper)``, or
        ``None`` if the last command was joint-space (``movej``). Unlike ``last_action`` this
        does not raise: a replay driven by ``movej`` legitimately has no EEF command."""
        return None if self._last_pose is None else (list(self._last_pose[0]), list(self._last_pose[1]), self._last_pose[2])

    def attach_recorder(self, recorder) -> None:
        """Route ``clear_recording_cache`` to a recorder's discard mark."""
        self._recorder = recorder

    def clear_recording_cache(self) -> None:
        """Drop frames captured so far. No-op when nothing is recording (e.g. the gate)."""
        if self._recorder is not None:
            self._recorder.mark_discard_point()

    def drive(self, generator, *, max_steps: int=3000) -> int:
        """Run a task generator to completion, stepping once per yield.

        One yield equals one physics step equals (later) one recorded frame. That is the
        invariant the primitives were written against -- ``move_to`` tests convergence before
        commanding precisely so every command it issues gets exactly one step.
        """
        n = 0
        for _ in generator:
            self.step()
            n += 1
            if n >= max_steps:
                raise TimeoutError(f'task exceeded {max_steps} control steps')
        return n

    def reset_to(self, joint_pos: list[float] | None=None) -> None:
        """Put the arm at a known pose. A deterministic start is part of the protocol."""
        jp = self.robot.data.default_joint_pos.clone()
        if joint_pos is not None:
            jp[:, self._arm.joint_ids] = torch.tensor([joint_pos], device=self._dev, dtype=jp.dtype).repeat(jp.shape[0], 1)
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        self.robot.reset()
        self._ik.reset()

# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import numpy as np

def q_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)

def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    (aw, ax, ay, az) = (a[..., 0], a[..., 1], a[..., 2], a[..., 3])
    (bw, bx, by, bz) = (b[..., 0], b[..., 1], b[..., 2], b[..., 3])
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)

def q_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation vector for unit quaternions (T,4) -> (T,3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    axis = np.where(s[..., None] > 1e-12, v / np.maximum(s[..., None], 1e-12), 0.0)
    return axis * angle[..., None]

def eef_delta_actions(actions_eef: np.ndarray, eef_pose: np.ndarray) -> np.ndarray:
    """(T,8) commanded pose+gripper and (T,7) achieved pose -> (T,7) [dxyz, rotvec, gripper]."""
    d_pos = actions_eef[:, :3] - eef_pose[:, :3]
    q_rel = q_mul(actions_eef[:, 3:7], q_conj(eef_pose[:, 3:7]))
    d_rot = q_to_rotvec(q_rel)
    return np.concatenate([d_pos, d_rot, actions_eef[:, 7:8]], axis=1).astype(np.float32)
ROT_REF_WXYZ = (0.0, 1.0, 0.0, 0.0)

def rotvec_to_q(r: np.ndarray) -> np.ndarray:
    """Rotation vectors (T,3) -> unit quaternions (T,4), (w, x, y, z)."""
    angle = np.linalg.norm(r, axis=-1)
    half = 0.5 * angle
    scale = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate([np.cos(half)[..., None], r * scale[..., None]], axis=-1)

def eef_abs_rotvec_actions(actions_eef: np.ndarray) -> np.ndarray:
    """(T,8) commanded pose+gripper -> (T,7) [xyz, rotvec(q_cmd * q_ref^-1), gripper].

    Exactly eef_delta's formula with the measured pose replaced by a constant (position anchor 0,
    rotation anchor ROT_REF_WXYZ): the absolute/residual contrast then differs only in its anchor.
    """
    q_ref = np.broadcast_to(np.asarray(ROT_REF_WXYZ, dtype=actions_eef.dtype), actions_eef[:, 3:7].shape)
    rot = q_to_rotvec(q_mul(actions_eef[:, 3:7], q_conj(q_ref)))
    return np.concatenate([actions_eef[:, :3], rot, actions_eef[:, 7:8]], axis=1).astype(np.float32)

def eef_abs_rotvec_to_pose(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of eef_abs_rotvec_actions: (T,7) -> (xyz (T,3), wxyz (T,4), gripper (T,))."""
    q_ref = np.broadcast_to(np.asarray(ROT_REF_WXYZ, dtype=np.float64), (len(a), 4))
    return (a[:, :3], q_mul(rotvec_to_q(np.asarray(a[:, 3:6], dtype=np.float64)), q_ref), a[:, 6])

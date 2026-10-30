# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
from escape_ood.collect.constants import FINAL_APPROACH_EXTRA_M, PREGRASP_OFFSET_M
Vec3 = list[float]
Quat = list[float]
GRASP_POINT_OFFSET_M = 0.1034

def mat_to_quat(r: list[Vec3]) -> Quat:
    """Rotation matrix (row-major 3x3) to quaternion ``(w, x, y, z)``.

    Branches on the largest diagonal term rather than always using the trace formula: the
    trace branch divides by ``sqrt(1 + trace)``, which loses precision and then blows up for
    rotations near 180 degrees — and a top-down grasp is a 180-degree rotation about a
    horizontal axis, so that degenerate case is the *common* one here, not an edge case.
    """
    (m00, m01, m02) = r[0]
    (m10, m11, m12) = r[1]
    (m20, m21, m22) = r[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]

def _normalize(v: Vec3) -> Vec3:
    n = math.sqrt(sum((c * c for c in v)))
    if n < 1e-12:
        raise ValueError(f'cannot normalize a zero vector: {v}')
    return [c / n for c in v]

def _cross(a: Vec3, b: Vec3) -> Vec3:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

def grasp_frame(azimuth_deg: float, tilt_deg: float=0.0) -> list[Vec3]:
    """Column-major-as-rows rotation matrix for a grasp at this azimuth and tilt.

    ``tilt_deg`` rotates the approach away from straight-down, toward the azimuth direction.
    Zero (straight down) is right for small primitives on a table; a nonzero tilt exists
    because a cluttered or fixture-bounded approach sometimes needs to come in at a slant.
    """
    phi = math.radians(azimuth_deg)
    tilt = math.radians(tilt_deg)
    heading = [math.cos(phi), math.sin(phi), 0.0]
    approach = _normalize([heading[0] * math.sin(tilt), heading[1] * math.sin(tilt), -math.cos(tilt)])
    eef_z = approach
    proj = sum((h * z for (h, z) in zip(heading, eef_z)))
    jaw = [h - proj * z for (h, z) in zip(heading, eef_z)]
    if math.sqrt(sum((c * c for c in jaw))) < 1e-08:
        jaw = [0.0, 0.0, 1.0]
        proj = sum((j * z for (j, z) in zip(jaw, eef_z)))
        jaw = [j - proj * z for (j, z) in zip(jaw, eef_z)]
    eef_x = _normalize(jaw)
    eef_y = _cross(eef_z, eef_x)
    return [[eef_x[0], eef_y[0], eef_z[0]], [eef_x[1], eef_y[1], eef_z[1]], [eef_x[2], eef_y[2], eef_z[2]]]

def grasp_quat(azimuth_deg: float, tilt_deg: float=0.0) -> Quat:
    """Quaternion ``(w, x, y, z)`` for a grasp at this azimuth and tilt."""
    return mat_to_quat(grasp_frame(azimuth_deg, tilt_deg))

def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product of two ``(w, x, y, z)`` quaternions: ``a`` after ``b``."""
    (aw, ax, ay, az) = a
    (bw, bx, by, bz) = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)

def rotvec_to_quat(v: Vec3) -> Quat:
    """Rotation vector (axis * angle, radians) -> unit quaternion ``(w, x, y, z)``.

    Inverse of the rotation-vector delta the eef-delta datasets carry, so an evaluator can turn
    a policy's ``[drx, dry, drz]`` back into a pose target: ``q_target = rotvec_to_quat(d) * q_now``.
    """
    angle = math.sqrt(sum((c * c for c in v)))
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(0.5 * angle) / angle
    return (math.cos(0.5 * angle), v[0] * s, v[1] * s, v[2] * s)

def canonical_azimuth(azimuth_deg: float) -> float:
    """Fold an azimuth into ``[0, 180)``, the period of a parallel-jaw grasp."""
    return float(azimuth_deg) % 180.0

def grasp_poses(object_pos: Vec3, *, azimuth_deg: float=0.0, tilt_deg: float=0.0, origin_to_centre_z: float=0.0, pregrasp_offset_m: float=PREGRASP_OFFSET_M) -> tuple[Vec3, Vec3, Quat]:
    pass
    frame = grasp_frame(azimuth_deg, tilt_deg)
    approach = [frame[0][2], frame[1][2], frame[2][2]]
    grasp = [float(object_pos[0]), float(object_pos[1]), float(object_pos[2]) + float(origin_to_centre_z)]
    standoff = pregrasp_offset_m + FINAL_APPROACH_EXTRA_M
    pregrasp = [g - a * standoff for (g, a) in zip(grasp, approach)]
    return (grasp, pregrasp, mat_to_quat(frame))

# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import os
import argparse
import json
import math
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from escape_ood.suites.spec import OBJECT_FOR_SUITE, SUITES, WORKSPACE, Suite, enumerate_conditions, object_keys_for, suite as get_suite
SCHEMA = 'escape_ood.scene'
TABLE_TOP_Z = 0.75
ROBOT_POS = [0.0, 0.0, TABLE_TOP_Z]
ROBOT_RPY = [0.0, 0.0, 0.0]
HEAD_CAM_WIDTH = 672
HEAD_CAM_HEIGHT = 376
HEAD_CAM_APERTURE_MM = 20.955
HEAD_CAM_FOCAL_MM = 14.15
HEAD_CAM = {'name': 'cam_head', 'pos': [1.1, 0.0, 1.4], 'look_at': [0.5, 0.0, 0.78], 'width': HEAD_CAM_WIDTH, 'height': HEAD_CAM_HEIGHT, 'focal_mm': HEAD_CAM_FOCAL_MM}
WRIST_CAM = {'name': 'cam_wrist', 'parent_body': 'panda_hand', 'offset_pos': [0.05, 0.0, 0.0], 'offset_rot': [0.707107, 0.0, 0.0, 0.707107], 'width': 424, 'height': 240, 'focal_mm': 7.73, 'clip_near': 0.01, 'clip_far': 5.0}
FIG_CAM = {'name': 'cam_fig', 'pos': [0.7, -1.3, 1.95], 'look_at': [0.47, 0.02, 0.8], 'width': 1280, 'height': 800, 'focal_mm': 14.0}
if os.environ.get('ESCAPE_FIG_CAM_POSE'):
    _v = [float(x) for x in os.environ['ESCAPE_FIG_CAM_POSE'].split(',')]
    FIG_CAM = dict(FIG_CAM, pos=_v[0:3], look_at=_v[3:6], focal_mm=_v[6])
if os.environ.get('ESCAPE_WRIST_LEGACY', '') == '1':
    WRIST_CAM = dict(WRIST_CAM, offset_pos=[0.0, 0.0, -0.04], offset_rot=[1.0, 0.0, 0.0, 0.0], focal_mm=12.0)
OBJECT_POOL: list[str] = list(dict.fromkeys(OBJECT_FOR_SUITE.values()))
OBJECT_CENTRE_Z = 0.025

def _norm(v):
    n = math.sqrt(sum((c * c for c in v)))
    return tuple((c / n for c in v))

def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

def head_cam_frame(cam: dict=HEAD_CAM):
    """(forward, right, up) unit vectors of the head camera in world coordinates.

    forward = look_at - pos; right = forward x world_up; up = right x forward. For the far-side
    camera looking back along -x this gives image-right = +y and image-up = -x, matching what
    the rendered frames show.
    """
    pos = tuple((float(v) for v in cam['pos']))
    tgt = tuple((float(v) for v in cam['look_at']))
    fwd = _norm(tuple((t - p for (t, p) in zip(tgt, pos))))
    right = _norm(_cross(fwd, (0.0, 0.0, 1.0)))
    up = _cross(right, fwd)
    return (fwd, right, up)

def project_to_head_cam(x_w: float, y_w: float, z_w: float, cam: dict=HEAD_CAM) -> tuple[float, float, float]:
    """World point -> (u, v, depth) in the head camera image. u right, v down, pixels.

    depth <= 0 means the point is beside or behind the camera; treat as out of frame.
    """
    pos = tuple((float(v) for v in cam['pos']))
    (fwd, right, up) = head_cam_frame(cam)
    d = (x_w - pos[0], y_w - pos[1], z_w - pos[2])
    depth = sum((a * b for (a, b) in zip(d, fwd)))
    if depth <= 1e-09:
        return (float('nan'), float('nan'), depth)
    (w, h) = (float(cam['width']), float(cam['height']))
    f_px = float(cam['focal_mm']) / HEAD_CAM_APERTURE_MM * w
    u = w / 2.0 + f_px * sum((a * b for (a, b) in zip(d, right))) / depth
    v = h / 2.0 - f_px * sum((a * b for (a, b) in zip(d, up))) / depth
    return (u, v, depth)

def in_head_frame(x_w: float, y_w: float, z_w: float, *, margin_frac: float=0.1, cam: dict=HEAD_CAM) -> bool:
    """Whether a world point lands inside the head image with ``margin_frac`` of border kept."""
    (u, v, depth) = project_to_head_cam(x_w, y_w, z_w, cam)
    if depth <= 0 or math.isnan(u):
        return False
    (w, h) = (float(cam['width']), float(cam['height']))
    return margin_frac * w <= u <= (1 - margin_frac) * w and margin_frac * h <= v <= (1 - margin_frac) * h

def _pool_entry(asset_name: str) -> dict:
    return {'asset_name': asset_name, 'library': 'escape_ood_assets', 'scale': [1.0, 1.0, 1.0]}

def _training_spawn_areas(s: Suite) -> list[dict]:
    """The suite's training cells, as named areas. Test cells are swept, never listed."""
    areas: list[dict] = []
    seen: set[tuple[float, float]] = set()
    for c in enumerate_conditions(s, object_keys_for(s.name)[:1], splits=('train',)):
        (x, y) = s.object_xy(c.params)
        k = (round(x, 4), round(y, 4))
        if k in seen:
            continue
        seen.add(k)
        areas.append({'id': f'train_area_{len(areas) + 1}', 'pos': [k[0], k[1], TABLE_TOP_Z]})
    return areas

def _cameras() -> list[dict]:
    """Both streams, for every suite."""
    return [dict(HEAD_CAM), dict(WRIST_CAM)]

def build_layout(s: Suite) -> dict:
    """One layout dict for one suite."""
    required = object_keys_for(s.name)
    pool = list(dict.fromkeys([*required, *OBJECT_POOL]))
    session = {'camera': {'distance_m': 1.8, 'orbit_center': [0.45, 0.0, TABLE_TOP_Z], 'pitch_deg': 24.0, 'preset': 'isometric', 'yaw_deg': -30.0}, 'object_pool': [_pool_entry(a) for a in pool], 'object_spawn_areas': _training_spawn_areas(s), 'props': [], 'robot': {'name': 'franka_panda', 'pos': ROBOT_POS, 'rpy': ROBOT_RPY, 'heading': 0.0}, 'scene': {'auto_pos_z': TABLE_TOP_Z, 'file_name': 'escape_ood_table.usd', 'name': 'escape_ood_table', 'path': '', 'pos': [0.0, 0.0, TABLE_TOP_Z], 'rot': [0.7071, 0.7071, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0]}, 'custom_cams': _cameras(), 'workspace': {'x_min': WORKSPACE.x_min, 'x_max': WORKSPACE.x_max, 'y_min': WORKSPACE.y_min, 'y_max': WORKSPACE.y_max, 'table_top_z': TABLE_TOP_Z}, 'suite': {'name': s.name, 'unit': s.unit, 'success_kind': s.success_kind}}
    return {'name': f'escape_ood_{s.name}', 'saved_at': 0.0, 'schema': SCHEMA, 'generated_by': 'escape_ood/suites/layout_gen.py', 'session': session}

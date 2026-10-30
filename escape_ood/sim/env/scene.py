# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import json
import os
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from escape_ood.collect.grasp import mat_to_quat
from escape_ood.suites.spec import object_start_xy, object_start_yaw, parking_slot, suite as get_suite
from escape_ood.suites.layout_gen import FIG_CAM, TABLE_TOP_Z, build_layout
_REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE_USD_DIR = _REPO_ROOT / 'data' / 'output' / 'assets' / 'scene'
OBJ_USD_DIR = _REPO_ROOT / 'data' / 'output' / 'assets' / 'escape_ood_assets'
YUP_TO_ZUP = (0.70710678, 0.70710678, 0.0, 0.0)
HOME_JOINT_POS = [-0.0157, -0.2754, 0.0159, -2.5895, 0.0059, 2.3142, 0.7809]
SUN_ROT = (0.887155, 0.0, 0.461472, 0.0)
HOME_JITTER_RAD = 0.02

def jittered_home(seed: int, scale: float=HOME_JITTER_RAD, base=None) -> list[float]:
    """A slightly perturbed home pose, deterministic in ``seed``.

    Every part of this benchmark is otherwise deterministic -- fixed spawn pose, zero initial
    velocity, a scripted expert -- so two demonstrations of one condition would be identical
    frame for frame. Recording five copies of one trajectory inflates the episode count without
    adding information.

    The arm's *starting configuration* is the one quantity no suite sweeps: every suite varies
    where the object (or target) sits. Perturbing the home pose therefore varies the trajectory
    without touching any condition's parameter vector, which spawn-pose jitter could not do --
    the spawn pose *is* the swept parameter.

    Seeded from the condition key and repeat index so a rerun reproduces the dataset exactly.
    """
    import random
    rng = random.Random(seed)
    return [j + rng.gauss(0.0, scale) for j in (HOME_JOINT_POS if base is None else base)]

def suite_home(suite_name: str) -> list[float]:
    """Nominal arm start shared by collection and evaluation."""
    return list(HOME_JOINT_POS)


def episode_seed(condition_key: str, repeat: int) -> int:
    """Stable seed for one (condition, repeat). ``hash()`` is salted per process, so not that."""
    import hashlib
    h = hashlib.sha256(f'{condition_key}#{repeat}'.encode()).digest()
    return int.from_bytes(h[:8], 'big')

SPAWN_CLEARANCE_M = 0.002

def thickness(usd_stem: str, directory: Path) -> float:
    """World-frame Z extent from the sidecar.

    ``world_size_m`` exists only on the scene props. The graspable assets carry
    ``bbox_size_m``, whose index 1 is the authored Y -- which becomes world Z after the spawn
    rotation.
    """
    meta = json.loads((directory / f'{usd_stem}.meta.json').read_text())
    if 'world_size_m' in meta:
        return float(meta['world_size_m'][2])
    return float(meta['bbox_size_m'][1])

def look_at_quat(eye, target) -> tuple[float, float, float, float]:
    """Quaternion orienting a camera at ``target``, in CameraCfg's "world" convention.

    Baked into the spawn config rather than applied at runtime: a Camera sensor is not
    initialised during scene setup, so ``set_world_poses_from_view`` there silently leaves the
    camera at the origin. Measured -- both cameras reported world pos (0,0,0) and rendered
    identical frames of the ground plane.
    """
    import math
    f = [t - e for (t, e) in zip(target, eye)]
    n = math.sqrt(sum((c * c for c in f))) or 1.0
    f = [c / n for c in f]
    up = [0.0, 0.0, 1.0]
    d = sum((u * c for (u, c) in zip(up, f)))
    z = [u - d * c for (u, c) in zip(up, f)]
    nz = math.sqrt(sum((c * c for c in z)))
    if nz < 1e-08:
        (z, nz) = ([0.0, 1.0, 0.0], 1.0)
    z = [c / nz for c in z]
    y = [z[1] * f[2] - z[2] * f[1], z[2] * f[0] - z[0] * f[2], z[0] * f[1] - z[1] * f[0]]
    return tuple(mat_to_quat([[f[0], y[0], z[0]], [f[1], y[1], z[1]], [f[2], y[2], z[2]]]))

def build_scene(suite_name: str, *, device: str='cuda:0', dt: float=1.0 / 120.0, object_keys: list[str] | None=None, with_cameras: bool=False, bright: bool=False):
    """(scene, sim, info) for one suite.

    ``object_keys`` spawns those graspable objects as rigid bodies; every episode parks them
    off-workspace and teleports the one it needs, which is how a condition's parameter vector
    becomes a scene without rebuilding it. Defaults to the layout's pool, which is every asset
    any suite uses, so one launch can serve every suite.
    """
    layout = build_layout(get_suite(suite_name))['session']
    sim = SimulationContext(SimulationCfg(dt=dt, device=device))
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    cfg.ground = AssetBaseCfg(prim_path='/World/GroundPlane', spawn=sim_utils.GroundPlaneCfg())
    cfg.dome = AssetBaseCfg(prim_path='/World/DomeLight', spawn=sim_utils.DomeLightCfg(intensity=900.0 if bright else 400.0, color=(0.85, 0.86, 0.9)))
    cfg.sun = AssetBaseCfg(prim_path='/World/Sun', spawn=sim_utils.DistantLightCfg(intensity=700.0, angle=1.5), init_state=AssetBaseCfg.InitialStateCfg(rot=SUN_ROT))
    t_table = thickness('escape_ood_table', SCENE_USD_DIR)
    tmeta = json.loads((SCENE_USD_DIR / 'escape_ood_table.meta.json').read_text())
    (tcx, tcy) = tmeta['world_centre_xy']
    cfg.table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table', spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD_DIR / 'escape_ood_table.usd')), init_state=AssetBaseCfg.InitialStateCfg(pos=(tcx, tcy, TABLE_TOP_Z - 0.5 * t_table), rot=YUP_TO_ZUP))
    cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    cfg.robot.init_state = cfg.robot.init_state.replace(pos=(0.0, 0.0, TABLE_TOP_Z))
    keys = object_keys if object_keys is not None else [o['asset_name'] for o in layout['object_pool']]
    half_h: dict[str, float] = {}
    entity_name: dict[str, str] = {}
    for (i, key) in enumerate(keys):
        h = thickness(key, OBJ_USD_DIR)
        half_h[key] = 0.5 * h
        entity_name[key] = f'obj_{i}'
        setattr(cfg, f'obj_{i}', RigidObjectCfg(prim_path=f'{{ENV_REGEX_NS}}/Obj_{key}', spawn=sim_utils.UsdFileCfg(usd_path=str(OBJ_USD_DIR / f'{key}.usd')), init_state=RigidObjectCfg.InitialStateCfg(pos=_parking(i), rot=YUP_TO_ZUP)))
    if with_cameras:
        for cam in layout['custom_cams']:
            _add_camera(cfg, cam)
        if os.environ.get('ESCAPE_FIG_CAM'):
            _add_camera(cfg, FIG_CAM)
    scene = InteractiveScene(cfg)
    sim.reset()
    return (scene, sim, {'built': {}, 'layout': layout, 'object_keys': keys, 'object_half_height': half_h, 'instance_name': entity_name, 'table_top_z': TABLE_TOP_Z})
SCENE_REALISED_PARAMS: dict[str, set[str]] = {}

def assert_condition_realised(suite_name: str, params: dict, built: dict) -> None:
    pass
    required = SCENE_REALISED_PARAMS.get(suite_name, set())
    problems = []
    for name in sorted(required):
        if name not in params:
            problems.append(f"condition has no '{name}'")
            continue
        want = float(params[name])
        got = built.get(name)
        if got is None:
            problems.append(f"'{name}' is not realised in the scene at all")
        elif abs(float(got) - want) > 1e-06:
            problems.append(f"'{name}': condition says {want}, scene built {got}")
    if problems:
        raise ValueError(f"scene does not realise condition for suite '{suite_name}': " + '; '.join(problems) + '. A prop cannot change after the stage is built, so run one launch per distinct value rather than iterating conditions against a fixed scene.')

def _parking(i: int) -> tuple[float, float, float]:
    """Off-workspace parking slot. Defined by the spec -- see ``spec.parking_slot``."""
    return parking_slot(i)

def _add_camera(cfg, cam: dict) -> None:
    data_types = ['rgb', 'distance_to_image_plane']
    if cam['name'] == 'cam_wrist':
        setattr(cfg, 'cam_wrist', CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{cam['parent_body']}/WristCam", update_period=0.0, width=int(cam['width']), height=int(cam['height']), data_types=data_types, spawn=sim_utils.PinholeCameraCfg(focal_length=float(cam['focal_mm']), focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(float(cam.get('clip_near', 0.1)), float(cam.get('clip_far', 2.0)))), offset=CameraCfg.OffsetCfg(pos=tuple(cam['offset_pos']), rot=tuple(cam['offset_rot']), convention='ros')))
        return
    setattr(cfg, cam['name'], CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/{cam['name']}", update_period=0.0, width=int(cam['width']), height=int(cam['height']), data_types=data_types, spawn=sim_utils.PinholeCameraCfg(focal_length=float(cam['focal_mm']), clipping_range=(0.05, 20.0)), offset=CameraCfg.OffsetCfg(pos=tuple(cam['pos']), rot=look_at_quat(cam['pos'], cam['look_at']), convention='world')))

def _quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    (aw, ax, ay, az) = a
    (bw, bx, by, bz) = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)

def spawn_rot(yaw_deg: float=0.0):
    """Spawn orientation for a prop: the Y-up correction, then a yaw about world Z.

    Constant for the cube. It becomes load-bearing with the YCB mug, whose handle makes the
    grasp depend on how the object is turned -- ``object_start_yaw`` is recorded per condition
    for exactly that reason.
    """
    import math
    if not yaw_deg:
        return YUP_TO_ZUP
    h = math.radians(float(yaw_deg)) * 0.5
    return _quat_mul((math.cos(h), 0.0, 0.0, math.sin(h)), YUP_TO_ZUP)

def park_objects(scene, sim, info: dict, keys) -> None:
    """Move objects off the table, to the parking slots they spawned in.

    The scene spawns every asset any suite uses so one launch can serve them all, and a
    condition places only what it needs. Without parking, a previous suite's object would still
    be sitting on the table -- an observation no condition describes, and a contamination that
    looks like nothing at all in the recorded action.
    """
    import torch
    for key in keys:
        inst = info['instance_name'].get(key)
        if inst is None:
            continue
        obj = scene[inst]
        i = int(str(inst).rsplit('_', 1)[-1])
        root = obj.data.default_root_state.clone()
        root[:, 0:3] = torch.tensor([list(_parking(i))], device=obj.device)
        root[:, 3:7] = torch.tensor([list(YUP_TO_ZUP)], device=obj.device)
        root[:, 7:] = 0.0
        obj.write_root_state_to_sim(root)
        obj.reset()

def reset_condition_scene(scene, sim, info: dict, condition) -> tuple[list[float], list[str]]:
    pass
    used = [condition.object_key]
    starts = [object_start_xy(condition)]
    yaws = [object_start_yaw(condition)]
    park_objects(scene, sim, info, [k for k in info['object_keys'] if k not in used])
    rest = [place_object(scene, sim, info, k, xy, yaw_deg=y) for (k, xy, y) in zip(used, starts, yaws)]
    return (rest, used)

def place_object(scene, sim, info: dict, key: str, xy, *, yaw_deg: float=0.0, settle_steps: int=45) -> float:
    """Teleport one object onto the table at ``xy`` and settle. Returns its resting world z.

    Settling matters: an object dropped and read immediately reports a height it has not
    reached yet, and the lift check is relative to that baseline.
    """
    import torch
    obj = scene[info['instance_name'][key]]
    z = info['table_top_z'] + info['object_half_height'][key] + SPAWN_CLEARANCE_M
    root = obj.data.default_root_state.clone()
    root[:, 0:3] = torch.tensor([[float(xy[0]), float(xy[1]), z]], device=obj.device)
    root[:, 3:7] = torch.tensor([list(spawn_rot(yaw_deg))], device=obj.device)
    root[:, 7:] = 0.0
    obj.write_root_state_to_sim(root)
    obj.reset()
    robot = scene['robot']
    robot.set_joint_position_target(robot.data.joint_pos.clone())
    for _ in range(settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    return float(obj.data.root_pos_w[0, 2].detach().cpu().item())

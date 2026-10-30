# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument('--suite', required=True)
_parser.add_argument('--split', default='interior', choices=('train', 'interior', 'test', 'all'), help='interior is the ID control; test is the sweep')
_parser.add_argument('--bands', default='', help='comma-separated band labels to restrict the test split to (R1..A3)')
_parser.add_argument('--policy', default='server', choices=('server', 'random', 'hold', 'expert', 'replay'))
_parser.add_argument('--replay-collect', default='data/output/collect', help='archive directory for --policy replay')
_parser.add_argument('--replay-space', default='joint', choices=('joint', 'eef_abs', 'eef_abs_rotvec', 'joint_chunk_absgrip', 'eef_chunk'), help='which recorded command --policy replay feeds back: joint `actions`, `actions_eef` verbatim, or `actions_eef` converted to eef_abs_rotvec. The non-joint choices prove a converted action space decodes back to the demonstration before training on it')
_parser.add_argument('--host', default='localhost')
_parser.add_argument('--port', type=int, default=5555)
_parser.add_argument('--episodes', type=int, default=1, help='rollouts per cell; >1 gives a per-cell rate')
_parser.add_argument('--max-steps', type=int, default=600, help='cap per rollout. The expert needs ~170-330 control steps.')
_parser.add_argument('--max-conditions', type=int, default=0, help='0 = every cell')
_parser.add_argument('--seed', type=int, default=0)
_parser.add_argument('--speak', default=None, help="raw instruction text to give the policy instead of the cell's own (the language-swap control); scoring stays on the cell")
_parser.add_argument('--hold-checks', type=int, default=3, help='consecutive satisfied checks before a rollout counts as a success; also what makes early exit safe')
_parser.add_argument('--check-every', type=int, default=15, help='control steps between success checks')
_parser.add_argument('--save-video', default='none', choices=('none', 'fail', 'success', 'all'))
_parser.add_argument('--video-fps', type=int, default=30)
_parser.add_argument('--tag', default=None, help='result-file suffix (default: the policy kind)')
_parser.add_argument('--out', default='data/output/eval')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
_app = AppLauncher(args_cli).app
import numpy as np
from escape_ood.suites.layout_gen import OBJECT_CENTRE_Z
from escape_ood.collect import success as S
from escape_ood.collect.constants import DISCARD_HOLD_STEPS
from escape_ood.collect.context import ObjectDropped, PhaseTimeout, TaskContext, TaskFailed
from escape_ood.collect.grasp import quat_mul, rotvec_to_quat
from escape_ood.splits.eef_delta import ROT_REF_WXYZ
from escape_ood.collect.tasks import task_for
from escape_ood.metrics.extent import format_summary, summarise
from escape_ood.sim.env.isaac_backend import IsaacMotionBackend
from escape_ood.sim.env.scene import assert_condition_realised, build_scene, episode_seed, jittered_home, reset_condition_scene, suite_home
from escape_ood.suites.spec import ENVELOPE, SUITES, band_label, enumerate_conditions, lang_obs_key, object_keys_for, object_start_xy, target_xys
JOINT_KEYS = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1']
EEF_DELTA_KEYS = ['eef_dx', 'eef_dy', 'eef_dz', 'eef_drx', 'eef_dry', 'eef_drz', 'gripper']
EEF_ABS_KEYS = ['eef_x', 'eef_y', 'eef_z', 'eef_qw', 'eef_qx', 'eef_qy', 'eef_qz', 'gripper']
EEF_ABS_ROTVEC_KEYS = ['eef_x', 'eef_y', 'eef_z', 'eef_rx', 'eef_ry', 'eef_rz', 'gripper']
JOINT_DELTA_ABSGRIP_KEYS = [f'd_{k}' for k in JOINT_KEYS[:7]] + [JOINT_KEYS[7]]
JOINT_CHUNK_ABSGRIP_KEYS = [f'c_{k}' for k in JOINT_KEYS[:7]] + [JOINT_KEYS[7]]
EEF_CHUNK_KEYS = ['eef_cx', 'eef_cy', 'eef_cz', 'eef_crx', 'eef_cry', 'eef_crz', 'gripper']
CAM_TO_MODALITY = {'cam_head': 'image', 'cam_wrist': 'wrist_image'}
LANGUAGE_KEY = lang_obs_key('task')

def reach_target_xy(cond) -> tuple[float, float]:
    """The cell the policy must reach: the destination for a place suite, else the object."""
    t = target_xys(cond)
    return tuple(t[0]) if t else tuple(object_start_xy(cond))

def grab_frames(scene, cams) -> dict:
    out = {}
    for cam in cams:
        rgb = scene[cam].data.output.get('rgb')
        if rgb is None:
            continue
        img = rgb[0, ..., :3].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        out[cam] = img
        if cam == 'cam_fig':
            d = scene[cam].data.output.get('distance_to_image_plane')
            if d is not None:
                out['cam_fig_depth'] = d[0, ..., 0].detach().cpu().numpy().astype(np.float16)
    return out

def _fmt_param(v) -> str:
    if isinstance(v, float):
        return f'{v:g}'.replace('-', 'm').replace('.', 'p')
    return str(v).replace('-', 'm').replace('.', 'p')

def write_video(frames: list, cond, ep: int, ok: bool, why: str) -> None:
    """Head and wrist side by side, captioned with the cell and the outcome."""
    import cv2
    out = Path(args_cli.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    vdir = out / 'video'
    vdir.mkdir(parents=True, exist_ok=True)
    head_key = 'cam_head' if 'cam_head' in frames[0] else next(iter(frames[0]))
    (h, w) = frames[0][head_key].shape[:2]
    has_wrist = 'cam_wrist' in frames[0]
    pane_w = w + (w // 2 if has_wrist else 0)
    canvas_h = h + 30
    params = '_'.join((f'{k}{_fmt_param(v)}' for (k, v) in sorted(cond.params.items())))
    tag = f"{cond.suite}_{('ok' if ok else 'FAIL')}_d{cond.sweep_coord:g}_{params}_ep{ep}"
    path = vdir / f'{tag}.mp4'
    label = f"{cond.suite}  {band_label(cond.params) or cond.split}  d={cond.sweep_coord:g}cm  {('SUCCESS' if ok else 'FAIL')}"
    sub = why[:70] if not ok else cond.instruction[:70]

    def composite(i, fr):
        canvas = np.zeros((canvas_h, pane_w, 3), np.uint8)
        canvas[:h, :w] = cv2.cvtColor(fr[head_key], cv2.COLOR_RGB2BGR)
        if has_wrist:
            wr = cv2.resize(cv2.cvtColor(fr['cam_wrist'], cv2.COLOR_RGB2BGR), (w // 2, h))
            canvas[:h, w:] = wr
        colour = (110, 200, 130) if ok else (90, 90, 235)
        cv2.putText(canvas, label, (8, h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        cv2.putText(canvas, f'step {i + 1}/{len(frames)}  {sub}', (8, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
        return canvas
    import shutil
    import subprocess
    if shutil.which('ffmpeg'):
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{pane_w}x{canvas_h}', '-r', str(args_cli.video_fps), '-i', '-', '-an', '-c:v', 'libx264', '-tag:v', 'avc1', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '23', '-movflags', '+faststart', str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for (i, fr) in enumerate(frames):
            proc.stdin.write(composite(i, fr).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            print(f'[EVAL] ffmpeg failed for {tag}', flush=True)
            return
    else:
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), float(args_cli.video_fps), (pane_w, canvas_h))
        for (i, fr) in enumerate(frames):
            vw.write(composite(i, fr))
        vw.release()
        print('[EVAL] WARNING no ffmpeg; wrote mp4v, which browsers cannot play', flush=True)
    print(f'[EVAL] video -> {path}', flush=True)
    if 'cam_fig' in frames[0]:
        fdir = vdir / f'{tag}_fig'
        fdir.mkdir(exist_ok=True)
        stride = int(os.environ.get('ESCAPE_FIG_STRIDE', '5'))
        for (i, fr) in enumerate(frames):
            if i % stride == 0 or i == len(frames) - 1:
                cv2.imwrite(str(fdir / f'{i:04d}.png'), cv2.cvtColor(fr['cam_fig'], cv2.COLOR_RGB2BGR))
                if 'cam_fig_depth' in fr:
                    np.save(fdir / f'{i:04d}_depth.npy', fr['cam_fig_depth'])
        print(f'[EVAL] figure frames -> {fdir}', flush=True)
EEF_STATE_KEYS = ['eef_x', 'eef_y', 'eef_z', 'eef_qw', 'eef_qx', 'eef_qy', 'eef_qz']

def build_observation(scene, backend, cams, instruction: str, cond=None) -> dict:
    """The (B=1, T=1, ...) observation dict the policy server expects."""
    joints = backend.getj() + [backend.gripper_pos()]
    state = {k: np.array([[[joints[i]]]], dtype=np.float32) for (i, k) in enumerate(JOINT_KEYS)}
    video = {}
    for cam in cams:
        key = CAM_TO_MODALITY.get(cam)
        rgb = scene[cam].data.output.get('rgb') if key else None
        if rgb is None:
            continue
        img = rgb[0, ..., :3].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        video[key] = img[None, None, ...]
    return {'video': video, 'state': state, 'language': {LANGUAGE_KEY: [[instruction]]}}

def action_space_of(chunk: dict) -> str:
    """Which action space a reply speaks, from its key set."""
    keys = set(chunk)
    if all((k in keys for k in JOINT_DELTA_ABSGRIP_KEYS)):
        return 'joint_delta_absgrip'
    if all((k in keys for k in JOINT_CHUNK_ABSGRIP_KEYS)):
        return 'joint_chunk_absgrip'
    if all((k in keys for k in EEF_CHUNK_KEYS)):
        return 'eef_chunk'
    if all((k in keys for k in JOINT_KEYS)):
        return 'joint'
    if all((k in keys for k in EEF_ABS_KEYS)):
        return 'eef_abs'
    if all((k in keys for k in EEF_ABS_ROTVEC_KEYS)):
        return 'eef_abs_rotvec'
    if all((k in keys for k in EEF_DELTA_KEYS)):
        return 'eef_delta'
    raise RuntimeError(f'policy reply matches no known action space: {sorted(keys)[:10]}')

def _series(chunk: dict, key: str) -> np.ndarray:
    v = np.asarray(chunk[key])
    while v.ndim > 2:
        v = v[0]
    return v
KEYS_FOR_SPACE = {'joint': JOINT_KEYS, 'joint_delta_absgrip': JOINT_DELTA_ABSGRIP_KEYS, 'joint_chunk_absgrip': JOINT_CHUNK_ABSGRIP_KEYS, 'eef_abs': EEF_ABS_KEYS, 'eef_abs_rotvec': EEF_ABS_ROTVEC_KEYS, 'eef_delta': EEF_DELTA_KEYS, 'eef_chunk': EEF_CHUNK_KEYS}

def keys_for(space: str) -> list[str]:
    try:
        return KEYS_FOR_SPACE[space]
    except KeyError:
        raise RuntimeError(f'no reply keys registered for action space {space!r}; known: {sorted(KEYS_FOR_SPACE)}') from None

def chunk_len(chunk: dict, space: str) -> int:
    return int(_series(chunk, keys_for(space)[0]).shape[0])

def decode_step(chunk: dict, space: str, t: int) -> list[float]:
    keys = keys_for(space)
    out = []
    for k in keys:
        v = _series(chunk, k)
        out.append(float(v[t][0] if v.ndim == 2 else v[t]))
    return out

def apply_action(backend, space: str, vals: list[float], anchor=None) -> float:
    """Send one decoded step to the backend. Returns the commanded gripper width.

    `anchor` = (joints, (pos, quat)) measured when the current chunk was queried; only the chunk-anchored
    spaces read it -- every other space is absolute or anchored to the state at THIS step.
"""

    def _j(targets, grip):
        backend.movej(targets, grip)

    def _p(pos, quat, grip):
        backend.movep(pos, quat, grip)
    if space == 'joint_chunk_absgrip':
        q0 = anchor[0]
        _j([q0[i] + vals[i] for i in range(7)], vals[7])
        return vals[7]
    if space == 'eef_chunk':
        (pos0, quat0) = anchor[1]
        q = quat_mul(rotvec_to_quat(vals[3:6]), tuple(quat0))
        _p([pos0[i] + vals[i] for i in range(3)], list(q), vals[6])
        return vals[6]
    if space == 'joint':
        _j(vals[:7], vals[7])
        return vals[7]
    if space == 'joint_delta_absgrip':
        q = backend.getj()
        _j([q[i] + vals[i] for i in range(7)], vals[7])
        return vals[7]
    if space == 'eef_abs':
        _p(list(vals[:3]), list(vals[3:7]), vals[7])
        return vals[7]
    if space == 'eef_abs_rotvec':
        q = quat_mul(rotvec_to_quat(vals[3:6]), ROT_REF_WXYZ)
        _p(list(vals[:3]), list(q), vals[6])
        return vals[6]
    (pos, quat) = backend.getp()
    target = [pos[i] + vals[i] for i in range(3)]
    q = quat_mul(rotvec_to_quat(vals[3:6]), tuple(quat))
    _p(target, list(q), vals[6])
    return vals[6]

class ZmqPolicyClient:
    pass

    def __init__(self, host: str, port: int, timeout_ms: int=30000):
        import zmq
        self._zmq = zmq
        self.ctx = zmq.Context()
        (self.host, self.port, self.timeout_ms) = (host, port, timeout_ms)
        self._connect()

    def _connect(self) -> None:
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(f'tcp://{self.host}:{self.port}')

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            import io
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {'__ndarray_class__': True, 'as_npy': buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and '__ndarray_class__' in obj:
            import io
            return np.load(io.BytesIO(obj['as_npy']), allow_pickle=False)
        return obj

    def _call(self, endpoint: str, data=None, requires_input: bool=True):
        import msgpack
        req = {'endpoint': endpoint}
        if requires_input:
            req['data'] = data
        try:
            self.sock.send(msgpack.packb(req, default=self._encode))
            msg = self.sock.recv()
        except self._zmq.error.Again:
            self._connect()
            raise
        if msg == b'ERROR':
            raise RuntimeError('policy server returned ERROR')
        resp = msgpack.unpackb(msg, object_hook=self._decode)
        if isinstance(resp, dict) and 'error' in resp:
            raise RuntimeError(f"policy server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call('ping', requires_input=False)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        """Tell the server a new rollout is starting. Best-effort.

        Policies that condition on a short observation history (Diffusion Policy uses the last
        two frames) must not carry frames across an episode boundary, and the protocol has no
        other signal for it. Servers that keep no state answer or ignore it harmlessly, so this
        never fails a rollout: a server predating the endpoint replies with an error, which is
        exactly the stateless case.
        """
        try:
            self._call('reset', requires_input=False)
        except Exception:
            pass

    def get_action(self, observation: dict) -> dict:
        resp = self._call('get_action', {'observation': observation, 'options': None})
        if isinstance(resp, (list, tuple)):
            resp = resp[0]
        if isinstance(resp, dict) and 'action' in resp and (not any((ks[0] in resp for ks in KEYS_FOR_SPACE.values()))):
            resp = resp['action']
        return resp

class RandomPolicy:
    """Uniform joint targets inside the soft limits. The protocol's random floor."""

    def __init__(self, backend, seed: int):
        self.rng = np.random.default_rng(seed)
        lo = backend.robot.data.soft_joint_pos_limits[0, backend._arm.joint_ids, 0]
        hi = backend.robot.data.soft_joint_pos_limits[0, backend._arm.joint_ids, 1]
        (self.lo, self.hi) = (lo.cpu().numpy(), hi.cpu().numpy())

    def get_action(self, _obs) -> dict:
        vals = list(self.rng.uniform(self.lo, self.hi)) + [float(self.rng.uniform(0.0, 0.04))]
        return {k: np.array([[v]], dtype=np.float32) for (k, v) in zip(JOINT_KEYS, vals)}

class HoldPolicy:
    """Hold the start pose. The no-op floor."""

    def __init__(self, home: list[float]):
        self.vals = list(home) + [0.04]

    def get_action(self, _obs) -> dict:
        return {k: np.array([[v]], dtype=np.float32) for (k, v) in zip(JOINT_KEYS, self.vals)}

class ReplayPolicy:
    """Feed a recorded episode's own actions back through the evaluation harness.

    The sharpest control there is, and the one that separates "the policy is bad" from "the
    action representation or the replay path is broken". The demonstration succeeded when it was
    collected; the evaluator resets to the same cell with the same ``episode_seed``, so replaying
    ``actions`` must reproduce it. If THIS fails, no policy trained on those actions can succeed
    and the number to fix is not a model's.

    Falls back to holding the last recorded action once the episode runs out, because the
    evaluator's step budget is longer than a demonstration.
    """

    def __init__(self, collect_dir, suite_name: str, space: str='joint'):
        import glob
        import h5py
        self._h5py = h5py
        self.index: dict[tuple, tuple[str, str]] = {}
        for path in sorted(glob.glob(str(Path(collect_dir) / f'{suite_name}*.hdf5'))):
            with h5py.File(path, 'r') as f:
                for name in f['data'].keys():
                    a = f['data'][name].attrs
                    key = (round(float(a['param_r']), 6), round(float(a['param_theta_deg']), 6), str(a['object_key']), int(a['attempt']))
                    self.index[key] = (path, name)
        self.space = space
        self.keys = KEYS_FOR_SPACE[space]
        self.actions = None
        self.anchors = None
        self.episode_seed = None
        self.t = 0

    def begin_episode(self, cond, ep: int) -> str:
        key = (round(float(cond.params['r']), 6), round(float(cond.params['theta_deg']), 6), str(cond.object_key), int(ep))
        hit = self.index.get(key)
        (self.actions, self.episode_seed, self.t) = (None, None, 0)
        if hit is None:
            return f'no recorded episode for {key}'
        (path, name) = hit
        with self._h5py.File(path, 'r') as f:
            g = f['data'][name]
            self.episode_seed = int(g.attrs['episode_seed'])
            if self.space == 'joint':
                self.actions = np.asarray(g['actions'][:], dtype=np.float32)
            elif self.space == 'eef_abs':
                self.actions = np.asarray(g['actions_eef'][:], dtype=np.float32)
            elif self.space == 'eef_abs_rotvec':
                from escape_ood.splits.eef_delta import eef_abs_rotvec_actions
                self.actions = eef_abs_rotvec_actions(np.asarray(g['actions_eef'][:], dtype=np.float64))
            elif self.space == 'joint_chunk_absgrip':
                self.actions = np.asarray(g['actions'][:], dtype=np.float64)
                self.anchors = np.asarray(g['obs']['joint_pos'][:], dtype=np.float64)
            else:
                self.actions = np.asarray(g['actions_eef'][:], dtype=np.float64)
                self.anchors = np.asarray(g['obs']['eef_pose'][:], dtype=np.float64)
        return ''

    def reset(self) -> None:
        self.t = 0

    def get_action(self, _obs) -> dict:
        if self.actions is None:
            raise RuntimeError('ReplayPolicy.begin_episode was not called for this episode')
        if self.space in ('joint_chunk_absgrip', 'eef_chunk'):
            from escape_ood.policy.act.chunk_anchor import anchored
            n = len(self.actions)
            rows = [min(self.t + k, n - 1) for k in range(16)]
            chunk = anchored(self.space, self.actions[rows], self.anchors[min(self.t, n - 1)])
            self.t += 16
            return {k: chunk[:, j:j + 1].astype(np.float32) for (j, k) in enumerate(self.keys)}
        i = min(self.t, len(self.actions) - 1)
        self.t += 1
        return {k: np.array([[float(self.actions[i, j])]], dtype=np.float32) for (j, k) in enumerate(self.keys)}

def pick_conditions(suite_name: str, object_keys: list[str]) -> list:
    splits = ('train', 'interior', 'test') if args_cli.split == 'all' else (args_cli.split,)
    conds = enumerate_conditions(SUITES[suite_name], object_keys, splits=splits)
    bands = {b.strip() for b in args_cli.bands.split(',') if b.strip()}
    if bands:
        conds = [c for c in conds if band_label(c.params) in bands]
    cells = [tuple((float(v) for v in c.split(','))) for c in os.environ.get('ESCAPE_EVAL_CELLS', '').split(';') if c]
    if cells:
        conds = [c for c in conds if (round(c.params['r'], 4), round(c.params['theta_deg'], 4)) in {(round(r, 4), round(t, 4)) for (r, t) in cells}]
    conds.sort(key=lambda c: (c.sweep_coord, c.key()))
    if args_cli.max_conditions:
        step = max(1, len(conds) // args_cli.max_conditions)
        conds = conds[::step][:args_cli.max_conditions]
    return conds

def evaluate(scene, sim, info, backend, policy, cond, cams, ep: int, state: dict) -> dict:
    """One rollout. Returns the record; never raises for a policy failure."""
    suite_name = cond.suite
    assert_condition_realised(suite_name, cond.params, info['built'])
    seed = episode_seed(cond.key(), ep)
    if hasattr(policy, 'begin_episode'):
        missing = policy.begin_episode(cond, ep)
        if missing:
            return {'key': cond.key(), 'suite': suite_name, 'split': cond.split, 'band': band_label(cond.params), 'sweep_coord': cond.sweep_coord, 'params': dict(cond.params), 'episode': ep, 'success': False, 'reason': missing, 'steps': 0, 'seconds': 0.0, 'skipped': True}
        if getattr(policy, 'episode_seed', None) is not None:
            seed = int(policy.episode_seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    backend.reset_to(jittered_home(seed, base=suite_home(suite_name)))
    (rest, used_keys) = reset_condition_scene(scene, sim, info, cond)
    obj_names = [info['instance_name'][k] for k in used_keys]
    (_hold_pos, _hold_quat) = backend.getp()
    for _ in range(DISCARD_HOLD_STEPS):
        backend.movep(list(_hold_pos), list(_hold_quat), backend.gripper_pos())
        backend.step()
    if cams:
        for _ in range(8):
            sim.render()
            scene.update(0.0)
    kind = S.success_kind(suite_name)
    reach_xy = reach_target_xy(cond)
    frames: list = []
    want_video = args_cli.save_video != 'none'

    def outcome() -> tuple[bool, str]:
        if kind == 'lift':
            return S.lifted(object_z_world=backend.object_pos_world(obj_names[0])[2], rest_z_world=rest[0], object_xy=backend.object_pos_in_base(obj_names[0]), eef_xy=backend.getp()[0])
        return (False, f'no predicate for success kind {kind!r}')
    spoken = cond.instruction if args_cli.speak is None else args_cli.speak
    (horizon, chunk, steps, held) = (0, None, 0, 0)
    space = state.get('action_space')
    eef_path: list[list[float]] = []
    (ok, why) = (False, 'did not finish')
    t0 = time.time()
    if policy == 'expert':
        ctx = TaskContext(backend)
        ctx.object_name = obj_names[0]
        state.setdefault('action_space', 'expert')
        try:
            for _ in task_for(cond)(cond).run(ctx):
                if cams:
                    sim.render()
                    scene.update(0.0)
                backend.step()
                steps += 1
                eef_path.append([round(float(v), 4) for v in backend.getp()[0]])
                if want_video:
                    frames.append(grab_frames(scene, cams))
                if steps >= args_cli.max_steps:
                    raise TimeoutError(f'exceeded {args_cli.max_steps} control steps')
            (ok, why) = outcome()
        except (PhaseTimeout, ObjectDropped, TaskFailed, TimeoutError) as e:
            (ok, why) = (False, f'{type(e).__name__}: {e}')
    else:
        for steps in range(1, args_cli.max_steps + 1):
            if cams:
                sim.render()
                scene.update(0.0)
            if chunk is None or horizon >= chunk_len(chunk, space):
                chunk = policy.get_action(build_observation(scene, backend, cams, spoken, cond))
                if space is None:
                    space = state['action_space'] = action_space_of(chunk)
                    print(f'[EVAL] policy speaks {space} (chunk {chunk_len(chunk, space)})', flush=True)
                horizon = 0
                anchor = (list(backend.getj()), backend.getp())
            vals = decode_step(chunk, space, horizon)
            grip_cmd = apply_action(backend, space, vals, anchor)
            offset = horizon
            horizon += 1
            backend.step()
            eef_path.append([round(float(v), 4) for v in backend.getp()[0]])
            if want_video:
                frames.append(grab_frames(scene, cams))
            if steps % args_cli.check_every == 0:
                (good, reason) = outcome()
                held = held + 1 if good else 0
                (ok, why) = (good, reason)
                if held >= args_cli.hold_checks:
                    break
    if not ok and policy != 'expert':
        (ok, why) = outcome()
    if want_video and frames:
        res = 'success' if ok else 'fail'
        if args_cli.save_video in ('all', res):
            write_video(frames, cond, ep, ok, why)
    (fx, fy) = backend.object_pos_world(obj_names[0])[:2]
    return {'key': cond.key(), 'suite': suite_name, 'split': cond.split, 'band': band_label(cond.params), 'sweep_coord': float(cond.sweep_coord), 'params': cond.params, 'episode': ep, 'success': bool(ok), 'reason': why, 'steps': steps, 'seconds': round(time.time() - t0, 2), 'early_exit': steps < args_cli.max_steps, 'hold_checks_final': None if policy == 'expert' else held, 'success_sustained': None if policy == 'expert' else bool(held >= args_cli.hold_checks), 'final_object_xy': [[round(float(fx), 4), round(float(fy), 4)]], 'target_xy': [[round(float(reach_xy[0]), 4), round(float(reach_xy[1]), 4)]], 'final_target_dist': [round(float(((fx - reach_xy[0]) ** 2 + (fy - reach_xy[1]) ** 2) ** 0.5), 4)], 'spoken_instruction': spoken, 'own_instruction': cond.instruction, 'eef_path': eef_path}

def main() -> int:
    suite_name = args_cli.suite
    if suite_name not in SUITES:
        raise SystemExit(f'[EVAL] unknown suite {suite_name!r}; known {sorted(SUITES)}')
    object_keys = object_keys_for(suite_name)
    if args_cli.speak is not None:
        print(f'[EVAL] spoken text overridden: {args_cli.speak!r} (scoring stays on the cell)', flush=True)
    (scene, sim, info) = build_scene(suite_name, device=args_cli.device, object_keys=object_keys, with_cameras=True)
    cams = [k for k in scene.keys() if k.startswith('cam')]
    backend = IsaacMotionBackend(scene, sim)
    if args_cli.policy == 'expert':
        (policy, tag) = ('expert', 'expert')
    elif args_cli.policy == 'random':
        (policy, tag) = (RandomPolicy(backend, args_cli.seed), 'random')
    elif args_cli.policy == 'hold':
        (policy, tag) = (HoldPolicy(suite_home(args_cli.suite)), 'hold')
    elif args_cli.policy == 'replay':
        policy = ReplayPolicy(args_cli.replay_collect, suite_name, args_cli.replay_space)
        print(f'[EVAL] replay index: {len(policy.index)} recorded episodes, space {args_cli.replay_space}', flush=True)
        tag = 'replay' if args_cli.replay_space == 'joint' else f'replay_{args_cli.replay_space}'
    else:
        policy = ZmqPolicyClient(args_cli.host, args_cli.port)
        if not policy.ping():
            raise SystemExit(f'[EVAL] no policy server at {args_cli.host}:{args_cli.port}')
        tag = 'server'
    tag = args_cli.tag or tag
    conds = pick_conditions(suite_name, object_keys)
    if not conds:
        raise SystemExit(f'[EVAL] no cells for {suite_name} split={args_cli.split} bands={args_cli.bands!r}')
    print(f'[EVAL] suite={suite_name} split={args_cli.split} policy={tag} cells={len(conds)} episodes/cell={args_cli.episodes} cameras={cams}', flush=True)
    (records, state) = ([], {})
    for (i, cond) in enumerate(conds, 1):
        for ep in range(args_cli.episodes):
            records.append(evaluate(scene, sim, info, backend, policy, cond, cams, ep, state))
        n_ok = sum((1 for r in records[-args_cli.episodes:] if r['success']))
        if i % 5 == 0 or n_ok == 0:
            rate = sum((1 for r in records if r['success'])) / max(len(records), 1)
            print(f"[EVAL] {i}/{len(conds)} {cond.split:8s} {str(cond.params.get('r')):5s}/{str(cond.params.get('theta_deg')):5s} d={cond.sweep_coord:5.1f}cm  {n_ok}/{args_cli.episodes} ok  running SR={100 * rate:.1f}%", flush=True)
    by_coord: dict[float, list[bool]] = {}
    for r in records:
        by_coord.setdefault(round(r['sweep_coord'], 4), []).append(r['success'])
    curve = [{'sweep_coord': c, 'n': len(v), 'n_success': sum(v), 'success_rate': round(sum(v) / len(v), 4)} for (c, v) in sorted(by_coord.items())]
    (total, n_ok) = (len(records), sum((1 for r in records if r['success'])))
    result = {'suite': suite_name, 'split': args_cli.split, 'bands': args_cli.bands, 'policy': tag, 'action_space': state.get('action_space'), 'wrist_legacy': os.environ.get('ESCAPE_WRIST_LEGACY', '') == '1', 'cameras': list(cams), 'episodes_per_condition': args_cli.episodes, 'max_steps': args_cli.max_steps, 'envelope': {'r_min': ENVELOPE.r_min, 'r_max': ENVELOPE.r_max, 'theta_max_deg': ENVELOPE.theta_max_deg}, 'n': total, 'n_success': n_ok, 'curve': curve, 'records': records}
    print(f'\n[EVAL] ==== {suite_name} / {args_cli.split} / {tag} ====', flush=True)
    print(f'[EVAL] {n_ok}/{total} = {100 * n_ok / max(total, 1):.1f}%', flush=True)
    print(format_summary(summarise(result)), flush=True)
    if args_cli.split == 'interior':
        rate = n_ok / max(total, 1)
        if rate < 0.8:
            print(f'[EVAL] interior control: {100 * rate:.1f}% -- BELOW 80%: the extent numbers are not interpretable, this is not a weak result but no result', flush=True)
        else:
            print(f'[EVAL] interior control: {100 * rate:.1f}% -- clears the 80% floor; extents measured against it are readable', flush=True)
    out = Path(args_cli.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    name = f'{suite_name}_{args_cli.split}' + (f"_{args_cli.bands.replace(',', '+')}" if args_cli.bands else '') + f'_{tag}.json'
    (out / name).write_text(json.dumps(result, indent=2) + '\n')
    print(f'[EVAL] wrote {out / name}', flush=True)
    return 0

def _hard_exit(code: int) -> None:
    """Isaac's non-daemon threads outlive a bare raise."""
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(code)
if __name__ == '__main__':
    try:
        _hard_exit(main())
    except SystemExit as e:
        _hard_exit(int(e.code) if isinstance(e.code, int) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _hard_exit(1)

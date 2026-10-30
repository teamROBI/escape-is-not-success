# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import io
import os
import pickle
import sys
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / 'third_party' / 'act'
EEF_ABS_KEYS = ['eef_x', 'eef_y', 'eef_z', 'eef_qw', 'eef_qx', 'eef_qy', 'eef_qz', 'gripper']
JOINT_KEYS = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1']
CAM_TO_MODALITY = {'cam_head': 'image', 'cam_wrist': 'wrist_image'}

def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {'__ndarray_class__': True, 'as_npy': buf.getvalue()}
    return obj

def _decode(obj):
    if isinstance(obj, dict) and '__ndarray_class__' in obj:
        return np.load(io.BytesIO(obj['as_npy']), allow_pickle=False)
    return obj

def _resize_frames(frames: np.ndarray, hw) -> np.ndarray:
    pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_act import _resize_frames as _rf
    return _rf(frames, tuple(hw))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='policy_best.ckpt from run_act_axis.py')
    ap.add_argument('--stats', default=None, help="dataset_stats.pkl (defaults to the checkpoint's own directory)")
    ap.add_argument('--port', type=int, default=6000)
    ap.add_argument('--host', default='127.0.0.1', help='listen address (default: localhost only)')
    ap.add_argument('--chunk-size', type=int, default=16)
    ap.add_argument('--hidden-dim', type=int, default=512)
    ap.add_argument('--dim-feedforward', type=int, default=3200)
    ap.add_argument('--kl-weight', type=int, default=10)
    ap.add_argument('--action-space', default='joint_abs', choices=('joint_abs', 'joint_delta_absgrip', 'joint_chunk_absgrip', 'eef_abs', 'eef_abs_rotvec', 'eef_delta', 'eef_chunk'), help='MUST match convert_act.py --action-space for this checkpoint; it selects the reply key names the evaluator dispatches on')
    ap.add_argument('--proprio', default='joint', choices=('joint',), help='MUST match convert_act --proprio for this checkpoint')
    ap.add_argument('--no-proprio', action='store_true', help='MUST match convert_act --no-proprio for this checkpoint')
    ap.add_argument('--cameras', nargs='*', default=['cam_head'])
    ap.add_argument('--scale', type=int, default=2, help='MUST match convert_act.py --scale for this checkpoint')
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    if not ckpt.exists():
        print(f'[ERROR] no checkpoint at {ckpt}')
        return 1
    stats_p = Path(args.stats) if args.stats else ckpt.parent / 'dataset_stats.pkl'
    if not stats_p.is_absolute():
        stats_p = (REPO_ROOT / stats_p).resolve()
    if not stats_p.exists():
        print(f"[ERROR] no dataset_stats.pkl beside the checkpoint ({stats_p}). ACT's normalisation is fitted at train time and cannot be recovered without it.")
        return 1
    sys.path.insert(0, str(ACT_DIR / 'detr'))
    sys.path.insert(0, str(ACT_DIR))
    os.chdir(ACT_DIR)
    EEF_DELTA_KEYS = ['eef_dx', 'eef_dy', 'eef_dz', 'eef_drx', 'eef_dry', 'eef_drz', 'gripper']
    REPLY_KEYS = {'joint_abs': JOINT_KEYS, 'joint_delta_absgrip': [f'd_{k}' for k in JOINT_KEYS[:7]] + [JOINT_KEYS[7]], 'joint_chunk_absgrip': [f'c_{k}' for k in JOINT_KEYS[:7]] + [JOINT_KEYS[7]], 'eef_abs': EEF_ABS_KEYS, 'eef_abs_rotvec': ['eef_x', 'eef_y', 'eef_z', 'eef_rx', 'eef_ry', 'eef_rz', 'gripper'], 'eef_delta': EEF_DELTA_KEYS, 'eef_chunk': ['eef_cx', 'eef_cy', 'eef_cz', 'eef_crx', 'eef_cry', 'eef_crz', 'gripper']}[args.action_space]
    EEF_STATE_KEYS = ['eef_x', 'eef_y', 'eef_z', 'eef_qw', 'eef_qx', 'eef_qy', 'eef_qz']
    PROPRIO_STATE_KEYS = JOINT_KEYS
    os.environ.setdefault('ESCAPE_ACTION_DIM', str(len(REPLY_KEYS)))
    os.environ.setdefault('ESCAPE_STATE_DIM', str(len(PROPRIO_STATE_KEYS)))
    import torch
    import detr.models.detr_vae as dv
    if not hasattr(dv, 'ESCAPE_STATE_DIM'):
        print('[ERROR] third_party/act is unpatched (still 14-dim bimanual). Run: bash scripts/setup/patch_act_state_dim.sh')
        return 1
    if getattr(dv, 'ESCAPE_ACTION_DIM', dv.ESCAPE_STATE_DIM) != len(REPLY_KEYS):
        print(f"[ERROR] action dim {getattr(dv, 'ESCAPE_ACTION_DIM', None)} != {len(REPLY_KEYS)} for {args.action_space}")
        return 1
    from policy import ACTPolicy
    cams = list(args.cameras)
    cfg = {'lr': 1e-05, 'num_queries': args.chunk_size, 'kl_weight': args.kl_weight, 'hidden_dim': args.hidden_dim, 'dim_feedforward': args.dim_feedforward, 'lr_backbone': 1e-05, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8, 'camera_names': cams}
    sys.argv = ['serve', '--ckpt_dir', str(ckpt.parent), '--policy_class', 'ACT', '--task_name', 'serve', '--seed', '0', '--num_epochs', '1', '--lr', '1e-5', '--batch_size', '1', '--kl_weight', str(args.kl_weight), '--chunk_size', str(args.chunk_size), '--hidden_dim', str(args.hidden_dim), '--dim_feedforward', str(args.dim_feedforward)]
    policy = ACTPolicy(cfg)
    missing = policy.load_state_dict(torch.load(ckpt, map_location='cpu'))
    policy.cuda().eval()
    with open(stats_p, 'rb') as f:
        stats = pickle.load(f)
    q_mean = np.asarray(stats['qpos_mean']).reshape(-1)
    q_std = np.asarray(stats['qpos_std']).reshape(-1)
    a_mean = np.asarray(stats['action_mean']).reshape(-1)
    a_std = np.asarray(stats['action_std']).reshape(-1)
    print(f'[SERVER] checkpoint : {ckpt}')
    print(f'[SERVER] load       : {missing}')
    print(f'[SERVER] state_dim  : {dv.ESCAPE_STATE_DIM}   chunk {args.chunk_size}')
    print(f'[SERVER] cameras    : {cams}   image decimation x{args.scale}')
    print(f'[SERVER] action     : {args.action_space} -> reply keys {REPLY_KEYS[:2]}...')
    print(f"[SERVER] proprio    : {args.proprio} ({len(PROPRIO_STATE_KEYS)}-dim){('  BLANKED' if args.no_proprio else '')}")
    print(f'[SERVER] language   : IGNORED (ACT has no language input)', flush=True)
    import msgpack
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f'tcp://{args.host}:{args.port}')
    print(f'[SERVER] listening on {args.host}:{args.port}', flush=True)
    n = 0
    while True:
        try:
            req = msgpack.unpackb(sock.recv(), object_hook=_decode)
        except Exception as e:
            print(f'[SERVER] bad request: {e}', flush=True)
            sock.send(b'ERROR')
            continue
        ep = req.get('endpoint')
        if ep == 'ping':
            sock.send(msgpack.packb({'ok': True}, default=_encode))
            continue
        if ep != 'get_action':
            sock.send(msgpack.packb({'error': f'unknown endpoint {ep!r}'}, default=_encode))
            continue
        try:
            obs = req['data']['observation']
            qpos = np.array([float(np.asarray(obs['state'][k]).reshape(-1)[0]) for k in PROPRIO_STATE_KEYS], dtype=np.float32)
            if args.no_proprio:
                qpos = np.zeros_like(qpos)
            qpos_n = (qpos - q_mean) / q_std
            imgs = []
            ref_hw = None
            for cam in cams:
                key = CAM_TO_MODALITY[cam]
                img = np.asarray(obs['video'][key])
                while img.ndim > 3:
                    img = img[0]
                if ref_hw is None:
                    ref_hw = img.shape[:2]
                elif img.shape[:2] != ref_hw:
                    img = _resize_frames(img[None], ref_hw)[0]
                if args.scale > 1:
                    img = img[::args.scale, ::args.scale, :]
                imgs.append(img)
            cur = np.stack(imgs, axis=0).astype(np.float32) / 255.0
            cur = np.transpose(cur, (0, 3, 1, 2))[None, ...]
            with torch.inference_mode():
                a = policy(torch.from_numpy(qpos_n[None, :]).float().cuda(), torch.from_numpy(cur).float().cuda())
            a = a.detach().cpu().numpy()[0]
            a = a * a_std + a_mean
            reply = {k: a[:, i].astype(np.float32).reshape(-1, 1) for (i, k) in enumerate(REPLY_KEYS)}
            sock.send(msgpack.packb(reply, default=_encode))
            n += 1
            if n % 200 == 0:
                print(f'[SERVER] {n} chunks served', flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(msgpack.packb({'error': str(e)}, default=_encode))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

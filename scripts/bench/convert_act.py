# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import glob
import re
import os
import shutil
import sys
from pathlib import Path
import h5py
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from escape_ood.suites.spec import SUITES
DEFAULT_CAMS = ('cam_head',)
_TRAIN_STEM = re.compile('^[A-Za-z0-9-]+(_p\\d+of\\d+)?$')

def _select_archives(col, suite: str, split: str) -> list:
    arcs = [Path(p) for p in sorted(glob.glob(str(col / f'{suite}*.hdf5')))]
    arcs = [p for p in arcs if p.stem == suite or p.stem.startswith(suite + '_')]
    if split == 'train':
        return [p for p in arcs if _TRAIN_STEM.match(p.stem)]
    return arcs

def episode_lengths(archives: list[Path], split: str, keep=None, only=None) -> list[int]:
    """Length of every episode that will be written, for the padding target."""
    out = []
    for arch in archives:
        with h5py.File(arch, 'r') as f:
            if 'data' not in f:
                continue
            for name in f['data']:
                g = f['data'][name]
                a = dict(g.attrs)
                if split != 'all' and str(a.get('split')) != split:
                    continue
                if not a.get('success', True):
                    continue
                if keep is not None and (not keep(a)):
                    continue
                if only is not None and (str(arch), name) not in only:
                    continue
                out.append(int(min(len(g['actions']), len(g['obs']['joint_pos']))))
    return out
ACTION_SPACES = ('joint_abs', 'joint_delta_absgrip', 'joint_chunk_absgrip', 'eef_abs', 'eef_abs_rotvec', 'eef_delta', 'eef_chunk')

def _actions_for(g, space: str) -> np.ndarray:
    """The action column for one episode, in the requested encoding."""
    if space in ('joint_abs', 'joint_chunk_absgrip'):
        return np.asarray(g['actions'][:], dtype=np.float32)
    if space == 'joint_delta_absgrip':
        a = np.asarray(g['actions'][:], dtype=np.float64)
        q = np.asarray(g['obs']['joint_pos'][:], dtype=np.float64)
        n = min(len(a), len(q))
        out = a[:n].copy()
        out[:, :7] -= q[:n, :7]
        return out.astype(np.float32)
    if space in ('eef_abs', 'eef_chunk'):
        return np.asarray(g['actions_eef'][:], dtype=np.float32)
    if space == 'eef_abs_rotvec':
        from escape_ood.splits.eef_delta import eef_abs_rotvec_actions
        return eef_abs_rotvec_actions(np.asarray(g['actions_eef'][:], dtype=np.float64))
    if space == 'eef_delta':
        from escape_ood.splits.eef_delta import eef_delta_actions
        return eef_delta_actions(np.asarray(g['actions_eef'][:], dtype=np.float64), np.asarray(g['obs']['eef_pose'][:], dtype=np.float64))
    raise ValueError(f'unknown action space {space!r}')
PROPRIO_KINDS = ('joint',)
PROPRIO_DIMS = {'joint': 8}

def _proprio_for(g, kind: str) -> np.ndarray:
    if kind == 'joint':
        return np.asarray(g['obs']['joint_pos'][:], dtype=np.float32)
    ep = np.asarray(g['obs']['eef_pose'][:], dtype=np.float32)
    grip = np.asarray(g['obs']['joint_pos'][:, 7:8], dtype=np.float32)
    n = min(len(ep), len(grip))
    raise ValueError(kind)

def _resize_frames(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, hw[0], hw[1], 3), nearest-neighbour, no new dependency."""
    (h, w) = hw
    yi = (np.arange(h) * frames.shape[1] // h).clip(0, frames.shape[1] - 1)
    xi = (np.arange(w) * frames.shape[2] // w).clip(0, frames.shape[2] - 1)
    return frames[:, yi][:, :, xi]

def convert_suite(archives: list[Path], out_dir: Path, cams: tuple[str, ...], split: str, scale: int=1, action_space: str='joint_abs', no_proprio: bool=False, proprio: str='joint', keep=None, only: set[tuple[str, str]] | None=None, pad_to: int | None=None) -> tuple[int, int]:
    """Convert selected episodes; pad_to can share a global target across archive workers."""
    required_length = max(episode_lengths(archives, split, keep, only), default=0)
    if pad_to is None:
        pad_to = required_length
    elif pad_to < required_length:
        raise ValueError(f'padding target {pad_to} is shorter than selected episode length {required_length}')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    frames = 0
    for arch in archives:
        with h5py.File(arch, 'r') as f:
            if 'data' not in f:
                continue
            for name in sorted(f['data'], key=lambda k: int(k.split('_')[-1])):
                g = f['data'][name]
                attrs = dict(g.attrs)
                if split != 'all' and str(attrs.get('split')) != split:
                    continue
                if not attrs.get('success', True):
                    continue
                if keep is not None and (not keep(attrs)):
                    continue
                if only is not None and (str(arch), name) not in only:
                    continue
                act = _actions_for(g, action_space)
                qpos = _proprio_for(g, proprio)
                if no_proprio:
                    qpos = np.zeros_like(qpos)
                T = min(len(act), len(qpos))
                (act, qpos) = (act[:T], qpos[:T])
                n_pad = pad_to - T
                if n_pad > 0:
                    act = np.concatenate([act, np.repeat(act[-1:], n_pad, axis=0)])
                    qpos = np.concatenate([qpos, np.repeat(qpos[-1:], n_pad, axis=0)])
                eef_pose = None
                if action_space == 'eef_chunk':
                    eef_pose = np.asarray(g['obs']['eef_pose'][:T], dtype=np.float32)
                    if n_pad > 0:
                        eef_pose = np.concatenate([eef_pose, np.repeat(eef_pose[-1:], n_pad, axis=0)])
                dst = out_dir / f'episode_{idx}.hdf5'
                with h5py.File(dst, 'w') as o:
                    o.attrs['sim'] = True
                    for (k, v) in attrs.items():
                        if k in ('suite', 'split', 'success', 'episode_seed', 'attempt') or str(k).startswith('param_'):
                            o.attrs[k] = v
                    o.attrs['source_archive'] = str(arch)
                    o.attrs['source_episode'] = name
                    o.create_dataset('/action', data=act)
                    obs = o.create_group('/observations')
                    obs.create_dataset('qpos', data=qpos)
                    obs.create_dataset('qvel', data=np.zeros_like(qpos))
                    if eef_pose is not None:
                        obs.create_dataset('eef_pose', data=eef_pose)
                    img = obs.create_group('images')
                    ref_hw = None
                    if len(cams) > 1:
                        ref_hw = tuple(np.asarray(g['obs'][cams[0]].shape[1:3]).tolist())
                    for cam in cams:
                        if cam not in g['obs']:
                            raise KeyError(f'{arch.name}/{name}: no camera {cam!r}')
                        frames_arr = np.asarray(g['obs'][cam][:T])
                        if ref_hw is not None and frames_arr.shape[1:3] != ref_hw:
                            frames_arr = _resize_frames(frames_arr, ref_hw)
                        if scale > 1:
                            frames_arr = frames_arr[:, ::scale, ::scale, :]
                        if n_pad > 0:
                            frames_arr = np.concatenate([frames_arr, np.repeat(frames_arr[-1:], n_pad, axis=0)])
                        img.create_dataset(cam, data=frames_arr, dtype='uint8', chunks=(1,) + frames_arr.shape[1:], compression='gzip', compression_opts=1)
                    o.attrs['true_length'] = T
                idx += 1
                frames += T
    return (idx, frames)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--collect', default='data/output/collect')
    ap.add_argument('--out', default='data/output/act')
    ap.add_argument('--suite', default=None)
    ap.add_argument('--split', default='train', choices=('train', 'interior', 'test', 'all'))
    ap.add_argument('--action-space', default='joint_abs', choices=ACTION_SPACES, help='joint_abs (default) | joint_delta_absgrip | joint_chunk_absgrip | eef_abs | eef_abs_rotvec | eef_chunk | eef_delta. ACT ties the action head width to state_dim, so the 8-dim spaces need no patch; eef_delta is 7-dim and does not fit that assumption.')
    ap.add_argument('--proprio', default='joint', choices=PROPRIO_KINDS, help='measured joint state (8 dimensions)')
    ap.add_argument('--no-proprio', action='store_true', help='blank the joint-state input (constant token). Proprio is an ABSOLUTE observation; this asks whether it is needed, and whether it is itself a confinement channel.')
    ap.add_argument('--cameras', nargs='*', default=list(DEFAULT_CAMS))
    ap.add_argument('--scale', type=int, default=2, help='decimate images by this factor (2 -> 188x336). Every episode is end-padded to the longest episode in this dataset.')
    args = ap.parse_args()
    col = REPO_ROOT / args.collect
    out = REPO_ROOT / args.out
    cams = tuple(args.cameras)
    suites = [args.suite] if args.suite else []
    if not suites:
        print('[ERROR] pass --suite lift')
        return 1
    keep = None
    total_e = total_f = 0
    for su in suites:
        arcs = _select_archives(col, su, args.split)
        if not arcs:
            print(f'[SKIP] {su}: no archive')
            continue
        (n, fr) = convert_suite(arcs, out / su, cams, args.split, args.scale, args.action_space, args.no_proprio, args.proprio, keep)
        if keep is not None and n == 0:
            print(f'[ERROR] {su}: the cell filter selected no episodes; an empty dataset must not be trained on')
            return 1
        total_e += n
        total_f += fr
        print(f'[OK] {su:10s} {n:4d} episodes  {fr:7d} frames -> {out / su}')
    print(f"[INFO] action space: {args.action_space}   proprio: {('BLANKED' if args.no_proprio else args.proprio)} ({PROPRIO_DIMS[args.proprio]}-dim)")
    print(f'\n[INFO] {total_e} episodes, {total_f} frames, cameras {list(cams)}, image decimation x{args.scale}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

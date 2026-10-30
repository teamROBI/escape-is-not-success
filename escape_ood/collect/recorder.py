# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from pathlib import Path
from typing import Any
import h5py
import numpy as np

class EpisodeRecorder:
    """Accumulate one episode's tensors, then append it to an axis archive.

    Deliberately not streaming: an episode is only known to be a keeper once its success
    predicate has run, and appending frame-by-frame would mean either rewriting the group on
    failure or leaving partial episodes in the file for a reader to filter.
    """

    def __init__(self, image_keys: list[str], depth_keys: list[str] | None=None) -> None:
        self.image_keys = list(image_keys)
        self.depth_keys = list(depth_keys or [])
        self.reset()

    def reset(self) -> None:
        self.actions: list[np.ndarray] = []
        self.joint_pos: list[np.ndarray] = []
        self.actions_eef: list[np.ndarray] = []
        self.eef_pose: list[np.ndarray] = []
        self.images: dict[str, list[np.ndarray]] = {k: [] for k in self.image_keys}
        self.depths: dict[str, list[np.ndarray]] = {k: [] for k in self.depth_keys}
        self._discard_from = 0

    def step(self, action: np.ndarray, joint_pos: np.ndarray, images: dict[str, np.ndarray] | None=None, *, depths: dict[str, np.ndarray] | None=None, eef_target: np.ndarray | None=None, eef_pose: np.ndarray | None=None) -> None:
        """One frame. ``eef_target`` is the commanded grasp-point pose + gripper (8), ``eef_pose``
        the achieved grasp-point pose (7). Both or neither, per episode: a frame missing one
        while others carry it is a driver bug and is refused at write time."""
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.joint_pos.append(np.asarray(joint_pos, dtype=np.float32))
        if eef_target is not None:
            self.actions_eef.append(np.asarray(eef_target, dtype=np.float32))
        if eef_pose is not None:
            self.eef_pose.append(np.asarray(eef_pose, dtype=np.float32))
        if images:
            for (k, img) in images.items():
                if k in self.images:
                    self.images[k].append(np.asarray(img, dtype=np.uint8))
        if depths:
            for (k, d) in depths.items():
                if k in self.depths:
                    self.depths[k].append(np.asarray(d, dtype=np.float32))

    def mark_discard_point(self) -> None:
        """Drop everything captured so far.

        The expert opens each episode with a discard hold while RTX loads textures; those
        frames show a half-rendered scene and must not reach the dataset. ``TaskContext.hold``
        signals it through ``clear_recording_cache``.
        """
        self._discard_from = len(self.actions)

    @property
    def n_frames(self) -> int:
        return max(0, len(self.actions) - self._discard_from)

    def _slice(self, seq: list) -> list:
        return seq[self._discard_from:]

    def write(self, path: Path, index: int, attrs: dict[str, Any], *, success: bool, compression: str | None='gzip', level: int=4) -> str:
        """Append this episode to ``path`` and return the group name.

        Opened in append mode per episode rather than held open for the run: a collection run
        is hours long, and a process killed with the file handle open loses everything since
        the last flush. Reopening costs milliseconds against a multi-second episode.
        """
        if self.n_frames == 0:
            raise ValueError('no frames recorded; nothing to write')
        path.parent.mkdir(parents=True, exist_ok=True)
        name = f'episode_{index}'
        with h5py.File(str(path), 'a') as f:
            data = f.require_group('data')
            if name in data:
                del data[name]
            g = data.create_group(name)
            g.create_dataset('actions', data=np.stack(self._slice(self.actions)))
            obs = g.create_group('obs')
            obs.create_dataset('joint_pos', data=np.stack(self._slice(self.joint_pos)))
            for (name, seq, where, dim) in (('actions_eef', self.actions_eef, g, 8), ('eef_pose', self.eef_pose, obs, 7)):
                if not seq:
                    continue
                if len(seq) != len(self.actions):
                    raise ValueError(f'{name}: {len(seq)} frames but {len(self.actions)} actions -- the driver passed it on some steps and not others')
                arr = np.stack(self._slice(seq))
                if arr.shape[1] != dim:
                    raise ValueError(f'{name}: expected {dim} floats per frame, got {arr.shape[1]}')
                where.create_dataset(name, data=arr)
            for (k, frames) in self.images.items():
                fr = self._slice(frames)
                if not fr:
                    continue
                obs.create_dataset(k, data=np.stack(fr), compression=compression, compression_opts=level if compression else None, chunks=(1,) + np.asarray(fr[0]).shape)
            for (k, frames) in self.depths.items():
                fr = self._slice(frames)
                if not fr:
                    continue
                if len(fr) != len(self._slice(self.actions)):
                    raise ValueError(f'depth_{k}: {len(fr)} frames but {len(self._slice(self.actions))} actions -- the driver passed depth on some steps and not others')
                arr = np.stack(fr)
                if arr.ndim != 3:
                    raise ValueError(f'depth_{k}: expected (n, H, W) metric depth, got shape {arr.shape}')
                obs.create_dataset(f'depth_{k}', data=arr, compression=compression, compression_opts=level if compression else None, chunks=(1,) + arr.shape[1:])
            for (key, value) in attrs.items():
                g.attrs[key] = value
            g.attrs['success'] = bool(success)
            g.attrs['n_frames'] = int(self.n_frames)
        return name

def next_episode_index(path: Path) -> int:
    """Next free index in an archive, so a resumed run appends rather than overwrites."""
    if not path.is_file():
        return 0
    with h5py.File(str(path), 'r') as f:
        if 'data' not in f:
            return 0
        idx = [int(n.rsplit('_', 1)[1]) for n in f['data'].keys() if n.startswith('episode_')]
        return max(idx) + 1 if idx else 0

def archive_summary(path: Path) -> dict:
    """Counts and frame totals, for checking a run without loading the images."""
    if not path.is_file():
        return {'episodes': 0, 'successful': 0, 'frames': 0}
    with h5py.File(str(path), 'r') as f:
        if 'data' not in f:
            return {'episodes': 0, 'successful': 0, 'frames': 0}
        eps = list(f['data'].keys())
        ok = sum((1 for n in eps if bool(f['data'][n].attrs.get('success', True))))
        frames = sum((int(f['data'][n].attrs.get('n_frames', 0)) for n in eps))
        return {'episodes': len(eps), 'successful': ok, 'frames': frames, 'size_mb': round(path.stat().st_size / 1000000.0, 1)}

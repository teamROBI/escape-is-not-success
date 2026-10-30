"""Chunk-anchored residual actions for ACT.

The per-control-step residuals (`joint_delta_absgrip`, `eef_delta`) bundle two things: a RELATIVE action
representation and per-step STATE FEEDBACK (every step of a chunk is applied to the state measured at that
step). These spaces keep the relative representation and remove the feedback inside a chunk: every action of a
chunk is expressed relative to the state measured ONCE, when the chunk was queried, and applied relative to that
same anchor. They are the per-step spaces with the anchor frozen at the query; at k = 0 they coincide.

  joint_chunk_absgrip  8: [u[s+k][:7] - q[s][:7], u[s+k][7]]       (arm relative to the query-time joints)
  eef_chunk            7: [p_cmd[s+k] - p[s], rotvec(R_cmd[s+k] R[s]^-1), g[s+k]]   (eef_delta's formula,
                          anchored at the query-time grasp-point pose)

The anchor depends on where a training chunk starts, so it cannot be precomputed into the dataset: the ACT
episode files keep ABSOLUTE commands in `/action` plus the anchor source (`/observations/qpos` for joint,
`/observations/eef_pose` for EEF), and `make_load_data` builds ACT's dataset with the anchoring done per sample.
`anchored()` is numpy-only and shared by training, the replay check and the tests.
"""
from __future__ import annotations
import os
import numpy as np
from escape_ood.splits.eef_delta import eef_delta_actions
SPACES = ('joint_chunk_absgrip', 'eef_chunk')

def anchored(space: str, action_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """(K, 8) absolute commands + the state measured at the chunk's start -> (K, D) anchored actions."""
    a = np.asarray(action_abs, dtype=np.float64)
    anc = np.asarray(anchor, dtype=np.float64)
    if space == 'joint_chunk_absgrip':
        out = a.copy()
        out[:, :7] -= anc[:7]
        return out.astype(np.float32)
    if space == 'eef_chunk':
        return eef_delta_actions(a, np.repeat(anc[None, :7], len(a), axis=0))
    raise ValueError(f'unknown chunk-anchored space {space!r}')

def anchor_key(space: str) -> str:
    return '/observations/qpos' if space == 'joint_chunk_absgrip' else '/observations/eef_pose'

def action_dim(space: str) -> int:
    return 8 if space == 'joint_chunk_absgrip' else 7

def norm_stats(space: str, dataset_dir, num_episodes: int, chunk: int) -> dict:
    """ACT's get_norm_stats, except the action statistics are taken over ANCHORED chunks -- every start frame,
    the first `chunk` actions (the part ACT trains on) -- because that is the quantity the policy regresses."""
    import h5py
    (qpos_all, rows) = ([], [])
    for ep in range(num_episodes):
        with h5py.File(os.path.join(dataset_dir, f'episode_{ep}.hdf5'), 'r') as root:
            qpos = root['/observations/qpos'][()]
            act = root['/action'][()]
            anc = root[anchor_key(space)][()]
        qpos_all.append(qpos)
        T = len(act)
        (s, k) = np.meshgrid(np.arange(T), np.arange(chunk), indexing='ij')
        (s, k) = (s.ravel(), k.ravel())
        keep = s + k < T
        (s, k) = (s[keep], k[keep])
        if space == 'joint_chunk_absgrip':
            r = act[s + k].astype(np.float64)
            r[:, :7] -= anc[s, :7]
        else:
            r = eef_delta_actions(act[s + k].astype(np.float64), anc[s, :7].astype(np.float64))
        rows.append(r)
    rows = np.concatenate(rows).astype(np.float64)
    qpos_all = np.stack(qpos_all).astype(np.float64)
    (a_mean, a_std) = (rows.mean(0), np.clip(rows.std(0, ddof=1), 0.01, np.inf))
    (q_mean, q_std) = (qpos_all.mean((0, 1)), np.clip(qpos_all.std((0, 1), ddof=1), 0.01, np.inf))
    return {'action_mean': a_mean.astype(np.float32), 'action_std': a_std.astype(np.float32), 'qpos_mean': q_mean.astype(np.float32), 'qpos_std': q_std.astype(np.float32), 'example_qpos': qpos}

def make_load_data(space: str, chunk: int):
    """A drop-in for ACT's utils.load_data that anchors each sampled chunk. Call after ACT is on sys.path."""
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from utils import EpisodicDataset

    class AnchoredEpisodicDataset(EpisodicDataset):

        def __getitem__(self, index):
            path = os.path.join(self.dataset_dir, f'episode_{self.episode_ids[index]}.hdf5')
            with h5py.File(path, 'r') as root:
                is_sim = root.attrs['sim']
                assert is_sim, "chunk anchoring assumes ACT's sim branch (actions from start_ts, unshifted)"
                episode_len = root['/action'].shape[0]
                start_ts = np.random.choice(episode_len)
                qpos = root['/observations/qpos'][start_ts]
                images = np.stack([root[f'/observations/images/{c}'][start_ts] for c in self.camera_names], 0)
                action = anchored(space, root['/action'][start_ts:], root[anchor_key(space)][start_ts])
            self.is_sim = is_sim
            padded = np.zeros((episode_len, action.shape[1]), dtype=np.float32)
            padded[:len(action)] = action
            is_pad = np.zeros(episode_len)
            is_pad[len(action):] = 1
            image_data = torch.einsum('k h w c -> k c h w', torch.from_numpy(images)) / 255.0
            action_data = (torch.from_numpy(padded).float() - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            qpos_data = (torch.from_numpy(qpos).float() - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']
            return (image_data, qpos_data, action_data, torch.from_numpy(is_pad).bool())

    def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
        print(f'\nData from: {dataset_dir}  (chunk-anchored actions: {space}, chunk {chunk})\n')
        train_ratio = 0.8
        shuffled = np.random.permutation(num_episodes)
        (train_idx, val_idx) = (shuffled[:int(train_ratio * num_episodes)], shuffled[int(train_ratio * num_episodes):])
        stats = norm_stats(space, dataset_dir, num_episodes, chunk)
        train_ds = AnchoredEpisodicDataset(train_idx, dataset_dir, camera_names, stats)
        val_ds = AnchoredEpisodicDataset(val_idx, dataset_dir, camera_names, stats)
        train_dl = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
        val_dl = DataLoader(val_ds, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
        return (train_dl, val_dl, stats, train_ds.is_sim)
    return load_data

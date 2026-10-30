# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path
import h5py
REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = REPO_ROOT / 'third_party' / 'act'

def dataset_facts(d: Path) -> tuple[int, int, list[str]]:
    """Episode count, longest episode, and camera names — read from the data, not declared.

    ACT's config wants these as constants. Deriving them means a re-collection that changes
    episode lengths cannot leave a stale number behind, which is the class of bug this project
    keeps paying for.
    """
    eps = sorted(glob.glob(str(d / 'episode_*.hdf5')))
    if not eps:
        raise SystemExit(f'no episodes in {d}')
    longest = 0
    cams: list[str] = []
    for p in eps:
        with h5py.File(p, 'r') as f:
            longest = max(longest, f['/action'].shape[0])
            if not cams:
                cams = sorted(f['/observations/images'].keys())
    return (len(eps), longest, cams)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', required=True, help='lift')
    ap.add_argument('--data', default='data/output/act')
    ap.add_argument('--out', default='data/output/train/act')
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-05)
    ap.add_argument('--chunk-size', type=int, default=16, help='predicted control steps per query')
    ap.add_argument('--hidden-dim', type=int, default=512)
    ap.add_argument('--dim-feedforward', type=int, default=3200)
    ap.add_argument('--kl-weight', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anchor-chunk', default=None, choices=('joint_chunk_absgrip', 'eef_chunk'), help='train on CHUNK-ANCHORED residual actions computed per sample from the absolute commands in /action (see escape_ood/policy/act/chunk_anchor.py); default: /action as is')
    args = ap.parse_args()
    data_dir = REPO_ROOT / args.data / args.suite
    (n_eps, ep_len, cams) = dataset_facts(data_dir)
    ckpt_dir = REPO_ROOT / args.out / args.suite
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    task = f'sim_escape_ood_{args.suite}'
    print(f'[ACT] suite     : {args.suite}')
    print(f'[ACT] episodes  : {n_eps}   longest {ep_len} steps')
    print(f'[ACT] cameras   : {cams}')
    print(f'[ACT] chunk     : {args.chunk_size}   epochs {args.epochs}   batch {args.batch_size}')
    print(f'[ACT] seed      : {args.seed}   lr {args.lr}   kl {args.kl_weight}   hidden {args.hidden_dim}   ff {args.dim_feedforward}')
    print(f'[ACT] ckpt      : {ckpt_dir}', flush=True)
    sys.path.insert(0, str(ACT_DIR / 'detr'))
    sys.path.insert(0, str(ACT_DIR))
    os.chdir(ACT_DIR)
    import constants
    constants.SIM_TASK_CONFIGS[task] = {'dataset_dir': str(data_dir), 'num_episodes': n_eps, 'episode_len': ep_len, 'camera_names': cams}
    import detr.models.detr_vae as dv
    if not hasattr(dv, 'ESCAPE_STATE_DIM'):
        raise SystemExit('third_party/act is unpatched: it still hardcodes the 14-dim bimanual state.\nRun: bash scripts/setup/patch_act_state_dim.sh')
    print(f"[ACT] state_dim : {dv.ESCAPE_STATE_DIM} (patched from upstream's 14)", flush=True)
    import types
    if 'sim_env' not in sys.modules:
        stub = types.ModuleType('sim_env')
        stub.BOX_POSE = [None]
        sys.modules['sim_env'] = stub
    if 'visualize_episodes' not in sys.modules:
        vis = types.ModuleType('visualize_episodes')
        vis.save_videos = lambda *a, **k: None
        sys.modules['visualize_episodes'] = vis
    import imitate_episodes
    sys.argv = ['imitate_episodes.py', '--ckpt_dir', str(ckpt_dir), '--policy_class', 'ACT', '--task_name', task, '--seed', str(args.seed), '--num_epochs', str(args.epochs), '--lr', str(args.lr), '--batch_size', str(args.batch_size), '--kl_weight', str(args.kl_weight), '--chunk_size', str(args.chunk_size), '--hidden_dim', str(args.hidden_dim), '--dim_feedforward', str(args.dim_feedforward)]
    cfg = {'eval': False, 'onscreen_render': False, 'ckpt_dir': str(ckpt_dir), 'policy_class': 'ACT', 'task_name': task, 'batch_size': args.batch_size, 'seed': args.seed, 'num_epochs': args.epochs, 'lr': args.lr, 'kl_weight': args.kl_weight, 'chunk_size': args.chunk_size, 'hidden_dim': args.hidden_dim, 'dim_feedforward': args.dim_feedforward, 'temporal_agg': False, 'lr_backbone': 1e-05, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8, 'position_embedding': 'sine', 'masks': False, 'dilation': False, 'pre_norm': False, 'camera_names': cams, 'num_queries': args.chunk_size}
    if args.anchor_chunk:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from escape_ood.policy.act.chunk_anchor import make_load_data
        imitate_episodes.load_data = make_load_data(args.anchor_chunk, args.chunk_size)
        print(f'[ACT] anchor    : chunk-anchored {args.anchor_chunk} (anchor = state at chunk start)', flush=True)
    imitate_episodes.main(cfg)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

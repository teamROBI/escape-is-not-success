# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import h5py
    import numpy as np
except ImportError:
    print('[ERROR] needs h5py. Run: venvs/sim/bin/python escape_ood/collect/tests/test_recorder.py', flush=True)
    sys.exit(1)
from escape_ood.collect.recorder import EpisodeRecorder, archive_summary, next_episode_index
from escape_ood.suites.spec import SUITES, enumerate_conditions, params_from_attrs
FAILURES: list[str] = []

def check(cond: bool, label: str) -> None:
    print(f"  {('ok   ' if cond else 'FAIL ')} {label}")
    if not cond:
        FAILURES.append(label)

def fake_episode(rec: EpisodeRecorder, n: int, *, discard_first: int=0, img_shape=(4, 6, 3), eef: bool=True) -> None:
    """Drive the recorder the way the collector does, warm-up frames included."""

    def eef_kw(i):
        if not eef:
            return {}
        return {'eef_target': [0.3 + 0.01 * i, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.04], 'eef_pose': [0.3 + 0.01 * (i - 1), 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]}
    for i in range(discard_first):
        rec.step(action=[float(i)] * 8, joint_pos=[0.0] * 8, images={k: np.zeros(img_shape, np.uint8) for k in rec.image_keys}, **eef_kw(i))
    if discard_first:
        rec.mark_discard_point()
    for i in range(n):
        rec.step(action=[float(i)] * 7 + [0.04], joint_pos=[float(i) * 0.1] * 7 + [0.04], images={k: np.full(img_shape, i % 255, np.uint8) for k in rec.image_keys}, **eef_kw(i))

def test_roundtrip() -> None:
    print('\nrecorder -> HDF5 -> back')
    cond = enumerate_conditions(SUITES['lift'], ['escape_ood_cube_red'], splits=('train',))[0]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'lift.hdf5'
        rec = EpisodeRecorder(image_keys=['cam_head'])
        fake_episode(rec, 12)
        rec.write(path, 0, cond.to_attrs(), success=True)
        with h5py.File(str(path), 'r') as f:
            g = f['data']['episode_0']
            check(g['actions'].shape == (12, 8), f"actions are (T,8): {g['actions'].shape}")
            check(g['obs']['joint_pos'].shape == (12, 8), f'state is (T,8) — the 8-dim single-arm convention the plan pins')
            check(g['obs']['cam_head'].shape == (12, 4, 6, 3), f"images are (T,H,W,3): {g['obs']['cam_head'].shape}")
            check(g['actions_eef'].shape == (12, 8), f"commanded EEF pose is (T,8): {g['actions_eef'].shape}")
            check(g['obs']['eef_pose'].shape == (12, 7), f"achieved EEF pose is (T,7): {g['obs']['eef_pose'].shape}")
            check(abs(float(g['actions_eef'][0, 0]) - 0.3) < 1e-06, 'EEF arrays are discard-sliced in step with the actions')
            check(bool(g.attrs['success']) is True, 'success flag written')
            check(int(g.attrs['n_frames']) == 12, 'n_frames matches the arrays')
            recovered = params_from_attrs(dict(g.attrs))
            check(set(recovered) == set(cond.params), f'parameter names survive: {sorted(recovered)}')
            same = all((abs(float(recovered[k]) - float(v)) < 1e-06 for (k, v) in cond.params.items() if isinstance(v, (int, float))))
            check(same, 'numeric parameter values survive the HDF5 attr round trip')
            check(str(g.attrs['suite']) == 'lift', 'suite recorded')
            check(abs(float(g.attrs['sweep_coord']) - cond.sweep_coord) < 1e-06, 'sweep coordinate recorded')

def test_discard() -> None:
    print('\nwarm-up frames are dropped, not recorded')
    cond = enumerate_conditions(SUITES['lift'], ['escape_ood_cube_red'], splits=('train',))[0]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'a.hdf5'
        rec = EpisodeRecorder(image_keys=['cam_head'])
        fake_episode(rec, 10, discard_first=7)
        check(rec.n_frames == 10, f'n_frames excludes the discarded prefix: {rec.n_frames}')
        rec.write(path, 0, cond.to_attrs(), success=True)
        with h5py.File(str(path), 'r') as f:
            g = f['data']['episode_0']
            check(g['actions'].shape[0] == 10, 'discarded frames absent from the archive')
            check(abs(float(g['actions'][0, 7]) - 0.04) < 1e-06, 'first kept frame is a real frame, not a warm-up one')

def test_resume_and_summary() -> None:
    print('\nappending, resuming and summarising')
    conds = enumerate_conditions(SUITES['lift'], ['escape_ood_cube_red'], splits=('train',))[:3]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'b.hdf5'
        check(next_episode_index(path) == 0, 'fresh archive starts at 0')
        rec = EpisodeRecorder(image_keys=[])
        for (i, c) in enumerate(conds):
            rec.reset()
            fake_episode(rec, 5 + i)
            rec.write(path, i, c.to_attrs(), success=i != 1)
        check(next_episode_index(path) == 3, 'resume index is one past the last episode')
        s = archive_summary(path)
        check(s['episodes'] == 3, f"episode count: {s['episodes']}")
        check(s['successful'] == 2, f"failed episodes are kept but marked: {s['successful']}")
        check(s['frames'] == 5 + 6 + 7, f"frame total: {s['frames']}")
        rec.reset()
        fake_episode(rec, 4)
        rec.write(path, next_episode_index(path), conds[0].to_attrs(), success=True)
        check(archive_summary(path)['episodes'] == 4, 'append does not clobber earlier episodes')

def test_converter_accepts_recorder_output() -> None:
    print('\nthe ACT converter reads what the recorder writes')
    sys.path.insert(0, str(REPO_ROOT / 'scripts/bench'))
    from convert_act import episode_lengths, convert_suite
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'lift.hdf5'
        rec = EpisodeRecorder(image_keys=[])
        both = enumerate_conditions(SUITES['lift'], ['escape_ood_cube_red'], splits=('train',))[:4] + enumerate_conditions(SUITES['lift'], ['escape_ood_cube_red'], splits=('test',))[:3]
        for i, c in enumerate(both):
            rec.reset()
            fake_episode(rec, 6)
            rec.write(path, i, c.to_attrs(), success=True)
        check(len(episode_lengths([path], 'train')) == 4, 'converter selects four training episodes')
        check(len(episode_lengths([path], 'test')) == 3, 'converter selects three test episodes')
        out = Path(td) / 'act'
        n, frames = convert_suite([path], out, (), 'train', 2, 'joint_abs')
        check((n, frames) == (4, 24), 'conversion excludes all held-out episodes')
        with h5py.File(out / 'episode_0.hdf5') as f:
            check(f['action'].shape == (6, 8), 'recorded commands survive ACT conversion')


def test_eef_optional_but_not_ragged() -> None:
    print('\nEEF datasets are optional per episode, never ragged')
    cond = enumerate_conditions(SUITES['lift'], ['escape_ood_cube_red'], splits=('train',))[0]
    with tempfile.TemporaryDirectory() as td:
        rec = EpisodeRecorder(image_keys=[])
        fake_episode(rec, 5, eef=False)
        rec.write(Path(td) / 'noeef.hdf5', 0, cond.to_attrs(), success=True)
        with h5py.File(str(Path(td) / 'noeef.hdf5'), 'r') as f:
            g = f['data']['episode_0']
            check('actions_eef' not in g and 'eef_pose' not in g['obs'], 'an episode driven without EEF simply lacks the datasets')
        rec = EpisodeRecorder(image_keys=[])
        fake_episode(rec, 3, eef=True)
        rec.step(action=[0.0] * 8, joint_pos=[0.0] * 8)
        try:
            rec.write(Path(td) / 'ragged.hdf5', 0, cond.to_attrs(), success=True)
            check(False, 'a ragged EEF record is refused')
        except ValueError:
            check(True, 'a ragged EEF record is refused')

def test_depth_roundtrip_and_not_a_camera() -> None:
    """Depth remains separate from RGB camera channels."""
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'a.hdf5'
        rec = EpisodeRecorder(image_keys=['cam_head'], depth_keys=['cam_head'])
        (n, H, W) = (5, 4, 6)
        for i in range(n):
            rec.step(action=np.zeros(8, np.float32), joint_pos=np.zeros(8, np.float32), images={'cam_head': np.zeros((H, W, 3), np.uint8)}, depths={'cam_head': np.full((H, W), 0.5 + i, np.float32)}, eef_target=np.zeros(8, np.float32), eef_pose=np.zeros(7, np.float32))
        rec.write(path, 0, {}, success=True)
        with h5py.File(path, 'r') as f:
            obs = f['data']['episode_0']['obs']
            check('depth_cam_head' in obs, 'depth dataset written as depth_cam_head')
            d = obs['depth_cam_head']
            check(d.shape == (n, H, W), f'depth shape {d.shape} == (n, H, W)')
            check(d.dtype == np.float32, f'depth dtype {d.dtype} is float32')
            check(abs(float(d[2][0, 0]) - 2.5) < 1e-06, 'depth values survive unscaled')
            cams = [k for k in obs.keys() if k.startswith('cam') and len(obs[k].shape) == 4]
            check(cams == ['cam_head'], f'camera discovery sees only cam_head, got {cams}')

def test_depth_must_not_be_ragged() -> None:
    """Depth on some steps but not others is a driver bug, and must not reach an archive."""
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'a.hdf5'
        rec = EpisodeRecorder(image_keys=[], depth_keys=['cam_head'])
        for i in range(4):
            rec.step(action=np.zeros(8, np.float32), joint_pos=np.zeros(8, np.float32), depths={'cam_head': np.zeros((4, 6), np.float32)} if i < 2 else None)
        try:
            rec.write(path, 0, {}, success=True)
        except ValueError as e:
            check('some steps and not others' in str(e), f'ragged depth refused: {e}')
        else:
            check(False, 'ragged depth was written instead of refused')

def test_empty_refused() -> None:
    print('\nan empty episode is refused rather than written')
    cond = enumerate_conditions(SUITES['lift'], ['escape_ood_cube_red'], splits=('train',))[0]
    with tempfile.TemporaryDirectory() as td:
        rec = EpisodeRecorder(image_keys=[])
        try:
            rec.write(Path(td) / 'c.hdf5', 0, cond.to_attrs(), success=True)
            check(False, 'writing zero frames should raise')
        except ValueError:
            check(True, 'writing zero frames raises rather than creating an empty group')

def main() -> int:
    test_roundtrip()
    test_discard()
    test_resume_and_summary()
    test_converter_accepts_recorder_output()
    test_eef_optional_but_not_ragged()
    test_depth_roundtrip_and_not_a_camera()
    test_depth_must_not_be_ragged()
    test_empty_refused()
    print()
    if FAILURES:
        print(f'[FAIL] {len(FAILURES)} check(s) failed:')
        for f in FAILURES:
            print(f'  - {f}')
        return 1
    print('[PASS] all checks passed')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

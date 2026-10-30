import math
import re
import sys
import tempfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from escape_ood.collect.grasp import quat_mul, rotvec_to_quat
from escape_ood.policy.act.chunk_anchor import action_dim, anchored, norm_stats
from escape_ood.splits.eef_delta import eef_delta_actions

def _episode(T=60, seed=0):
    """A smooth synthetic episode: commanded joints/pose a little ahead of the measured ones."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)[:, None]
    q = 0.3 * np.sin(2 * np.pi * t * rng.uniform(0.5, 1.5, 8)) + rng.normal(0, 0.01, (T, 8))
    u = np.roll(q, -3, axis=0)
    u[-3:] = q[-1]
    yaw = 0.4 * np.sin(2 * np.pi * t[:, 0])
    quat = np.array([quat_mul(rotvec_to_quat((0.02 * math.sin(y), 0.0, y)), (0.0, 1.0, 0.0, 0.0)) for y in yaw])
    pose = np.concatenate([0.4 + 0.1 * np.sin(2 * np.pi * t * np.array([1, 2, 3])), quat], axis=1)
    cmd = np.concatenate([np.roll(pose, -2, axis=0), rng.uniform(0, 0.04, (T, 1))], axis=1)
    cmd[-2:, :7] = pose[-1]
    return (u, q, cmd, pose)

def test_first_step_equals_per_step_residual():
    (u, q, cmd, pose) = _episode()
    for s in (0, 7, 30):
        j = anchored('joint_chunk_absgrip', u[s:s + 16], q[s])
        assert np.allclose(j[0, :7], u[s, :7] - q[s, :7]) and np.isclose(j[0, 7], u[s, 7])
        e = anchored('eef_chunk', cmd[s:s + 16], pose[s])
        assert np.allclose(e[0], eef_delta_actions(cmd[s:s + 1], pose[s:s + 1])[0], atol=1e-06)

def test_decode_with_the_query_anchor_recovers_absolute_commands():
    (u, q, cmd, pose) = _episode()
    s = 11
    j = anchored('joint_chunk_absgrip', u[s:s + 16], q[s]).astype(np.float64)
    assert np.allclose(j[:, :7] + q[s, :7], u[s:s + 16, :7], atol=1e-06) and np.allclose(j[:, 7], u[s:s + 16, 7])
    e = anchored('eef_chunk', cmd[s:s + 16], pose[s]).astype(np.float64)
    for k in range(16):
        p = pose[s, :3] + e[k, :3]
        qk = np.array(quat_mul(rotvec_to_quat(tuple(e[k, 3:6])), tuple(pose[s, 3:7])))
        assert np.allclose(p, cmd[s + k, :3], atol=1e-06)
        assert abs(abs(float(qk @ cmd[s + k, 3:7])) - 1.0) < 1e-06
        assert np.isclose(e[k, 6], cmd[s + k, 7], atol=1e-06)

def test_later_steps_are_NOT_the_per_step_residual():
    """The point of the space: past k = 0 the anchor is the query-time state, not the state at s + k."""
    (u, q, cmd, pose) = _episode()
    s = 5
    j = anchored('joint_chunk_absgrip', u[s:s + 16], q[s])
    per_step = u[s + 8, :7] - q[s + 8, :7]
    assert not np.allclose(j[8, :7], per_step)

def test_norm_stats_shapes():
    import h5py
    with tempfile.TemporaryDirectory() as d:
        for ep in range(3):
            (u, q, cmd, pose) = _episode(seed=ep)
            with h5py.File(f'{d}/episode_{ep}.hdf5', 'w') as f:
                f.attrs['sim'] = True
                f.create_dataset('/observations/qpos', data=q.astype(np.float32))
                f.create_dataset('/observations/eef_pose', data=pose.astype(np.float32))
                f.create_dataset('/action', data=u.astype(np.float32))
        st = norm_stats('joint_chunk_absgrip', d, 3, 16)
        assert st['action_mean'].shape == (action_dim('joint_chunk_absgrip'),) and (st['action_std'] >= 0.01).all()
        assert st['qpos_mean'].shape == (8,)

def test_reply_keys_agree_and_decode_reads_the_anchor():
    ev = (REPO / 'escape_ood/sim/tools/eval_policy.py').read_text().replace("'", '"')
    sv = (REPO / 'scripts/bench/run_act_policy_server.py').read_text().replace("'", '"')
    assert '"joint_chunk_absgrip": [f"c_{k}" for k in JOINT_KEYS[:7]] + [JOINT_KEYS[7]]' in sv
    assert re.search('JOINT_CHUNK_ABSGRIP_KEYS = \\[f"c_\\{k\\}" for k in JOINT_KEYS\\[:7\\]\\] \\+ \\[JOINT_KEYS\\[7\\]\\]', ev)
    keys = re.search('EEF_CHUNK_KEYS = (\\[[^\\]]*\\])', ev).group(1)
    assert f'"eef_chunk": {keys}' in sv
    for space in ('joint_chunk_absgrip', 'eef_chunk'):
        body = ev.split(f'if space == "{space}":')[1].split('return')[0]
        assert 'anchor[' in body, f'{space} decode does not use the query-time anchor'
    assert 'anchor = (list(backend.getj()), backend.getp())' in ev
if __name__ == '__main__':
    for (name, fn) in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)

import math
import re
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from escape_ood.collect.grasp import grasp_quat, quat_mul, rotvec_to_quat
from escape_ood.splits.eef_delta import ROT_REF_WXYZ, eef_abs_rotvec_actions, eef_abs_rotvec_to_pose, eef_delta_actions, q_mul, q_to_rotvec

def _top_down_path(n=400, seed=0):
    """Commanded poses around the top-down grasp: yaw sweep +-70 deg plus small tilts, both quaternion signs."""
    rng = np.random.default_rng(seed)
    yaw = np.linspace(-math.radians(70), math.radians(70), n)
    tilt = 0.06 * np.sin(np.linspace(0, 6 * math.pi, n))[:, None] * np.array([1.0, -0.5, 0.0])
    r = np.concatenate([tilt[:, :2], yaw[:, None]], axis=1)
    q = np.array([quat_mul(rotvec_to_quat(tuple(v)), ROT_REF_WXYZ) for v in r])
    q *= np.where(rng.random(n) < 0.5, -1.0, 1.0)[:, None]
    pos = rng.uniform([0.3, -0.6, 0.0], [0.8, 0.6, 0.3], size=(n, 3))
    grip = rng.uniform(0.0, 0.04, size=(n, 1))
    return np.concatenate([pos, q, grip], axis=1)

def test_reference_is_the_nominal_grasp():
    assert np.allclose(ROT_REF_WXYZ, grasp_quat(0.0, 0.0), atol=1e-12), grasp_quat(0.0, 0.0)

def test_round_trip_is_exact():
    a = _top_down_path()
    enc = eef_abs_rotvec_actions(a).astype(np.float64)
    assert enc.shape == (len(a), 7)
    (pos, q, g) = eef_abs_rotvec_to_pose(enc)
    assert np.abs(pos - a[:, :3]).max() < 1e-06
    assert np.abs(g - a[:, 7]).max() < 1e-06
    dots = np.abs(np.sum(q * a[:, 3:7], axis=1))
    assert dots.min() > 1 - 1e-09, dots.min()

def test_evaluator_decode_matches():
    """apply_action's stdlib decode (grasp.quat_mul / rotvec_to_quat) == the numpy inverse."""
    enc = eef_abs_rotvec_actions(_top_down_path(50)).astype(np.float64)
    (_, q_np, _) = eef_abs_rotvec_to_pose(enc)
    for (row, qn) in zip(enc, q_np):
        qs = np.array(quat_mul(rotvec_to_quat(tuple(row[3:6])), ROT_REF_WXYZ))
        assert abs(abs(float(qs @ qn)) - 1.0) < 1e-09

def test_no_wrap_where_a_plain_rotvec_wraps():
    """The reason for the anchor: a plain rotation vector of a top-down grasp sits at |r| ~ pi and flips sign."""
    a = _top_down_path()
    plain = q_to_rotvec(a[:, 3:7])
    anchored = eef_abs_rotvec_actions(a)[:, 3:6]
    assert np.abs(np.diff(plain, axis=0)).max() > 3.0
    assert np.abs(np.diff(anchored, axis=0)).max() < 0.05
    assert np.linalg.norm(anchored, axis=1).max() < math.radians(75)

def test_is_eef_delta_with_a_constant_anchor():
    a = _top_down_path(100)
    anchor = np.tile(np.array([0.0, 0.0, 0.0, *ROT_REF_WXYZ]), (len(a), 1))
    assert np.allclose(eef_abs_rotvec_actions(a), eef_delta_actions(a, anchor), atol=1e-06)

def test_reply_keys_agree():
    ev = (REPO / 'escape_ood/sim/tools/eval_policy.py').read_text().replace("'", '"')
    sv = (REPO / 'scripts/bench/run_act_policy_server.py').read_text().replace("'", '"')
    keys = re.search('EEF_ABS_ROTVEC_KEYS = (\\[[^\\]]*\\])', ev).group(1)
    assert f'"eef_abs_rotvec": {keys}' in sv, 'server and evaluator disagree on eef_abs_rotvec reply keys'
    assert 'ROT_REF_WXYZ' in ev.split('if space == "eef_abs_rotvec":')[1].split('return')[0]
if __name__ == '__main__':
    for (name, fn) in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)

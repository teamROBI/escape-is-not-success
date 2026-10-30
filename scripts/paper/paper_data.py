import json, sys, os, statistics as st
from functools import lru_cache
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from escape_ood.metrics.extent import reach_fraction, escape_distance, direction_of, strict_success
from escape_ood.suites.spec import ENVELOPE
DATA = Path(os.environ.get('ESCAPE_RESULTS_DIR', REPO / 'data/output'))
EVAL = DATA / 'eval'
MANIFEST = json.load(open(REPO / 'results/manifest.json'))
ESCAPE_M = 0.01
APPROACH_STAGE = 0.25

def cell(frame, arm, gripper, cameras, rotation=None, anchor=None):
    """The first manifest cell with these interface fields. Variant cells (the rotation-vector control, the
    query-anchored residuals) carry a `rotation` / `anchor` field and are only returned when asked for by it."""
    for c in MANIFEST['cells']:
        if (c['frame'], c['arm_command'], c['gripper_command'], c['cameras'], c.get('rotation'), c.get('anchor')) == (frame, arm, gripper, cameras, rotation, anchor):
            return c
    raise KeyError((frame, arm, gripper, cameras))

@lru_cache(maxsize=None)
def records(run, split='test'):
    """A run's evaluation records. `provenance[run].eval_tag` names the evaluation to read when it differs from the
    run name (e.g. a re-evaluation under the comparators' scene settings); the checkpoint is always the run's own."""
    suffix = '_trn' if split == 'train' else ''
    tag = MANIFEST['provenance'].get(run, {}).get('eval_tag', run)
    return json.load(open(EVAL / f'lift_{split}_{tag}{suffix}.json'))['records']

def scored(run, direction='radial'):
    """Per-episode dicts: band (cm), escape (m), approach, success, path."""
    out = []
    for r in records(run):
        t = r['target_xy'][0]
        if direction_of(t) != direction:
            continue
        out.append({'band': round(r['sweep_coord']), 'escape': escape_distance(r['eef_path']), 'approach': reach_fraction(r['eef_path'], t), 'success': strict_success(r), 'path': r['eef_path'], 'target': t})
    return out

def stage_counts(run, direction='radial'):
    """(n, escaped > 1 cm, approach >= 0.25, strict success) for one run."""
    e = scored(run, direction)
    return (len(e), sum((x['escape'] > ESCAPE_M for x in e)), sum((x['approach'] >= APPROACH_STAGE for x in e)), sum((x['success'] for x in e)))

def band_medians(run):
    """{band_cm: median radial approach} for one run."""
    b = {}
    for x in scored(run):
        b.setdefault(x['band'], []).append(x['approach'])
    return {k: st.median(v) for (k, v) in sorted(b.items())}

def band_successes(run):
    b = {}
    for x in scored(run):
        b[x['band']] = b.get(x['band'], 0) + x['success']
    return dict(sorted(b.items()))

def split_success(run, split):
    rs = records(run, split)
    return (sum((strict_success(r) for r in rs)), len(rs))

def expert_summary():
    """Scripted expert through the same evaluator: terminal successes and the range of per-band MEDIAN approach.
    Bands are the spec's named test bands (R1-R5 radial, A1-A3 angular), via the record's `band` label.
    The expert branch applies its own final lift check, not the policy's three-check hold, so `success` here is a
    terminal success (strict_success falls back to it for records without hold fields)."""
    recs = json.load(open(DATA / MANIFEST['expert_run']['file']))['records']
    out = {'strict': sum((strict_success(r) for r in recs)), 'terminal': sum((bool(r['success']) for r in recs)), 'n': len(recs)}
    for d in ('radial', 'angular'):
        b = {}
        for r in recs:
            t = r['target_xy'][0]
            if direction_of(t) == d:
                b.setdefault(r['band'], []).append(reach_fraction(r['eef_path'], t))
        med = [st.median(v) for v in b.values()]
        out[d] = (min(med), max(med))
    return out

def archive_parity():
    """The primary collection vs the camera recollection, matched by episode_seed: (identical, total, max |diff|)
    over the recorded action arrays. Needs h5py/numpy (run with the sim venv)."""
    import glob, h5py, numpy as np

    def index(d):
        out = {}
        for f in sorted(glob.glob(str(DATA / d / '*.hdf5'))):
            with h5py.File(f) as h:
                for k in h['data']:
                    g = h['data'][k]
                    if g.attrs.get('split') == 'train':
                        out[int(g.attrs['episode_seed'])] = np.asarray(g['actions'][:])
        return out
    (a, b) = (index(MANIFEST['archives']['primary']), index(MANIFEST['archives']['camera']))
    common = sorted(set(a) & set(b))
    same = 0
    worst = 0.0
    for s in common:
        if a[s].shape == b[s].shape and np.array_equal(a[s], b[s]):
            same += 1
        elif a[s].shape == b[s].shape:
            worst = max(worst, float(np.abs(a[s] - b[s]).max()))
        else:
            worst = float('inf')
    return (same, len(common), worst)


CUBE_CENTRE_Z = 0.025

def approach_3d_check(direction='radial'):
    """Approach is scored in the table plane. How close does the grasp point get in 3D? Over every Table 1 run's test
    episodes in `direction`: the closest 3D distance from the grasp-point path to the cube's INITIAL centre, split into
    strict lifts, approach >= 0.25 without a lift, and the rest. Returns {group: (n, median_m)}."""
    import math
    groups = {'lift': [], 'approach_no_lift': [], 'rest': []}
    for c in MANIFEST['cells']:
        for run in c['runs'].values():
            for r in records(run):
                t = r['target_xy'][0]
                if direction_of(t) != direction:
                    continue
                d3 = min((math.dist(p, (t[0], t[1], CUBE_CENTRE_Z)) for p in r['eef_path']))
                if strict_success(r):
                    groups['lift'].append(d3)
                elif reach_fraction(r['eef_path'], t) >= APPROACH_STAGE:
                    groups['approach_no_lift'].append(d3)
                else:
                    groups['rest'].append(d3)
    return {k: (len(v), st.median(v)) for (k, v) in groups.items()}

def approach_push_check():
    """Does pushing the cube distort target approach (measured to the cube's START position)? Over every failed test
    episode (no strict lift) of every manifest run: failures with approach >= 0.25, how many moved the cube > 1 cm and
    > 5 cm, and how many failed episodes change the 0.25 classification when approach is scored against the cube's
    FINAL position instead (a final position inside the envelope counts as reached)."""
    import numpy as np
    out = {'failed': 0, 'hi': 0, 'hi_moved_1cm': 0, 'hi_moved_5cm': 0, 'reclassified': 0}
    for c in MANIFEST['cells']:
        for r in c['runs'].values():
            for x in records(r):
                t = x['target_xy'][0]
                a = reach_fraction(x['eef_path'], t)
                if a is None or strict_success(x):
                    continue
                fo = x['final_object_xy']
                fo = np.asarray(fo[0] if isinstance(fo[0], list) else fo, dtype=float)
                d = float(np.linalg.norm(fo - np.asarray(t, dtype=float)))
                a_final = reach_fraction(x['eef_path'], list(fo)) if ENVELOPE.distance_outside(*fo) > 1e-06 else 1.0
                out['failed'] += 1
                if a >= APPROACH_STAGE:
                    out['hi'] += 1
                    out['hi_moved_1cm'] += d > 0.01
                    out['hi_moved_5cm'] += d > 0.05
                out['reclassified'] += (a >= APPROACH_STAGE) != (a_final >= APPROACH_STAGE)
    return out

def envelope_sensitivity(halos=(0, 1, 2)):
    """Headline counts under other envelope margins. The paper's envelope extends HALF a lattice step beyond the
    outermost demonstrations (2.5 cm radially, 5 deg angularly). halo = k scales that margin: 0 is the training
    lattice itself, 2 is twice the margin. Returns {k: {run: (radial escape > 1 cm, radial approach >= 0.25,
    angular approach >= 0.25)}} over every manifest run; strict completion does not depend on the envelope."""
    from escape_ood.metrics.extent import reach_fraction, escape_distance, direction_of
    from escape_ood.suites.spec import Envelope
    out = {}
    for k in halos:
        env = Envelope(r_min=0.35 - 0.025 * k, r_max=0.5 + 0.025 * k, theta_max_deg=30.0 + 5.0 * k)
        res = {}
        for c in MANIFEST['cells']:
            for r in c['runs'].values():
                esc = appr_r = appr_a = 0
                for x in records(r):
                    t = x['target_xy'][0]
                    a = reach_fraction(x['eef_path'], t, env)
                    if direction_of(t) == 'radial':
                        esc += escape_distance(x['eef_path'], env) > ESCAPE_M
                        appr_r += a is not None and a >= APPROACH_STAGE
                    else:
                        appr_a += a is not None and a >= APPROACH_STAGE
                res[r] = (esc, appr_r, appr_a)
        out[k] = res
    return out

def eval_repeatability():
    """How reproducible is one evaluation of a checkpoint? Compares episode outcomes (escape > 1 cm, approach >= 0.25,
    strict lift) between two evaluations of the same checkpoint on the interior + test splits, matched by condition and
    episode. Pairs: an unchanged-setting repeat (*_rep vs the original) for two runs, and every control run's two
    wrist-mount settings (its eval_tag vs the run name). Returns {pair label: (episodes, same outcome)}."""

    def outc(x):
        t = x['target_xy'][0]
        return (escape_distance(x['eef_path']) > ESCAPE_M, (reach_fraction(x['eef_path'], t) or 0) >= APPROACH_STAGE, strict_success(x))
    key = lambda x: (round(x['target_xy'][0][0], 4), round(x['target_xy'][0][1], 4), x['episode'])

    def load(tag, split):
        return json.load(open(EVAL / f'lift_{split}_{tag}.json'))['records']

    def agree(a, b):
        n = same = 0
        for sp in ('interior', 'test'):
            A = {key(x): x for x in load(a, sp)}
            for y in load(b, sp):
                n += 1
                same += outc(A[key(y)]) == outc(y)
        return (n, same)
    out = {f'{r} repeat': agree(r, r + '_rep') for r in ('act_joint_chunk', 'act_eef_chunk')}
    n = same = 0
    for (r, pv) in MANIFEST['provenance'].items():
        if pv.get('eval_tag') and pv['eval_tag'] != r:
            (a, b) = agree(r, pv['eval_tag'])
            n += a
            same += b
    out['controls, two mount settings'] = (n, same)
    return out

def native_vs_strict():
    """Radial test completions per run under the evaluator's own success flag (one passing check suffices) versus
    strict_success (three consecutive checks). Returns {run: (native, strict)} for every manifest run."""
    out = {}
    for c in MANIFEST['cells']:
        for r in c['runs'].values():
            recs = [x for x in records(r) if direction_of(x['target_xy'][0]) == 'radial']
            out[r] = (sum((bool(x['success']) for x in recs)), sum((strict_success(x) for x in recs)))
    return out

def eval_repeatability_detail():
    """Per-outcome detail behind the repeatability sentence. (1) For the two seed-0 anchoring-control runs re-run with
    an unchanged setting (*_rep): episodes (interior + test) whose escape > 1 cm, approach >= 0.25 and strict outcome
    flipped. (2) For every control run evaluated twice (its eval_tag and the run name): the largest per-seed change in
    radial escape, approach >= 0.25 and strict counts."""
    key = lambda x: (round(x['target_xy'][0][0], 4), round(x['target_xy'][0][1], 4), x['episode'])

    def comp(x):
        t = x['target_xy'][0]
        return (escape_distance(x['eef_path']) > ESCAPE_M, (reach_fraction(x['eef_path'], t) or 0) >= APPROACH_STAGE, strict_success(x))
    load = lambda tag, sp: json.load(open(EVAL / f'lift_{sp}_{tag}.json'))['records']
    flips = {}
    for r in ('act_joint_chunk', 'act_eef_chunk'):
        (f, n) = ([0, 0, 0], 0)
        for sp in ('interior', 'test'):
            A = {key(x): x for x in load(r, sp)}
            for y in load(r + '_rep', sp):
                (a, b) = (comp(A[key(y)]), comp(y))
                n += 1
                for i in range(3):
                    f[i] += a[i] != b[i]
        flips[r] = (n, tuple(f))
    radial = lambda tag: [x for x in load(tag, 'test') if direction_of(x['target_xy'][0]) == 'radial']
    mx = [0, 0, 0]
    for (r, pv) in MANIFEST['provenance'].items():
        if pv.get('eval_tag') and pv['eval_tag'] != r:
            (a, b) = (radial(r), radial(pv['eval_tag']))
            for i in range(3):
                mx[i] = max(mx[i], abs(sum((comp(x)[i] for x in a)) - sum((comp(x)[i] for x in b))))
    return {'rerun_flips': flips, 'controls_max_count_change': tuple(mx)}


def grasp_height_check(thresholds_m=(0.02, 0.03)):
    """Final grasp-point height above the cube's rest centre (CUBE_CENTRE_Z) over every strict test completion of the
    13 paper cells (radial + angular). A 5 cm cube tipped on an edge raises its centre by (sqrt(2)-1)*2.5 cm = 1.04 cm,
    on a corner by 1.83 cm; a grasp point ending well above that is not a finger a tipped cube leans on.
    Returns (n, median_m, {threshold: count at or above})."""
    z = []
    for c in MANIFEST['cells']:
        for r in c['runs'].values():
            for d in ('radial', 'angular'):
                z += [x['path'][-1][2] - CUBE_CENTRE_Z for x in scored(r, d) if x['success']]
    return (len(z), st.median(z), {t: sum((v >= t for v in z)) for t in thresholds_m})

# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Iterator, Literal
ParamValue = float | int | str
Params = dict[str, ParamValue]
Split = Literal['train', 'interior', 'test']
Region = Literal['inside', 'radial', 'angular', 'corner', 'near']

@dataclass(frozen=True)
class Workspace:
    """Bounding rectangle of everything a condition can ask for, robot base frame, metres.

    Generous on purpose: the reachability probe sweeps it to find where the arm stops, and the
    table is generated ``0.10`` m larger than it on every side. The z of the table top is the
    base frame's z = 0.
    """
    x_min: float = 0.05
    x_max: float = 0.85
    y_min: float = -0.6
    y_max: float = 0.6
    z_table: float = 0.0

    def contains(self, x: float, y: float, *, margin: float=0.0) -> bool:
        return self.x_min + margin <= x <= self.x_max - margin and self.y_min + margin <= y <= self.y_max - margin
WORKSPACE = Workspace()

def polar_xy(r: float, theta_deg: float) -> tuple[float, float]:
    """Base-frame (x, y) of a polar condition. theta is measured from +x, positive toward +y."""
    t = math.radians(float(theta_deg))
    return (float(r) * math.cos(t), float(r) * math.sin(t))

def xy_polar(x: float, y: float) -> tuple[float, float]:
    """Inverse of :func:`polar_xy`: (r, theta_deg)."""
    return (math.hypot(x, y), math.degrees(math.atan2(y, x)))

@dataclass(frozen=True)
class Envelope:
    """The annular sector demonstrations are collected in: r in [r_min, r_max], |theta| <= theta_max.

    ``boundary_distance`` is the quantity the whole benchmark is organised around. Outside the
    envelope it is how far a point is from the trained region, split by *how* it is outside:

    * **radial** (|theta| within the sector, r beyond r_max): ``r - r_max`` -- straight out.
    * **angular** (r within the annulus, |theta| beyond the sector): arc length
      ``r * (|theta| - theta_max)`` -- how far the arm must swing at that radius. Arc length
      rather than chord because it is the distance *along the trained radius*, which is what a
      policy that has only ever reached at that radius would have to produce.
    * **near** (r below r_min): ``r_min - r``. Not swept -- the arm cannot grasp at its base --
      but defined, so the metric never sees an undefined value.
    * **corner** (outside in both): the Euclidean distance to the nearest envelope corner. No
      condition is placed there (a test asserts it) because a single scalar cannot say whether
      the failure was radial or angular; the value exists only so a rollout that wanders into
      a corner still scores.

    Inside it is negative: minus the distance to the nearest boundary, so "every training point
    is at least 2 cm inside" is ``boundary_distance <= -0.02``.
    """
    r_min: float
    r_max: float
    theta_max_deg: float

    def region(self, x: float, y: float) -> Region:
        (r, th) = xy_polar(x, y)
        in_r = self.r_min <= r <= self.r_max
        in_t = abs(th) <= self.theta_max_deg
        if in_r and in_t:
            return 'inside'
        if in_t:
            return 'radial' if r > self.r_max else 'near'
        if in_r:
            return 'angular'
        return 'corner'

    def contains(self, x: float, y: float) -> bool:
        return self.region(x, y) == 'inside'

    def boundary_distance(self, x: float, y: float) -> float:
        """Signed metres outside the envelope (negative inside). See the class docstring."""
        (r, th) = xy_polar(x, y)
        dth = math.radians(abs(th) - self.theta_max_deg)
        reg = self.region(x, y)
        if reg == 'inside':
            return -min(r - self.r_min, self.r_max - r, r * -dth)
        if reg == 'radial':
            return r - self.r_max
        if reg == 'near':
            return self.r_min - r
        if reg == 'angular':
            return r * dth
        rc = self.r_max if r > self.r_max else self.r_min
        (cx, cy) = polar_xy(rc, math.copysign(self.theta_max_deg, th))
        return math.hypot(x - cx, y - cy)

    def distance_outside(self, x: float, y: float) -> float:
        """Euclidean metres from (x, y) to the nearest point of the envelope; 0 inside.

        The denominator of the reach fraction: how far the object sits from anything the policy
        was trained on, as a straight-line distance, so it is directly comparable with the
        closest distance the grasp point ever gets to the object. Equals ``boundary_distance``
        for a radial point; for an angular point it is the chord to the boundary ray, not the
        arc, so it is never larger than ``boundary_distance``.
        """
        if self.region(x, y) == 'inside':
            return 0.0
        tm = math.radians(self.theta_max_deg)

        def to_arc(radius: float) -> float:
            (r, th) = (math.hypot(x, y), math.atan2(y, x))
            if abs(th) <= tm:
                return abs(r - radius)
            return min((math.hypot(x - radius * math.cos(s * tm), y - radius * math.sin(s * tm)) for s in (-1.0, 1.0)))

        def to_ray(sign: float) -> float:
            (ux, uy) = (math.cos(sign * tm), math.sin(sign * tm))
            t = min(max(x * ux + y * uy, self.r_min), self.r_max)
            return math.hypot(x - t * ux, y - t * uy)
        return min(to_arc(self.r_min), to_arc(self.r_max), to_ray(1.0), to_ray(-1.0))
ENVELOPE = Envelope(r_min=0.325, r_max=0.525, theta_max_deg=35.0)
TRAIN_R: list[float] = [0.35, 0.4, 0.45, 0.5]
TRAIN_THETA: list[float] = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
INTERIOR_R: list[float] = [0.375, 0.425, 0.475]
RADIAL_BAND_R: list[float] = [0.575, 0.625, 0.675, 0.725, 0.775]
ANGULAR_BAND_THETA: list[float] = [45.0, 55.0, 65.0]
_TOL = 1e-06

def _is_train_cell(p: Params) -> bool:
    (r, th) = (float(p['r']), float(p['theta_deg']))
    return any((abs(r - tr) < _TOL for tr in TRAIN_R)) and any((abs(th - tt) < _TOL for tt in TRAIN_THETA))

def envelope_split(p: Params) -> Split:
    """train if on the lattice; interior if inside the envelope; test otherwise."""
    if _is_train_cell(p):
        return 'train'
    (x, y) = polar_xy(float(p['r']), float(p['theta_deg']))
    return 'interior' if ENVELOPE.contains(x, y) else 'test'

def envelope_coord(p: Params) -> float:
    """Centimetres outside the envelope; 0 for anything inside."""
    (x, y) = polar_xy(float(p['r']), float(p['theta_deg']))
    return max(0.0, ENVELOPE.boundary_distance(x, y)) * 100.0

def envelope_grid() -> Iterator[Params]:
    """The shared cell set every Tier 1 suite sweeps: train lattice, interior, radial, angular."""
    for r in TRAIN_R:
        for th in TRAIN_THETA:
            yield {'r': r, 'theta_deg': th}
    for r in INTERIOR_R:
        for th in TRAIN_THETA:
            yield {'r': r, 'theta_deg': th}
    for r in RADIAL_BAND_R:
        for th in TRAIN_THETA:
            yield {'r': r, 'theta_deg': th}
    for r in TRAIN_R:
        for th in ANGULAR_BAND_THETA:
            for sign in (-1.0, 1.0):
                yield {'r': r, 'theta_deg': sign * th}

def band_label(p: Params) -> str | None:
    """'R1'..'R5' / 'A1'..'A3' for a test cell, None for train and interior.

    Labels are for reporting and for the skyline (``is_withheld_from_training``). The curve is
    drawn against ``envelope_coord``, never against these.
    """
    if envelope_split(p) != 'test':
        return None
    (r, th) = (float(p['r']), abs(float(p['theta_deg'])))
    for (i, br) in enumerate(RADIAL_BAND_R):
        if abs(r - br) < _TOL:
            return f'R{i + 1}'
    for (i, bt) in enumerate(ANGULAR_BAND_THETA):
        if abs(th - bt) < _TOL:
            return f'A{i + 1}'
    raise ValueError(f'test cell {p} is on no band -- envelope_grid and the band lists disagree')

@dataclass(frozen=True)
class Condition:
    pass
    suite: str
    params: Params
    object_key: str
    instruction: str
    split: Split
    sweep_coord: float

    def key(self) -> str:
        """Stable identifier, used for filenames and for --conditions selection."""
        ps = ','.join((f'{k}={_fmt(v)}' for (k, v) in sorted(self.params.items())))
        seed_object = self.object_key.replace('escape_ood_cube_red', 'tango_cube_red')
        return f'{self.suite}[{ps}]@{seed_object}'

    def to_attrs(self) -> dict[str, ParamValue]:
        """Flat, HDF5-attr-safe dict. Round-trips via :func:`params_from_attrs`."""
        out: dict[str, ParamValue] = {'suite': self.suite, 'object_key': self.object_key, 'split': self.split, 'sweep_coord': float(self.sweep_coord), 'instruction': self.instruction}
        for (k, v) in self.params.items():
            out[f'param_{k}'] = v
        return out

def params_from_attrs(attrs: dict[str, ParamValue]) -> Params:
    """Recover the parameter vector from HDF5 episode attrs written by :meth:`to_attrs`."""
    return {k[len('param_'):]: v for (k, v) in attrs.items() if str(k).startswith('param_')}

def _fmt(v: ParamValue) -> str:
    return f'{v:g}' if isinstance(v, float) else str(v)

@dataclass(frozen=True)
class Suite:
    pass
    name: str
    grid: Callable[[], Iterator[Params]]
    split: Callable[[Params], Split]
    sweep_coord: Callable[[Params], float]
    instruction: Callable[[Params, str], str]
    object_xy: Callable[[Params], tuple[float, float]]
    target_xy: Callable[[Params], tuple[float, float] | None]
    success_kind: str
    unit: str = 'cm'
    notes: str = ''

    def in_train(self, p: Params) -> bool:
        return self.split(p) == 'train'

def enumerate_conditions(suite: Suite, object_keys: list[str], splits: tuple[Split, ...]=('train', 'interior', 'test')) -> list[Condition]:
    """All conditions for one suite, bucketed into splits by the suite's own definition."""
    conds: list[Condition] = []
    for params in suite.grid():
        split = suite.split(params)
        if split not in splits:
            continue
        coord = 0.0 if split != 'test' else round(float(suite.sweep_coord(params)), 6)
        for obj in object_keys:
            conds.append(Condition(suite=suite.name, params=dict(params), object_key=obj, instruction=suite.instruction(params, obj), split=split, sweep_coord=coord))
    return conds

@dataclass(frozen=True)
class SweepSample:
    """A sampled evaluation set plus an explicit account of what was left out.

    Train and interior conditions are always kept in full: they are the controls every
    extent number is read against, and there are few of them. Test conditions are stratified by
    ``sweep_coord`` and sampled, and ``dropped`` / ``bin_counts`` state the bound rather than
    hiding it.
    """
    conditions: list[Condition]
    n_train: int
    n_interior: int
    n_test_sampled: int
    n_test_total: int
    bin_counts: list[int]
    bin_edges: list[float]

    @property
    def dropped(self) -> int:
        return self.n_test_total - self.n_test_sampled

    def summary(self) -> str:
        empty = sum((1 for c in self.bin_counts if c == 0))
        return f'{len(self.conditions)} conditions: {self.n_train} train + {self.n_interior} interior (all kept), {self.n_test_sampled}/{self.n_test_total} test sampled ({self.dropped} dropped) across {len(self.bin_counts)} sweep bins' + (f'; {empty} bin(s) EMPTY' if empty else '')

def sample_sweep(suite: Suite, object_keys: list[str], *, n_bins: int=10, per_bin: int=6, seed: int=0) -> SweepSample:
    """Keep every train and interior condition; stratify test by ``sweep_coord`` and sample.

    One stratum per distinct coordinate when there are few (the bands are discrete), equal-width
    bins over the *observed* test range otherwise. Deterministic from ``seed``. An empty bin is
    reported, not silently tolerated.
    """
    import random
    rng = random.Random(seed)
    train = enumerate_conditions(suite, object_keys, splits=('train',))
    interior = enumerate_conditions(suite, object_keys, splits=('interior',))
    test = enumerate_conditions(suite, object_keys, splits=('test',))
    kept = list(train) + list(interior)
    if not test:
        return SweepSample(kept, len(train), len(interior), 0, 0, [], [])
    distinct = sorted({round(c.sweep_coord, 6) for c in test})
    (lo, hi) = (distinct[0], distinct[-1])
    n = min(n_bins, len(distinct))
    if n <= 1 or hi <= lo:
        take = sorted(rng.sample(test, min(per_bin * max(n, 1), len(test))), key=lambda c: c.key())
        return SweepSample(kept + take, len(train), len(interior), len(take), len(test), [len(take)], [lo, hi])
    if len(distinct) <= n_bins:
        groups: dict[float, list[Condition]] = {d: [] for d in distinct}
        for c in test:
            groups[round(c.sweep_coord, 6)].append(c)
        buckets = [groups[d] for d in distinct]
        edges = list(distinct)
    else:
        width = (hi - lo) / n
        buckets = [[] for _ in range(n)]
        for c in test:
            i = min(int((round(c.sweep_coord, 6) - lo) / width + 1e-09), n - 1)
            buckets[i].append(c)
        edges = [round(lo + i * width, 6) for i in range(n + 1)]
    picked: list[Condition] = []
    counts: list[int] = []
    for b in buckets:
        take = sorted(rng.sample(b, min(per_bin, len(b))), key=lambda c: c.key()) if b else []
        picked.extend(take)
        counts.append(len(take))
    return SweepSample(conditions=kept + picked, n_train=len(train), n_interior=len(interior), n_test_sampled=len(picked), n_test_total=len(test), bin_counts=counts, bin_edges=edges)
_OBJ_PHRASE = {'escape_ood_cube_red': 'red block'}
_COLOURS = ('red', 'blue', 'green', 'yellow', 'black', 'white')

def _obj_phrase(asset_name: str) -> str:
    pass
    name = str(asset_name)
    if name in _OBJ_PHRASE:
        return _OBJ_PHRASE[name]
    parts = [t for t in name.split('_') if t and t not in ('escape_ood', 'ycb')]
    if len(parts) > 1 and parts[-1] in _COLOURS:
        parts = [parts[-1]] + parts[:-1]
    return ' '.join(parts)
OBJECT_FOR_SUITE: dict[str, str] = {'lift': 'escape_ood_cube_red'}

def object_keys_for(suite_name: str) -> list[str]:
    """Graspable objects a suite uses. One definition, because four callers need to agree."""
    return [OBJECT_FOR_SUITE[suite_name]]
LANG_OBS_KEY = {'task': 'annotation.human.action.task_description'}

def lang_obs_key(channel: str) -> str:
    """Observation key for a language channel; raises on anything Tier 1 does not carry."""
    try:
        return LANG_OBS_KEY[channel]
    except KeyError:
        raise ValueError(f'unknown language channel {channel!r}; Tier 1 carries {sorted(LANG_OBS_KEY)} only') from None
PARKING_X0 = 3.0
PARKING_DX = 0.3
PARKING_Y = -1.2
PARKING_Z = 0.05

def parking_slot(i: int) -> tuple[float, float, float]:
    """Off-workspace parking position for spawn index ``i``, clear of the camera frustum."""
    return (PARKING_X0 + PARKING_DX * float(i), PARKING_Y, PARKING_Z)

def _swept_xy(p: Params) -> tuple[float, float]:
    return polar_xy(float(p['r']), float(p['theta_deg']))

def _no_target(p: Params) -> None:
    return None
LIFT = Suite(name='lift', grid=envelope_grid, split=envelope_split, sweep_coord=envelope_coord, instruction=lambda p, obj: f'pick up the {_obj_phrase(obj)}', object_xy=_swept_xy, target_xy=_no_target, success_kind='lift', notes='Reach + grasp + lift. The object sits at the swept cell; no goal location.')
SUITES: dict[str, Suite] = {'lift': LIFT}

def suite(name: str) -> Suite:
    if name not in SUITES:
        raise KeyError(f"Unknown suite '{name}'. Available: {', '.join(sorted(SUITES))}")
    return SUITES[name]

def object_start_xy(condition: Condition) -> tuple[float, float]:
    """Where the graspable spawns for a condition. Gate, collector, renderer and evaluator all
    read this one function, so the gate cannot certify a placement collection never runs."""
    return suite(condition.suite).object_xy(condition.params)

def object_start_yaw(condition: Condition) -> float:
    """Yaw the object spawns at, degrees. Constant in Tier 1; becomes load-bearing for the mug."""
    return float(condition.params.get('yaw_deg', 0.0))

def target_xys(condition: Condition) -> list[tuple[float, float]]:
    """Goal positions for a condition: one for the place suites, none for lift/press."""
    t = suite(condition.suite).target_xy(condition.params)
    return [] if t is None else [t]

def is_withheld_from_training(suite_name: str, params: Params, *, include_bands: tuple[str, ...]=()) -> bool:
    """Whether an episode at these params stays OUT of a training dataset.

    Only ``train`` episodes are converted by default. A skyline dataset opts a band back in
    (``include_bands=("R2",)``) to ask whether the behaviour is learnable at that reach when
    shown -- the control that separates *cannot generalize* from *cannot represent*. Interior
    episodes are never trained on; they are the ID control.
    """
    split = suite(suite_name).split(params)
    if split == 'train':
        return False
    if split == 'test' and band_label(params) in include_bands:
        return False
    return True

def expected_train_episodes(suite_name: str, demos_per_condition: int) -> int:
    """What a complete training dataset for one suite must contain. The fine-tune launcher
    refuses to start on anything smaller unless told the shortfall is deliberate."""
    s = suite(suite_name)
    n = len(enumerate_conditions(s, object_keys_for(suite_name), splits=('train',)))
    return n * int(demos_per_condition)

def instruction_from_attrs(attrs: dict[str, ParamValue]) -> str:
    """Re-derive an episode's instruction from its recorded HDF5 attrs.

    The archive stores the string it was collected under, but that string is a pure function of
    (suite, params, object) and the spec owns that function. Re-deriving makes a wording fix a
    re-conversion instead of a re-collection. Raises rather than guessing.
    """
    name = attrs.get('suite')
    if name is None:
        raise KeyError("episode attrs carry no 'suite'; cannot re-derive the instruction")
    obj = attrs.get('object_key')
    if obj is None:
        raise KeyError(f"episode attrs carry no 'object_key' (suite {name})")
    return suite(str(name)).instruction(params_from_attrs(attrs), str(obj))

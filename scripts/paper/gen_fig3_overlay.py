import os
import glob
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from paper_plot_style import plt, OI, FIG_DIR
from paper_data import DATA
from escape_ood.suites.layout_gen import FIG_CAM, TABLE_TOP_Z, project_to_head_cam
from escape_ood.suites.spec import ENVELOPE
from escape_ood.metrics.extent import escape_distance, reach_fraction, strict_success
MEDIA = Path(os.environ.get('ESCAPE_MEDIA_DIR', DATA / 'media/fig3_rollouts'))
OUT = FIG_DIR / 'fig3_overlay.pdf'
POLICIES = [('act', 'joint absolute, head camera'), ('act_eef_abs_wo', 'EEF absolute, wrist camera'), ('act_joint_chunk', 'joint residual$^\\ddagger$, head camera')]
CASES = [(0.725, 20.0, 0, 'cube 20 cm out, $\\theta$ = 20°')]
CROP = (240, 990, 48, 520)
N_GHOST = 6
(FS, FS_SMALL) = (7.5, 7.2)

def tagfmt(v):
    return ('m' if v < 0 else '') + f'{abs(v):g}'.replace('.', 'p')

def run_dirs(tag):
    """A policy's render directories: MEDIA/<tag>, or MEDIA/<tag>_<job> when its cells were split across jobs.
    Matched exactly, so 'act' never picks up 'act_eef_abs_wo'."""
    return [p for p in (MEDIA / tag, *sorted(MEDIA.glob(f'{tag}_[a-z]'))) if p.is_dir()]

def frames(tag, r, th, ep):
    d = [g for rd in run_dirs(tag) for g in glob.glob(str(rd / 'video' / f'lift_*_r{tagfmt(r)}_theta_deg{tagfmt(th)}_ep{ep}_fig'))]
    assert len(d) == 1, (tag, r, th, ep, d)
    files = sorted(glob.glob(d[0] + '/[0-9][0-9][0-9][0-9].png'))
    out = []
    for f in files:
        dp = f[:-4] + '_depth.npy'
        out.append((int(Path(f).stem), cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), np.load(dp).astype(np.float32) if Path(dp).exists() else None))
    return out

def record(tag, r, th, ep):
    res = [g for rd in run_dirs(tag) for g in glob.glob(str(rd / 'lift_test_*.json'))]
    assert res, (tag, res)
    for f in res:
        for x in json.load(open(f))['records']:
            if abs(x['params']['r'] - r) < 1e-06 and abs(x['params']['theta_deg'] - th) < 1e-06 and (x['episode'] == ep):
                return x
    raise KeyError((tag, r, th, ep))

def composite(fr):
    """Onion-skin one episode with EXACT arm masks from the figure camera's depth.

    Clean plate: per pixel, the colour from the frame where that pixel was farthest away -- the scene with every
    moving part removed (the robot base never moves, so it stays). A pose at time t is wherever something is nearer
    than the clean plate. Earlier poses are drawn as tinted translucent silhouettes with an outline, the first pose
    faintest; the last frame's moving parts are drawn solid in their true colour. Depth ignores shadows, so unlike a
    colour difference the masks carry no shadow.
    """
    rgb = np.stack([f[1] for f in fr]).astype(np.float32)
    dep = np.stack([f[2] for f in fr])
    far = dep.argmax(0)
    plate = np.take_along_axis(rgb, far[None, ..., None].repeat(3, -1), 0)[0]
    plate_d = dep.max(0)
    idx = np.linspace(0, len(fr) - 1, N_GHOST + 1).round().astype(int)
    out = plate.copy()
    tint = np.array([135, 135, 135], np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for (k, i) in enumerate(idx):
        m = (plate_d - dep[i] > 0.01).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        mf = m.astype(np.float32)[..., None]
        if i == idx[-1]:
            out = out * (1 - mf) + rgb[i] * mf
            continue
        a = 0.28 + 0.32 * k / max(1, len(idx) - 2)
        ghost = 0.55 * rgb[i] + 0.45 * tint
        out = out * (1 - a * mf) + ghost * (a * mf)
        (cs, _) = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        o = np.clip(out, 0, 255).astype(np.uint8)
        cv2.drawContours(o, cs, -1, (85, 85, 85), 1, lineType=cv2.LINE_AA)
        out = o.astype(np.float32)
    return (np.clip(out, 0, 255).astype(np.uint8), [fr[i][0] for i in idx])

def uv(x, y, z):
    (u, v, _) = project_to_head_cam(x, y, z, cam=FIG_CAM)
    return (u - CROP[0], v - CROP[2])

def envelope_ring():
    e = ENVELOPE
    th = [-e.theta_max_deg + k * 2 * e.theta_max_deg / 60 for k in range(61)]
    pts = [(e.r_max, t) for t in th] + [(e.r_min, t) for t in reversed(th)]
    pts.append(pts[0])
    return [uv(r * math.cos(math.radians(t)), r * math.sin(math.radians(t)), TABLE_TOP_Z) for (r, t) in pts]

def main():
    (u0, u1, v0, v1) = CROP
    (W, gap, head, rowgap) = (5.0, 0.06, 0.17, 0.04)
    lab = 0.14 if len(CASES) > 1 else 0.0
    tile_w = (W - lab - 2 * gap) / 3
    tile_h = tile_w * (v1 - v0) / (u1 - u0)
    rowlab = 0.0
    H = head + len(CASES) * (tile_h + rowlab) + (len(CASES) - 1) * rowgap
    fig = plt.figure(figsize=(W, H))
    ring = envelope_ring()
    for (j, (tag, name)) in enumerate(POLICIES):
        x = lab + j * (tile_w + gap)
        short = name.replace('absolute', 'abs.').replace('residual', 'res.')
        fig.text((x + tile_w / 2) / W, 1.0, short, fontsize=FS_SMALL, ha='center', va='top')
        for (i, (r, th, ep, label)) in enumerate(CASES):
            fr = frames(tag, r, th, ep)
            (img, steps) = composite(fr)
            rec = record(tag, r, th, ep)
            (e, a, s) = (escape_distance(rec['eef_path']), reach_fraction(rec['eef_path'], rec['target_xy'][0]), strict_success(rec))
            top = H - head - i * (tile_h + rowlab + rowgap) - rowlab
            ax = fig.add_axes([x / W, (top - tile_h) / H, tile_w / W, tile_h / H])
            ax.imshow(img[v0:v1, u0:u1])
            ax.fill(*zip(*ring), color='#1f4e9c', alpha=0.18, lw=0)
            ax.plot(*zip(*ring), color='#1f4e9c', lw=0.7, ls=(0, (2.5, 1.2)))
            p = [uv(q[0], q[1], q[2] + TABLE_TOP_Z) for q in rec['eef_path']]
            if s:
                (col, ls) = (OI['vermillion'], '-')
            elif (a or 0) >= 0.25:
                (col, ls) = (OI['orange'], '-')
            elif e > 0.01:
                (col, ls) = (OI['green'], (0, (2.2, 1.2)))
            else:
                (col, ls) = ('#8a8a8a', '-')
            ax.plot(*zip(*p), color='#151515', lw=2.1, ls=ls, alpha=0.8, solid_capstyle='round')
            ax.plot(*zip(*p), color=col, lw=1.0, ls=ls, solid_capstyle='round')
            ax.plot(*p[-1], marker='o', ms=2.6, color='black', zorder=5)
            ax.set_xlim(0, u1 - u0)
            ax.set_ylim(v1 - v0, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            box = dict(boxstyle='square,pad=0.12', fc='white', ec='none', alpha=0.85)
            ax.text(0.97, 0.04, f'escape {100 * e:.1f} cm\napproach {a or 0:.2f}', transform=ax.transAxes, fontsize=FS_SMALL - 0.2, ha='right', va='bottom', linespacing=1.0, bbox=box)
            ax.text(0.97, 0.96, 'completed' if s else 'not completed', transform=ax.transAxes, fontsize=FS_SMALL - 0.2, ha='right', va='top', color='white' if s else 'black', bbox=dict(boxstyle='square,pad=0.12', fc=OI['vermillion'] if s else 'white', ec='none', alpha=0.9))
            if j == 0 and lab:
                fig.text(0.0, (top - tile_h / 2) / H, label.replace('cube ', '').replace(' the envelope', ''), fontsize=FS_SMALL, ha='left', va='center', rotation=90, color='#444444')
            print(tag, r, th, ep, 'frames', len(fr), 'ghosts at steps', steps, f'e={e:.3f} a={a} s={s}')
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix('.png'), dpi=250)
    print('wrote', OUT, f'{W:.2f} x {H:.2f} in')
if __name__ == '__main__':
    main()

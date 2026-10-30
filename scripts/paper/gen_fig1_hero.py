"""Fig 1 (hero): same success, different failure stage.

(a) The task as the policy sees it: its head-camera frame with the (unmarked) envelope, training and test
    positions projected onto the table (gen_fig0_scene.draw).
(b) One column per ACT policy (see POLICIES). Top: every seed-0 radial grasp-point path, top-down and oriented like
    (a) (robot at the top, its left on the right), coloured by the furthest stage reached (illustrative).
    Bottom: radial test episodes (of 70) that escaped the envelope (> 1 cm), approached the object (target
    approach >= 0.25) and completed (strict success); bar = mean over 3 training seeds, whisker = min-max.
    The stages are nested: approach >= 0.25 implies the closest point is >= 1.25 cm outside the envelope.
"""
import math
import statistics as st
from matplotlib.patches import Polygon
from paper_plot_style import plt, OI, save_fig
import paper_data as D
import gen_fig0_scene as SCENE
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'white'
POLICIES = [('joint absolute, head camera', D.cell('joint', 'absolute', 'absolute', 'head')), ('EEF absolute, wrist camera', D.cell('EEF', 'absolute', 'absolute', 'wrist')), ('joint residual$^\\ddagger$, head camera', D.cell('joint', 'residual', 'absolute', 'head', anchor='query'))]
STAGES = [('escaped > 1 cm', OI['green']), ('approach $\\geq$ 0.25', OI['orange']), ('strict completion', OI['vermillion'])]
HATCH = [None, '////', None]
INSIDE = '#b8b8b8'
(FS, FS_SMALL) = (7.0, 6.5)
(W, H) = (5.5, 1.74)
fig = plt.figure(figsize=(W, H))

def axes_in(left, bottom, width, height):
    return fig.add_axes([left / W, bottom / H, width / W, height / H])
(COL_X0, COL_W, COL_GAP) = (1.86, 1.18, 0.06)
(MAP_XLIM, MAP_YLIM) = ((-0.44, 0.44), (-0.84, -0.22))
MAP_H = COL_W * (MAP_YLIM[1] - MAP_YLIM[0]) / (MAP_XLIM[1] - MAP_XLIM[0])
HEAD_Y = H - 0.02
MAP_Y = HEAD_Y - 0.23 - MAP_H
(BAR_Y, BAR_H) = (0.24, MAP_Y - 0.24 - 0.09)
(img_h, img_w) = (SCENE.CROP[3] - SCENE.CROP[2], SCENE.CROP[1] - SCENE.CROP[0])
A_H = MAP_H
A_W = A_H * img_w / img_h
ax = axes_in(0.0, MAP_Y, A_W, A_H)
SCENE.draw(ax, lw=0.8, s_train=2.2, s_test=5.0, lw_test=0.5)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(True)
    s.set_linewidth(0.4)
fig.text(0.0, HEAD_Y / H, '(a) task: lift the cube', fontsize=FS, ha='left', va='top', fontweight='bold')
fig.text(0.0, (HEAD_Y - 0.13) / H, 'head camera (inset: wrist crop)', fontsize=FS_SMALL, ha='left', va='top', color='#444444')
import h5py
WRIST = ('collect_wrist/lift_p0of3.hdf5', 'episode_6', 14)
with h5py.File(D.DATA / WRIST[0]) as h5:
    g = h5['data'][WRIST[1]]
    assert g.attrs['split'] == 'train' and abs(g.attrs['param_r'] - 0.45) < 1e-06, dict(g.attrs)
    wimg = g['obs/cam_wrist'][WRIST[2]][105:215]
I_W = 0.42 * A_W
I_H = I_W * wimg.shape[0] / wimg.shape[1]
iax = axes_in(A_W - I_W - 0.025, MAP_Y + A_H - I_H - 0.025, I_W, I_H)
iax.imshow(wimg)
iax.set_xticks([])
iax.set_yticks([])
for sp in iax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(0.6)
    sp.set_color('white')
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color='#1f4e9c', lw=0.8, ls=(0, (3, 1.5))), Line2D([], [], color='#1f4e9c', marker='o', ms=2.2, lw=0), Line2D([], [], color=SCENE.RADIAL_C, marker='x', ms=3.2, mew=0.6, lw=0), Line2D([], [], color=SCENE.ANGULAR_C, marker='o', ms=2.6, mew=0.5, mfc='none', lw=0)], ['envelope (overlay)', 'training positions', 'radial test positions', 'angular test positions'], loc='upper left', bbox_to_anchor=(0.0, (BAR_Y + BAR_H + 0.02) / H), frameon=False, fontsize=FS_SMALL, handlelength=1.4, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.0, borderpad=0.0)
E = D.ENVELOPE

def sector(r0, r1, t):
    ts = [math.radians(-t + 2 * t * i / 40) for i in range(41)]
    outer = [(r1 * math.sin(a), -r1 * math.cos(a)) for a in ts]
    inner = [(r0 * math.sin(a), -r0 * math.cos(a)) for a in reversed(ts)]
    return outer + inner

def stage(x):
    if x['success']:
        return 3
    if x['approach'] >= D.APPROACH_STAGE:
        return 2
    if x['escape'] > D.ESCAPE_M:
        return 1
    return 0
fig.text(COL_X0 / W, HEAD_Y / H, '(b) three ACT policies, radial test positions', fontsize=FS, ha='left', va='top', fontweight='bold')
for (pi, (name, c)) in enumerate(POLICIES):
    x0 = COL_X0 + pi * (COL_W + COL_GAP)
    fig.text((x0 + COL_W / 2) / W, (HEAD_Y - 0.13) / H, name.replace('\n', ', '), fontsize=FS_SMALL, ha='center', va='top')
    ax = axes_in(x0, MAP_Y, COL_W, MAP_H)
    ax.add_patch(Polygon(sector(E.r_min, E.r_max, E.theta_max_deg), closed=True, facecolor='#e8e8e8', edgecolor='#9a9a9a', linewidth=0.5, zorder=0))
    eps = D.scored(c['runs']['0'])
    for level in (0, 1, 2, 3):
        col = INSIDE if level == 0 else STAGES[level - 1][1]
        for x in eps:
            if stage(x) != level:
                continue
            p = x['path']
            ax.plot([q[1] for q in p], [-q[0] for q in p], color=col, lw=(0.45, 0.6, 1.0, 1.6)[level], ls=(0, (2.2, 1.2)) if level == 1 else '-', alpha=(0.3, 0.6, 0.8, 1.0)[level], zorder=1 + level)
    tx = {tuple(x['target']) for x in eps}
    ax.scatter([t[1] for t in tx], [-t[0] for t in tx], s=4, marker='x', color='black', lw=0.5, zorder=6)
    if pi == 0:
        ax.plot([-0.395, -0.295], [-0.815, -0.815], color='black', lw=0.8, solid_capstyle='butt', zorder=7)
        ax.text(-0.345, -0.805, '10 cm', ha='center', va='bottom', fontsize=FS_SMALL - 0.5, zorder=7)
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color('#9a9a9a')
    ax = axes_in(x0 + 0.06, BAR_Y, COL_W - 0.06, BAR_H)
    per_seed = [D.stage_counts(r) for r in c['runs'].values()]
    for (si, (label, col)) in enumerate(STAGES):
        vals = [p[1 + si] for p in per_seed]
        m = st.mean(vals)
        y = len(STAGES) - 1 - si
        ax.barh(y, m, 0.72, color=col, edgecolor='white', linewidth=0, hatch=HATCH[si])
        if max(vals) > 0:
            ax.errorbar(m, y, xerr=[[m - min(vals)], [max(vals) - m]], fmt='none', ecolor='#333333', elinewidth=0.6, capsize=1.2, capthick=0.6)
        ax.text(max(vals) + 2, y, '0' if m == 0 else f'{m:.1f}', ha='left', va='center', fontsize=FS_SMALL)
    ax.set_xlim(0, 82)
    ax.set_ylim(-0.55, len(STAGES) - 0.45)
    ax.spines['bottom'].set_bounds(0, 70)
    ax.set_xticks([0, 35, 70])
    ax.tick_params(axis='x', labelsize=FS_SMALL, length=2, pad=1)
    ax.set_yticks(range(len(STAGES)))
    ax.set_yticklabels([s[0] for s in reversed(STAGES)] if pi == 0 else [], fontsize=FS_SMALL)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
fig.text((COL_X0 + 1.5 * COL_W + COL_GAP) / W, 0.0, 'radial test episodes (of 70)', fontsize=FS_SMALL, ha='center', va='bottom')
save_fig(fig, 'fig1_hero')

import statistics as st
from matplotlib.lines import Line2D
from paper_plot_style import plt, OI, FIG_DIR
import paper_data as D

def band_counts(run):
    """{band_cm: (episodes, approach >= 0.25, strict lifts)} over one run's radial test episodes."""
    b = {}
    for x in D.scored(run):
        (n, a, s) = b.get(x['band'], (0, 0, 0))
        b[x['band']] = (n + 1, a + (x['approach'] >= D.APPROACH_STAGE), s + bool(x['success']))
    return dict(sorted(b.items()))

def series(cell, k):
    """Bands and the mean / min / max over seeds of outcome k (1 = approach, 2 = lift), in % of the band's episodes."""
    per = [band_counts(r) for r in cell['runs'].values()]
    bands = sorted(per[0])
    assert all((p[b][0] == 14 for p in per for b in bands)), 'expected 14 radial episodes per band and seed'
    vals = [[100 * p[b][k] / p[b][0] for p in per] for b in bands]
    return (bands, [st.mean(v) for v in vals], [min(v) for v in vals], [max(v) for v in vals])

def cam_series(arm, k):
    """Camera rows (EEF, collection B): per camera input, the mean / min / max over seeds of outcome k in % of 70."""
    out = []
    for cam in CAMS:
        c = next((x for x in D.MANIFEST['cells'] if x['role'] == 'camera panel' and x['arm_command'] == arm and (x['cameras'] == cam)))
        v = [100 * D.stage_counts(r)[k + 1] / D.stage_counts(r)[0] for r in c['runs'].values()]
        out.append((st.mean(v), min(v), max(v)))
    return ([o[0] for o in out], [o[1] for o in out], [o[2] for o in out])
CAMS = ['head', 'head+wrist', 'wrist']
(W, H) = (5.5, 1.36)
(fig, axes) = plt.subplots(1, 3, figsize=(W, H), sharey=True, gridspec_kw={'width_ratios': [1, 1, 0.62]})
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.215, top=0.755, wspace=0.09)
ABS = '#555555'
STY = {1: (OI['orange'], 'o', -0.35), 2: (OI['vermillion'], 's', 0.35)}
for (ax, frame) in zip(axes, ['joint', 'EEF']):
    ab = D.cell(frame, 'absolute', 'absolute', 'head')
    assert all((v == 0 for k in (1, 2) for v in series(ab, k)[3])), 'absolute-arm cells are not all zero'
    ax.axhline(0, color=ABS, lw=1.3, ls=(0, (3.5, 2)), zorder=1)
    ax.plot(series(ab, 1)[0], series(ab, 1)[1], color=ABS, lw=0, marker='o', ms=4.2, mfc='white', mew=0.9, zorder=3)
    res = D.cell(frame, 'residual', 'absolute', 'head')
    for (k, (col, mk, dx)) in STY.items():
        (x, m, lo, hi) = series(res, k)
        xs = [v + dx for v in x]
        ax.errorbar(xs, m, yerr=[[a - b for (a, b) in zip(m, lo)], [b - a for (a, b) in zip(m, hi)]], fmt='none', ecolor=col, elinewidth=1.0, capsize=2.0, capthick=1.0, zorder=2)
        ax.plot(xs, m, color=col, lw=2.2 if k == 1 else 1.8, marker=mk, ms=5.0, mec='white', mew=0.5, zorder=3)
    ax.set_title(f'{frame} frame, collection A', fontsize=7.5, pad=2.5)
    ax.set_xticks(x)
    ax.set_xlim(3, 27)
    ax.set_ylim(-6, 106)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
ax = axes[2]
ab_lift = cam_series('absolute', 2)
assert all((v == 0 for v in ab_lift[2])), 'absolute EEF camera rows are expected to lift in none'
ax.axhline(0, color=ABS, lw=1.3, ls=(0, (3.5, 2)), zorder=1)
(m, lo, hi) = cam_series('absolute', 1)
ax.errorbar(range(3), m, yerr=[[a - b for (a, b) in zip(m, lo)], [b - a for (a, b) in zip(m, hi)]], fmt='none', ecolor=ABS, elinewidth=0.8, capsize=1.5, zorder=2)
ax.plot(range(3), m, color=ABS, lw=0, marker='o', ms=4.2, mfc='white', mew=0.9, zorder=3)
for (k, (col, mk, dx)) in STY.items():
    (m, lo, hi) = cam_series('residual', k)
    xs = [i + 0.12 * dx / 0.35 for i in range(3)]
    ax.errorbar(xs, m, yerr=[[a - b for (a, b) in zip(m, lo)], [b - a for (a, b) in zip(m, hi)]], fmt='none', ecolor=col, elinewidth=1.0, capsize=2.0, capthick=1.0, zorder=2)
    ax.plot(xs, m, color=col, lw=0, marker=mk, ms=4.7, mec='white', mew=0.5, zorder=3)
ax.set_title('EEF frame, cameras, collection B', fontsize=7.5, pad=2.5)
ax.set_xticks(range(3))
ax.set_xticklabels(['head', 'head+\nwrist', 'wrist'])
ax.set_xlim(-0.5, 2.5)
ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
axes[0].set_ylabel('radial episodes (%)', fontsize=7.5, labelpad=2)
fig.text(0.4, 0.02, 'distance beyond the demonstrated envelope (cm)', ha='center', va='bottom', fontsize=7.5)
handles = [Line2D([], [], color=ABS, lw=1.3, ls=(0, (3.5, 2))), Line2D([], [], color=ABS, lw=0, marker='o', ms=4.2, mfc='white', mew=0.9), Line2D([], [], color=OI['orange'], lw=1.8, marker='o', ms=4.7, mec='white', mew=0.5), Line2D([], [], color=OI['vermillion'], lw=1.8, marker='s', ms=4.7, mec='white', mew=0.5)]
labels = ['absolute arm: strict completion (0%)', 'absolute arm: approach$\\,\\geq\\,$0.25', 'residual arm: approach$\\,\\geq\\,$0.25', 'residual arm: strict completion']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7.4, handlelength=1.8, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)
fig.savefig(FIG_DIR / 'fig2_approach.pdf')
fig.savefig(FIG_DIR / 'fig2_approach.png', dpi=220)
for frame in ['joint', 'EEF']:
    c = D.cell(frame, 'residual', 'absolute', 'head')
    print(frame, 'residual approach %', [round(v, 1) for v in series(c, 1)[1]], 'lift %', [round(v, 1) for v in series(c, 2)[1]])
print('Saved:', FIG_DIR / 'fig2_approach.pdf')

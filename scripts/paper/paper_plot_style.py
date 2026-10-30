"""Shared matplotlib style for every paper figure (publication preset, vector PDF)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
FONT_SIZE = 9
matplotlib.rcParams.update({'font.size': FONT_SIZE, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1, 'legend.fontsize': FONT_SIZE - 1, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
OI = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999'}
FIG_DIR = Path(os.environ.get('ESCAPE_FIGURES_DIR', Path(__file__).resolve().parents[2] / 'outputs/figures'))
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    """Vector PDF for the paper + a PNG preview for render-then-verify."""
    fig.savefig(FIG_DIR / f'{name}.pdf')
    fig.savefig(FIG_DIR / f'{name}.png', dpi=200)
    print(f'Saved: {FIG_DIR / name}.pdf (+ .png preview)')

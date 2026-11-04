"""Shared, print-sized style for the MIA manuscript's scientific figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1 / 25.4
FULL_WIDTH = 182 * MM
SINGLE_WIDTH = 88 * MM
COLORS = {
    'NoJEPA': '#486581',
    'DirectReadout': '#D58A36',
    'ForecastAttention': '#247B73',
    'ForecastMean': '#8B6CA8',
    'direct_no_jepa': '#486581',
    'direct_readout': '#D58A36',
    'forecast_attention': '#247B73',
    'forecast_mean': '#8B6CA8',
}
TEXT = '#172B3A'
MUTED = '#687B89'
GRID = '#E3E9ED'


def apply_style():
    """Apply at final physical size; do not shrink text after plotting."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'axes.titleweight': 'bold',
        'axes.labelcolor': TEXT,
        'text.color': TEXT,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': .65,
        'axes.edgecolor': MUTED,
        'grid.color': GRID,
        'grid.linewidth': .45,
        'lines.linewidth': 1.35,
        'lines.markersize': 3.5,
        'xtick.major.width': .6,
        'ytick.major.width': .6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'mathtext.fontset': 'dejavusans',
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def panel_label(ax, label, x=-.13, y=1.10):
    return ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
                   weight='bold', va='top', ha='left', color=TEXT,
                   clip_on=False)


def save_figure(fig, path, png=True, dpi=600, tight=True):
    """Save vector PDF and optional 600-dpi PNG at the declared figure size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {'bbox_inches': 'tight', 'pad_inches': .035} if tight else {}
    fig.savefig(path.with_suffix('.pdf'), **kw)
    if png:
        fig.savefig(path.with_suffix('.png'), dpi=dpi, **kw)


apply_style()

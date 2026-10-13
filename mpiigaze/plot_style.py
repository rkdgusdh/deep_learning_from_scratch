# coding: utf-8
"""Shared matplotlib style for the notebooks (one palette, recessive axes and grid)."""
import matplotlib as mpl

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'   # categorical slots 1-3, in fixed order
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    'seq_blue', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
TEXT, TEXT2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def apply():
    mpl.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 130, 'savefig.bbox': 'tight',
        'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
        'axes.edgecolor': GRID, 'axes.labelcolor': TEXT2, 'axes.titlecolor': TEXT,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'xtick.color': TEXT2, 'ytick.color': TEXT2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.frameon': False, 'legend.fontsize': 8,
        'axes.prop_cycle': mpl.cycler(color=[BLUE, ORANGE, AQUA]),
        'lines.linewidth': 2, 'patch.linewidth': 0,
    })

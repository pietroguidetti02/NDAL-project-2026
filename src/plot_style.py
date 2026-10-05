"""Shared paper-style settings so every figure in the manuscript uses the same fonts, sizes and encodings."""
import os
import matplotlib.pyplot as plt

# Figures are drawn at these sizes and scaled down by LaTeX: a single-column figure (6.0 in) placed at
# \columnwidth (3.5 in) and a double-column one (12.0 in) at \textwidth (7.16 in) both render 14 pt text at ~8 pt.
FIG_COLUMN = (6.0, 3.8)
FIG_COLUMN_TALL = (6.0, 5.2)
FIG_DOUBLE = (12.0, 4.0)

# Colorblind-validated categorical slots; identity is always doubled by a marker so figures survive B&W print.
MODEL_STYLE = {
    'XGBoost': {'color': '#2a78d6', 'marker': 'o'},
    'MLP':     {'color': '#eb6834', 'marker': 's'},
    'LSTM':    {'color': '#1baf7a', 'marker': '^'},
}

# Training scope is encoded with line style / hatch, never with a new hue, so it composes with MODEL_STYLE.
SCOPE_STYLE = {
    'Centralized': {'linestyle': '-',  'hatch': ''},
    'Federated':   {'linestyle': '--', 'hatch': '///'},
    'Local':       {'linestyle': ':',  'hatch': '...'},
}

# Link types use hues disjoint from the model slots so "blue" never means two things in the paper.
LINK_STYLE = {
    'Fiber':  {'color': '#4a3aa7', 'linestyle': '-',  'marker': 'D'},
    'Mobile': {'color': '#e34948', 'linestyle': '--', 'marker': 'v'},
}

NEUTRAL = {
    'ink': '#0b0b0b',
    'secondary': '#52514e',
    'muted': '#898781',
    'grid': '#e1e0d9',
    'idle': '#c3c2b7',
    'network': '#52514e',
}

MODEL_ALIASES = {'NN': 'MLP', 'MLP_NN': 'MLP', 'XGB': 'XGBoost'}


def canonical_model(name):
    return MODEL_ALIASES.get(name, name)


def apply_paper_style():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # IEEE PDF eXpress rejects Type 3 fonts, which is matplotlib's default for PDF/PS output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.edgecolor": NEUTRAL['secondary'],
        "axes.grid": True,
        "grid.color": NEUTRAL['grid'],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "hatch.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def line_kwargs(model, scope='Centralized', markevery=None):
    style = MODEL_STYLE[canonical_model(model)]
    kwargs = {'color': style['color'], 'marker': style['marker'],
              'linestyle': SCOPE_STYLE[scope]['linestyle'],
              'markerfacecolor': 'white', 'markeredgewidth': 1.4}
    if markevery is not None:
        kwargs['markevery'] = markevery
    return kwargs


def bar_kwargs(model, scope='Centralized'):
    return {'color': MODEL_STYLE[canonical_model(model)]['color'],
            'hatch': SCOPE_STYLE[scope]['hatch'],
            'edgecolor': 'white', 'linewidth': 1.0}


def scope_legend_handles(scopes, extra=()):
    """Neutral-colored legend swatches for scopes, so one legend serves panels drawn in different model colors."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=NEUTRAL['muted'], edgecolor='white', hatch=SCOPE_STYLE[s]['hatch'], label=label)
               for s, label in scopes]
    handles += [Patch(facecolor=color, edgecolor='white', label=label) for color, label in extra]
    return handles


def save_figure(fig, path_without_ext, formats=('pdf', 'png')):
    """Saves a figure as vector PDF (for the paper) and PNG (for quick preview)."""
    os.makedirs(os.path.dirname(path_without_ext) or '.', exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{path_without_ext}.{fmt}")
    plt.close(fig)

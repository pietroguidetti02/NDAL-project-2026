"""Regenerates the manuscript figures from the saved experiment outputs (no retraining)."""
import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, auc

sys.path.append(os.getcwd())
from src.preprocessor import clean_data
from src.federated import communication_time_s
from src.plot_style import (apply_paper_style, save_figure, line_kwargs, bar_kwargs, scope_legend_handles,
                            FIG_COLUMN, FIG_COLUMN_TALL, FIG_DOUBLE, MODEL_STYLE, SCOPE_STYLE,
                            LINK_STYLE, NEUTRAL)

# All paper results live in results/paper/ (see results/paper/README.md for provenance).
PAPER = 'results/paper'
SWEEPS = {
    'S1': ('Single dataset', f'{PAPER}/accuracy/sweep_S1_1hz'),
    'S2': ('Spatial split', f'{PAPER}/accuracy/sweep_S2_1hz'),
    'S3': ('Temporal split', f'{PAPER}/accuracy/sweep_S3_1hz'),
}
SWEEP_MODELS = {'XGBoost': 'XGBoost', 'NN': 'MLP', 'LSTM': 'LSTM'}
FL_SEED_DIRS = sorted(glob.glob(f'{PAPER}/accuracy/fl_seed*'))
# Training-time figure: seed 1 ran alone on the server, so its timings are not contaminated by other jobs.
FL_DIR = f'{PAPER}/accuracy/fl_seed1'
_PI = f'{PAPER}/latency/rpi3b_30000'
PI_RUNS = {'MLP': _PI, 'LSTM': _PI}
# Same 08 script and settings on the x86 server, drawn as reference ticks on the Pi figure.
X86_MACHINES = [
    ('x86 server, Intel i9', {'MLP': f'{PAPER}/latency/i9_30000', 'LSTM': f'{PAPER}/latency/i9_30000'}),
]


def load_link(window, link, tunnel):
    df = clean_data(pd.read_csv(f'dataset/{window}/{link}-{tunnel}.csv'))
    df['time'] = pd.to_datetime(df['time'])
    return df.set_index('time')


def pr_auc_from_predictions(csv_path):
    d = pd.read_csv(csv_path)
    precision, recall, _ = precision_recall_curve(d['y_true'], d['y_prob'])
    # Same trapezoidal PR-AUC as the original sweep comparison, so numbers match earlier reports.
    return auc(recall, precision)


def draw_link_timeseries(ax_d, ax_l, start='2025-03-05 18:30', end='2025-03-05 20:00', link='cpe_a-cpe_b'):
    bin_width = pd.Timedelta('1min')
    for k, (tunnel, name) in enumerate([('fiber', 'Fiber'), ('mobile', 'Mobile')]):
        df = load_link('first_capture_window', link, tunnel).loc[start:end]
        color = LINK_STYLE[name]['color']
        label = 'Mobile (4G)' if name == 'Mobile' else name
        ax_d.plot(df.index, df['delay_ms'].rolling(30, min_periods=1).median(), color=color, linewidth=1.4, label=label)
        loss_pct = df['packet_loss'].resample('1min').mean() * 100
        ax_l.bar(loss_pct.index + (k - 0.5) * bin_width * 0.45, loss_pct.values, width=bin_width * 0.45,
                 color=color, align='edge', linewidth=0)
    ax_d.set_ylabel('Delay [ms]')
    ax_d.legend(loc='upper left')
    ax_l.set_ylabel('Loss [%]')
    ax_l.set_xlabel('Time (5 Mar 2025)')
    ax_l.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


def draw_delay_cdf(ax):
    for tunnel, name in [('fiber', 'Fiber'), ('mobile', 'Mobile')]:
        delays = np.concatenate([clean_data(pd.read_csv(f))['delay_ms'].dropna().values
                                 for f in glob.glob(f'dataset/*/*-{tunnel}.csv')])
        x = np.sort(delays)
        y = np.arange(1, len(x) + 1) / len(x)
        style = LINK_STYLE[name]
        ax.plot(x, y, color=style['color'], linestyle=style['linestyle'], linewidth=1.8,
                label=f'{name} (4G)' if name == 'Mobile' else name)
    ax.set_xscale('log')
    ax.set_xlabel('Delay [ms]')
    ax.set_ylabel('ECDF')
    ax.set_ylim(0, 1.01)
    ax.legend(loc='lower right')


def fig_link_timeseries(out_dir):
    fig, (ax_d, ax_l) = plt.subplots(2, 1, sharex=True, figsize=FIG_COLUMN_TALL,
                                     gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})
    draw_link_timeseries(ax_d, ax_l)
    save_figure(fig, os.path.join(out_dir, 'fig_link_timeseries'))


def fig_delay_cdf(out_dir):
    fig, ax = plt.subplots(figsize=FIG_COLUMN)
    draw_delay_cdf(ax)
    save_figure(fig, os.path.join(out_dir, 'fig_delay_cdf'))


def fig_link_characterization(out_dir):
    """Double-column figure: (a) delay/loss time series on the left, (b) delay ECDF on the right."""
    fig = plt.figure(figsize=(FIG_DOUBLE[0], FIG_DOUBLE[1] * 1.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[2, 1], hspace=0.08, wspace=0.25)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[1, 0], sharex=ax_d)
    ax_c = fig.add_subplot(gs[:, 1])
    draw_link_timeseries(ax_d, ax_l)
    plt.setp(ax_d.get_xticklabels(), visible=False)
    draw_delay_cdf(ax_c)
    ax_c.get_legend().remove()
    ax_d.set_title('(a) Link A$\\rightarrow$B, 18:30-20:00')
    ax_c.set_title('(b) All links, both capture windows')
    save_figure(fig, os.path.join(out_dir, 'fig_link_characterization'))


DEFAULT_SCENARIO = 'S3'


def fig_horizon(out_dir, scenario=DEFAULT_SCENARIO, N=30):
    _, sweep_dir = SWEEPS[scenario]
    fig, ax = plt.subplots(figsize=FIG_COLUMN)
    for file_tag, model in SWEEP_MODELS.items():
        xs, ys = [], []
        for path in glob.glob(os.path.join(sweep_dir, f'N_{N}_X_*', f'mobile_{file_tag}_predictions.csv')):
            xs.append(int(re.search(r'N_\d+_X_(\d+)', path).group(1)))
            ys.append(pr_auc_from_predictions(path))
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], **line_kwargs(model), label=model)
    ax.set_xlabel('Prediction horizon $X$ [s]')
    ax.set_ylabel('PR-AUC')
    ax.set_xticks(sorted(set(xs)))
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    save_figure(fig, os.path.join(out_dir, 'fig_horizon'))


def fig_pr_curves(out_dir, scenario=DEFAULT_SCENARIO, N=30, X=5):
    _, sweep_dir = SWEEPS[scenario]
    run_dir = os.path.join(sweep_dir, f'N_{N}_X_{X}')
    fig, ax = plt.subplots(figsize=FIG_COLUMN)
    for file_tag, model in SWEEP_MODELS.items():
        d = pd.read_csv(os.path.join(run_dir, f'mobile_{file_tag}_predictions.csv'))
        precision, recall, _ = precision_recall_curve(d['y_true'], d['y_prob'])
        # Markers spaced along the curve keep the models apart in B&W print.
        ax.plot(recall, precision, **line_kwargs(model, markevery=0.1),
                label=f'{model} (PR-AUC {auc(recall, precision):.2f})')
        op = json.load(open(os.path.join(run_dir, f'mobile_{file_tag}_metrics_summary.json')))
        ax.plot(op['recall'], op['precision'], linestyle='none', marker=MODEL_STYLE[model]['marker'],
                color=MODEL_STYLE[model]['color'], markersize=9, markeredgecolor='white', markeredgewidth=1.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))  # drop y=0 so it does not collide with x=0 at the origin
    ax.legend(loc='upper right')
    save_figure(fig, os.path.join(out_dir, 'fig_pr_curves'))


def fig_scenarios(out_dir, N=30, X=5):
    fig, ax = plt.subplots(figsize=FIG_COLUMN)
    width = 0.26
    for k, (file_tag, model) in enumerate(SWEEP_MODELS.items()):
        paths = [os.path.join(d, f'N_{N}_X_{X}', f'mobile_{file_tag}_predictions.csv') for _, d in SWEEPS.values()]
        values = [pr_auc_from_predictions(p) if os.path.exists(p) else np.nan for p in paths]
        pos = np.arange(len(SWEEPS)) + (k - 1) * width
        ax.bar(pos, values, width, **bar_kwargs(model), label=model)
        for p, v in zip(pos, values):
            ax.text(p, (0 if np.isnan(v) else v) + 0.01, 'n/a' if np.isnan(v) else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=10, color=NEUTRAL['secondary'])
    ax.set_xticks(np.arange(len(SWEEPS)))
    ax.set_xticklabels([label for label, _ in SWEEPS.values()])
    ax.set_ylabel('PR-AUC')
    ax.set_ylim(0, 1)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', ncol=3)
    save_figure(fig, os.path.join(out_dir, 'fig_scenarios'))


def load_fl_summaries():
    """Multi-seed runs (results/exp_federated_*_seed*) when present, otherwise the single reference run."""
    seed_files = [os.path.join(d, 'federated_sweeping_summary.csv') for d in FL_SEED_DIRS]
    if seed_files:
        s = pd.concat([pd.read_csv(f) for f in seed_files], ignore_index=True)
    else:
        s = pd.read_csv(os.path.join(FL_DIR, 'federated_sweeping_summary.csv'))
        s['Seed'] = 0
    s['Scope'] = s['Type'].str.replace(r'_CPE_[A-C]', '', regex=True)
    return s


def fig_fl_f1(out_dir):
    s = load_fl_summaries()
    n_seeds = s['Seed'].nunique()
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharey=True)
    width = 0.26
    for ax, model in zip(axes, ['MLP', 'LSTM']):
        d = s[s['Model'] == model]
        n_values = sorted(d['N'].unique())
        for k, scope in enumerate(['Local', 'Federated', 'Centralized']):
            # Local is first averaged over the 3 CPEs of each seed, then every scope is summarized across seeds.
            per_seed = d[d['Scope'] == scope].groupby(['N', 'Seed'])['F1_Score'].mean().groupby('N')
            mean = per_seed.mean().reindex(n_values)
            if n_seeds > 1:
                std = per_seed.std().reindex(n_values)
                yerr = [std, std]
            elif scope == 'Local':
                g = d[d['Scope'] == 'Local'].groupby('N')['F1_Score']
                yerr = [mean - g.min().reindex(n_values), g.max().reindex(n_values) - mean]
            else:
                yerr = None
            pos = np.arange(len(n_values)) + (k - 1) * width
            ax.bar(pos, mean, width, yerr=yerr, capsize=3, error_kw={'elinewidth': 1, 'ecolor': NEUTRAL['ink']},
                   **bar_kwargs(model, scope))
        ax.set_xticks(np.arange(len(n_values)))
        ax.set_xticklabels([f'N = {n}' for n in n_values])
        ax.set_title(model)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('F1-score' + (f' (mean $\\pm$ std, {n_seeds} seeds)' if n_seeds > 1 else ''))
    axes[0].set_ylim(0, max(0.4, axes[0].get_ylim()[1], axes[1].get_ylim()[1]))
    local_label = 'Local (mean over CPEs)' if n_seeds > 1 else 'Local (mean, min-max over CPEs)'
    fig.legend(handles=scope_legend_handles([('Local', local_label),
                                             ('Federated', 'Federated'), ('Centralized', 'Centralized')]),
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    save_figure(fig, os.path.join(out_dir, 'fig_fl_f1'))


def fig_fl_times(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    cpes = ['CPE_A', 'CPE_B', 'CPE_C']
    for ax, model in zip(axes, ['MLP', 'LSTM']):
        files = sorted(glob.glob(os.path.join(FL_DIR, f'FL_{model}_N*_timing_records.csv')),
                       key=lambda p: int(p.split('_N')[1].split('_')[0]))
        x, ticks, network_s = 0, [], None
        for path in files:
            n = int(path.split('_N')[1].split('_')[0])
            t = pd.read_csv(path)
            slowest = t[cpes].max(axis=1)
            network_s = t['Network'].mean()
            for c in cpes:
                compute = t[c].mean()
                idle = (slowest - t[c]).mean()
                ax.bar(x, compute, 0.8, color=MODEL_STYLE[model]['color'], edgecolor='white',
                       label='Local compute' if x == 0 else None)
                ax.bar(x, idle, 0.8, bottom=compute, color=NEUTRAL['idle'], hatch='///', edgecolor='white',
                       label='Idle (waiting for straggler)' if x == 0 else None)
                ax.bar(x, network_s, 0.8, bottom=compute + idle, color=NEUTRAL['network'], edgecolor='white',
                       label='Network' if x == 0 else None)
                ticks.append((x, c.replace('CPE_', '')))
                x += 1
            ax.text(x - 2, -0.13, f'N = {n}', transform=ax.get_xaxis_transform(), ha='center', va='top')
            x += 0.8
        ax.set_xticks([p for p, _ in ticks])
        ax.set_xticklabels([label for _, label in ticks])
        ax.set_title(f'{model} (network: {network_s:.1f} s/round)')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Time per FedAvg round [s]')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=NEUTRAL['muted'], edgecolor='white', label='Local compute'),
                        Patch(facecolor=NEUTRAL['idle'], hatch='///', edgecolor='white', label='Idle (waiting for slowest CPE)'),
                        Patch(facecolor=NEUTRAL['network'], edgecolor='white', label='Network')],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.subplots_adjust(bottom=0.2)
    save_figure(fig, os.path.join(out_dir, 'fig_fl_times'))


def fig_pi_latency(out_dir, rtt_ms, capacity_mbps, deadline_s=1.0):
    available = {m: os.path.join(d, 'realtime_inference_results.csv') for m, d in PI_RUNS.items()
                 if os.path.exists(os.path.join(d, 'realtime_inference_results.csv'))}
    if not available:
        print('[!] No Raspberry Pi results found, skipping fig_pi_latency.')
        return
    x86, x86_label = {}, None
    for label, runs in X86_MACHINES:
        x86 = {m: pd.read_csv(os.path.join(d, 'realtime_inference_results.csv')).query('Model == @m')
               for m, d in runs.items() if os.path.exists(os.path.join(d, 'realtime_inference_results.csv'))}
        if x86:
            x86_label = label
            break
    fig, axes = plt.subplots(1, len(available), figsize=FIG_DOUBLE if len(available) > 1 else FIG_COLUMN)
    axes = np.atleast_1d(axes)
    width = 0.36
    for ax, (model, path) in zip(axes, available.items()):
        d = pd.read_csv(path)
        d = d[d['Model'] == model]
        n_values = sorted(d['N'].unique())
        for k, scope in enumerate(['Local', 'Federated']):
            g = d[d['Scope'] == scope].groupby('N')['compute_ms']
            med = g.median().reindex(n_values) / 1000
            p5, p95 = g.quantile(0.05).reindex(n_values) / 1000, g.quantile(0.95).reindex(n_values) / 1000
            pos = np.arange(len(n_values)) + (k - 0.5) * width
            ax.bar(pos, med, width, **bar_kwargs(model, scope))
            net = 0.0
            if scope == 'Federated':
                payload = d.loc[d['Scope'] == 'Federated', 'payload_bytes'].max()
                net = communication_time_s(payload, rtt_ms, capacity_mbps)
                ax.bar(pos, [net] * len(n_values), width, bottom=med, color=NEUTRAL['network'], edgecolor='white')
                ax.text(0.02, 0.97, f'Update: {payload / 1000:.0f} kB, network: {net * 1000:.0f} ms/round',
                        transform=ax.transAxes, ha='left', va='top', fontsize=12, color=NEUTRAL['secondary'])
            # Whiskers span the 5th-95th percentile of the whole step (network is a constant offset).
            ax.errorbar(pos, med + net, yerr=[med - p5, p95 - med], fmt='none', capsize=3,
                        elinewidth=1, ecolor=NEUTRAL['ink'])
            if model in x86 and not x86[model].empty:
                x86_med = x86[model][x86[model]['Scope'] == scope].groupby('N')['compute_ms'].median().reindex(n_values) / 1000
                ax.plot(pos, x86_med + net, linestyle='none', marker='_', markersize=22, markeredgewidth=2.5,
                        color=NEUTRAL['ink'], zorder=5)
        top = ax.get_ylim()[1]
        if top >= deadline_s:
            ax.axhline(deadline_s, color=NEUTRAL['ink'], linewidth=1, linestyle='-.')
            ax.text(-0.45, deadline_s, f'$X$ = {deadline_s:g} s', va='bottom', ha='left', fontsize=12)
        ax.set_ylim(0, top * 1.12)
        ax.set_xticks(np.arange(len(n_values)))
        ax.set_xticklabels([f'N = {n}' for n in n_values])
        ax.set_title(model)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Latency per step on Raspberry Pi [s]')
    handles = scope_legend_handles(
        [('Local', 'Local: features + inference + update'), ('Federated', 'Federated: slowest CPE + FedAvg')],
        extra=[(NEUTRAL['network'], 'Network (analytical)')])
    if x86:
        handles.append(Line2D([], [], linestyle='none', marker='_', markersize=16, markeredgewidth=2.5,
                              color=NEUTRAL['ink'], label=f'{x86_label}, same code (median)'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2 if x86 else 3)
    save_figure(fig, os.path.join(out_dir, 'fig_pi_latency'))


FIGURES = {
    'link_characterization': fig_link_characterization,
    'link_timeseries': fig_link_timeseries,
    'delay_cdf': fig_delay_cdf,
    'horizon': fig_horizon,
    'pr_curves': fig_pr_curves,
    'scenarios': fig_scenarios,
    'fl_f1': fig_fl_f1,
    'fl_times': fig_fl_times,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the paper figures from saved results.")
    parser.add_argument('--only', nargs='*', default=None, help=f"Subset of figures: {list(FIGURES) + ['pi_latency']}")
    parser.add_argument('--output_dir', type=str, default='results/paper/figures')
    parser.add_argument('--rtt_ms', type=float, default=150.0, help="RTT charged to each FedAvg transfer.")
    parser.add_argument('--capacity_mbps', type=float, default=10.0, help="Assumed uplink/downlink capacity.")
    args = parser.parse_args()

    apply_paper_style()
    os.makedirs(args.output_dir, exist_ok=True)
    selected = args.only or list(FIGURES) + ['pi_latency']
    for name in selected:
        print(f'[*] {name}')
        if name == 'pi_latency':
            fig_pi_latency(args.output_dir, args.rtt_ms, args.capacity_mbps)
        else:
            FIGURES[name](args.output_dir)
    print(f'[*] Figures saved in {args.output_dir}')

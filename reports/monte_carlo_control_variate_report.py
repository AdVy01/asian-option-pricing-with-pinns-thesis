import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods.monte_carlo_control_variate import AsianOptionMC

COLORS = {'plain': '#1f77b4', 'cv': '#2ca02c', 'discrete': '#d62728', 'continuous': '#2ca02c'}


def _run(S0, K, T, M, r, div, sigma, paths, option_type='call', seed=100,
         average='continuous'):
    """Price one option and return all estimates as a dict (one table row)."""
    t0 = time.perf_counter()
    opt = AsianOptionMC(option_type, S0, K, T, M, r, div, sigma, paths,
                        seed=seed, average=average)
    cv, cv_lo, cv_hi = opt.value_with_control_variate
    plain, plain_lo, plain_hi = opt.value
    return {
        'S0': S0, 'K': K, 'T': T, 'r': r, 'div': div, 'sigma': sigma,
        'M': M, 'paths': paths, 'average': average,
        'V_cv': cv, 'cv_lower': cv_lo, 'cv_upper': cv_hi, 'cv_half_width': (cv_hi - cv_lo) / 2,
        'V_plain': plain, 'plain_lower': plain_lo, 'plain_upper': plain_hi,
        'plain_half_width': (plain_hi - plain_lo) / 2,
        'seconds': time.perf_counter() - t0,
    }


def _finish(fig, title, save_path):
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')


def accuracy_comparison_in_num_of_paths(S0, K, T, M, r, div, sigma, sample_sizes,
                                        show_basic_mc=True, average='continuous',
                                        option_type='call', seed=100,
                                        title=None, save_path=None):
    """Price and 95% confidence interval as the number of paths grows.

    Left panel: estimates with their confidence bands.
    Right panel: confidence-interval half-width on log-log axes, with a
    1/sqrt(n) guide line; the vertical gap between the two lines is the
    variance reduction from the control variate.

    All sample sizes use the same seed, so each smaller sample consists of the
    first paths of the larger ones (common random numbers). The curves are then
    smooth, and differences between sample sizes are not just noise.
    """
    df = pd.DataFrame([_run(S0, K, T, M, r, div, sigma, n, option_type, seed, average)
                       for n in sample_sizes])
    n = df['paths'].to_numpy()

    fig, (ax_price, ax_width) = plt.subplots(1, 2, figsize=(13, 5))

    if show_basic_mc:
        ax_price.plot(n, df.V_plain, marker='o', color=COLORS['plain'], label='Standard MC')
        ax_price.fill_between(n, df.plain_lower, df.plain_upper, color=COLORS['plain'], alpha=0.15)
    ax_price.plot(n, df.V_cv, marker='s', linestyle='--', color=COLORS['cv'],
                  label='MC with control variate')
    ax_price.fill_between(n, df.cv_lower, df.cv_upper, color=COLORS['cv'], alpha=0.25)
    ax_price.set(xlabel='Number of paths', ylabel='Option price',
                 title='Estimate with 95% confidence interval')
    ax_price.legend()

    if show_basic_mc:
        ax_width.loglog(n, df.plain_half_width, marker='o', color=COLORS['plain'], label='Standard MC')
    ax_width.loglog(n, df.cv_half_width, marker='s', linestyle='--', color=COLORS['cv'],
                    label='MC with control variate')
    guide = df.cv_half_width.iloc[0] * np.sqrt(n[0] / n)
    ax_width.loglog(n, guide, color='grey', linestyle=':', label=r'$\propto 1/\sqrt{n}$')
    ax_width.set(xlabel='Number of paths', ylabel='CI half-width',
                 title='Width of the 95% confidence interval')
    ax_width.legend()

    for ax in (ax_price, ax_width):
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
    _finish(fig, title, save_path)
    return df, fig


def accuracy_in_time_steps(S0, K, T, r, div, sigma, num_of_paths, time_steps,
                           averages=('discrete', 'continuous'), reference=None,
                           reference_label='Continuous reference',
                           option_type='call', seed=100, title=None, save_path=None):
    """Price and 95% confidence interval as the number of time steps M grows.

    averages  : which averages to show; 'discrete' = M monitoring dates,
                'continuous' = trapezoid approximation of the continuous average.
    reference : optional exact/continuous price (e.g. from an FD solver of the
                PDE) drawn as a horizontal line.

    The confidence band only reflects Monte Carlo noise. The distance of the curve
    from the reference is the time-discretisation bias, which the band does not show.
    """
    df = pd.DataFrame([_run(S0, K, T, M, r, div, sigma, num_of_paths, option_type, seed, avg)
                       for avg in averages for M in time_steps])

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {'discrete': 'Discrete average (M dates)',
              'continuous': 'Continuous average (trapezoid)'}
    for avg in averages:
        d = df[df.average == avg]
        ax.plot(d.M, d.V_cv, marker='s', linestyle='--', color=COLORS[avg], label=labels[avg])
        ax.fill_between(d.M, d.cv_lower, d.cv_upper, color=COLORS[avg], alpha=0.25)
    if reference is not None:
        ax.axhline(reference, color='black', linewidth=1, label=reference_label)

    ax.set_xscale('log')
    ax.set(xlabel='Number of time steps M', ylabel='Option price',
           title=f'MC with control variate, {num_of_paths:,} paths, 95% CI')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    _finish(fig, title, save_path)
    return df, fig


def reference_table(S_values, sigmas, K, T, r, div, M, paths, average='continuous',
                    option_type='call', seed=100):
    """Benchmark prices on the grid S_values x sigmas (as in Park et al., Table 4).

    Returns one row per (S0, sigma) with the control-variate price, its 95%
    confidence interval and the plain MC price for comparison.
    """
    rows = [_run(S0, K, T, M, r, div, sigma, paths, option_type, seed, average)
            for S0, sigma in product(S_values, sigmas)]
    return pd.DataFrame(rows)[['S0', 'sigma', 'V_cv', 'cv_lower', 'cv_upper',
                               'cv_half_width', 'V_plain', 'plain_half_width', 'seconds']]
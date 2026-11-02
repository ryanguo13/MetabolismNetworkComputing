"""Render the completed M1 evidence tables and figures without rerunning solvers."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/m1-completion'
    read = lambda name: json.loads((out / name).read_text(encoding='utf-8'))
    summaries, net, runs = read('kecoli-summary.json'), read('kecoli-net-flux.json'), read('kecoli-runs.json')
    acceptance = read('acceptance.json')
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, marker in (('BDF', 'o'), ('Radau', 's'), ('CVODE-KLU', '^')):
        rows = [row for row in summaries if row['method'] == method]
        flux = [next(row for row in net['configs'] if row['method'] == method and row['rtol'] == s['rtol']) for s in rows]
        axes[0].loglog([s['concentration_error']['1e-05'] for s in rows], [s['median_seconds'] for s in rows], marker=marker, label=method)
        axes[1].loglog([f['net_flux_error']['0.001'] for f in flux], [s['median_seconds'] for s in rows], marker=marker, label=method)
    for ax, label in zip(axes, ('Observed concentration error / target 1e-5', 'Observed source net-flux error / target 1e-3')):
        ax.axvline(1, linestyle='--', color='gray', linewidth=1)
        ax.grid(which='both', alpha=.2)
        ax.set(xlabel=label, ylabel='Median integration + output (s)', title='k-ecoli457: 3003 states, shared Python RHS')
        ax.legend()
    figure.savefig(out / 'large-work-precision.png', dpi=180)
    plt.close(figure)
    costs, labels = [], []
    for method in ('BDF', 'Radau', 'CVODE-KLU'):
        group = [row for row in runs if row['method'] == method and row['rtol'] == 1e-8]
        total = np.median([row['integration_output_seconds'] for row in group])
        rhs = np.median([row['counts_and_costs']['rhs_seconds'] for row in group])
        jac = np.median([row['counts_and_costs']['jacobian_seconds'] for row in group])
        factor = np.median([row['counts_and_costs'].get('lu_factor_seconds', 0) for row in group])
        solve = np.median([row['counts_and_costs'].get('linear_solve_seconds', 0) for row in group])
        costs.append([rhs, jac, factor, solve, max(0, total - rhs - jac - factor - solve)])
        labels.append(method)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bottom = np.zeros(3)
    for values, label in zip(np.array(costs).T, ('RHS', 'Jacobian', 'SciPy LU factor', 'SciPy linear solve', 'Other incl. native CVODE linear algebra')):
        ax.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    ax.set(ylabel='Median measured time components (s)', title='Same rtol=1e-8; cost attribution only')
    ax.legend(fontsize=7)
    fig.savefig(out / 'cost-breakdown.png', dpi=180)
    plt.close(fig)

    lines = ['# M1 completion evidence', '', f"M1: **{acceptance['M1']}**; M2 admission: **{acceptance['M2_admission']}**.", '',
             'Raw repeated runs, reference arrays, source-executor fixture and gate decisions are stored alongside this table.', '',
             'k-ecoli457: original aerobic glucose source initialization, ACALD 2x; fixed source-coordinate window [0,1].',
             'All solvers use the same Python RHS and analytic sparse Jacobian in the same Linux runtime. Seven interleaved isolated workers per configuration after warm-up.', '',
             '| Method | rtol | Median s | IQR s | Concentration E at 1e-5 | Net flux E at 1e-3 | Max worker RSS MiB | Failed |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        f = next(row for row in net['configs'] if row['method'] == s['method'] and row['rtol'] == s['rtol'])
        lines.append(f"| {s['method']} | {s['rtol']:.0e} | {s['median_seconds']:.3f} | {s['iqr_seconds']:.3f} | {s['concentration_error']['1e-05']:.4g} | {f['net_flux_error']['0.001']:.4g} | {s['peak_rss_bytes']/2**20:.1f} | {s['failures']} |")
    lines += ['', 'E≤1 passes the stated metric on the fixed grid. Coarse configurations that fail accuracy remain in the table.',
              'Worker RSS includes inherited model/reference buffers and postprocessing; it is not incremental solver-only memory.',
              'CVODE LU factor time is not exposed separately. Its complete native work remains in total time.', '',
              '![Work precision](large-work-precision.png)', '', '![Costs](cost-breakdown.png)', '',
              '![Public MathSBML curation comparison](chassagnole-curation.png)', '',
              'See [M1 completion report](../../docs/m1-completion-report.md) for model eligibility, numerical limits and reproduction commands.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('M1 completion tables and figures rendered')


if __name__ == '__main__':
    main()

"""Plot saved local metrics; no solver runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'results/m3-research'
    rows = json.loads((out/'output-probe.json').read_text())['runs']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout='constrained')
    methods = ('strang', 'balanced', 'mri2')
    for ax, start in zip(axes, (.001, .01)):
        selected = [next(r for r in rows if r['start'] == start and r['h'] == 1e-4
                         and r['inner_rtol'] == 1e-10 and r['method'] == m) for m in methods]
        for offset, field, label, color in [(-.25, 'endpoint', 'Endpoint', '#3657a7'),
                (0, 'linear_grid', 'Linear output', '#d79d32'),
                (.25, 'hermite_grid', 'Full-RHS Hermite', '#178276')]:
            ax.bar(np.arange(3)+offset, [r[field]['net_flux'] for r in selected],
                   width=.24, label=label, color=color)
        ax.axhline(1, color='#b33a3a', linestyle='--', linewidth=1)
        ax.set_xticks(range(3), ['Strang', 'Balanced', 'MRI2'])
        ax.set_yscale('log')
        ax.set_ylim(3e-4, 100)
        ax.set_title(f't = {start:g}, H = 1e-4, inner rtol = 1e-10')
        ax.set_ylabel('Net-flux error / local budget (1e-4)')
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('M3 local probes: coupling and output errors\nDevelopment evidence only; dashed line = local budget', fontsize=12)
    fig.savefig(out/'coupling-output.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

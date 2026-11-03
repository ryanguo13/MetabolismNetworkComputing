"""Audit saved M2 experiments and render the report without new integrations."""

import argparse
import hashlib
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from .m2_experiments import OUT, ROOT, digest, error, save


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def test_source_hashes():
    paths = list((ROOT / 'tests').glob('test_*.py'))
    paths += [ROOT / 'metabolism' / name for name in ('splitting.py', 'sbml.py', 'kecoli.py', 'cvode.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(paths)}


def run_tests():
    """Record checks in the same Windows + Ubuntu runtimes used for M1/M2."""
    commands = [[sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']]
    commands += [['wsl', '-d', 'Ubuntu-24.04', '-u', 'root', '--cd', str(ROOT), '--',
                  'python3', '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v']
                 for pattern in ('test_cvode.py', 'test_kecoli.py')]
    records, logs = [], []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs += ['$ ' + ' '.join(command), result.stdout]
        records.append({'command': command, 'returncode': result.returncode, 'source_sha256_LF': test_source_hashes()})
    (OUT / 'tests.txt').write_text('\n'.join(line.rstrip() for line in '\n'.join(logs).splitlines()) + '\n',
                                 encoding='utf-8', newline='\n')
    save(OUT / 'tests.json', records)


def summarize(name):
    runs = read(f'{name}-runs.json')
    registration = read(f'{name}-registration.json')
    rows = []
    for tol in registration['inner_rtol']:
        for count in registration['steps']:
            for method in ('Lie', 'Strang'):
                selected = [r for r in runs if r['inner_rtol'] == tol and r['steps'] == count and r['method'] == method]
                good = [r for r in selected if r['status'] == 'success']
                seconds = [r['partition_and_integration_seconds'] for r in good]
                row = {'method': method, 'steps': count, 'inner_rtol': tol,
                       'successes': len(good), 'failures': len(selected) - len(good),
                       'all_seconds': seconds, 'median_seconds': float(np.median(seconds)) if seconds else None,
                       'iqr_seconds': float(np.ptp(np.quantile(seconds, [.25, .75]))) if seconds else None}
                if good:
                    for key in ('observed_error', 'common_grid_error', 'flux_error', 'invariant_error', 'net_flux_error'):
                        if key in good[0]:
                            row[key] = good[0][key]
                    row['max_substep_invariant_error'] = max(r['stats']['max_substep_invariant_error'] for r in good)
                    row['median_fast_seconds'] = float(np.median([r['stats']['fast_seconds'] for r in good]))
                    row['median_slow_seconds'] = float(np.median([r['stats']['slow_seconds'] for r in good]))
                    row['median_ledger_seconds'] = float(np.median([r['stats']['ledger_seconds'] for r in good]))
                rows.append(row)
    data = np.load(OUT / f'{name}-trajectories.npz')
    sensitivity = []
    for method in ('Lie', 'Strang'):
        for count in registration['steps']:
            loose = data[f'{method}-{count}-1e-09']
            tight = data[f'{method}-{count}-1e-10']
            indices = np.arange(count + 1) * ((len(data['reference_times']) - 1) // count)
            ref = data['reference'][:, indices]
            subflow_change = error(loose, tight, data['scale'])
            split_error = error(tight, ref, data['scale'])
            sensitivity.append({'method': method, 'steps': count, 'subflow_change': subflow_change,
                                'split_error': split_error, 'change_over_split_error': subflow_change / max(split_error, 1e-15)})
    return registration, runs, rows, sensitivity


def audit(b0, cases):
    prior = json.loads((ROOT / 'results/m1-completion/acceptance.json').read_text())
    m1_files = {**prior['evidence_sha256'], 'results/m1-completion/case-registration.json': prior['registration_sha256']}
    checks = {'M1_admission_and_evidence_intact': prior['M2_admission'] == 'pass' and
              all(digest(ROOT / path) == sha for path, sha in m1_files.items())}
    weak = [r for r in b0['linear'] if r['fast_slow_ratio'] == 1]
    checks['weak_chain_Lie_first_and_Strang_second_order'] = len(weak) == 8 and all(
        max(abs(np.array(r['orders'][-2:]) - (1 if r['method'] == 'Lie' else 2))) < .06 for r in weak)
    checks['enzyme_orders_and_two_distinct_pools'] = len(b0['enzyme']) == 2 and all(
        max(abs(np.array(r['orders'][-2:]) - (1 if r['method'] == 'Lie' else 2))) < .06
        and max(r['two_pool_errors']) < 1e-10 and max(r['numerical_vs_exact_split']) < 1e-10 for r in b0['enzyme'])
    checks['independent_enzyme_reference_budget'] = b0['enzyme_independent_reference_max_difference'] < 1e-10
    checks['stiffness_far_and_near_equilibrium_scans_saved'] = len(b0['linear']) == 32 and all(
        np.isfinite(r['trajectory_errors']).all() and max(r['invariant_errors']) < 1e-10
        and min(r['min_raw_concentrations']) > -1e-12 for r in b0['linear'])
    checks['Robertson_zero_initial_layer_detected_without_clipping'] = len(b0['robertson']) == 8 and all(
        r['status'] == 'success' and r['invariant_error'] < 1e-10 and
        r['stats']['significant_negative_rhs_components'] == 0 and r['stats']['clipping_count'] == 0 for r in b0['robertson'])
    negative = b0['negative_detection']
    checks['explicit_negative_failure_path'] = (negative['detected'] and
        negative['stats']['significant_negative_rhs_components'] > 0 and negative['stats']['clipping_count'] == 0)
    for name, (registration, runs, rows, sensitivity) in cases.items():
        prefix = name + '_'
        model_path = ROOT / ('models/k-ecoli457.zip' if name == 'kecoli' else 'models/BIOMD0000000064.xml')
        checks[prefix + 'model_and_reference_budget'] = (digest(model_path) == registration['model_sha256'] and
            max(registration['reference_checks_at_strictest_targets'].values()) < .05)
        checks[prefix + 'seven_interleaved_repeats_and_warmups'] = (len(runs) == 84 and len(rows) == 12 and
            len(read(f'{name}-warmups.json')) == 12 and all(r['successes'] == 7 and r['failures'] == 0 for r in rows)
            and all(sorted(r['repeat'] for r in runs if r['method'] == row['method'] and r['steps'] == row['steps']
                           and r['inner_rtol'] == row['inner_rtol']) == list(range(7)) for row in rows))
        checks[prefix + 'substeps_invariants_and_raw_domain'] = all(
            r['status'] == 'success' and r['invariant_error'] < 1e-10 and
            r['stats']['max_substep_invariant_error'] < 1e-10 and r['stats']['significant_negative_rhs_components'] == 0 and
            r['stats']['clipping_count'] == r['stats']['projection_count'] == r['stats']['fallback_count'] == 0 for r in runs)
        checks[prefix + 'common_grid_errors_decrease_with_macro_refinement'] = all(
            all(a > b for a, b in zip(values[:-1], values[1:]))
            for values in [[r['common_grid_error']['0.001'] for r in rows if r['inner_rtol'] == tol and r['method'] == method]
                           for tol in registration['inner_rtol'] for method in ('Lie', 'Strang')])
        checks[prefix + 'subflow_tightening_separates_splitting_error'] = max(r['change_over_split_error'] for r in sensitivity) < .05
    teusink = cases['teusink'][1]
    # The normalization is fixed by the carbon weights and the original x0.
    from .sbml import SBMLModel
    from .kecoli import KEcoliModel
    small = SBMLModel(ROOT / 'models/BIOMD0000000064.xml')
    carbon = np.array([6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 2, 0, 21, 21])
    normalizer = max(1., float(carbon @ small.y0))
    checks['open_ledger_with_independent_quadrature'] = all(abs(r['stats']['ledger_residual']) / normalizer < 1e-10
        and r['stats']['ledger_quadrature_difference'] / normalizer < 1e-10 for r in teusink)
    large = KEcoliModel(ROOT / 'models/k-ecoli457.zip')
    for name, model, prior_path in [('teusink', small, 'results/m1/BIOMD0000000064-reference.npz'),
                                     ('kecoli', large, 'results/m1-completion/kecoli-reference.npz')]:
        reg, runs, _, _ = cases[name]
        data = np.load(OUT / f'{name}-trajectories.npz')
        prior_data = np.load(ROOT / prior_path)
        valid = np.array_equal(data['scale'], prior_data['scale'])
        f_ref = np.column_stack([model.flux(t, y) for t, y in zip(data['reference_times'], data['reference'].T)])
        f_cross = np.column_stack([model.flux(t, y) for t, y in zip(data['reference_times'], data['cross_reference'].T)])
        target = min(reg['concentration_and_elementary_flux_targets'])
        valid = valid and error(data['cross_reference'], data['reference'], data['scale'], target) < .05
        valid = valid and error(f_cross, f_ref, data['flux_scale'], target) < .05
        for row in [r for r in runs if r['repeat'] == 0 and r['status'] == 'success']:
            states = data[f"{row['method']}-{row['steps']}-{row['inner_rtol']}"]
            indices = np.arange(row['steps'] + 1) * ((len(data['reference_times']) - 1) // row['steps'])
            f = np.column_stack([model.flux(0., y) for y in states.T])
            for r in reg['concentration_and_elementary_flux_targets']:
                valid = valid and np.isclose(error(states, data['reference'][:, indices], data['scale'], r),
                                             row['observed_error'][str(r)], rtol=1e-10, atol=1e-10)
                valid = valid and np.isclose(error(f, f_ref[:, indices], data['flux_scale'], r),
                                             row['flux_error'][str(r)], rtol=1e-10, atol=1e-10)
        checks[name + '_saved_trajectories_recompute_metrics'] = bool(valid)
    mapping = large.data['network_data']['rxn_f_bInd_rxnInd'].astype(int)
    registered = cases['kecoli'][0]
    all_ids = np.unique(mapping)
    expected = np.concatenate([np.flatnonzero(mapping == all_ids[i]) for i in registered['fast_groups']]).tolist()
    checks['whole_reversible_pairs_and_reduced_fast_coordinates'] = (
        expected == registered['fast_directional_columns'] and
        all(r['stats']['fast_columns'] == expected and len(r['stats']['fast_updated_states']) == 267 for r in cases['kecoli'][1]) and
        all(r['stats']['fast_columns'] == [6, 7] and len(r['stats']['fast_updated_states']) == 6 for r in teusink))
    tests = read('tests.json')
    checks['automated_checks_pass_in_registered_runtimes'] = len(tests) == 3 and all(
        r['returncode'] == 0 and r['source_sha256_LF'] == test_source_hashes() for r in tests)
    return checks


def plots(b0, cases):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for method, marker in [('Lie', 'o'), ('Strang', 's')]:
        row = next(r for r in b0['linear'] if r['case'] == 'reversible_chain' and r['fast_slow_ratio'] == 1
                   and r['initial'] == 'far' and r['method'] == method)
        axes[0, 0].loglog(1 / np.array(row['steps']), row['trajectory_errors'], marker=marker, label=method)
        row = next(r for r in b0['enzyme'] if r['method'] == method)
        axes[0, 1].loglog(1 / np.array(row['steps']), row['exact_flow_errors'], marker=marker, label=method)
    for ratio in (1., 1e2, 1e4, 1e6):
        row = next(r for r in b0['linear'] if r['case'] == 'reversible_chain' and r['fast_slow_ratio'] == ratio
                   and r['initial'] == 'far' and r['method'] == 'Strang')
        axes[1, 0].loglog(1 / np.array(row['steps']), row['trajectory_errors'], 'o-', label=f'fast multiplier {ratio:g}')
    for name, (reg, _, rows, _) in cases.items():
        for method, style in [('Lie', '--'), ('Strang', '-')]:
            selected = [r for r in rows if r['method'] == method and r['inner_rtol'] == 1e-10]
            axes[1, 1].loglog([reg['window'][1] / r['steps'] for r in selected],
                              [r['common_grid_error']['0.001'] for r in selected], marker='o', linestyle=style,
                              label=f'{name} {method}')
    for ax, title in zip(axes.ravel(), ['Linear chain: weak stiffness', 'Explicit enzyme mechanism',
                                        'Strang: stiff order reduction', 'Real models: bounded short windows']):
        ax.set(title=title, xlabel='Macro step (model time coordinate)',
               ylabel='Scaled trajectory error' if ax is not axes[1, 1] else 'Common-grid error / target 1e-3')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=.2)
    axes[1, 1].set_xticks([3.125e-6, 1.25e-5, 6.25e-5], ['3.125e-6', '1.25e-5', '6.25e-5'])
    axes[1, 1].xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(OUT / 'convergence.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (name, (_, _, rows, _)) in zip(axes, cases.items()):
        selected = [r for r in rows if r['inner_rtol'] == 1e-10]
        fast = np.array([r['median_fast_seconds'] for r in selected])
        slow = np.array([r['median_slow_seconds'] for r in selected])
        x = np.arange(len(selected))
        ax.bar(x, fast, label='Fast Radau (incl. ledger if used)')
        ax.bar(x, slow, bottom=fast, label='Slow RK45 (incl. ledger if used)')
        ax.plot(x, [r['median_seconds'] for r in selected], 'k.', label='Total incl. partition setup')
        ax.set_xticks(x, [f"{r['method']}\n{r['steps']} steps" for r in selected], fontsize=8)
        ax.set(title=name + ' (own runtime; short window)', ylabel='Median seconds; 7 interleaved repeats')
        ax.legend(fontsize=7)
    fig.savefig(OUT / 'restart-cost.png', dpi=180)
    plt.close(fig)


def report(b0, cases, checks):
    passed = all(checks.values())
    lines = ['# M2 固定无 PE 分裂验收', '', '日期：2026-11-03（Asia/Shanghai）。', '',
             f"**M2 {'通过' if passed else '未通过'}：{sum(checks.values())}/{len(checks)} 项保存证据检查通过。** "
             '实现固定 Lie（快整步、慢整步）和 Strang（快半步、慢整步、快半步）；快流 Radau、慢流 RK45。没有 PE、QSSA、动态分组、外层自适应、投影或浓度裁剪。', '',
             '## 小系统正确性', '',
             '线性 `A ⇌ B → C` 与 `A ⇌ B ⇌ C` 使用独立完整矩阵指数作参考，子流为精确可逆对公式。'
             '扫描快反应速率的名义倍率 1、10²、10⁴、10⁶，快对速率为 `(3,2)×倍率`，慢对为 `(0.4,0.2)` 或 `(0.4,0)`。'
             '准确的孤立组非零模态比另存于 JSON；名义倍率不冒称完整 Jacobian 刚度比。'
             '远离／处于初始快平衡两类初值，每种执行 8、16、32、64、128 个宏步。'
             '弱刚性最后两档 Lie 阶数接近 1、Strang 接近 2。', '',
             '| 小系统 | 方法 | 最后两档观测阶数 |', '|---|---|---|']
    for row in b0['linear']:
        if row['fast_slow_ratio'] == 1 and row['initial'] == 'far':
            lines.append(f"| {row['case']} | {row['method']} | {row['orders'][-2]:.4f}, {row['orders'][-1]:.4f} |")
    for row in b0['enzyme']:
        lines.append(f"| E+S ⇌ ES → E+P | {row['method']} | {row['orders'][-2]:.4f}, {row['orders'][-1]:.4f} |")
    largest_pair_difference = max(max(r['numerical_vs_exact_split']) for r in b0['enzyme'])
    largest_pool = max(max(r['two_pool_errors']) for r in b0['enzyme'])
    lines += ['', f'酶结合流用 Riccati 方程的精确解、催化流用指数解；没有消去 ES。数值子流组合与精确子流组合最大差 '
              f'`{largest_pair_difference:.3g}`，两个独立守恒池最大绝对误差 `{largest_pool:.3g}`。完整 ODE 的 Radau／DOP853 参考差 '
              f"`{b0['enzyme_independent_reference_max_difference']:.3g}`。", '',
              'Robertson 初始层从 `[1,0,0]` 开始，8 次固定分裂均保留原不可逆反应；步长细化误差下降。'
              '特意构造的越出非负域流触发明确失败，未经裁剪的负值及失败原因保存在 `b0.json`。', '',
              '## 刚性边界', '',
              '高快慢倍率下 Strang 在当前网格上不能维持二阶。例如可逆链、远平衡初值、倍率 10⁶ 的最后观测阶约为 '
              f"{next(r for r in b0['linear'] if r['case']=='reversible_chain' and r['fast_slow_ratio']==1e6 and r['initial']=='far' and r['method']=='Strang')['orders'][-1]:.3f}。"
              '倍率 10² 的跨初始层网格还出现非单调误差。这些是保留的负结果；不能据 Strang 名义阶数直接采用 M3 的除以 3 估计器。', '',
              '## 真实模型短窗探针', '',
              '使用原始模型初值、M1 冻结的浓度／通量尺度；分组在整条轨迹固定。只缩减化学计量列不更新的坐标，'
              '速率仍读取完整状态；大模型使用原始正反向对和解析稀疏 Jacobian。'
              '短窗属于 M1 登记窗口的子集，**不代表已通过 M1 全窗口的浓度或通量精度门**。没有对子流路径作任意 Strang 插值。', '',
              '| 模型 | 窗口 | 快更新坐标 | 最严格新参考交叉差（目标单位） |', '|---|---|---:|---|']
    for name, (reg, _, _, _) in cases.items():
        values = ', '.join(f'{k}: {v:.3g}' for k, v in reg['reference_checks_at_strictest_targets'].items())
        lines.append(f"| {name} | 0–{reg['window'][1]:g} {reg['time_unit']} | {267 if name=='kecoli' else 6} | {values} |")
    lines += ['', '下表为较紧子流容差 `rtol=10^-10`。E 为各自全部宏步端点的误差；另存所有方法共用粗网格的误差，'
              '用于公平的细化趋势检查。每配置预热一次、正式交错重复七次，保存全部成功和失败记录。', '',
              '| 模型 | 方法 | 宏步数 | 浓度 E(10^-3) | 基元／净反应速率 E(10^-3) | 总体净通量 E(10^-3) | 中位秒（IQR） |',
              '|---|---|---:|---:|---:|---:|---:|']
    for name, (_, _, rows, _) in cases.items():
        for row in rows:
            if row['inner_rtol'] != 1e-10:
                continue
            net = f"{row['net_flux_error']['0.001']:.4g}" if 'net_flux_error' in row else '—'
            lines.append(f"| {name} | {row['method']} | {row['steps']} | {row['observed_error']['0.001']:.4g} | "
                         f"{row['flux_error']['0.001']:.4g} | {net} | {row['median_seconds']:.3f} ({row['iqr_seconds']:.3f}) |")
    large_rows = cases['kecoli'][2]
    lines += ['', f"大模型本次扫描的最好浓度 E(10^-3) 为 `{min(r['observed_error']['0.001'] for r in large_rows):.4g}`，"
              f"最好总体净通量 E(10^-3) 为 `{min(r['net_flux_error']['0.001'] for r in large_rows):.4g}`。"
              '这些数字直接决定相应配置是否达标；求解器成功不等于精度通过。'
              'M2 的继续条件是弱刚性基础阶数、正确收支及负值检测，本次真实模型探针不承担 M3 的完整误差控制验收。']
    sensitivity = max(r['change_over_split_error'] for _, _, _, sensitivity in cases.values() for r in sensitivity)
    invariant = max(r['stats']['max_substep_invariant_error'] for _, runs, _, _ in cases.values() for r in runs)
    lines += ['', f'子流容差 `10^-9 → 10^-10` 引起的浓度轨迹变化，最大仅为对应分裂误差的 `{sensitivity:.3g}`。'
              f'所有正式运行的接受子步／子流末态守恒误差最大 `{invariant:.3g}`。'
              '监测 RHS 参数（含 Newton／有限差分探针）、接受子步、子流末态和宏步；显著负值直接报错，'
              '未声称能观测求解器内部未传给 RHS 的全部线性代数状态。', '',
              'Teusink 的 Strang 误差在当前强刚性初始层没有呈现干净的二阶趋势：'
              '全部宏步端点的浓度误差可能随细化反而上升，因为更细的网格采到了更早的初始层；'
              '共用粗网格误差也趋于停滞。不能用粗网格的单调性掩盖这一现象。', '',
              'Teusink 对编码反应列做碳权重浓度记账，独立 4／8 点 Gauss 积分核验开放贡献；'
              '高能磷酸 P 权重为零，NAD／NADH 权重相同。该记账不是完整培养体系物理总质量。'
              '大模型检查 456 个真正酶池，未加入 Biomass_Bulk 伪池。', '',
              '## 成本与下一阶段边界', '',
              '每轨迹计时包含分区矩阵构建、子求解器启动、RHS、Jacobian、分解、接受子步诊断和开放收支积分；'
              '模型读取与一次性固定分组另列 setup。参考生成、误差后处理和绘图离线执行。'
              '快／慢时间已包含收支审计，ledger_seconds 是其中的子集，不能重复相加。'
              '子求解器失败前的 RHS 调用仍计数；nfev/njev/nlu 仅累计已返回的子求解器结果。'
              '没有取得每轨迹进程峰值内存，本阶段不比较内存效率。', '',
              'Windows Teusink 与 WSL 大模型时间不能交叉比较；本阶段也未对相同短窗重测成熟基线，'
              '因此不计算加速倍数。反复重启的成本已经可见。', '',
              '**M3 可继续做：**冻结／动态分组消融、滞回、外层误差控制及原子回滚；'
              '先校准强耦合、切换与初始层的估计器，必要时缩步或完整系统回退。'
              '先测试完整登记窗口，再谈达标效率。M3 尚未实现，PE 仍不准入。', '',
              '## 复现', '', '```powershell',
              'uv run python -m unittest discover -s tests -v',
              'uv run python -m metabolism.m2_experiments --case b0',
              'uv run python -m metabolism.m2_experiments --case teusink', '```', '',
              'Ubuntu／WSL（沿用 M1 环境与软件包清单）：', '', '```bash',
              'python3 -m metabolism.m2_experiments --case kecoli', '```', '',
              '最后回到 Windows：', '', '```powershell', 'uv run python -m metabolism.m2_report --run-tests',
              'uv run python -m metabolism.m2_report --check', '```', '',
              '`--check` 复核保存证据及指纹，失败返回非零退出码；不会运行数值求解器。'
              '测试额外覆盖保留完整调控依赖、共享物种、非单位体积适配、拒绝非自治 Chassagnole 和非法反应分组。', '',
              '![收敛与阶降低](../results/m2/convergence.png)', '',
              '![子流与重启成本](../results/m2/restart-cost.png)', '']
    (ROOT / 'docs/m2-completion-report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check', action='store_true', help='Validate saved evidence and hashes, without rerendering')
    parser.add_argument('--run-tests', action='store_true', help='Record Windows + WSL checks before rendering')
    args = parser.parse_args()
    if args.check and args.run_tests:
        parser.error('--check does not run test processes')
    if args.run_tests:
        run_tests()
    b0 = read('b0.json')
    cases = {name: summarize(name) for name in ('teusink', 'kecoli')}
    checks = audit(b0, cases)
    if args.check:
        registered = read('acceptance.json')
        checks['saved_evidence_sha256'] = all(digest(ROOT / path) == sha for path, sha in registered['evidence_sha256'].items())
        checks['saved_check_results_match_recomputed'] = registered['checks'] == {k: v for k, v in checks.items() if not k.startswith('saved_')}
    else:
        for name, (_, _, rows, sensitivity) in cases.items():
            save(OUT / f'{name}-summary.json', {'configs': rows, 'subflow_tolerance_sensitivity': sensitivity})
        plots(b0, cases)
        report(b0, cases, checks)
        paths = [p for p in OUT.iterdir() if p.name != 'acceptance.json']
        paths += [ROOT / 'docs/m2-completion-report.md', ROOT / 'metabolism/splitting.py',
                  ROOT / 'metabolism/m2_experiments.py', ROOT / 'metabolism/m2_report.py', ROOT / 'tests/test_splitting.py']
        save(OUT / 'acceptance.json', {'M2': 'complete' if all(checks.values()) else 'incomplete',
                                      'M3_admission': 'pass' if all(checks.values()) else 'fail',
                                      'checks': checks, 'evidence_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
                                      'limits': ['Real-model probes are bounded subwindows', 'No full-window precision qualification',
                                                 'No speedup or PE claim', 'M3 estimator must be calibrated for stiff order reduction']})
    print(json.dumps(checks, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit('M2 checks failed')


if __name__ == '__main__':
    main()

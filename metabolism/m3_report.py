"""Audit M3 saved trajectories, costs and stopping decision. No new ODE solves."""

import argparse
import hashlib
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .m3_experiments import OUT, ROOT
from .m2_experiments import digest, error, linear_chain, save


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def source_hashes():
    files = sorted((ROOT / 'tests').rglob('test_*.py'))
    files += [ROOT / 'metabolism' / p for p in ('adaptive.py', 'splitting.py', 'sbml.py', 'kecoli.py', 'cvode.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files}


def run_tests():
    commands = [[sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']]
    commands += [['wsl', '-d', 'Ubuntu-24.04', '-u', 'root', '--cd', str(ROOT), '--', 'python3', '-m',
                  'unittest', 'discover', '-s', 'tests/m3' if pattern == 'test_adaptive.py' else 'tests', '-p', pattern, '-v']
                 for pattern in ('test_adaptive.py', 'test_cvode.py', 'test_kecoli.py')]
    records, logs = [], []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        records.append({'command': command, 'returncode': result.returncode, 'source_sha256_LF': source_hashes()})
        logs += ['$ ' + ' '.join(command), result.stdout]
    save(OUT / 'tests.json', records)
    (OUT / 'tests.txt').write_text('\n'.join(line.rstrip() for line in '\n'.join(logs).splitlines()) + '\n',
                                  encoding='utf-8', newline='\n')


def summary(name):
    reg, runs = read(f'{name}-registration.json'), read(f'{name}-runs.json')
    rows = []
    for config in reg['configs']:
        selected = [r for r in runs if r['config'] == config['id']]
        good = [r for r in selected if r['status'] == 'success']
        values = [r['seconds'] for r in good]
        row = {'config': config['id'], 'target': config['target'], 'successes': len(good),
               'failures': len(selected) - len(good), 'all_seconds': values,
               'median_seconds': float(np.median(values)) if values else None,
               'iqr_seconds': float(np.ptp(np.quantile(values, [.25, .75]))) if values else None}
        if good:
            row['observed_error'] = good[0]['observed_error']
            for key in ('flux_errors', 'net_flux_errors', 'relative_open_ledger_error'):
                if key in good[0]:
                    row[key] = good[0][key]
            for key in ('accepted_steps', 'rejected_steps', 'partition_switches', 'group_evaluations', 'gradient_evaluations',
                        'false_acceptances_prevented', 'split_seconds', 'audit_seconds', 'fallback_seconds',
                        'classification_seconds', 'partition_seconds'):
                row[key] = float(np.median([r['stats'][key] for r in good]))
            row['fallback_fraction'] = float(np.median([sum(f['remaining_fraction'] for f in r['stats']['fallbacks']) for r in good]))
            row['fallback_reasons'] = sorted({f['reason'] for r in good for f in r['stats']['fallbacks']})
        rows.append(row)
    return reg, runs, rows


def historical_hashes():
    first = json.loads((ROOT / 'results/m1-completion/acceptance.json').read_text(encoding='utf-8'))
    second = json.loads((ROOT / 'results/m2/acceptance.json').read_text(encoding='utf-8'))
    records = {**first['evidence_sha256'], **second['evidence_sha256'],
               'results/m1-completion/case-registration.json': first['registration_sha256']}
    return all(digest(ROOT / path) == sha for path, sha in records.items())


def validate(cases):
    cal, validation, screening, tests = [read(p) for p in ('calibration.json', 'validation.json', 'screening.json', 'tests.json')]
    checks = {'M1_M2_evidence_unchanged': historical_hashes(),
              '144_exact_estimator_calibrations': len(cal['rows']) == 144,
              'naive_false_accepts_exist_and_all_are_audited': sum(r['naive_false_accept'] for r in cal['rows']) > 0 and
                  all(not r['naive_false_accept'] or r['independent_audit_rejects'] for r in cal['rows']),
              'RMS_hidden_component_guard': cal['rms_counterexample']['wrms'] < 1 < cal['rms_counterexample']['maximum'],
              'stiff_initial_layers_nonlinear_pools_event_errors_pass': len(validation['cases']) >= 13 and
                  all(r['error'] <= 1 for r in validation['cases']),
              'hysteresis_reduces_chatter_in_registered_trace': validation['hysteresis_chatter'][0]['switches'] < validation['hysteresis_chatter'][1]['switches'],
              'rank_one_screen_matches_original_on_four_states': len(screening['snapshots']) == 4 and
                  all(r['max_relative_difference'] < 1e-10 for r in screening['snapshots']),
              'screening_optimization_measured_with_repeats': all(len(r['optimized_including_gradient_seconds']) == 7 and
                  len(r['original_group_loop_excluding_gradient_seconds']) == 7 for r in screening['snapshots']),
              'tests_pass_in_Windows_and_native_WSL': len(tests) == 4 and all(r['returncode'] == 0 and
                  r['source_sha256_LF'] == source_hashes() for r in tests)}
    sensitivity = read('sensitivities.json')
    checks['group_threshold_development_sensitivity'] = len(sensitivity['group_thresholds']) == 3 and all(
        r['error'] <= 1 for r in sensitivity['group_thresholds'])
    checks['tenfold_inner_tightening_with_frozen_maps'] = len(sensitivity['inner_tolerances']) == 4 and all(
        r['difference_in_local_target_units'] < .05 for r in sensitivity['inner_tolerances'])
    all_stats = []
    for v in validation['cases']:
        all_stats.extend(v.get('segment_stats', [v.get('stats')]))
    if screening['macro_probe']['stats'] is not None:
        all_stats += [screening['macro_probe']['stats']]
    checks['larger_macro_probe_and_failures_are_recorded'] = screening['macro_probe']['status'] in ('success', 'failed', 'timeout')
    all_stats += [r['stats'] for r in sensitivity['group_thresholds']]
    for name, (reg, runs, rows) in cases.items():
        expected = len(reg['configs']) * 7
        checks[name + '_all_repeats_recorded'] = len(runs) == expected and len(read(f'{name}-warmups.json')) == len(rows) and all(
            r['successes'] == 7 and r['failures'] == 0 for r in rows)
        checks[name + '_global_grid_concentration_error_passes'] = all(r['status'] == 'success' and r['observed_error'] <= 1 for r in runs)
        all_stats += [r['stats'] for r in runs if r['status'] == 'success']
        data = np.load(OUT / f'{name}-trajectories.npz')
        recomputed = True
        targets = {c['id']: c['target'] for c in reg['configs']}
        for run in [r for r in runs if r['repeat'] == 0]:
            if run['status'] != 'success':
                recomputed = False
                continue
            key = run.get('trajectory_key', run['config'])
            value = data[key]
            reference = data['reference']
            scale = data['scale'] if 'scale' in data else np.ones(len(value))
            recomputed = recomputed and np.isclose(error(value, reference, scale, targets[run['config']]), run['observed_error'], rtol=1e-10, atol=1e-10)
            recomputed = recomputed and np.isfinite(value).all() and not np.any(value < -1e-12 * scale[:, None])
        checks[name + '_saved_trajectory_metrics_and_domain_recompute'] = bool(recomputed)
        if name != 'toy':
            from .sbml import SBMLModel
            from .kecoli import KEcoliModel
            large = name.startswith('kecoli')
            model = KEcoliModel(ROOT / 'models/k-ecoli457.zip') if large else SBMLModel(ROOT / 'models/BIOMD0000000064.xml')
            ref_path = ROOT / reg['reference_path']
            prior = np.load(ref_path)
            checks[name + '_original_window_grid_scale_and_reference'] = (digest(ref_path) == reg['reference_sha256'] and
                digest(model.path) == reg['model_sha256'] and np.array_equal(prior['times'], data['times']) and
                np.array_equal(prior['scale'], data['scale']) and np.array_equal(prior['concentrations'], data['reference']))
            flux_ref = np.column_stack([model.flux(t, y) for t, y in zip(data['times'], data['reference'].T)])
            flux_ok = True
            if large:
                net_scale = np.array(json.loads((ROOT / 'results/m1-completion/kecoli-net-flux.json').read_text())['net_flux_scale'])
                conservation = model.pools
            else:
                conservation = model.L.T
            for run in [r for r in runs if r['repeat'] == 0 and r['status'] == 'success']:
                values = data[run.get('trajectory_key', run['config'])]
                flux = np.column_stack([model.flux(t, y) for t, y in zip(data['times'], values.T)])
                target = targets[run['config']]
                flux_ok = flux_ok and np.isclose(error(flux, flux_ref, data['flux_scale'], target), run['flux_errors'][str(target)], rtol=1e-10, atol=1e-9)
                invariant = float(np.max(abs(conservation @ (values - model.y0[:, None]))) /
                                  max(1e-12, float(np.max(abs(conservation @ model.y0)))))
                flux_ok = flux_ok and invariant < 1e-10 and np.isclose(invariant, run['invariant_error'], rtol=1e-8, atol=1e-14)
                if large:
                    flux_ok = flux_ok and np.isclose(error(model.net_map @ flux, model.net_map @ flux_ref, net_scale, 1e-3),
                                                    run['net_flux_errors']['0.001'], rtol=1e-9, atol=1e-8)
            checks[name + '_flux_metrics_recompute'] = bool(flux_ok)
            checks[name + '_registered_flux_invariants_and_ledger_pass'] = all(
                r['flux_errors'][str(targets[r['config']])] <= 1 and r['invariant_error'] < 1e-10 and
                r.get('relative_open_ledger_error', 0.) < 1e-10 and
                r.get('net_flux_errors', {'0.001': 0.})['0.001'] <= 1 for r in runs if r['status'] == 'success')
    accepted = [a for s in all_stats for a in s['attempts'] if a['status'] == 'accepted']
    rejected = [a for s in all_stats for a in s['attempts'] if a['status'] == 'rejected']
    checks['every_accepted_split_has_midpoint_endpoint_observable_audit'] = len(accepted) > 0 and all(
        a['estimator_wrms'] <= 1 and max(a['audit_max'], a['audit_flux_max'], a['audit_net_flux_max']) <= 1 for a in accepted)
    checks['every_rejection_has_atomic_rollback_evidence'] = len(rejected) > 0 and all(a['rollback_verified'] for a in rejected) and all(
        s['rollback_checks'] == s['rejected_steps'] for s in all_stats)
    checks['no_projection_clipping_and_committed_pool_errors_pass'] = all(s['clipping_count'] == s['projection_count'] == 0 and
        s['max_committed_pool_error'] < 1e-10 for s in all_stats)
    event = next(v for v in validation['cases'] if v['case'] == 'known_input_event')
    checks['event_boundary_landed_and_history_reset'] = (len(event['segment_stats']) == 2 and
        event['segment_stats'][1]['attempts'][0]['time'] == .1 and
        event['segment_stats'][1]['attempts'][0]['accepted_state_index'] == 0)
    # Performance is a separate decision, never folded into solver success.
    dynamic = [r for name in ('teusink', 'kecoli') for r in cases[name][2] if r['config'].startswith('dynamic')]
    useful = any(r['fallback_fraction'] < .5 for r in dynamic)
    return checks, useful


def render(cases, checks, useful):
    cal, screening, validation = read('calibration.json'), read('screening.json'), read('validation.json')
    probe = screening['macro_probe']
    probe_stats = probe['stats']
    probe_note = (f"探针状态 `{probe['status']}`，耗时 {probe['seconds']:.3f} 秒。"
                  + (f"保存 {probe_stats['accepted_steps']} 次接受、{probe_stats['rejected_steps']} 次拒步；"
                     f"回退原因：{', '.join(f['reason'] for f in probe_stats['fallbacks']) or '无'}。"
                     if probe_stats is not None else '内部记录不可取得，未填入推测数据。'))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ratio in (1., 100., 1e4, 1e6):
        rows = [r for r in cal['rows'] if r['fast_multiplier'] == ratio and r['snapshot'] == 0 and r['initial'] == 'far' and r['reversible']]
        axes[0].loglog([r['h'] for r in rows], [r['underestimation_ratio'] for r in rows], 'o-', label=f'multiplier {ratio:g}')
    axes[0].axhline(2., color='gray', linestyle='--')
    axes[0].set(title='Estimator calibration (exact flows)', xlabel='Macro step', ylabel='True endpoint error / step-doubling estimate')
    axes[0].legend(fontsize=7)
    toy_rows = cases['toy'][2]
    axes[1].bar(np.arange(len(toy_rows)), [r['median_seconds'] for r in toy_rows])
    axes[1].set_xticks(np.arange(len(toy_rows)), [r['config'].replace('dynamic-', '').replace('single-threshold', 'off').replace('hysteresis', 'on') for r in toy_rows], rotation=35, fontsize=7)
    axes[1].set(title='Shared chain: 7 repeats', ylabel='Audited end-to-end median seconds')
    names = ['Teusink', 'k-ecoli457']
    ratios = []
    for name in ('teusink', 'kecoli'):
        candidate = next(r for r in cases[name][2] if r['config'] == 'dynamic-5-hysteresis-1e-05')
        control = next(r for r in cases[name + '-full'][2] if r['target'] == 1e-5)
        ratios.append(candidate['median_seconds'] / control['median_seconds'])
    axes[2].bar(names, ratios)
    axes[2].axhline(1., color='gray', linestyle='--')
    axes[2].set(title='Full-window hybrid vs full-only control', ylabel='Time ratio (larger is slower)')
    for ax in axes:
        ax.grid(True, axis='y', alpha=.2)
    fig.savefig(OUT / 'calibration-cost.png', dpi=180)
    plt.close(fig)
    lines = ['# M3 动态分组、步长与回退阶段报告', '', '实施：2026-11-03 至 2026-11-04；验收：2026-11-04（Asia/Shanghai）。', '',
             f"**正确性检查 {sum(checks.values())}/{len(checks)} 通过；M3 完成本阶段实验与继续／停止判断。** "
             '当前实现是逐步审计的分裂—完整 ODE 混合原型。实际效率条件未得到支持，**不准入 M4／PE**。', '',
             '## 实现及误差边界', '',
             '按完整反应组的尺度化 Jacobian 无穷范数筛查；进入阈值 1、退出 0.3（无滞回消融取 1）。'
             '比较固定分组与每 1／5／20 个接受宏步更新；拒步立即重估。慢系统的尺度化 Jacobian 范数乘步长若超过 2，'
             '视为保守耦合报警，回退完整系统；这不是非线性稳定性证明。', '',
             '粗步和两个细步始终使用同一冻结分组；估计 `(y_f-y_c)/3`，接受 `y_f`，不作外推。'
             '控制器按 `clip(0.2,2,0.9 E^(-1/3))` 更新；分组改变禁止继承增长，控制器没有隐式 PI 历史。'
             '拒步后核验时间、状态、分组、缓存、分组历史及已提交开放收支均未变化；仅诊断计数和日志保留。', '',
             'M2 的刚性阶降低意味着不能直接信任二阶估计。每次候选都从相同起点用独立 Radau 完整 ODE 审计中点和终点，'
             '同时检查浓度最大分量、WRMS、速率及已登记净通量。局部审计预算为外层目标的 0.1 倍；'
             '子流初值为局部容差的 0.05 倍（上限 10^-8）。端点误差超过估计的 2 倍且大于 0.1 个局部目标单位时，'
             '判为超出已校准区间，回退完整系统。超标拒步最多连续 4 次；不通过缩小偏差审计频率制造收益。', '',
             '完整系统审计和回退是两种不同用途：审计只决定候选是否可信，不把其轨迹冒充已接受分裂解；'
             '回退明确从最后已接受状态继续。合成／小模型用稀疏 BDF 回退，大模型用同 RHS、解析 Jacobian 的 CVODE-KLU。'
             '在已知常值输入边界停止并换用单独模型对象，浓度连续，分组与控制历史重新初始化；不跨事件套自治公式。', '',
             '## 校准、反例与消融', '',
             f"144 个精确子流／完整矩阵指数局部样例中，朴素 WRMS 规则有 **{sum(r['naive_false_accept'] for r in cal['rows'])} 个错误接受**；"
             '逐步中点／终点最大分量审计全部拦截。另构造 100 维单分量反例：WRMS 小于 1，而最大分量超过 1。'
             '阈值 1.5／2／4 的敏感性记录保留，阈值不能替代独立审计，也不是一般误差定理。', '',
             '分组进入阈值另在开发链测试 0.5／1／2（退出阈值取进入值的 0.3 倍）；'
             '共享链、显式酶机制及 Teusink 两个轨迹状态还执行固定分组、相同宏步的子流容差 10^-8 → 10^-9 检查。'
             '这些是单次精度敏感性核验，不是重复计时或留出集泛化证据。', '',
             '共享线性链的固定／动态、1／5／20 周期、滞回开／关共 7 配置，每配置一次预热及 7 次交错重复。'
             '高倍率初始层、非线性双酶池、Robertson 零初值和已知输入切换均以完整系统独立参考检查；'
             f"登记验证中最大全轨迹误差为 `{max(v['error'] for v in validation['cases']):.4g}` 个目标单位。", '',
             '| 共享链配置 | 浓度 E(10^-3) | 接受／拒步 | 分组切换 | 回退覆盖时间 | 中位秒（IQR） |',
             '|---|---:|---:|---:|---:|---:|']
    for r in toy_rows:
        lines.append(f"| {r['config']} | {r['observed_error']:.4g} | {r['accepted_steps']:g}/{r['rejected_steps']:g} | "
                     f"{r['partition_switches']:g} | {100*r['fallback_fraction']:.3f}% | {r['median_seconds']:.4f} ({r['iqr_seconds']:.4f}) |")
    lines += ['', '低频更新在本例中可能错过步长增长后需要的快组，触发慢耦合保护。'
              '这能安全回退，但不能把更早回退造成的较低耗时归功于动态分裂。', '',
              '## 真实模型完整登记窗口', '',
              '沿用 M1 原始初值、模型指纹、物种／通量尺度、公共输出网格和完整窗口：Teusink 0–10 min；'
              'k-ecoli457 0–1 来源积分坐标。浓度／速率目标测试 10^-3 和 10^-5，大模型净通量目标为 10^-3。'
              '每种目标均重复完整的 7 配置；同一诊断包装另运行完整系统控制组。所有配置各预热一次并交错重复 7 次。'
              '连同共享链共 273 次正式运行、39 次预热，全部完成；单次开发校准与探针不计入正式重复数。', '',
              '| 模型 | 配置／目标 | 浓度 E | 速率 E | 净通量 E(10^-3) | 回退覆盖时间 | 中位秒（IQR） |',
              '|---|---|---:|---:|---:|---:|---:|']
    for name in ('teusink', 'kecoli'):
        for r in cases[name][2] + cases[name + '-full'][2]:
            net = f"{r['net_flux_errors']['0.001']:.3g}" if 'net_flux_errors' in r else '—'
            lines.append(f"| {name} | {r['config']} | {r['observed_error']:.3g} | {r['flux_errors'][str(r['target'])]:.3g} | "
                         f"{net} | {100*r['fallback_fraction']:.8f}% | {r['median_seconds']:.3f} ({r['iqr_seconds']:.3f}) |")
    lines += ['', '**这些达标轨迹主要由完整系统回退取得，不是纯分裂完整窗口资格。** '
              '为限制开发成本，真实模型登记最多 8 次分裂尝试；初始层公共网格极密，强制落步很早耗尽预算，'
              '回退覆盖接近整个窗口。该工程上限和输出策略本身限制了这批结果；不能从它推导所有多速率方法都无效。'
              '另有不受早期输出落步限制的大模型宏步探针。默认慢耦合保护触发正常回退；'
              '为单独检验估计器，另一次开发探针固定 145 个快组、仅在该探针将慢耦合阈值放宽到 10^6。'
              + probe_note + '候选的步长、WRMS、独立浓度／通量误差和回滚证据逐次保留。'
              '该配置不用于正式计时或生产接受条件。', '',
              '## 一次优化核验与成本判断', '',
              '按来源的相反化学计量列合并正反速率梯度，精确计算同一组 Jacobian 范数；'
              '不再逐组构造稀疏 Jacobian。四个轨迹状态均与原逐组算法吻合，保存每状态七次测量。'
              '原循环时间不含梯度，优化时间包含梯度；该不对称方向不夸大优化优势。', '',
              '| 状态坐标 | 原逐组中位秒 | 优化（含梯度）中位秒 | 最大相对差 |', '|---:|---:|---:|---:|']
    for r in screening['snapshots']:
        lines.append(f"| {r['time']:.4g} | {np.median(r['original_group_loop_excluding_gradient_seconds']):.5f} | "
                     f"{np.median(r['optimized_including_gradient_seconds']):.5f} | {r['max_relative_difference']:.3g} |")
    lines += ['', '总时间包括分组预处理／筛查、分区矩阵构建、粗细双算、子求解器重启、Jacobian／分解、'
              '独立审计、失败重试、收支诊断与完整系统回退。模型解析单列 setup；参考生成、误差后处理与绘图离线。'
              'full-only 控制组经过同一诊断包装和严格回退容差，用于量化额外尝试开销，**不是对各精度档最快成熟基线的重新选优**；'
              '不能据该控制组宣称超越 BDF／Radau／CVODE。M1 强基线原始结果仍保留。', '',
              '即使某个混合配置的总时间低于 full-only 控制组，也不能据此认定动态分组产生加速：'
              '短暂预积分改变了完整系统求解器的启动状态与步长历史，而分裂覆盖时间几乎为零。'
              '该实现层差异与动态分组贡献必须分开，本阶段没有支持后者的证据。', '',
              '原生 CVODE 只暴露 RHS 参数和输出，未提供所有内部接受状态或独立 LU 次数。'
              '日志分别报告 SciPy 可用 nlu 与原生 linear setup，不互相替代。未采集隔离 worker 峰值内存。'
              '所有最小值未经裁剪；拒绝试算的负值／失败与已提交轨迹分开保存。', '',
              '**继续／停止决定：**逐步完整系统审计仍不可省，动态分组未展示覆盖其代价的实际收益；'
              '在核验一次分组频率、稀疏实现及重启成本后，按项目规划停止当前以加速为目标的 PE 扩展。'
              'M4 不准入，M5／M6 未自动开展。后续若要继续，应先解决输出／重启策略和廉价但可靠的估计器，'
              '再以较大分裂预算及同实际精度的强基线重验，不能删除安全检查来得到加速。', '',
              '## 复现', '', '```powershell',
              'uv run python -m metabolism.m3_experiments --case calibration',
              'uv run python -m metabolism.m3_experiments --case toy',
              'uv run python -m metabolism.m3_experiments --case sensitivities',
              'uv run python -m metabolism.m3_experiments --case teusink',
              'uv run python -m metabolism.m3_experiments --case teusink-full', '```', '',
              'Ubuntu／WSL（M1 环境）：', '', '```bash',
              'python3 -m metabolism.m3_experiments --case kecoli',
              'python3 -m metabolism.m3_experiments --case kecoli-full',
              'python3 -m metabolism.m3_experiments --case screening', '```', '',
              '`screening` 默认复用已保存的四状态筛查计时，并重跑宏步探针；加 `--refresh-screening` 才重新测量筛查。'
              '探针记录内部 20 秒及外部 30 秒超时边界，失败／超时也保存，不补造不可取得的内部计数。', '',
              '最后回到 Windows：', '', '```powershell',
              'uv run python -m metabolism.m3_report --run-tests',
              'uv run python -m metabolism.m3_report --check', '```', '',
              '`--check` 只重算保存轨迹的误差、验收条件和 SHA-256，不运行积分。M1／M2 的验收文件及其证据保持原样；'
              'M3 额外核验这些历史指纹，新的测试源码集合独立登记。', '',
              '![校准与成本](../results/m3/calibration-cost.png)', '']
    (ROOT / 'docs/m3-completion-report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-tests', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.run_tests and args.check:
        parser.error('--check never executes test processes')
    if args.run_tests:
        run_tests()
    names = ('toy', 'teusink', 'kecoli', 'teusink-full', 'kecoli-full')
    cases = {name: summary(name) for name in names}
    checks, useful = validate(cases)
    if args.check:
        attestation = read('acceptance.json')
        checks['evidence_sha256'] = all(digest(ROOT / path) == sha for path, sha in attestation['evidence_sha256'].items())
        checks['saved_results_match_recomputed'] = attestation['checks'] == {k: v for k, v in checks.items() if k not in ('evidence_sha256', 'saved_results_match_recomputed')}
    else:
        for name, (_, _, rows) in cases.items():
            save(OUT / f'{name}-summary.json', rows)
        render(cases, checks, useful)
        paths = [p for p in OUT.iterdir() if p.name != 'acceptance.json']
        paths += [ROOT / 'docs/m3-completion-report.md', ROOT / 'metabolism/adaptive.py',
                  ROOT / 'metabolism/m3_experiments.py', ROOT / 'metabolism/m3_report.py', ROOT / 'tests/m3/test_adaptive.py',
                  ROOT / 'tests/m3/__init__.py']
        save(OUT / 'acceptance.json', {'M3': 'completed_with_stop' if all(checks.values()) else 'incomplete',
            'correctness': 'pass' if all(checks.values()) else 'fail', 'efficiency_supported': useful,
            'M4_admission': 'not_admitted', 'checks': checks,
            'evidence_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
            'decision': 'Preserve correctness prototype; stop PE efficiency extension under the current full-step audit, restart and output strategy.',
            'limits': ['Full-window hybrid is predominantly full-ODE fallback', 'No pure-splitting qualification or speedup claim',
                       'No claim of global error guarantee beyond registered evaluation grids', 'No PE, M5 or M6 executed']})
    print(json.dumps(checks, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit('M3 correctness evidence failed')


if __name__ == '__main__':
    main()

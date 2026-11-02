# M1 验收与 M2 准入报告

日期：2026-11-02（Asia/Shanghai）。**M1 已完成，M2 准入通过；M2 算法尚未实施。**

[验收 JSON](../results/m1-completion/acceptance.json) 对 13 项证据逐项检查，并保存输入证据的 SHA-256；[冻结配置](../results/m1-completion/case-registration.json) 给出允许进入 M2 的模型、窗口、误差档位和范围。[首轮报告](m1-baseline-report.md) 保留开发历史，当前结论以本文为准。

## 验收结果

| M1 要求 | 实际证据 | 结论 |
|---|---|---|
| 至少两个公开模型的语义和公开轨迹／稳态可复现 | Teusink 14 状态、Chassagnole 18 状态；Python 与 RoadRunner 导数／通量一致；前者复现修订 Table 4 稳态，后者对照公开 MathSBML 复现图 | 通过 |
| 参考解足够一致并做容差／网格核验 | 小模型 Radau／BDF／CVODE，补充更紧浓度与通量核验；大模型再与 Octave `ode15s` 原方程执行对照 | 在登记档位通过 |
| 至少一个可执行的中等规模或更大真实候选 | k-ecoli457 的实际展开系统：3003 状态、5239 单向反应、2666 完整反应组；原包参数和初值已核验 | 通过 |
| 强基线与可解释成本 | 同一 Linux 运行时、相同 Python RHS／解析稀疏 Jacobian；BDF、Radau、CVODE-BDF+KLU；预热后每配置 7 次交错计时，记录进程峰值内存和成本 | 通过 |
| 不变量、开放收支、原始负值 | 456 个真实酶池、伪池排除有代数依据；开放系统记账；保存未经裁剪的输出与可访问内部状态最小值 | 达标配置通过 |
| 存在实际快慢结构且有继续研究价值 | 多个轨迹状态上，Teusink 局部快块和 k-ecoli457 完整可逆组均显著移除局部最快模态；大模型 SciPy 时间主要花在线性分解 | 允许进入 M2，尚非加速结论 |

## 补齐的模型证据

**Chassagnole。** BioModels 的 `curation_notes.txt` 明确其公开图由 MathSBML 得到、复现论文图 4–6。我们保存原图及指纹，按图中坐标标定 9 条代谢物曲线，固定取样后对照原始 SBML 在 0–40 s 的数值解。81 个点最大偏差约 **3.01 像素**，低于预定 5 像素门槛；原图一个横向像素约 0.22 s。这是粗栅格尺度的公开执行结果复现，不能称为高精度实验数据拟合，也没有取得论文原始浓度观测数据。

其 7 条显式时间辅因子规则仍按原式执行。**该模型不进入首版自治 Strang 分裂**，保留为完整系统基线与语义验证对象；不冻结辅因子、不悄然扩展为非自治方法。

**k-ecoli457。** 使用原包 `parallel_automated.m`、`perturbModel.m` 的好氧葡萄糖条件：代谢物归一化初值设为 1，关闭 446–455 号替代碳源通道，ACALD 酶量取 2 倍，得到实际瞬态而非几乎静止的默认状态。源包初值与 Python 初值逐项差异为 **0**。

在两个登记状态上直接调用原始 `mass_balance_ode.m`，Python 导数与原函数的最大绝对差异为 **0**；与原始生成 Jacobian 的最大分量尺度化差异约 **1.11×10^-10**。Octave 8.4.0 的 `ode15s` 使用相同方程的向量化乘积和原始 Jacobian 生成完整轨迹；向量化 RHS 同时与未修改原函数核对，最大浮点加法顺序差约 `4.66×10^-10`。原始源码留在忽略的本地缓存，仓库保存调用脚本、输入和执行输出，不改写下载包。

原入口中的稳态／CPU 时间终止逻辑是求解停止条件，不是化学输入事件。所有比较方法统一改为登记的固定窗口 `[0,1]`；不将其单位擅自标为物理秒。此窗口是数值开发用例，未声称复现论文中全部扰动实验。

另按原 `get_Vnet.m` 的“首个基元步骤”及转运符号规则恢复 457 个总体反应通量。默认未过表达状态与来源包的稳态通量，在落实上述 10 个条件开关后最大尺度化差约 `2.09×10^-12`。原 `Data.mat` 中关闭通道仍留有 ±100 的替代碳源占位通量；这些差异显式登记，未混同于当前好氧条件。

## 精度档位及结构边界

小模型补测 `rtol=10^-10,10^-11`，并以更紧 BDF／CVODE 对 Radau `10^-12` 核验浓度和通量。最严格 `10^-7` 目标下，全部参考差异仍小于 5%；Chassagnole 的最大通量参考差异约 **0.0196 个目标单位**。BDF、Radau、CVODE 各有同时满足浓度与通量要求的配置。

大模型冻结的浓度档位是 **10^-3、10^-5**，基元单向通量同档；原始总体净通量冻结为 **10^-2、10^-3**。这些不是所有模型统一使用同一内部容差。

- 在浓度 `10^-5` 目标下，CVODE 容差收紧、BDF、Radau 和 Octave 的最大参考差异约 **0.00363 个目标单位**；单向通量参考差异同样小于 5%。
- 原始净通量是大正反向通量之差，对状态误差很敏感；`10^-5` 净通量档参考差异最高约 1.79 个目标单位，不满足参考预算。在登记的 `10^-3` 净通量档，最大差异约 **0.0179 个目标单位**，通过 5% 门槛。
- 曾试更紧 Radau `rtol=10^-10`，运行超过 120 s 后停止；这个成本／精度限制保留在审计中，没有计为成功参考。按原协议的舍入／病态限制条款登记实际可核验档位，不宣称大模型达到 `10^-7` 浓度或 `10^-5` 净通量精度。

自然酶池的 Gram 矩阵为对角，基底条件数约 **3.06**；456 个真实池满足 `LᵀS=0`。`Biomass_Bulk` 对应伪池不满足该式，已据此排除。参考解最大酶池绝对漂移约 `1.25×10^-10`，按协议以最大初始池量 2 归一化后约 **6.27×10^-11**，通过 `10^-10` 门槛。基线的粗容差配置若结构或精度不达标仍保留；只冻结同时达标的配置。

Chassagnole 的开放收支用碳数加权浓度记账，按实际编码的反应列逐项纳入非闭合贡献，包括带生物量缩放的 PTS 列。独立自适应积分核验的最大相对残差约 **5.41×10^-13**。该记账不是全培养体系物理总质量断言，更未对开放系统投影回初始总量。

## 成本与继续条件

本轮增加 **84 个小模型严格容差计时批次**和 **84 个大模型正式计时批次**；加上首轮 168 个，共 **336 个正式批次**，均运行成功。求解器报告成功与精度达标分开记录，失败精度的粗容差配置仍在[完整结果表](../results/m1-completion/README.md)中。

大模型在 `rtol=10^-8` 时，BDF／Radau／CVODE-KLU 的每轨迹中位积分及输出耗时约 **2.53／7.60／1.23 s**；三者同时达到登记的浓度、单向通量、净通量和酶池门槛。这里提供了实际误差曲线，不把同容差本身当作公平比较的依据。配置由开发集选择，尚不是留出扰动的泛化结论。

SciPy 成本分解显示 LU 分解约占本用例时间的八成，RHS 仅占较小部分；CVODE 的内部线性代数计入总耗时，但接口未单独暴露 LU 用时／次数，不能用线性求解器 setup 次数替代 LU 次数。隔离 worker 峰值 RSS 约 290–299 MiB，包含继承的模型／参考数组及后处理，不冒称纯求解器增量内存。Windows 小模型计时与 Linux 大模型计时不跨环境拼接比较。

Teusink 在 `h=10^-5 min` 下，两个完整净反应 `vGAPDH/vPGK` 更新 6 个状态；多个状态上，剩余慢 Jacobian 的最大模态约降低 **86–106 倍**。到 `h≥10^-3 min`，快块可能扩至全网，不能假定快块始终局部。

k-ecoli457 按来源映射组成 **2573 个正反向对和 93 个单向组**。在 `h=10^-5` 下，多状态上约 145 个快组更新 267 个状态，最大连通块约 202 个状态；余下慢 Jacobian 的最大模态约降低 **1.87–1.89 万倍**。这是局部线性诊断，仍须检查显式子步稳定性和完整速率依赖。遍历所有组构造诊断的当前开销约 0.4–0.5 s，必须在 M3 优化和计入，不能免费忽略。

这些结果满足“存在实际分离且分解成本值得研究”的 M1 继续条件。**M2 仅获准进行无 PE 固定 Lie／Strang 的阶数、守恒、负值和成本验证；尚未证明分裂有净加速，也不据此进入 PE。** 93-metabolite core 包未取得，但已由可执行 k-ecoli457 满足规模候选要求，不再是 M2 阻塞项。

## 复现

Windows 项目环境：

```powershell
uv sync --locked
uv run python -m metabolism.fetch_models
uv run python -m unittest discover -s tests -v
uv run python -m metabolism.curation
uv run python -m metabolism.calibration
uv run python -m metabolism.diagnostics
uv run python -m metabolism.prepare_source
```

Ubuntu 24.04／WSL 安装运行时可用 `sudo bash scripts/setup-linux.sh`；本次实际版本另存 `linux-environment.json` 和 `linux-packages.txt`。从项目目录运行：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
octave --no-gui --quiet --eval "addpath(fullfile(pwd, 'scripts')); kecoli_reference(pwd);"
python3 -m unittest discover -s tests -p test_cvode.py -v
python3 -m unittest discover -s tests -p test_kecoli.py -v
python3 -m metabolism.large_benchmark --reference
python3 -m metabolism.large_benchmark --benchmark
python3 -m metabolism.net_flux_audit
python3 -m metabolism.large_diagnostics
```

最后在 Windows 环境运行：

```powershell
uv run python -m metabolism.acceptance
uv run python -m metabolism.completion_report
```

`acceptance` 会重新核验保存的证据并返回非零退出码表示未通过；它不会重新运行全部求解器。冻结配置以当前开发证据登记，未来 20 个保持正反向关系的留出扰动仅注册于 M5，尚未运行。

# MetabolismNetworkComputing

面向刚性代谢动力学 ODE 的**自适应快—慢反应算子分裂**研究。

**当前阶段：M2 已完成，22 项验收检查通过；M3 尚未开始。** 固定 Lie／Strang、精确小系统、守恒／开放收支和负值检测已验证，真实模型短窗保留全部 168 次正式运行及精度负结果。强刚性下出现阶降低，大模型当前宏步未达到精度目标；没有分裂算法加速结论。调研及实施记录日期：2026-11-01 至 2026-11-03（Asia/Shanghai）。

## 研究问题

在相同轨迹精度下，能否利用代谢网络的可逆反应、局部时间尺度与化学计量结构，降低相对于成熟稀疏隐式求解器的总计算成本，同时控制守恒误差、负浓度和长期漂移？

建议先验证**不做近似降阶的自适应分裂**，再引入**有适用条件检查和误差核验的部分平衡（PE）加速**。动态分组、自适应步长和 PE 均有先行工作；本项目的潜在贡献需要靠代谢模型上的误差控制、结构性质和公平实验建立。

## 文档导航

| 文档 | 内容 |
|---|---|
| [原始构想](grok_idea.md) | 用户提供的原文；即本次所指的 idea 文件，保留原文件名 |
| [项目规划](docs/project-plan.md) | 范围、假设、算法设计、阶段任务、继续／停止条件 |
| [文献综述与阅读笔记](docs/literature-review.md) | 经典与近期工作、阅读深度、适用边界、创新性对照 |
| [基准与评估协议](docs/benchmark-protocol.md) | 候选模型、基线、指标、消融、鲁棒性、公平计时 |
| [检索与证据记录](docs/research-log.md) | 检索范围、来源、全文阅读定位、未解决问题 |
| [进度记录](docs/progress.md) | 本轮已完成项及明确未执行项 |
| [M1 首轮实施报告](docs/m1-baseline-report.md) | 模型语义、参考解核验、基线结果与剩余验收项 |
| [M1 验收与 M2 准入报告](docs/m1-completion-report.md) | 补齐证据、实际精度上限、规模基线、准入范围及复现命令 |
| [M2 固定分裂验收报告](docs/m2-completion-report.md) | 基础阶数、刚性阶降低、守恒／开放收支、负值检测及真实模型短窗成本 |
| [BibTeX 文献库](references.bib) | 与综述编号对应的可引用条目 |

## 运行 M1

需要 [uv](https://docs.astral.sh/uv/)；依赖由 `uv.lock` 锁定，Python 3.12、CPU float64：

```powershell
uv sync --locked
uv run python -m metabolism.fetch_models
uv run python -m unittest discover -s tests -v
uv run python -m metabolism.benchmark
```

下载器核对 [模型清单](models/manifest.json) 中的 SHA-256；已有文件也必须通过核验。两个 CC0 SBML 随仓库保存；k-ecoli457 原始 ZIP 留作本地缓存，可按清单重新下载。运行结果保存在 [results/m1](results/m1)，含参考轨迹、完整重复记录、语义审计、环境版本及工作量—精度图。完整基线需数分钟；重复次数下限为 7，短轨迹按所有方法相同的批大小计时。

首轮 CVODE 使用 RoadRunner 编译 RHS，不能从首轮图推断纯算法优势。已补齐的大模型比较使用相同 Python RHS／解析稀疏 Jacobian 的 BDF、Radau 与 CVODE-KLU；其结果及全部复现步骤见 [M1 验收报告](docs/m1-completion-report.md)。336 个正式计时批次全部运行成功，精度不达标的粗容差配置仍保留。

```powershell
uv run python -m metabolism.acceptance
```

该命令核验保存证据并输出 `M1: complete`、`M2_admission: pass`。M2 准入对象为合成正确性样例及 Teusink／k-ecoli457 自治模型；Chassagnole 含显式时间规则，只保留完整系统基线。大模型浓度／单向通量最严格档为 `10^-5`，原始净通量为 `10^-3`，小模型仍保留浓度及通量 `10^-7`。适配器对不支持的 SBML 语义明确报错。

## 运行 M2

```powershell
uv run python -m metabolism.m2_experiments --case b0
uv run python -m metabolism.m2_experiments --case teusink
```

在 M1 的 Ubuntu／WSL 环境运行 `python3 -m metabolism.m2_experiments --case kecoli`，最后回到 Windows：

```powershell
uv run python -m metabolism.m2_report --run-tests
uv run python -m metabolism.m2_report --check
```

[M2 结果](results/m2)保留全部重复、原始轨迹、负结果和指纹。真实模型只运行登记窗口内的短窗探针；没有全窗口精度或加速结论。动态分组、外层误差控制和回退属于 M3；PE 仍未准入。

关键近期对照：Zeng、Xie、Wang 的 [2026 年 CLE 快—慢分裂预印本](https://arxiv.org/abs/2604.00140v1)。它研究随机系统；本项目首先研究确定性 ODE，不直接沿用其随机误差公式或性能结论。

# MetabolismNetworkComputing

面向刚性代谢动力学 ODE 的**自适应快—慢反应算子分裂**研究。

**当前阶段：M1 模型与基线已启动。** 已取得两个公开 SBML 模型及 k-ecoli457 原始补充包，实现有限 SBML 语义适配器，并运行 BDF／Radau／CVODE 基线试验。M1 验收尚未全部完成；尚无分裂算法或算法加速结论。调研及实施记录日期：2026-11-01 至 2026-11-02（Asia/Shanghai）。

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

当前不支持任意 SBML：适配器对未支持的语义明确报错。Chassagnole 的显式时间规则用于完整系统基线，暂不纳入首版自治分裂。CVODE 使用 RoadRunner 编译 RHS 和默认稠密线性代数，图中与 SciPy 的差异包含软件实现成本。M2–M6 仍按验收门推进。

关键近期对照：Zeng、Xie、Wang 的 [2026 年 CLE 快—慢分裂预印本](https://arxiv.org/abs/2604.00140v1)。它研究随机系统；本项目首先研究确定性 ODE，不直接沿用其随机误差公式或性能结论。

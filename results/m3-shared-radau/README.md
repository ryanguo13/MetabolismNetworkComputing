# 阶段共享与强基线挑战

2026-11-08，开发研究；M4 不准入。见 [报告](../../docs/m3-stage-sharing.md)。

- `registration.json`、`runs.json`、`cost-0.npz`–`cost-5.npz`：六个已知历史状态、五种消融／对照、两档内层容差，各一次预热和三次计时，共 240 条记录。
- `fairness/registration.json`、`fairness/runs.json`、`fairness/cost-{2,3,5}.npz`：第一批后追加登记的起步／容差挑战，三个配置、五种选项、四档容差，同样 240 条记录。是适应性开发补查，不是独立确认实验。
- `tests.txt`：7 项自检；`summary.json`：两批全部曲线、目标判定、计时中位数／IQR、工作量和门槛决定。
- `environment.json`、`manifest.json`：运行环境、父提交、实际 Radau 源码指纹及本轮源码／数据清单。

使用真实计算历史；未把参考状态作为免费启动。两边计时都不含独立误差核算、必要完整审计、估计器或启动。保存了预热、全部不合格结果；没有修改历史证据。

只读核验：`bash scripts/run-macos.sh -m research.m3_shared_report`。

# 单宏步 MRI2 与紧凑残差估计

2026-11-08。保留完整审计，没有在线接受或 M4 准入。

- `registration.json`：48 个开发候选和 48 条成本记录的冻结协议、源码指纹。
- `development.json`、`dev-*.npz`：两模型四起点的候选端点／曲线、三个预设估计网格的轨迹与判据、独立参考。
- `cost.json`、`cost-*.npz`：结构缓存重用、候选与完整系统的开发计时；准确性事后计算在两边计时外。
- `partition-registration.json`、`partitions.json`、`partitions-*.npz`：成本失败后登记的 36 个快块范围探针，含负值与不合格结果。
- `summary.json`：本目录与三阶目录的独立重算；`tests.txt`：4 项新增自检。
- `environment.json`、`manifest.json`：环境与本轮全部源码／证据指纹。

见 [报告](../../docs/m3-macro-redesign.md)。只读复核：`bash scripts/run-macos.sh -m research.m3_macro_report`。

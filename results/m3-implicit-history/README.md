# 隐式慢历史诊断证据

2026-11-08，开发研究，M4 不准入。见[报告](../../docs/m3-implicit-history.md)。

- `stability-registration.json`、`stability.json`、`stability.npz`：300 个精确快流、冻结线性、隐式端点历史配置。
- `local-registration.json`、`local.json`、候选及参考 NPZ：**保留但排除出科学比较的首轮**；三／四点历史输入切片错误导致 72 条输入检查失败，全轮 108 条均不作为科学结论。
- 修正后的全部 108 条非线性尝试与 18 个对照位于相邻 [`m3-implicit-local-corrected`](../m3-implicit-local-corrected/README.md)，不是覆盖首轮文件。
- `literature-provenance.json`：定向查询、8 条元数据、实际全文读取章节与版本、PDF 指纹及检索技能出处。
- `tests.txt`、`summary.json`、`environment.json`、`manifest.json`：5 项自检，全部线性矩阵与非线性尝试重新求解、曲线／账本／参考／决策核验，两个目录统一证据清单。

默认只读复核会重新求解候选与独立参考：`bash scripts/run-macos.sh -m research.m3_implicit_report`。不得重跑原驱动覆盖已登记证据。

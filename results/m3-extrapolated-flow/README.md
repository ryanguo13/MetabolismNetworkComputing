# 一次快流历史外推：局部与自生成历史证据

2026-11-08。开发研究，M4 未准入；见 [报告](../../docs/m3-extrapolated-flow.md)。

- 相邻 `../m3-ledger-order/` 保存预登记的 144 条原 MRI2／MRI3 账本分解和 18 组独立参考。其中 3 组粗步参考积分未通过 8/16 点一致性，明确标记并排除资格。
- `registration.json`、`runs.json`、`path-*.npz`：一次快流候选的 144 条局部筛查，含 108 候选、36 对照；过去慢 RHS 来自真实计算历史，未用参考解填充。
- `path-registration.json`、`paths.json`、`trajectory-*.npz`、`path-reference-*.npz`：9 条自生成历史路径、144 宏步、18 个连续完整对照；完整求解启动计入候选时间。失败局部判定仍保留，不计接受覆盖。
- `tests.txt`、`summary.json`、`environment.json`、`manifest.json`：4 项自检、独立曲线／误差／两套账本／累计结果重算、环境与指纹；本清单也涵盖相邻账本诊断目录。
- `literature-provenance.json`：实际查询、标识符、读取版本与章节、PDF 指纹及检索技能来源。

本轮只有单次开发计时；未计独立审计、误差评分和分组成本，无正式加速声明。只读复核：`bash scripts/run-macos.sh -m research.m3_ledger_flow_report`。

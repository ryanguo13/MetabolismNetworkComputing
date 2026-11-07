# SPC 完整预测器内部求解容差补测

首批大模型预测残差在快速下降后停滞；本目录独立登记 `1e-9/1e-8` 两档内部求解容差，保留原实际误差、通量、池、两套收支与独立完整参考门槛。

144 个 SPC、12 个同容差完整 SDIRK、18 个成熟 Radau 共 174 条记录全部完成。SPC 宽松／严格档分别通过 90／60 条。相对最快合格成熟 Radau 的最好单次比值约 1.13，同优化的完整 SDIRK 多数更快；M4 未准入。见[完整报告](../../docs/m3-spc.md)。

本目录由 [`../m3-spc/manifest.json`](../m3-spc/manifest.json) 统一保存指纹。只读复核：`bash scripts/run-macos.sh -m research.m3_spc_report`。

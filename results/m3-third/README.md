# MRI-GARK-ERK33a 开发证据

2026-11-08。采用 Sandu arXiv:1808.02759v3 §6.2 的三阶方法，并核对 SUNDIALS 6.4.1 系数；三个快子积分，最后一个为仿射慢驱动，尚无嵌入对或在线控制器。

`registration.json` 保存文献版本／指纹、系数核对来源、冻结配置与源码；`runs.json` 保存 18 个候选及 18 个完整系统对照；每个 NPZ 对应一个区间，含两级独立参考、优化参考、候选端点／曲线及影子误差轨迹。计时均为单次开发测量。

端点、整段输出和物理收支分开审计。当前没有同精度成本优势，M4 未准入。见 [报告](../../docs/m3-macro-redesign.md)及[统一机器汇总](../m3-single/summary.json)。只读复核：`bash scripts/run-macos.sh -m research.m3_macro_report`。

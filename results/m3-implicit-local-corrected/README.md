# 修正历史输入后的非线性局部实验

本目录独立保存修正后的 108 条候选、18 个完整对照和三个真实历史启动。首轮输入错误及原文件保留在相邻目录，见[完整报告](../../docs/m3-implicit-history.md)。

48 条候选迭代收敛，38 条通过两档局部精度及保护。48 条不收敛、12 条遇到负状态，全部保留。三个步长的最快合格候选相对完整对照时间比为 .248/.192/.119；均为未计历史、分组、独立审计的单次乐观开发计时，不构成正式加速或 M4 准入证据。

本目录由 [`../m3-implicit-history/manifest.json`](../m3-implicit-history/manifest.json) 统一管理指纹。只读复核：`bash scripts/run-macos.sh -m research.m3_implicit_report`。

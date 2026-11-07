# 多步／切换／扰动影子压力测试

12 条强制诊断路径、96 步；任何误差拒绝均保存，未当作生产控制器接受。

`registration.json` 冻结场景和源码；`runs.json` 保存逐步组、实际误差、影子评分、局部与全路径误差；`summary.json` 为运行摘要。每条路径一个 NPZ，包含起点、候选节点／输出、局部和全路径两级独立参考、两级误差估计轨迹。`packaging.json` 保存原压缩包 SHA-256、数组总数和全部分片指纹。原包一个参考数组解压失败，已从保存起点和网格独立重算并复现原误差指标；损坏与恢复记录分别见 `archive-integrity-failure.json`、`reference-recovery.json`。该成员不声称与原始字节相同。

结果和局限见 [报告](../../docs/m3-followup-evaluation.md)。96 步无观察到的错误接受，不等于全窗口可靠性；M4 未准入。统一复核：`bash scripts/run-macos.sh -m research.m3_followup_report`。

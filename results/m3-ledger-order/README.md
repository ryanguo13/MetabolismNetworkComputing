# MRI 端点／输出账本误差阶诊断

144 个候选、18 组独立参考；`registration.json` 为运行前登记，`runs.json` 保留所有结果，`early.npz`／`late.npz` 保存参考、候选端点和 Hermite 系数。

参考端点构造的 Hermite 仅作输出误差隔离，不是可接受候选。早期三个最大 H 的参考 8/16 点积分预算不一致，已标记并排除资格。见 [报告](../../docs/m3-extrapolated-flow.md)。

本目录由 [联合清单](../m3-extrapolated-flow/manifest.json) 和 `research.m3_ledger_flow_report` 只读核验。

# 第一版影子误差估计

2026-11-07。公式、局限与后续决策见 [评估报告](../../docs/m3-shadow-evaluation.md)。

- `registration.json`：事前冻结公式、开发数据、源码、新状态和局部筛选门槛。
- `development.json/.npz`：68 个历史候选的影子判断及两级误差传播轨迹。
- `new-states.json/.npz`：36 个新候选、独立参考／收紧参考、候选节点／曲线、误差轨迹、实际误差和假想决定。
- `ordering-probe.json`：同一开发矩阵的四种 LU 排序，一次预热和三次轮转重复。未作为性能资格实验。

两模型均没有观察到错误接受，但均未通过成本门槛。所有判断为影子记录，未移除完整 ODE 审计。两批数据及源码统一由 [优化版指纹](../m3-shadow-fast/manifest.json)核验。

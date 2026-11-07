# M3 宏步增长与慢历史稳定性证据

本目录保存 2026-11-08 的开发研究。M4 准入标准没有改变，当前结果不构成准入或正式性能声明。

- `registration.json`、`runs.json`：三条登记路径、三／四点历史、三种增长倍率、两档内层容差，共 36 条轨迹和 294 个宏步。
- `reference-{2,4,5}.npz`：每条路径的初始状态、公共查询网格和基线曲线；默认 verifier 会用冻结源码重新积分严格参考，不重复保存大数组。
- `*-*.npz`：每条候选轨迹的启动历史、宏步曲线、慢驱动多项式、端点和查询映射。
- `stability-registration.json`、`stability.json`、`stability.npz`：两个 Teusink 状态、十个分组阈值、五个步长的 100 个冻结线性稳定性配置。
- `tests.txt`、`summary.json`、`environment.json`、`manifest.json`：测试、独立复核摘要、运行环境和指纹。

严格 `1e-6` 目标通过 18/36 条，宽松 `1e-4` 目标通过 20/36 条。最快候选与同路径最快合格连续完整基线的单次时间比均小于 1（路径 2/4/5 严格档分别约 0.261/0.622/0.809），因此没有 1.5 倍加速证据。冻结线性诊断显示，`h>=0.001` 时通过筛查的最小快活跃维数仍为 14；缩小到 8–9 个状态需要约 `h=1e-4`。

验证命令：

```bash
bash scripts/run-macos.sh -m unittest research.test_growth_history research.test_history_stability -v
bash scripts/run-macos.sh -m research.m3_growth_report
```

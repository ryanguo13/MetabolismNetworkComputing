# M3 耦合与输出研究证据

日期：2026-11-07。结论见 [研究报告](../../docs/m3-research-resolution.md)。

- `probe.json`：24 个独立大模型局部探针、所有方法和失败条件、分组、独立参考收紧、原生工作量、精确线性自检。
- `probe.npz`：从原初值计算的两个起点、冻结尺度、参考轨迹、两级数值轨迹。数值初值不来自保存参考。
- `output-probe.json`：原始误差重算、完整 RHS 三次 Hermite 输出、非负与守恒检查、输入及脚本 SHA-256。
- `sources.json`：实际取得的文献／官方页面链接、版本、内容 SHA-256；版权全文只留在忽略的本地阅读缓存。
- `coupling-output.png`：选定开发样例的局部误差图，绘图代码见 `research/plot_m3_coupling.py`。

局部浓度／单向通量／净通量目标均为 `1e-4`；不是全窗口误差。记录中的 `sampled_pass` 只代表指定局部采样点，不授予新求解器资格。没有 PE、没有正式加速比，也没有 M4 准入。

运行：

```bash
bash scripts/run-macos.sh -m research.m3_coupling_probe
bash scripts/run-macos.sh -m research.m3_output_probe
bash scripts/run-macos.sh -m research.plot_m3_coupling
```

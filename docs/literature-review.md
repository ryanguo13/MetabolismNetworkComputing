# 文献综述与阅读笔记

检索／阅读：2026-11-01 至 2026-11-02（Asia/Shanghai）。范围：确定性刚性反应 ODE、算子分裂、部分平衡、结构性质、代谢动力学模型，并核查近期随机生化分裂工作。

**阅读标记：**`F`＝已取得全文并阅读与本计划相关的方法／分析／实验部分，不代表所有证明逐行复核；`A`＝摘要或出版页面／节选；`M`＝书目信息。未拿到全文的条目不伪装成全文精读。BibTeX 键与下表编号一一对应，全文版本和定位见 [证据记录](research-log.md)。

## 1. 核心结论

1. 快慢分裂、宏／微步自适应、PE 和保持结构的方法均已有成熟文献。候选贡献应放在**代谢反应分组的适用条件、分裂与 PE 偏差分开核验、多个守恒池与回退、真实模型公平评估**，而非泛称发明快慢分裂。
2. 时间尺度分离与近平衡是不同条件。远离平衡的快速过程可需要隐式积分；近平衡净通量很小的反应对可能仍含极短弛豫时间。[R4–R5]
3. 经典 Strang 二阶是有前提的。刚性初始层、强耦合、动态分组和近似子流可能破坏朴素误差估计。[R2–R3]
4. 可证守恒与可证非负性需要不同论证；PDS 中的总和守恒不能直接等同于一般生化网的多元素、多池守恒。[R9]
5. 已发现题目高度相近的 2026 年预印本 [R7]。它处理 CLE，不能作为确定性代谢 ODE 的现成理论，也不能忽略其对创新空间的限制。

## 2. 文献地图

| 编号／BibTeX 键 | 文献 | 阅读 | 对本计划的作用 |
|---|---|---|---|
| R1 / `Strang1968` | Strang (1968), *On the Construction and Comparison of Difference Schemes*. DOI [10.1137/0705041](https://doi.org/10.1137/0705041) | M；相关格式另由 R3 全文核对 | 对称组合方法的经典来源，不作为创新点 |
| R2 / `Sportisse2000` | Sportisse (2000), *An Analysis of Operator Splitting Techniques in the Stiff Case*. DOI [10.1006/jcph.2000.6495](https://doi.org/10.1006/jcph.2000.6495) | A；R3 交叉支持 | 刚性极限下传统小步展开可能不适用，算子顺序重要 |
| R3 / `Descombes2011` | Descombes et al. (2011), *Adaptive time splitting method for multi-scale evolutionary partial differential equations*. DOI [10.1142/S1793744211000412](https://doi.org/10.1142/S1793744211000412) | F | shifted-Strang、估计器有效域与额外校准 |
| R4 / `Guidry2013PE` | Guidry, Billings & Hix (2013), *Explicit integration of extremely stiff reaction networks: partial equilibrium methods*. DOI [10.1088/1749-4699/6/1/015003](https://doi.org/10.1088/1749-4699/6/1/015003) | F，开放预印本版本 | 反应进度变量、平衡弛豫、重叠约束与重构 |
| R5 / `Guidry2013QSS` | Guidry & Harris (2013), *… quasi-steady-state methods*. DOI [10.1088/1749-4699/6/1/015002](https://doi.org/10.1088/1749-4699/6/1/015002) | A | QSS 在近平衡时仍可能缺乏速度竞争力 |
| R6 / `Zhang2021` | Yu Zhang & Wenli Du (2021；2020 online), *Intelligent Time-Scale Operator-Splitting Integration for Chemical Reaction Systems*. DOI [10.1109/TNNLS.2020.3006348](https://doi.org/10.1109/TNNLS.2020.3006348) | A | 在线快慢反应识别已存在；必须纳入直接先行工作 |
| R7 / `Zeng2026` | Zeng, Xie & Wang (2026), *Adaptive Fast–Slow Operator Splitting for Multiscale Biochemical Stochastic Dynamics*. [arXiv:2604.00140v1](https://arxiv.org/abs/2604.00140v1) | F，预印本 | 最新近题对照；CLE 误差分解及宏／微步联合控制 |
| R8 / `Sandu2019` | Sandu (2019), *A Class of Multirate Infinitesimal GARK Methods*. DOI [10.1137/18M1205492](https://doi.org/10.1137/18M1205492) | F，arXiv v3 | 更强的多速率对照；耦合与阶条件不能忽略 |
| R9 / `Kopecz2018` | Kopecz & Meister (2018), *On order conditions for modified Patankar–Runge–Kutta schemes*. DOI [10.1016/j.apnum.2017.09.004](https://doi.org/10.1016/j.apnum.2017.09.004) | F | 正性／守恒证明的条件与适用边界 |
| R10 / `Teusink2000` | Teusink et al. (2000), *Can yeast glycolysis be understood in terms of in vitro kinetics of the constituent enzymes? Testing biochemistry*. DOI [10.1046/j.1432-1327.2000.01527.x](https://doi.org/10.1046/j.1432-1327.2000.01527.x) | A + BioModels 说明 | 首个真实小型动力学候选；模型本身也有生物学局限 |
| R11 / `Chassagnole2002` | Chassagnole et al. (2002), *Dynamic modeling of the central carbon metabolism of Escherichia coli*. DOI [10.1002/bit.10288](https://doi.org/10.1002/bit.10288) | A + 模型目录 | PTS／糖酵解／PPP 瞬态候选 |
| R12 / `Khodayari2016` | Khodayari & Maranas (2016), *A genome-scale Escherichia coli kinetic metabolic model k-ecoli457 satisfying flux data for multiple mutant strains*. DOI [10.1038/ncomms13806](https://doi.org/10.1038/ncomms13806) | F | 中等规模及更大网络的来源与转换成本 |
| R13 / `Ren2014` | Ren et al. (2014), *Dynamic adaptive chemistry with operator splitting schemes for reactive flow simulations*. DOI [10.1016/j.jcp.2014.01.016](https://doi.org/10.1016/j.jcp.2014.01.016) | M / 检索摘要有限 | 动态化学简化与分裂的邻近先行工作；详细方法待全文 |
| R14 / `Zhang2026Component` | Jingchao Zhang et al. (2026), *A hybrid implicit-explicit time integration for stiff chemically reacting flows based on adaptive component-splitting method*. DOI [10.1016/j.compfluid.2026.106977](https://doi.org/10.1016/j.compfluid.2026.106977) | A，含出版页节选 | 近期 IMEX／分量选择更新；不是代谢反应 PE 的直接验证 |
| D1 / `CVODEDocs` | LLNL, [CVODE 官方说明](https://computing.llnl.gov/projects/sundials/cvode) | 官方页面 | 完整系统 BDF、稀疏直接／Krylov 强基线 |
| D2 / `ARKODEDocs` | LLNL, [ARKODE 官方说明](https://computing.llnl.gov/projects/sundials/arkode) | 页面检索内容 | IMEX 与现有数值求解器的比较入口；具体 MRI 配置待实施核验 |

以上共 14 篇论文与 2 项官方软件资料，其中 6 篇取得全文并阅读关键部分。全文阅读不代表结论已在本项目复现。

## 3. 关键论文阅读卡

### R3：自适应分裂必须考虑误差估计器失效

论文用二阶 Strang 与偏移后的低阶 Strang 组合构造局部估计，研究反应扩散问题。偏移参数引入 `O(ε h²)` 差值；其差值并不是直接等于标准 Strang 的 `O(h³)` 局部误差。作者进一步估计临界步长、调整偏移量，并在 BZ 反应扩散与脉冲气体放电问题上验证。

对本项目最有用的是第 2 节和第 5 节：**步长接受条件依赖估计器有效区间，不能只看两个近似结果相差小。** 论文还强调子问题求解精度需要足够高，否则无法将主要误差归因于分裂。

采用：第一版以步长加倍作为透明基准；强刚性与切换点做独立局部参考检查；后续才评估低成本嵌入估计器。

不直接采用：PDE 情形的临界步长表达式、最优偏移参数或“快算子最后即可避免所有降阶”的泛化。动态反应分区和 PE 近似需要额外分析。

### R4–R5：PE 是额外近似，重构并非免费的后处理

R4 第 4 节用反应进度变量与守恒关系把简单可逆反应组约化为低维方程。式 (23) 给出示例反应组的平衡弛豫时间；式 (24) 按物种到平衡浓度的相对距离判别近平衡。第 4.3 节以反应向量正交空间解释守恒。

第 6.2 节比较三种平衡恢复方式：非线性约束求解、平衡丰度约束迭代、多个反应组平衡值平均。最后一种随后做总核子数归一化；论文明确指出共享组平均会破坏原来的守恒并可能长期累积。

采用：以反应组为单位判断；区分弛豫速度与当前近平衡程度；对共享变量的组联合重构；记录平衡恢复成本。

不直接采用：核反应网络中的单个总量归一化、温度依赖速率设定和性能结论。代谢网络常有多个独立池、开放交换和已经约化的酶动力学；一个整体缩放通常无法满足全部约束。

R5 摘要指出 QSS 方法离平衡较远时可有效，但趋近平衡时速度会失去竞争力，需要 PE 补充。它不支持“QSSA 单独解决所有刚性”的说法，也不提供本项目模型的速度保证。

### R6：动态快慢识别已被研究

该文摘要描述预训练反向传播网络在线识别快慢反应，用来替代昂贵 Jacobian 特征分解，并据此做时间尺度算子分裂。正式卷期为 **2021, 32(8), 3366–3376**，DOI 中的 2020 对应在线发表时期。

影响：本项目不能把“在线辨认快反应并混合求解”作为新颖性。应比较可解释分组的成本、失效边界及守恒／PE 资格控制。

限制：本轮取得摘要与书目，未获取全文，因此不声称已经确认其具体分类阈值、守恒性质或全部基线公平性。补读全文是形成论文贡献声明前的必做项。

### R7：2026 年近题预印本，值得参考也需独立核查

已阅读 v1 全文中的模型、误差分析、控制器和实验。作者研究随机 CLE 的 Lie–Trotter 分裂，提出单步均方误差形式 `h²(A(x)+B(x)/N)+o(h²)`，据此用 PI 调整宏步与快微步数。实验为三物种六反应随机网络，比较 EM、Ilie–PI 和 FS–MSE–PI，用 SSA 终态分布作为参照。

与本项目的边界：

- 这是随机误差与分布精度问题；确定性 ODE 的局部截断误差阶数、控制器指数与参考解不能直接搬用。
- 文中实验明确给定快慢反应列表；这里的“adaptive”主要体现步长和微步数，并不自动等同于本计划的状态依赖动态反应重分组。
- SSA 对应 CME；与 SSA 的分布偏差混合了 CLE 模型近似与数值积分误差。未来若扩展随机路线，需要同一 CLE 的高精度强／弱参考，而不仅是 SSA。
- 文中使用相同宏步数作比较，但不同方法微步数与分类开销不同；本项目采用同实际精度、全开销计时。

**全文复核时发现的待澄清点，尚未独立复现：**式 (6) 把宏步随机增量均分为 `ΔW/N`，而定理 3 证明使用独立微增量、方差 `h/N`；两种构造不同，不能不加说明地共用推导。第 5 节参数列表中的 `κ5` 集合与表 1 的 `0.01/0.1/1` 也有不一致。均方误差交叉项消失的条件及零 propensity 附近适用性需要进一步核查。因此本计划将它视为重要近题预印本，**不把其定理视作本项目已核实的理论保障**。

潜在差异是确定性酶动力学、完整可逆组、动态分区及 PE 偏差／多守恒池核验；这些差异是否足够新仍需进一步检索和实验，不能在此确认。

### R8：多速率方法给出了更强的比较对象

第 2 节 MRI-GARK 把慢阶段信息作为 forcing 加入快子问题，借此传递快慢耦合；第 3–4 节讨论阶条件与稳定性，第 8 节给出数值验证。论文明确关注时间尺度差异和快过程对慢动力学的作用。

启发：大时间尺度比不代表可以忽略耦合；只与普通 Euler 或简单固定 Strang 比较不足以显示优势。若快慢耦合强，带耦合的多速率方法可能更合适。

不在首版自行实现高阶 GARK；M5 优先调用成熟实现作补充基线。尚未逐条复核全部高阶条件，因此不据此承诺本计划的动态分组方法具有相同阶数。

### R9：正性保证有明确的系统类别

已读 PDS 定义、MPRK 定义、Lemma 2.6–2.7、阶条件与结论。正性证明使用适当 Patankar 权重与 M-matrix 结构；总和守恒使用 `p_ij=d_ji` 等条件。文章引言还明确区分其所研究的总量守恒与生化意义上的守恒。

采用：将非负性、线性不变量、精度分别验收；未来选择 Patankar 扩展时先验证 PDS 映射和多个元素平衡，且计入线性求解开销。

不采用：把“Patankar 无条件正性”推广为任意 SBML 速率律的无条件正确解，或把保正当作精度保证。当前首版使用拒步／回退处理负值，不承诺无条件保正。

### R10–R12：模型选型应区分数值与生物学验证

Teusink 的摘要及 BioModels 说明显示，加入旁支对得到实验观察的稳定稳态有重要作用，体外参数也不能完整解释所有体内通量。它适合真实酶动力学数值测试，但不应把数值解更准确等同于生物学模型更正确。

Chassagnole 摘要描述了 PTS 与糖酵解、PPP 的耦合，以及用瞬态浓度观测进行验证与参数估计，适合作为第二个真实候选。具体事件、辅助变量和 SBML 表达方式尚待核查。

R12 全文给出 k-ecoli457 的 **337 metabolites、457 model reactions、295 regulatory interactions**；前序 core model 为 93 metabolites / 138 reactions。其主要参数约束来自多个菌株的稳态通量，论文另评估浓度、酶参数和产率；这不自动证明所有瞬态轨迹可靠。

方法／可用性部分说明：模型使用 MATLAB，补充资料提供参数、化学计量与可执行 MAT 文件。文中报告过积分耗时，但不能用于预测当前硬件与求解器上的速度。M1 必须检查实际模型包、展开维度、初值、归一化和移植一致性；因此不将 k-ecoli457 写成“直接可用的 SBML benchmark”。

## 4. 创新性对照与研究空白的表达

| 已有内容 | 不能声称的新意 | 本项目拟核验的更窄问题 |
|---|---|---|
| R1–R3 的分裂与误差自适应 | 首创 Strang／自适应分裂 | 动态反应分组切换处的误差估计是否可靠 |
| R4–R5 的 PE／QSS | 首创消除近平衡刚性 | 可逆酶速率、共享反应块、多守恒池下 PE 是否安全且划算 |
| R6 的在线快慢识别 | 首创动态分类 | 不依赖训练的分组能否提供可解释资格检查与成本优势 |
| R7 的随机生化快慢分裂 | 首创生化快慢自适应 | 确定性 ODE、动态完整反应组、分裂与 PE 偏差的分离验证 |
| R8 的成熟多速率方法 | 首创宏微步控制 | 与全隐式及耦合多速率法相比的适用区域 |
| R9 的保正保守格式 | 首创正性或守恒 | 对一般代谢池与开放边界进行正确收支检查 |

本轮得到的是**可检验的研究定位**，不是“已证明文献空白”。后续投稿前还需引文追踪、直接先行工作全文阅读和最新数据库复查。

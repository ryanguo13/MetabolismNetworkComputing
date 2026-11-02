# 检索与证据记录

开始：2026-11-01；整理完成：2026-11-02（Asia/Shanghai）。本轮是定向研究调研，不是穷尽检索或系统综述。

## 1. 检索方式与取舍

使用网页语义检索发现来源，再通过 DOI / Crossref、arXiv、出版社、BioModels 和 LLNL 官方资料核验。对六篇关键论文下载公开 PDF，提取正文并阅读相关章节。原始论文文件留作本地阅读缓存，仓库提交原创阅读笔记、来源链接与版本指纹。

检索式包括：

- `adaptive operator splitting stiff chemical kinetics fast slow reactions error control`
- `adaptive time splitting stiff systems Descombes Duarte Massot error estimate`
- `adaptive fast slow reaction partitioning operator splitting metabolic biochemical reaction networks multirate`
- `Guidry algebraically stabilized explicit integration partial equilibrium stiff reaction networks 2013`
- `2024 2025 2026 adaptive operator splitting stiff chemical kinetics partial equilibrium`
- `Intelligent Time-Scale Operator-Splitting pdf`
- `Patankar Runge Kutta unconditional positivity conservation order conditions modified`
- `A Class of Multirate Infinitesimal GARK Methods arxiv`
- `kinetic metabolic model Chassagnole 2002 Escherichia coli Teusink 2000 glycolysis BIOMD`
- `Khodayari Maranas 2016 genome scale kinetic model Escherichia coli 457 reactions k-ecoli457`
- `SUNDIALS CVODE BDF sparse Jacobian ARKODE multirate methods`

保留能改变方法选择、理论边界、强基线或模型选型的来源。随机混合模拟、反应扩散 RDME、神经算子和 CFD 加速工作作为邻近方向筛查，不把它们混同于确定性代谢 ODE 的直接证据。搜索返回的年份／摘要并不总等于正式出版信息，BibTeX 优先采用 DOI 注册元数据并与正文核对。

## 2. 全文阅读定位

页码为下载 PDF 的物理页，可能与期刊正文页码不同。`F` 表示读过相关关键章节，不表示验证了全部理论证明或复现了实验。

| 对应条目 | 实际阅读版本 | PDF 页数 | 重点定位 |
|---|---|---:|---|
| R3 Descombes | [arXiv:1104.3697v2](https://arxiv.org/pdf/1104.3697v2) | 31 | §2 分裂与偏移格式；§3 的误差结构；§5.1/5.3（约 pp.16–23）有效步长与控制流程；§6–7 的验证与结论 |
| R4 Guidry PE | [arXiv:1112.4738v2](https://arxiv.org/pdf/1112.4738v2) | 48 | §4（约 pp.8–14），式 (23)–(24)、守恒与反应组；§6.2（pp.23–26）平衡重构；§9–12 实验范围和结论 |
| R7 Zeng | [arXiv:2604.00140v1](https://arxiv.org/pdf/2604.00140v1) | 11 | §2–4 模型、式 (6)、定理 3、宏微步控制；§5–6 实验、表 1、参考解选择与结论 |
| R8 Sandu | [arXiv:1808.02759v3](https://arxiv.org/pdf/1808.02759v3) | 40 | §2 Definition 2.2；§3–4 阶条件／耦合稳定性框架；§8–9 数值例子与结论。高阶系数未逐项复核 |
| R9 Kopecz | [arXiv:1702.04589](https://arxiv.org/pdf/1702.04589) | 24 | §1 系统定义与生化守恒边界；§2 Lemma 2.6–2.7；§3 阶条件；末尾结论。下载文本日期 2018-01-06 |
| R12 Khodayari | [出版社 PDF](https://www.nature.com/articles/ncomms13806.pdf) | 12 | Abstract / Results 的规模与验证范围；Methods、Code availability、Data availability 的模型分发与实现信息 |

R3 PDF 排版页眉含 2018 时间戳，不能因此改写期刊发表年 2011。R4 阅读的是期刊发表前的开放预印本版本，不宣称逐字等同期刊最终版。R8 阅读 v3，而 BibTeX 引用 2019 年期刊版本；涉及具体公式时应标注所用版本。

下载文件 SHA-256，用于定位本轮实际阅读内容：

| 来源 | SHA-256 |
|---|---|
| R3 | `9930e91dcc87dc43085ba3ddb7206563d924598c2109213b44f22587f0f6619c` |
| R4 | `135ad96833727b18a73054bb3c1f0dfbd06e00a5692e84871bd3b3e4f5480ccd` |
| R7 | `df0212da9142f907608ad4a4240db15857faa769561df7b46835bae5a4e90a25` |
| R8 | `70a9b85334170f14d38a895bb27f20aba0cb61f77515f3ba77c4137a59d05730` |
| R9 | `9bc4927d772b2794bdd12227362dcb71c4528262a0de3eaa50d39bd5b25f9b14` |
| R12 | `7bfdaf440925ee23ccc8659c0376092ce5244a9c0301b85a55cbca64f5a8180c` |

## 3. 元数据与模型来源核验

13 篇带期刊 DOI 的文献通过 Crossref 核验题名、作者和卷期；R7 通过 arXiv 题名页及 PDF 确认作者、版本和 2026-03-31 首次提交日期。检索到的 2026 文献在本次检索日期之前；不将它们按模型训练知识的时间界限遗漏。

模型身份还参考：

- [BioModels Teusink 模型说明](https://www.ebi.ac.uk/biomodels/content/model-of-the-month?month=01&year=2007)
- [ODEbase 的 BIOMD0000000064 目录](https://www.odebase.org/detail/1374)：只作规模线索，非独立 ODE 维度证明。
- [Chassagnole PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/17590932/)
- [k-ecoli457 出版页面及补充资料入口](https://www.nature.com/articles/ncomms13806)

本轮没有下载 SBML / MAT 模型包、提取模型参数、导入求解器或生成参考轨迹。论文 PDF 下载仅为文献阅读。

## 4. 获取限制与后续必读

- R6 DOI 正文读取超时；取得摘要和 IEEE 卷期线索并由 Crossref 核验，未获得全文。正式讨论算法差异前必须补读。
- R1 为书目核验；R2 为摘要级阅读；相关分裂公式及刚性问题另由 R3 全文支持。
- R5 为摘要级；R10–R11 为摘要／模型说明级，M1 需要结合原模型补充材料核验具体方程。
- R13 仅有元数据与有限检索片段，不推测其具体误差控制公式。
- R14 读到出版页摘要、Highlights 和部分引言／结论，未取得完整方法与实验，因此不引用其加速百分比作为本项目依据。
- R7 预印本存在需澄清的符号／实验参数一致性问题，见阅读笔记；本轮未复现、未联系作者，也未对所有证明作正式审稿判断。
- 没有证明“截至当前不存在相同方法”。M6 前需补足引文追踪和最新论文复查。

## 5. 规划决策的证据链

| 规划决策 | 支持来源 | 证据强度／限制 |
|---|---|---|
| 先无 PE，后单独评估 PE | R3、R4、R5 | 方法与误差来源可区分；收益待实验 |
| PE 同时检查弛豫与近平衡程度 | R4 §4 | 核反应实例清楚；酶动力学须重新验证 |
| 共享反应组不能简单独立修正 | R4 §6.2 | 原文明确存在重叠与守恒恢复问题 |
| 不以两种近似解接近就认定无偏差 | R3 有效域分析 + PE 改变动力学的事实 | 本项目采用独立全系统参考步作设计保障，尚无一般误差定理 |
| 多个守恒池、正性、热力学分别处理 | R4 §4.3、R9 §1–2 | 结构条件不同；开放系统还需收支项 |
| 使用强隐式／多速率基线 | D1、D2、R8 | 可用能力已核对，具体本机配置待实施 |
| 真实模型从小到中等规模递进 | R10–R12 + 原始构想 | 模型身份与论文规模已核对，执行性未验证 |

本记录用于让后续研究者复核“哪些结论来自哪篇文献、读到什么程度、哪些仍是假设”，不充当实验日志。

## 6. 2026-11-02 M1 补充来源（不改写 M0 阅读深度）

- BioModels 模型下载与文件清单 API 实际取得 B1／B2 SBML、B2 原始 `curation_image.jpeg` 和 `curation_notes.txt`；所有来源及指纹登记于 `models/manifest.json`。B2 图注明确 MathSBML 复现论文图 4–6；此证据替代缺少原论文观测数据时的公开执行结果核对，不声称读到了原论文全文。
- 出版社 Supplementary Software 1 的嵌套 ZIP 实际取得 k-ecoli457 `Data.mat`、原始 RHS、生成 Jacobian、初值条件和净通量恢复代码；通过 Octave 8.4.0 实际执行与 Python 逐项对照，见 M1 验收报告。
- 当前科学结论来自保存的执行记录，未补读到的论文全文仍保持此前的阅读级别；没有联系作者或制造缺失实验数据。

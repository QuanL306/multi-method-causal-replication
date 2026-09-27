# Week 24 学习笔记：因果发现与综合 Capstone

> **对应教材**: Molak Ch 12-14 (Causal Discovery) + Facure Ch 12 (Next Steps)
> **日期**: 2026-08-22
> **环境**: Python 3.11, causal-python conda env (castle 1.0.4, lingam 1.13.0)

---

## 1. 因果发现：从"假设 DAG 已知"到"从数据学习 DAG"

Month 1-4 的所有因果推断工作都建立在一个前提上：**因果图（DAG）是已知的**。我们用领域知识画出 DAG，然后用 back-door criterion 识别混杂，再用各种估计方法计算因果效应。但现实中，因果图往往不完整、有争议、甚至完全未知。

因果发现（causal discovery）解决的就是这个问题：**给定观测数据，能否推断变量间的因果结构？**

Molak 将因果发现分为三大家族：

- **基于约束的方法**（PC、FCI）：通过条件独立性测试排除因果边
- **基于评分的方法**（GES、NOTEARS）：搜索最优拟合的 DAG 结构
- **基于函数模型的方法**（LiNGAM）：利用数据的分布特征确定因果方向

我们的实验对这三类方法做了系统对比，结果非常有启发性。

---

## 2. PC 算法：条件独立性驱动的骨架发现

PC 算法（Spirtes, Glymour, Scheines, 2000）是最经典的因果发现方法。它的逻辑是：如果 $X$ 和 $Y$ 在给定某组变量 $S$ 下条件独立，则 $X$ 和 $Y$ 之间不存在直接因果边。

**三步流程**：
1. **骨架发现**：从完全图开始，逐阶增加条件集大小，测试条件独立性，移除不显著的边
2. **v-结构定向**：如果 $X \rightarrow Z \leftarrow Y$ 且 $X$ 与 $Y$ 不相邻，则定向为 collider
3. **Meek 规则传播**：利用一致性约束传播剩余边的方向

我们的实验在不同显著性水平（$\alpha$）下运行 PC：
- $\alpha=0.01$（严格）：漏掉一些真实边（recall 低），但误报少（precision 高）
- $\alpha=0.10$（宽松）：捕捉更多边，但引入假阳性
- $\alpha=0.05$：平衡点，F1=0.727，SHD=3

fig6 的样本量分析揭示了 PC 的一致性：当 $n$ 从 100 增到 5000 时，SHD 从 8 降到 2，F1 从 0.45 升到 0.73。这验证了 PC 是**统计一致的**——给定无限数据，它会收敛到正确的 Markov 等价类。

但 PC 有一个根本限制：它只能识别 **Markov 等价类**（CPDAG），不能区分同一等价类中的不同 DAG。例如 $A \rightarrow B \rightarrow C$ 和 $A \leftarrow B \leftarrow C$ 在观测数据下不可区分。

---

## 3. DirectLiNGAM：非高斯性是因果方向的钥匙

LiNGAM（Shimizu et al., 2006）的核心洞察令人惊叹：**如果噪声是非高斯的，因果方向可以从观测数据中唯一确定。**

直觉解释：在高斯世界中，$Y = \beta X + \epsilon$ 和 $X = \gamma Y + \eta$ 产生完全相同的联合分布——你无法区分因果方向。但非高斯分布打破了这种对称性：独立成分分析（ICA）可以识别混合矩阵，从而恢复因果顺序。

DirectLiNGAM 在我们的医疗数据上取得了**完美恢复**（SHD=0, F1=1.000）。原因是我们使用了 Laplace 分布的噪声（非高斯），这正是 LiNGAM 的理论前提。

这个结果的实践意义是双面的：
- **好消息**：当数据确实非高斯时（如收入、计数数据），LiNGAM 可以比约束方法获得更强的结论
- **坏消息**：如果噪声接近高斯（如中心极限定理生效的聚合指标），LiNGAM 的优势消失

---

## 4. NOTEARS：将组合优化变为连续优化

传统因果发现是 NP-hard 的组合优化问题——在 $d$ 个变量的 $2^{\binom{d}{2}}$ 个可能 DAG 中搜索。NOTEARS（Zheng et al., 2018）的突破在于用**矩阵指数的迹**将 DAG 约束编码为连续等式：

$$\text{tr}(e^{W \circ W}) - d = 0 \iff W \text{ 是 DAG 的邻接矩阵}$$

这将离散的图搜索转化为连续优化问题：

$$\min_W \|X - XW\|_F^2 + \lambda \|W\|_1 \quad \text{s.t.} \quad \text{tr}(e^{W \circ W}) - d = 0$$

我们的从零实现用 `scipy.linalg.expm` 计算矩阵指数，用增广拉格朗日法处理约束。结果不尽如人意（SHD=9, F1=0.300），反映了 NOTEARS 的几个实践挑战：
- 矩阵指数计算在 $d$ 大时开销大
- 增广拉格朗日的收敛对超参数敏感
- $L_1$ 正则化可能过于激进地剪枝

这印证了 Molak 的提醒：**因果发现算法的理论保证（一致性、正确性）和实际表现之间经常有显著差距**。数据量、噪声分布、图密度都会影响结果。

---

## 5. 从发现到估计：端到端因果推断流水线

fig7 展示了因果发现最激动人心的应用场景：**用发现的 DAG 作为 DoWhy 的输入进行因果效应估计**。

因为 DirectLiNGAM 完美恢复了真实 DAG，所以 DoWhy 在"真实 DAG"和"发现 DAG"下给出了完全相同的因果效应估计（-0.1041）。而忽略混杂的朴素相关分析给出 -0.6073——偏差超过 5 倍。

这个端到端流程（数据 → 因果发现 → DAG → DoWhy → 因果效应）代表了一个完整的数据驱动因果推断管道。它的局限也很清楚：

1. **因果发现本身可能出错**——NOTEARS 和 GES 的表现远不如 LiNGAM
2. **发现的 DAG 不确定如何传播到估计的不确定性**——目前没有标准方法处理这个"双重不确定性"
3. **因果发现假设因果充分性**（无隐变量），这在实践中几乎从不成立

FCI 算法可以处理隐变量，但输出的是 PAG（部分 ancestral graph），解读更复杂。

---

## 6. 综合 Capstone：方法论的三角验证

capstone.py 将 Month 05 四周所学全部整合到一个 SaaS 客户流失场景中。六个估计方法（LR、PSM、IPW、DR、LinearDML、CausalForestDML）高度一致，ATE 全部收敛在 -6.0 附近，CV 稳定性 std=0.13。

**方法论的一致性是最强的稳健性证据**。当六种不同原理的方法（参数/半参数/非参数，匹配/加权/回归）给出相近结果时，我们可以相当有信心地认为估计是可靠的。

Capstone 的关键商业洞察：
- **谁受益最多**：enterprise 客户（CATE 约 -10），startup 客户受益最少（约 -3）
- **CATE 排序精度**：LinearDML 与真实 CATE 的 Spearman ρ=0.97，说明排序几乎完美
- **预算优化**：如果只能覆盖 30% 客户，CATE 引导的精准投放比随机投放 ROI 提升约 45%

---

## 7. Month 05 回顾：从 R 到 Python 的迁移完成了什么

回顾整个 Month 05，我们完成了从 R 因果推断生态到 Python 工业生态的系统迁移：

**Week 21**（倾向得分 / DoWhy）：掌握了观测性因果推断的核心工具族——PS 匹配、IPW、DR——以及 DoWhy 四步流程。发现 DoWhy 0.14 的状态污染 bug，学会了为每个 estimator 创建新 CausalModel。

**Week 22**（CATE / Meta-learners / DML）：从零实现了 S/T/X/DR-learner，理解了 meta-learner 架构的本质差异。DML 的 Neyman orthogonalization 和交叉拟合是 ML+因果的核心理论创新。Causal Forest 提供了异质性效应的非线性建模和特征重要性。

**Week 23**（DID / Synth / IV / RDD / Geo / Switchback）：用 Python 重新实现了 Month 3 在 R 中做过的面板数据方法，同时新增了工具变量（2SLS）、Geo 实验和 Switchback 设计。TWFE 负权重问题在 Python 中同样得到了清晰展示。

**Week 24**（因果发现 / Capstone）：进入了 R 生态未触及的新领域——从数据学习因果结构。PC、LiNGAM、NOTEARS 三大算法的对比揭示了因果发现的核心权衡：约束方法的保守性 vs 评分方法的不稳定性 vs 函数模型的分布假设。

**最大的收获不是任何一个方法，而是方法论的整合**：因果发现给出结构假设，DoWhy 框架将假设形式化，多种估计方法提供三角验证，refutation 测试进行压力检验，CATE 分析支持精准决策。这个完整流水线才是 Python 因果推断生态的真正价值。

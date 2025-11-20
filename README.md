# Optimization-
A rookie
# LASSO Optimization Algorithms Comparison & Analysis
# LASSO 回归优化算法收敛性对比分析

## 📖 项目简介 (Introduction)

本项目旨在通过数值实验，比较不同优化算法在求解 **LASSO (Least Absolute Shrinkage and Selection Operator)** 回归问题时的收敛性能。

目标函数定义为：
$$\min_{\beta} \frac{1}{2n} \| y - X\beta \|_2^2 + \lambda \| \beta \|_1$$

本项目不仅实现了经典的优化算法，还重点探究了 **ADMM** 算法中惩罚参数 $\rho$ 对收敛速度的影响，并通过 **Monte Carlo (蒙特卡洛)** 随机实验验证了算法的鲁棒性。

## 🛠️ 实验设置 (Experimental Setup)

为了消除单次数据生成的偶然性，准确评估算法的平均性能和稳定性，我们采用了以下实验设置：

* **数据维度**: 样本量 $n=200$, 特征数 $p=50$。
* **实验机制**: 独立重复实验 **50~100 次** (Randomized Trials)。
* **可视化方案**:
    * **细淡色线 (Clouds)**: 展示单次实验的收敛轨迹，反映算法的方差（稳定性）。
    * **粗实线 (Mean)**: 展示所有实验的平均收敛路径，反映算法的期望性能。

## 🚀 实现算法 (Algorithms Implemented)

1.  **Coordinate Descent (BCD)**: 块坐标下降法
2.  **FISTA**: 快速迭代收缩阈值算法 (Nesterov 加速)
3.  **ADMM**: 交替方向乘子法 (包含参数敏感性分析)
4.  **Huber Gradient Descent**: 基于 Huber 平滑逼近的梯度下降法

## 📊 核心分析与发现 (Key Findings)

### 1. ADMM 参数调优分析 (Parameter Tuning for ADMM)

在 ADMM 算法中，惩罚参数 $\rho$ (rho) 是影响收敛效率的关键超参数。为了寻找最优解，我们对 $\rho$ 进行了敏感性测试，取值范围为 $\{0.5, 1, 2, 5\}$。

![LASSO Convergence Plot](Figure_2.jpg)
*(图示：100次随机实验下的算法收敛曲线对比)*

根据实验结果（如上图所示），我们观察到：

* **$\rho = 1$ (Magenta/品红)**: **表现最佳**。收敛曲线下降最快，且方差较小，说明在该问题规模下，$\rho=1$ 最能平衡原始残差和对偶残差的下降。
* **$\rho = 2$ (Cyan/青色)**: 收敛速度次之，略慢于 $\rho=1$。
* **$\rho = 0.5$ (Orange/橙色)**: 收敛速度中等。由于步长偏离最优值，导致迭代效率下降。
* **$\rho = 5$ (Purple/紫色)**: **表现最差**。较大的 $\rho$ 值导致每一步的更新步长过小（Over-penalized），显著拖慢了收敛过程。

**结论**: 针对当前的 LASSO 问题设置，**$\rho=1$ 是最优参数选择**。

### 2. 算法综合性能对比 (Performance Comparison)

通过横向对比不同算法的收敛曲线（Suboptimality $f(x_k) - f^*$），我们得出以下结论：

* **Coordinate Descent (红色)**: **统治级表现**。
    由于 LASSO 问题的 Hessian 矩阵通常不是完全稠密的，坐标下降法利用贪心策略逐个更新坐标，展现出了极快的收敛速度（近似垂直下降），在极短的迭代步数内达到了机器精度。

* **FISTA (深蓝色)**: **高效但伴有震荡**。
    作为一阶加速算法，FISTA 的收敛速度远快于普通梯度下降。图中深蓝色曲线呈现出的**波纹状 (Ripples)** 是由于动量 (Momentum) 项引入的惯性效应导致的。虽然存在震荡，但其能够迅速逼近最优解。

* **ADMM (品红, $\rho=1$)**: **稳健**。
    在调优参数后，ADMM 展现了良好的线性收敛能力。虽然在单机串行环境下不如坐标下降法高效，但其优势在于可分解性，适合处理分布式大规模数据。

* **Huber Gradient (绿色)**: **收敛最慢**。
    由于 Huber 方法是对 $L_1$ 范数进行了平滑近似，它无法像近端算法（Proximal Algorithms）那样利用软阈值算子将系数精确压缩为 0。受限于平滑参数 $\delta$ 和梯度下降的特性，其收敛轨迹平缓，且存在系统性误差（无法精确收敛到 $f^*$）。

## 💻 运行指南 (Quick Start)

### 依赖环境
确保您的环境中安装了以下 Python 库：
```bash
pip install numpy matplotlib scikit-learn

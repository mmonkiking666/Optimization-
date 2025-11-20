import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from numpy.linalg import norm, cholesky, solve
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# --- 1. 参数设置 ---
n_samples = 200
n_features = 50
max_iter = 100
n_trials = 100  # 实验重复次数
lambda_ratio = 0.1

# 颜色和标签配置
algo_configs = {
    'Coordinate Desc': {'color': 'firebrick', 'style': '-'},
    'Huber Gradient':  {'color': 'limegreen', 'style': '-'},
    'FISTA':           {'color': 'darkblue',  'style': '-'},
    'ADMM (rho=0.5)':  {'color': 'orange',    'style': '-'},
    'ADMM (rho=1)':    {'color': 'magenta',   'style': '-'},
    'ADMM (rho=2)':    {'color': 'cyan',      'style': '-'},
    # 【新增】 ADMM rho=5，使用紫色
    'ADMM (rho=5)':    {'color': 'purple',    'style': '-'}
}

# 初始化结果字典
results = {name: [] for name in algo_configs.keys()}


# --- 2. 辅助函数 ---

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def lasso_objective(beta, X, y, n, lam):
    residual = X @ beta - y
    l2_loss = (0.5 / n) * (residual @ residual)
    l1_norm = lam * norm(beta, 1)
    return l2_loss + l1_norm


# --- 3. 算法求解器 ---

# [1] Huber Gradient
def gradient_descent_huber(X, y, n, p, lam, max_iter, f_star, delta=1e-2):
    beta = np.zeros(p)
    history = []
    L_f = norm(X.T @ X / n, ord=2)
    L_g = lam / delta
    alpha = 1.0 / (L_f + L_g)

    for k in range(max_iter):
        grad_f = (X.T @ (X @ beta - y)) / n
        grad_g_huber = lam * np.clip(beta / delta, -1, 1)
        beta = beta - alpha * (grad_f + grad_g_huber)
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


# [2] FISTA (加速 Proximal Gradient)
def fista(X, y, n, lam, max_iter, f_star):
    beta = np.zeros(n_features)
    z = np.zeros(n_features)
    t = 1.0
    history = []
    L = norm(X.T @ X / n, ord=2)
    alpha = 1.0 / L
    for k in range(max_iter):
        beta_old = beta.copy()
        grad_z = (X.T @ (X @ z - y)) / n
        beta = soft_threshold(z - alpha * grad_z, alpha * lam)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = beta + ((t - 1) / t_new) * (beta - beta_old)
        t = t_new
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


# [3] ADMM
def admm(X, y, n, p, lam, rho, max_iter, f_star):
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    history = []
    I = np.identity(p)
    L_cho = cholesky(X.T @ X / n + rho * I)
    for k in range(max_iter):
        rhs = (X.T @ y / n) + rho * (z - u)
        beta = solve(L_cho.T, solve(L_cho, rhs))
        z = soft_threshold(beta + u, lam / rho)
        u = u + beta - z
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


# [4] Coordinate Descent
def coordinate_descent(X, y, n, p, lam, max_iter, f_star):
    beta = np.zeros(p)
    history = []
    A_j = np.zeros(p)
    for j in range(p):
        A_j[j] = (X[:, j] @ X[:, j]) / n
        if A_j[j] == 0: A_j[j] = 1e-8
    for k in range(max_iter):
        for j in range(p):
            old_beta_j = beta[j]
            residual_no_j = y - (X @ beta - X[:, j] * old_beta_j)
            c_j = (X[:, j] @ residual_no_j) / n
            beta[j] = soft_threshold(c_j / A_j[j], lam / A_j[j])
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


# --- 4. 主循环 ---

print(f"Starting {n_trials} random trials...")

for i in range(n_trials):
    if (i + 1) % 5 == 0:
        print(f"Processing trial {i + 1}/{n_trials}...")

    X = np.random.randn(n_samples, n_features)
    true_beta = np.zeros(n_features)
    true_beta[:10] = np.random.uniform(-5, 5, 10)
    y = X @ true_beta + np.random.randn(n_samples) * 0.5

    lam_max = norm(X.T @ y, ord=np.inf) / n_samples
    lam = lam_max * lambda_ratio

    lasso_sklearn = Lasso(alpha=lam, fit_intercept=False, tol=1e-14, max_iter=20000)
    lasso_sklearn.fit(X, y)
    f_star = lasso_objective(lasso_sklearn.coef_, X, y, n_samples, lam)

    # 运行算法
    results['Coordinate Desc'].append(coordinate_descent(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Huber Gradient'].append(gradient_descent_huber(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['FISTA'].append(fista(X, y, n_samples, lam, max_iter, f_star))

    # ADMM 变体 (0.5, 1, 2, 5)
    results['ADMM (rho=0.5)'].append(admm(X, y, n_samples, n_features, lam, 0.5, max_iter, f_star))
    results['ADMM (rho=1)'].append(admm(X, y, n_samples, n_features, lam, 1.0, max_iter, f_star))
    results['ADMM (rho=2)'].append(admm(X, y, n_samples, n_features, lam, 2.0, max_iter, f_star))
    results['ADMM (rho=5)'].append(admm(X, y, n_samples, n_features, lam, 5.0, max_iter, f_star)) # 新增

print("All trials complete. Plotting...")

# --- 5. 绘图 ---

plt.figure(figsize=(10, 8))
k_axis = np.arange(1, max_iter + 1)

for name, histories in results.items():
    data_matrix = np.array(histories)
    mean_curve = np.mean(data_matrix, axis=0)
    color = algo_configs[name]['color']

    # 绘制云雾
    for single_run in data_matrix:
        plt.plot(k_axis, single_run, color=color, alpha=0.15, linewidth=0.8)

    # 绘制主线
    plt.plot(k_axis, mean_curve, color=color, linestyle='-', linewidth=3, label=name)

plt.yscale('log')
plt.xlabel('Iteration k', fontsize=14)
plt.ylabel('Suboptimality $f(x_k) - f^*$', fontsize=14)
plt.title(f'LASSO Convergence: {n_trials} Randomized Trials', fontsize=16)
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.ylim(bottom=1e-10)
plt.xlim(0, max_iter)

plt.tight_layout()
plt.show()
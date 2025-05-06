import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mpl
import seaborn as sns

# -------------------------------
# 使用 PGF 输出（用于 LaTeX 插图）
# -------------------------------
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "xelatex",  # 或 xelatex、lualatex
    "text.usetex": True,
    "pgf.rcfonts": False,  # 禁止自动应用 matplotlib 字体
    # "axes.labelsize": 12,
    # "axes.titlesize": 14,
    "axes.unicode_minus": False,
    "text.latex.preamble": r"""
        \usepackage{fontspec}
        \setmainfont{Times New Roman}  % 让 LaTeX 使用 Times New Roman
        \setsansfont{Arial}           % 可选，设置无衬线字体
        \setmonofont{Courier New}     % 可选，设置等宽字体
    """,
    "mathtext.default": 'regular',
    'mathtext.fontset': "custom",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # 尽量匹配 LaTeX 的 Times 字体
})
sns.set_theme(style="whitegrid")

# -------------------------------
# 1. 生成源信号
# -------------------------------
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)
s2 = signal.sawtooth(2 * np.pi * time)
s3 = signal.square(2 * np.pi * time)

S = np.c_[s1, s2, s3]
S += 0.1 * np.random.normal(size=S.shape)
S /= S.std(axis=0)

# -------------------------------
# 2. 混合信号
# -------------------------------
A = np.random.rand(3, 3)
X = S @ A.T

# -------------------------------
# 3. FastICA 实现
# -------------------------------
def g(x): return np.tanh(x)
def g_prime(x): return 1 - np.tanh(x) ** 2

def whiten(X):
    X -= X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    d, E = np.linalg.eigh(cov)
    D_inv = np.diag(1. / np.sqrt(d))
    return X @ E @ D_inv @ E.T

def fastica(X, n_components, max_iter=200, tol=1e-4):
    X = whiten(X)
    n_samples, n_features = X.shape
    W = np.zeros((n_components, n_features))

    for i in range(n_components):
        w = np.random.rand(n_features)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            wx = X @ w
            gwx = g(wx)
            g_wx = g_prime(wx)
            w_new = (X.T @ gwx - g_wx.mean() * w) / n_samples
            for j in range(i):
                w_new -= np.dot(w_new, W[j]) * W[j]
            w_new /= np.linalg.norm(w_new)
            if np.abs(np.abs(np.dot(w_new, w)) - 1) < tol:
                break
            w = w_new
        W[i, :] = w
    return X @ W.T

# -------------------------------
# 4. 分离信号
# -------------------------------
S_est = fastica(X, n_components=3)

# -------------------------------
# 5. 可视化并导出 PGF
# -------------------------------
plt.figure(figsize=(6, 5))

plt.subplot(3, 1, 1)
plt.title("Original Signals")
plt.plot(S, linewidth=1)
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
plt.plot(X, linewidth=1)
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.title("Recovered Signals")
plt.plot(S_est, linewidth=1)
plt.ylabel("Amplitude")
plt.xlabel("Sample Index")

plt.tight_layout()
plt.savefig("fastica_result.pgf")
# plt.show()

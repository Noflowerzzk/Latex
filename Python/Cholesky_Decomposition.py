import numpy as np

def cholesky_decomposition(A):
    """ 对称正定矩阵 A 进行 Cholesky 分解，返回 L """
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i, k] * L[j, k] for k in range(j))

            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_k)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]

    return L

# 测试
A = np.array([[5, -2, 0],
              [-2, 3, -1],
              [0, -1, 1]], dtype=float)

L = cholesky_decomposition(A)
print("L:\n", L)
print("L.T:\n", L.T)

import numpy as np

def lr_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)  # 单位下三角矩阵
    R = A.astype(float)  # 确保计算不受整型影响
    
    for i in range(n):
        for j in range(i + 1, n):
            factor = R[j, i] / R[i, i]
            L[j, i] = factor  # 记录到 L 矩阵
            R[j] -= factor * R[i]  # 消元
    
    return L, R

A = np.array([[2, -1, 3],
              [1, 2, 1],
              [2, 4, -2]], dtype=float)

L, R = lr_decomposition(A)
print("L:")
print(L)
print("R:")
print(R)

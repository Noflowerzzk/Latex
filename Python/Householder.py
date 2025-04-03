import numpy as np

def householder_qr(A):
    """ 使用 Householder 变换计算矩阵 A 的 QR 分解 """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for k in range(n):
        # 选取列向量
        x = R[k:, k]
        
        # 计算法向量 v
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (1 if x[0] >= 0 else -1)
        v = x + e1
        v = v / np.linalg.norm(v)

        # 构造 Householder 变换矩阵 H_k = E - 2uu.T
        H_k = np.eye(m)
        H_k[k:, k:] -= 2.0 * np.outer(v, v)

        # 更新 R 和 Q
        R = H_k @ R
        Q = Q @ H_k.T  # 注意 Q 需要不断右乘 H_k 的转置

    return Q, R

# 测试
A = np.array([[3, 14, 9],
              [6, 43, 3],
              [6, 22, 15]], dtype=float)

Q_A, R_A = householder_qr(A)

print("Q_A:\n", Q_A)
print("R_A:\n", R_A)

B = np.array([[1, 1, 1],
              [2, -1, -1],
              [2, -4, 10]], dtype=float)

Q_B, R_B = householder_qr(B)

print("Q_B\n", Q_B)
print("R_B:\n", R_B)

print(np.linalg.inv(R_B) @ Q_B.T)